"""Training loop for the deep models.

Speed choices (the previous project took 2–8 h per model at ~3,600 windows/s):

- data stays on the GPU (``WindowSet``); a batch is one index-gather;
- each epoch draws ``windows_per_epoch`` random windows instead of every
  overlapping window (neighbouring windows differ by one sample, so a full
  pass is mostly repetition);
- early stopping is checked on a fixed, evenly spaced validation subset;
  the full validation set is scored once, after training;
- bf16 autocast and TF32 matmuls on CUDA, large batches (1024 by default).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .windows import WindowSet, evenly_spaced

LOSSES = ("mse", "huber", "mae")


@dataclass
class TrainSettings:
    batch_size: int = 1024
    max_epochs: int = 60
    windows_per_epoch: int = 200_000
    val_windows: int = 50_000
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    huber_delta: float = 1.0
    amp: bool = True
    grad_clip: float = 1.0
    max_minutes: float | None = None
    compile: bool = False
    eval_batch_size: int = 2048
    # Channel-independent PatchTST runs batch × channels sequences; cap that product to fit memory.
    ci_sequence_budget: int = 8192
    # iTransformer attends across channels: batch × heads × channels² score elements per layer.
    attention_budget: int = 2**27

    @classmethod
    def from_dict(cls, doc: dict[str, Any] | None) -> TrainSettings:
        doc = dict(doc or {})
        unknown = set(doc) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown train settings: {sorted(unknown)}")
        out = cls(**doc)
        if out.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}: {out.loss}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss_fn(settings: TrainSettings) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if settings.loss == "huber":
        return nn.HuberLoss(delta=settings.huber_delta)
    if settings.loss == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


def _autocast(device: torch.device, enabled: bool):
    # bf16 needs no loss scaling (RTX 30/40 series); older GPUs simply train in fp32.
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def configure_backend(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predict(model: nn.Module, ws: WindowSet, idx: np.ndarray, *, batch_size: int = 4096,
            amp: bool = True) -> np.ndarray:
    """Standardized predictions ``(n, H, T)`` for windows ``idx``."""

    model.eval()
    parts = []
    for _, tidx in ws.iter_batches(idx, batch_size):
        x, _ = ws.gather(tidx)
        with _autocast(ws.device, amp):
            parts.append(model(x).float().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros((0, len(ws.horizons), ws.y.shape[1]), np.float32)


@torch.no_grad()
def _val_loss(model: nn.Module, ws: WindowSet, idx: np.ndarray, settings: TrainSettings,
              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
    model.eval()
    total, count = 0.0, 0
    for _, tidx in ws.iter_batches(idx, settings.eval_batch_size):
        x, y = ws.gather(tidx)
        with _autocast(ws.device, settings.amp):
            pred = model(x)
        total += float(loss_fn(pred.float(), y)) * len(tidx)
        count += len(tidx)
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train: WindowSet,
    val: WindowSet,
    settings: TrainSettings,
    *,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train with early stopping on validation loss; restores the best weights."""

    device = train.device
    configure_backend(device)
    torch.manual_seed(seed)
    model.to(device)
    runner = torch.compile(model) if settings.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, settings.patience // 3))
    loss_fn = _loss_fn(settings)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    val_idx = evenly_spaced(len(val), settings.val_windows)
    per_epoch = min(settings.windows_per_epoch, len(train))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float]] = []
    best_loss, best_state, best_epoch, bad_epochs = math.inf, None, 0, 0
    status = "MAX_EPOCHS"
    started = time.perf_counter()
    train_seconds = 0.0
    windows_seen = 0
    for epoch in range(1, settings.max_epochs + 1):
        runner.train()
        epoch_start = time.perf_counter()
        order = torch.randperm(len(train), device=device, generator=generator)[:per_epoch]
        loss_sum = torch.zeros((), device=device)
        batches = 0
        for start in range(0, per_epoch, settings.batch_size):
            batch = order[start : start + settings.batch_size]
            if len(batch) < 2:  # BatchNorm needs two samples
                continue
            x, y = train.gather(batch)
            with _autocast(device, settings.amp):
                pred = runner(x)
            loss = loss_fn(pred.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if settings.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            loss_sum += loss.detach()
            batches += 1
            windows_seen += len(batch)
        train_loss = float(loss_sum) / max(batches, 1)
        if not math.isfinite(train_loss):
            status = "NAN_LOSS"
            log(f"epoch {epoch}: loss is not finite; stopping")
            break
        train_seconds += time.perf_counter() - epoch_start
        val_loss = _val_loss(runner, val, val_idx, settings, loss_fn)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                        "seconds": round(time.perf_counter() - epoch_start, 3)})
        improved = val_loss < best_loss * (1 - 1e-4)
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        log(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  "
            f"{history[-1]['seconds']:.1f}s{'  *' if improved else ''}")
        if bad_epochs >= settings.patience:
            status = "EARLY_STOPPED"
            break
        if settings.max_minutes and (time.perf_counter() - started) / 60 > settings.max_minutes:
            status = "TIME_LIMIT"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return {
        "status": status,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
        "train_seconds": round(train_seconds, 2),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "windows_seen": windows_seen,
        "windows_per_second": round(windows_seen / train_seconds, 1) if train_seconds else None,
        "peak_gpu_mb": round(peak_mb, 1) if peak_mb is not None else None,
        "batch_size": settings.batch_size,
    }
