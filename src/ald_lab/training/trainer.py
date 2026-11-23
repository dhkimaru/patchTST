"""Training loop for the deep models.

Speed choices (the previous project took 2–8 h per model at ~3,600 windows/s):

- data stays on the GPU (``WindowSet``); a batch is one index-gather;
- each epoch draws ``windows_per_epoch`` random windows instead of every
  overlapping window (neighbouring windows differ by one sample, so a full
  pass is mostly repetition);
- early stopping is checked on a fixed, evenly spaced validation subset;
  the full validation set is scored once, after training;
- bf16 autocast and TF32 matmuls on CUDA, large batches (1024 by default).

When a batch does not fit in GPU memory (a search can ask for 500 patches × batch 2048), the batch is split into
parts whose gradients are added before one optimizer step, so the effective batch size stays the one asked for
(BatchNorm statistics then come from each part). Evaluation halves its batch instead.

Each epoch also records the validation MAE in model space (standardized, or the scaled correction to the recipe
template); for one target and data setting it is proportional to the MAE in physical units, so it is comparable
across losses. A hyperparameter search reads it through ``on_epoch`` to stop weak trials early (``hpo.py``).
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
LR_SCHEDULES = ("plateau", "cosine", "constant")


@dataclass
class TrainSettings:
    batch_size: int = 1024
    max_epochs: int = 60
    windows_per_epoch: int = 200_000
    val_windows: int = 50_000
    patience: int = 8
    # Early stopping never fires before this epoch (a run in 140~146 stopped at epoch 34 and was 25 % worse).
    min_epochs: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    huber_delta: float = 1.0
    # plateau: halve the rate when validation stalls (the original behaviour); cosine: decay to 1 % of ``lr`` over
    # ``max_epochs``; constant. ``warmup_epochs`` ramps the rate up linearly first (cosine and constant).
    lr_schedule: str = "plateau"
    warmup_epochs: int = 0
    amp: bool = True
    grad_clip: float = 1.0
    max_minutes: float | None = None
    compile: bool = False
    eval_batch_size: int = 2048
    # Recent runs count more: a training window is drawn with weight 0.5 ** (age / half-life), age = how many training
    # runs (in run order) are newer than its run. 0 = every window equally likely (the default).
    recency_halflife_runs: float = 0.0
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
        if out.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}: {out.lr_schedule}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss_fn(settings: TrainSettings) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if settings.loss == "huber":
        return nn.HuberLoss(delta=settings.huber_delta)
    if settings.loss == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


def is_oom(exc: BaseException) -> bool:
    """True for a GPU out-of-memory error."""

    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _over_batches(ws: WindowSet, idx: np.ndarray, batch_size: int,
                  fn: Callable[[torch.Tensor, torch.Tensor], Any]) -> tuple[list[Any], int]:
    """``fn(x, y)`` for each batch of ``idx`` in order, halving the batch while it runs out of GPU memory.
    Returns the outputs and the batch size that fitted."""

    while True:
        out: list[Any] = []
        oom = False
        try:
            for _, tidx in ws.iter_batches(idx, batch_size):
                out.append(fn(*ws.gather(tidx)))
        except Exception as exc:
            if not is_oom(exc) or batch_size <= 1:
                raise
            oom = True  # retried outside the handler so the failed batch's tensors are released first
        if not oom:
            return out, batch_size
        out = []
        batch_size = max(1, batch_size // 2)
        _free(ws.device)


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
    """Standardized predictions ``(n, H, T)`` for windows ``idx`` (the batch halves while it does not fit)."""

    model.eval()

    def one(x: torch.Tensor, _y: torch.Tensor) -> np.ndarray:
        with _autocast(ws.device, amp):
            return model(x).float().cpu().numpy()

    parts, _ = _over_batches(ws, idx, batch_size, one)
    return np.concatenate(parts) if parts else np.zeros((0, len(ws.horizons), ws.y.shape[1]), np.float32)


@torch.no_grad()
def _val_loss(model: nn.Module, ws: WindowSet, idx: np.ndarray, settings: TrainSettings,
              loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              batch_size: int | None = None) -> tuple[float, float, int]:
    """Validation loss and MAE (model space) on windows ``idx``, and the batch size that fitted in memory."""

    model.eval()

    def one(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, int]:
        with _autocast(ws.device, settings.amp):
            pred = model(x)
        pred = pred.float()
        return float(loss_fn(pred, y)) * len(x), float((pred - y).abs().mean()) * len(x), len(x)

    parts, used = _over_batches(ws, idx, batch_size or settings.eval_batch_size, one)
    count = sum(p[2] for p in parts)
    return sum(p[0] for p in parts) / max(count, 1), sum(p[1] for p in parts) / max(count, 1), used


def _lr_factor(settings: TrainSettings) -> Callable[[int], float]:
    """Per-epoch multiplier of ``lr`` for the cosine and constant schedules (epoch counted from 0)."""

    warm = max(0, int(settings.warmup_epochs))
    span = max(1, settings.max_epochs - warm)

    def factor(epoch: int) -> float:
        if epoch < warm:
            return (epoch + 1) / (warm + 1)
        if settings.lr_schedule == "constant":
            return 1.0
        progress = min(1.0, (epoch - warm) / span)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def train_model(
    model: nn.Module,
    train: WindowSet,
    val: WindowSet,
    settings: TrainSettings,
    *,
    seed: int = 0,
    log: Callable[[str], None] = print,
    on_epoch: Callable[[int, float], bool] | None = None,
) -> dict[str, Any]:
    """Train with early stopping on validation loss; restores the best weights.

    ``on_epoch(epoch, val_mae)`` is called after every epoch; returning True stops training with status
    ``PRUNED`` (a hyperparameter search gave up on this trial). The best weights so far are still restored.
    """

    device = train.device
    configure_backend(device)
    torch.manual_seed(seed)
    model.to(device)
    runner = torch.compile(model) if settings.compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    if settings.lr_schedule == "plateau":
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                                    patience=max(1, settings.patience // 3))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor(settings))
    loss_fn = _loss_fn(settings)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    val_idx = evenly_spaced(len(val), settings.val_windows)
    per_epoch = min(settings.windows_per_epoch, len(train))
    weights = getattr(train, "window_weights", None)
    draw = torch.as_tensor(weights, dtype=torch.float32, device=device) if weights is not None else None
    if draw is not None:
        per_epoch = settings.windows_per_epoch
        log(f"recent runs weighted: half-life {settings.recency_halflife_runs:g} runs, "
            f"oldest run weight {float(draw.min() / draw.max()):.3f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float]] = []
    best_loss, best_state, best_epoch, bad_epochs = math.inf, None, 0, 0
    parts = 1  # each batch is computed in this many pieces once it runs out of GPU memory (gradients added up)
    eval_batch = settings.eval_batch_size

    def step(batch: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for piece in torch.tensor_split(batch, parts):
            x, y = train.gather(piece)
            with _autocast(device, settings.amp):
                pred = runner(x)
            loss = loss_fn(pred.float(), y) * (len(piece) / len(batch))  # a mean over the whole batch
            loss.backward()
            total += loss.detach()
        return total

    status = "MAX_EPOCHS"
    started = time.perf_counter()
    train_seconds = 0.0
    windows_seen = 0
    for epoch in range(1, settings.max_epochs + 1):
        runner.train()
        epoch_start = time.perf_counter()
        if draw is not None:  # weighted draw with replacement (windows of recent runs come up more often)
            order = torch.multinomial(draw, per_epoch, replacement=True, generator=generator)
        else:
            order = torch.randperm(len(train), device=device, generator=generator)[:per_epoch]
        loss_sum = torch.zeros((), device=device)
        batches = 0
        for start in range(0, per_epoch, settings.batch_size):
            batch = order[start : start + settings.batch_size]
            if len(batch) < 2:  # BatchNorm needs two samples
                continue
            while True:
                oom = False
                try:
                    loss = step(batch)
                except Exception as exc:
                    # BatchNorm needs two samples per piece; past that the setting cannot train on this GPU.
                    if not is_oom(exc) or len(batch) // (parts * 2) < 2:
                        raise
                    oom = True
                if not oom:
                    break
                optimizer.zero_grad(set_to_none=True)
                _free(device)
                parts *= 2
                log(f"GPU 메모리 부족: batch {len(batch)}을 {parts}조각으로 나눠 계산 (실효 batch는 그대로)")
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
        val_loss, val_mae, used = _val_loss(runner, val, val_idx, settings, loss_fn, eval_batch)
        if used < eval_batch:
            log(f"GPU 메모리 부족: 검증 batch {eval_batch} → {used}")
            eval_batch = used
        lr = optimizer.param_groups[0]["lr"]  # the rate this epoch trained with
        if settings.lr_schedule == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae,
                        "lr": lr, "seconds": round(time.perf_counter() - epoch_start, 3)})
        improved = val_loss < best_loss * (1 - 1e-4)
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        log(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}  "
            f"{history[-1]['seconds']:.1f}s{'  *' if improved else ''}")
        if on_epoch is not None and on_epoch(epoch, val_mae):
            status = "PRUNED"
            log(f"epoch {epoch}: stopped by the search (pruned)")
            break
        if bad_epochs >= settings.patience and epoch >= settings.min_epochs:
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
        "best_val_mae": min((h["val_mae"] for h in history), default=None),
        "history": history,
        "train_seconds": round(train_seconds, 2),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "windows_seen": windows_seen,
        "windows_per_second": round(windows_seen / train_seconds, 1) if train_seconds else None,
        "peak_gpu_mb": round(peak_mb, 1) if peak_mb is not None else None,
        "batch_size": settings.batch_size,
        "batch_parts": parts,
        "eval_batch_size": eval_batch,
    }
