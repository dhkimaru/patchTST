"""Train-only scaling and device-resident sliding windows.

The previous project cut every window on the CPU inside a ``DataLoader``
(one Python call per sample, batch 128) and reached about 3,600 windows/s on
an RTX 4090 while the GPU sat at 150 MB. Here a partition's standardized
series lives on the GPU as one ``(rows, channels)`` tensor; a batch of
windows is a single index-gather ``x[ends[:, None] + offsets]``, so the
input pipeline costs microseconds per batch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .prepare import Bundle

REGIONS = ("all", "process", "cycle", "noncycle")  # noncycle: process rows outside ALD cycles


@dataclass
class Scaler:
    """Per-column standardization fitted on train rows only."""

    columns: list[str]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray, index: list[int] | None = None) -> np.ndarray:
        mean = self.mean if index is None else self.mean[index]
        std = self.std if index is None else self.std[index]
        return np.asarray(values, dtype=np.float64) * std + mean

    def to_dict(self) -> dict[str, object]:
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, doc: dict[str, object]) -> Scaler:
        return cls(list(doc["columns"]), np.asarray(doc["mean"], dtype=np.float64),
                   np.asarray(doc["std"], dtype=np.float64))


def fit_scaler(bundle: Bundle, columns: list[str], chunk: int = 262_144) -> Scaler:
    index = bundle.column_index(columns)
    total = np.zeros(len(index))
    total_sq = np.zeros(len(index))
    for start in range(0, bundle.n_rows, chunk):
        block = np.asarray(bundle.values[start : start + chunk][:, index], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += (block**2).sum(axis=0)
    n = max(bundle.n_rows, 1)
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 0.0))
    std[std < 1e-8] = 1.0
    return Scaler(list(columns), mean, std)


def constant_columns(bundle: Bundle, columns: list[str]) -> list[str]:
    """Columns that never change in this partition (useless as inputs)."""

    index = bundle.column_index(columns)
    lo = np.full(len(index), np.inf)
    hi = np.full(len(index), -np.inf)
    for start in range(0, bundle.n_rows, 262_144):
        block = np.asarray(bundle.values[start : start + 262_144][:, index])
        lo = np.minimum(lo, block.min(axis=0))
        hi = np.maximum(hi, block.max(axis=0))
    return [c for c, a, b in zip(columns, lo, hi) if b - a <= 1e-9]


COPY_THRESHOLD = 0.9999


def near_copies(bundle: Bundle, columns: list[str], threshold: float = COPY_THRESHOLD,
                max_rows: int = 200_000) -> pd.DataFrame:
    """Column pairs with |r| ≥ ``threshold`` on (a sample of) this partition's rows.

    A virtual sensor must not be "estimated" from a copy of itself, so these pairs are removed
    from the inputs together with the target. Copies inside a sensor family (VG11 ↔ PressAct ↔
    AUXMon_VG11) already go with the family; this catches duplicates across families. The
    threshold is 0.9999, not 0.999: adjacent heater zones share the temperature ramp (r ≈ 0.9998
    on train runs) yet are different sensors, and estimating a zone from its neighbours is a
    legitimate virtual sensor.
    """

    rows = evenly_spaced(bundle.n_rows, max_rows)
    values = np.asarray(np.asarray(bundle.values)[rows][:, bundle.column_index(columns)], dtype=np.float64)
    varies = values.std(axis=0) > 0
    names = [c for c, v in zip(columns, varies) if v]
    corr = np.corrcoef(values[:, varies], rowvar=False)
    a, b = np.nonzero(np.triu(np.abs(corr) >= threshold, k=1))
    return pd.DataFrame({"column_a": [names[i] for i in a], "column_b": [names[j] for j in b],
                         "pearson_r": corr[a, b]})


def window_ends(bundle: Bundle, context: int, horizons: tuple[int, ...], region: str = "all") -> np.ndarray:
    """Rows that can end a window: the whole window and every horizon stay inside one segment."""

    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}: {region}")
    rows = np.arange(bundle.n_rows, dtype=np.int64)
    bounds = bundle.seg_bounds[np.asarray(bundle.seg_id)]
    ok = (rows - context + 1 >= bounds[:, 0]) & (rows + max(horizons) < bounds[:, 1])
    if region == "process":
        ok &= np.asarray(bundle.in_process)
    elif region == "cycle":
        ok &= np.asarray(bundle.in_cycle)
    elif region == "noncycle":
        ok &= np.asarray(bundle.in_process) & ~np.asarray(bundle.in_cycle)
    return rows[ok]


def evenly_spaced(n: int, limit: int | None) -> np.ndarray:
    """At most ``limit`` positions spread evenly over ``range(n)``."""

    if limit is None or n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, limit).round().astype(np.int64))


class WindowSet:
    """One partition's standardized inputs/targets on a device, plus its valid window ends.

    ``x``: ``(rows, C)`` inputs, ``y``: ``(rows, T)`` targets, both standardized.
    A window ending at row ``e`` sees ``x[e-L+1 .. e]`` and is scored on
    ``y[e + h]`` for each horizon ``h`` (``h = 0`` is a virtual-sensor nowcast).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ends: np.ndarray,
        *,
        context: int,
        horizons: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> None:
        if len(ends) == 0:
            raise ValueError("no valid windows; check context length, horizons and region")
        self.device = torch.device(device)
        self.x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device=self.device)
        self.ends_np = np.asarray(ends, dtype=np.int64)
        self.ends = torch.as_tensor(self.ends_np, device=self.device)
        self.context = int(context)
        self.horizons = tuple(int(h) for h in horizons)
        self.offsets = torch.arange(-self.context + 1, 1, device=self.device)
        self.h = torch.as_tensor(self.horizons, device=self.device)

    def __len__(self) -> int:
        return len(self.ends_np)

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[1])

    def gather(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``idx`` indexes ``ends``; returns ``x (B, L, C)`` and ``y (B, H, T)``."""

        e = self.ends[idx]
        x = self.x[e[:, None] + self.offsets]
        y = self.y[e[:, None] + self.h]
        return x, y

    def target_rows(self, idx: np.ndarray) -> np.ndarray:
        """Bundle rows each prediction refers to, ``(B, H)``."""

        return self.ends_np[idx][:, None] + np.asarray(self.horizons)[None, :]

    def iter_batches(self, idx: np.ndarray, batch_size: int):
        for start in range(0, len(idx), batch_size):
            part = idx[start : start + batch_size]
            yield part, torch.as_tensor(part, device=self.device)


def offset_targets(ws: WindowSet, offset: np.ndarray, scale: float = 1.0) -> WindowSet:
    """The same windows with targets ``(y - offset) / scale``, sharing the inputs on the device.

    Used to train a model on the correction to a baseline (the recipe template) instead of the raw
    value. ``scale`` puts that correction on a unit scale: VG11 spans 1–760 Torr, so in globally
    standardized units a 0.05 Torr correction inside ALD cycles would be invisible to the loss.
    """

    out = copy.copy(ws)
    out.y = (ws.y - torch.as_tensor(offset, dtype=torch.float32, device=ws.device)) / scale
    return out


def residual_scale(ws: WindowSet, offset: np.ndarray) -> float:
    """Spread of ``y - offset`` over the windows' target rows (train partition)."""

    rows = ws.target_rows(np.arange(len(ws))).reshape(-1)
    resid = ws.y.cpu().numpy()[rows] - np.asarray(offset)[rows]
    return float(max(resid.std(), 1e-6))
