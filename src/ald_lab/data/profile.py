"""Column profiling across runs: ranges, constants, exact duplicates, near-copies.

Statistics are accumulated per run so all 74 runs never need to sit in memory
at once. Exact duplicates are columns whose values are byte-identical in every
run; near-copies are pairs with |Pearson r| at or above a threshold on a
row subsample (e.g. ``AUXMon_VG11`` is ``VG11 / 10``).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NEAR_COPY_THRESHOLD = 0.999


@dataclass
class _ColumnAccumulator:
    count: int = 0
    missing: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf
    runs_constant: int = 0
    max_unique_per_run: int = 0
    binary: bool = True
    constant_values: set[float] = field(default_factory=set)


class ColumnProfiler:
    def __init__(self, columns: list[str], subsample_every: int = 20) -> None:
        self.columns = list(columns)
        self.subsample_every = subsample_every
        self._acc = {c: _ColumnAccumulator() for c in self.columns}
        self._hashes: dict[str, list[str]] = defaultdict(list)
        self._samples: list[np.ndarray] = []
        self.n_runs = 0

    def add_run(self, frame: pd.DataFrame) -> None:
        self.n_runs += 1
        values = frame[self.columns].to_numpy(dtype=np.float64)
        for j, column in enumerate(self.columns):
            col = values[:, j]
            acc = self._acc[column]
            finite = col[np.isfinite(col)]
            acc.count += finite.size
            acc.missing += col.size - finite.size
            if finite.size:
                acc.total += float(finite.sum())
                acc.total_sq += float(np.square(finite).sum())
                lo, hi = float(finite.min()), float(finite.max())
                acc.minimum = min(acc.minimum, lo)
                acc.maximum = max(acc.maximum, hi)
                unique = np.unique(finite)
                acc.max_unique_per_run = max(acc.max_unique_per_run, unique.size)
                if acc.binary and not np.isin(unique, (0.0, 1.0)).all():
                    acc.binary = False
                if lo == hi:
                    acc.runs_constant += 1
                    acc.constant_values.add(lo)
            self._hashes[column].append(hashlib.sha1(np.ascontiguousarray(col).tobytes()).hexdigest())
        self._samples.append(values[:: self.subsample_every])

    def _duplicate_groups(self) -> dict[str, str]:
        """Map each duplicate column to the first column (in file order) it equals."""

        canonical: dict[tuple[str, ...], str] = {}
        duplicate_of: dict[str, str] = {}
        for column in self.columns:
            key = tuple(self._hashes[column])
            if key in canonical:
                duplicate_of[column] = canonical[key]
            else:
                canonical[key] = column
        return duplicate_of

    def near_copy_pairs(self, threshold: float = NEAR_COPY_THRESHOLD) -> pd.DataFrame:
        sample = np.concatenate(self._samples, axis=0)
        std = np.nanstd(sample, axis=0)
        usable = [i for i, s in enumerate(std) if np.isfinite(s) and s > 0]
        sub = pd.DataFrame(sample[:, usable], columns=[self.columns[i] for i in usable])
        corr = sub.corr().to_numpy()
        names = list(sub.columns)
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = corr[i, j]
                if np.isfinite(r) and abs(r) >= threshold:
                    pairs.append((names[i], names[j], float(r)))
        return pd.DataFrame(pairs, columns=["column_a", "column_b", "pearson_r"])

    def result(self) -> pd.DataFrame:
        duplicate_of = self._duplicate_groups()
        rows = []
        for column in self.columns:
            acc = self._acc[column]
            mean = acc.total / acc.count if acc.count else np.nan
            var = acc.total_sq / acc.count - mean**2 if acc.count else np.nan
            globally_constant = (
                acc.count > 0 and acc.runs_constant == self.n_runs and len(acc.constant_values) == 1
            )
            if acc.count == 0:
                status = "ALL_MISSING"
            elif globally_constant:
                status = "CONSTANT"
            elif column in duplicate_of:
                status = "DUPLICATE"
            else:
                status = "OK"
            rows.append(
                {
                    "column": column,
                    "status": status,
                    "duplicate_of": duplicate_of.get(column) if status == "DUPLICATE" else None,
                    "count": acc.count,
                    "missing": acc.missing,
                    "min": acc.minimum if acc.count else np.nan,
                    "max": acc.maximum if acc.count else np.nan,
                    "mean": mean,
                    "std": float(np.sqrt(max(var, 0.0))) if acc.count else np.nan,
                    "binary": bool(acc.binary and acc.count),
                    "runs_constant": acc.runs_constant,
                    "max_unique_per_run": acc.max_unique_per_run,
                }
            )
        return pd.DataFrame(rows)
