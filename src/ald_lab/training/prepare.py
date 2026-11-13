"""Turn raw runs into model-ready arrays once, and cache them.

A *bundle* holds every usable signal of one split partition (train,
validation or test) at the chosen sampling interval, plus the row metadata
that windows, baselines and metrics need:

- contiguous *segments*: a run is cut wherever the 0.5 s clock breaks (gaps,
  backward jumps), so no window ever spans a run boundary or a time gap;
- ``in_cycle`` (row is inside an ALD cycle step) and ``in_process``
  (between the run's START and END steps);
- ``step_key`` / ``step_pos`` (recipe step and position inside it) for the
  recipe-template baseline and the step-transition breakdown.

Bundles are plain ``.npy`` files under ``work/prepared/`` so training
processes memory-map them instead of re-reading 2.4 GB of CSV.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.catalog import CYCLE_STEPS, EXPECTED_INTERVAL_S, INTERVAL_TOLERANCE_S, process_window, step_segments
from ..data.columns import load_column_dictionary
from ..data.raw import list_trace_files, load_trace, run_id_from_path, signal_columns
from ..data.splits import RunSplit
from ..paths import LabPaths

BUNDLE_VERSION = "b2"  # b2: step names per step key (per-step error breakdown)
# Template positions are shared past this many 0.5 s rows (20 min), as in the variability analysis.
POSITION_CAP_ROWS = 2400
ARRAYS = ("values", "run_index", "seg_id", "seg_bounds", "in_cycle", "in_process", "step_key", "step_pos")
# How 0.5 s rows are grouped into blocks: "run" counts blocks from the start of each continuous segment
# (the original pipeline); "step" restarts them at every step change, so no block mixes two steps
# (a step with an odd number of rows ends in a single-row block).
BIN_ALIGNS = ("run", "step")


class PrepareError(ValueError):
    """Raised when runs cannot be turned into a bundle."""


@dataclass
class Bundle:
    partition: str
    sample_seconds: float
    columns: list[str]
    run_ids: list[str]
    values: np.ndarray  # (N, C) float32, physical units
    run_index: np.ndarray  # (N,) int32 → run_ids
    seg_id: np.ndarray  # (N,) int32 → seg_bounds
    seg_bounds: np.ndarray  # (S, 2) int64 [start, end) rows
    in_cycle: np.ndarray  # (N,) bool
    in_process: np.ndarray  # (N,) bool
    step_key: np.ndarray  # (N,) int64, crc32 of "recipe|step_id"
    step_pos: np.ndarray  # (N,) int32, samples since the step started (capped)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    def column_index(self, names: list[str]) -> list[int]:
        lookup = {c: i for i, c in enumerate(self.columns)}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise PrepareError(f"columns not in the prepared data: {missing}")
        return [lookup[n] for n in names]


def usable_columns(column_dictionary: pd.DataFrame | None = None) -> list[str]:
    """Signals the dictionary marks OK (drops constant and duplicate exports)."""

    columns, _ = load_column_dictionary() if column_dictionary is None else (column_dictionary, None)
    return columns.loc[columns["status"] == "OK", "column"].tolist()


def _step_key(recipe: pd.Series, step_id: pd.Series) -> np.ndarray:
    combo = recipe.astype(str) + "|" + step_id.astype(str)
    codes, uniques = pd.factorize(combo)
    hashed = np.array([zlib.crc32(u.encode("utf-8")) for u in uniques], dtype=np.int64)
    return hashed[codes]


def _block_starts(a: int, b: int, factor: int, step_change: np.ndarray, align: str) -> np.ndarray:
    """First raw row of every block in the segment [a, b)."""

    if align == "run" or factor == 1:
        n = (b - a) // factor * factor
        return a + factor * np.arange(n // factor)
    cuts = a + 1 + np.flatnonzero(step_change[a + 1 : b])
    starts = [np.arange(s, e, factor) for s, e in zip(np.r_[a, cuts], np.r_[cuts, b])]
    return np.concatenate(starts)


def run_arrays(frame: pd.DataFrame, columns: list[str], factor: int, align: str = "run",
               ) -> dict[str, np.ndarray | int]:
    """Arrays for one run: contiguous segments, block-averaged by ``factor`` rows (see ``BIN_ALIGNS``)."""

    if align not in BIN_ALIGNS:
        raise PrepareError(f"bin_align must be one of {BIN_ALIGNS}: {align!r}")

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PrepareError(f"run is missing columns {missing}")
    signals = frame[columns]
    n_nan = int(signals.isna().to_numpy().sum())
    values = signals.ffill().bfill().fillna(0.0).to_numpy(np.float32)

    delta = frame["timestamp"].diff().dt.total_seconds().to_numpy()
    breaks = np.flatnonzero(np.abs(delta[1:] - EXPECTED_INTERVAL_S) > INTERVAL_TOLERANCE_S) + 1
    edges = np.concatenate([[0], breaks, [len(frame)]])

    key = _step_key(frame["recipe"], frame["step_id"])
    step_names = {str(int(k)): str(n) for k, n in
                  pd.Series(frame["step_name"].to_numpy(), index=key).groupby(level=0).first().items()}
    step_change = np.r_[True, key[1:] != key[:-1]]
    group = np.cumsum(step_change)
    pos = pd.Series(np.zeros(len(frame))).groupby(group).cumcount().to_numpy()
    pos = np.minimum(pos, POSITION_CAP_ROWS)
    in_cycle = frame["step_name"].isin(CYCLE_STEPS).to_numpy()
    start, end = process_window(step_segments(frame))
    ts = frame["timestamp"]
    in_process = np.zeros(len(frame), dtype=bool)
    if start is not None:
        in_process = (ts >= start).to_numpy() & ((ts < end).to_numpy() if end is not None else True)

    pieces: dict[str, list[np.ndarray]] = {
        k: [] for k in ("values", "in_cycle", "in_process", "step_key", "step_pos", "source_rows", "source_starts")}
    lengths = []
    for a, b in zip(edges[:-1], edges[1:]):
        starts = _block_starts(int(a), int(b), factor, step_change, align)
        if not len(starts):
            continue
        if align == "run":
            last = starts + factor - 1
            block = values[starts[0] : last[-1] + 1]
            if factor > 1:
                block = block.reshape(-1, factor, block.shape[1]).mean(axis=1)
        else:
            last = np.r_[starts[1:], b] - 1
            counts = (last - starts + 1).astype(np.float32)[:, None]
            block = np.add.reduceat(values[a:b], starts - a, axis=0) / counts
        pieces["values"].append(block.astype(np.float32))
        pieces["in_cycle"].append(in_cycle[last])
        pieces["in_process"].append(in_process[last])
        pieces["step_key"].append(key[last])
        pieces["step_pos"].append((pos[last] // factor).astype(np.int32))
        pieces["source_rows"].append(last)  # raw row closing each block (timestamps for predictions)
        pieces["source_starts"].append(starts)  # raw row opening each block (scoring on the 0.5 s rows)
        lengths.append(len(starts))
    if not lengths:
        raise PrepareError("run has no usable rows")
    out: dict[str, np.ndarray | int] = {k: np.concatenate(v) for k, v in pieces.items()}
    out["segment_lengths"] = np.asarray(lengths, dtype=np.int64)
    out["step_names"] = step_names  # type: ignore[assignment]
    out["n_nan"] = n_nan
    out["n_segments_raw"] = len(edges) - 1
    return out


def _bundle_key(split_id: str, partition: str, run_ids: list[str], sample_seconds: float, columns: list[str],
                bin_align: str = "run") -> str:
    doc = {"v": BUNDLE_VERSION, "split": split_id, "part": partition, "runs": run_ids, "dt": sample_seconds, "cols": columns}
    if bin_align != "run":  # the original grouping keeps its existing cache keys
        doc["align"] = bin_align
    payload = json.dumps(doc, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class TraceSource:
    """Find a run's trace: parquet cache first, raw CSV (``ALD_DATA_ROOT``) second."""

    def __init__(self, paths: LabPaths) -> None:
        self.paths = paths
        self.csv: dict[str, Path] = {}
        if paths.data_root is not None and paths.data_root.is_dir():
            self.csv = {run_id_from_path(p): p for p in list_trace_files(paths.data_root)}

    def load(self, run_id: str) -> pd.DataFrame:
        if run_id in self.csv:
            return load_trace(self.csv[run_id], self.paths.trace_cache)
        cached = sorted(self.paths.trace_cache.glob(f"{run_id}.*.parquet"))
        if cached:
            return pd.read_parquet(cached[-1])
        raise PrepareError(
            f"{run_id}: no CSV under ALD_DATA_ROOT and no parquet cache in {self.paths.trace_cache}"
        )


def build_bundle(
    paths: LabPaths,
    split: RunSplit,
    partition: str,
    *,
    sample_seconds: float,
    columns: list[str] | None = None,
    include_test: bool = False,
    rebuild: bool = False,
    workers: int = 4,
    bin_align: str = "run",
) -> Bundle:
    """Load (or build and cache) one partition's bundle."""

    factor = round(sample_seconds / EXPECTED_INTERVAL_S)
    if factor < 1 or abs(factor * EXPECTED_INTERVAL_S - sample_seconds) > 1e-9:
        raise PrepareError(f"sample_seconds must be a multiple of {EXPECTED_INTERVAL_S}: {sample_seconds}")
    if bin_align not in BIN_ALIGNS:
        raise PrepareError(f"bin_align must be one of {BIN_ALIGNS}: {bin_align!r}")
    run_ids = split.runs(partition, include_test=include_test)
    columns = list(columns or usable_columns())
    key = _bundle_key(split.split_id, partition, run_ids, sample_seconds, columns, bin_align)
    tag = "" if bin_align == "run" else f"_{bin_align}bin"
    directory = paths.work_root / "prepared" / f"{split.split_id}_{sample_seconds:g}s{tag}_{partition}_{key}"
    if directory.is_dir() and not rebuild and (directory / "meta.json").is_file():
        return load_bundle(directory)

    source = TraceSource(paths)
    started = time.perf_counter()
    # Columns the export does not contain (another tool version) are dropped and reported.
    present = set(signal_columns(source.load(run_ids[0])))
    missing_columns = [c for c in columns if c not in present]
    columns = [c for c in columns if c in present]

    def one(run_id: str) -> dict[str, np.ndarray | int]:
        frame = source.load(run_id)
        absent = [c for c in columns if c not in signal_columns(frame)]
        if absent:
            raise PrepareError(f"{run_id} lacks columns {absent}")
        return run_arrays(frame, columns, factor, bin_align)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        parts = list(pool.map(one, run_ids))

    bundle = assemble_bundle(partition, sample_seconds, columns, run_ids, parts)
    bundle.stats.update(missing_columns=missing_columns, build_seconds=round(time.perf_counter() - started, 1),
                        key=key, bin_align=bin_align)
    save_bundle(bundle, directory)
    return bundle


def assemble_bundle(partition: str, sample_seconds: float, columns: list[str], run_ids: list[str],
                    parts: list[dict[str, np.ndarray | int]]) -> Bundle:
    """Concatenate per-run arrays (from :func:`run_arrays`) into one bundle."""

    offsets = np.cumsum([0] + [len(p["values"]) for p in parts])
    seg_bounds, seg_id, run_index = [], [], []
    for r, part in enumerate(parts):
        start = offsets[r]
        for length in part["segment_lengths"]:
            seg_id.append(np.full(length, len(seg_bounds), dtype=np.int32))
            seg_bounds.append((start, start + length))
            start += length
        run_index.append(np.full(len(part["values"]), r, dtype=np.int32))
    return Bundle(
        partition=partition,
        sample_seconds=float(sample_seconds),
        columns=list(columns),
        run_ids=list(run_ids),
        values=np.concatenate([p["values"] for p in parts]),
        run_index=np.concatenate(run_index),
        seg_id=np.concatenate(seg_id),
        seg_bounds=np.asarray(seg_bounds, dtype=np.int64),
        in_cycle=np.concatenate([p["in_cycle"] for p in parts]),
        in_process=np.concatenate([p["in_process"] for p in parts]),
        step_key=np.concatenate([p["step_key"] for p in parts]),
        step_pos=np.concatenate([p["step_pos"] for p in parts]),
        stats={
            "runs": len(run_ids),
            "rows": int(offsets[-1]),
            "columns": len(columns),
            "segments": len(seg_bounds),
            "runs_split_by_time_breaks": int(sum(int(p["n_segments_raw"]) > 1 for p in parts)),
            "nan_filled": int(sum(int(p["n_nan"]) for p in parts)),
            "cycle_rows": int(sum(int(p["in_cycle"].sum()) for p in parts)),
            "process_rows": int(sum(int(p["in_process"].sum()) for p in parts)),
            "step_names": {k: v for p in parts for k, v in dict(p.get("step_names") or {}).items()},
        },
    )


def bundle_from_traces(frames: list[tuple[str, pd.DataFrame]], columns: list[str], sample_seconds: float,
                       bin_align: str = "run") -> tuple[Bundle, pd.DataFrame, list[str]]:
    """A bundle from traces outside any split (new runs to predict), not cached.

    Columns a trace lacks are filled with NaN (→ 0 after the usual gap filling) and returned, so the
    caller can refuse missing model inputs but still predict a virtual sensor whose sensor is absent.
    Also returns, per bundle row, the timestamp and step name of the raw row closing its block.
    """

    factor = round(sample_seconds / EXPECTED_INTERVAL_S)
    absent: set[str] = set()
    parts, meta = [], []
    for run_id, frame in frames:
        lacking = [c for c in columns if c not in frame.columns]
        if lacking:
            absent.update(lacking)
            frame = frame.assign(**{c: np.nan for c in lacking})
        part = run_arrays(frame, columns, factor, bin_align)
        rows = np.asarray(part["source_rows"])
        parts.append(part)
        meta.append(pd.DataFrame({"run_id": run_id, "timestamp": frame["timestamp"].to_numpy()[rows],
                                  "step_name": frame["step_name"].to_numpy()[rows]}))
    bundle = assemble_bundle("predict", sample_seconds, columns, [r for r, _ in frames], parts)
    return bundle, pd.concat(meta, ignore_index=True), sorted(absent)


def save_bundle(bundle: Bundle, directory: Path) -> None:
    """Write next to the target and rename; if another process finished first, keep theirs."""

    tmp = directory.with_name(f"{directory.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in ARRAYS:
        np.save(tmp / f"{name}.npy", getattr(bundle, name))
    meta = {
        "version": BUNDLE_VERSION,
        "partition": bundle.partition,
        "sample_seconds": bundle.sample_seconds,
        "columns": bundle.columns,
        "run_ids": bundle.run_ids,
        "stats": bundle.stats,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if (directory / "meta.json").is_file():  # built concurrently by another run, or a forced rebuild
        stale = directory.with_name(f"{directory.name}.old-{os.getpid()}")
        directory.replace(stale)
        shutil.rmtree(stale, ignore_errors=True)
    elif directory.exists():
        shutil.rmtree(directory)
    tmp.replace(directory)


def load_bundle(directory: Path, *, mmap: bool = True) -> Bundle:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None) for name in ARRAYS}
    return Bundle(
        partition=meta["partition"],
        sample_seconds=float(meta["sample_seconds"]),
        columns=list(meta["columns"]),
        run_ids=list(meta["run_ids"]),
        stats=dict(meta.get("stats") or {}, path=str(directory)),
        **arrays,
    )
