"""Read equipment trace CSVs (``standard_trace_NNN.csv``) into typed frames.

Each CSV is one batch run sampled every 0.5 s. The first six columns are
metadata; the rest are numeric signals. Parsed runs are cached as parquet
next to the work root, keyed by file size and modification time.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = {
    "No.": "row_no",
    "Recipe(Table) Name": "recipe",
    "Step ID": "step_id",
    "Step Name": "step_name",
}
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S.%f"
# Trailing empty column produced by the equipment export.
DROPPED_COLUMNS = ("Unnamed: 206",)
RUN_FILE_PATTERN = re.compile(r"standard_trace_(\d{3})\.csv")
CACHE_VERSION = "v1"


class TraceFormatError(ValueError):
    """Raised when a trace CSV does not match the expected export layout."""


def run_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    if RUN_FILE_PATTERN.fullmatch(name) is None:
        raise TraceFormatError(f"unexpected trace file name: {name}")
    return name[: -len(".csv")]


def run_no_from_id(run_id: str) -> int:
    return int(run_id.rsplit("_", 1)[1])


def list_trace_files(data_root: Path) -> list[Path]:
    files = sorted(p for p in data_root.iterdir() if RUN_FILE_PATTERN.fullmatch(p.name))
    if not files:
        raise TraceFormatError(f"no standard_trace_NNN.csv files in {data_root}")
    return files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trace_csv(path: str | Path) -> pd.DataFrame:
    """Parse one trace CSV into ``timestamp`` + metadata + float32 signals."""

    path = Path(path)
    frame = pd.read_csv(path, low_memory=False)
    missing = [c for c in (*META_COLUMNS, DATE_COLUMN, TIME_COLUMN) if c not in frame.columns]
    if missing:
        raise TraceFormatError(f"{path.name} is missing metadata columns {missing}")
    timestamp = pd.to_datetime(
        frame[DATE_COLUMN].astype(str) + " " + frame[TIME_COLUMN].astype(str),
        format=TIMESTAMP_FORMAT,
    )
    signal_columns = [
        c
        for c in frame.columns
        if c not in META_COLUMNS and c not in (DATE_COLUMN, TIME_COLUMN) and c not in DROPPED_COLUMNS
    ]
    signals = frame[signal_columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    meta = frame[list(META_COLUMNS)].rename(columns=META_COLUMNS)
    meta["row_no"] = meta["row_no"].astype(np.int64)
    meta["step_id"] = meta["step_id"].astype(np.int64)
    meta["recipe"] = meta["recipe"].astype(str)
    meta["step_name"] = meta["step_name"].astype(str)
    out = pd.concat([pd.DataFrame({"timestamp": timestamp}), meta, signals], axis=1)
    return out


def signal_columns(frame: pd.DataFrame) -> list[str]:
    reserved = {"timestamp", *META_COLUMNS.values()}
    return [c for c in frame.columns if c not in reserved]


def _cache_path(cache_dir: Path, source: Path) -> Path:
    stat = source.stat()
    key = f"{CACHE_VERSION}-{stat.st_size}-{int(stat.st_mtime)}"
    return cache_dir / f"{source.stem}.{key}.parquet"


def load_trace(source: str | Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Read a trace, using (and filling) the parquet cache when given."""

    source = Path(source)
    if cache_dir is None:
        return read_trace_csv(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(cache_dir, source)
    if cached.is_file():
        return pd.read_parquet(cached)
    frame = read_trace_csv(source)
    for stale in cache_dir.glob(f"{source.stem}.*.parquet"):
        stale.unlink()
    tmp = cached.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(cached)
    return frame
