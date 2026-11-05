"""What the In tab shows about the data: which runs, rows and columns were left out and why, and what the
preprocessing did to one validation run.

Everything comes from finished outputs (the data overview tables, the split file, the prepared bundles'
metadata) except the example, which reads one validation run's trace and passes it through the training
pipeline's own ``run_arrays`` — so the picture is what training saw, not a re-implementation. No torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.catalog import CYCLE_START_STEP
from ..data.columns import load_column_dictionary
from ..data.raw import DROPPED_COLUMNS, META_COLUMNS
from ..paths import LabPaths

# CSV columns that are not signals: the four metadata columns, date and time, the export's empty last column.
INFO_COLUMNS = len(META_COLUMNS) + 2 + len(DROPPED_COLUMNS)
ISSUE_NAMES = {"MISSING_RUN_FILE": "파일 없음", "BACKWARD_TIMESTAMP_JUMP": "시간 역행", "SAMPLING_GAP": "0.5초 간격 끊김",
               "CYCLE_COUNT": "사이클 수가 다름", "THICKNESS_OUTLIER": "두께 이상", "EXSITU_SHEET_FORMULA": "엑셀 수식 차이",
               "ABNORMAL_END": "비정상 종료 기록", "EXSITU_DATE_MISMATCH": "두께 기록 날짜 불일치",
               "EXSITU_TITLE_MISMATCH": "두께 기록 제목 불일치", "NO_EXSITU": "두께 기록 없음",
               "EXSITU_MISSING": "두께 파일 없음"}
EXAMPLE_POINTS = 1200  # min/max pairs kept for the whole-run line
ZOOM_SECONDS = 120


@dataclass
class DataView:
    columns: pd.DataFrame | None = None  # data_overview/columns.parquet: status, duplicate_of, name_ko, unit, …
    issues: pd.DataFrame | None = None
    runs: pd.DataFrame | None = None
    bundles: dict[str, dict[str, Any]] = field(default_factory=dict)  # partition → prepared bundle stats (1 s)
    example: dict[str, Any] | None = None


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def _with_dictionary(columns: pd.DataFrame | None) -> pd.DataFrame | None:
    """Names and units from the current column dictionary; the overview keeps the ones it was built with."""

    if columns is None or "column" not in columns:
        return columns
    try:
        dictionary, _ = load_column_dictionary()
    except (OSError, ValueError, KeyError):
        return columns
    ref = dictionary.set_index("column")
    out = columns.copy()
    for name in ("unit", "name_ko"):
        if name in out:
            current = out["column"].map(ref[name].where(ref[name].notna() & (ref[name].astype(str) != "")))
            out[name] = current.where(current.notna(), out[name])
    return out


def bundle_stats(paths: LabPaths, split: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Stats of the newest prepared bundle per partition that holds the whole partition (not a smoke subset)."""

    out: dict[str, tuple[float, dict[str, Any]]] = {}
    root = paths.work_root / "prepared"
    if not split or not root.is_dir():
        return {}
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        part = meta.get("partition")
        if part not in ("train", "validation", "test") or len(meta.get("run_ids", [])) != len(split.get(part) or []):
            continue
        mtime = meta_path.stat().st_mtime
        if part not in out or mtime > out[part][0]:
            out[part] = (mtime, {**(meta.get("stats") or {}), "sample_seconds": meta.get("sample_seconds")})
    return {k: v for k, (_, v) in out.items()}


def _bands(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Runs of True in ``mask`` as (start, end) in the units of ``t``."""

    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(float(t[a]), float(t[min(b, len(t) - 1)])) for a, b in zip(edges[::2], edges[1::2])]


def _envelope(t: np.ndarray, y: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Min and max of each bucket, in time order, so short spikes survive thinning."""

    if len(y) <= 2 * buckets:
        return t, y
    xs, ys = [], []
    for part_t, part_y in zip(np.array_split(t, buckets), np.array_split(y, buckets)):
        lo, hi = int(np.argmin(part_y)), int(np.argmax(part_y))
        for i in sorted({lo, hi}):
            xs.append(part_t[i])
            ys.append(part_y[i])
    return np.asarray(xs), np.asarray(ys)


def example_run(paths: LabPaths, run_id: str, column: str = "VG11") -> dict[str, Any] | None:
    """One run through the training pipeline's ``run_arrays``: the 1 s series with its process and ALD-cycle
    flags, and two minutes inside the ALD cycles at 0.5 s next to the 1 s averages built from them."""

    from ..training.prepare import PrepareError, TraceSource, run_arrays

    try:
        frame = TraceSource(paths).load(run_id)
    except (PrepareError, OSError, ValueError):
        return None
    if column not in frame.columns:
        return None
    arr = run_arrays(frame, [column], factor=2)
    ts = frame["timestamp"]
    t0 = ts.iloc[0]
    rows = np.asarray(arr["source_rows"])
    minutes = (ts.iloc[rows] - t0).dt.total_seconds().to_numpy() / 60
    values = np.asarray(arr["values"])[:, 0].astype(float)
    t_env, y_env = _envelope(minutes, values, EXAMPLE_POINTS)
    out: dict[str, Any] = {
        "run_id": run_id, "column": column, "raw_rows": len(frame), "rows_1s": len(values),
        "segments": len(arr["segment_lengths"]), "nan_filled": int(arr["n_nan"]),
        "t": t_env.round(3).tolist(), "y": y_env.round(4).tolist(),
        "process": _bands(minutes, np.asarray(arr["in_process"])), "cycle": _bands(minutes, np.asarray(arr["in_cycle"])),
        "process_rows": int(np.asarray(arr["in_process"]).sum()), "cycle_rows": int(np.asarray(arr["in_cycle"]).sum()),
    }
    flashes = np.flatnonzero((frame["step_name"] == CYCLE_START_STEP).to_numpy())
    if len(flashes):
        # a DCS flash in the middle of the cycles, with 10 s of the step before it
        start = max(int(flashes[len(flashes) // 2]) - 20, 0)
        stop = min(start + int(ZOOM_SECONDS / 0.5), len(frame))
        raw_t = (ts.iloc[start:stop] - ts.iloc[start]).dt.total_seconds().to_numpy()
        inside = (rows >= start) & (rows < stop)
        # a 1 s value averages the block's two rows: draw it at the block's middle (0.25 s before its last row)
        block_t = (ts.iloc[rows[inside]] - ts.iloc[start]).dt.total_seconds().to_numpy() - 0.25
        out.update({
            "zoom_raw_t": raw_t.round(2).tolist(), "zoom_raw_y": frame[column].iloc[start:stop].astype(float).round(4).tolist(),
            "zoom_t": block_t.round(2).tolist(), "zoom_y": values[inside].round(4).tolist(),
            "zoom_steps": [(float((ts.iloc[i] - ts.iloc[start]).total_seconds()), str(frame["step_name"].iloc[i]))
                           for i in range(start, stop) if i == start or frame["step_name"].iloc[i] != frame["step_name"].iloc[i - 1]],
        })
    return out


def load_data_view(paths: LabPaths, split: dict[str, Any] | None, *, examples: bool = True) -> DataView:
    overview = paths.overview_dir
    view = DataView(columns=_with_dictionary(_read(overview / "columns.parquet")), issues=_read(overview / "issues.parquet"),
                    runs=_read(overview / "runs.parquet"), bundles=bundle_stats(paths, split))
    if examples and split and split.get("validation"):
        view.example = example_run(paths, str(split["validation"][0]))
    return view
