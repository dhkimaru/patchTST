"""Per-run catalog: timing, sampling integrity, recipe/step structure, quality flags."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .raw import run_no_from_id

EXPECTED_INTERVAL_S = 0.5
INTERVAL_TOLERANCE_S = 0.01
# Step that opens each ALD cycle (DCS flash). Used to count cycles per run.
CYCLE_START_STEP = "SI-FL1"
# Steps that make up one ALD deposition cycle (DCS supply → purge → NH3 + plasma → purge).
CYCLE_STEPS = ("SI-FL1", "SI-EVA1", "CHANGE", "N-PRE1", "N-FL1", "N-FL2", "N-EVA1", "CLOSE1", "pre-NH3P")


def step_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive rows with the same (recipe, step) into segments."""

    key = frame["recipe"] + "\x1f" + frame["step_id"].astype(str)
    segment_id = key.ne(key.shift()).cumsum()
    grouped = frame.groupby(segment_id, sort=False)
    segments = pd.DataFrame(
        {
            "recipe": grouped["recipe"].first(),
            "step_id": grouped["step_id"].first(),
            "step_name": grouped["step_name"].first(),
            "start": grouped["timestamp"].first(),
            "end": grouped["timestamp"].last(),
            "rows": grouped.size(),
        }
    ).reset_index(drop=True)
    segments["duration_s"] = segments["rows"] * EXPECTED_INTERVAL_S
    return segments


def sampling_report(timestamps: pd.Series) -> dict[str, float | int]:
    delta = timestamps.diff().dt.total_seconds().to_numpy()[1:]
    gap = delta > EXPECTED_INTERVAL_S + INTERVAL_TOLERANCE_S
    backward = delta < 0
    duplicate = delta == 0
    return {
        "n_gaps": int(gap.sum()),
        "max_gap_s": float(delta[gap].max()) if gap.any() else 0.0,
        "n_backward_jumps": int(backward.sum()),
        "min_backward_s": float(delta[backward].min()) if backward.any() else 0.0,
        "n_duplicate_timestamps": int(duplicate.sum()),
        "regular_fraction": float(
            np.mean(np.abs(delta - EXPECTED_INTERVAL_S) <= INTERVAL_TOLERANCE_S)
        )
        if len(delta)
        else 1.0,
    }


def process_window(segments: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Start of the first ``START`` step and start of the last ``END`` step after it.

    Trace files carry STANDBY padding before and after the process; operator
    logs (and the ex-situ workbook) record this START..END window instead.
    """

    starts = segments.index[segments["step_name"] == "START"]
    if len(starts) == 0:
        return None, None
    first = starts[0]
    ends = segments.index[(segments["step_name"] == "END") & (segments.index > first)]
    process_start = segments.at[first, "start"]
    process_end = segments.at[ends[-1], "start"] if len(ends) else None
    return process_start, process_end


def catalog_run(run_id: str, frame: pd.DataFrame, source: Path | None = None) -> dict[str, object]:
    segments = step_segments(frame)
    report = sampling_report(frame["timestamp"])
    process_start, process_end = process_window(segments)
    recipe_rows = frame["recipe"].value_counts()
    flags: list[str] = []
    if report["n_backward_jumps"]:
        flags.append("BACKWARD_TIMESTAMP_JUMP")
    if report["n_duplicate_timestamps"]:
        flags.append("DUPLICATE_TIMESTAMPS")
    if report["n_gaps"]:
        flags.append("SAMPLING_GAP")
    row: dict[str, object] = {
        "run_id": run_id,
        "run_no": run_no_from_id(run_id),
        "source_file": source.name if source is not None else f"{run_id}.csv",
        "size_bytes": source.stat().st_size if source is not None else None,
        "rows": int(len(frame)),
        "start": frame["timestamp"].min(),
        "end": frame["timestamp"].max(),
        "duration_h": float((frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 3600),
        "process_start": process_start,
        "process_end": process_end,
        "n_step_segments": int(len(segments)),
        "n_unique_steps": int(frame["step_id"].nunique()),
        "n_cycles": int((segments["step_name"] == CYCLE_START_STEP).sum()),
        "main_recipe": str(recipe_rows.index[0]),
        "recipe_rows": {str(k): int(v) for k, v in recipe_rows.items()},
        "quality_flags": flags,
        **report,
    }
    return row


def build_catalog(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    catalog = pd.DataFrame(list(rows)).sort_values("run_no").reset_index(drop=True)
    expected = set(range(int(catalog["run_no"].min()), int(catalog["run_no"].max()) + 1))
    catalog.attrs["missing_run_numbers"] = sorted(expected - set(catalog["run_no"]))
    return catalog


def step_table(segments_by_run: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate step segments over runs: how often and how long each step runs."""

    frames = [seg.assign(run_id=run_id) for run_id, seg in segments_by_run.items()]
    allseg = pd.concat(frames, ignore_index=True)
    table = (
        allseg.groupby(["recipe", "step_id", "step_name"])
        .agg(
            occurrences=("rows", "size"),
            runs=("run_id", "nunique"),
            total_rows=("rows", "sum"),
            median_duration_s=("duration_s", "median"),
        )
        .reset_index()
        .sort_values("total_rows", ascending=False)
    )
    return table
