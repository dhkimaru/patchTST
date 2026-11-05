"""Parse the ex-situ metrology workbook and link it to trace runs.

The workbook stacks one block per process run on each sheet. A block starts at
a row whose column A reads ``진행날짜`` and whose title sits one row above in
column C. Columns A/B carry run info (date, start/end time, recipe, trace
IDs), columns C/D the recipe conditions, and columns E..L the thickness table:
point number, x, y and one thickness value (Å) per monitored boat slot (WAP).
Summary rows (AVERAGE, MAX, MIN, RANGE, U/F, WtoW) follow the points.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_MARKER = "진행날짜"
N_WAP = 5
CONDITION_FIELDS = {
    "TEMP_U": "temp_u",
    "TEMP_CU": "temp_cu",
    "TEMP_C": "temp_c",
    "TEMP_CL": "temp_cl",
    "TEMP_L": "temp_l",
    "RF F.PW": "rf_forward_power_w",
    "DCS": "dcs_slm",
    "NH3": "nh3_slm",
    "LOOP": "loop",
    "PURGE MODE": "purge_mode",
    "WAP": "wap_slots",
    "Boat Rotation": "boat_rotation_rpm",
    "ToF": "tof_installed",
    "N2 PURGE MODE": "n2_purge_mode",
    "NORMAL END": "normal_end",
}
SUMMARY_LABELS = ("AVERAGE", "MAX", "MIN", "RANGE", "U/F", "WtoW")
# Measurement layouts by ring radius (mm). 9pt: center, 4 at 70, 4 at 145.
# 49pt: center, 8 at 46.7, 16 at 93.3, 24 at 140. Radii differ, so means and
# U/F from different layouts are not directly comparable.
LAYOUTS = {"9pt": (0.0, 70.0, 145.0), "49pt": (0.0, 46.7, 93.3, 140.0)}
RUN_TITLE_PATTERN = re.compile(r"연속진행\s*(\d+)")
LINK_TOLERANCE = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class ExsituTables:
    runs: pd.DataFrame
    points: pd.DataFrame


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    text = str(value).strip().upper().removesuffix("W").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _flag(value: object) -> bool | None:
    text = str(value).strip().upper() if value is not None else ""
    if text == "O":
        return True
    if text == "X":
        return False
    return None


def _as_time(value: object) -> dt.time | None:
    if isinstance(value, dt.datetime):
        return value.time()
    if isinstance(value, dt.time):
        return value
    if isinstance(value, str):
        try:
            return dt.time.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _as_date(value: object) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return None


def _combine(date: dt.date | None, start: dt.time | None, end: dt.time | None):
    if date is None or start is None:
        return None, None
    started = pd.Timestamp(dt.datetime.combine(date, start))
    if end is None:
        return started, None
    ended = pd.Timestamp(dt.datetime.combine(date, end))
    if ended < started:
        ended += pd.Timedelta(days=1)
    return started, ended


def _parse_block(ws, row: int, block_id: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    cell = lambda r, c: ws.cell(r, c).value  # noqa: E731
    info: dict[str, object] = {
        "block_id": block_id,
        "sheet": ws.title,
        "title": str(cell(row - 1, 3) or cell(row - 1, 2) or "").strip(),
        "sheet_row": row,
    }
    trace_ids: list[int] = []
    run_date = start = end = None
    for r in range(row, row + 16):
        key, value = cell(r, 1), cell(r, 2)
        if key == BLOCK_MARKER:
            run_date = _as_date(value)
        elif key == "Start Time":
            start = _as_time(value)
        elif key == "End Time":
            end = _as_time(value)
        elif key == "Recipe":
            info["recipe"] = None if value is None else str(value)
        elif isinstance(key, str) and key.startswith("DATA TIME"):
            info["data_time_s"] = _number(value)
        if key in ("TraceData", None) and _number(value) is not None and r > row + 4:
            trace_ids.append(int(_number(value)))
        label = cell(r, 3)
        if label in CONDITION_FIELDS:
            name = CONDITION_FIELDS[label]
            raw = cell(r, 4)
            if name in ("tof_installed", "n2_purge_mode", "normal_end"):
                info[name] = _flag(raw)
            elif name == "wap_slots":
                info[name] = None if raw is None else str(raw)
            else:
                info[name] = _number(raw)
    info["run_date"] = run_date
    info["start"], info["end"] = _combine(run_date, start, end)
    info["trace_ids"] = ",".join(str(t) for t in trace_ids)

    points: list[dict[str, object]] = []
    summary: dict[str, list[float | None]] = {}
    r = row + 1
    while r <= ws.max_row:
        label = cell(r, 5)
        if cell(r, 1) == BLOCK_MARKER or label == "MAP":
            break
        number = _number(label)
        if number is not None and not isinstance(label, str):
            values = [_number(cell(r, c)) for c in range(8, 8 + N_WAP)]
            if any(v is not None for v in values):
                for wap_index, value in enumerate(values, start=1):
                    points.append(
                        {
                            "block_id": block_id,
                            "point": int(number),
                            "x_mm": _number(cell(r, 6)),
                            "y_mm": _number(cell(r, 7)),
                            "wap_index": wap_index,
                            "thickness_a": value,
                        }
                    )
        elif label in SUMMARY_LABELS:
            summary[str(label)] = [_number(cell(r, c)) for c in range(8, 8 + N_WAP)]
            if label == "WtoW":
                break
        r += 1

    info.update(_block_stats(points, summary))
    return info, points


def _layout_name(radii: list[float], n_points: int) -> str:
    if n_points == 0:
        return "none"
    for name, rings in LAYOUTS.items():
        if len(rings) == len(radii) and np.allclose(rings, radii, atol=1.0):
            return name
    return f"{n_points}pt"


def _block_stats(points: list[dict[str, object]], summary: dict[str, list[float | None]]) -> dict[str, object]:
    """Statistics recomputed from the raw points, plus a cross-check against the sheet.

    Workbook formulas (verified): U/F = range / (2·mean) · 100 per wafer, and
    WtoW = range of the five wafer means / (2·their mean) · 100.
    """

    out: dict[str, object] = {}
    frame = pd.DataFrame(points, columns=["point", "x_mm", "y_mm", "wap_index", "thickness_a"])
    frame = frame.dropna(subset=["thickness_a"])
    n_points = int(frame["point"].nunique())
    radius = np.hypot(frame["x_mm"].astype(float), frame["y_mm"].astype(float)).round(1)
    radii = sorted(set(radius.tolist()))
    out["n_points"] = n_points
    out["layout"] = _layout_name(radii, n_points)
    out["ring_radii_mm"] = "/".join(f"{r:g}" for r in radii) if radii else None
    means: list[float] = []
    max_sheet_diff = 0.0
    for wap in range(1, N_WAP + 1):
        values = frame.loc[frame["wap_index"] == wap, "thickness_a"].to_numpy(dtype=float)
        if values.size == 0:
            for key in ("thk_mean", "thk_range", "uf_pct"):
                out[f"{key}_wap{wap}"] = None
            continue
        mean, rng = float(values.mean()), float(values.max() - values.min())
        out[f"thk_mean_wap{wap}"] = mean
        out[f"thk_range_wap{wap}"] = rng
        out[f"uf_pct_wap{wap}"] = rng / (2 * mean) * 100
        means.append(mean)
        for label, calc in (("AVERAGE", mean), ("U/F", rng / (2 * mean) * 100)):
            sheet = (summary.get(label) or [None] * N_WAP)[wap - 1]
            if sheet is not None:
                max_sheet_diff = max(max_sheet_diff, abs(sheet - calc))
    out["thk_mean"] = float(np.mean(means)) if means else None
    ufs = [out[f"uf_pct_wap{w}"] for w in range(1, N_WAP + 1) if out.get(f"uf_pct_wap{w}") is not None]
    out["uf_pct_mean"] = float(np.mean(ufs)) if ufs else None
    out["wtow_pct"] = (max(means) - min(means)) / (2 * np.mean(means)) * 100 if len(means) > 1 else None
    out["sheet_check_max_abs_diff"] = max_sheet_diff if means else None
    if n_points:
        center = frame.loc[radius == 0.0, "thickness_a"]
        edge_r = radii[-1]
        edge = frame.loc[radius == edge_r, "thickness_a"]
        out["thk_center"] = float(center.mean()) if len(center) else None
        out["thk_edge"] = float(edge.mean())
        out["edge_radius_mm"] = float(edge_r)
        out["edge_vs_center_pct"] = (
            (out["thk_edge"] - out["thk_center"]) / out["thk_center"] * 100 if out["thk_center"] else None
        )
    else:
        out.update(thk_center=None, thk_edge=None, edge_radius_mm=None, edge_vs_center_pct=None)
    return out


def radial_profile(points: pd.DataFrame) -> pd.DataFrame:
    """Mean thickness per (block, WAP, ring radius): the wafer's center-to-edge shape."""

    frame = points.dropna(subset=["thickness_a"]).copy()
    frame["radius_mm"] = np.hypot(frame["x_mm"].astype(float), frame["y_mm"].astype(float)).round(1)
    return (
        frame.groupby(["block_id", "wap_index", "radius_mm"])["thickness_a"]
        .agg(thk_mean="mean", n="size")
        .reset_index()
    )


def parse_exsitu_workbook(path: str | Path) -> ExsituTables:
    import openpyxl

    workbook = openpyxl.load_workbook(Path(path), data_only=True)
    runs: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for ws in workbook.worksheets:
        for row in range(1, ws.max_row + 1):
            if ws.cell(row, 1).value == BLOCK_MARKER:
                block_id = f"B{len(runs) + 1:03d}"
                info, block_points = _parse_block(ws, row, block_id)
                runs.append(info)
                points.extend(block_points)
    run_table = pd.DataFrame(runs)
    point_table = pd.DataFrame(
        points, columns=["block_id", "point", "x_mm", "y_mm", "wap_index", "thickness_a"]
    )
    point_table["radius_mm"] = np.hypot(point_table["x_mm"].astype(float), point_table["y_mm"].astype(float)).round(1)
    point_table = point_table.merge(run_table[["block_id", "layout"]], on="block_id", how="left")
    return ExsituTables(runs=run_table, points=point_table)


def _trace_boundaries(catalog: pd.DataFrame) -> pd.DataFrame:
    """Long table of (run_id, boundary, timestamp) for every known run boundary."""

    columns = [c for c in ("start", "end", "process_start", "process_end") if c in catalog.columns]
    long = catalog.melt(id_vars=["run_id"], value_vars=columns, var_name="boundary", value_name="ts")
    return long.dropna(subset=["ts"]).reset_index(drop=True)


def _seconds_of_day(ts: pd.Timestamp) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def link_to_traces(exsitu_runs: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach ``run_id`` to ex-situ blocks.

    The workbook's start/end times are operator records: usually the process
    START..END window, sometimes the file start or the moment an alarm stopped
    the run. A block links to the trace run that has any boundary (file
    start/end, process start/end) within five minutes of the block's start or
    end. Blocks with an unusable date (e.g. a typo) fall back to the
    ``연속진행 N`` number in their title, only on the sheet that holds the
    time-matched blocks, and only if the time of day still agrees.
    """

    linked = exsitu_runs.copy()
    linked["run_id"] = None
    linked["link_method"] = None
    linked["link_diff_s"] = np.nan
    bounds = _trace_boundaries(catalog)
    # A block start can only meet a run start, a block end only a run end:
    # consecutive files abut, so a block end often equals the next file's start.
    same_side = {"start": ("start", "process_start"), "end": ("end", "process_end")}
    candidates: list[tuple[float, int, str, str]] = []
    for idx, block in linked.iterrows():
        best: tuple[float, str, str] | None = None
        for side, allowed in same_side.items():
            ts = block[side]
            if pd.isna(ts):
                continue
            pool = bounds[bounds["boundary"].isin(allowed)]
            diff = (pool["ts"] - ts).abs()
            pos = diff.idxmin()
            if diff[pos] <= LINK_TOLERANCE and (best is None or diff[pos].total_seconds() < best[0]):
                best = (diff[pos].total_seconds(), pool.at[pos, "run_id"], f"{side}~{pool.at[pos, 'boundary']}")
        if best is not None:
            candidates.append((best[0], idx, best[1], best[2]))
    claimed: set[str] = set()
    for diff_s, idx, run_id, method in sorted(candidates):
        if run_id in claimed:
            continue
        claimed.add(run_id)
        linked.at[idx, "run_id"] = run_id
        linked.at[idx, "link_method"] = method
        linked.at[idx, "link_diff_s"] = diff_s

    time_matched = linked["run_id"].notna()
    if time_matched.any():
        home_sheet = linked.loc[time_matched, "sheet"].mode().iloc[0]
        by_no = catalog.set_index("run_no")
        taken = set(linked["run_id"].dropna())
        pending = linked[(linked["run_id"].isna()) & (linked["sheet"] == home_sheet)]
        for idx, block in pending.iterrows():
            match = RUN_TITLE_PATTERN.search(str(block["title"]))
            if match is None or int(match.group(1)) not in by_no.index or pd.isna(block["start"]):
                continue
            candidate = by_no.loc[int(match.group(1))]
            if candidate["run_id"] in taken:
                continue
            run_bounds = bounds[bounds["run_id"] == candidate["run_id"]]
            deltas = [
                abs(_seconds_of_day(ts) - _seconds_of_day(block[side]))
                for ts in run_bounds["ts"]
                for side in ("start", "end")
                if not pd.isna(block[side])
            ]
            if deltas and min(deltas) <= LINK_TOLERANCE.total_seconds():
                linked.at[idx, "run_id"] = candidate["run_id"]
                linked.at[idx, "link_method"] = "title_number+time_of_day"
                linked.at[idx, "link_diff_s"] = float(min(deltas))
                taken.add(candidate["run_id"])

    title_no = linked["title"].map(
        lambda t: int(m.group(1)) if (m := RUN_TITLE_PATTERN.search(str(t))) else None
    )
    run_no = linked["run_id"].map(lambda r: int(str(r).rsplit("_", 1)[1]) if isinstance(r, str) else None)
    linked["title_run_no"] = title_no.astype("Int64")
    linked["title_consistent"] = [
        None if (pd.isna(r) or pd.isna(t)) else bool(int(t) == int(r))
        for r, t in zip(run_no, title_no)
    ]
    return linked
