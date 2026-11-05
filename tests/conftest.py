"""Synthetic trace CSVs and metrology workbooks shaped like the real exports.

공개본: 공정 조건(온도 · RF · 유량 · LOOP 등), 레시피 이름, boat 위치, 두께 값은 실제와 무관한 임의 값이다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CYCLE = [(119, "SI-FL1", 4), (117, "SI-EVA1", 6), (115, "N-FL1", 8), (111, "N-EVA1", 4)]
# The workbook's 9-point layout: center, four at 70 mm, four at 145 mm.
NINE_POINT_XY = [(0, 0), (0, 70), (-70, 0), (0, -70), (70, 0), (0, 145), (-145, 0), (0, -145), (145, 0)]


def make_trace(
    start: str,
    *,
    n_cycles: int = 3,
    standby_rows: int = 10,
    seed: int = 0,
    gap_after_row: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = [(65534, "STANDBY")] * standby_rows + [(1, "START")] * 4
    for _ in range(n_cycles):
        for step_id, name, rows in CYCLE:
            steps += [(step_id, name)] * rows
    steps += [(65535, "END")] * 3 + [(65534, "STANDBY")] * standby_rows
    n = len(steps)
    times = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * 0.5, unit="s")
    if gap_after_row is not None:
        times = times.where(np.arange(n) <= gap_after_row, times + pd.Timedelta(seconds=4))
    vg11 = rng.normal(1.0, 0.1, n).round(3)
    frame = pd.DataFrame(
        {
            "No.": np.arange(1, n + 1),
            "Recipe(Table) Name": "MAIN_RECIPE",
            "Step ID": [s[0] for s in steps],
            "Step Name": [s[1] for s in steps],
            "Date": times.strftime("%m/%d/%Y"),
            "Time": [t.strftime("%H:%M:%S.") + str(t.microsecond // 100000) for t in times],
            "VG11": vg11,
            "PressAct": vg11,  # exact copy, like the real export
            "AUXMon_VG11": (vg11 / 10).round(4),
            "TempAct_C": rng.normal(500, 1, n).round(1),
            "ValveAct_29": [1.0 if s[1] == "SI-FL1" else 0.0 for s in steps],
            "MFCMon_F2": 0.0,
            "Unnamed: 206": np.nan,
        }
    )
    return frame


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    root = tmp_path / "traces"
    root.mkdir()
    make_trace("2025-04-15 10:00:00", seed=1).to_csv(root / "standard_trace_001.csv", index=False)
    make_trace("2025-04-15 12:00:00", seed=2, n_cycles=2).to_csv(root / "standard_trace_002.csv", index=False)
    # Long standby padding: this file's process window starts 8+ minutes in.
    make_trace("2025-04-15 14:00:00", seed=3, standby_rows=1000, gap_after_row=20).to_csv(
        root / "standard_trace_004.csv", index=False
    )
    return root


def _write_block(ws, top: int, title: str, date: dt.date | None, start: dt.time, end: dt.time,
                 values: list[list[float]]) -> int:
    ws.cell(top, 3, title)
    r = top + 1
    rows_a = [("진행날짜", dt.datetime.combine(date, dt.time()) if date else None), ("Start Time", start),
              ("End Time", end), ("측정날짜", None), ("Recipe", "MAIN_RECIPE"), ("TraceData", 826),
              (None, 827), (None, None), (None, None), (None, None), (None, None), (None, None),
              (None, None), ("DATA TIME [sec]", 0.5)]
    conditions = [("TEMP_U", 500), ("TEMP_CU", 500), ("TEMP_C", 500), ("TEMP_CL", 500), ("TEMP_L", 500),
                  ("RF F.PW", "200W"), ("DCS", 1.0), ("NH3", 4.0), ("LOOP", 90), ("PURGE MODE", 3),
                  ("WAP", "10,35,60,85,110"), ("Boat Rotation", 2.0), ("ToF", "O"), ("N2 PURGE MODE", "X"),
                  ("NORMAL END", "O")]
    for i, (key, value) in enumerate(rows_a):
        if key is not None:
            ws.cell(r + i, 1, key)
        if value is not None:
            ws.cell(r + i, 2, value)
    for i, (key, value) in enumerate(conditions):
        ws.cell(r + i, 3, key)
        ws.cell(r + i, 4, value)
    for c, label in enumerate(["POINT", "x", "y", "1", "2", "3", "4", "5"], start=5):
        ws.cell(r, c, label)
    for p, row_values in enumerate(values, start=1):
        x, y = NINE_POINT_XY[p - 1]
        ws.cell(r + p, 5, p)
        ws.cell(r + p, 6, x)
        ws.cell(r + p, 7, y)
        for w, v in enumerate(row_values):
            ws.cell(r + p, 8 + w, v)
    arr = np.array(values)
    summary_row = r + len(values) + 1
    stats = {
        "AVERAGE": arr.mean(axis=0),
        "MAX": arr.max(axis=0),
        "MIN": arr.min(axis=0),
        "RANGE": arr.max(axis=0) - arr.min(axis=0),
        "U/F": (arr.max(axis=0) - arr.min(axis=0)) / (2 * arr.mean(axis=0)) * 100,
    }
    for i, (label, vals) in enumerate(stats.items()):
        ws.cell(summary_row + i, 5, label)
        for w, v in enumerate(vals):
            ws.cell(summary_row + i, 8 + w, float(v))
    return summary_row + len(stats) + 2


@pytest.fixture
def exsitu_xlsx(tmp_path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "5.F2 Clean_1차_이후 80"
    values = [[100.0 + 0.1 * p + w for w in range(5)] for p in range(9)]
    # Run 1: START at row 10 -> 10:00:05; END step begins after standby + start + cycles.
    top = _write_block(ws, 1, "STANDARD 연속진행 1", dt.date(2025, 4, 15), dt.time(10, 0, 5),
                       dt.time(10, 0, 5 + 33), values)
    # Run 3 has no trace file; its end equals the start of run 4's file (must NOT link).
    top = _write_block(ws, top, "STANDARD 연속진행 3", dt.date(2025, 4, 15), dt.time(13, 0, 0),
                       dt.time(14, 0, 0), values)
    # Run 2 with a typo'd year: links by title number + time of day.
    top = _write_block(ws, top, "STANDARD 연속진행 2", dt.date(2021, 4, 15), dt.time(12, 0, 5),
                       dt.time(12, 0, 5 + 22), values)
    path = tmp_path / "metrology.xlsx"
    wb.save(path)
    return path
