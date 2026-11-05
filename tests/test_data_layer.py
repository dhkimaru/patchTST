from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ald_lab.data.catalog import build_catalog, catalog_run, process_window, step_segments
from ald_lab.data.columns import core_columns, load_column_dictionary
from ald_lab.data.exsitu import link_to_traces, parse_exsitu_workbook
from ald_lab.data.profile import ColumnProfiler
from ald_lab.data.raw import (
    TraceFormatError,
    list_trace_files,
    load_trace,
    read_trace_csv,
    run_id_from_path,
    signal_columns,
)


def test_read_trace_types_and_dropped_column(trace_dir):
    frame = read_trace_csv(trace_dir / "standard_trace_001.csv")
    assert frame["timestamp"].iloc[1] - frame["timestamp"].iloc[0] == pd.Timedelta(seconds=0.5)
    assert "Unnamed: 206" not in frame.columns
    assert frame["VG11"].dtype == np.float32
    assert {"recipe", "step_id", "step_name", "row_no"} <= set(frame.columns)
    assert signal_columns(frame) == ["VG11", "PressAct", "AUXMon_VG11", "TempAct_C", "ValveAct_29", "MFCMon_F2"]


def test_run_id_rejects_unexpected_names(tmp_path):
    with pytest.raises(TraceFormatError):
        run_id_from_path(tmp_path / "trace_1.csv")
    assert run_id_from_path("x/standard_trace_007.csv") == "standard_trace_007"


def test_parquet_cache_round_trip(trace_dir, tmp_path):
    source = trace_dir / "standard_trace_001.csv"
    first = load_trace(source, tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == 1
    second = load_trace(source, tmp_path / "cache")
    pd.testing.assert_frame_equal(first, second)


def test_catalog_counts_cycles_gaps_and_process_window(trace_dir):
    rows = []
    for path in list_trace_files(trace_dir):
        frame = read_trace_csv(path)
        rows.append(catalog_run(run_id_from_path(path), frame, path))
    catalog = build_catalog(rows)
    assert catalog["n_cycles"].tolist() == [3, 2, 3]
    assert catalog.attrs["missing_run_numbers"] == [3]
    run4 = catalog.set_index("run_no").loc[4]
    assert run4["n_gaps"] == 1 and run4["max_gap_s"] == pytest.approx(4.5)
    assert "SAMPLING_GAP" in run4["quality_flags"]
    run1 = catalog.set_index("run_no").loc[1]
    assert run1["process_start"] == pd.Timestamp("2025-04-15 10:00:05")
    assert run1["process_end"] == pd.Timestamp("2025-04-15 10:00:40")


def test_step_segments_collapse_consecutive_rows(trace_dir):
    frame = read_trace_csv(trace_dir / "standard_trace_002.csv")
    segments = step_segments(frame)
    assert segments["rows"].sum() == len(frame)
    assert (segments["step_name"] == "SI-FL1").sum() == 2
    start, end = process_window(segments)
    assert start < end


def test_profiler_flags_constant_duplicate_and_near_copy(trace_dir):
    profiler = None
    for path in list_trace_files(trace_dir):
        frame = read_trace_csv(path)
        profiler = profiler or ColumnProfiler(signal_columns(frame), subsample_every=1)
        profiler.add_run(frame)
    result = profiler.result().set_index("column")
    assert result.loc["MFCMon_F2", "status"] == "CONSTANT"
    assert result.loc["PressAct", "status"] == "DUPLICATE"
    assert result.loc["PressAct", "duplicate_of"] == "VG11"
    assert result.loc["MFCMon_F2", "duplicate_of"] is None
    assert result.loc["ValveAct_29", "binary"]
    pairs = profiler.near_copy_pairs()
    assert {"VG11", "AUXMon_VG11"} <= set(pairs["column_a"]) | set(pairs["column_b"])


def test_exsitu_parse_and_link(trace_dir, exsitu_xlsx):
    tables = parse_exsitu_workbook(exsitu_xlsx)
    blocks = tables.runs.set_index("title")
    run1 = blocks.loc["STANDARD 연속진행 1"]
    assert run1["n_points"] == 9
    assert run1["rf_forward_power_w"] == 200.0
    assert run1["tof_installed"] == True and run1["n2_purge_mode"] == False  # noqa: E712
    assert run1["thk_mean_wap1"] == pytest.approx(100.4)
    assert run1["trace_ids"] == "826,827"
    assert len(tables.points) == 3 * 9 * 5
    assert run1["layout"] == "9pt" and run1["ring_radii_mm"] == "0/70/145"
    # U/F = range / (2·mean) · 100 per wafer, recomputed from the raw points.
    assert run1["uf_pct_wap1"] == pytest.approx(0.8 / (2 * 100.4) * 100)
    assert run1["sheet_check_max_abs_diff"] == pytest.approx(0.0, abs=1e-9)

    rows = [catalog_run(run_id_from_path(p), read_trace_csv(p), p) for p in list_trace_files(trace_dir)]
    linked = link_to_traces(tables.runs, build_catalog(rows)).set_index("title")
    assert linked.loc["STANDARD 연속진행 1", "run_id"] == "standard_trace_001"
    # No trace file for run 3: its end equals run 4's file start, which must not link.
    assert linked.loc["STANDARD 연속진행 3", "run_id"] is None
    # Typo'd year falls back to the title number, checked against the time of day.
    assert linked.loc["STANDARD 연속진행 2", "run_id"] == "standard_trace_002"
    assert linked.loc["STANDARD 연속진행 2", "link_method"] == "title_number+time_of_day"
    assert linked["title_consistent"].dropna().all()


def test_column_dictionary_is_consistent():
    columns, parameters = load_column_dictionary()
    assert len(columns) == 200
    # 공개본 사전에는 현업 파라미터 대응·중요도가 없다 (비공개 저장소에만 있음).
    assert parameters.empty and columns["param_id"].isna().all() and core_columns(columns) == []
    assert set(columns["status"]) <= {"OK", "CONSTANT", "DUPLICATE"}
    assert not columns.loc[columns["status"] != "OK", "primary"].any()
