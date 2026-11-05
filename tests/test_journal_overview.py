from __future__ import annotations

import json

import pytest

from ald_lab.data.overview import build_data_overview, load_overview
from ald_lab.journal import Journal, JournalEntry
from ald_lab.paths import resolve_paths


def test_journal_records_success_and_failure(tmp_path):
    journal = Journal(tmp_path)
    with journal.record("train", "PatchTST trial", config={"lr": 1e-3}) as entry:
        entry.metrics["mae"] = 0.2
        entry.strengths.append("SI-FL1 피크를 잘 따라감")
    with pytest.raises(RuntimeError):
        with journal.record("train", "broken trial"):
            raise RuntimeError("CUDA OOM")
    entries = journal.entries()
    assert [e.status for e in entries] == ["ok", "failed"]
    assert entries[0].metrics == {"mae": 0.2} and entries[0].duration_s >= 0
    assert "CUDA OOM" in entries[1].error
    frame = journal.to_frame()
    assert list(frame["title"]) == ["PatchTST trial", "broken trial"]


def test_journal_rejects_unknown_stage():
    with pytest.raises(ValueError):
        JournalEntry(stage="deploy", title="x", started_at="2025-01-01T00:00:00")


def test_build_overview_end_to_end(trace_dir, exsitu_xlsx, tmp_path):
    paths = resolve_paths(data_root=trace_dir, exsitu_xlsx=exsitu_xlsx, work_root=tmp_path / "work")
    result = build_data_overview(paths, workers=1)
    assert result.summary["n_runs"] == 3
    assert result.summary["missing_run_numbers"] == [3]
    assert result.summary["exsitu_linked_runs"] == 2
    tables = load_overview(result.output_dir)
    assert set(tables["issues"]["kind"]) >= {"MISSING_RUN_FILE", "SAMPLING_GAP", "NO_EXSITU"}
    assert tables["columns"].set_index("column").loc["VG11", "unit"] == "Torr"
    entries = Journal(paths.journal_dir).entries("data")
    assert len(entries) == 1 and entries[0].status == "ok"
    assert json.loads((result.output_dir / "summary.json").read_text())["n_rows"] == tables["runs"]["rows"].sum()


def test_overview_survives_a_wrong_workbook_path(trace_dir, tmp_path):
    paths = resolve_paths(data_root=trace_dir, exsitu_xlsx=tmp_path / "nowhere.xlsx", work_root=tmp_path / "work")
    result = build_data_overview(paths, workers=1)
    assert result.summary["n_runs"] == 3 and result.summary["exsitu_linked_runs"] == 0
    assert "EXSITU_MISSING" in set(load_overview(result.output_dir)["issues"]["kind"])
