"""Shared, cached data access for dashboard pages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ..data.catalog import step_segments
from ..data.overview import load_overview
from ..data.splits import load_split
from ..journal import Journal
from ..paths import LabPaths, resolve_paths
from .theme import register_templates


def lab_paths() -> LabPaths:
    """Resolve paths from ``streamlit run app.py -- --work-root DIR`` or env vars."""

    work_root = None
    argv = sys.argv[1:]
    if "--work-root" in argv:
        idx = argv.index("--work-root")
        if idx + 1 < len(argv):
            work_root = argv[idx + 1]
    return resolve_paths(work_root=work_root or os.environ.get("ALD_WORK_ROOT"))


def theme_mode() -> str:
    context_theme = getattr(getattr(st, "context", None), "theme", None)
    mode = getattr(context_theme, "type", None) if context_theme is not None else None
    return "dark" if mode == "dark" else "light"


def _dir_signature(path: Path) -> float:
    if not path.exists():
        return 0.0
    return max((p.stat().st_mtime for p in path.iterdir()), default=path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _overview(path_str: str, signature: float) -> dict[str, object]:
    return load_overview(Path(path_str))


def overview() -> dict[str, object]:
    paths = lab_paths()
    return _overview(str(paths.overview_dir), _dir_signature(paths.overview_dir))


@st.cache_data(show_spinner=False)
def _analysis(path_str: str, signature: float) -> dict[str, object]:
    root = Path(path_str)
    table_path = root / "param_variability.parquet"
    summary_path = root / "param_variability.json"
    return {
        "variability": pd.read_parquet(table_path) if table_path.is_file() else None,
        "variability_summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None,
    }


def analysis() -> dict[str, object]:
    root = lab_paths().work_root / "analysis"
    return _analysis(str(root), _dir_signature(root))


@st.cache_data(show_spinner=False)
def split_table(path_str: str) -> pd.DataFrame:
    split = load_split(path_str)
    rows = [{"run_id": r, "partition": p} for p in ("train", "validation", "test") for r in getattr(split, p)]
    rows += [{"run_id": r, "partition": "excluded"} for r in split.excluded]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _journal_frame(path_str: str, signature: float) -> pd.DataFrame:
    return Journal(Path(path_str)).to_frame()


def journal_frame() -> pd.DataFrame:
    paths = lab_paths()
    return _journal_frame(str(paths.journal_dir), _dir_signature(paths.journal_dir))


@st.cache_data(show_spinner=False, max_entries=8)
def trace(run_id: str, cache_dir: str) -> pd.DataFrame:
    """Load one run from the parquet cache written by the overview builder."""

    matches = sorted(Path(cache_dir).glob(f"{run_id}.*.parquet"))
    if not matches:
        raise FileNotFoundError(f"{run_id} is not in the trace cache {cache_dir}")
    return pd.read_parquet(matches[-1])


@st.cache_data(show_spinner=False, max_entries=8)
def segments(run_id: str, cache_dir: str) -> pd.DataFrame:
    return step_segments(trace(run_id, cache_dir))


def setup_page() -> None:
    register_templates()


def _tree_signature(path: Path, pattern: str) -> float:
    if not path.exists():
        return 0.0
    return max((p.stat().st_mtime for p in path.glob(pattern)), default=0.0)


@st.cache_data(show_spinner=False)
def _experiments(path_str: str, signature: float) -> pd.DataFrame:
    rows = []
    for summary in Path(path_str).glob("*/*/summary.json"):
        doc = json.loads(summary.read_text(encoding="utf-8"))
        rows.append({
            "experiment": doc.get("experiment"), "stamp": summary.parent.name, "task": doc.get("task"),
            "smoke": bool(doc.get("smoke")), "jobs": doc.get("jobs"), "failed": doc.get("failed"),
            "wall_minutes": doc.get("wall_minutes"), "dir": str(summary.parent),
            "has_test": (summary.parent / "test_summary.csv").is_file(),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("stamp", ascending=False).reset_index(drop=True) if len(frame) else frame


def experiments() -> pd.DataFrame:
    root = lab_paths().work_root / "experiments"
    return _experiments(str(root), _tree_signature(root, "*/*/*summary*"))


@st.cache_data(show_spinner=False)
def experiment_detail(exp_dir: str, signature: float) -> dict[str, object]:
    root = Path(exp_dir)
    out: dict[str, object] = {
        "summary": json.loads((root / "summary.json").read_text(encoding="utf-8")),
        "table": pd.read_csv(root / "summary.csv"),
        "results": pd.read_csv(root / "results.csv"),
        "test": pd.read_csv(root / "test_summary.csv") if (root / "test_summary.csv").is_file() else None,
        "prep": json.loads((root / "prep.json").read_text(encoding="utf-8")) if (root / "prep.json").is_file() else None,
        "config": (root / "experiment.yaml").read_text(encoding="utf-8") if (root / "experiment.yaml").is_file() else "",
    }
    histories = {}
    for job_dir in (root / "jobs").glob("*"):
        hist = job_dir / "history.json"
        if hist.is_file():
            histories[job_dir.name] = pd.DataFrame(json.loads(hist.read_text(encoding="utf-8")))
    out["histories"] = histories
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def prediction_sample(job_dir: str) -> pd.DataFrame | None:
    path = Path(job_dir) / "predictions_sample.parquet"
    return pd.read_parquet(path) if path.is_file() else None


@st.cache_data(show_spinner=False)
def _ledger(path_str: str, signature: float) -> pd.DataFrame:
    from ..ledger import collect_ledger

    return collect_ledger(Path(path_str))


def ledger() -> pd.DataFrame:
    """Every finished job of every run (cumulative records); refreshed as jobs finish."""

    work_root = lab_paths().work_root
    root = work_root / "experiments"
    signature = max(_tree_signature(root, "*/*/jobs/*"), _tree_signature(root, "*/*/*.csv"),
                    _tree_signature(work_root / "queue", "*/request.json"))
    return _ledger(str(work_root), signature)

