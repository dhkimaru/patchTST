"""Which trained jobs represent an experiment (no torch import: the report uses this too)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.columns import load_column_dictionary

WEIGHT_FILES = ("model.pt", "lightgbm.joblib", "ridge.npz", "estimator.joblib", "template.parquet")


@lru_cache(maxsize=1)
def dictionary_units() -> dict[str, str]:
    """Units in the current column dictionary; they win over the unit stored when a run was trained."""

    dictionary, _ = load_column_dictionary()
    units = dictionary.set_index("column")["unit"].fillna("").astype(str)
    return {column: unit for column, unit in units.items() if unit}


def unit_of(target: str, stored: Any = "") -> str:
    return dictionary_units().get(target) or ("" if stored is None or pd.isna(stored) else str(stored))


def selected_jobs(exp_dir: Path, scope: str = "best") -> list[dict[str, Any]]:
    """Per target, the best validation candidate (``best``) or the best of each model (``per_model``),
    resolved to the seed with the lowest validation MAE."""

    table = pd.read_csv(exp_dir / "summary.csv")
    results = pd.read_csv(exp_dir / "results.csv")
    results = results[results["status"] == "ok"]
    picks = []
    for target, group in table.groupby("target", sort=False):
        group = group.sort_values("mae")
        base = group[group["is_baseline"]]
        candidates = group[~group["is_baseline"]]
        chosen = candidates.head(1) if scope == "best" else candidates.groupby("model", sort=False).head(1)
        for row in chosen.itertuples():
            jobs = results[(results["target"] == target) & (results["label"] == row.label)].sort_values("mae")
            if not len(jobs):
                continue
            job = jobs.iloc[0]
            picks.append({
                "target": target, "label": row.label, "model": row.model, "job_id": job["job_id"],
                "job_dir": exp_dir / "jobs" / str(job["job_id"]), "seed": int(job["seed"]), "mae": float(row.mae),
                "improvement_pct": None if pd.isna(row.improvement_pct) else float(row.improvement_pct),
                "baseline_mae": float(base["mae"].iloc[0]) if len(base) else None,
                "unit": unit_of(target, row.unit),
                "mae_cycle": float(getattr(row, "mae_cycle", float("nan"))),
                "mae_other": float(getattr(row, "mae_other", float("nan"))),
            })
    return picks


def weight_files(directory: Path) -> list[dict[str, Any]]:
    return [{"name": n, "bytes": (directory / n).stat().st_size} for n in WEIGHT_FILES if (directory / n).is_file()]
