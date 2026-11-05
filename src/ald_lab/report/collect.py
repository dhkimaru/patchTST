"""Gather everything the report shows from ``work/`` (and the committed history files).

Reads only finished outputs: the data overview, experiment folders
(``summary.json`` and friends, ``jobs/<id>/job.json``…), model packages and
the journal. Nothing here imports torch, so a report can be built anywhere
the outputs are.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..journal import Journal
from ..launch import HP_SUFFIX
from ..ledger import collect_ledger
from ..paths import PROJECT_ROOT, LabPaths
from ..training.selection import selected_jobs, weight_files
from .dataview import DataView, load_data_view
from .plan import load_plan

HISTORY_DIR = PROJECT_ROOT / "configs" / "history"
# Reporting order; experiments not listed follow alphabetically.
EXPERIMENT_ORDER = ("g1_forecast_vg11", "g2_thickness", "g1_virtual_sensor", "g1_forecast_params", "g3_core_vs_all")
EXAMPLE_POINTS = 900
ZOOM_SECONDS = 600


@dataclass
class Pick:
    """The validation-selected model of one target, with what the report needs from its job folder."""

    target: str
    label: str
    model: str
    job_id: str
    job_dir: Path
    seed: int
    mae: float
    improvement_pct: float | None
    baseline_mae: float | None
    unit: str
    mae_cycle: float
    mae_other: float
    doc: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    weights: list[dict[str, Any]] = field(default_factory=list)
    example: pd.DataFrame | None = None
    example_zoom: pd.DataFrame | None = None


@dataclass
class Experiment:
    name: str
    stamp: str
    directory: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    table: pd.DataFrame
    results: pd.DataFrame
    prep: dict[str, Any] | None
    test: pd.DataFrame | None
    picks: list[Pick]
    package: dict[str, Any] | None
    jobs: list[dict[str, Any]] = field(default_factory=list)  # jobs.json: settings of every job
    test_horizons: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    curves: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return str(self.summary.get("task"))

    @property
    def smoke(self) -> bool:
        return bool(self.summary.get("smoke"))


@dataclass
class ReportData:
    created: datetime
    work_root: Path
    overview: dict[str, Any] | None
    split: dict[str, Any] | None
    experiments: list[Experiment]
    running: list[dict[str, Any]]
    journal: pd.DataFrame
    histories: list[dict[str, Any]]
    notes: list[str]
    data_view: DataView | None = None  # In tab: exclusions, preprocessing example
    plan: dict[str, Any] | None = None  # 다음 실험 tab (configs/plan/next_experiments.yaml)
    ledger: pd.DataFrame = field(default_factory=pd.DataFrame)  # every job of every run (Hyper Parameter ⑦)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def _example(job_dir: Path, limit: int = EXAMPLE_POINTS) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The first validation run's predictions: thinned over the whole run, and ten minutes at full
    resolution from the middle of its ALD cycles (where the pressure pulses are)."""

    path = job_dir / "predictions_sample.parquet"
    if not path.is_file():
        return None, None
    frame = pd.read_parquet(path)
    zoom = None
    if "in_cycle" in frame and frame["in_cycle"].any():
        cycle = frame[frame["in_cycle"]]
        start = float(cycle["t_s"].iloc[len(cycle) // 2])
        zoom = frame[(frame["t_s"] >= start) & (frame["t_s"] < start + ZOOM_SECONDS)].reset_index(drop=True)
        zoom = zoom if len(zoom) > 10 else None
    if len(frame) > limit:
        frame = frame.iloc[:: max(1, len(frame) // limit)].head(limit)
    return frame.reset_index(drop=True), zoom


def load_experiment_dir(directory: Path, *, examples: bool = True) -> Experiment:
    summary = _read_json(directory / "summary.json")
    config = yaml.safe_load((directory / "experiment.yaml").read_text(encoding="utf-8"))
    picks = []
    for raw in selected_jobs(directory, "best"):
        pick = Pick(**raw)
        pick.doc = _read_json(pick.job_dir / "job.json") or {}
        pick.validation = _read_json(pick.job_dir / "metrics_validation.json") or {}
        pick.weights = [{**w, "sha256": _short_hash(pick.job_dir / w["name"])} for w in weight_files(pick.job_dir)]
        if examples:
            pick.example, pick.example_zoom = _example(pick.job_dir)
        picks.append(pick)
    test = pd.read_csv(directory / "test_summary.csv") if (directory / "test_summary.csv").is_file() else None
    results = pd.read_csv(directory / "results.csv")
    experiment = Experiment(
        name=str(summary["experiment"]), stamp=directory.name, directory=directory, summary=summary, config=config,
        table=pd.read_csv(directory / "summary.csv"), results=results,
        prep=_read_json(directory / "prep.json"), test=test, picks=picks, package=None,
        jobs=_read_json(directory / "jobs.json") or [],
    )
    experiment.test_horizons = _test_horizons(experiment)
    experiment.curves = _curves(experiment)
    return experiment


def _ok(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["status"] == "ok"] if "status" in results else results


def _test_horizons(exp: Experiment) -> dict[tuple[str, str], dict[str, float]]:
    """Test-run MAE per horizon of each final-test candidate, averaged over its seeds (metrics_test.json)."""

    out: dict[tuple[str, str], dict[str, float]] = {}
    if exp.test is None:
        return out
    ok = _ok(exp.results)
    for target, label in zip(exp.test["target"], exp.test["label"]):
        values: dict[str, list[float]] = {}
        for job_id in ok.loc[(ok["target"] == target) & (ok["label"] == label), "job_id"]:
            metrics = _read_json(exp.directory / "jobs" / str(job_id) / "metrics_test.json") or {}
            for key, value in (metrics.get("by_horizon") or {}).items():
                if value is not None:
                    values.setdefault(key, []).append(float(value))
        if values:
            out[(target, label)] = {key: sum(v) / len(v) for key, v in values.items()}
    return out


def _curves(exp: Experiment) -> dict[str, dict[str, Any]]:
    """Per target: the epoch history of the selected model if it is a deep model, else of the best deep model,
    plus what the selected classical model decided by itself (trees, regularisation)."""

    ok = _ok(exp.results)
    if "kind" not in ok or not (ok["kind"] == "deep").any():
        return {}
    deep = ok[ok["kind"] == "deep"]
    out = {}
    for pick in exp.picks:
        rows = deep[deep["target"] == pick.target]
        if not len(rows):
            continue
        if pick.model in set(rows["model"]) and pick.label in set(rows["label"]):
            label, job_id = pick.label, pick.job_id
        else:
            label = rows.groupby("label")["mae"].mean().idxmin()
            job_id = rows[rows["label"] == label].sort_values("mae").iloc[0]["job_id"]
        job_dir = exp.directory / "jobs" / str(job_id)
        history = _read_json(job_dir / "history.json") or []
        training = (_read_json(job_dir / "metrics_validation.json") or {}).get("training") or {}
        out[pick.target] = {
            "label": label, "is_pick": label == pick.label, "history": history,
            "best_epoch": training.get("best_epoch"), "epochs": training.get("epochs"),
            "status": training.get("status"), "mae": float(rows[rows["label"] == label]["mae"].mean()),
            "pick_label": pick.label, "pick_model": pick.model,
            "pick_training": pick.validation.get("training") or {},
        }
    return out


def _order(name: str) -> tuple[int, str]:
    base = name.removesuffix("_smoke")
    return (EXPERIMENT_ORDER.index(base) if base in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER), name)


def find_experiments(work_root: Path, *, include_smoke: bool = False, only: list[str] | None = None,
                     examples: bool = True) -> tuple[list[Experiment], list[dict[str, Any]]]:
    """Latest finished run of each experiment, plus experiments still running (no summary yet)."""

    root = work_root / "experiments"
    finished, running = [], []
    if not root.is_dir():
        return finished, running
    for exp in sorted(p for p in root.iterdir() if p.is_dir()):
        if (exp.name.endswith("_smoke") and not include_smoke) or (only and exp.name not in only):
            continue
        if exp.name.removesuffix("_smoke").endswith(HP_SUFFIX):  # dashboard runs: cumulative records only
            continue
        stamps = sorted((p for p in exp.iterdir() if p.is_dir()), reverse=True)
        done = [s for s in stamps if (s / "summary.json").is_file()]
        if stamps and stamps[0] not in done:
            jobs = _read_json(stamps[0] / "jobs.json") or []
            finished_jobs = sum(1 for _ in (stamps[0] / "jobs").glob("*/metrics_validation.json"))
            running.append({"name": exp.name, "stamp": stamps[0].name, "jobs": len(jobs), "done": finished_jobs})
        if done:
            experiment = load_experiment_dir(done[0], examples=examples)
            package = work_root / "packages" / f"{experiment.name}_{experiment.stamp}" / "package.json"
            experiment.package = _read_json(package)
            if experiment.package is not None:
                experiment.package["path"] = str(package.parent)
            finished.append(experiment)
    finished.sort(key=lambda e: _order(e.name))
    return finished, running


def collect(paths: LabPaths, *, include_smoke: bool = False, only: list[str] | None = None, examples: bool = True,
            notes: list[str] | None = None) -> ReportData:
    experiments, running = find_experiments(paths.work_root, include_smoke=include_smoke, only=only,
                                            examples=examples)
    split_path = PROJECT_ROOT / "configs" / "splits" / "split_v2.yaml"
    split = yaml.safe_load(split_path.read_text(encoding="utf-8")) if split_path.is_file() else None
    journal = Journal(paths.journal_dir).to_frame() if (paths.journal_dir / "journal.jsonl").is_file() \
        else pd.DataFrame()
    histories = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(HISTORY_DIR.glob("*.yaml"))]
    histories.sort(key=lambda h: h.get("order", 99))
    return ReportData(
        created=datetime.now(), work_root=paths.work_root, overview=_read_json(paths.overview_dir / "summary.json"),
        split=split, experiments=experiments, running=running, journal=journal, histories=histories,
        notes=list(notes or []), data_view=load_data_view(paths, split, examples=examples), plan=load_plan(),
        ledger=collect_ledger(paths.work_root, include_smoke=include_smoke),
    )
