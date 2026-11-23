"""Hyperparameter search to saturation (극한 탐색): Optuna on top of the training pipeline.

A study YAML (``configs/hpo/*.yaml``) names one base experiment, one target, one model and a search space. Every
trial is an ordinary job of :func:`ald_lab.training.runner.run_job` on the validation runs of the base experiment's
split; the test runs are never opened. Trials are proposed by TPE (multivariate, constant liar for parallel
workers) and weak ones are stopped early by Hyperband on the per-epoch validation MAE. One worker per GPU pulls
trials from a shared journal file under ``work/hpo/<name>/``, so a stopped study continues where it left off.

Saturation is the stopping rule of the 극한 탐색: the best value found in the last ``saturation.window`` finished
trials improves on the best before them by less than max(``min_gain_pct``, seed SD %). The report also gives the
expected best value after n trials (Dodge et al., 2019) and fANOVA importances. Trials keep only metrics and
training history (no weights); the chosen settings are retrained with several seeds (``ald-hpo top``) and judged by
cross-validation like any other experiment.

    ald-hpo run configs/hpo/x1_vs_nh3_full.yaml --gpus 0,1,2,3
    ald-hpo report configs/hpo/x1_vs_nh3_full.yaml          # also written at the end of a run
    ald-hpo top configs/hpo/x1_vs_nh3_full.yaml --k 5 --seeds 42,43,44
    ald-hpo reference --experiment g1_forecast_params        # current settings: seed mean ± SD per target
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .paths import PROJECT_ROOT, LabPaths, resolve_paths

OBJECTIVES = ("mae", "mae_cycle", "mae_other")
SAMPLERS = ("tpe", "random")
PRUNERS = ("hyperband", "halving", "median", "none")
DERIVED = {"stride_ratio": "stride_seconds = patch_seconds × stride_ratio",
           "d_ff_mult": "d_ff = d_model × d_ff_mult"}
CONTEXT_KEY = "window.context_seconds"
INPUT_KEY = "data.input_set"  # a named input set (all, core, core_ab, legacy_primary, A_all, A_plus) per trial
STUDY_TAG = "hpo"  # experiment.yaml ``study``: search trials stay out of the main records (selection bias)


class StudyError(ValueError):
    """Raised for an invalid study YAML."""


@dataclass
class StudySpec:
    name: str
    base: str
    target: str
    model: str = "patchtst"
    input_set: str | None = None
    objective: str = "mae"
    trials: int = 150
    seed: int = 42
    description: str = ""
    sampler: dict[str, Any] = field(default_factory=dict)
    pruner: dict[str, Any] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    space: dict[str, dict[str, Any]] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    enqueue: list[dict[str, Any]] = field(default_factory=list)
    enqueue_from: dict[str, Any] = field(default_factory=dict)
    saturation: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def window(self) -> int:
        return int(self.saturation.get("window", 50))

    @property
    def min_gain_pct(self) -> float:
        return float(self.saturation.get("min_gain_pct", 1.0))


def _base_path(base: str) -> Path:
    path = Path(base)
    if not path.suffix:
        path = PROJECT_ROOT / "configs" / "experiments" / f"{base}.yaml"
    elif not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    return path


def _norm_dist(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        value = {"type": "categorical", "choices": value}
    if not isinstance(value, dict) or "type" not in value:
        raise StudyError(f"space.{key}: a list of choices or {{type: float|int|categorical, ...}}")
    kind = value["type"]
    if kind == "categorical":
        if not value.get("choices"):
            raise StudyError(f"space.{key}: categorical needs choices")
        return {"type": kind, "choices": list(value["choices"])}
    if kind in ("float", "int"):
        if "low" not in value or "high" not in value:
            raise StudyError(f"space.{key}: {kind} needs low and high")
        low, high = float(value["low"]), float(value["high"])
        if not low < high:
            raise StudyError(f"space.{key}: low must be below high")
        log = bool(value.get("log", False))
        if log and low <= 0:
            raise StudyError(f"space.{key}: log scale needs low > 0")
        out: dict[str, Any] = {"type": kind, "low": low, "high": high, "log": log}
        if value.get("step") is not None:
            if log:
                raise StudyError(f"space.{key}: step and log cannot be combined")
            out["step"] = float(value["step"])
        if kind == "int":
            out["low"], out["high"] = int(low), int(high)
            out["step"] = int(out.get("step", 1))
        return out
    raise StudyError(f"space.{key}: type must be float, int or categorical, not {kind!r}")


def load_study(path: str | Path) -> StudySpec:
    from .training.trainer import TrainSettings

    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "space" not in doc:
        raise StudyError(f"{path}: a study needs a search space (space:)")
    spec = StudySpec(
        name=str(doc["name"]), base=str(doc["base"]), target=str(doc["target"]),
        model=str(doc.get("model", "patchtst")), input_set=doc.get("input_set"),
        objective=str(doc.get("objective", "mae")), trials=int(doc.get("trials", 150)), seed=int(doc.get("seed", 42)),
        description=str(doc.get("description") or ""), sampler=dict(doc.get("sampler") or {}),
        pruner=dict(doc.get("pruner") or {}), fixed=dict(doc.get("fixed") or {}),
        space={str(k): _norm_dist(str(k), v) for k, v in (doc.get("space") or {}).items()},
        train=dict(doc.get("train") or {}), enqueue=[dict(e) for e in doc.get("enqueue") or []],
        enqueue_from=dict(doc.get("enqueue_from") or {}), saturation=dict(doc.get("saturation") or {}), path=path,
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.name):
        raise StudyError("name may use letters, digits, '_', '-', '.' only")
    if spec.objective not in OBJECTIVES:
        raise StudyError(f"objective must be one of {OBJECTIVES}")
    if spec.trials < 1:
        raise StudyError("trials must be at least 1")
    if not spec.space:
        raise StudyError("space is empty")
    if str(spec.sampler.get("kind", "tpe")) not in SAMPLERS:
        raise StudyError(f"sampler.kind must be one of {SAMPLERS}")
    if str(spec.pruner.get("kind", "hyperband")) not in PRUNERS:
        raise StudyError(f"pruner.kind must be one of {PRUNERS}")
    fields = set(TrainSettings.__dataclass_fields__)
    for key in [*spec.space, *spec.fixed, *spec.train]:
        name = key[len("train."):] if key.startswith("train.") else (key if key in spec.train else None)
        if name is not None and name not in fields:
            raise StudyError(f"{key}: unknown train setting")
    TrainSettings.from_dict(spec.train)
    for i, params in enumerate(spec.enqueue):
        for key, value in params.items():
            dist = spec.space.get(key)
            if dist and dist["type"] == "categorical" and value not in dist["choices"]:
                raise StudyError(f"enqueue[{i}].{key}={value!r} is not among the choices {dist['choices']}")
    if not _base_path(spec.base).is_file():
        raise StudyError(f"base experiment not found: {spec.base}")
    for derived, target in (("stride_ratio", "stride_seconds"), ("d_ff_mult", "d_ff")):
        if derived in spec.space and target in spec.fixed:
            raise StudyError(f"space.{derived} and fixed.{target} contradict each other; keep one")
    from .training.config import INPUT_SETS

    sets = spec.space.get(INPUT_KEY, {}).get("choices", []) + ([spec.fixed[INPUT_KEY]] if INPUT_KEY in spec.fixed else [])
    unknown = [v for v in sets if v not in INPUT_SETS and v != "default"]
    if unknown:
        raise StudyError(f"{INPUT_KEY}: unknown input sets {unknown}; choose from {INPUT_SETS}")
    return spec


def is_study_file(path: str | Path) -> bool:
    """True for a study YAML (has ``space`` and ``base``), False for an experiment YAML."""

    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(doc, dict) and "space" in doc and "base" in doc


# ----------------------------------------------------------------------------- experiment and trials

def study_experiment(spec: StudySpec, *, smoke: bool = False):
    """The base experiment narrowed to the study's target, model and input set (trials override per job)."""

    from .training.config import ModelSpec, apply_smoke, load_experiment
    from .training.trainer import TrainSettings

    cfg = load_experiment(_base_path(spec.base))
    if spec.target not in cfg.targets and not smoke:
        raise StudyError(f"{spec.target} is not a target of {spec.base}")
    cfg.name = spec.name
    cfg.targets = [spec.target]
    cfg.description = spec.description or f"{spec.target} · {spec.model} 하이퍼파라미터 탐색"
    input_set = spec.input_set or next(iter(cfg.input_sets))
    if input_set not in cfg.input_sets:
        raise StudyError(f"input_set {input_set!r} is not in {list(cfg.input_sets)}")
    # The study's own set first (baseline, default trials); the named sets let a trial pick another (data.input_set).
    from .training.config import INPUT_SETS

    cfg.input_sets = {input_set: cfg.input_sets[input_set],
                      **{name: name for name in INPUT_SETS if name != input_set}}
    cfg.train = TrainSettings.from_dict({**cfg.train.to_dict(), **spec.train})
    cfg.seeds = [spec.seed]
    cfg.models = [ModelSpec(cfg.baseline)]
    raw = copy.deepcopy(cfg.raw)
    raw.update(name=spec.name, targets=[spec.target], seeds=[spec.seed], study=f"{STUDY_TAG}:{spec.name}",
               description=cfg.description)
    raw["train"] = {**(raw.get("train") or {}), **spec.train}
    raw.setdefault("data", {})["input_sets"] = dict(cfg.input_sets)
    raw["models"] = [cfg.baseline, {"name": spec.model, "label": f"{spec.model}_search",
                                    "params": dict(spec.fixed)}]
    cfg.raw = raw
    if smoke:
        cfg = apply_smoke(cfg)
        cfg.raw = {**raw, "name": cfg.name}
    return cfg


def max_epochs_of(spec: StudySpec, cfg: Any) -> int:
    dist = spec.space.get("train.max_epochs")
    if dist:
        return int(max(dist["choices"]) if dist["type"] == "categorical" else dist["high"])
    return int(spec.fixed.get("train.max_epochs", cfg.train.max_epochs))


def suggest(trial: Any, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, dist in space.items():
        if dist["type"] == "categorical":
            out[key] = trial.suggest_categorical(key, dist["choices"])
        elif dist["type"] == "int":
            out[key] = trial.suggest_int(key, dist["low"], dist["high"], step=dist.get("step", 1), log=dist["log"])
        else:
            out[key] = trial.suggest_float(key, dist["low"], dist["high"], step=dist.get("step"), log=dist["log"])
    return out


def materialize(params: dict[str, Any], context_seconds: float) -> tuple[dict[str, Any], float]:
    """Job params (derived keys resolved) and the window length of one trial."""

    out = dict(params)
    context = float(out.pop(CONTEXT_KEY, context_seconds))
    # A derived key never overrides an explicit value (the current settings ride along with their own d_ff).
    ratio = out.pop("stride_ratio", None)
    if ratio is not None:
        out.setdefault("stride_seconds", round(float(out.get("patch_seconds", 16)) * float(ratio), 3))
    mult = out.pop("d_ff_mult", None)
    if mult is not None:
        out.setdefault("d_ff", round(float(out.get("d_model", 128)) * float(mult)))
    return out, context


def trial_problems(model: str, params: dict[str, Any], context_seconds: float) -> list[str]:
    from .models.constraints import combo_problems

    model_params = {k: v for k, v in params.items() if not k.startswith("train.") and k != "residual"}
    problems = combo_problems(model, model_params, context_seconds)
    epochs = params.get("train.max_epochs")
    warm = params.get("train.warmup_epochs")
    if epochs is not None and warm is not None and int(warm) >= int(epochs):
        problems.append("warmup이 최대 epoch보다 깁니다")
    return problems


def with_context(cfg: Any, context_seconds: float) -> Any:
    if abs(float(context_seconds) - cfg.context_seconds) < 1e-9:
        return cfg
    out = copy.copy(cfg)
    out.context_seconds = float(context_seconds)
    return out


# ----------------------------------------------------------------------------- Optuna plumbing

def _optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise StudyError("optuna가 없습니다: pip install -e '.[train]' (optuna 포함)") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def study_dir(paths: LabPaths, name: str) -> Path:
    return Path(paths.work_root) / "hpo" / name


def storage_for(paths: LabPaths, name: str) -> Any:
    optuna = _optuna()
    from optuna.storages import JournalStorage

    try:
        from optuna.storages.journal import JournalFileBackend as Backend
    except ImportError:  # optuna < 4
        from optuna.storages import JournalFileStorage as Backend  # type: ignore[no-redef]
    folder = study_dir(paths, name)
    folder.mkdir(parents=True, exist_ok=True)
    del optuna
    return JournalStorage(Backend(str(folder / "study.journal")))


def make_sampler(spec: StudySpec, worker: int = 0) -> Any:
    optuna = _optuna()
    conf = spec.sampler
    seed = int(conf.get("seed", 0)) + 1009 * worker  # workers must not draw the same random trials
    if str(conf.get("kind", "tpe")) == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(
        seed=seed, multivariate=bool(conf.get("multivariate", True)), group=bool(conf.get("group", False)),
        n_startup_trials=int(conf.get("n_startup_trials", 20)), constant_liar=bool(conf.get("constant_liar", True)),
    )


def make_pruner(spec: StudySpec, max_epochs: int) -> Any:
    optuna = _optuna()
    conf = spec.pruner
    kind = str(conf.get("kind", "hyperband"))
    min_resource = int(conf.get("min_resource", 40))
    if kind == "none":
        return optuna.pruners.NopPruner()
    if kind == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=int(conf.get("n_startup_trials", 10)),
                                           n_warmup_steps=min_resource)
    if kind == "halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=min_resource,
                                                      reduction_factor=int(conf.get("reduction_factor", 3)))
    return optuna.pruners.HyperbandPruner(min_resource=min_resource, max_resource=max(max_epochs, min_resource + 1),
                                          reduction_factor=int(conf.get("reduction_factor", 3)))


def open_study(spec: StudySpec, paths: LabPaths, *, worker: int = 0, max_epochs: int = 100) -> Any:
    optuna = _optuna()
    return optuna.create_study(study_name=spec.name, storage=storage_for(paths, spec.name), direction="minimize",
                               sampler=make_sampler(spec, worker), pruner=make_pruner(spec, max_epochs),
                               load_if_exists=True)


def _counts(study: Any) -> dict[str, int]:
    from optuna.trial import TrialState

    out = {"complete": 0, "pruned": 0, "fail": 0, "running": 0, "waiting": 0}
    names = {TrialState.COMPLETE: "complete", TrialState.PRUNED: "pruned", TrialState.FAIL: "fail",
             TrialState.RUNNING: "running", TrialState.WAITING: "waiting"}
    for t in study.get_trials(deepcopy=False):
        if t.state in names:
            out[names[t.state]] += 1
    return out


def _budget_left(study: Any, budget: int) -> bool:
    c = _counts(study)
    if c["fail"] >= max(20, budget):
        return False
    return c["complete"] + c["pruned"] + c["running"] < budget


# ----------------------------------------------------------------------------- reference (current settings)

def reference_rows(work_root: Path, experiment: str, *, model: str = "patchtst", metric: str = "mae",
                   min_seeds: int = 1) -> pd.DataFrame:
    """Per target the settings with the lowest seed-mean ``metric`` among runs of ``experiment`` (any study tag)."""

    from .ledger import collect_ledger

    led = collect_ledger(work_root)
    if not len(led):
        return pd.DataFrame()
    led = led[(led["base"] == experiment) & (led["model"] == model) & (led["job_status"] == "ok")
              & led[metric].notna()]
    if not len(led):
        return pd.DataFrame()
    led = led.assign(context=led["context_seconds"].astype(str))
    rows = []
    for (target, settings, context, input_set), g in led.groupby(["target", "settings", "context", "input_set"]):
        values = g[metric].astype(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append({"target": target, "unit": g["unit"].iloc[0], "settings": settings,
                     "params": g["params"].iloc[0], "context_seconds": float(g["context_seconds"].iloc[0]),
                     "input_set": input_set, "seeds": int(g["seed"].nunique()), "metric": metric, "mean": mean,
                     "sd": sd, "sd_pct": (sd / mean * 100) if sd is not None and mean else None,
                     "exp_dir": g["exp_dir"].iloc[0]})
    frame = pd.DataFrame(rows)
    enough = frame[frame["seeds"] >= min_seeds]
    frame = enough if len(enough) else frame
    best = frame.loc[frame.groupby("target")["mean"].idxmin()]
    return best.sort_values("target").reset_index(drop=True)


def _reference_trial(spec: StudySpec, work_root: Path, cfg: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Params of the current best settings (``enqueue_from``) split into space keys and the rest."""

    conf = spec.enqueue_from
    if not conf:
        return None
    best = reference_rows(work_root, str(conf["experiment"]), model=spec.model,
                          metric=str(conf.get("metric", spec.objective)), min_seeds=int(conf.get("min_seeds", 1)))
    best = best[best["target"] == spec.target] if len(best) else best
    if not len(best):
        return None
    row = best.iloc[0]
    params = dict(row["params"] or {})
    try:  # the run's own train settings, so the first trial reproduces the current setting as it was trained
        run_doc = yaml.safe_load((Path(row["exp_dir"]) / "experiment.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        run_doc = {}
    for key, value in (run_doc.get("train") or {}).items():
        params.setdefault(f"train.{key}", value)
    if abs(float(row["context_seconds"]) - cfg.context_seconds) > 1e-9:
        params[CONTEXT_KEY] = float(row["context_seconds"])
    if row["input_set"] not in ("default", next(iter(cfg.input_sets))):
        params[INPUT_KEY] = row["input_set"]
    in_space, extra = {}, {}
    for key, value in params.items():
        dist = spec.space.get(key)
        if dist is None:
            extra[key] = value
        elif dist["type"] == "categorical":
            (in_space if value in dist["choices"] else extra)[key] = value
        elif dist["low"] <= float(value) <= dist["high"]:
            in_space[key] = value
        else:
            extra[key] = value
    return in_space, extra


# ----------------------------------------------------------------------------- running

def _trial_job(spec: StudySpec, cfg: Any, number: int, params: dict[str, Any], role: str | None,
               input_set: str | None = None):
    import hashlib

    from .training.config import Job

    label = f"{spec.model}#{'ref' if role == 'reference' else f't{number:04d}'}"
    digest = hashlib.sha1(json.dumps([spec.name, number, params], sort_keys=True, default=str).encode()).hexdigest()[:6]
    job_id = re.sub(r"[^A-Za-z0-9_.=-]+", "_", f"{spec.target}__{spec.model}__t{number:04d}__s{spec.seed}__{digest}")
    return Job(job_id, spec.target, spec.model, "deep", label, params, spec.seed, dict(params), spec.trials,
               input_set or next(iter(cfg.input_sets)))


def run_trials(spec: StudySpec, cfg: Any, paths: LabPaths, split: Any, exp_dir: Path, baselines: dict[str, Any],
               parent_id: str | None, device: str, worker: int, n_jobs: int, results: Any = None,
               cache: Any = None) -> list[dict[str, Any]]:
    """One worker: ask, train, tell until the budget is used. Returns (and puts on ``results``) trial records."""

    import torch

    from .journal import Journal
    from .training.runner import DataCache, run_job

    optuna = _optuna()
    from optuna.trial import TrialState

    torch.set_num_threads(max(1, n_jobs))
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    journal = Journal(paths.journal_dir)
    cache = cache or DataCache(cfg, paths, split, device)
    study = open_study(spec, paths, worker=worker, max_epochs=max_epochs_of(spec, cfg))
    out = []
    while _budget_left(study, spec.trials):
        trial = study.ask()
        role = trial.user_attrs.get("role")
        params = {**spec.fixed, **suggest(trial, spec.space), **dict(trial.user_attrs.get("extra_params") or {})}
        input_set = params.pop(INPUT_KEY, None)
        if input_set == "default":
            input_set = None
        job_params, context = materialize(params, cfg.context_seconds)
        if cfg.smoke:  # a smoke check stays tiny whatever the space says
            job_params.update({k: getattr(cfg.train, k[len("train."):]) for k in job_params
                               if k in ("train.max_epochs", "train.windows_per_epoch", "train.val_windows")})
            job_params.update({k: 0 for k in job_params if k in ("train.min_epochs", "train.warmup_epochs")})
        problems = trial_problems(spec.model, job_params, context)
        if problems:
            trial.set_user_attr("problems", problems)
            study.tell(trial, state=TrialState.FAIL)
            continue
        if abs(context - cfg.context_seconds) > 1e-9:
            job_params[CONTEXT_KEY] = context
        tcfg = with_context(cfg, context)
        job = _trial_job(spec, tcfg, trial.number, job_params, role, input_set)
        run_params = {k: v for k, v in job_params.items() if k != CONTEXT_KEY}
        job = type(job)(**{**job.to_dict(), "params": run_params})

        def on_epoch(epoch: int, value: float, trial: Any = trial, keep: bool = role == "reference") -> bool:
            trial.report(float(value), epoch)
            return False if keep else bool(trial.should_prune())  # the current setting always trains to the end

        started = time.perf_counter()
        try:
            result = run_job(job, tcfg, cache, exp_dir, journal, baselines=baselines, parent_id=parent_id,
                             n_jobs=n_jobs, log_prefix=f"[{device} · trial {trial.number}] ", on_epoch=on_epoch,
                             save_artifacts=False)
        except Exception as exc:  # noqa: BLE001  (recorded in the journal by run_job; the study goes on)
            traceback.print_exc()
            study.tell(trial, state=TrialState.FAIL)
            record = {"job_id": job.job_id, "target": job.target, "model": job.model, "label": job.label,
                      "seed": job.seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "trial": trial.number, "job": {**job.to_dict(), "params": job_params}}
        else:
            value = result.get(spec.objective)
            pruned = result.get("train_status") == "PRUNED"
            trial.set_user_attr("job_id", job.job_id)
            trial.set_user_attr("epochs", result.get("epochs"))
            trial.set_user_attr("mae", result.get("mae"))
            if pruned:
                study.tell(trial, state=TrialState.PRUNED)
            elif value is None or not math.isfinite(float(value)):
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, float(value))
            record = {**result, "trial": trial.number, "pruned": pruned, "objective": value,
                      "trial_seconds": round(time.perf_counter() - started, 1),
                      "job": {**job.to_dict(), "params": job_params}}
        out.append(record)
        if results is not None:
            results.put(record)
    del optuna
    return out


def _worker_entry(spec: StudySpec, cfg: Any, paths: LabPaths, split: Any, exp_dir: Path, baselines: dict[str, Any],
                  parent_id: str | None, device: str, worker: int, n_jobs: int, results: Any) -> None:
    try:
        run_trials(spec, cfg, paths, split, exp_dir, baselines, parent_id, device, worker, n_jobs, results)
    except Exception:  # noqa: BLE001  (the main process notices the worker is gone)
        traceback.print_exc()


def _write_json(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_study(path: str | Path, paths: LabPaths, *, devices: list[str], trials: int | None = None,
              smoke: bool = False) -> Path:
    """Run (or continue) a study; returns the run folder ``work/experiments/<name>/<stamp>``."""

    import shutil

    import torch

    from .data.splits import load_split
    from .journal import Journal
    from .training.config import expand_jobs
    from .training.runner import (
        DataCache,
        _prepare_entry,
        run_job,
        smoke_split,
        summarize,
    )

    spec = load_study(path)
    if trials is not None:
        spec.trials = int(trials)
    if smoke:
        spec.name = f"{spec.name}_smoke"
        spec.trials = min(spec.trials, 3)
        spec.pruner = {**spec.pruner, "min_resource": 1}
        spec.sampler = {**spec.sampler, "n_startup_trials": 1}
    cfg = study_experiment(spec, smoke=smoke)
    cfg.name = spec.name
    exp_dir = Path(paths.work_root) / "experiments" / spec.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(cfg.raw, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")
    shutil.copyfile(spec.path, exp_dir / "study.yaml")
    split = load_split(cfg.split_path())
    if cfg.smoke:
        split = smoke_split(split)
    journal = Journal(paths.journal_dir)
    started = time.perf_counter()
    main_device = devices[0]
    cache = DataCache(cfg, paths, split, main_device)
    parent_id = _prepare_entry(cfg, cache, split, journal, exp_dir)
    jobs_doc: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    for job in [j for j in expand_jobs(cfg) if j.model == cfg.baseline]:
        result = run_job(job, cfg, cache, exp_dir, journal, baselines=baselines, parent_id=parent_id,
                         n_jobs=os.cpu_count() or 1)
        results.append(result)
        jobs_doc.append(job.to_dict())
        baselines[job.target] = json.loads((exp_dir / result["job_dir"] / "metrics_validation.json")
                                           .read_text())["metrics"]
    _write_json(exp_dir / "jobs.json", jobs_doc)

    optuna = _optuna()
    from optuna.trial import TrialState

    study = open_study(spec, paths, max_epochs=max_epochs_of(spec, cfg))
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)  # left over by a stopped run
    if not study.get_trials(deepcopy=False):
        ref = _reference_trial(spec, Path(paths.work_root), cfg)
        if ref is not None:
            in_space, extra = ref
            study.enqueue_trial(in_space, user_attrs={"role": "reference", "extra_params": extra})
        for params in spec.enqueue:
            study.enqueue_trial(params, user_attrs={"role": "enqueued"})
    del optuna
    print(f"study {spec.name}: {spec.target} · {spec.model} · objective {spec.objective} · "
          f"{spec.trials} trials on {', '.join(devices)} (journal {study_dir(paths, spec.name)})", flush=True)

    def collect(record: dict[str, Any]) -> None:
        results.append({k: v for k, v in record.items() if k != "job"})
        jobs_doc.append(record["job"])
        _write_json(exp_dir / "jobs.json", jobs_doc)

    n_jobs = max(1, (os.cpu_count() or 1) // len(devices))
    if len(devices) == 1:
        for record in run_trials(spec, cfg, paths, split, exp_dir, baselines, parent_id, main_device, 0, n_jobs,
                                 cache=cache):
            collect(record)
    else:
        del cache
        if main_device.startswith("cuda"):
            torch.cuda.empty_cache()
        ctx = torch.multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        workers = [ctx.Process(target=_worker_entry, args=(spec, cfg, paths, split, exp_dir, baselines, parent_id,
                                                           device, i, n_jobs, queue))
                   for i, device in enumerate(devices)]
        for w in workers:
            w.start()
        while True:
            try:
                collect(queue.get(timeout=30))
            except Exception:  # noqa: BLE001  (queue.Empty after the timeout)
                if not any(w.is_alive() for w in workers):
                    break
        for w in workers:
            w.join()
        while not queue.empty():
            collect(queue.get_nowait())
    summarize(cfg, results, exp_dir, journal, wall_seconds=time.perf_counter() - started)
    report_study(spec, paths, exp_dir=exp_dir, journal=journal)
    return exp_dir


# ----------------------------------------------------------------------------- saturation report

def expected_best(values: list[float]) -> list[float]:
    """Expected minimum of n draws (with replacement) from ``values`` for n = 1..len(values) (Dodge et al., 2019)."""

    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if not n:
        return []
    ge = (n - np.arange(n)) / n          # P(V ≥ v_i) for sorted v (ties handled by summing their weights)
    gt = (n - np.arange(n) - 1) / n      # P(V > v_i)
    return [float(np.sum(v * (ge**k - gt**k))) for k in range(1, n + 1)]


def saturation(values_in_order: list[float], window: int, min_gain_pct: float,
               seed_sd_pct: float | None) -> dict[str, Any]:
    """Did the last ``window`` finished trials still improve the best value by more than the threshold?"""

    threshold = max(min_gain_pct, seed_sd_pct or 0.0)
    n = len(values_in_order)
    out: dict[str, Any] = {"window": window, "threshold_pct": threshold, "finished": n,
                           "seed_sd_pct": seed_sd_pct, "min_gain_pct": min_gain_pct}
    if n <= window:
        out.update(verdict="판단 보류", reason=f"완료 trial {n}개 ≤ 판정 구간 {window}개", gain_pct=None)
        return out
    before = float(np.min(values_in_order[: n - window]))
    now = float(np.min(values_in_order))
    gain = (before - now) / before * 100 if before else 0.0
    out.update(gain_pct=gain, best_before=before, best=now,
               verdict="포화" if gain < threshold else "개선 중",
               reason=f"최근 {window}개에서 최고 기록 {gain:.2f}% 개선 (기준 {threshold:.2f}%)")
    return out


def _importances(study: Any) -> dict[str, float]:
    optuna = _optuna()
    try:
        from optuna.importance import FanovaImportanceEvaluator

        return {k: float(v) for k, v in optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=0)).items()}
    except Exception as exc:  # noqa: BLE001  (too few trials, one varied parameter, missing sklearn …)
        print(f"importance skipped: {type(exc).__name__}: {exc}")
        return {}


def _seed_sd_pct(spec: StudySpec, paths: LabPaths) -> float | None:
    if spec.saturation.get("seed_sd_pct") is not None:
        return float(spec.saturation["seed_sd_pct"])
    ref = study_dir(paths, spec.name).parent / "reference.json"
    if ref.is_file():
        doc = json.loads(ref.read_text(encoding="utf-8"))
        for row in doc.get("rows", []):
            if row.get("target") == spec.target and row.get("sd_pct") is not None:
                return float(row["sd_pct"])
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def report_study(spec: StudySpec | str | Path, paths: LabPaths, *, exp_dir: Path | None = None,
                 journal: Any = None) -> dict[str, Any]:
    """Saturation curve, importances and top trials of a study → ``work/hpo/<name>/summary.*`` and
    ``work/reports/hpo_<name>.txt`` (read through the lab server's /api/report)."""

    from optuna.trial import TrialState

    from .training.selection import unit_of

    if not isinstance(spec, StudySpec):
        spec = load_study(spec)
    study = open_study(spec, paths)
    trials = study.get_trials(deepcopy=False)
    rows = []
    for t in trials:
        rows.append({"number": t.number, "state": t.state.name, "value": t.value,
                     "role": t.user_attrs.get("role", ""), "epochs": t.user_attrs.get("epochs"),
                     "last_step": t.last_step, "seconds": t.duration.total_seconds() if t.duration else None,
                     "problems": "; ".join(t.user_attrs.get("problems", [])), **{f"p.{k}": v for k, v in t.params.items()}})
    frame = pd.DataFrame(rows)
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    done.sort(key=lambda t: (t.datetime_complete or datetime.max, t.number))
    values = [float(t.value) for t in done]
    best_curve = list(np.minimum.accumulate(values)) if values else []
    expected = expected_best(values)
    sd_pct = _seed_sd_pct(spec, paths)
    sat = saturation(values, spec.window, spec.min_gain_pct, sd_pct)
    importances = _importances(study) if len(done) >= 10 else {}
    ref = next((t for t in done if t.user_attrs.get("role") == "reference"), None)
    best = min(done, key=lambda t: t.value) if done else None
    unit = unit_of(spec.target, None)
    counts = _counts(study)
    top = sorted(done, key=lambda t: t.value)[:10]
    summary = {
        "name": spec.name, "target": spec.target, "model": spec.model, "objective": spec.objective, "unit": unit,
        "base": spec.base, "budget": spec.trials, "counts": counts,
        "best": {"number": best.number, "value": best.value, "params": best.params} if best else None,
        "reference": {"number": ref.number, "value": ref.value} if ref else None,
        "gain_vs_reference_pct": ((ref.value - best.value) / ref.value * 100) if ref and best and ref.value else None,
        "saturation": sat, "importances": importances,
        "best_so_far": [{"n": i + 1, "value": float(v)} for i, v in enumerate(best_curve)],
        "expected_best": [{"n": i + 1, "value": v} for i, v in enumerate(expected)],
        "top": [{"number": t.number, "value": t.value, "params": t.params, "role": t.user_attrs.get("role", "")}
                for t in top],
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    folder = study_dir(paths, spec.name)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / "summary.json", summary)
    frame.to_csv(folder / "trials.csv", index=False)
    text = study_text(summary)
    (folder / "summary.txt").write_text(text, encoding="utf-8")
    reports = Path(paths.work_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"hpo_{spec.name}.txt").write_text(text, encoding="utf-8")
    write_overview(paths)
    if exp_dir is not None:
        _write_json(exp_dir / "hpo_summary.json", summary)
        frame.to_csv(exp_dir / "trials.csv", index=False)
        (exp_dir / "hpo_summary.txt").write_text(text, encoding="utf-8")
    if journal is not None:
        with journal.record("hpo", f"[{spec.name}] 탐색 요약 ({spec.target} · {spec.model})",
                            config={"study": spec.name, "target": spec.target, "model": spec.model,
                                    "objective": spec.objective, "budget": spec.trials, "space": spec.space,
                                    "fixed": spec.fixed}, tags=[spec.name, spec.target, spec.model, "hpo"]) as entry:
            entry.summary = (f"완료 {counts['complete']} · 조기 중단 {counts['pruned']} · 실패 {counts['fail']} · "
                             f"최고 {spec.objective} {_fmt(best.value if best else None)} {unit} · {sat['verdict']}")
            entry.metrics = {"counts": counts, "best": summary["best"], "saturation": sat,
                             "importances": importances, "gain_vs_reference_pct": summary["gain_vs_reference_pct"]}
            entry.artifacts = {"summary": str(folder / "summary.txt")}
            if summary["gain_vs_reference_pct"] is not None and summary["gain_vs_reference_pct"] > 0:
                entry.strengths.append(f"지금 설정 대비 검증 {spec.objective} {summary['gain_vs_reference_pct']:.1f}% 감소")
            if sat["verdict"] == "포화":
                entry.strengths.append(sat["reason"])
            entry.weaknesses.append("검증 run 하나로 고른 값이라 낙관적: 상위 설정은 seed 3개와 교차검증으로 다시 확인")
            if sat["verdict"] != "포화":
                entry.weaknesses.append(f"아직 포화 아님: {sat['reason']}")
    return summary


def study_text(summary: dict[str, Any]) -> str:
    unit = summary.get("unit") or ""
    c = summary["counts"]
    sat = summary["saturation"]
    lines = [f"## 하이퍼파라미터 탐색: {summary['name']} ({summary['target']} · {summary['model']})",
             (f"기준 실험 {summary['base']} · 목표 지표 검증 {summary['objective']} ({unit or '단위 없음'}) · "
              f"예산 {summary['budget']} trial · 갱신 {summary['updated']}"),
             f"완료 {c['complete']} · 조기 중단 {c['pruned']} · 실패 {c['fail']} · 실행 중 {c['running']} · 대기 {c['waiting']}",
             ""]
    best, ref = summary.get("best"), summary.get("reference")
    if best:
        lines.append(f"[최고] trial {best['number']}: {_fmt(best['value'])} {unit}")
    if ref:
        gain = summary.get("gain_vs_reference_pct")
        lines.append(f"[지금 설정] trial {ref['number']}: {_fmt(ref['value'])} {unit}"
                     + (f" · 최고가 {gain:.1f}% 낮음 (최고 ÷ 지금 설정 - 1 = {-gain:.1f}%)" if gain is not None else ""))
    lines += ["", f"[포화 판정] {sat['verdict']}: {sat['reason']}"
              + (f" · seed 표준편차 {sat['seed_sd_pct']:.2f}%" if sat.get("seed_sd_pct") is not None else
                 " · seed 표준편차 기록 없음(최소 개선 기준만 씀)")]
    curve = summary.get("best_so_far") or []
    expected = summary.get("expected_best") or []
    if curve:
        marks = sorted({n for n in (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, len(curve)) if n <= len(curve)})
        lines += ["", "[포화 곡선] 완료 trial 수별 최고 기록과 기대 최고치(그만큼만 돌렸다면 기대되는 최고값)",
                  "trial 수  최고 기록  기대 최고치"]
        for n in marks:
            lines.append(f"{n:>8d}  {_fmt(curve[n - 1]):>9s}  {_fmt(expected[n - 1]):>10s}")
    imp = summary.get("importances") or {}
    if imp:
        lines += ["", "[설정 중요도] fANOVA, 성능 변동에 대한 기여 몫(합 1)"]
        lines += [f"- {k}: {v:.3f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])]
    top = summary.get("top") or []
    if top:
        lines += ["", "[상위 trial]"]
        for t in top:
            params = " · ".join(f"{k}={_fmt(v, 3)}" for k, v in sorted(t["params"].items()))
            role = " (지금 설정)" if t.get("role") == "reference" else ""
            lines.append(f"- trial {t['number']}{role}: {_fmt(t['value'])} {unit} | {params}")
    return "\n".join(lines) + "\n"


def write_overview(paths: LabPaths) -> Path:
    """One table over every study under ``work/hpo/`` → ``work/reports/hpo_overview.txt``."""

    rows = []
    root = Path(paths.work_root) / "hpo"
    for path in sorted(root.glob("*/summary.json")) if root.is_dir() else []:
        s = json.loads(path.read_text(encoding="utf-8"))
        c, sat = s["counts"], s["saturation"]
        imp = s.get("importances") or {}
        rows.append({
            "탐색": s["name"], "대상": s["target"], "지표": s["objective"], "단위": s.get("unit") or "",
            "완료/중단/실패": f"{c['complete']}/{c['pruned']}/{c['fail']}", "예산": s["budget"],
            "지금 설정": _fmt((s.get("reference") or {}).get("value")), "최고": _fmt((s.get("best") or {}).get("value")),
            "지금 대비 (%)": _fmt(-s["gain_vs_reference_pct"], 3) if s.get("gain_vs_reference_pct") is not None else "-",
            "포화": sat["verdict"], "중요도 1위": max(imp, key=imp.get) if imp else "-",
        })
    text = ["## 하이퍼파라미터 탐색 전체 (검증 run, 시험 run 미사용)",
            "'지금 대비' = 최고 ÷ 지금 설정 - 1 (음수면 나아짐). '포화' = 최근 판정 구간에서 최고 기록 개선이 기준 미만.", ""]
    text.append(pd.DataFrame(rows).to_string(index=False) if rows else "탐색 기록이 없습니다.")
    reports = Path(paths.work_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "hpo_overview.txt"
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- follow-up experiments

def top_configs(spec: StudySpec | str | Path, paths: LabPaths, *, k: int = 5, seeds: list[int] | None = None,
                targets: list[str] | None = None, split: str | None = None, name: str | None = None,
                study_tag: str | None = None) -> list[dict[str, Any]]:
    """Experiment documents that retrain the ``k`` best trials with several seeds (one per window length)."""

    from optuna.trial import TrialState

    if not isinstance(spec, StudySpec):
        spec = load_study(spec)
    study = open_study(spec, paths)
    done = sorted((t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE),
                  key=lambda t: t.value)[:k]
    if not done:
        raise StudyError(f"{spec.name}: 완료된 trial이 없습니다")
    base = yaml.safe_load(_base_path(spec.base).read_text(encoding="utf-8"))
    groups: dict[tuple[float, str | None], list[dict[str, Any]]] = {}
    for rank, t in enumerate(done, start=1):
        params = {**spec.fixed, **t.params, **dict(t.user_attrs.get("extra_params") or {})}
        input_set = params.pop(INPUT_KEY, None)
        job_params, context = materialize(params, float((base.get("window") or {}).get("context_seconds", 192)))
        groups.setdefault((context, None if input_set == "default" else input_set), []).append(
            {"name": spec.model, "label": f"{spec.name}_r{rank}", "params": job_params})
    docs = []
    for (context, input_set), models in groups.items():
        doc = copy.deepcopy(base)
        suffix = "" if len(groups) == 1 else f"_w{context:g}" + (f"_{input_set}" if input_set else "")
        doc["name"] = (name or f"{spec.name}_top{k}") + suffix
        doc["targets"] = targets or [spec.target]
        doc["seeds"] = seeds or [42, 43, 44]
        doc.setdefault("window", {})["context_seconds"] = context
        doc["train"] = {**(doc.get("train") or {}), **spec.train}
        if split:
            doc.setdefault("data", {})["split"] = split
        if input_set:
            doc.setdefault("data", {})["input_sets"] = {input_set: input_set}
        elif spec.input_set:
            sets = (doc.get("data") or {}).get("input_sets") or {}
            if spec.input_set in sets:
                doc["data"]["input_sets"] = {spec.input_set: sets[spec.input_set]}
        doc["study"] = f"hpo_top:{spec.name}" if study_tag is None else study_tag
        if not doc["study"]:
            doc.pop("study")
        doc["description"] = f"{spec.name} 상위 {len(models)}개 설정을 seed {len(doc['seeds'])}개로 다시 학습"
        doc["models"] = models
        docs.append(doc)
    return docs


# ----------------------------------------------------------------------------- CLI

def _csv(value: str | None) -> list[str] | None:
    return [s.strip() for s in value.split(",") if s.strip()] if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ald-hpo", description="하이퍼파라미터 탐색 (Optuna, 포화까지)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="탐색 실행 (중단된 탐색은 이어서)")
    p_run.add_argument("study")
    p_run.add_argument("--gpus", default="auto")
    p_run.add_argument("--trials", type=int, help="예산(trial 수)을 이 값으로")
    p_run.add_argument("--smoke", action="store_true", help="trial 3개 · epoch 2로 끝까지 도는지만 확인")
    p_report = sub.add_parser("report", help="포화 곡선 · 중요도 · 상위 trial (work/reports/hpo_<이름>.txt)")
    p_report.add_argument("study", nargs="?", help="생략하면 전체 요약표만 다시 만든다")
    p_top = sub.add_parser("top", help="상위 k개 설정을 seed 여러 개로 다시 학습할 실험 YAML")
    p_top.add_argument("study")
    p_top.add_argument("--k", type=int, default=5)
    p_top.add_argument("--seeds", default="42,43,44")
    p_top.add_argument("--targets", help="같은 설정을 쓸 대상 (기본: 탐색 대상)")
    p_top.add_argument("--split", help="분할 (예: configs/splits/cv1.yaml)")
    p_top.add_argument("--name")
    p_top.add_argument("--study-tag", help="실험 YAML의 study 값 (기본 hpo_top:<탐색 이름>, 빈 값이면 본 기록)")
    p_top.add_argument("--out", help="저장 폴더 (기본: configs/experiments)")
    p_top.add_argument("--submit", action="store_true", help="대기열에 바로 넣는다")
    p_ref = sub.add_parser("reference", help="지금 설정의 seed 평균 ± 표준편차 (누적 기록에서, 학습 없음)")
    p_ref.add_argument("--experiment", required=True, help="기준 실험 이름 (예: g1_forecast_params)")
    p_ref.add_argument("--model", default="patchtst")
    p_ref.add_argument("--metric", default="mae", choices=OBJECTIVES)
    p_ref.add_argument("--min-seeds", type=int, default=2)
    p_ref.add_argument("--write-configs", action="store_true",
                       help="대상마다 지금 설정을 다시 재는 실험 YAML을 configs/experiments/에 쓴다")
    p_ref.add_argument("--seeds", default="42,43,44")
    for p in (p_run, p_report, p_top, p_ref):
        p.add_argument("--work-root")
        p.add_argument("--data-root")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, work_root=args.work_root)
    if args.command == "run":
        from .training.runner import parse_devices

        exp_dir = run_study(args.study, paths, devices=parse_devices(args.gpus), trials=args.trials, smoke=args.smoke)
        print(f"\n결과: {exp_dir}/hpo_summary.txt")
        return 0
    if args.command == "report":
        if args.study:
            print(study_text(report_study(args.study, paths)))
        else:
            print(write_overview(paths).read_text(encoding="utf-8"))
        return 0
    if args.command == "top":
        docs = top_configs(args.study, paths, k=args.k, seeds=[int(s) for s in _csv(args.seeds) or []],
                           targets=_csv(args.targets), split=args.split, name=args.name, study_tag=args.study_tag)
        out = Path(args.out) if args.out else PROJECT_ROOT / "configs" / "experiments"
        out.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            target = out / f"{doc['name']}.yaml"
            target.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"저장: {target}")
            if args.submit:
                from .launch import ensure_worker, submit

                record = submit(paths.work_root, doc, gpus="auto", source="hpo")
                print(f"대기열: {record['id']}")
        if args.submit:
            ensure_worker(paths.work_root)
        return 0
    if args.command == "reference":
        rows = reference_rows(paths.work_root, args.experiment, model=args.model, metric=args.metric,
                              min_seeds=args.min_seeds)
        if not len(rows):
            print(f"{args.experiment}: 누적 기록에 {args.model} 결과가 없습니다")
            return 1
        folder = Path(paths.work_root) / "hpo"
        folder.mkdir(parents=True, exist_ok=True)
        doc = {"experiment": args.experiment, "model": args.model, "metric": args.metric,
               "updated": datetime.now().isoformat(timespec="seconds"),
               "rows": rows.drop(columns=["params"]).to_dict(orient="records")}
        ref_path = folder / "reference.json"
        old = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.is_file() else {}
        merged = {r["target"]: r for r in old.get("rows", [])}
        merged.update({r["target"]: r for r in doc["rows"]})
        _write_json(ref_path, {**doc, "rows": list(merged.values())})
        show = rows[["target", "unit", "seeds", "mean", "sd", "sd_pct", "context_seconds", "settings"]]
        print(f"## 지금 설정 ({args.experiment}, {args.model}, 검증 {args.metric}, seed {args.min_seeds}개 이상 우선)")
        print(show.to_string(index=False, formatters={"mean": "{:.4g}".format,
                                                      "sd": lambda v: "-" if pd.isna(v) else f"{v:.2g}",
                                                      "sd_pct": lambda v: "-" if pd.isna(v) else f"{v:.2f}%"}))
        print(f"\n저장: {ref_path}")
        if args.write_configs:
            seeds = [int(s) for s in _csv(args.seeds) or []]
            out = PROJECT_ROOT / "configs" / "experiments"
            for row in rows.itertuples():
                base_doc = yaml.safe_load((Path(row.exp_dir) / "experiment.yaml").read_text(encoding="utf-8"))
                doc = copy.deepcopy(base_doc)
                safe = re.sub(r"[^A-Za-z0-9]+", "", row.target)
                doc.update(name=f"x0_ref_{args.experiment}_{safe}", targets=[row.target], seeds=seeds,
                           description=f"지금 설정 재측정 ({args.experiment}, seed {len(seeds)}개)")
                doc.setdefault("window", {})["context_seconds"] = row.context_seconds
                doc["models"] = [{"name": args.model, "label": f"{args.model}_ref", "params": dict(row.params)}]
                if row.input_set != "default":
                    sets = (doc.get("data") or {}).get("input_sets") or {}
                    if row.input_set in sets:
                        doc["data"]["input_sets"] = {row.input_set: sets[row.input_set]}
                path = out / f"{doc['name']}.yaml"
                path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"저장: {path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
