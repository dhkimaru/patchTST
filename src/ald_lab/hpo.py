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
from datetime import datetime, timedelta
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
# 1: keys missing from the record were drawn by the sampler; 2: filled with the run's own defaults, but the train
# block came from any run with the same model settings (runs that trained differently were averaged together);
# 3: runs are grouped by their train block too, so the mean and the reproduced setting come from the same runs.
REF_VERSION = 3
# A current-setting trial further than this from its recorded mean (max of the %, 3 seed SD) is flagged.
REF_TOLERANCE_PCT = 10.0
LIVE_EVERY = 5  # rewrite the study report after this many finished trials (the dashboard reads it)


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
                   min_seeds: int = 1, ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per target the settings with the lowest seed-mean ``metric`` among runs of ``experiment`` (any study tag).
    ``ledger`` reuses an already collected record."""

    from .ledger import collect_ledger

    led = collect_ledger(work_root) if ledger is None else ledger
    if not len(led):
        return pd.DataFrame()
    led = led[(led["base"] == experiment) & (led["model"] == model) & (led["job_status"] == "ok")
              & led[metric].notna()]
    if not len(led):
        return pd.DataFrame()
    led = led.assign(context=led["context_seconds"].astype(str),
                     train=led["train"].fillna("") if "train" in led else "")
    rows = []
    for (target, settings, train, context, input_set), g in led.groupby(
            ["target", "settings", "train", "context", "input_set"]):
        values = g[metric].astype(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        per_seed = {int(s): float(v) for s, v in g.groupby("seed")[metric].mean().items()}
        fit = g["fit_minutes"].dropna() if "fit_minutes" in g else pd.Series(dtype=float)
        rows.append({"target": target, "unit": g["unit"].iloc[0], "settings": settings, "train": train,
                     "per_seed": per_seed, "fit_minutes": float(fit.mean()) if len(fit) else None,
                     "params": g["params"].iloc[0], "context_seconds": float(g["context_seconds"].iloc[0]),
                     "input_set": input_set, "seeds": int(g["seed"].nunique()), "metric": metric, "mean": mean,
                     "sd": sd, "sd_pct": (sd / mean * 100) if sd is not None and mean else None,
                     "exp_dir": g["exp_dir"].iloc[0]})
    frame = pd.DataFrame(rows)
    enough = frame[frame["seeds"] >= min_seeds]
    frame = enough if len(enough) else frame
    best = frame.loc[frame.groupby("target")["mean"].idxmin()]
    return best.sort_values("target").reset_index(drop=True)


def save_reference_rows(paths: LabPaths, experiment: str, model: str, rows: pd.DataFrame) -> Path:
    """Merge ``rows`` of :func:`reference_rows` into ``work/hpo/reference.json`` (one row per experiment, model and
    target; a later call wins) and rewrite ``work/reports/hpo_reference.txt``."""

    folder = Path(paths.work_root) / "hpo"
    folder.mkdir(parents=True, exist_ok=True)
    new_rows = rows.drop(columns=["params"]).assign(experiment=experiment, model=model)
    ref_path = folder / "reference.json"
    old = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.is_file() else {}
    key = lambda r: f"{r.get('experiment')}|{r.get('model')}|{r['target']}"
    merged = {key(r): r for r in old.get("rows", [])}
    merged.update({key(r): r for r in new_rows.to_dict(orient="records")})
    all_rows = sorted(merged.values(), key=lambda r: (str(r.get("experiment")), r["target"]))
    _write_json(ref_path, {"updated": datetime.now().isoformat(timespec="seconds"), "rows": all_rows})
    table = pd.DataFrame(all_rows).reindex(columns=["experiment", "target", "unit", "metric", "seeds", "mean", "sd",
                                                    "sd_pct", "context_seconds", "settings", "train"])
    reports = Path(paths.work_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "hpo_reference.txt").write_text(
        "## 지금 설정의 기준점 (누적 기록, 검증 run, 학습 없음)\n"
        "실험·대상마다 seed 평균이 가장 낮은 설정. sd = seed 간 표준편차(단위는 대상 단위), sd_pct = sd ÷ 평균 × 100.\n"
        "settings = 모델 설정, train = 그 run의 실험 파일 학습 설정(같은 모델 설정이라도 학습 설정이 다르면 따로 묶음).\n\n"
        + table.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n", encoding="utf-8")
    return ref_path


_BASELINE_MISS: dict[str, float] = {}


def record_baseline(spec: StudySpec, paths: LabPaths, cache: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """What a study is measured against: the seed mean of the current setting in the cumulative record
    (``enqueue_from``). Read from ``work/hpo/reference.json``; the first time it is missing it is taken from the
    record and saved (``cache`` shares one collected record between studies). None without a record."""

    conf = spec.enqueue_from
    if not conf:
        return None
    experiment, metric = str(conf["experiment"]), str(conf.get("metric", spec.objective))
    key = f"{experiment}|{spec.model}|{spec.target}|{metric}"
    path = Path(paths.work_root) / "hpo" / "reference.json"
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", []) if path.is_file() else []
    for r in rows:
        if (f"{r.get('experiment')}|{r.get('model', spec.model)}|{r.get('target')}|{r.get('metric', metric)}" == key
                and r.get("mean") is not None):
            return r
    if time.time() - _BASELINE_MISS.get(key, 0.0) < 600:  # looked a few minutes ago and found nothing
        return None
    from .ledger import collect_ledger

    cache = {} if cache is None else cache
    if "ledger" not in cache:
        cache["ledger"] = collect_ledger(paths.work_root)
    found = reference_rows(paths.work_root, experiment, model=spec.model, metric=metric,
                           min_seeds=int(conf.get("min_seeds", 1)), ledger=cache["ledger"])
    mine = found[found["target"] == spec.target] if len(found) else found
    if not len(mine):
        _BASELINE_MISS[key] = time.time()
        return None
    save_reference_rows(paths, experiment, spec.model, found)
    return {**mine.drop(columns=["params"]).iloc[0].to_dict(), "experiment": experiment, "model": spec.model}


def _reference_trial(spec: StudySpec, work_root: Path, cfg: Any
                     ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Params of the current best settings (``enqueue_from``) split into space keys and the rest, and what the
    record expects of them (seed mean, SD, seed count)."""

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
    run_train = dict(run_doc.get("train") or {})
    for key, value in run_train.items():
        params.setdefault(f"train.{key}", value)
    params[CONTEXT_KEY] = float(row["context_seconds"])
    if row["input_set"] not in ("default", next(iter(cfg.input_sets))):
        params[INPUT_KEY] = row["input_set"]
    params.update(_unrecorded_defaults(spec, cfg, params, run_train))
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
    expected = {"mean": float(row["mean"]), "sd": None if pd.isna(row["sd"]) else float(row["sd"]),
                "seeds": int(row["seeds"]), "train": str(row.get("train") or ""), "settings": str(row["settings"]),
                "per_seed": {str(k): v for k, v in dict(row.get("per_seed") or {}).items()}}
    return in_space, extra, expected


def _model_defaults(model: str, sample_seconds: float) -> dict[str, Any]:
    """Constructor defaults of a deep model as the job params name them (seconds for patch, stride, kernel)."""

    from .models import defaults

    out: dict[str, Any] = {"revin": True, "revin_center": "mean", "head_dropout": 0.0}  # build_deep_model
    seconds = {"patch_len": "patch_seconds", "stride": "stride_seconds", "kernel": "kernel_seconds"}
    for key, value in defaults.DEEP.get(model, {}).items():
        if key in seconds:
            out[seconds[key]] = value * sample_seconds
        else:
            out[key] = value
    return out


def _unrecorded_defaults(spec: StudySpec, cfg: Any, params: dict[str, Any], run_train: dict[str, Any]) -> dict[str, Any]:
    """Values the current setting used for keys its record does not name: the run's train settings or the
    defaults (TrainSettings, model constructor). Without them the sampler would fill them in, and the first trial
    would not be the current setting (REF_VERSION 1)."""

    from .training.trainer import TrainSettings

    train_defaults = TrainSettings().to_dict()
    model_defaults = _model_defaults(spec.model, cfg.sample_seconds)
    out: dict[str, Any] = {}
    for key in spec.train:  # the study's own training rules (min_epochs, patience …) do not apply to it
        out.setdefault(f"train.{key}", run_train.get(key, train_defaults.get(key)))
    for key in [*spec.space, *spec.fixed]:
        if key in params or key in out:
            continue
        if key.startswith("train."):
            name = key[len("train."):]
            out[key] = run_train.get(name, train_defaults.get(name))
        elif key == INPUT_KEY:
            out[key] = "default"
        elif key == "stride_ratio":
            patch = float(params.get("patch_seconds", model_defaults.get("patch_seconds", 16)))
            stride = float(params.get("stride_seconds", model_defaults.get("stride_seconds", 8)))
            out[key] = round(stride / patch, 4) if patch else None
        elif key == "d_ff_mult":
            d_model = float(params.get("d_model", model_defaults.get("d_model", 128)))
            d_ff = float(params.get("d_ff", model_defaults.get("d_ff", 256)))
            out[key] = round(d_ff / d_model, 4) if d_model else None
        elif key in model_defaults:
            out[key] = model_defaults[key]
    return {k: v for k, v in out.items() if v is not None}


def enqueue_reference(spec: StudySpec, paths: LabPaths, cfg: Any = None, study: Any = None) -> bool:
    """Put the current setting (REF_VERSION) first in line unless the study already has it. True when queued.
    Works on a running study too: its workers take the waiting trial next."""

    cfg = cfg or study_experiment(spec)
    study = study or open_study(spec, paths, max_epochs=max_epochs_of(spec, cfg))
    if any(int(t.user_attrs.get("ref_version", 1)) >= REF_VERSION and t.user_attrs.get("role") == "reference"
           for t in study.get_trials(deepcopy=False)):
        return False
    ref = _reference_trial(spec, Path(paths.work_root), cfg)
    if ref is None:
        return False
    in_space, extra, expected = ref
    study.enqueue_trial(in_space, user_attrs={"role": "reference", "ref_version": REF_VERSION, "extra_params": extra,
                                              "expected": expected})
    return True


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
               cache: Any = None, on_record: Any = None) -> list[dict[str, Any]]:
    """One worker: ask, train, tell until the budget is used. Returns trial records and hands each one to
    ``results`` (a queue, worker processes) or ``on_record`` (the main process) as soon as it is done."""

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
        trial.set_user_attr("actual", {**job_params, CONTEXT_KEY: context, INPUT_KEY: input_set or "default"})
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
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:500])
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
            trial.set_user_attr("train_status", result.get("train_status"))
            if (result.get("batch_parts") or 1) > 1:
                trial.set_user_attr("batch_parts", result["batch_parts"])
            if pruned:
                study.tell(trial, state=TrialState.PRUNED)
            elif value is None or not math.isfinite(float(value)):
                trial.set_user_attr("error", f"검증 {spec.objective}가 NaN: 학습 발산 ({result.get('train_status')})")
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, float(value))
            record = {**result, "trial": trial.number, "pruned": pruned, "objective": value,
                      "trial_seconds": round(time.perf_counter() - started, 1),
                      "job": {**job.to_dict(), "params": job_params}}
        out.append(record)
        if results is not None:
            results.put(record)
        if on_record is not None:
            on_record(record)
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
    fresh = not study.get_trials(deepcopy=False)
    enqueue_reference(spec, paths, cfg, study)  # also for a study begun before the current REF_VERSION
    if fresh:
        for params in spec.enqueue:
            study.enqueue_trial(params, user_attrs={"role": "enqueued"})
    del optuna
    run_info = {"started": datetime.now().isoformat(timespec="seconds"), "workers": len(devices)}
    print(f"study {spec.name}: {spec.target} · {spec.model} · objective {spec.objective} · "
          f"{spec.trials} trials on {', '.join(devices)} (journal {study_dir(paths, spec.name)})", flush=True)

    def collect(record: dict[str, Any]) -> None:
        results.append({k: v for k, v in record.items() if k != "job"})
        jobs_doc.append(record["job"])
        _write_json(exp_dir / "jobs.json", jobs_doc)
        if (len(results) - 1) % LIVE_EVERY == 0:  # the dashboard follows the study while it runs
            try:
                report_study(spec, paths, importance=False, run=run_info)
            except Exception as exc:  # noqa: BLE001  (a report problem must not stop the search)
                print(f"live report skipped: {type(exc).__name__}: {exc}", flush=True)

    n_jobs = max(1, (os.cpu_count() or 1) // len(devices))
    if len(devices) == 1:
        run_trials(spec, cfg, paths, split, exp_dir, baselines, parent_id, main_device, 0, n_jobs, cache=cache,
                   on_record=collect)
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
    report_study(spec, paths, exp_dir=exp_dir, journal=journal, run={**run_info, "finished": True})
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
        rows = json.loads(ref.read_text(encoding="utf-8")).get("rows", [])
        experiment = str(spec.enqueue_from.get("experiment") or "")
        mine = [r for r in rows if r.get("target") == spec.target and r.get("sd_pct") is not None]
        exact = [r for r in mine if r.get("experiment") == experiment]
        for row in exact or mine:  # the study's own reference experiment first, any other run of the target next
            return float(row["sd_pct"])
    return None


def reference_check(spec: StudySpec, paths: LabPaths, ref: Any) -> dict[str, Any] | None:
    """How far the current-setting trial landed from the seed mean the record has for it. A trial outside
    max(REF_TOLERANCE_PCT, 3 seed SD) did not reproduce the current setting, so '지금 대비' is not trustworthy."""

    if ref is None or ref.value is None:
        return None
    expected = ref.user_attrs.get("expected")
    if not expected:  # queued before the expectation was stored: the table of ``ald-hpo reference``
        path = study_dir(paths, spec.name).parent / "reference.json"
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows", []) if path.is_file() else []
        experiment = str(spec.enqueue_from.get("experiment") or "")
        mine = [r for r in rows if r.get("target") == spec.target and r.get("experiment") == experiment
                and r.get("model", spec.model) == spec.model and r.get("mean")]
        expected = {"mean": mine[0]["mean"], "sd": mine[0].get("sd"), "seeds": mine[0].get("seeds")} if mine else None
    if not expected or not expected.get("mean"):
        return None
    mean = float(expected["mean"])
    sd = expected.get("sd")
    sd = None if sd is None or (isinstance(sd, float) and not math.isfinite(sd)) else float(sd)
    gap = (float(ref.value) - mean) / mean * 100
    limit = max(REF_TOLERANCE_PCT, 3 * sd / mean * 100) if sd else REF_TOLERANCE_PCT
    return {"mean": mean, "sd": sd, "seeds": expected.get("seeds"), "gap_pct": gap, "limit_pct": limit,
            "ok": abs(gap) <= limit}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _local(when: Any) -> datetime | None:
    """A naive local time (Optuna's clock) from an ISO string or datetime."""

    if when is None:
        return None
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when)
        except ValueError:
            return None
    return when.astimezone().replace(tzinfo=None) if when.tzinfo else when


def progress(trials: list[Any], budget: int, *, since: Any = None, workers: int | None = None,
             now: datetime | None = None) -> dict[str, Any]:
    """Finished trials, pace and the expected end of a study. The pace is finished trials per hour of wall time
    since ``since`` (the run's start: it already includes the workers in parallel); without it, the median
    trial duration divided by ``workers``."""

    from optuna.trial import TrialState

    now = now or datetime.now()
    finished = [t for t in trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    left = max(0, budget - len(finished))
    out: dict[str, Any] = {"finished": len(finished), "budget": budget, "left": left, "per_trial_min": None,
                           "eta_min": None, "eta_at": None, "basis": ""}
    start = _local(since)
    recent = [t for t in finished if start and t.datetime_complete and t.datetime_complete >= start]
    if start and recent:
        elapsed = (now - start).total_seconds() / 60
        per = elapsed / len(recent)
        out.update(per_trial_min=per, basis=f"이번 실행 {len(recent)}개 기준")
    else:
        durations = [t.duration.total_seconds() / 60 for t in finished if t.duration]
        if durations:
            per = float(np.median(durations)) / max(1, workers or 1)
            out.update(per_trial_min=per, basis=f"trial {len(durations)}개 중앙값 ÷ 작업자 {max(1, workers or 1)}")
    if out["per_trial_min"] is not None:
        eta = out["per_trial_min"] * left
        out.update(eta_min=eta, eta_at=(now + timedelta(minutes=eta)).isoformat(timespec="minutes"))
    return out


def _reference_of(done: list[Any]) -> tuple[Any, int]:
    """The newest current-setting trial of the newest version (an old version stays in the record, unused)."""

    refs = [t for t in done if t.user_attrs.get("role") == "reference"]
    if not refs:
        return None, 0
    ref = max(refs, key=lambda t: (int(t.user_attrs.get("ref_version", 1)), t.number))
    return ref, int(ref.user_attrs.get("ref_version", 1))


FAIL_UNKNOWN = "기록 없음: 검증 지표가 NaN이었거나 작업자가 중간에 멈춤"
_JOURNAL_FAILS: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}


def fail_kind(text: str) -> str:
    """A short name for why a trial failed, from its error text."""

    low = text.lower()
    if "out of memory" in low or "outofmemory" in low:
        return "GPU 메모리 부족"
    if "NaN" in text or "not finite" in low:
        return "학습 발산 (NaN)"
    if "worker exited" in low:
        return "작업자 중단"
    return "오류"


def journal_failures(paths: LabPaths, name: str) -> dict[str, str]:
    """``label → error`` of the failed trial jobs of study ``name`` in the pipeline journal. Runs of older code kept
    the error only there (newer runs also store it on the trial). Read incrementally: only new lines each call."""

    path = Path(paths.journal_dir) / "journal.jsonl"
    if not path.is_file():
        return {}
    key = (str(path), name)
    size = path.stat().st_size
    offset, found = _JOURNAL_FAILS.get(key, (0, {}))
    if size < offset:  # the file was replaced: read it again
        offset, found = 0, {}
    if size > offset:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(size - offset)
        whole = chunk.rfind(b"\n") + 1  # a line still being written waits for the next call
        needle = f'"experiment": {json.dumps(name)}'
        found = dict(found)
        for raw in chunk[:whole].decode("utf-8", errors="replace").splitlines():
            if '"status": "failed"' not in raw or needle not in raw:
                continue
            try:
                doc = json.loads(raw)
            except ValueError:
                continue
            config = doc.get("config") or {}
            if config.get("experiment") == name and config.get("label"):
                found[str(config["label"])] = str(doc.get("error") or "")
        offset += whole
        _JOURNAL_FAILS[key] = (offset, found)
    return found


def brief_settings(params: dict[str, Any]) -> str:
    """The settings that decide memory and time: window, patch, patch count, size, batch, and the size of one
    layer's attention scores (batch × heads × patches² × 4 bytes, the tensor that runs out of memory first)."""

    p = dict(params)
    context, patch = p.get(CONTEXT_KEY), p.get("patch_seconds")
    stride = p.get("stride_seconds")
    if stride is None and patch is not None and p.get("stride_ratio") is not None:
        stride = float(patch) * float(p["stride_ratio"])
    parts = []
    if context is not None:
        parts.append(f"window {float(context):g}초")
    if patch is not None:
        parts.append(f"patch {float(patch):g}초" + (f" 간격 {float(stride):g}초" if stride else ""))
    tokens = None
    if context is not None and patch is not None and stride:
        tokens = int((float(context) - float(patch)) // float(stride)) + 2  # PatchTST pads one stride at the end
        parts.append(f"patch {tokens}개")
    for key, label in (("d_model", "d_model "), ("n_layers", "층 "), ("n_heads", "head "), ("d_ff", "d_ff "),
                       ("train.batch_size", "batch "), ("train.lr", "lr ")):
        if p.get(key) is not None:
            value = p[key]
            parts.append(f"{label}{value:.0e}" if key == "train.lr" else f"{label}{value}")
    batch, heads = p.get("train.batch_size"), p.get("n_heads")
    if tokens and batch and heads:
        size = int(batch) * int(heads) * tokens**2 * 4 / 2**30
        parts.append(f"attention 점수 한 층 {size:.2f} GiB" if size < 10 else f"attention 점수 한 층 {size:.0f} GiB")
    return ", ".join(parts)


def failures(spec: StudySpec, trials: list[Any], paths: LabPaths) -> list[dict[str, Any]]:
    """Every failed trial with a reason (``kind``), the error text and its memory-relevant settings."""

    from optuna.trial import TrialState

    out, logged = [], None
    for t in trials:
        if t.state != TrialState.FAIL:
            continue
        problems = list(t.user_attrs.get("problems") or [])
        text = "; ".join(problems) or t.user_attrs.get("error")
        if not text:
            if logged is None:
                logged = journal_failures(paths, spec.name)
            role = "ref" if t.user_attrs.get("role") == "reference" else f"t{t.number:04d}"
            text = logged.get(f"{spec.model}#{role}")
        kind = "설정 조합 불가" if problems else (fail_kind(text) if text else "기록 없음")
        out.append({"trial": t.number, "kind": kind, "error": (text or FAIL_UNKNOWN)[:300],
                    "settings": brief_settings(_actual(t))})
    return out


def failure_kinds(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _trained_with(t: Any) -> dict[str, Any]:
    """Everything a trial trained with: sampled or enqueued values, the extras riding along, the resolved job."""

    return {**t.params, **dict(t.user_attrs.get("extra_params") or {}), **dict(t.user_attrs.get("actual") or {})}


def reference_trials(spec: StudySpec, paths: LabPaths, trials: list[Any]) -> dict[str, Any] | None:
    """Every current-setting trial (all versions) with what it trained with, the keys on which they differ and the
    record they should reproduce. This is how a reference that missed its record is traced."""

    refs = [t for t in trials if t.user_attrs.get("role") == "reference"]
    if not refs:
        return None
    used = [_trained_with(t) for t in refs]
    keys = sorted({k for u in used for k in u})
    differ = [k for k in keys if len({json.dumps(u.get(k), sort_keys=True, default=str) for u in used}) > 1]
    record = None
    for t in refs:
        expected = t.user_attrs.get("expected")
        if expected and expected.get("settings"):
            record = {"settings": expected["settings"], "train": expected.get("train", "")}
    if record is None:
        path = study_dir(paths, spec.name).parent / "reference.json"
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows", []) if path.is_file() else []
        experiment = str(spec.enqueue_from.get("experiment") or "")
        mine = [r for r in rows if r.get("target") == spec.target and r.get("experiment") == experiment]
        if mine:
            record = {"settings": mine[0].get("settings", ""), "train": mine[0].get("train", "")}
    return {"trials": [{"number": t.number, "version": int(t.user_attrs.get("ref_version", 1)), "state": t.state.name,
                        "value": t.value, "epochs": t.user_attrs.get("epochs"), "params": u}
                       for t, u in zip(refs, used)],
            "differ": differ, "same": {k: used[0].get(k) for k in keys if k not in differ}, "record": record}


def _actual(t: Any) -> dict[str, Any]:
    """The settings a trial really trained with (derived and extra keys resolved), else the sampled ones."""

    return dict(t.user_attrs.get("actual") or t.params)


def report_study(spec: StudySpec | str | Path, paths: LabPaths, *, exp_dir: Path | None = None,
                 journal: Any = None, importance: bool = True, run: dict[str, Any] | None = None) -> dict[str, Any]:
    """Saturation curve, importances, top trials and pace of a study → ``work/hpo/<name>/summary.*`` and
    ``work/reports/hpo_<name>.txt`` (read through the lab server's /api/report). ``run`` holds the start time
    and worker count of the run in progress; ``importance=False`` skips fANOVA (the live updates)."""

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
                     "problems": "; ".join(t.user_attrs.get("problems", [])), "error": t.user_attrs.get("error", ""),
                     **{f"p.{k}": v for k, v in _actual(t).items()}})
    frame = pd.DataFrame(rows)
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    done.sort(key=lambda t: (t.datetime_complete or datetime.max, t.number))
    values = [float(t.value) for t in done]
    best_curve = list(np.minimum.accumulate(values)) if values else []
    expected = expected_best(values)
    sd_pct = _seed_sd_pct(spec, paths)
    sat = saturation(values, spec.window, spec.min_gain_pct, sd_pct)
    old = study_dir(paths, spec.name) / "summary.json"
    importances = (_importances(study) if importance and len(done) >= 10
                   else (json.loads(old.read_text(encoding="utf-8")).get("importances") or {}) if old.is_file() else {})
    ref, ref_version = _reference_of(done)
    best = min(done, key=lambda t: t.value) if done else None
    unit = unit_of(spec.target, None)
    counts = _counts(study)
    top = sorted(done, key=lambda t: t.value)[:10]
    run = dict(run or {})
    pace = progress(trials, spec.trials, since=None if run.get("finished") else run.get("started"),
                    workers=run.get("workers"))
    summary = {
        "name": spec.name, "target": spec.target, "model": spec.model, "objective": spec.objective, "unit": unit,
        "base": spec.base, "budget": spec.trials, "counts": counts, "progress": pace, "run": run,
        "best": {"number": best.number, "value": best.value, "params": _actual(best)} if best else None,
        "reference": {"number": ref.number, "value": ref.value, "version": ref_version,
                      "check": reference_check(spec, paths, ref)} if ref else None,
        "gain_vs_reference_pct": ((ref.value - best.value) / ref.value * 100) if ref and best and ref.value else None,
        "saturation": sat, "importances": importances,
        "best_so_far": [{"n": i + 1, "value": float(v)} for i, v in enumerate(best_curve)],
        "expected_best": [{"n": i + 1, "value": v} for i, v in enumerate(expected)],
        "top": [{"number": t.number, "value": t.value, "params": _actual(t), "role": t.user_attrs.get("role", ""),
                 "ref_version": t.user_attrs.get("ref_version")} for t in top],
        "failures": failures(spec, trials, paths),
        "references": reference_trials(spec, paths, trials),
        "refcheck": refcheck_results(spec, paths),
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


def _minutes(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 60:.0f}초"
    return f"{value:.1f}분" if value < 90 else f"{value / 60:.1f}시간"


def study_text(summary: dict[str, Any]) -> str:
    unit = summary.get("unit") or ""
    c = summary["counts"]
    sat = summary["saturation"]
    pace = summary.get("progress") or {}
    lines = [f"## 하이퍼파라미터 탐색: {summary['name']} ({summary['target']} · {summary['model']})",
             (f"기준 실험 {summary['base']} · 목표 지표 검증 {summary['objective']} ({unit or '단위 없음'}) · "
              f"예산 {summary['budget']} trial · 갱신 {summary['updated']}"),
             f"완료 {c['complete']} · 조기 중단 {c['pruned']} · 실패 {c['fail']} · 실행 중 {c['running']} · 대기 {c['waiting']}"]
    if pace.get("per_trial_min") is not None:
        end = (f" · 남은 {pace['left']}개 예상 {_minutes(pace['eta_min'])} (끝 {pace['eta_at'].replace('T', ' ')})"
               if pace.get("left") else "")
        lines.append(f"[진행] trial당 {_minutes(pace['per_trial_min'])} ({pace['basis']}){end}")
    lines.append("")
    best, ref = summary.get("best"), summary.get("reference")
    if best:
        lines.append(f"[최고] trial {best['number']}: {_fmt(best['value'])} {unit}")
    if ref:
        gain = summary.get("gain_vs_reference_pct")
        old = (" · 이전 방식 기준 trial(기록에 없던 설정을 탐색기가 골라 지금 설정과 다를 수 있음, "
               "ald-hpo requeue-reference로 다시 넣기)" if ref.get("version", 1) < REF_VERSION else "")
        if ref.get("version", 1) == 2:
            old = (" · 2판 기준 trial(학습 설정이 다른 run을 섞은 기록에서 학습 설정을 가져왔음, "
                   "ald-hpo requeue-reference로 다시 넣기)")
        lines.append(f"[지금 설정] trial {ref['number']}: {_fmt(ref['value'])} {unit}"
                     + (f" · 최고 ÷ 지금 설정 - 1 = {-gain:.1f}%" if gain is not None else "") + old)
        check = ref.get("check")
        if check:
            seeds = f"seed {check['seeds']}개 " if check.get("seeds") else ""
            lines.append(f"[재현 확인] 누적 기록 {seeds}평균 {_fmt(check['mean'])} {unit}, 이번 trial과 차이 "
                         f"{check['gap_pct']:+.1f}% (허용 ±{check['limit_pct']:.1f}%): "
                         + ("재현됨" if check["ok"] else "재현 안 됨, '지금 대비'는 참고만"))
    lines += ["", f"[포화 판정] {sat['verdict']}: {sat['reason']}"
              + (f" · seed 표준편차 {sat['seed_sd_pct']:.2f}%" if sat.get("seed_sd_pct") is not None else
                 " · seed 표준편차 기록 없음(최소 개선 기준만 씀)")]
    curve = [p["value"] for p in summary.get("best_so_far") or []]
    expected = [p["value"] for p in summary.get("expected_best") or []]
    if curve:
        marks = sorted({n for n in (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, len(curve)) if n <= len(curve)})
        lines += ["", "[포화 곡선] 완료 trial 수별 최고 기록과 기대 최고치(그만큼만 돌렸다면 기대되는 최고값)",
                  "trial 수  최고 기록  기대 최고치"]
        for n in marks:
            lines.append(f"{n:>8d}  {_fmt(curve[n - 1]):>9s}  {_fmt(expected[n - 1]):>10s}")
    refs = summary.get("references")
    if refs:
        lines += ["", ("[지금 설정 trial 비교] 기준 trial마다 실제로 학습한 설정. 기준 trial끼리 다른 값만 trial별로 "
                       "적고, 나머지는 '공통'에 적는다")]
        for t in refs["trials"]:
            diff = " · ".join(f"{k}={_fmt(t['params'].get(k), 4)}" for k in refs["differ"]) or "차이 없음"
            lines.append(f"- trial {t['number']} ({t['version']}판, {t['state']}, {_fmt(t['value'])} {unit}, "
                         f"{t.get('epochs') or '-'} epoch): {diff}")
        lines.append("- 공통: " + " · ".join(f"{k}={_fmt(v, 4)}" for k, v in sorted(refs["same"].items())))
        if refs.get("record"):
            lines.append(f"- 누적 기록의 설정: {refs['record']['settings']} | 학습 설정: {refs['record'].get('train') or '-'}")
    check = summary.get("refcheck")
    if check:
        lines += ["", f"[재현 실험] 지금 설정을 일반 학습 경로로 다시 학습 (검증 {check['metric']}, 요청 {check.get('submitted')})",
                  "    seed   누적 기록   다시 학습"]
        for seed in sorted({*check["record"], *check["rerun"]}):
            note = "  (탐색 seed)" if seed == check["study_seed"] else ""
            lines.append(f"{seed:>8d}  {_fmt(check['record'].get(seed)):>10s}  {_fmt(check['rerun'].get(seed)):>10s}{note}")
        lines.append(f"판정: {check['verdict']}")
    fails = summary.get("failures") or []
    if fails:
        kinds = ", ".join(f"{k} {v}" for k, v in failure_kinds(fails).items())
        lines += ["", f"[실패] {len(fails)}개: {kinds}"]
        lines += [f"- trial {f['trial']} ({f['kind']}): {f['settings']} | {f['error']}" for f in fails]
    imp = summary.get("importances") or {}
    if imp:
        lines += ["", "[설정 중요도] fANOVA, 성능 변동에 대한 기여 몫(합 1)"]
        lines += [f"- {k}: {v:.3f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])]
    top = summary.get("top") or []
    if top:
        lines += ["", "[상위 trial] 실제로 학습한 설정"]
        for t in top:
            params = " · ".join(f"{k}={_fmt(v, 3)}" for k, v in sorted(t["params"].items()))
            role = " (지금 설정)" if t.get("role") == "reference" else ""
            lines.append(f"- trial {t['number']}{role}: {_fmt(t['value'])} {unit} | {params}")
    return "\n".join(lines) + "\n"


def refresh_report(paths: LabPaths, name: str, *, max_age_seconds: float = 30.0) -> bool:
    """Rewrite ``reports/hpo_<name>.txt`` from the study journal when it is missing or older than the journal (a
    study run by older code writes it only at the end). True when rewritten."""

    journal = study_dir(paths, name) / "study.journal"
    if not journal.is_file():
        return False
    report = Path(paths.work_root) / "reports" / f"hpo_{name}.txt"
    if report.is_file() and report.stat().st_mtime >= journal.stat().st_mtime - max_age_seconds:
        return False
    path = next((p for p in study_files() if p.stem == name), None)
    if path is None:
        return False
    spec = load_study(path)
    old = study_dir(paths, name) / "summary.json"
    run = (json.loads(old.read_text(encoding="utf-8")).get("run") or {}) if old.is_file() else {}
    if not run.get("started"):  # a run of older code kept no start time: the queue request has it
        from .launch import list_requests

        req = next((r for r in list_requests(Path(paths.work_root), include_hidden=True)
                    if r.get("kind") == "hpo" and str(r.get("name")) == name and not r.get("smoke")), None)
        if req and req.get("started"):
            run = {**run, "started": req["started"], "workers": _workers(req.get("gpus")),
                   "finished": req.get("status") != "running"}
    report_study(spec, paths, importance=False, run=run)
    return True


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
            "예상 끝": ((s.get("progress") or {}).get("eta_at") or "-").replace("T", " ")
            if (s.get("progress") or {}).get("left") else "-",
        })
    text = ["## 하이퍼파라미터 탐색 전체 (검증 run, 시험 run 미사용)",
            "'지금 대비' = 최고 ÷ 지금 설정 - 1 (음수면 나아짐). '포화' = 최근 판정 구간에서 최고 기록 개선이 기준 미만.", ""]
    text.append(pd.DataFrame(rows).to_string(index=False) if rows else "탐색 기록이 없습니다.")
    reports = Path(paths.work_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "hpo_overview.txt"
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- dashboard status

STATE_NAMES = {"running": "실행 중", "queued": "대기", "done": "완료", "partial": "멈춤", "new": "미실행",
               "failed": "실패", "cancelled": "취소"}
_STATUS_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}


def study_files(folder: Path | None = None) -> list[Path]:
    folder = folder or PROJECT_ROOT / "configs" / "hpo"
    return sorted(folder.glob("*.yaml")) if folder.is_dir() else []


def _workers(gpus: Any) -> int:
    text = str(gpus or "auto")
    if text == "cpu":
        return 1
    if text == "auto":
        return max(1, len([g for g in os.environ.get("GPUS", "0,1,2,3").split(",") if g.strip()]))
    return max(1, len([g for g in text.split(",") if g.strip()]))


def _live_info(spec: StudySpec, paths: LabPaths, request: dict[str, Any] | None) -> dict[str, Any]:
    """Counts, best, current setting, saturation and pace of one study straight from its journal."""

    from optuna.trial import TrialState

    study = open_study(spec, paths)
    trials = study.get_trials(deepcopy=False)
    done = sorted((t for t in trials if t.state == TrialState.COMPLETE),
                  key=lambda t: (t.datetime_complete or datetime.max, t.number))
    values = [float(t.value) for t in done]
    best = min(done, key=lambda t: t.value) if done else None
    ref, version = _reference_of(done)
    pending_ref = any(t.user_attrs.get("role") == "reference" and int(t.user_attrs.get("ref_version", 1)) > version
                      and t.state in (TrialState.RUNNING, TrialState.WAITING) for t in trials)
    fails = failures(spec, trials, paths)
    sat = saturation(values, spec.window, spec.min_gain_pct, _seed_sd_pct(spec, paths))
    running = bool(request and request.get("status") == "running")
    summary_path = study_dir(paths, spec.name) / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    workers = _workers(request.get("gpus")) if request else int((old.get("run") or {}).get("workers") or 4)
    pace = progress(trials, spec.trials, since=request.get("started") if running else None, workers=workers)
    imp = old.get("importances") or {}
    return {"counts": _counts(study), "finished": pace["finished"], "progress": pace,
            "best": best.value if best else None, "best_number": best.number if best else None,
            "reference": ref.value if ref else None, "ref_version": version, "ref_pending": pending_ref,
            "ref_check": reference_check(spec, paths, ref),
            "failures": fails, "failure_kinds": failure_kinds(fails),
            "gain_pct": ((ref.value - best.value) / ref.value * 100) if ref and best and ref.value else None,
            "saturation": sat["verdict"], "saturation_reason": sat["reason"],
            "importance_top": max(imp, key=imp.get) if imp else None}


METRIC_NAMES = {"mae": "검증 MAE", "mae_cycle": "사이클 안 MAE", "mae_other": "사이클 밖 MAE"}


def study_label(spec: StudySpec, task: str) -> str:
    """What a study searches, in a few words: the task, the kind of model and the metric it optimizes."""

    metric = METRIC_NAMES.get(spec.objective, spec.objective)
    if task == "forecast":
        return f"미래값 예측, {metric}"
    if "cycleonly" in spec.base:  # its validation windows all lie inside the ALD cycles
        return "가상 센서 0.5초 사이클 전용 모델, " + ("사이클 안 MAE" if spec.objective == "mae" else metric)
    if "res05" in spec.base:
        return f"가상 센서 0.5초 공정 전체 모델, {metric}"
    return f"가상 센서 1초 공정 전체 모델, {metric}"


def request_label(q: dict[str, Any]) -> tuple[str, str | None]:
    """(kind, study) of a queue request: 극한 탐색, 재현 확인 or 상위 5개 확정 of a study, or 실험."""

    name = str(q.get("name") or "")
    if q.get("kind") == "hpo":
        return "극한 탐색", name.removesuffix("_smoke")
    if name.endswith(REFCHECK_SUFFIX):
        return "재현 확인", name[: -len(REFCHECK_SUFFIX)]
    match = re.match(r"^(.*)_top\d+(?:_w.*)?$", name)
    if match:
        return "상위 5개 확정", match.group(1)
    return "실험", None


def top_results(spec: StudySpec, paths: LabPaths) -> dict[str, Any] | None:
    """The best of the top settings retrained with several seeds (상위 5개 확정): seed mean of the study's metric."""

    from .ledger import _run_rows

    root = Path(paths.work_root) / "experiments"
    dirs = sorted(d for d in root.glob(f"{spec.name}_top*") if d.is_dir()) if root.is_dir() else []
    values: dict[str, dict[int, float]] = {}
    for folder in dirs:
        for exp_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for r in _run_rows(exp_dir, None):
                value = r.get(spec.objective)
                if (r["model"] == spec.model and r["target"] == spec.target and r.get("job_status", "ok") == "ok"
                        and value is not None and not pd.isna(value)):
                    values.setdefault(str(r.get("label")), {})[int(r["seed"])] = float(value)
    if not values:
        return None
    means = {label: (float(np.mean(list(v.values()))), len(v)) for label, v in values.items()}
    best = min(means, key=lambda k: means[k][0])
    return {"label": best, "mean": means[best][0], "seeds": means[best][1], "settings": len(means)}


def queue_plan(rows: list[dict[str, Any]], requests: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """The running request and the queued ones in the order the worker takes them, each with its place, a readable
    title and expected start and end. A study's time comes from the pace of studies of the same task (or another
    task's, marked), another request's from its ``estimate_minutes``; after a request of unknown length the later
    times are left open."""

    from .launch import remaining_minutes, run_order

    by_request = {r["request"]: r for r in rows if r.get("request")}
    by_name = {r["name"]: r for r in rows if "name" in r}
    paces: dict[str, list[float]] = {}
    for row in rows:
        per = (row.get("progress") or {}).get("per_trial_min")
        if per:
            paces.setdefault(row.get("task", ""), []).append(per)
    every = [v for vs in paces.values() for v in vs]
    clock, known = now, True
    plan = []
    for place, q in enumerate(run_order(requests)):
        kind, study = request_label(q)
        row = by_request.get(q["id"]) if kind == "극한 탐색" else None
        item = {"id": q["id"], "status": q["status"], "place": place, "kind": kind, "study": study,
                "title": f"{kind} {study}" if study else str(q.get("memo") or q.get("name")),
                "n_jobs": q.get("n_jobs"), "progress": q.get("progress"), "start_at": None, "end_at": None, "basis": ""}
        minutes = None
        if q["status"] == "running":
            item["start_at"] = q.get("started")
            eta = (row.get("progress") or {}).get("eta_min") if row else None
            minutes = eta if eta is not None else remaining_minutes(q)
        else:
            item["start_at"] = clock.isoformat(timespec="minutes") if known else None
            if row is not None and every:
                same = paces.get(row.get("task", ""))
                minutes = float(np.median(same or every)) * max(0, row["budget"] - row["finished"])
                item["basis"] = "같은 과제 탐색의 속도로 계산" if same else "다른 과제 탐색의 속도를 빌린 추정"
            elif q.get("estimate_minutes") is not None:
                minutes = float(q["estimate_minutes"])
                item["basis"] = "기록의 학습 시간으로 추정"
            elif kind == "재현 확인" and study in by_name and (by_name[study].get("record") or {}).get("fit_minutes"):
                fit = float(by_name[study]["record"]["fit_minutes"])  # seeds train side by side, one per GPU
                minutes = fit * -(-int(q.get("n_jobs") or 1) // _workers(q.get("gpus")))
                item["basis"] = "기록의 학습 시간으로 추정"
        if minutes is None or not known:
            known = False
        else:
            clock = (now if q["status"] == "running" else clock) + timedelta(minutes=float(minutes))
            item["end_at"] = clock.isoformat(timespec="minutes")
        plan.append(item)
        if row is not None:  # a study counts its trials (the request also counts the baseline job)
            item.update(progress=(row.get("finished", 0), row["budget"]), unit="trial")
            row.update(place=place, start_at=item["start_at"], end_at=item["end_at"], end_basis=item["basis"])
    return plan


def studies_status(paths: LabPaths, folder: Path | None = None) -> dict[str, Any]:
    """Every study file with its state, progress, result against the record and place in the queue, plus the queue
    itself in run order (studies, reproduction checks, top-5 confirmations and other experiments), for the panel."""

    from .launch import list_requests
    from .training.config import load_experiment
    from .training.selection import unit_of

    requests = list_requests(Path(paths.work_root), include_hidden=True)  # newest first
    latest: dict[str, dict[str, Any]] = {}
    for r in requests:
        if r.get("kind") == "hpo" and not r.get("smoke"):
            latest.setdefault(str(r["name"]), r)
    ledger_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for path in study_files(folder):
        try:
            spec = load_study(path)
            task = load_experiment(_base_path(spec.base)).task
        except Exception as exc:  # noqa: BLE001  (a broken file shows as an error row)
            rows.append({"name": path.stem, "file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        req = latest.get(spec.name)
        row: dict[str, Any] = {
            "name": spec.name, "file": path.name, "description": spec.description, "label": study_label(spec, task),
            "target": spec.target, "model": spec.model, "objective": spec.objective,
            "unit": unit_of(spec.target, None), "budget": spec.trials, "task": task,
            "request": req.get("id") if req else None, "request_status": req.get("status") if req else None,
            "report": f"hpo_{spec.name}.txt", "finished": 0, "counts": None, "progress": None,
            "has_reference_source": bool(spec.enqueue_from), "refcheck": None, "record": None, "top": None,
        }
        journal = study_dir(paths, spec.name) / "study.journal"
        if journal.is_file():
            key = (journal.stat().st_mtime, row["request_status"], (req or {}).get("started"))
            cached = _STATUS_CACHE.get(spec.name)
            if cached and cached[0] == key and row["request_status"] != "running":
                info = cached[1]
            else:
                try:
                    info = _live_info(spec, paths, req)
                except Exception as exc:  # noqa: BLE001  (optuna missing, journal being written …)
                    info = {"error": f"{type(exc).__name__}: {exc}"}
                _STATUS_CACHE[spec.name] = (key, info)
            row.update(info)
        for key_name, reader, extra in (("record", record_baseline, (ledger_cache,)),
                                        ("refcheck", refcheck_results, ()), ("top", top_results, ())):
            try:  # each can exist before the study ever ran
                row[key_name] = reader(spec, paths, *extra)
            except Exception as exc:  # noqa: BLE001  (a half-written file: try again on the next refresh)
                row[key_name] = None
                row.setdefault("notes", []).append(f"{key_name}: {type(exc).__name__}: {exc}")
        status = row["request_status"]
        if status in ("running", "queued"):
            row["state"] = status
        elif row["finished"] >= spec.trials:
            row["state"] = "done"
        elif journal.is_file():
            row["state"] = "partial"
        else:
            row["state"] = status if status in ("failed", "cancelled") else "new"
        rows.append(row)

    now = datetime.now()
    plan = queue_plan(rows, requests, now)
    for row in rows:  # follow-up requests of a study (재현 확인, 상위 5개 확정) waiting or running
        row["followups"] = [i for i in plan if i["study"] == row["name"] and i["kind"] != "극한 탐색"]
    ends = [i["end_at"] for i in plan if i["end_at"]]
    states = {k: sum(1 for r in rows if r.get("state") == k) for k in STATE_NAMES}
    return {"rows": rows, "states": states, "finished": sum(r.get("finished", 0) for r in rows),
            "budget": sum(r.get("budget", 0) for r in rows if "error" not in r), "queue": plan,
            "queue_end": ends[-1] if ends and all(i["end_at"] for i in plan) else None,
            "updated": now.isoformat(timespec="seconds")}


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


REFCHECK_SUFFIX = "_refcheck"


def full_trial_minutes(spec: StudySpec, paths: LabPaths) -> float | None:
    """Median minutes of the study's trials that trained to the end (not pruned): the length of one retraining."""

    from optuna.trial import TrialState

    done = [t.duration.total_seconds() / 60 for t in open_study(spec, paths).get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.duration]
    return float(np.median(done)) if done else None


def refcheck_doc(spec: StudySpec, paths: LabPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    """An experiment that retrains the current setting through the normal training path (not the search) with the
    record's seeds and the study's seed, from the record run's own experiment file: data, window and train exactly
    as they were. Whether the record comes back tells a seed effect from a search-path or data difference apart.
    Returns the document and the record it should reproduce (kept in ``work/hpo/<name>/refcheck.json``)."""

    conf = spec.enqueue_from
    if not conf:
        raise StudyError(f"{spec.name}: enqueue_from가 없어 지금 설정을 알 수 없습니다")
    metric = str(conf.get("metric", spec.objective))
    rows = reference_rows(paths.work_root, str(conf["experiment"]), model=spec.model, metric=metric,
                          min_seeds=int(conf.get("min_seeds", 1)))
    rows = rows[rows["target"] == spec.target] if len(rows) else rows
    if not len(rows):
        raise StudyError(f"{spec.name}: 누적 기록에 지금 설정이 없습니다")
    row = rows.iloc[0]
    doc = yaml.safe_load((Path(row["exp_dir"]) / "experiment.yaml").read_text(encoding="utf-8"))
    record_seeds = sorted(int(s) for s in (row["per_seed"] or {}))
    doc["name"] = f"{spec.name}{REFCHECK_SUFFIX}"
    doc["targets"] = [spec.target]
    doc["seeds"] = sorted({*record_seeds, int(spec.seed)})
    doc["models"] = [{"name": spec.model, "label": f"{spec.model}_refcheck", "params": dict(row["params"] or {})}]
    doc.setdefault("window", {})["context_seconds"] = float(row["context_seconds"])
    data = doc.get("data") or {}
    if row["input_set"] != "default" and row["input_set"] in (data.get("input_sets") or {}):
        data["input_sets"] = {row["input_set"]: data["input_sets"][row["input_set"]]}
    doc["study"] = f"hpo_ref:{spec.name}"
    doc["description"] = (f"{spec.name}: 지금 설정을 일반 학습 경로로 다시 학습 (기록 seed {record_seeds} + 탐색 seed "
                          f"{spec.seed}, 기록 run의 실험 파일 그대로)")
    record = {"experiment": doc["name"], "metric": metric, "mean": float(row["mean"]),
              "fit_minutes": None if row.get("fit_minutes") is None or pd.isna(row.get("fit_minutes"))
              else float(row["fit_minutes"]),
              "sd": None if pd.isna(row["sd"]) else float(row["sd"]),
              "per_seed": {str(k): float(v) for k, v in dict(row["per_seed"] or {}).items()},
              "record_run": str(row["exp_dir"]), "submitted": datetime.now().isoformat(timespec="seconds")}
    return doc, record


def refcheck_results(spec: StudySpec, paths: LabPaths) -> dict[str, Any] | None:
    """The reproduction experiment against its record, seed by seed, with a verdict (None before one was queued)."""

    from .ledger import _run_rows

    side = study_dir(paths, spec.name) / "refcheck.json"
    if not side.is_file():
        return None
    info = json.loads(side.read_text(encoding="utf-8"))
    metric = info["metric"]
    root = Path(paths.work_root) / "experiments" / info["experiment"]
    rerun: dict[int, float] = {}
    for exp_dir in sorted(root.iterdir()) if root.is_dir() else []:  # later runs replace earlier ones
        for r in _run_rows(exp_dir, None) if exp_dir.is_dir() else []:
            value = r.get(metric)
            if (r["model"] == spec.model and r["target"] == spec.target and r.get("job_status", "ok") == "ok"
                    and value is not None and not pd.isna(value)):
                rerun[int(r["seed"])] = float(value)
    record = {int(k): float(v) for k, v in (info.get("per_seed") or {}).items()}
    out: dict[str, Any] = {"metric": metric, "record": record, "rerun": rerun, "study_seed": int(spec.seed),
                           "record_mean": info.get("mean"), "submitted": info.get("submitted"), "verdict": None}
    common = [s for s in record if s in rerun]
    if not common:
        out["verdict"] = "결과 기다리는 중"
        return out
    rec = float(np.mean([record[s] for s in common]))
    new = float(np.mean([rerun[s] for s in common]))
    sd = info.get("sd")
    limit = max(REF_TOLERANCE_PCT, 3 * sd / rec * 100) if sd and rec else REF_TOLERANCE_PCT
    out.update(common=common, record_common=rec, rerun_common=new, gap_pct=(new - rec) / rec * 100, limit_pct=limit)
    reproduced = abs(out["gap_pct"]) <= limit
    mine = rerun.get(int(spec.seed))
    if not reproduced:
        out["verdict"] = (f"일반 학습 경로에서도 기록이 재현되지 않음({out['gap_pct']:+.1f}%): 기록 이후 코드, 데이터 또는 "
                          "실험 파일 밖의 설정이 바뀌었을 수 있음")
    elif mine is None:
        out["verdict"] = "기록 seed는 재현됨, 탐색 seed 결과 기다리는 중"
    else:
        seed_gap = (mine - rec) / rec * 100
        out["study_seed_gap_pct"] = seed_gap
        out["verdict"] = (f"기록 seed는 재현됨. 탐색 seed {spec.seed}는 기록 평균 대비 {seed_gap:+.1f}%: "
                          + ("seed 차이가 큼(trial 하나의 값은 이만큼 흔들림)" if abs(seed_gap) > limit else
                             "seed 차이 수준이 아님, 기준 trial과 다르면 탐색 경로를 확인"))
    return out


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
    p_req = sub.add_parser("requeue-reference",
                           help="지금 설정 trial을 (다시) 맨 앞에 넣는다 · 실행 중인 탐색에도 바로 들어감")
    p_req.add_argument("study")
    p_ref = sub.add_parser("reference", help="지금 설정의 seed 평균 ± 표준편차 (누적 기록에서, 학습 없음)")
    p_ref.add_argument("--experiment", required=True, help="기준 실험 이름 (예: g1_forecast_params)")
    p_ref.add_argument("--model", default="patchtst")
    p_ref.add_argument("--metric", default="mae", choices=OBJECTIVES)
    p_ref.add_argument("--min-seeds", type=int, default=2)
    p_ref.add_argument("--write-configs", action="store_true",
                       help="대상마다 지금 설정을 다시 재는 실험 YAML을 configs/experiments/에 쓴다")
    p_ref.add_argument("--seeds", default="42,43,44")
    for p in (p_run, p_report, p_top, p_ref, p_req):
        p.add_argument("--work-root")
        p.add_argument("--data-root")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, work_root=args.work_root)
    if args.command == "run":
        from .training.runner import parse_devices

        exp_dir = run_study(args.study, paths, devices=parse_devices(args.gpus), trials=args.trials, smoke=args.smoke)
        print(f"\n결과: {exp_dir}/hpo_summary.txt")
        return 0
    if args.command == "requeue-reference":
        spec = load_study(args.study)
        if enqueue_reference(spec, paths):
            print(f"{spec.name}: 지금 설정 trial을 맨 앞에 넣었습니다 (실행 중이면 다음 trial로 학습)")
        else:
            print(f"{spec.name}: 이미 들어가 있거나, 누적 기록에서 지금 설정을 찾지 못했습니다")
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
        ref_path = save_reference_rows(paths, args.experiment, args.model, rows)
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
