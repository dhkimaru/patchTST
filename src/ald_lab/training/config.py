"""Experiment YAML → jobs.

An experiment names one task, its data settings, targets and a list of
models. A model may carry a ``grid`` of hyperparameters; every combination
becomes a job and the best one is chosen on validation (never on test).
See ``configs/experiments/*.yaml``.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..data.columns import core_columns, grade_columns
from ..data.splits import DEFAULT_SPLIT
from ..models.classic import CLASSIC_MODELS
from ..models.constraints import combo_problems
from ..models.deep import BACKBONES
from ..paths import PROJECT_ROOT
from .trainer import TrainSettings
from .windows import REGIONS

TASKS = ("forecast", "virtual_sensor", "thickness")
BASELINES = {"forecast": "persistence", "virtual_sensor": "template", "thickness": "mean"}
BASELINE_LABELS = {"persistence": "현재값 유지", "template": "레시피 평균", "schedule": "레시피 일정 평균",
                   "mean": "학습 평균"}
VM_MODELS = ("mean", "ridge", "pls", "lightgbm")
# The previous project's 38-signal schema (F_PRIMARY without Step ID), kept for comparison.
LEGACY_PRIMARY = (
    "MFCMon_N2-1", "MFCMon_N2-2", "MFCMon_N2-3", "MFCMon_N2-4", "MFCMon_F.PWR", "MFCMon_L.POS",
    "MFCMon_P.POS", "MFCMon_DCS", "MFCMon_NH3", "MFCMon_F2", "APCValveMon", "PressAct", "AUXMon_VG11",
    "VG11", "VG12", "VG13", "TempAct_U", "TempAct_CU", "TempAct_C", "TempAct_CL", "TempAct_L",
    "ValveAct_2", "ValveAct_3", "ValveAct_4", "ValveAct_5", "ValveAct_9", "ValveAct_12", "ValveAct_14",
    "ValveAct_16", "ValveAct_26", "ValveAct_28", "ValveAct_29", "ValveAct_60", "ValveAct_63",
    "ValveAct_73", "ValveAct_80", "ValveAct_89", "ValveAct_90",
)
INPUT_SETS = ("all", "core", "core_ab", "legacy_primary", "A_all", "A_plus")
SMOKE = {"train_runs": 4, "val_runs": 2, "max_epochs": 2, "windows_per_epoch": 4096, "val_windows": 2048}


class ConfigError(ValueError):
    """Raised for an invalid experiment YAML."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    label: str | None = None
    # Random search: draw ``trials`` valid combinations from ``grid`` instead of training all of them.
    search: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        if self.name in BACKBONES:
            return "deep"
        if self.name in CLASSIC_MODELS:
            return "classic"
        return "vm"


@dataclass(frozen=True)
class Job:
    job_id: str
    target: str
    model: str
    kind: str
    label: str
    params: dict[str, Any]
    seed: int
    variant: dict[str, Any]
    grid_size: int
    input_set: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ExperimentConfig:
    name: str
    task: str
    targets: list[str]
    models: list[ModelSpec]
    description: str = ""
    split: str = str(DEFAULT_SPLIT)
    sample_seconds: float = 1.0
    region: str = "all"
    inputs: Any = "all"
    input_sets: dict[str, Any] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)
    exclude_target_family: bool = True
    context_seconds: float = 192.0
    horizons_seconds: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0])
    train: TrainSettings = field(default_factory=TrainSettings)
    seeds: list[int] = field(default_factory=lambda: [42])
    vm: dict[str, Any] = field(default_factory=dict)
    smoke: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline(self) -> str:
        return BASELINES[self.task]

    @property
    def context_steps(self) -> int:
        return max(1, round(self.context_seconds / self.sample_seconds))

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        if self.task == "virtual_sensor":
            return (0,)
        return tuple(max(1, round(h / self.sample_seconds)) for h in self.horizons_seconds)

    @property
    def horizon_labels(self) -> list[float]:
        return [0.0] if self.task == "virtual_sensor" else [float(h) for h in self.horizons_seconds]

    def split_path(self) -> Path:
        path = Path(self.split)
        if not path.suffix:
            path = PROJECT_ROOT / "configs" / "splits" / f"{self.split}.yaml"
        return path


def _models(doc: list[Any], task: str) -> list[ModelSpec]:
    out = []
    for item in doc:
        if isinstance(item, str):
            item = {"name": item}
        name = str(item["name"])
        search = dict(item.get("search") or {})
        spec = ModelSpec(name=name, params=dict(item.get("params") or {}), grid=dict(item.get("grid") or {}),
                         label=item.get("label"), search=search)
        if search and (int(search.get("trials", 0)) < 1 or not spec.grid):
            raise ConfigError(f"{name}: search needs trials ≥ 1 and a grid to draw from")
        allowed = VM_MODELS if task == "thickness" else (*BACKBONES, *CLASSIC_MODELS)
        if name not in allowed:
            raise ConfigError(f"model {name!r} is not available for task {task}; choose from {sorted(allowed)}")
        if task == "virtual_sensor" and name == "persistence":
            raise ConfigError("persistence needs the target as an input; it does not apply to virtual sensors")
        for key, values in spec.grid.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"grid {name}.{key} must be a non-empty list")
        out.append(spec)
    return out


def load_experiment(path: str | Path) -> ExperimentConfig:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    task = doc.get("task")
    if task not in TASKS:
        raise ConfigError(f"task must be one of {TASKS}: {task!r}")
    data = dict(doc.get("data") or {})
    window = dict(doc.get("window") or {})
    cfg = ExperimentConfig(
        name=str(doc["name"]),
        task=task,
        description=str(doc.get("description") or ""),
        targets=[str(t) for t in doc["targets"]],
        models=_models(doc.get("models") or [], task),
        split=str(data.get("split", DEFAULT_SPLIT)),
        sample_seconds=float(data.get("sample_seconds", 1.0)),
        region=str(data.get("region", "all")),
        inputs=data.get("inputs", "all"),
        input_sets=dict(data.get("input_sets") or {}),
        exclude=[str(c) for c in data.get("exclude") or []],
        exclude_target_family=bool(data.get("exclude_target_family", True)),
        context_seconds=float(window.get("context_seconds", 192)),
        horizons_seconds=[float(h) for h in window.get("horizons_seconds", [10, 20, 30])],
        train=TrainSettings.from_dict(doc.get("train")),
        seeds=[int(s) for s in doc.get("seeds", [42])],
        vm=dict(doc.get("vm") or {}),
        raw=copy.deepcopy(doc),
    )
    if not cfg.input_sets:
        cfg.input_sets = {"default": cfg.inputs}
    if cfg.baseline not in [m.name for m in cfg.models]:
        cfg.models.insert(0, ModelSpec(cfg.baseline))
    if cfg.region not in REGIONS:
        raise ConfigError(f"data.region must be one of {REGIONS}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg.name):
        raise ConfigError("name may use letters, digits, '_', '-', '.' only")
    if not cfg.models:
        raise ConfigError("models is empty")
    return cfg


def apply_smoke(cfg: ExperimentConfig) -> ExperimentConfig:
    """Tiny version of an experiment for checking that everything runs end to end."""

    out = copy.deepcopy(cfg)
    out.smoke = True
    out.name = f"{cfg.name}_smoke"
    out.train.max_epochs = SMOKE["max_epochs"]
    out.train.windows_per_epoch = SMOKE["windows_per_epoch"]
    out.train.val_windows = SMOKE["val_windows"]
    out.seeds = out.seeds[:1]
    smoke_models = []
    for spec in out.models:
        params = dict(spec.params)
        if spec.name == "lightgbm":
            params.update(n_estimators=50, max_train_windows=20_000, max_val_windows=5_000)
        if spec.name == "ridge":
            params.update(max_train_windows=20_000, max_val_windows=5_000)
        grid = {k: v[:2] for k, v in spec.grid.items()}
        search = {**spec.search, "trials": min(2, int(spec.search.get("trials", 2)))} if spec.search else {}
        smoke_models.append(ModelSpec(spec.name, params, grid, spec.label, search))
    out.models = smoke_models
    return out


def search_combos(spec: ModelSpec, context_seconds: float) -> list[dict[str, Any]]:
    """``trials`` distinct combinations of ``spec.grid`` drawn at random (seeded) among those the model accepts."""

    keys = list(spec.grid)
    trials = int(spec.search.get("trials", 1))
    rng = random.Random(int(spec.search.get("seed", 0)))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(trials * 200):
        combo = {k: rng.choice(spec.grid[k]) for k in keys}
        key = json.dumps(combo, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        if not combo_problems(spec.name, {**spec.params, **combo}, context_seconds):
            out.append(combo)
            if len(out) == trials:
                break
    return out


def _variant_label(name: str, variant: dict[str, Any], label: str | None) -> str:
    base = label or name
    if not variant:
        return base
    return base + "[" + ",".join(f"{k}={v}" for k, v in variant.items()) + "]"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def expand_jobs(cfg: ExperimentConfig, *, only_models: list[str] | None = None,
                only_targets: list[str] | None = None) -> list[Job]:
    jobs = []
    set_names = list(cfg.input_sets)
    for target in cfg.targets:
        if only_targets and target not in only_targets:
            continue
        for spec in cfg.models:
            if only_models and not ({spec.name, spec.label} & set(only_models)) and spec.name != cfg.baseline:
                continue
            keys = list(spec.grid)
            if spec.search:
                variants = search_combos(spec, cfg.context_seconds)
            else:
                variants = [dict(zip(keys, combo)) for combo in itertools.product(*(spec.grid[k] for k in keys))] or [{}]
            stochastic = spec.kind == "deep" or spec.name == "lightgbm"
            # Baselines ignore the inputs (persistence, template, mean): run them once per target.
            sets = set_names[:1] if spec.name == cfg.baseline else set_names
            for input_set in sets:
                for trial, variant in enumerate(variants, start=1):
                    params = {**spec.params, **variant}
                    label = (f"{spec.label or spec.name}#{trial:02d}" if spec.search
                             else _variant_label(spec.name, variant, spec.label))
                    if len(set_names) > 1 and spec.name != cfg.baseline:
                        label += f"@{input_set}"
                    for seed in (cfg.seeds if stochastic else cfg.seeds[:1]):
                        digest = hashlib.sha1(json.dumps([target, label, params, seed, input_set], sort_keys=True,
                                                         default=str).encode()).hexdigest()[:6]
                        job_id = _safe(f"{target}__{spec.label or spec.name}__s{seed}__{digest}")
                        jobs.append(Job(job_id, target, spec.name, spec.kind, label, params, seed, variant,
                                        len(variants), input_set))
    # Baselines first (every other job is compared against them), then grouped by target and input set
    # so a worker can reuse the prepared window sets of neighbouring jobs.
    jobs.sort(key=lambda j: (j.model != cfg.baseline, cfg.targets.index(j.target), set_names.index(j.input_set),
                             j.kind == "deep"))
    return jobs


def resolve_inputs(
    cfg: ExperimentConfig,
    target: str,
    *,
    usable: list[str],
    dictionary: pd.DataFrame,
    near_copies: pd.DataFrame | None = None,
    input_set: str | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Input columns for one target and the reasons columns were removed."""

    spec = cfg.input_sets.get(input_set, cfg.inputs) if input_set else cfg.inputs
    if isinstance(spec, str):
        if spec == "all":
            base = list(usable)
        elif spec == "core":
            base = core_columns(dictionary, ("A",))
        elif spec == "core_ab":
            base = core_columns(dictionary, ("A", "B"))
        elif spec == "legacy_primary":
            base = list(LEGACY_PRIMARY)
        elif spec == "A_all":
            base = grade_columns(dictionary, ("A",))
        elif spec == "A_plus":
            base = grade_columns(dictionary, ("A",), plus=True)
        else:
            raise ConfigError(f"data.inputs must be a list or one of {INPUT_SETS}: {spec}")
    else:
        base = [str(c) for c in spec]
    removed: dict[str, list[str]] = {}
    unusable = [c for c in base if c not in usable]
    if unusable:
        removed["상수·중복 컬럼"] = unusable
    inputs = [c for c in base if c in usable]
    if cfg.task == "forecast":
        if target not in inputs:
            inputs = [target, *inputs]
    elif cfg.task == "virtual_sensor":
        drop = {target}
        copies = set()
        if near_copies is not None and len(near_copies):
            for row in near_copies.itertuples():
                if target in (row.column_a, row.column_b):
                    copies |= {row.column_a, row.column_b}
        drop |= copies
        if cfg.exclude_target_family:
            family = dictionary.set_index("column")["family"]
            fam = family.get(target)
            if fam is not None:
                drop |= set(family.index[family == fam])
        removed["대상과 같은 계열·복사본"] = sorted(c for c in inputs if c in drop)
        inputs = [c for c in inputs if c not in drop]
    extra = [c for c in inputs if c in cfg.exclude and c != target]
    if extra:
        removed["설정에서 제외"] = extra
    inputs = [c for c in inputs if c not in extra]
    if not inputs:
        raise ConfigError(f"{target}: no input columns left")
    return inputs, removed
