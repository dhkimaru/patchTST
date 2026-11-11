"""Run an experiment end to end.

1. Prepare data once (bundles under ``work/prepared/``) and journal it as the
   *전처리* stage.
2. Run the task's baseline per target (persistence / recipe template).
3. Run every other job, one worker process per GPU, and journal each one as
   *학습* (or *하이퍼파라미터 탐색* when it is part of a grid).
4. Summarize: per-target ranking on validation, chosen grid variants,
   improvement over the baseline, time — written to ``summary.md`` and the
   journal as *평가*.

Test runs are never opened here; see :func:`run_final_test`.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.columns import load_column_dictionary
from ..data.splits import RunSplit, load_split
from ..journal import Journal
from ..models.classic import CLASSIC_REGISTRY, RecipeTemplate, StepSchedule
from ..models.deep import build_deep_model, count_parameters, resolve_params
from ..paths import LabPaths
from .config import (
    BASELINE_LABELS,
    SMOKE,
    ExperimentConfig,
    Job,
    apply_smoke,
    expand_jobs,
    load_experiment,
    resolve_inputs,
)
from .evaluate import describe, improvement, regression_metrics
from .prepare import POSITION_CAP_ROWS, Bundle, build_bundle, load_bundle, usable_columns
from .trainer import TrainSettings, predict, train_model
from .windows import (Scaler, WindowSet, constant_columns, fit_scaler, offset_targets, residual_scale, window_ends,
                      with_recency_weights)
from .windows import near_copies as find_near_copies

PREDICTION_SAMPLE_ROWS = 25_000
SAVE_VALIDATION_PREDICTIONS = os.environ.get("ALD_SAVE_VAL_PREDICTIONS", "1") != "0"  # predictions_val.parquet per job
# Window sets kept per process; jobs are ordered by target so neighbours share them.
TARGET_CACHE = 2


def parse_devices(spec: str | None) -> list[str]:
    """``"0,1,2,3"`` → cuda devices; ``"cpu"``; default: every visible GPU, else CPU."""

    if spec in (None, "", "auto"):
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]
    if spec == "cpu":
        return ["cpu"]
    return [f"cuda:{int(part)}" for part in str(spec).split(",") if part.strip()]


def smoke_split(split: RunSplit) -> RunSplit:
    return RunSplit(
        split_id=split.split_id,
        train=split.train[: SMOKE["train_runs"]],
        validation=split.validation[: SMOKE["val_runs"]],
        test=split.test,
        excluded=split.excluded,
    )


def _units(dictionary: pd.DataFrame) -> dict[str, str]:
    return dictionary.set_index("column")["unit"].fillna("").astype(str).to_dict()


def job_settings(cfg: ExperimentConfig, job_params: dict[str, Any], model: str,
                 n_inputs: int) -> tuple[TrainSettings, dict[str, Any], str, str | None]:
    """Effective train settings, model params and target mode of one deep job (training and final test).

    ``train.<field>`` entries in params/grid override the experiment's train settings. ``residual``
    (``auto`` | ``template`` | ``none``) makes the model learn the correction to the recipe template;
    ``auto`` does so for virtual sensors, like the classical models. Channel-independent PatchTST
    runs batch × channels sequences, so its batch is capped to fit GPU memory.
    """

    settings = TrainSettings.from_dict({**cfg.train.to_dict(), **{
        k[len("train."):]: v for k, v in job_params.items() if k.startswith("train.")}})
    residual = str(job_params.get("residual", "auto"))
    if residual == "auto":
        residual = "template" if cfg.task == "virtual_sensor" else "none"
    if residual not in ("template", "none"):
        raise ValueError(f"deep models support residual: auto | template | none, not {residual!r}")
    params = resolve_params({k: v for k, v in job_params.items()
                             if not k.startswith("train.") and k != "residual"}, cfg.sample_seconds)
    cap = None
    if model == "patchtst" and params.get("channel_mode", "independent") == "independent":
        cap = max(32, settings.ci_sequence_budget // n_inputs)
    elif model == "itransformer":
        cap = max(32, settings.attention_budget // (int(params.get("n_heads", 8)) * n_inputs**2))
    note = None
    if cap is not None and cap < max(settings.batch_size, settings.eval_batch_size):
        note = (f"{model}: batch {settings.batch_size} → {min(cap, settings.batch_size)}, "
                f"eval batch → {min(cap * 2, settings.eval_batch_size)} ({n_inputs} channels)")
        settings.batch_size = min(cap, settings.batch_size)
        settings.eval_batch_size = min(cap * 2, settings.eval_batch_size)
    return settings, params, residual, note


def make_window_set(
    bundle: Bundle,
    inputs: list[str],
    target: str,
    scaler: Scaler,
    *,
    context: int,
    horizons: tuple[int, ...],
    region: str,
    device: str,
    chunk: int = 262_144,
) -> WindowSet:
    """Standardize ``inputs`` and ``target`` with ``scaler`` and put them on ``device``."""

    columns = scaler.columns
    index = bundle.column_index(columns)
    x = np.empty((bundle.n_rows, len(inputs)), dtype=np.float32)
    y = np.empty((bundle.n_rows, 1), dtype=np.float32)
    t = columns.index(target)
    for start in range(0, bundle.n_rows, chunk):
        block = scaler.transform(np.asarray(bundle.values[start : start + chunk][:, index]))
        x[start : start + chunk] = block[:, : len(inputs)]
        y[start : start + chunk, 0] = block[:, t]
    ends = window_ends(bundle, context, horizons, region)
    return WindowSet(x, y, ends, context=context, horizons=horizons, device=device)


@dataclass
class TargetData:
    target: str
    inputs: list[str]
    removed: dict[str, list[str]]
    scaler: Scaler
    train: WindowSet
    val: WindowSet
    target_channels: list[int] | None
    template: RecipeTemplate | None = None

    def attach_template(self, train_bundle: Bundle, val_bundle: Bundle) -> RecipeTemplate:
        if self.template is None:
            y_train = self.train.y.cpu().numpy()
            self.template = RecipeTemplate().fit(train_bundle, y_train)
            self.train.template = self.template.values(train_bundle)
            self.val.template = self.template.values(val_bundle)
        return self.template

    def attach_schedule(self, train_bundle: Bundle, val_bundle: Bundle) -> StepSchedule:
        """Recipe template + step plan for the ``schedule`` baseline (future step timing from the plan only)."""

        self.attach_template(train_bundle, val_bundle)
        cap = POSITION_CAP_ROWS // max(1, round(train_bundle.sample_seconds / 0.5))
        plan = StepSchedule(cap).fit(train_bundle)
        for ws, bundle in ((self.train, train_bundle), (self.val, val_bundle)):
            ws.recipe, ws.schedule = self.template, plan
            ws.step_key, ws.step_pos = np.asarray(bundle.step_key), np.asarray(bundle.step_pos)
        return plan


class DataCache:
    """Bundles plus the most recent per-target window sets of one process."""

    def __init__(self, cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, device: str,
                 *, rebuild: bool = False) -> None:
        self.cfg, self.paths, self.device = cfg, paths, device
        self.dictionary, _ = load_column_dictionary()
        columns = usable_columns(self.dictionary)
        self.train_bundle = build_bundle(paths, split, "train", sample_seconds=cfg.sample_seconds, columns=columns,
                                         rebuild=rebuild, bin_align=cfg.bin_align)
        self.val_bundle = build_bundle(paths, split, "validation", sample_seconds=cfg.sample_seconds,
                                       columns=columns, rebuild=rebuild, bin_align=cfg.bin_align)
        self.usable = [c for c in self.train_bundle.columns if c in set(self.val_bundle.columns)]
        # Copies (|r| ≥ COPY_THRESHOLD = 0.9999) measured on train runs only; virtual sensors drop them with the target.
        self.near_copies = find_near_copies(self.train_bundle, self.usable)
        self._cache: OrderedDict[tuple[str, str], TargetData] = OrderedDict()

    def target_data(self, target: str, input_set: str = "default") -> TargetData:
        key = (target, input_set)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        while len(self._cache) >= TARGET_CACHE:
            self._cache.popitem(last=False)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        inputs, removed = resolve_inputs(self.cfg, target, usable=self.usable, dictionary=self.dictionary,
                                         near_copies=self.near_copies, input_set=input_set)
        constant = [c for c in constant_columns(self.train_bundle, inputs) if c != target]
        if constant:
            removed["학습 run에서 변화 없음"] = constant
            inputs = [c for c in inputs if c not in constant]
        union = inputs + ([target] if target not in inputs else [])
        scaler = fit_scaler(self.train_bundle, union)
        shape = {"context": self.cfg.context_steps, "horizons": self.cfg.horizon_steps, "region": self.cfg.region,
                 "device": self.device}
        train = make_window_set(self.train_bundle, inputs, target, scaler, **shape)
        val = make_window_set(self.val_bundle, inputs, target, scaler, **shape)
        channels = [inputs.index(target)] if target in inputs else None
        self._cache[key] = TargetData(target, inputs, removed, scaler, train, val, channels)
        return self._cache[key]


def _prediction_sample(bundle: Bundle, ws: WindowSet, idx: np.ndarray, y_true: np.ndarray,
                       y_pred: np.ndarray, horizons: list[float]) -> pd.DataFrame:
    """All windows of the first validation run (for the dashboard's example plot)."""

    rows = ws.target_rows(idx)
    run = np.asarray(bundle.run_index)[rows[:, 0]]
    keep = np.flatnonzero(run == run[0])[:PREDICTION_SAMPLE_ROWS]
    first_row = int(np.flatnonzero(np.asarray(bundle.run_index) == run[0])[0])
    frame = pd.DataFrame({
        "run_id": bundle.run_ids[int(run[0])],
        "t_s": (ws.ends_np[idx][keep] - first_row) * bundle.sample_seconds,
        "in_cycle": np.asarray(bundle.in_cycle)[ws.ends_np[idx][keep]],
    })
    for i, h in enumerate(horizons):
        frame[f"true_{h:g}s"] = y_true[keep, i, 0]
        frame[f"pred_{h:g}s"] = y_pred[keep, i, 0]
    return frame


def _validation_predictions(bundle: Bundle, ws: WindowSet, idx: np.ndarray, y_true: np.ndarray,
                            y_pred: np.ndarray, horizons: list[float]) -> pd.DataFrame:
    """Every validation window, kept for later studies that combine models without retraining.

    Windows are keyed by run and the row (within that run, at the job's ``sample_seconds``) that ends the
    context, so jobs with other context lengths can be matched on the windows they share.
    """

    ends = ws.ends_np[idx]
    run_index = np.asarray(bundle.run_index)
    run = run_index[ends]
    first = np.searchsorted(run_index, np.arange(len(bundle.run_ids)))
    frame = pd.DataFrame({
        "run_id": pd.Categorical.from_codes(run.astype(np.int32), categories=list(bundle.run_ids)),
        "row": (ends - first[run]).astype(np.int32),
        "in_cycle": np.asarray(bundle.in_cycle)[ends],
    })
    for i, h in enumerate(horizons):
        frame[f"true_{h:g}s"] = y_true[:, i, 0].astype(np.float32)
        frame[f"pred_{h:g}s"] = y_pred[:, i, 0].astype(np.float32)
    return frame


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {k: metrics[k] for k in ("mae", "rmse", "r2", "p95_abs_error", "n_windows")}
    for group in ("by_horizon", "by_region", "by_phase"):
        for key, value in metrics[group].items():
            out[f"mae_{key}"] = value
    return out


def run_job(
    job: Job,
    cfg: ExperimentConfig,
    cache: DataCache,
    exp_dir: Path,
    journal: Journal,
    *,
    baselines: dict[str, dict[str, Any]],
    parent_id: str | None,
    n_jobs: int,
    log_prefix: str = "",
) -> dict[str, Any]:
    job_dir = exp_dir / "jobs" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "train.log"

    def log(message: str) -> None:
        line = f"{log_prefix}[{job.target} · {job.label} · s{job.seed}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    units = _units(cache.dictionary)
    unit = units.get(job.target, "")
    stage = "hpo" if job.grid_size > 1 else "train"
    title = f"[{cfg.name}] {job.target} · {job.label}" + (f" · seed {job.seed}" if len(cfg.seeds) > 1 else "")
    config = {"experiment": cfg.name, "task": cfg.task, "target": job.target, "model": job.model,
              "label": job.label, "params": job.params, "seed": job.seed, "device": cache.device,
              "sample_seconds": cfg.sample_seconds, "context_seconds": cfg.context_seconds,
              "horizons_seconds": cfg.horizon_labels, "region": cfg.region,
              "inputs": cfg.input_sets.get(job.input_set, cfg.inputs), "input_set": job.input_set,
              "train": cfg.train.to_dict() if job.kind == "deep" else None, "smoke": cfg.smoke}
    with journal.record(stage, title, config=config, parent_ids=[parent_id] if parent_id else None,
                        tags=[cfg.name, cfg.task, job.target, job.model]) as entry:
        data = cache.target_data(job.target, job.input_set)
        log(f"inputs {len(data.inputs)}  train windows {len(data.train):,}  val windows {len(data.val):,}")
        val_idx = np.arange(len(data.val))
        training: dict[str, Any] = {}
        n_params = None
        started = time.perf_counter()
        model_config: dict[str, Any] = {}
        settings = cfg.train
        if job.kind == "deep":
            settings, params, residual, note = job_settings(cfg, job.params, job.model, len(data.inputs))
            torch.manual_seed(job.seed)  # initial weights follow the job's seed
            np.random.seed(job.seed)
            model_config = {"name": job.model, "params": params, "n_channels": len(data.inputs),
                            "context": cfg.context_steps, "n_horizons": len(cfg.horizon_steps), "n_targets": 1,
                            "target_channels": data.target_channels, "residual": residual}
            train_ws, val_ws, scale = data.train, data.val, 1.0
            if residual == "template":
                data.attach_template(cache.train_bundle, cache.val_bundle)
                scale = residual_scale(data.train, data.train.template)
                model_config["residual_scale"] = scale
                train_ws = offset_targets(data.train, data.train.template, scale)
                val_ws = offset_targets(data.val, data.val.template, scale)
            model = build_deep_model(job.model, params, n_channels=len(data.inputs), context=cfg.context_steps,
                                     n_horizons=len(cfg.horizon_steps), n_targets=1,
                                     target_channels=data.target_channels)
            n_params = count_parameters(model)
            log(f"parameters {n_params:,}")
            if note:
                log(note)
            if residual == "template":
                log("target: correction to the recipe template")
            if settings.recency_halflife_runs > 0:
                train_ws = with_recency_weights(train_ws, cache.train_bundle, settings.recency_halflife_runs)
            training = train_model(model, train_ws, val_ws, settings, seed=job.seed, log=log)
            fit_seconds = time.perf_counter() - started
            infer_start = time.perf_counter()
            pred = predict(model, val_ws, val_idx, batch_size=settings.eval_batch_size, amp=settings.amp)
            if residual == "template":
                pred = pred * scale + data.val.template[data.val.target_rows(val_idx)]
                data.template.save(job_dir)
            torch.save({"state_dict": model.state_dict(), "model_config": model_config}, job_dir / "model.pt")
            (job_dir / "history.json").write_text(json.dumps(training["history"], indent=1))
        else:
            model = CLASSIC_REGISTRY[job.model](job.params)
            ctx = {"seed": job.seed, "n_jobs": n_jobs, "target_channels": data.target_channels}
            if job.model == "template" or cfg.task == "virtual_sensor" or job.params.get("residual") == "template":
                data.attach_template(cache.train_bundle, cache.val_bundle)
            plan = data.attach_schedule(cache.train_bundle, cache.val_bundle) if job.model == "schedule" else None
            training = model.fit(data.train, data.val, ctx) or {}
            if plan is not None:
                model.info.update(plan.info)
                training = {**training, **plan.info}
            fit_seconds = time.perf_counter() - started
            infer_start = time.perf_counter()
            pred = model.predict(data.val, val_idx, ctx)
            model.save(job_dir)
            if job.model in ("template", "schedule") or getattr(model, "residual", None) == "template":
                data.template.save(job_dir)
            if plan is not None:
                plan.save(job_dir)
        infer_seconds = time.perf_counter() - infer_start
        rows = data.val.target_rows(val_idx)
        t = data.scaler.columns.index(job.target)
        y_true = data.scaler.inverse(data.val.y.cpu().numpy()[rows], [t])
        y_pred = data.scaler.inverse(pred, [t])
        metrics = regression_metrics(y_true, y_pred, rows=rows, bundle=cache.val_bundle,
                                     horizons_seconds=cfg.horizon_labels)
        baseline = baselines.get(job.target)
        is_baseline = job.model == cfg.baseline
        strengths, weaknesses = describe(
            metrics, unit=unit, baseline=None if is_baseline else baseline,
            baseline_label=BASELINE_LABELS.get(cfg.baseline, "기준선"),
            training=training if job.kind == "deep" else None, patience=settings.patience,
        )
        gain = None if is_baseline or baseline is None else improvement(metrics["mae"], baseline["mae"])
        train_summary = {k: v for k, v in training.items() if k != "history"}
        result = {
            "job_id": job.job_id, "target": job.target, "model": job.model, "kind": job.kind,
            "label": job.label, "variant": job.variant, "grid_size": job.grid_size, "seed": job.seed,
            "status": "ok", "unit": unit, "n_inputs": len(data.inputs), "n_params": n_params,
            "fit_seconds": round(fit_seconds, 2), "infer_seconds": round(infer_seconds, 2),
            "improvement_pct": gain, "is_baseline": is_baseline, **_flat_metrics(metrics),
            "epochs": training.get("epochs"), "best_epoch": training.get("best_epoch"),
            "train_status": training.get("status"), "windows_per_second": training.get("windows_per_second"),
            "peak_gpu_mb": training.get("peak_gpu_mb"), "job_dir": f"jobs/{job.job_id}",
        }
        (job_dir / "job.json").write_text(json.dumps({
            "job": job.to_dict(), "experiment": cfg.name, "task": cfg.task, "inputs": data.inputs,
            "removed_inputs": data.removed, "scaler": data.scaler.to_dict(), "model_config": model_config,
            "sample_seconds": cfg.sample_seconds, "bin_align": cfg.bin_align, "context_steps": cfg.context_steps,
            "horizon_steps": list(cfg.horizon_steps), "horizons_seconds": cfg.horizon_labels,
            "region": cfg.region, "device": cache.device,
        }, ensure_ascii=False, indent=1, default=str))
        (job_dir / "metrics_validation.json").write_text(json.dumps(
            {"metrics": metrics, "training": train_summary, "result": result, "strengths": strengths,
             "weaknesses": weaknesses}, ensure_ascii=False, indent=1, default=str))
        _prediction_sample(cache.val_bundle, data.val, val_idx, y_true, y_pred, cfg.horizon_labels).to_parquet(
            job_dir / "predictions_sample.parquet", index=False)
        if SAVE_VALIDATION_PREDICTIONS:
            _validation_predictions(cache.val_bundle, data.val, val_idx, y_true, y_pred, cfg.horizon_labels).to_parquet(
                job_dir / "predictions_val.parquet", index=False)

        speed = ""
        if training.get("windows_per_second"):
            speed = f", 초당 {training['windows_per_second']:,.0f} window"
        entry.summary = (
            f"{job.target} 검증 MAE {metrics['mae']:.4g} {unit}".rstrip()
            + (f" (기준선 대비 {gain:+.1f} %)" if gain is not None else "")
            + f" · 학습 {fit_seconds / 60:.1f}분{speed} · 입력 {len(data.inputs)}개"
        )
        entry.metrics = {**_flat_metrics(metrics), "improvement_pct": gain, "fit_seconds": round(fit_seconds, 2),
                         "infer_seconds": round(infer_seconds, 2), "n_params": n_params, **train_summary}
        entry.artifacts = {"job_dir": str(job_dir)}
        entry.strengths, entry.weaknesses = strengths, weaknesses
        log(entry.summary)
    return result


def _worker(device: str, cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, exp_dir: Path,
            baselines: dict[str, dict[str, Any]], parent_id: str | None, n_jobs: int, jobs: Any,
            results: Any) -> None:
    torch.set_num_threads(max(1, n_jobs))
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    journal = Journal(paths.journal_dir)
    cache = DataCache(cfg, paths, split, device)
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            results.put(run_job(job, cfg, cache, exp_dir, journal, baselines=baselines, parent_id=parent_id,
                                n_jobs=n_jobs, log_prefix=f"[{device}] "))
        except Exception as exc:  # recorded as failed in the journal by Journal.record
            traceback.print_exc()
            results.put({"job_id": job.job_id, "target": job.target, "model": job.model, "label": job.label,
                         "seed": job.seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def _run_parallel(jobs: list[Job], devices: list[str], cfg: ExperimentConfig, paths: LabPaths, split: RunSplit,
                  exp_dir: Path, baselines: dict[str, dict[str, Any]], parent_id: str | None) -> list[dict[str, Any]]:
    ctx = torch.multiprocessing.get_context("spawn")
    job_queue, result_queue = ctx.Queue(), ctx.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in devices:
        job_queue.put(None)
    n_jobs = max(1, (os.cpu_count() or 1) // len(devices))
    workers = [ctx.Process(target=_worker, args=(d, cfg, paths, split, exp_dir, baselines, parent_id, n_jobs,
                                                 job_queue, result_queue)) for d in devices]
    for w in workers:
        w.start()
    results = []
    while len(results) < len(jobs):
        try:
            results.append(result_queue.get(timeout=30))
        except Exception:
            if not any(w.is_alive() for w in workers):
                break
    for w in workers:
        w.join()
    done = {r["job_id"] for r in results}
    for job in jobs:
        if job.job_id not in done:
            results.append({"job_id": job.job_id, "target": job.target, "model": job.model, "label": job.label,
                            "seed": job.seed, "status": "failed", "error": "worker exited before finishing"})
    return results


def _prepare_entry(cfg: ExperimentConfig, cache: DataCache, split: RunSplit, journal: Journal,
                   exp_dir: Path) -> str:
    title = f"[{cfg.name}] 학습 데이터 준비"
    config = {"experiment": cfg.name, "task": cfg.task, "split": split.split_id,
              "sample_seconds": cfg.sample_seconds, "bin_align": cfg.bin_align, "region": cfg.region, "inputs": cfg.inputs,
              "context_seconds": cfg.context_seconds, "horizons_seconds": cfg.horizon_labels, "smoke": cfg.smoke}
    with journal.record("preprocess", title, config=config, tags=[cfg.name, cfg.task]) as entry:
        targets = {}
        n_train = len(window_ends(cache.train_bundle, cfg.context_steps, cfg.horizon_steps, cfg.region))
        n_val = len(window_ends(cache.val_bundle, cfg.context_steps, cfg.horizon_steps, cfg.region))
        for target in cfg.targets:
            for input_set in cfg.input_sets:
                inputs, removed = resolve_inputs(cfg, target, usable=cache.usable, dictionary=cache.dictionary,
                                                 near_copies=cache.near_copies, input_set=input_set)
                key = target if len(cfg.input_sets) == 1 else f"{target}@{input_set}"
                targets[key] = {"inputs": len(inputs), "removed": removed}
        tb, vb = cache.train_bundle.stats, cache.val_bundle.stats
        entry.metrics = {
            "train_runs": tb.get("runs"), "validation_runs": vb.get("runs"), "test_runs_sealed": len(split.test),
            "train_rows": tb.get("rows"), "validation_rows": vb.get("rows"), "usable_columns": len(cache.usable),
            "train_segments": tb.get("segments"), "nan_filled": (tb.get("nan_filled") or 0) + (vb.get("nan_filled") or 0),
            "train_windows": n_train, "validation_windows": n_val, "targets": targets,
        }
        entry.summary = (
            f"run {tb.get('runs')}/{vb.get('runs')} (학습/검증, 시험 {len(split.test)}개 봉인) · "
            f"{cfg.sample_seconds:g}초 간격 {tb.get('rows', 0) + vb.get('rows', 0):,}행 · 신호 {len(cache.usable)}개 · "
            f"window {cfg.context_seconds:g}초"
            + (f" → {', '.join(f'{h:g}' for h in cfg.horizon_labels)}초 뒤" if cfg.task == "forecast" else " → 현재값")
        )
        entry.strengths = [
            "run 단위 분할: 같은 run이 학습과 검증에 함께 들어가지 않음",
            "정규화 통계는 학습 run에서만 계산",
            "시간 끊김·역행에서 구간을 나눠 window가 끊김을 넘지 않음",
        ]
        if cfg.sample_seconds > 0.5:
            entry.weaknesses.append(f"0.5초 원본을 {cfg.sample_seconds:g}초 평균으로 줄여 짧은 스파이크가 완화됨")
        (exp_dir / "prep.json").write_text(json.dumps({"bundles": {"train": tb, "validation": vb},
                                                       "targets": targets}, ensure_ascii=False, indent=1,
                                                      default=str))
    return entry.entry_id


def run_experiment(
    config_path: str | Path,
    paths: LabPaths,
    *,
    devices: list[str],
    smoke: bool = False,
    only_models: list[str] | None = None,
    only_targets: list[str] | None = None,
    dry_run: bool = False,
    rebuild: bool = False,
) -> Path | None:
    cfg = load_experiment(config_path)
    if smoke:
        cfg = apply_smoke(cfg)
    jobs = expand_jobs(cfg, only_models=only_models, only_targets=only_targets)
    print(f"experiment {cfg.name}: task {cfg.task}, {len(jobs)} jobs on {', '.join(devices)}")
    for job in jobs:
        print(f"  {job.job_id:60s} {job.label}")
    if dry_run:
        return None
    exp_dir = paths.work_root / "experiments" / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, exp_dir / "experiment.yaml")
    (exp_dir / "jobs.json").write_text(json.dumps([j.to_dict() for j in jobs], ensure_ascii=False, indent=1,
                                                  default=str))
    journal = Journal(paths.journal_dir)
    split = load_split(cfg.split_path())
    if cfg.smoke:
        split = smoke_split(split)
    started = time.perf_counter()
    if cfg.task == "thickness":
        from .vm import run_thickness

        results = run_thickness(cfg, paths, split, jobs, exp_dir, journal)
        summarize(cfg, results, exp_dir, journal, wall_seconds=time.perf_counter() - started)
        return exp_dir

    main_device = devices[0]
    cache = DataCache(cfg, paths, split, main_device, rebuild=rebuild)
    parent_id = _prepare_entry(cfg, cache, split, journal, exp_dir)
    n_cpu = os.cpu_count() or 1
    baselines: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    first, rest = [j for j in jobs if j.model == cfg.baseline], [j for j in jobs if j.model != cfg.baseline]
    for job in first:
        result = run_job(job, cfg, cache, exp_dir, journal, baselines=baselines, parent_id=parent_id, n_jobs=n_cpu)
        results.append(result)
        metrics = json.loads((exp_dir / result["job_dir"] / "metrics_validation.json").read_text())["metrics"]
        baselines[job.target] = metrics
    if len(devices) == 1 or len(rest) <= 1:
        for job in rest:
            try:
                results.append(run_job(job, cfg, cache, exp_dir, journal, baselines=baselines,
                                       parent_id=parent_id, n_jobs=n_cpu))
            except Exception as exc:
                traceback.print_exc()
                results.append({"job_id": job.job_id, "target": job.target, "model": job.model,
                                "label": job.label, "seed": job.seed, "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"})
    else:
        del cache
        if main_device.startswith("cuda"):
            torch.cuda.empty_cache()
        results += _run_parallel(rest, devices, cfg, paths, split, exp_dir, baselines, parent_id)
    summarize(cfg, results, exp_dir, journal, wall_seconds=time.perf_counter() - started)
    return exp_dir


def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("-")
            elif isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(cfg: ExperimentConfig, results: list[dict[str, Any]], exp_dir: Path, journal: Journal,
              *, wall_seconds: float) -> pd.DataFrame:
    """Aggregate seeds, rank per target on validation, write summary files and a journal entry."""

    frame = pd.DataFrame(results)
    frame.to_csv(exp_dir / "results.csv", index=False)
    ok = frame[frame["status"] == "ok"]
    rows = []
    for (target, label), group in ok.groupby(["target", "label"], sort=False):
        first = group.iloc[0]
        rows.append({
            "target": target, "label": label, "model": first["model"], "is_baseline": bool(first.get("is_baseline")),
            "grid_size": int(first.get("grid_size") or 1), "seeds": len(group),
            "mae": float(group["mae"].mean()), "mae_std": float(group["mae"].std(ddof=0)) if len(group) > 1 else None,
            "improvement_pct": float(group["improvement_pct"].mean()) if group["improvement_pct"].notna().any() else None,
            "fit_minutes": float(group["fit_seconds"].mean() / 60),
            "windows_per_second": float(group["windows_per_second"].mean())
            if "windows_per_second" in group and group["windows_per_second"].notna().any() else None,
            "unit": first.get("unit", ""),
            **{k: float(group[k].mean()) for k in group.columns if k.startswith("mae_") and group[k].notna().any()},
        })
    table = pd.DataFrame(rows)
    selected: dict[str, Any] = {}
    grid_choices: list[dict[str, Any]] = []
    if len(table):
        table = table.sort_values(["target", "mae"]).reset_index(drop=True)
        table["rank"] = table.groupby("target").cumcount() + 1
        for target, group in table.groupby("target", sort=False):
            candidates = group[~group["is_baseline"]]
            base = group[group["is_baseline"]]
            if len(candidates):
                best = candidates.iloc[0]
                selected[target] = {"label": best["label"], "model": best["model"], "mae": best["mae"],
                                    "improvement_pct": best["improvement_pct"],
                                    "baseline_mae": float(base["mae"].iloc[0]) if len(base) else None,
                                    "unit": best["unit"]}
            for model, mgroup in candidates.groupby("model"):
                if int(mgroup["grid_size"].iloc[0]) > 1:
                    grid_choices.append({"target": target, "model": model, "chosen": mgroup.iloc[0]["label"],
                                         "mae": mgroup.iloc[0]["mae"], "tried": len(mgroup)})
    table.to_csv(exp_dir / "summary.csv", index=False)
    failed = frame[frame["status"] != "ok"]
    summary = {"experiment": cfg.name, "task": cfg.task, "smoke": cfg.smoke, "baseline": cfg.baseline,
               "wall_minutes": round(wall_seconds / 60, 2), "jobs": len(frame), "failed": len(failed),
               "selected": selected, "grid_choices": grid_choices, "exp_dir": str(exp_dir)}
    (exp_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))

    base_label = BASELINE_LABELS.get(cfg.baseline, "기준선")
    lines = [f"# {cfg.name} — 검증 결과", "", cfg.description, "",
             f"- 과제: {cfg.task} · 기준선: {base_label} · 작업 {len(frame)}개 (실패 {len(failed)}) · "
             f"총 {wall_seconds / 60:.1f}분", "- 선택은 검증 run 기준이며 시험 run은 열지 않았다.", ""]
    for target, info in selected.items():
        gain = info["improvement_pct"]
        base_mae = "-" if info["baseline_mae"] is None else f"{info['baseline_mae']:.4g}"
        lines.append(f"- **{target}**: {info['label']} — MAE {info['mae']:.4g} {info['unit']} "
                     f"(기준선 {base_mae}" + (f", {gain:+.1f} %" if gain is not None else "") + ")")
    if grid_choices:
        lines += ["", "## 하이퍼파라미터 선택", ""]
        lines += [f"- {g['target']} · {g['model']}: {g['tried']}개 중 {g['chosen']} (MAE {g['mae']:.4g})"
                  for g in grid_choices]
    if len(table):
        show = [c for c in ["target", "rank", "label", "seeds", "mae", "mae_std", "improvement_pct",
                            "mae_cycle", "mae_other", "fit_minutes", "windows_per_second"] if c in table]
        lines += ["", "## 전체 비교 (검증)", "", _md_table(table[show])]
    if len(failed):
        lines += ["", "## 실패한 작업", ""]
        lines += [f"- {r.label} ({r.target}, seed {r.seed}): {r.error}" for r in failed.itertuples()]
    (exp_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with journal.record("evaluate", f"[{cfg.name}] 모델 비교 (검증)", config={"experiment": cfg.name,
                        "task": cfg.task, "smoke": cfg.smoke}, tags=[cfg.name, cfg.task]) as entry:
        entry.summary = " · ".join(
            f"{t}: {i['label']} {i['improvement_pct']:+.1f} %" if i["improvement_pct"] is not None
            else f"{t}: {i['label']}" for t, i in selected.items()) or "완료된 작업 없음"
        entry.metrics = {"selected": selected, "grid_choices": grid_choices, "jobs": len(frame),
                         "failed": len(failed), "wall_minutes": round(wall_seconds / 60, 2)}
        entry.artifacts = {"summary": str(exp_dir / "summary.md"), "exp_dir": str(exp_dir)}
        for target, info in selected.items():
            gain = info["improvement_pct"]
            if gain is not None and gain > 0:
                entry.strengths.append(f"{target}: {info['label']}가 {base_label} 대비 오차 {gain:.1f} % 감소")
            else:
                entry.weaknesses.append(f"{target}: 어떤 모델도 {base_label}보다 낫지 않음")
        for r in failed.itertuples():
            entry.weaknesses.append(f"실패: {r.label} ({r.target}) — {r.error}")
    print((exp_dir / "summary.md").read_text(encoding="utf-8"))
    return table


def load_predictor(job_dir: Path, device: str) -> tuple[dict[str, Any], Any]:
    """Rebuild a trained job: returns ``(job_doc, model)``."""

    doc = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    name = doc["job"]["model"]
    if doc["job"]["kind"] == "deep":
        state = torch.load(job_dir / "model.pt", map_location=device, weights_only=False)
        mc = state["model_config"]
        model = build_deep_model(mc["name"], mc["params"], n_channels=mc["n_channels"], context=mc["context"],
                                 n_horizons=mc["n_horizons"], n_targets=mc["n_targets"],
                                 target_channels=mc["target_channels"])
        model.load_state_dict(state["state_dict"])
        return doc, model.to(device)
    model = CLASSIC_REGISTRY[name](doc["job"]["params"])
    model.load_state(job_dir)
    return doc, model


def window_set_for_job(doc: dict[str, Any], bundle: Bundle, device: str) -> WindowSet:
    """The job's windows (same inputs, train-fitted scaler, context, horizons, region) on new runs."""

    return make_window_set(bundle, doc["inputs"], doc["job"]["target"], Scaler.from_dict(doc["scaler"]),
                           context=int(doc["context_steps"]), horizons=tuple(int(h) for h in doc["horizon_steps"]),
                           region=doc["region"], device=device)


def predict_job(job_dir: Path, doc: dict[str, Any], model: Any, ws: WindowSet, bundle: Bundle,
                cfg: ExperimentConfig) -> np.ndarray:
    """Standardized predictions ``(n, H, 1)`` of a trained job for every window of ``ws``.

    The same path as validation: deep models add their scaled correction to the recipe template,
    classical models get the template or the last value as their base.
    """

    idx = np.arange(len(ws))
    target = doc["job"]["target"]
    if doc["job"]["kind"] == "deep":
        settings, _, residual, _ = job_settings(cfg, doc["job"]["params"], doc["job"]["model"], len(doc["inputs"]))
        pred = predict(model, ws, idx, batch_size=settings.eval_batch_size, amp=settings.amp)
        if residual == "template":
            template = RecipeTemplate.load(job_dir).values(bundle)
            pred = pred * float(doc["model_config"].get("residual_scale", 1.0)) + template[ws.target_rows(idx)]
        return pred
    ws.template = None
    if doc["job"]["model"] == "template" or getattr(model, "residual", None) == "template":
        ws.template = RecipeTemplate.load(job_dir).values(bundle)
    if doc["job"]["model"] == "schedule":
        ws.recipe, ws.schedule = RecipeTemplate.load(job_dir), StepSchedule.load(job_dir)
        ws.step_key, ws.step_pos = np.asarray(bundle.step_key), np.asarray(bundle.step_pos)
    channels = [doc["inputs"].index(target)] if target in doc["inputs"] else None
    return model.predict(ws, idx, {"target_channels": channels})


def run_final_test(exp_dir: Path, paths: LabPaths, *, device: str, scope: str = "per_model") -> pd.DataFrame:
    """Score the validation-selected candidates on the sealed test runs (once, at the end)."""

    exp_dir = Path(exp_dir)
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = load_experiment(exp_dir / "experiment.yaml")
    if summary.get("smoke"):
        cfg = apply_smoke(cfg)
    split = load_split(cfg.split_path())
    journal = Journal(paths.journal_dir)
    table = pd.read_csv(exp_dir / "summary.csv")
    results = pd.read_csv(exp_dir / "results.csv")
    results = results[results["status"] == "ok"]
    picks = []
    for target, group in table.groupby("target"):
        group = group.sort_values("mae")
        if scope == "best":
            chosen = pd.concat([group[group["is_baseline"]], group[~group["is_baseline"]].head(1)])
        else:
            chosen = group.groupby("model", sort=False).head(1)
        picks += [(target, label) for label in chosen["label"]]
    if cfg.task == "thickness":
        from .vm import final_test_thickness

        return final_test_thickness(cfg, paths, split, exp_dir, picks, journal)

    dictionary, _ = load_column_dictionary()
    units = _units(dictionary)
    test_bundle = build_bundle(paths, split, "test", sample_seconds=cfg.sample_seconds,
                               columns=usable_columns(dictionary), include_test=True, bin_align=cfg.bin_align)
    rows_out = []
    # One test window set per (target, inputs): every job of that pair used the same train-fitted scaler.
    window_sets: dict[tuple[str, tuple[str, ...]], WindowSet] = {}
    for target, label in picks:
        for job in results[(results["target"] == target) & (results["label"] == label)].itertuples():
            job_dir = exp_dir / "jobs" / job.job_id
            doc, model = load_predictor(job_dir, device)
            scaler = Scaler.from_dict(doc["scaler"])
            key = (target, tuple(doc["inputs"]))
            if key not in window_sets:
                window_sets.clear()  # jobs come grouped by target; keep one set on the device
                window_sets[key] = window_set_for_job(doc, test_bundle, device)
            ws = window_sets[key]
            pred = predict_job(job_dir, doc, model, ws, test_bundle, cfg)
            rows = ws.target_rows(np.arange(len(ws)))
            t = scaler.columns.index(target)
            y_true = scaler.inverse(ws.y.cpu().numpy()[rows], [t])
            metrics = regression_metrics(y_true, scaler.inverse(pred, [t]), rows=rows, bundle=test_bundle,
                                         horizons_seconds=cfg.horizon_labels)
            (job_dir / "metrics_test.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
            rows_out.append({"target": target, "label": label, "model": job.model, "seed": job.seed,
                             "is_baseline": job.model == cfg.baseline, "unit": units.get(target, ""),
                             **_flat_metrics(metrics)})
    test = pd.DataFrame(rows_out)
    horizon_cols = [f"mae_{h:g}s" for h in cfg.horizon_labels if f"mae_{h:g}s" in test and h > 0]
    agg = test.groupby(["target", "label", "model", "is_baseline", "unit"], sort=False).agg(
        mae=("mae", "mean"), mae_std=("mae", "std"), seeds=("mae", "size"),
        mae_cycle=("mae_cycle", "mean"), mae_other=("mae_other", "mean"),
        **{c: (c, "mean") for c in horizon_cols}).reset_index()
    agg["improvement_pct"] = np.nan
    for target, group in agg.groupby("target"):
        base = group[group["is_baseline"]]
        if len(base):
            agg.loc[group.index, "improvement_pct"] = (1 - group["mae"] / float(base["mae"].iloc[0])) * 100
    agg = agg.sort_values(["target", "mae"])
    agg.to_csv(exp_dir / "test_summary.csv", index=False)
    lines = [f"# {cfg.name} — 시험 run 결과 (최종)", "",
             f"- 시험 run {len(split.test)}개. 후보는 검증 결과로 미리 고른 것({scope})이며 시험 결과로 다시 고르지 않는다.", "",
             _md_table(agg[["target", "label", "seeds", "mae", "mae_std", "improvement_pct", *horizon_cols,
                            "mae_cycle", "mae_other"]])]
    (exp_dir / "test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with journal.record("evaluate", f"[{cfg.name}] 시험 run 최종 평가", config={"experiment": cfg.name,
                        "scope": scope}, tags=[cfg.name, cfg.task, "test"]) as entry:
        best = agg[~agg["is_baseline"]].groupby("target").head(1)
        entry.summary = " · ".join(f"{r.target}: {r.label} MAE {r.mae:.4g} ({r.improvement_pct:+.1f} %)"
                                   for r in best.itertuples())
        entry.metrics = {"rows": agg.to_dict(orient="records")}
        entry.artifacts = {"test_summary": str(exp_dir / "test_summary.md")}
    print("\n".join(lines))
    return agg
