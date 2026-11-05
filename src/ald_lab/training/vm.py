"""Run-level thickness prediction (virtual metrology, G2).

Each run becomes one row: per input signal, the mean and standard deviation
over ALD-cycle rows and the mean over the whole process window. Targets come
from the ex-situ workbook (49-point layout, recomputed from raw points by
``ald-build-overview``). With about 70 runs the models are small and are
scored by leave-one-run-out cross-validation over train + validation runs;
test runs stay sealed for ``ald-final-test``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.columns import load_column_dictionary
from ..data.splits import RunSplit
from ..journal import Journal
from ..models.defaults import VM_LIGHTGBM, VM_PLS
from ..paths import LabPaths
from .config import BASELINE_LABELS, ExperimentConfig, Job, resolve_inputs
from .prepare import Bundle, build_bundle, usable_columns

DEFAULT_EXCLUDED_RUNS = ("standard_trace_001", "standard_trace_002", "standard_trace_055", "standard_trace_057")
TARGET_UNITS = {"thk_mean": "Å", "uf_pct_mean": "%", "wtow_pct": "%", "thk_center": "Å", "thk_edge": "Å",
                "edge_vs_center_pct": "%"}


class VMError(ValueError):
    """Raised when run-level targets or features are missing."""


def load_targets(paths: LabPaths, targets: list[str]) -> pd.DataFrame:
    path = paths.overview_dir / "exsitu_blocks.parquet"
    if not path.is_file():
        raise VMError("ex-situ table missing: run ald-build-overview --exsitu <workbook.xlsx> first")
    blocks = pd.read_parquet(path)
    linked = blocks.dropna(subset=["run_id"])
    linked = linked[linked["layout"] == "49pt"]
    missing = [t for t in targets if t not in linked.columns]
    if missing:
        raise VMError(f"unknown thickness targets {missing}")
    return linked.drop_duplicates("run_id").set_index("run_id")[targets]


def run_features(bundle: Bundle, inputs: list[str]) -> pd.DataFrame:
    index = bundle.column_index(inputs)
    run_index = np.asarray(bundle.run_index)
    in_cycle = np.asarray(bundle.in_cycle)
    in_process = np.asarray(bundle.in_process)
    rows = []
    for r, run_id in enumerate(bundle.run_ids):
        mask = run_index == r
        values = np.asarray(bundle.values[np.flatnonzero(mask)][:, index], dtype=np.float64)
        cyc = values[in_cycle[mask]]
        proc = values[in_process[mask]]
        feat = {"run_id": run_id, "cycle_minutes": len(cyc) * bundle.sample_seconds / 60}
        for j, name in enumerate(inputs):
            feat[f"{name}__cycle_mean"] = cyc[:, j].mean() if len(cyc) else np.nan
            feat[f"{name}__cycle_std"] = cyc[:, j].std() if len(cyc) else np.nan
            feat[f"{name}__process_mean"] = proc[:, j].mean() if len(proc) else np.nan
        rows.append(feat)
    return pd.DataFrame(rows).set_index("run_id")


def _estimator(model: str, params: dict[str, Any], seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "ridge":
        from sklearn.linear_model import RidgeCV

        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 29)))
    if model == "pls":
        from sklearn.cross_decomposition import PLSRegression

        n_components = int(params.get("n_components", VM_PLS["n_components"]))
        return make_pipeline(StandardScaler(), PLSRegression(n_components=n_components))
    if model == "lightgbm":
        try:
            import lightgbm as lgb

            p = {**VM_LIGHTGBM, **params}
            return lgb.LGBMRegressor(n_estimators=int(p["n_estimators"]),
                                     learning_rate=float(p["learning_rate"]),
                                     num_leaves=int(p["num_leaves"]),
                                     min_child_samples=int(p["min_child_samples"]),
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                     random_state=seed, verbose=-1, n_jobs=1)  # ~60 rows: threads only add overhead
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=7,
                                                 min_samples_leaf=5, random_state=seed)
    raise VMError(f"unknown thickness model {model}")


def _fit_predict(model: str, params: dict[str, Any], seed: int, x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray) -> np.ndarray:
    if model == "mean":
        return np.full(len(x_test), float(np.mean(y_train)))
    est = _estimator(model, params, seed)
    x_train = np.nan_to_num(x_train)
    est.fit(x_train, y_train)
    return np.asarray(est.predict(np.nan_to_num(x_test))).reshape(-1)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    err = np.abs(pred - y)
    sst = float(((y - y.mean()) ** 2).sum())
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "r2": float(1 - ((pred - y) ** 2).sum() / sst) if sst > 0 else None,
            "max_abs_error": float(err.max()), "target_std": float(y.std()), "n_runs": int(len(y))}


def _dataset(cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, partitions: tuple[str, ...],
             include_test: bool = False, input_set: str = "default",
             ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]]:
    dictionary, _ = load_column_dictionary()
    bundles = [build_bundle(paths, split, part, sample_seconds=cfg.sample_seconds, columns=usable_columns(dictionary),
                            include_test=include_test) for part in partitions]
    usable = [c for c in bundles[0].columns if all(c in b.columns for b in bundles)]
    inputs, removed = resolve_inputs(cfg, cfg.targets[0], usable=usable, dictionary=dictionary, input_set=input_set)
    features = [run_features(bundle, inputs) for bundle in bundles]
    x = pd.concat(features)
    targets = load_targets(paths, cfg.targets)
    excluded = [str(r) for r in cfg.vm.get("exclude_runs", DEFAULT_EXCLUDED_RUNS)]
    runs = [r for r in x.index if r in targets.index and r not in excluded]
    x = x.loc[runs]
    if "test" not in partitions:  # features that never vary across the fitting runs carry nothing
        x = x.loc[:, x.std(ddof=0).fillna(0) > 0]
    return x, targets.loc[runs], inputs, removed


def run_thickness(cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, jobs: list[Job], exp_dir: Path,
                  journal: Journal) -> list[dict[str, Any]]:
    with journal.record("preprocess", f"[{cfg.name}] run 단위 특성 만들기",
                        config={"experiment": cfg.name, "task": cfg.task, "inputs": cfg.inputs,
                                "sample_seconds": cfg.sample_seconds}, tags=[cfg.name, cfg.task]) as prep:
        datasets = {name: _dataset(cfg, paths, split, ("train", "validation"), input_set=name)
                    for name in cfg.input_sets}
        x, y, inputs, removed = next(iter(datasets.values()))
        excluded = [str(r) for r in cfg.vm.get("exclude_runs", DEFAULT_EXCLUDED_RUNS)]
        prep.summary = (f"학습+검증 run {len(x)}개 (이상 run {len(excluded)}개 제외, 시험 {len(split.test)}개 봉인) · "
                        f"입력 신호 {len(inputs)}개 → run 특성 {x.shape[1]}개")
        prep.metrics = {"runs": len(x), "features": int(x.shape[1]), "inputs": len(inputs),
                        "excluded_runs": excluded, "removed_inputs": removed,
                        "target_std": {t: float(y[t].std()) for t in cfg.targets}}
        prep.strengths = ["run 하나 = 표본 하나: 같은 run이 학습과 평가에 동시에 들어가지 않음 (leave-one-run-out)"]
        prep.weaknesses = [f"표본이 {len(x)}개뿐이라 특성 {x.shape[1]}개 대비 매우 적음 — 단순 모델만 의미 있음"]
    parent = prep.entry_id
    results = []
    baselines: dict[str, dict[str, Any]] = {}
    for job in jobs:
        x, y, inputs, _ = datasets[job.input_set]
        runs = list(x.index)
        job_dir = exp_dir / "jobs" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        unit = TARGET_UNITS.get(job.target, "")
        stage = "hpo" if job.grid_size > 1 else "train"
        title = f"[{cfg.name}] {job.target} · {job.label}"
        try:
            with journal.record(stage, title, config={"experiment": cfg.name, "task": cfg.task,
                                "target": job.target, "model": job.model, "params": job.params, "seed": job.seed},
                                parent_ids=[parent], tags=[cfg.name, cfg.task, job.target, job.model]) as entry:
                started = time.perf_counter()
                target = y[job.target].to_numpy(np.float64)
                values = x.to_numpy(np.float64)
                pred = np.empty(len(runs))
                for i in range(len(runs)):
                    keep = np.arange(len(runs)) != i
                    pred[i] = _fit_predict(job.model, job.params, job.seed, values[keep], target[keep], values[i:i + 1])[0]
                fit_seconds = time.perf_counter() - started
                metrics = _metrics(target, pred)
                is_baseline = job.model == cfg.baseline
                if is_baseline:
                    baselines[job.target] = metrics
                base = baselines.get(job.target)
                gain = None if is_baseline or base is None else (1 - metrics["mae"] / base["mae"]) * 100
                worst = runs[int(np.argmax(np.abs(pred - target)))]
                strengths, weaknesses = [], []
                if gain is not None:
                    line = f"학습 평균 예측 대비 MAE {base['mae']:.3g} → {metrics['mae']:.3g} {unit}"
                    (strengths if gain >= 5 else weaknesses).append(f"{line} ({gain:+.1f} %)")
                if metrics["r2"] is not None and not is_baseline:
                    (strengths if metrics["r2"] > 0.3 else weaknesses).append(
                        f"run 간 변동 설명력 R² {metrics['r2']:.2f} (run 간 표준편차 {metrics['target_std']:.3g} {unit})")
                weaknesses.append(f"가장 크게 틀린 run: {worst} (오차 {np.abs(pred - target).max():.3g} {unit})")
                pd.DataFrame({"run_id": runs, "true": target, "pred": pred}).to_parquet(
                    job_dir / "predictions_cv.parquet", index=False)
                (job_dir / "job.json").write_text(json.dumps({"job": job.to_dict(), "experiment": cfg.name,
                                                              "task": cfg.task, "inputs": inputs,
                                                              "features": list(x.columns)},
                                                             ensure_ascii=False, indent=1, default=str))
                result = {"job_id": job.job_id, "target": job.target, "model": job.model, "kind": "vm",
                          "label": job.label, "variant": job.variant, "grid_size": job.grid_size,
                          "seed": job.seed, "status": "ok", "unit": unit, "n_inputs": len(inputs),
                          "fit_seconds": round(fit_seconds, 3), "improvement_pct": gain,
                          "is_baseline": is_baseline, **metrics, "job_dir": f"jobs/{job.job_id}"}
                (job_dir / "metrics_validation.json").write_text(json.dumps(
                    {"metrics": metrics, "result": result, "strengths": strengths, "weaknesses": weaknesses},
                    ensure_ascii=False, indent=1, default=str))
                entry.summary = (f"{job.target} leave-one-run-out MAE {metrics['mae']:.3g} {unit}"
                                 + (f" (학습 평균 대비 {gain:+.1f} %)" if gain is not None else "")
                                 + f" · {len(runs)} run · {fit_seconds:.1f}초")
                entry.metrics = {**metrics, "improvement_pct": gain, "fit_seconds": round(fit_seconds, 3)}
                entry.strengths, entry.weaknesses = strengths, weaknesses
                print(entry.summary, flush=True)
            results.append(result)
        except Exception as exc:
            results.append({"job_id": job.job_id, "target": job.target, "model": job.model, "label": job.label,
                            "seed": job.seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return results


def final_test_thickness(cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, exp_dir: Path,
                         picks: list[tuple[str, str]], journal: Journal) -> pd.DataFrame:
    sets = {}
    for name in cfg.input_sets:
        x_fit, y_fit, _, _ = _dataset(cfg, paths, split, ("train", "validation"), input_set=name)
        x_test, y_test, _, _ = _dataset(cfg, paths, split, ("test",), include_test=True, input_set=name)
        sets[name] = (x_fit, y_fit, x_test.reindex(columns=x_fit.columns), y_test)
    results = pd.read_csv(exp_dir / "results.csv")
    rows = []
    for target, label in picks:
        for job in results[(results["target"] == target) & (results["label"] == label)].itertuples():
            spec = json.loads((exp_dir / "jobs" / job.job_id / "job.json").read_text())["job"]
            params = spec["params"]
            x_fit, y_fit, x_test, y_test = sets[spec.get("input_set", "default")]
            pred = _fit_predict(job.model, params, int(job.seed), x_fit.to_numpy(), y_fit[target].to_numpy(),
                                x_test.to_numpy())
            rows.append({"target": target, "label": label, "model": job.model, "seed": job.seed,
                         "is_baseline": job.model == cfg.baseline, "unit": TARGET_UNITS.get(target, ""),
                         **_metrics(y_test[target].to_numpy(np.float64), pred)})
    test = pd.DataFrame(rows)
    agg = test.groupby(["target", "label", "model", "is_baseline", "unit"], sort=False)["mae"].mean().reset_index()
    agg["improvement_pct"] = np.nan
    for target, group in agg.groupby("target"):
        base = group[group["is_baseline"]]
        if len(base):
            agg.loc[group.index, "improvement_pct"] = (1 - group["mae"] / float(base["mae"].iloc[0])) * 100
    agg = agg.sort_values(["target", "mae"])
    agg.to_csv(exp_dir / "test_summary.csv", index=False)
    with journal.record("evaluate", f"[{cfg.name}] 시험 run 최종 평가", config={"experiment": cfg.name},
                        tags=[cfg.name, cfg.task, "test"]) as entry:
        entry.summary = f"시험 run {len(x_test)}개, 학습 평균 기준선 ({BASELINE_LABELS['mean']}) 대비"
        entry.metrics = {"rows": agg.to_dict(orient="records")}
    print(agg.to_string(index=False))
    return agg


def fit_final_model(cfg: ExperimentConfig, paths: LabPaths, split: RunSplit, spec: dict[str, Any]) -> dict[str, Any]:
    """Refit one thickness job on every fitting run (train + validation) for export.

    Leave-one-run-out keeps no single fitted model; this is the model that would be used on new
    runs. The returned state is what ``ald-predict`` loads from ``estimator.joblib``.
    """

    x, y, inputs, _ = _dataset(cfg, paths, split, ("train", "validation"), input_set=spec.get("input_set", "default"))
    target = spec["target"]
    values = np.nan_to_num(x.to_numpy(np.float64))
    truth = y[target].to_numpy(np.float64)
    estimator = None
    if spec["model"] != "mean":
        estimator = _estimator(spec["model"], spec["params"], int(spec["seed"]))
        estimator.fit(values, truth)
    return {"model": spec["model"], "estimator": estimator, "mean": float(truth.mean()), "features": list(x.columns),
            "inputs": inputs, "target": target, "unit": TARGET_UNITS.get(target, ""), "runs": list(x.index)}
