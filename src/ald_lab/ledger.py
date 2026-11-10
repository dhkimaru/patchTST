"""Cumulative experiment records: one row per finished job of every run in ``work/experiments/``.

Unlike the report's tabs (the newest run of each experiment), this keeps every run, so it answers "which
settings gave which result" across days. Rows are read from each run's ``results.csv`` when the run is
finished and from ``jobs/<id>/metrics_validation.json`` while it is still training, so a running experiment
grows a row per finished job. Settings come from ``jobs.json`` and ``experiment.yaml``. Aggregates only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .launch import HP_SUFFIX, read_request

BASE_COLUMNS = ("when", "experiment", "base", "stamp", "status", "source", "memo", "task", "target", "unit", "model",
                "label", "input_set", "seed", "settings", "mae", "improvement_pct", "mae_cycle", "mae_other",
                "test_mae", "fit_minutes", "n_params", "context_seconds", "study", "exp_dir", "job_id")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _when(stamp: str) -> pd.Timestamp | None:
    when = pd.to_datetime(stamp, format="%Y%m%d-%H%M%S", errors="coerce")
    return None if pd.isna(when) else when


def settings_text(params: dict[str, Any]) -> str:
    return " · ".join(f"{k}={v}" for k, v in sorted(params.items())) or "기본값"


def _requests(work_root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    root = work_root / "queue"
    if root.is_dir():
        for folder in root.iterdir():
            doc = read_request(folder) if folder.is_dir() else None
            if doc and doc.get("exp_dir"):
                out[str(Path(doc["exp_dir"]).resolve())] = doc
    return out


def _run_rows(exp_dir: Path, request: dict[str, Any] | None) -> list[dict[str, Any]]:
    jobs = {j["job_id"]: j for j in _read_json(exp_dir / "jobs.json") or []}
    if not jobs:
        return []
    config = yaml.safe_load((exp_dir / "experiment.yaml").read_text(encoding="utf-8")) \
        if (exp_dir / "experiment.yaml").is_file() else {}
    summary = _read_json(exp_dir / "summary.json")
    finished = summary is not None
    results: dict[str, dict[str, Any]] = {}
    if (exp_dir / "results.csv").is_file():
        frame = pd.read_csv(exp_dir / "results.csv")
        results = {str(r["job_id"]): r for r in frame.to_dict(orient="records")}
    name = exp_dir.parent.name
    task = config.get("task") or (summary or {}).get("task")
    context = (config.get("window") or {}).get("context_seconds", 192 if task != "thickness" else None)
    rows = []
    for job_id, job in jobs.items():
        res = results.get(job_id)
        if res is None:
            doc = _read_json(exp_dir / "jobs" / job_id / "metrics_validation.json")
            if not doc:
                continue  # not finished yet
            res = doc.get("result") or {**doc.get("metrics", {}), "status": "ok"}
        test = _read_json(exp_dir / "jobs" / job_id / "metrics_test.json") or {}
        test_mae = test.get("mae", (test.get("metrics") or {}).get("mae"))
        fit = res.get("fit_seconds")
        rows.append({
            "when": _when(exp_dir.name), "experiment": name,
            "base": name.removesuffix("_smoke").removesuffix(HP_SUFFIX), "stamp": exp_dir.name,
            "status": "완료" if finished else "실행 중",
            "source": "대시보드" if request and request.get("source") == "dashboard"
            else ("대기열" if request else "터미널"),
            "memo": (request or {}).get("memo") or config.get("description", ""), "task": task,
            "target": job["target"], "unit": res.get("unit"), "model": job["model"], "label": job.get("label"),
            "input_set": job.get("input_set", "default"), "seed": job.get("seed"),
            "settings": settings_text(job.get("params") or {}), "params": job.get("params") or {},
            "job_status": res.get("status", "ok"), "mae": res.get("mae"), "improvement_pct": res.get("improvement_pct"),
            "mae_cycle": res.get("mae_cycle"), "mae_other": res.get("mae_other"), "test_mae": test_mae,
            "fit_minutes": float(fit) / 60 if fit is not None and not pd.isna(fit) else None, "n_params": res.get("n_params"),
            "context_seconds": context, "study": str(config.get("study") or ""), "exp_dir": str(exp_dir), "job_id": job_id, "smoke": name.endswith("_smoke"),
            **{k: v for k, v in res.items() if k.startswith("mae_") and k[4:].endswith("s")
               and k[4:-1].replace(".", "").isdigit()},
        })
    return rows


def collect_ledger(work_root: Path, *, include_smoke: bool = False) -> pd.DataFrame:
    """Every finished job of every run, newest run first."""

    work_root = Path(work_root)
    root = work_root / "experiments"
    if not root.is_dir():
        return pd.DataFrame(columns=list(BASE_COLUMNS))
    requests = _requests(work_root)
    rows = []
    for exp_dir in root.glob("*/*"):
        if not exp_dir.is_dir() or (exp_dir.parent.name.endswith("_smoke") and not include_smoke):
            continue
        rows += _run_rows(exp_dir, requests.get(str(exp_dir.resolve())))
    if not rows:
        return pd.DataFrame(columns=list(BASE_COLUMNS))
    frame = pd.DataFrame(rows)
    return frame.sort_values(["when", "target", "mae"], ascending=[False, True, True]).reset_index(drop=True)


def main_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the main protocol. A side study (``study:`` in the experiment YAML, e.g. a cross-validation fold or
    longer horizons) is scored on other runs or other horizons, so records and decisions leave it out."""

    if "study" not in frame:
        return frame
    return frame[frame["study"].fillna("").astype(str) == ""]


def best_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Lowest validation MAE per (task, target) over every run of the main protocol, baselines aside."""

    frame = main_records(frame)
    ok = frame[(frame.get("job_status", "ok") == "ok") & frame["mae"].notna()]
    ok = ok[~ok["model"].isin(("persistence", "template", "schedule", "mean"))]
    if not len(ok):
        return ok
    idx = ok.groupby(["task", "target"])["mae"].idxmin()
    return ok.loc[idx].sort_values(["task", "target"]).reset_index(drop=True)


def best_over_time(frame: pd.DataFrame, task: str, target: str) -> pd.DataFrame:
    """Per run (time) the best MAE of ``target`` and the best so far — how the record improved."""

    frame = main_records(frame)
    ok = frame[(frame["task"] == task) & (frame["target"] == target) & frame["mae"].notna()
               & ~frame["model"].isin(("persistence", "template", "schedule", "mean"))]
    if not len(ok):
        return ok
    per_run = ok.loc[ok.groupby("exp_dir")["mae"].idxmin()].sort_values("when")
    per_run = per_run.assign(best_so_far=per_run["mae"].cummin())
    return per_run.reset_index(drop=True)


DEEP_MODELS = ("patchtst", "itransformer", "dlinear", "gru")


def _minutes_text(value: float) -> str:
    if value < 1:
        return f"{value * 60:.0f}초"
    return f"{value:.1f}분" if value < 90 else f"{value / 60:.1f}시간"


def estimate_parts(records: pd.DataFrame, jobs: list[dict[str, Any]], task: str, n_devices: int) -> tuple[str, str]:
    """Rough wall time of ``jobs`` and how it was reached: the median training time of the same task and model in
    earlier runs (deep models share ``n_devices`` GPUs, the others run one after another). Training time only."""

    if not len(records) or not jobs:
        return "기록 없음", "같은 과제의 이전 학습 기록이 없어 추정 불가"
    past = records[(records["task"] == task) & records["fit_minutes"].notna() & ~records["smoke"].astype(bool)]
    medians = past.groupby("model")["fit_minutes"].median()
    serial = parallel = 0.0
    counts: dict[str, int] = {}
    unknown = set()
    for job in jobs:
        minutes = medians.get(job["model"])
        if minutes is None or pd.isna(minutes):
            unknown.add(job["model"])
            continue
        counts[job["model"]] = counts.get(job["model"], 0) + 1
        if job["model"] in DEEP_MODELS:
            parallel += float(minutes)
        else:
            serial += float(minutes)
    total = serial + parallel / max(1, n_devices)
    text = f"약 {_minutes_text(total)}" if total >= 1 / 60 else "1분 미만"
    parts = [f"{model} {n}개 × {_minutes_text(float(medians[model]))}" for model, n in counts.items()
             if float(medians[model]) * n >= 1 / 60]
    basis = " + ".join(parts) or "기준선만"
    if any(m in DEEP_MODELS for m in counts) and n_devices > 1:
        basis += f" (딥러닝은 GPU {n_devices}장에 나눔)"
    basis += " · 이전 학습 시간 중앙값 기준 (데이터 준비·평가 제외, 큰 모델은 더 걸릴 수 있음)"
    if unknown:
        basis += f" · 기록 없는 모델: {', '.join(sorted(unknown))}"
    return text, basis


def estimate_text(records: pd.DataFrame, jobs: list[dict[str, Any]], task: str, n_devices: int) -> str:
    """:func:`estimate_parts` in one line (the Streamlit launcher)."""

    text, basis = estimate_parts(records, jobs, task, n_devices)
    return text if text == "기록 없음" else f"{text} — {basis}"
