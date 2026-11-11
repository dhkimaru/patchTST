"""Run a trained model on new trace CSVs.

``ald-predict <model folder> --csv <file or folder>`` takes a job folder
(``work/experiments/<name>/<stamp>/jobs/<id>``) or a package written by
``ald-export-model``. Both hold the same files: the weights, ``job.json``
(inputs, train-fitted scaler, window shape) and, for virtual sensors, the
recipe template. Predictions go through the same code as validation and the
final test (:func:`ald_lab.training.runner.predict_job`), so a run from the
validation set reproduces its validation error.

Output: one row per prediction with the time it was made, the step, and for
each horizon the predicted value (and the measured one when the CSV has it).
Thickness models give one row per run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.raw import list_trace_files, read_trace_csv, run_id_from_path
from .config import ExperimentConfig, apply_smoke, load_experiment
from .prepare import bundle_from_traces


class PredictError(ValueError):
    """Raised when a model folder or the given traces cannot be used."""


def experiment_for(model_dir: Path) -> ExperimentConfig:
    """The experiment a model folder came from (package: copied next to it; job: two levels up)."""

    for root in (model_dir, model_dir.parent.parent):
        path = root / "experiment.yaml"
        if path.is_file():
            cfg = load_experiment(path)
            summary = root / "summary.json"
            if summary.is_file() and json.loads(summary.read_text(encoding="utf-8")).get("smoke"):
                cfg = apply_smoke(cfg)
            return cfg
    raise PredictError(f"experiment.yaml not found next to {model_dir} or two levels above it")


def trace_files(spec: str | Path) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        return list_trace_files(path)
    if path.is_file():
        return [path]
    raise PredictError(f"no such file or folder: {path}")


def _run_id(path: Path) -> str:
    try:
        return run_id_from_path(path)
    except ValueError:
        return path.stem


def predict_traces(model_dir: str | Path, files: list[Path], *, device: str = "cpu") -> pd.DataFrame:
    """Predictions of the model in ``model_dir`` for each trace file."""

    from .runner import load_predictor, predict_job, window_set_for_job
    from .windows import Scaler

    model_dir = Path(model_dir)
    if not (model_dir / "job.json").is_file():
        raise PredictError(f"{model_dir} is not a model folder (no job.json)")
    doc = json.loads((model_dir / "job.json").read_text(encoding="utf-8"))
    cfg = experiment_for(model_dir)
    frames = [(_run_id(p), read_trace_csv(p)) for p in files]
    if doc["task"] == "thickness":
        return _predict_thickness(model_dir, doc, cfg, frames)

    target = doc["job"]["target"]
    scaler = Scaler.from_dict(doc["scaler"])
    bundle, meta, absent = bundle_from_traces(frames, scaler.columns, float(doc["sample_seconds"]),
                                              str(doc.get("bin_align", "run")))
    missing_inputs = [c for c in absent if c in doc["inputs"]]
    if missing_inputs:
        raise PredictError(f"the traces lack model inputs: {missing_inputs}")
    _, model = load_predictor(model_dir, device)
    ws = window_set_for_job(doc, bundle, device)
    pred = predict_job(model_dir, doc, model, ws, bundle, cfg)
    idx = np.arange(len(ws))
    t = scaler.columns.index(target)
    y_pred = scaler.inverse(pred, [t])
    rows = ws.target_rows(idx)
    y_true = None if target in absent else scaler.inverse(ws.y.cpu().numpy()[rows], [t])
    ends = ws.ends_np
    out = meta.iloc[ends][["run_id", "timestamp", "step_name"]].reset_index(drop=True)
    out["in_cycle"] = np.asarray(bundle.in_cycle)[ends]
    for i, h in enumerate(doc["horizons_seconds"]):
        suffix = "" if float(h) == 0 else f"_+{float(h):g}s"
        if y_true is not None:
            out[f"actual{suffix}"] = y_true[:, i, 0]
        out[f"pred{suffix}"] = y_pred[:, i, 0]
    out.attrs.update(target=target, unit=_unit(model_dir), has_actual=y_true is not None)
    return out


def _unit(model_dir: Path) -> str:
    path = model_dir / "metrics_validation.json"
    if not path.is_file():
        return ""
    return str(json.loads(path.read_text(encoding="utf-8")).get("result", {}).get("unit") or "")


def _predict_thickness(model_dir: Path, doc: dict[str, Any], cfg: ExperimentConfig,
                       frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    from .vm import run_features

    path = model_dir / "estimator.joblib"
    if not path.is_file():
        raise PredictError("thickness models are fitted on all runs by ald-export-model; "
                           "predict from the exported package")
    import joblib

    state = joblib.load(path)
    bundle, _, absent = bundle_from_traces(frames, state["inputs"], cfg.sample_seconds)
    if absent:
        raise PredictError(f"the traces lack model inputs: {absent}")
    x = run_features(bundle, state["inputs"]).reindex(columns=state["features"])
    if state["estimator"] is None:
        pred = np.full(len(x), float(state["mean"]))
    else:
        pred = np.asarray(state["estimator"].predict(np.nan_to_num(x.to_numpy(np.float64)))).reshape(-1)
    out = pd.DataFrame({"run_id": list(x.index), f"pred_{state['target']}": pred})
    out.attrs.update(target=state["target"], unit=state.get("unit", ""), has_actual=False)
    return out


def describe_errors(frame: pd.DataFrame) -> list[str]:
    """MAE per horizon when the traces also contain the measured target."""

    lines = []
    for col in [c for c in frame.columns if c.startswith("pred")]:
        actual = col.replace("pred", "actual", 1)
        if actual in frame:
            err = (frame[col] - frame[actual]).abs()
            cyc = err[frame["in_cycle"]] if "in_cycle" in frame else err.iloc[:0]
            lines.append(f"{col}: MAE {err.mean():.4g}" + (f" (ALD 사이클 {cyc.mean():.4g})" if len(cyc) else ""))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="학습된 모델로 새 trace CSV를 예측합니다 (job 폴더 또는 ald-export-model 패키지)."
    )
    parser.add_argument("model_dir", help="모델 폴더: jobs/<id> 또는 work/packages/.../<대상>")
    parser.add_argument("--csv", required=True, help="trace CSV 파일 또는 standard_trace_NNN.csv가 있는 폴더")
    parser.add_argument("--out", help="출력 CSV (기본: <모델 폴더>/predictions.csv)")
    parser.add_argument("--device", default="cpu", help="'cpu' 또는 'cuda:0'")
    parser.add_argument("--html", action="store_true",
                        help="예측을 시간 순서대로 재생하는 HTML도 만든다 (<출력 CSV 이름>_player.html)")
    args = parser.parse_args(argv)
    files = trace_files(args.csv)
    frame = predict_traces(args.model_dir, files, device=args.device)
    out = Path(args.out) if args.out else Path(args.model_dir) / "predictions.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")  # BOM: Excel opens Korean text correctly
    print(f"{frame.attrs.get('target')}: run {len(files)}개 → 예측 {len(frame):,}행 → {out}")
    for line in describe_errors(frame):
        print("  " + line)
    if args.html:
        page = write_player(frame, Path(args.model_dir), out.with_name(out.stem + "_player.html"))
        print(f"재생 화면: {page}" if page else "두께 예측은 run 하나에 값 하나라 재생 화면을 만들지 않습니다.")
    return 0


def write_player(frame: pd.DataFrame, model_dir: Path, out: Path) -> Path | None:
    """The report's inference player for these predictions, as one offline HTML file."""

    from ..models.cards import model_card
    from ..report.player import prediction_series
    from ..report.render import player_page

    if "timestamp" not in frame:
        return None
    doc = json.loads((model_dir / "job.json").read_text(encoding="utf-8"))
    job = doc["job"]
    model = model_card(job["model"])["name"] + ("" if job.get("label", job["model"]) == job["model"]
                                                 else f" ({job['label']})")
    items = prediction_series(frame, model=model, context=float(doc["context_steps"]) * float(doc["sample_seconds"]))
    if not items:
        return None
    target = frame.attrs.get("target", "")
    title = f"{target} 추론 재생 ({model})"
    meta = f"모델 폴더 {model_dir} · run {len(items)}개 · " + ("실측 있음 (오차 표시)" if frame.attrs.get("has_actual")
                                                            else "실측 없음 (예측만)")
    out.write_text(player_page(items, title=title, meta=meta), encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
