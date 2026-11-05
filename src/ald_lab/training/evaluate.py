"""Physical-unit metrics with breakdowns, and plain-language strengths/weaknesses.

Breakdowns answer where a model helps: per horizon, inside vs outside ALD
cycles, right after a recipe step change vs stable, and per run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .prepare import Bundle

TRANSITION_SECONDS = 10.0


def _mae(err: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = err if mask is None else err[mask]
    return float(values.mean()) if values.size else None


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    rows: np.ndarray,
    bundle: Bundle,
    horizons_seconds: list[float],
    transition_seconds: float = TRANSITION_SECONDS,
) -> dict[str, Any]:
    """Metrics for ``(n, H, T)`` arrays in physical units; ``rows`` are the bundle rows ``(n, H)``."""

    err = np.abs(y_pred - y_true)  # (n, H, T)
    sq = (y_pred - y_true) ** 2
    sst = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum()
    in_cycle = np.asarray(bundle.in_cycle)[rows][..., None]
    transition = (np.asarray(bundle.step_pos)[rows] * bundle.sample_seconds < transition_seconds)[..., None]
    in_cycle = np.broadcast_to(in_cycle, err.shape)
    transition = np.broadcast_to(transition, err.shape)
    run_of = np.asarray(bundle.run_index)[rows[:, 0]]
    by_run = {}
    for r in np.unique(run_of):
        by_run[bundle.run_ids[int(r)]] = float(err[run_of == r].mean())
    labels = [f"{h:g}s" for h in horizons_seconds]
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(sq.mean())),
        "r2": float(1 - sq.sum() / sst) if sst > 0 else None,
        "p95_abs_error": float(np.quantile(err, 0.95)),
        "by_horizon": {label: float(err[:, i].mean()) for i, label in enumerate(labels)},
        "by_region": {"cycle": _mae(err, in_cycle), "other": _mae(err, ~in_cycle)},
        "by_phase": {"transition": _mae(err, transition), "stable": _mae(err, ~transition)},
        "by_run": by_run,
        "n_windows": int(err.shape[0]),
        "transition_seconds": transition_seconds,
    }


def improvement(mae: float | None, baseline_mae: float | None) -> float | None:
    if mae is None or not baseline_mae:
        return None
    return float((1 - mae / baseline_mae) * 100)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def describe(
    metrics: dict[str, Any],
    *,
    unit: str = "",
    baseline: dict[str, Any] | None = None,
    baseline_label: str = "기준선",
    training: dict[str, Any] | None = None,
    patience: int | None = None,
) -> tuple[list[str], list[str]]:
    """Rule-based strengths and weaknesses (Korean, for the dashboard)."""

    strengths: list[str] = []
    weaknesses: list[str] = []
    u = f" {unit}" if unit and unit != "-" else ""
    if baseline is not None:
        gain = improvement(metrics["mae"], baseline.get("mae"))
        line = f"{baseline_label} 대비 MAE {_fmt(baseline.get('mae'))} → {_fmt(metrics['mae'])}{u}"
        if gain is not None and gain >= 5:
            strengths.append(f"{line} ({gain:.1f} % 감소)")
        elif gain is not None and gain <= -2:
            weaknesses.append(f"{line} ({-gain:.1f} % 증가) — 기준선보다 나쁨")
        elif gain is not None:
            weaknesses.append(f"{line} — 기준선과 차이가 거의 없음 ({gain:+.1f} %)")
        for key, name in (("cycle", "ALD 사이클 구간"), ("other", "사이클 외 구간(대기·승온·배기)")):
            g = improvement(metrics["by_region"].get(key), baseline["by_region"].get(key))
            if g is None:
                continue
            if g >= 5:
                strengths.append(f"{name}: 기준선 대비 {g:.1f} % 개선")
            elif g <= -2:
                weaknesses.append(f"{name}: 기준선보다 {-g:.1f} % 나쁨")
    horizons = list(metrics["by_horizon"].items())
    if len(horizons) > 1 and horizons[0][1]:
        ratio = horizons[-1][1] / horizons[0][1]
        if ratio >= 2:
            weaknesses.append(
                f"예측 거리가 길수록 오차 증가: {horizons[0][0]} {_fmt(horizons[0][1])} → "
                f"{horizons[-1][0]} {_fmt(horizons[-1][1])}{u} ({ratio:.1f}배)"
            )
    phase = metrics["by_phase"]
    if phase.get("transition") and phase.get("stable"):
        ratio = phase["transition"] / phase["stable"]
        secs = metrics.get("transition_seconds", TRANSITION_SECONDS)
        if ratio >= 2:
            weaknesses.append(f"step 전환 직후({secs:g}초 이내) 오차가 안정 구간의 {ratio:.1f}배")
        elif ratio <= 1.3:
            strengths.append(f"step 전환 직후에도 오차가 안정 구간과 비슷함 ({ratio:.1f}배)")
    by_run = metrics.get("by_run") or {}
    if len(by_run) >= 3:
        values = np.array(list(by_run.values()))
        worst = max(by_run, key=by_run.get)
        ratio = by_run[worst] / float(np.median(values)) if np.median(values) > 0 else np.inf
        if ratio >= 2:
            weaknesses.append(f"{worst}에서 오차가 다른 run 중앙값의 {ratio:.1f}배 (run별 편차 큼)")
        else:
            strengths.append(f"run별 오차가 고름 (최대 run이 중앙값의 {ratio:.1f}배)")
    if training:
        best, epochs = training.get("best_epoch") or 0, training.get("epochs") or 0
        if training.get("status") == "MAX_EPOCHS" and best == epochs:
            weaknesses.append("마지막 epoch까지 검증 오차가 줄고 있었음 — 더 학습하면 나아질 수 있음")
        if patience and best and best <= 3 and epochs >= best + patience:
            weaknesses.append(f"{best} epoch에서 최고 성능 후 검증 오차 증가 — 과적합 또는 학습률 과다 가능성")
        if training.get("status") == "NAN_LOSS":
            weaknesses.append("학습 중 손실이 발산(NaN)함")
    return strengths, weaknesses
