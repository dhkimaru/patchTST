"""Page 3: experiments — which models and hyperparameters were tried, and what was chosen on validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from ..state import experiment_detail, experiments, journal_frame, prediction_sample, theme_mode
from ..theme import CONTEXT_GRAY, tokens
from ..ui import chart, empty_state, kpi_row, table
from .launcher import render_launcher, render_queue
from .records import render_records

TASK_LABELS = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서", "thickness": "두께(VM)"}
BASELINE_LABELS = {"persistence": "현재값 유지", "template": "레시피 평균", "schedule": "레시피 일정 평균",
                   "mean": "학습 평균"}
COLUMN_LABELS = {
    "label": "모델", "seeds": "seed 수", "mae": "MAE", "mae_std": "MAE 표준편차", "improvement_pct": "기준선 대비 개선(%)",
    "mae_cycle": "MAE·사이클", "mae_other": "MAE·사이클 외", "mae_transition": "MAE·step 전환 직후",
    "mae_stable": "MAE·안정 구간", "fit_minutes": "학습(분)", "windows_per_second": "초당 window",
}


def _signature(exp_dir: str) -> float:
    root = Path(exp_dir)
    return max((p.stat().st_mtime for p in root.glob("*.*")), default=0.0)


def _model_bars(sub: pd.DataFrame, unit: str, target: str) -> None:
    t = tokens(theme_mode())
    gray = CONTEXT_GRAY[theme_mode()]
    sub = sub.sort_values("mae", ascending=False)
    colors = [gray if b else t.series[0] for b in sub["is_baseline"]]
    text = [("기준선" if b else (f"{g:+.1f} %" if pd.notna(g) else "")) for b, g in
            zip(sub["is_baseline"], sub["improvement_pct"])]
    error = sub["mae_std"].fillna(0).to_numpy() if "mae_std" in sub else None
    fig = go.Figure(go.Bar(
        x=sub["mae"], y=sub["label"], orientation="h", marker_color=colors, text=text, textposition="outside",
        error_x=dict(type="data", array=error, visible=bool(error is not None and error.any())),
        hovertemplate="%{y}<br>MAE %{x:.4g}<extra></extra>",
    ))
    fig.update_layout(title=f"모델별 검증 MAE — {target} (짧을수록 좋음, 회색 = 기준선)",
                      xaxis_title=f"MAE ({unit})" if unit else "MAE", yaxis_title=None, showlegend=False,
                      margin=dict(l=10, r=60, t=60, b=40))
    chart(fig, height=120 + 34 * len(sub))


def _learning_curves(rows: pd.DataFrame, histories: dict[str, pd.DataFrame], order: list[str]) -> None:
    t = tokens(theme_mode())
    fig = go.Figure()
    shown = rows.drop_duplicates("label")
    for _, row in shown.iterrows():
        hist = histories.get(row["job_id"])
        if hist is None or not len(hist):
            continue
        color = t.series[order.index(row["label"]) % len(t.series)] if row["label"] in order else t.muted
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], mode="lines+markers", name=row["label"],
                                 line=dict(color=color, width=2), marker=dict(size=4)))
    if not fig.data:
        return
    fig.update_layout(title="검증 손실 (epoch별, 첫 seed)", xaxis_title="epoch", yaxis_title="검증 손실 (표준화 단위)",
                      yaxis_type="log", legend=dict(orientation="h", y=-0.25))
    chart(fig, height=420)


def _prediction_example(rows: pd.DataFrame, unit: str, best: str | None, exp_dir: str) -> None:
    labels = rows.drop_duplicates("label")
    options = labels["label"].tolist()
    label = st.selectbox("예측 예시를 볼 모델", options, index=options.index(best) if best in options else 0,
                         key="pred_model")
    job_id = labels.loc[labels["label"] == label, "job_id"].iloc[0]
    sample = prediction_sample(str(Path(exp_dir) / "jobs" / job_id))
    if sample is None or not len(sample):
        st.caption("예측 예시 파일이 없습니다.")
        return
    horizons = [c[len("true_"):] for c in sample.columns if c.startswith("true_")]
    c1, c2 = st.columns([1, 3])
    horizon = c1.selectbox("예측 거리", horizons, index=len(horizons) - 1, key="pred_h")
    minutes = sample["t_s"] / 60
    lo, hi = float(minutes.min()), float(minutes.max())
    start = lo
    if hi - lo > 10:
        # Open on ALD cycles (where models differ most), five minutes after they begin.
        cycle = minutes[sample["in_cycle"].astype(bool)]
        default = float(cycle.iloc[0]) + 5 if len(cycle) else lo
        start = c2.slider("구간 시작(분)", lo, hi - 10, min(max(default, lo), hi - 10), key="pred_start")
    part = sample[(minutes >= start) & (minutes <= start + 10)]
    t = tokens(theme_mode())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part["t_s"] / 60, y=part[f"true_{horizon}"], name="실측", mode="lines",
                             line=dict(color=t.text_secondary, width=2)))
    fig.add_trace(go.Scatter(x=part["t_s"] / 60, y=part[f"pred_{horizon}"], name=f"예측 ({label})", mode="lines",
                             line=dict(color=t.series[0], width=2, dash="dot")))
    run_id = sample["run_id"].iloc[0]
    fig.update_layout(title=f"{run_id} · {horizon} 예측 vs 실측 (10분 구간)", xaxis_title="run 시작 후 (분)",
                      yaxis_title=unit or "값", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right"))
    chart(fig, height=380)


def _thickness_scatter(rows: pd.DataFrame, unit: str, exp_dir: str) -> None:
    labels = rows.drop_duplicates("label")
    label = st.selectbox("교차검증 예측을 볼 모델", labels["label"].tolist(), key="vm_model")
    path = Path(exp_dir) / "jobs" / labels.loc[labels["label"] == label, "job_id"].iloc[0] / "predictions_cv.parquet"
    if not path.is_file():
        return
    cv = pd.read_parquet(path)
    t = tokens(theme_mode())
    lo, hi = float(min(cv["true"].min(), cv["pred"].min())), float(max(cv["true"].max(), cv["pred"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="예측 = 실측",
                             line=dict(color=CONTEXT_GRAY[theme_mode()], dash="dash")))
    fig.add_trace(go.Scatter(x=cv["true"], y=cv["pred"], mode="markers", name=label, text=cv["run_id"],
                             marker=dict(color=t.series[0], size=8),
                             hovertemplate="%{text}<br>실측 %{x:.3f}<br>예측 %{y:.3f}<extra></extra>"))
    fig.update_layout(title=f"run별 leave-one-out 예측 — {label}", xaxis_title=f"실측 ({unit})",
                      yaxis_title=f"예측 ({unit})", showlegend=False)
    chart(fig, height=420)


def _journal_records() -> None:
    journal = journal_frame()
    runs = journal[journal["stage"].isin(["hpo", "train"])] if len(journal) else journal
    if not len(runs):
        return
    with st.expander(f"모든 학습 기록 ({len(runs)}건, journal)"):
        frame = pd.DataFrame([{
            "시작": r.started_at.tz_convert(None), "단계": "탐색" if r.stage == "hpo" else "학습", "제목": r.title,
            "상태": r.status, "소요(초)": r.duration_s, "요약": r.summary,
        } for r in runs.sort_values("started_at", ascending=False).itertuples()])
        table(frame, height=360)


def render() -> None:
    st.title("3. 하이퍼파라미터·학습")
    st.markdown(
        "하이퍼파라미터를 골라 학습을 돌리고(①), 진행을 보고(②), 지금까지 돌린 모든 조합의 결과를 비교합니다(③). "
        "모델 선택은 검증 run의 MAE로만 하고, 시험 run은 최종 후보를 정한 뒤 한 번만 엽니다."
    )
    tabs = st.tabs(["① 새 실험 실행", "② 진행 상황", "③ 누적 기록", "④ 실험별 상세"])
    with tabs[0]:
        render_launcher()
    with tabs[1]:
        render_queue()
    with tabs[2]:
        render_records()
    with tabs[3]:
        _detail()


def _detail() -> None:
    exps = experiments()
    if not len(exps):
        empty_state(
            "아직 실험 결과 없음",
            "서버에서 `ald-train configs/experiments/g1_forecast_vg11.yaml` 처럼 실행하면 결과가 여기에 쌓입니다.",
        )
        _journal_records()
        return
    labels = [f"{r.experiment} · {r.stamp}" + (" (smoke)" if r.smoke else "") for r in exps.itertuples()]
    pick = st.selectbox("실험", range(len(exps)), format_func=lambda i: labels[i])
    row = exps.iloc[pick]
    detail = experiment_detail(row["dir"], _signature(row["dir"]))
    summary, frame, results = detail["summary"], detail["table"], detail["results"]
    config = yaml.safe_load(detail["config"]) if detail["config"] else {}
    task = summary.get("task")
    baseline = BASELINE_LABELS.get(summary.get("baseline"), "기준선")
    if config.get("description"):
        st.markdown(f"**{config['description']}**")
    kpi_row([
        ("과제", TASK_LABELS.get(task, task), None),
        ("기준선", baseline, "모든 모델의 개선율은 이 기준선의 검증 MAE 대비입니다."),
        ("작업", f"{summary.get('jobs')}개 (실패 {summary.get('failed')})", "모델 × 하이퍼파라미터 조합 × seed"),
        ("총 소요", f"{summary.get('wall_minutes', 0):.1f}분", "데이터 준비부터 요약까지"),
    ])
    if summary.get("smoke"):
        st.info("smoke 실행입니다. run 몇 개·epoch 2로 끝까지 도는지만 확인한 결과라 성능 비교에 쓰지 않습니다.")
    if not len(frame):
        st.warning("성공한 작업이 없습니다. 아래 실패 기록을 확인하세요.")

    selected = summary.get("selected") or {}
    if selected:
        st.subheader("대상별 선택 결과 (검증 run)")
        table(pd.DataFrame([{
            "대상": target, "선택 모델": info["label"], "MAE": info["mae"], f"{baseline} MAE": info.get("baseline_mae"),
            "개선(%)": info.get("improvement_pct"), "단위": info.get("unit"),
        } for target, info in selected.items()]), column_config={
            "MAE": st.column_config.NumberColumn(format="%.4g"),
            f"{baseline} MAE": st.column_config.NumberColumn(format="%.4g"),
            "개선(%)": st.column_config.NumberColumn(format="%+.1f"),
        })

    targets = list(dict.fromkeys(frame["target"])) if len(frame) else []
    if targets:
        target = st.selectbox("대상", targets) if len(targets) > 1 else targets[0]
        sub = frame[frame["target"] == target]
        unit = str(sub["unit"].dropna().iloc[0]) if sub["unit"].notna().any() else ""
        _model_bars(sub, unit, target)
        cols = [c for c in ["label", "seeds", "mae", "mae_std", "improvement_pct"] if c in sub]
        cols += [c for c in sub.columns if c.startswith("mae_") and c[4:].endswith("s") and c[4:-1].replace(".", "").isdigit()]
        cols += [c for c in ["mae_cycle", "mae_other", "mae_transition", "mae_stable", "fit_minutes",
                             "windows_per_second"] if c in sub and sub[c].notna().any()]
        view = sub[cols].rename(columns=lambda c: COLUMN_LABELS.get(c, f"MAE·{c[4:]}" if c.startswith("mae_") else c))
        table(view, column_config={c: st.column_config.NumberColumn(format="%.4g") for c in view.columns
                                   if c not in ("모델", "seed 수")})
        st.caption("MAE·10s 등은 예측 거리별 오차, 사이클은 ALD 사이클 step 구간, step 전환 직후는 step이 바뀐 뒤 10초 이내입니다.")

        grid = [g for g in summary.get("grid_choices") or [] if g["target"] == target]
        if grid:
            st.subheader("하이퍼파라미터 선택")
            for g in grid:
                st.markdown(f"- **{g['model']}**: {g['tried']}개 조합 중 **{g['chosen']}** (검증 MAE {g['mae']:.4g})")
                tried = sub[(sub["model"] == g["model"])][["label", "mae"]].sort_values("mae")
                table(tried.rename(columns={"label": "조합", "mae": "검증 MAE"}),
                      column_config={"검증 MAE": st.column_config.NumberColumn(format="%.4g")})

        ok = results[(results["status"] == "ok") & (results["target"] == target)]
        order = list(dict.fromkeys(ok["label"]))
        if task == "thickness":
            _thickness_scatter(ok, unit, row["dir"])
        else:
            deep = ok[ok["kind"] == "deep"]
            if len(deep):
                st.subheader("학습 곡선")
                _learning_curves(deep, detail["histories"], order)
                cols = [c for c in ["label", "seed", "epochs", "best_epoch", "train_status", "fit_seconds",
                                    "windows_per_second", "peak_gpu_mb", "n_params"] if c in deep]
                table(deep[cols].rename(columns={"label": "모델", "epochs": "epoch", "best_epoch": "최고 epoch",
                                                 "train_status": "종료 이유", "fit_seconds": "학습(초)",
                                                 "windows_per_second": "초당 window", "peak_gpu_mb": "GPU 최대(MB)",
                                                 "n_params": "파라미터 수"}))
            st.subheader("예측 예시 (검증 run)")
            _prediction_example(ok, unit, (selected.get(target) or {}).get("label"), row["dir"])

    test = detail["test"]
    st.subheader("시험 run 최종 결과")
    if test is None:
        st.caption(f"아직 열지 않았습니다. 최종 후보가 정해지면 서버에서 `ald-final-test {row['dir']}` 를 실행합니다.")
    else:
        st.caption("검증으로 미리 고른 후보만 평가했습니다. 이 표로 모델을 다시 고르지 않습니다.")
        shown = test.drop(columns=[c for c in ("model", "is_baseline") if c in test])
        shown = shown.rename(columns={"target": "대상", "unit": "단위", **COLUMN_LABELS})
        table(shown, column_config={c: st.column_config.NumberColumn(format="%.4g") for c in shown.columns
                                    if c not in ("대상", "모델", "단위", "seed 수")})
    failed = results[results["status"] != "ok"] if "status" in results else results.iloc[0:0]
    if len(failed):
        st.subheader("실패한 작업")
        table(failed[[c for c in ["label", "target", "seed", "error"] if c in failed]])
    with st.expander("실험 계획서 (YAML)"):
        st.code(detail["config"], language="yaml")
    if detail["prep"]:
        with st.expander("데이터 준비 기록"):
            st.json(detail["prep"], expanded=False)
    _journal_records()
