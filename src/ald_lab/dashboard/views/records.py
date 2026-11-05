"""Page 3, tab ③: cumulative records — every combination ever trained and what it scored.

Reads :func:`ald_lab.ledger.collect_ledger` (every run in ``work/experiments/``, finished jobs only), so it
grows while an experiment is still training.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...ledger import best_by_target, best_over_time
from ...models.cards import model_card
from ..state import ledger, theme_mode
from ..theme import tokens
from ..ui import chart, empty_state, kpi_row, table

TASK_LABELS = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서", "thickness": "두께(VM)"}
BASELINES = ("persistence", "template", "mean")
MODEL_ORDER = ("ridge", "pls", "lightgbm", "patchtst", "itransformer", "dlinear", "gru")


def _model_name(model: str) -> str:
    return model_card(model)["name"]


def _progress_chart(frame: pd.DataFrame, task: str, target: str, unit: str) -> None:
    """Every trained combination of one target over time (dots, colour = model) and the best so far (line)."""

    t = tokens(theme_mode())
    ok = frame[(frame["task"] == task) & (frame["target"] == target) & frame["mae"].notna()
               & ~frame["model"].isin(BASELINES)]
    if not len(ok):
        return
    fig = go.Figure()
    for i, model in enumerate(m for m in MODEL_ORDER if m in set(ok["model"])):
        sub = ok[ok["model"] == model]
        fig.add_trace(go.Scatter(
            x=sub["when"], y=sub["mae"], mode="markers", name=_model_name(model),
            marker={"size": 8, "color": t.series[i % len(t.series)], "opacity": 0.8},
            customdata=sub[["settings", "input_set", "seed", "memo"]].to_numpy(),
            hovertemplate="%{x|%m-%d %H:%M}<br>MAE %{y:.4g}<br>%{customdata[0]}<br>입력 %{customdata[1]} · "
                          "seed %{customdata[2]}<br>%{customdata[3]}<extra>" + _model_name(model) + "</extra>"))
    best = best_over_time(frame, task, target)
    if len(best) > 1:
        fig.add_trace(go.Scatter(x=best["when"], y=best["best_so_far"], mode="lines", name="지금까지 최고",
                                 line={"color": t.text_secondary, "width": 2, "shape": "hv"}))
    fig.update_layout(title=f"{target} — 학습한 조합 전부 (점)와 지금까지 최고 (선)", yaxis_title=f"검증 MAE ({unit})",
                      xaxis_title="실행 시각", height=380, legend={"orientation": "h", "y": -0.25},
                      margin={"t": 50, "b": 10})
    chart(fig)


def _setting_chart(rows: pd.DataFrame, unit: str) -> None:
    """MAE by the value of one setting, over every run of one model."""

    keys = sorted({k for p in rows["params"] for k in p})
    if not keys:
        st.caption("이 모델은 바꾼 설정이 없습니다 (기본값으로만 학습).")
        return
    key = st.selectbox("설정", keys, key="rec_key")
    values = rows["params"].map(lambda p: p.get(key, "기본값"))
    frame = rows.assign(value=values.astype(str))
    order = sorted(frame["value"].unique(), key=lambda v: (not _is_number(v), float(v) if _is_number(v) else 0, v))
    t = tokens(theme_mode())
    fig = go.Figure(go.Box(x=frame["value"], y=frame["mae"], boxpoints="all", jitter=0.3, pointpos=0,
                           marker={"color": t.series[0], "size": 6}, line={"color": t.series[0]}, name=key,
                           customdata=frame[["settings", "when"]].astype(str).to_numpy(),
                           hovertemplate="%{x}<br>MAE %{y:.4g}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": order, "title": key},
                      yaxis_title=f"검증 MAE ({unit})", height=320, showlegend=False, margin={"t": 20, "b": 10})
    chart(fig)
    st.caption("점 하나가 학습 한 번입니다. 다른 설정도 함께 바뀐 실행이 섞여 있으니, 같은 실행 안의 비교는 ④ 실험별 상세에서 봅니다.")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def render_records() -> None:
    frame = ledger()
    if not len(frame):
        empty_state("아직 기록 없음", "실험이 하나라도 끝나면(또는 작업 하나가 끝나면) 여기에 한 줄씩 쌓입니다.")
        return
    ok = frame[frame["mae"].notna()]
    runs = frame["exp_dir"].nunique()
    kpi_row([
        ("학습 기록", f"{len(ok):,}개", "끝난 작업 수 (모델 × 조합 × 대상 × seed, 기준선 포함)"),
        ("실험 실행", f"{runs}회", "work/experiments/ 아래 실행 폴더 수"),
        ("대시보드에서 실행", f"{frame.loc[frame['source'] != '터미널', 'exp_dir'].nunique()}회", None),
        ("최근 실행", f"{frame['when'].max():%m-%d %H:%M}" if frame["when"].notna().any() else "-", None),
    ])
    st.subheader("대상별 지금까지 최고")
    best = best_by_target(frame)
    if len(best):
        table(pd.DataFrame({
            "과제": best["task"].map(TASK_LABELS), "대상": best["target"], "검증 MAE": best["mae"],
            "개선율(%)": best["improvement_pct"], "모델": best["label"], "설정": best["settings"],
            "입력": best["input_set"], "seed": best["seed"], "언제": best["when"].dt.strftime("%m-%d %H:%M"),
            "실험": best["experiment"],
        }), column_config={"검증 MAE": st.column_config.NumberColumn(format="%.4g"),
                           "개선율(%)": st.column_config.NumberColumn(format="%+.1f")})

    st.subheader("한 대상의 기록")
    c = st.columns(3)
    tasks = [x for x in TASK_LABELS if x in set(frame["task"])]
    task = c[0].selectbox("과제", tasks, format_func=TASK_LABELS.get, key="rec_task")
    sub = frame[frame["task"] == task]
    target = c[1].selectbox("대상", list(dict.fromkeys(sub["target"])), key="rec_target")
    sub = sub[sub["target"] == target]
    models = [m for m in MODEL_ORDER + BASELINES if m in set(sub["model"])]
    chosen = c[2].multiselect("모델", models, default=models, format_func=_model_name, key="rec_models")
    sub = sub[sub["model"].isin(chosen)]
    unit = str(sub["unit"].dropna().iloc[0]) if sub["unit"].notna().any() else ""
    _progress_chart(frame[frame["model"].isin(chosen)], task, target, unit)
    order = st.radio("정렬", ["MAE 낮은 순", "최근 순"], horizontal=True, key="rec_order")
    view = sub.sort_values("mae" if order == "MAE 낮은 순" else "when", ascending=order == "MAE 낮은 순")
    horizon = [c for c in view.columns if c.startswith("mae_") and c[4:].endswith("s") and view[c].notna().any()]
    shown = pd.DataFrame({
        "언제": view["when"].dt.strftime("%m-%d %H:%M"), "모델": view["label"], "설정": view["settings"],
        "입력": view["input_set"], "seed": view["seed"], "검증 MAE": view["mae"], "개선율(%)": view["improvement_pct"],
        **{f"MAE {c[4:]}": view[c] for c in horizon},
        "사이클 MAE": view["mae_cycle"], "시험 MAE": view["test_mae"], "학습(분)": view["fit_minutes"],
        "출처": view["source"], "메모": view["memo"], "상태": view["status"],
    })
    table(shown, height=420, column_config={
        **{k: st.column_config.NumberColumn(format="%.4g") for k in shown.columns if "MAE" in k},
        "개선율(%)": st.column_config.NumberColumn(format="%+.1f"),
        "학습(분)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("이 표를 CSV로", shown.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"records_{task}_{target}.csv", mime="text/csv")

    trained = [m for m in chosen if m not in BASELINES]
    if trained:
        st.subheader("설정 값별 MAE (모든 실행)")
        model = st.selectbox("모델", trained, format_func=_model_name, key="rec_setting_model")
        _setting_chart(sub[sub["model"] == model], unit)
