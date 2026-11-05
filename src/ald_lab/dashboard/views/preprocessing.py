"""Page 2: preprocessing — what was removed, split and scaled, and why."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...data.splits import DEFAULT_SPLIT
from ..state import journal_frame, overview, split_table, theme_mode
from ..theme import CONTEXT_GRAY, tokens
from ..ui import chart, empty_state, missing_overview_notice, table

PLANNED_STEPS = [
    ("컬럼 선별", "상수·완전 중복 제거, VG11 복사본(PressAct·AUXMon_VG11) 표시", "자동 점검 완료"),
    ("Run 선별", "시간 역행(071) 처리, 두께 이상 run(55·57) 제외 여부 결정", "예정"),
    ("구간 분할", "0.5초 간격이 끊긴 지점에서 분할, window가 구간을 넘지 않게", "예정"),
    ("리샘플링", "0.5초 원본 유지 또는 1초 decimation (실험 축)", "예정"),
    ("Run 단위 분할", "학습/검증/시험을 run(파일) 단위로, 시간 순서 유지 (split_v2 계승). 교차검증 fold는 예정", "설정 완료"),
    ("정규화", "학습 run에서만 평균·표준편차 계산", "예정"),
]


PARTITION_LABELS = {"train": "학습", "validation": "검증", "test": "시험 (봉인)", "excluded": "제외"}


def _split_section(tables) -> None:
    st.subheader("Run 단위 분할 (split_v2)")
    st.markdown(
        "이전 프로젝트의 split_v2를 그대로 씁니다. run을 시작 시각 순으로 놓고 앞 70%를 학습, 다음 15%를 검증, "
        "마지막 15%를 시험으로 나눴습니다. **시험 run은 최종 후보 평가 전까지 열지 않습니다** "
        "(코드에서도 `include_test=True` 없이는 시험 run을 꺼낼 수 없습니다)."
    )
    split = split_table(str(DEFAULT_SPLIT))
    runs = tables["runs"][["run_id", "start", "duration_h"]].merge(split, on="run_id", how="left")
    runs["partition"] = runs["partition"].fillna("excluded")
    t = tokens(theme_mode())
    colors = {"train": t.series[0], "validation": t.series[1], "test": t.series[2], "excluded": CONTEXT_GRAY[theme_mode()]}
    fig = go.Figure()
    for part in ("train", "validation", "test", "excluded"):
        sub = runs[runs["partition"] == part]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(x=sub["start"], y=[PARTITION_LABELS[part]] * len(sub), mode="markers",
                                 name=f"{PARTITION_LABELS[part]} {len(sub)}", marker=dict(size=10, color=colors[part]),
                                 customdata=sub[["run_id"]].to_numpy(),
                                 hovertemplate="%{customdata[0]}<br>%{x|%m-%d %H:%M}<extra></extra>"))
    fig.update_layout(title="run 시작 시각과 분할", xaxis_title="시작 시각",
                      yaxis=dict(categoryorder="array",
                                 categoryarray=[PARTITION_LABELS[p] for p in ("excluded", "test", "validation", "train")]))
    chart(fig, height=300)
    stats = (
        runs.groupby("partition")
        .agg(runs=("run_id", "size"), first=("start", "min"), last=("start", "max"), hours=("duration_h", "sum"))
        .reindex(["train", "validation", "test", "excluded"]).dropna(how="all").reset_index()
    )
    stats["partition"] = stats["partition"].map(PARTITION_LABELS)
    table(stats, column_config={
        "partition": "구분", "runs": st.column_config.NumberColumn("run 수", format="%d"),
        "first": st.column_config.DatetimeColumn("첫 run", format="YYYY-MM-DD"),
        "last": st.column_config.DatetimeColumn("마지막 run", format="YYYY-MM-DD"),
        "hours": st.column_config.NumberColumn("데이터 시간 (h)", format="%.0f"),
    })


def render() -> None:
    st.title("2. 전처리")
    tables = overview()
    summary = tables.get("summary")
    if not summary:
        missing_overview_notice()
        return
    st.markdown("원본 데이터에서 학습 입력을 만들기까지의 단계와, 각 단계에서 무엇이 얼마나 바뀌었는지 기록합니다.")

    st.subheader("컬럼 선별 (자동 점검 결과)")
    status = summary["column_status"]
    total = summary["n_signal_columns"]
    after_constant = total - status.get("CONSTANT", 0)
    after_duplicate = after_constant - status.get("DUPLICATE", 0)
    stages = [("원본 신호", total), ("상수 제거 후", after_constant), ("완전 중복 제거 후", after_duplicate)]
    t = tokens(theme_mode())
    ramp = [t.sequential[5], t.sequential[4], t.sequential[3]]
    fig = go.Figure(go.Bar(
        x=[v for _, v in stages], y=[k for k, _ in stages], orientation="h", marker_color=ramp,
        text=[f"{v}개" for _, v in stages], textposition="outside",
        hovertemplate="%{y}: %{x}개<extra></extra>"))
    fig.update_layout(title="남는 컬럼 수", yaxis=dict(autorange="reversed"), xaxis_title="컬럼 수", showlegend=False)
    chart(fig, height=300)
    st.caption(
        f"근사 복사 쌍 {summary['n_near_copy_pairs']}개(|r|≥0.999)는 아직 제거하지 않았습니다. "
        "특히 PressAct·AUXMon_VG11은 VG11의 복사본이라 가상 센서 평가에서 VG11과 함께 빼야 공정한 비교가 됩니다. "
        f"핵심 파라미터 실험에서는 중요도 A 항목의 대표 컬럼 {len(summary['core_columns_A'])}개를 별도 입력 세트로 씁니다."
    )

    st.subheader("전처리 단계")
    rows = [{"단계": n, "내용": d, "상태": s} for n, d, s in PLANNED_STEPS]
    table(pd.DataFrame(rows))

    _split_section(tables)

    journal = journal_frame()
    entries = journal[journal["stage"] == "preprocess"] if len(journal) else journal
    st.subheader("전처리 실행 기록")
    if not len(entries):
        empty_state("아직 실행 기록 없음",
                    "전처리 파이프라인이 실행되면 설정, 입력/출력 행·컬럼 수, 소요 시간이 여기에 시간순으로 쌓입니다.")
        return
    for row in entries.sort_values("started_at").itertuples():
        with st.container(border=True):
            st.markdown(f"**{row.title}** · {row.started_at.tz_convert(None):%Y-%m-%d %H:%M} · {row.duration_s:,.1f}초")
            if row.summary:
                st.write(row.summary)
            if row.metrics:
                st.json(row.metrics, expanded=False)
            with st.expander("설정"):
                st.code(json.dumps(row.config, ensure_ascii=False, indent=2), language="json")
