"""Page 4: every recorded stage in time order, with duration, strengths and weaknesses."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from ...journal import STAGE_LABELS, STAGES
from ..state import journal_frame, theme_mode
from ..theme import STATUS, tokens
from ..ui import chart, empty_state, kpi_row


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}초"
    if seconds < 5400:
        return f"{seconds / 60:.1f}분"
    return f"{seconds / 3600:.1f}시간"


def render() -> None:
    st.title("4. 타임라인·결과")
    st.markdown("데이터 점검부터 학습·평가까지 모든 단계를 시간 순서로 보여 줍니다. 각 기록에는 소요 시간과 강점·약점이 함께 남습니다.")
    journal = journal_frame()
    if not len(journal):
        empty_state("아직 기록 없음", "`ald-build-overview`를 실행하면 첫 기록(원본 데이터 점검)이 생깁니다.")
        return
    journal = journal.copy()
    journal["start_local"] = journal["started_at"].dt.tz_convert(None)
    journal["duration_s"] = journal["duration_s"].fillna(0.0)
    kpi_row(
        [
            ("기록", f"{len(journal)}건", None),
            ("총 소요 시간", _fmt_duration(journal["duration_s"].sum()), "모든 단계 합계"),
            ("실패", f"{(journal['status'] == 'failed').sum()}건", None),
            ("기간", f"{journal['start_local'].min():%m-%d} ~ {journal['start_local'].max():%m-%d}", None),
        ]
    )
    t = tokens(theme_mode())
    stage_order = [s for s in STAGES if s in set(journal["stage"])]
    fig = go.Figure()
    for i, stage in enumerate(stage_order):
        part = journal[journal["stage"] == stage]
        fig.add_trace(go.Bar(
            x=part["duration_s"] * 1000, base=part["start_local"], y=[STAGE_LABELS[stage]] * len(part),
            orientation="h", name=STAGE_LABELS[stage], marker_color=t.series[i], width=0.4,
            customdata=part[["title", "status"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>%{base|%m-%d %H:%M} 시작<br>상태 %{customdata[1]}<extra></extra>",
        ))
        failed = part[part["status"] == "failed"]
        if len(failed):
            fig.add_trace(go.Scatter(x=failed["start_local"], y=[STAGE_LABELS[stage]] * len(failed), mode="markers",
                                     marker=dict(symbol="x", size=10, color=STATUS["critical"]), name="실패",
                                     showlegend=False, hoverinfo="skip"))
    fig.update_layout(title="단계별 실행 시점과 길이", xaxis=dict(type="date", title="시각"),
                      yaxis=dict(categoryorder="array", categoryarray=[STAGE_LABELS[s] for s in reversed(stage_order)]),
                      barmode="overlay")
    chart(fig, height=140 + 60 * len(stage_order))

    st.subheader("기록 (최신순)")
    for row in journal.sort_values("started_at", ascending=False).itertuples():
        status = ":green[■] 완료" if row.status == "ok" else (":red[■] 실패" if row.status == "failed" else ":orange[■] 진행 중")
        with st.container(border=True):
            st.markdown(
                f"**{row.title}** · {STAGE_LABELS.get(row.stage, row.stage)} · "
                f"{row.start_local:%Y-%m-%d %H:%M} · {_fmt_duration(row.duration_s)} · {status}"
            )
            if row.summary:
                st.write(row.summary)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**강점**")
                for item in row.strengths or ["—"]:
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("**약점·주의**")
                for item in row.weaknesses or ["—"]:
                    st.markdown(f"- {item}")
            if row.error:
                st.error(row.error)
            with st.expander("설정·지표·산출물"):
                st.code(json.dumps({"config": row.config, "metrics": row.metrics, "artifacts": row.artifacts,
                                    "git_commit": row.git_commit, "host": row.host},
                                   ensure_ascii=False, indent=2, default=str), language="json")
