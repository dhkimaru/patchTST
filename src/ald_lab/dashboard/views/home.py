"""Landing page: research goals and progress of the four reporting stages."""

from __future__ import annotations

import streamlit as st

from ...journal import STAGE_LABELS
from ..state import journal_frame, lab_paths, overview

STAGE_CARDS = [
    ("1. 원본 데이터", ("data",), "run·컬럼·공정 구조·두께 측정·품질 이슈"),
    ("2. 전처리", ("preprocess",), "컬럼 선별, 구간 분할, 정규화, 학습/검증/시험 분할"),
    ("3. 하이퍼파라미터·학습", ("hpo", "train"), "설정을 골라 학습 실행, 진행 상황, 모든 조합의 누적 기록"),
    ("4. 타임라인·결과", ("evaluate",), "소요 시간, 모델의 강점·약점을 시간순으로"),
]


def render() -> None:
    st.title("ALD Lab")
    st.markdown(
        "PEALD SiN 공정 trace 데이터로 두 가지를 연구합니다.\n\n"
        "- **가상 센서**: 어떤 센서가 빠져도 나머지 신호로 그 값을 추정 (센서를 더 달지 않아도 단 것과 같은 효과)\n"
        "- **두께 예측 (Virtual Metrology)**: 공정 trace로 Ex-situ 두께·균일도를 예측\n\n"
        "아래 네 단계는 보고 순서(원본 설명 → 전처리 → 하이퍼파라미터·학습 → 소요 시간·강점·약점)를 따릅니다."
    )
    reports = sorted((lab_paths().work_root / "reports").glob("ald_report_2*.html"))
    if reports:
        latest = reports[-1]
        st.download_button(f"보고서 내려받기 (HTML, {latest.stem[-13:]})", latest.read_bytes(), file_name=latest.name,
                           mime="text/html", help="노트북에서 인터넷 없이 열리는 파일 하나. 새로 만들기: bash scripts/make_report.sh")
    journal = journal_frame()
    cols = st.columns(len(STAGE_CARDS))
    for col, (title, stages, body) in zip(cols, STAGE_CARDS):
        done = journal[journal["stage"].isin(stages) & (journal["status"] == "ok")] if len(journal) else journal
        with col.container(border=True):
            st.markdown(f"**{title}**")
            if len(done):
                last = done["started_at"].max()
                st.markdown(f":green[■] 기록 {len(done)}건")
                st.caption(f"최근 {last.tz_convert(None):%Y-%m-%d %H:%M} UTC · {body}")
            else:
                st.markdown(":gray[■] 아직 기록 없음")
                st.caption(body)

    tables = overview()
    summary = tables.get("summary")
    if summary:
        st.subheader("데이터 한눈에")
        st.markdown(
            f"- run **{summary['n_runs']}개** ({summary['first_start'][:10]} ~ {summary['last_end'][:10]}), "
            f"총 **{summary['total_hours']:.0f}시간**, {summary['n_rows']:,}행\n"
            f"- 신호 컬럼 **{summary['n_signal_columns']}개** 중 사용 가능 "
            f"**{summary['column_status'].get('OK', 0)}개**\n"
            f"- Ex-situ 두께 측정과 연결된 run **{summary['exsitu_linked_runs']}개**"
        )

    if len(journal):
        st.subheader("최근 기록")
        recent = journal.sort_values("started_at", ascending=False).head(5)
        for row in recent.itertuples():
            stage = STAGE_LABELS.get(row.stage, row.stage)
            duration = f"{row.duration_s:,.0f}초" if row.duration_s == row.duration_s else "-"
            st.markdown(
                f"- `{row.started_at.tz_convert(None):%m-%d %H:%M}` **{row.title}** · {stage} · {duration} · {row.status}"
            )
