"""Page 1: explain the raw data before any modelling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ...analysis.variability import CLASSES, CROSS_FEASIBLE_R2
from ...data.catalog import CYCLE_START_STEP
from ..state import analysis, lab_paths, overview, segments, theme_mode, trace
from ..theme import CONTEXT_GRAY, SEVERITY_LABEL, STATUS, tokens
from ..ui import chart, empty_state, kpi_row, missing_overview_notice, severity_text, table

MAX_POINTS_PER_TRACE = 4000
DEFAULT_SIGNALS = ["VG11", "MFCMon_DCS", "MFCMon_NH3"]
CYCLE_SIGNALS = ["VG11", "APCValveMon", "MFCMon_DCS", "MFCMon_NH3", "MFCMon_N2-3",
                 "MFCMon_F.PWR", "ValveAct_29", "ValveAct_89"]
DOE_FACTORS = [("dcs_slm", "DCS 유량 (SLM)"), ("nh3_slm", "NH3 유량 (SLM)"),
               ("rf_forward_power_w", "RF 파워 (W)"), ("temp_c", "공정 온도 (℃)")]


def render() -> None:
    st.title("1. 원본 데이터")
    tables = overview()
    summary = tables.get("summary")
    if not summary:
        missing_overview_notice()
        return
    tabs = st.tabs(["요약", "Run 목록", "공정 구조", "컬럼 사전", "신호 보기", "Ex-situ 두께", "가상 센서 대상",
                    "데이터 이슈"])
    with tabs[0]:
        _summary(tables)
    with tabs[1]:
        _runs(tables)
    with tabs[2]:
        _process_structure(tables)
    with tabs[3]:
        _column_dictionary(tables)
    with tabs[4]:
        _signal_viewer(tables)
    with tabs[5]:
        _exsitu(tables)
    with tabs[6]:
        _virtual_sensor_targets(tables)
    with tabs[7]:
        _issues(tables)


# ----------------------------------------------------------------- helpers


def _short_date(text: str) -> str:
    ts = pd.Timestamp(text)
    return f"{ts.month}/{ts.day}"


def _labels(tables) -> dict[str, str]:
    columns = tables["columns"]
    out = {}
    for row in columns.itertuples():
        unit = f" [{row.unit}]" if isinstance(row.unit, str) else ""
        name = f" · {row.name_ko}" if isinstance(row.name_ko, str) else ""
        out[row.column] = f"{row.column}{name}{unit}"
    return out


def _minmax_decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep each bucket's min and max so spikes survive downsampling."""

    n = len(x)
    if n <= max_points:
        return x, y
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    keep: list[int] = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        if b <= a or np.all(np.isnan(seg)):
            continue
        keep.extend(sorted({a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))}))
    idx = np.asarray(keep)
    return x[idx], y[idx]


def _run_severity(tables) -> pd.Series:
    issues = tables["issues"]
    order = {"critical": 2, "warning": 1, "info": 0}
    if issues is None or issues.empty:
        return pd.Series(dtype=object)
    worst = issues.assign(rank=issues["severity"].map(order)).sort_values("rank").groupby("run_id").tail(1)
    return worst.set_index("run_id")["severity"]


def _linked_blocks(tables) -> pd.DataFrame | None:
    blocks = tables.get("exsitu_blocks")
    if blocks is None:
        return None
    return blocks[blocks["run_id"].notna()].copy()


# -------------------------------------------------------------------- tabs


def _summary(tables) -> None:
    s = tables["summary"]
    status = s["column_status"]
    st.markdown(
        "PEALD SiN(DCS·NH3) 배치 장비의 **trace 데이터**입니다. CSV 파일 하나가 배치 **run 하나**이고, "
        f"0.5초마다 {s['n_signal_columns']}개 신호를 기록합니다. 한 run은 보트 로딩 → 승온·안정화 → "
        f"**ALD 사이클 반복** → 하강·언로딩 순서로 진행되며 중앙값 {s['median_run_hours']:.1f}시간 걸립니다. "
        "run마다 boat 5개 위치에서 웨이퍼 두께를 49점씩 측정한 **Ex-situ 데이터**가 연결됩니다."
    )
    kpi_row(
        [
            ("Run", f"{s['n_runs']}개", f"번호 누락: {', '.join(map(str, s['missing_run_numbers']))}"),
            ("기간", f"{_short_date(s['first_start'])}–{_short_date(s['last_end'])}",
             f"{s['first_start'][:10]} ~ {s['last_end'][:10]}"),
            ("데이터량", f"{s['total_hours']:.0f}시간", f"{s['n_rows']:,}행, 0.5초 간격"),
            ("신호 컬럼", f"{status.get('OK', 0)} / {s['n_signal_columns']}",
             f"상수 {status.get('CONSTANT', 0)}개, 완전 중복 {status.get('DUPLICATE', 0)}개 제외"),
            ("두께 연결", f"{s['exsitu_linked_runs']} run", f"Ex-situ 블록 {s['exsitu_blocks']}개 중"),
        ]
    )
    runs = tables["runs"]
    recipe_hours: dict[str, float] = {}
    for raw in runs["recipe_rows"]:
        for recipe, rows in json.loads(raw).items():
            recipe_hours[recipe] = recipe_hours.get(recipe, 0.0) + rows * 0.5 / 3600
    hours = pd.Series(recipe_hours).sort_values()
    t = tokens(theme_mode())
    fig = go.Figure(
        go.Bar(x=hours.values, y=hours.index, orientation="h", marker_color=t.series[0],
               text=[f"{v:,.0f}h" for v in hours.values], textposition="outside",
               hovertemplate="%{y}: %{x:,.1f}시간<extra></extra>")
    )
    fig.update_layout(title="레시피별 누적 시간 (전체 run 합계)", xaxis_title="시간 (h)", yaxis_title=None,
                      showlegend=False)
    chart(fig, height=320)
    st.caption(
        "가장 긴 레시피가 본 공정(대기·승온·안정화·하강 포함)이고, 나머지는 ALD 사이클을 이루는 하위 레시피입니다."
    )


def _runs(tables) -> None:
    runs = tables["runs"].copy()
    severity = _run_severity(tables)
    runs["severity"] = runs["run_id"].map(severity).fillna("ok")
    blocks = _linked_blocks(tables)
    if blocks is not None:
        runs = runs.merge(blocks[["run_id", "thk_mean", "uf_pct_mean", "title"]], on="run_id", how="left")
    issues = tables["issues"]
    kinds = issues.groupby("run_id")["kind"].agg(", ".join) if issues is not None and len(issues) else {}
    runs["issues"] = runs["run_id"].map(kinds).fillna("")

    labels = {"ok": "정상", "info": "참고", "warning": "경고", "critical": "치명"}
    colors = {"ok": STATUS["good"], "info": CONTEXT_GRAY[theme_mode()], "warning": STATUS["warning"],
              "critical": STATUS["critical"]}
    fig = go.Figure()
    for key in ("ok", "info", "warning", "critical"):
        part = runs[runs["severity"] == key]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["start"], y=part["duration_h"], mode="markers", name=labels[key],
                marker=dict(color=colors[key], size=10),
                customdata=np.stack([part["run_id"], part["n_cycles"], part["issues"]], axis=1),
                hovertemplate="%{customdata[0]}<br>시작 %{x|%m-%d %H:%M}<br>길이 %{y:.2f}h"
                              "<br>사이클 %{customdata[1]}회<br>%{customdata[2]}<extra></extra>",
            )
        )
    fig.update_layout(title="Run 시작 시각과 길이 (색 = 가장 심한 이슈)", xaxis_title="시작 시각",
                      yaxis_title="run 길이 (h)")
    chart(fig, height=360)
    view = runs[["run_id", "start", "duration_h", "rows", "n_cycles", "main_recipe", "thk_mean",
                 "uf_pct_mean", "severity", "issues"]].copy()
    view["severity"] = view["severity"].map(labels)
    table(
        view,
        height=420,
        column_config={
            "run_id": "Run",
            "start": st.column_config.DatetimeColumn("시작", format="YYYY-MM-DD HH:mm"),
            "duration_h": st.column_config.NumberColumn("길이 (h)", format="%.2f"),
            "rows": st.column_config.NumberColumn("행 수", format="%d"),
            "n_cycles": "ALD 사이클",
            "main_recipe": "주 레시피",
            "thk_mean": st.column_config.NumberColumn("평균 두께 (Å)", format="%.1f"),
            "uf_pct_mean": st.column_config.NumberColumn("U/F (%)", format="%.2f"),
            "severity": "상태",
            "issues": "이슈",
        },
    )


def _process_structure(tables) -> None:
    st.markdown(
        "한 run 안에서 ALD 사이클은 **SI-FL1(DCS 공급) → SI-EVA1(배기) → CHANGE → N-PRE1 → "
        "N-FL1/N-FL2(NH3 + 플라즈마) → N-EVA1(배기) → CLOSE1** 순서로 반복됩니다. "
        "아래에서 run과 사이클을 골라 한 사이클 동안 신호가 어떻게 움직이는지 볼 수 있습니다."
    )
    runs = tables["runs"]
    paths = lab_paths()
    c1, c2, c3 = st.columns([2, 1, 1])
    run_id = c1.selectbox("Run", runs["run_id"].tolist(), index=min(18, len(runs) - 1), key="cycle_run")
    seg = segments(run_id, str(paths.trace_cache))
    cycle_starts = seg.index[seg["step_name"] == CYCLE_START_STEP].tolist()
    if not cycle_starts:
        empty_state("사이클 없음", "이 run에는 SI-FL1 step이 없습니다.")
        return
    cycle_no = c2.number_input("사이클 번호", min_value=1, max_value=len(cycle_starts), value=min(20, len(cycle_starts)))
    n_show = c3.number_input("표시할 사이클 수", min_value=1, max_value=5, value=2)
    first = cycle_starts[cycle_no - 1]
    last_idx = cycle_no - 1 + n_show
    end_ts = seg.at[cycle_starts[last_idx], "start"] if last_idx < len(cycle_starts) else seg.iloc[-1]["end"]
    start_ts = seg.at[first, "start"]
    frame = trace(run_id, str(paths.trace_cache))
    window = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)]
    window_seg = seg[(seg["start"] >= start_ts) & (seg["start"] < end_ts)]
    labels = _labels(tables)
    signals = [s for s in CYCLE_SIGNALS if s in window.columns]
    t = tokens(theme_mode())
    x = (window["timestamp"] - start_ts).dt.total_seconds()
    fig = make_subplots(rows=len(signals) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.6] + [1.0] * len(signals))
    shades = [CONTEXT_GRAY[theme_mode()], t.grid]
    for i, s in enumerate(window_seg.itertuples()):
        x0 = (s.start - start_ts).total_seconds()
        width = max(s.duration_s, 0.5)
        fig.add_trace(go.Bar(x=[width], y=["step"], base=[x0], orientation="h", marker_color=shades[i % 2],
                             text=s.step_name if width >= 6 else "", textposition="inside",
                             insidetextanchor="middle", textfont=dict(color=t.text_primary, size=11),
                             hovertemplate=f"{s.step_name} · {s.recipe}<br>{x0:.1f}s ~ {x0 + width:.1f}s<extra></extra>",
                             showlegend=False), row=1, col=1)
    for r, signal in enumerate(signals, start=2):
        fig.add_trace(go.Scatter(x=x, y=window[signal], mode="lines", line=dict(color=t.series[0], width=2),
                                 name=signal, showlegend=False,
                                 hovertemplate=f"{signal}: %{{y:.4g}}<br>%{{x:.1f}}s<extra></extra>"),
                      row=r, col=1)
        fig.update_yaxes(title_text=signal, title_font=dict(size=11), row=r, col=1)
    for s in window_seg.itertuples():
        if s.step_name == CYCLE_START_STEP:
            x0 = (s.start - start_ts).total_seconds()
            fig.add_vline(x=x0, line_width=1, line_color=t.axis)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="사이클 시작 후 경과 시간 (s)", row=len(signals) + 1, col=1)
    fig.update_layout(title=f"{run_id} · 사이클 {cycle_no}부터 {n_show}개", bargap=0.0, barmode="overlay",
                      height=150 + 110 * len(signals), margin=dict(l=110))
    chart(fig)
    st.caption(" · ".join(labels.get(s, s) for s in signals))

    st.subheader("Step 목록 (전체 run 합계)")
    steps = tables["steps"].copy()
    table(
        steps,
        height=360,
        column_config={
            "recipe": "레시피", "step_id": "Step ID", "step_name": "Step",
            "occurrences": st.column_config.NumberColumn("등장 횟수", format="%d"),
            "runs": "run 수",
            "total_rows": st.column_config.NumberColumn("총 행 수", format="%d"),
            "median_duration_s": st.column_config.NumberColumn("1회 길이 중앙값 (s)", format="%.1f"),
        },
    )


def _column_dictionary(tables) -> None:
    columns = tables["columns"].copy()
    s = tables["summary"]
    status = s["column_status"]
    kpi_row(
        [
            ("전체 신호", f"{len(columns)}", None),
            ("사용 가능", f"{status.get('OK', 0)}", "상수·완전 중복이 아닌 컬럼"),
            ("상수", f"{status.get('CONSTANT', 0)}", "모든 run에서 같은 값"),
            ("완전 중복", f"{status.get('DUPLICATE', 0)}", "다른 컬럼과 모든 값이 동일"),
            ("근사 복사 쌍", f"{s['n_near_copy_pairs']}", "|r| ≥ 0.999 (예: PressAct ≈ VG11)"),
            ("핵심(A) 대표", f"{len(s['core_columns_A'])}", "중요도 A 파라미터의 대표 컬럼"),
        ]
    )
    st.caption(
        "중요도(A/B/C)는 현업 파라미터 자료의 항목을 컬럼에 대응시킨 것입니다. "
        "**대응 신뢰도**: inferred_high/low는 데이터 패턴으로 추정한 값이라 현업 확인이 필요합니다."
    )
    c1, c2, c3, c4 = st.columns(4)
    groups = sorted(columns["group"].dropna().unique())
    pick_groups = c1.multiselect("그룹", groups, placeholder="전체")
    pick_importance = c2.multiselect("중요도", ["A", "B", "C", "미지정"], placeholder="전체")
    pick_status = c3.multiselect("상태", ["OK", "CONSTANT", "DUPLICATE"], placeholder="전체")
    query = c4.text_input("검색", placeholder="컬럼명 또는 설명")
    view = columns
    if pick_groups:
        view = view[view["group"].isin(pick_groups)]
    if pick_importance:
        imp = view["importance"].fillna("미지정")
        view = view[imp.isin(pick_importance)]
    if pick_status:
        view = view[view["status"].isin(pick_status)]
    if query:
        text = (view["column"] + " " + view["name_ko"].fillna("") + " " + view["notes"].fillna("")).str.lower()
        view = view[text.str.contains(query.lower(), regex=False)]
    view = view[["column", "name_ko", "group", "unit", "param_id", "importance", "primary", "mapping",
                 "status", "duplicate_of", "min", "max", "mean", "std", "notes"]]
    table(
        view,
        height=480,
        column_config={
            "column": "컬럼", "name_ko": "설명", "group": "그룹", "unit": "단위", "param_id": "항목",
            "importance": "중요도", "primary": st.column_config.CheckboxColumn("대표"),
            "mapping": "대응 신뢰도", "status": "상태", "duplicate_of": "중복 대상",
            "min": st.column_config.NumberColumn("최소", format="%.4g"),
            "max": st.column_config.NumberColumn("최대", format="%.4g"),
            "mean": st.column_config.NumberColumn("평균", format="%.4g"),
            "std": st.column_config.NumberColumn("표준편차", format="%.4g"),
            "notes": "비고",
        },
    )
    params = tables.get("parameters")
    if params is not None:
        st.subheader("파라미터 설명서 항목 ↔ 대표 컬럼")
        primary = columns[columns["primary"]].groupby("param_id")["column"].agg(", ".join)
        related = columns.groupby("param_id")["column"].count()
        params = params.assign(대표_컬럼=params["param_id"].map(primary).fillna("—"),
                               관련_컬럼_수=params["param_id"].map(related).fillna(0).astype(int))
        table(params[["param_id", "name_ko", "name_en", "unit", "importance", "scope", "대표_컬럼", "관련_컬럼_수"]],
              column_config={"param_id": "항목", "name_ko": "이름", "name_en": "영문", "unit": "단위",
                             "importance": "중요도", "scope": "구분"})
    near = tables.get("near_copies")
    if near is not None and len(near):
        with st.expander(f"근사 복사 쌍 {len(near)}개 (|r| ≥ 0.999)"):
            table(near.sort_values("pearson_r", key=np.abs, ascending=False),
                  column_config={"column_a": "컬럼 A", "column_b": "컬럼 B",
                                 "pearson_r": st.column_config.NumberColumn("상관계수 r", format="%.6f")})


def _signal_viewer(tables) -> None:
    runs = tables["runs"]
    columns = tables["columns"]
    labels = _labels(tables)
    usable = columns.loc[columns["status"] == "OK", "column"].tolist()
    c1, c2 = st.columns([1, 2])
    pick_runs = c1.multiselect("Run (최대 3개)", runs["run_id"].tolist(), default=[runs["run_id"].iloc[18]],
                               max_selections=3)
    pick_cols = c2.multiselect("신호", usable, default=[c for c in DEFAULT_SIGNALS if c in usable],
                               format_func=lambda c: labels.get(c, c), max_selections=8)
    if not pick_runs or not pick_cols:
        empty_state("선택 필요", "run과 신호를 하나 이상 고르세요.")
        return
    paths = lab_paths()
    meta = runs.set_index("run_id")
    max_minutes = float(meta.loc[pick_runs, "duration_h"].max() * 60)
    window = st.slider("공정 시작(START step) 기준 시간 범위 (분)", -60.0, max_minutes, (0.0, min(max_minutes, 300.0)),
                       step=1.0)
    t = tokens(theme_mode())
    fig = make_subplots(rows=len(pick_cols), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for j, run_id in enumerate(pick_runs):
        frame = trace(run_id, str(paths.trace_cache))
        anchor = meta.loc[run_id, "process_start"]
        anchor = frame["timestamp"].iloc[0] if pd.isna(anchor) else anchor
        minutes = ((frame["timestamp"] - anchor).dt.total_seconds() / 60).to_numpy()
        mask = (minutes >= window[0]) & (minutes <= window[1])
        for r, col in enumerate(pick_cols, start=1):
            xs, ys = _minmax_decimate(minutes[mask], frame[col].to_numpy()[mask], MAX_POINTS_PER_TRACE)
            fig.add_trace(
                go.Scattergl(x=xs, y=ys, mode="lines", name=run_id, legendgroup=run_id, showlegend=r == 1,
                             line=dict(color=t.series[j], width=1.5),
                             hovertemplate=f"{run_id}<br>{col}: %{{y:.4g}}<br>%{{x:.1f}}분<extra></extra>"),
                row=r, col=1,
            )
            fig.update_yaxes(title_text=col, title_font=dict(size=11), row=r, col=1)
    fig.update_xaxes(title_text="공정 시작 후 경과 시간 (분)", row=len(pick_cols), col=1)
    fig.update_layout(height=160 + 190 * len(pick_cols), title="신호 비교 (구간마다 최소·최대를 보존해 표시)",
                      margin=dict(l=110), showlegend=len(pick_runs) > 1)
    chart(fig)
    st.caption(" · ".join(labels.get(c, c) for c in pick_cols))


def _layout_maps() -> go.Figure:
    """Where the 9-point and 49-point layouts measure on a 300 mm wafer."""

    t = tokens(theme_mode())
    fig = make_subplots(rows=1, cols=2, subplot_titles=["9점 (DOE 시트)", "49점 (STANDARD 시트)"])
    theta = np.linspace(0, 2 * np.pi, 200)
    rings = {"9pt": {0.0: 1, 70.0: 4, 145.0: 4}, "49pt": {0.0: 1, 46.7: 8, 93.3: 16, 140.0: 24}}
    for c, layout in enumerate(("9pt", "49pt"), start=1):
        fig.add_trace(go.Scatter(x=150 * np.cos(theta), y=150 * np.sin(theta), mode="lines",
                                 line=dict(color=t.axis, width=1), hoverinfo="skip", showlegend=False), row=1, col=c)
        xs, ys, text = [], [], []
        for radius, count in rings[layout].items():
            for k in range(count):
                angle = np.pi / 2 + 2 * np.pi * k / count
                xs.append(radius * np.cos(angle))
                ys.append(radius * np.sin(angle))
                text.append(f"r = {radius:g} mm")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=8, color=t.series[0]),
                                 text=text, hovertemplate="%{text}<extra></extra>", showlegend=False), row=1, col=c)
        fig.update_xaxes(scaleanchor=f"y{c if c > 1 else ''}", range=[-160, 160], showticklabels=False,
                         showgrid=False, row=1, col=c)
        fig.update_yaxes(range=[-160, 160], showticklabels=False, showgrid=False, row=1, col=c)
    return fig


def _exsitu(tables) -> None:
    blocks = tables.get("exsitu_blocks")
    points = tables.get("exsitu_points")
    radial = tables.get("exsitu_radial")
    if blocks is None:
        empty_state("Ex-situ 데이터 없음", "개요 생성 시 --exsitu 로 Ex-situ 워크북 경로를 주세요.")
        return
    t = tokens(theme_mode())
    counts = blocks["layout"].value_counts()
    st.markdown(
        "워크북의 각 블록은 run 하나이고, boat의 **5개 위치(WAP, 보통 slot 11·36·61·86·110)** 웨이퍼를 측정했습니다. "
        "웨이퍼 한 장의 측정점 배치는 **두 종류**입니다. 측정 반경이 달라서 평균·U/F를 배치끼리 그대로 비교하면 안 됩니다. "
        "아래 값은 모두 원시 측정점에서 다시 계산했습니다(U/F = 범위 ÷ (2 × 평균) × 100)."
    )
    kpi_row(
        [
            ("49점 블록", f"{counts.get('49pt', 0)}", "STANDARD 시트 · 반경 0 / 46.7 / 93.3 / 140 mm"),
            ("9점 블록", f"{counts.get('9pt', 0)}", "DOE 시트 · 반경 0 / 70 / 145 mm"),
            ("측정값 없음", f"{counts.get('none', 0)}", "조건 기록만 있고 두께가 비어 있는 블록"),
            ("trace 연결", f"{blocks['run_id'].notna().sum()} run", "모두 49점 배치"),
        ]
    )
    chart(_layout_maps(), height=320)

    st.subheader("STANDARD 연속 run · 49점 (trace와 연결)")
    linked = _linked_blocks(tables).sort_values("run_id")
    linked["run_no"] = linked["run_id"].str[-3:].astype(int)
    issues = tables["issues"]
    outlier_runs = set(issues.loc[issues["kind"] == "THICKNESS_OUTLIER", "run_id"]) if issues is not None else set()
    normal = linked[~linked["run_id"].isin(outlier_runs)]
    kpi_row(
        [
            ("두께 중앙값", f"{normal['thk_mean'].median():.1f} Å", "이상 run 제외, boat 5위치 평균"),
            ("U/F 중앙값", f"{normal['uf_pct_mean'].median():.2f} %", "웨이퍼 내 균일도 (낮을수록 균일)"),
            ("WtoW 중앙값", f"{normal['wtow_pct'].median():.2f} %", "boat 위치 간 균일도"),
            ("가장자리 − 중심", f"{normal['edge_vs_center_pct'].median():+.2f} %", "r=140 mm 링 평균 대비 중심"),
            ("두께 이상 run", f"{len(outlier_runs)}", "같은 배치 안에서 robust z ≥ 5"),
        ]
    )
    show_outliers = st.toggle("이상 run 포함", value=False)
    view = linked if show_outliers else normal
    # Reindex over every run number so missing or excluded runs break the line
    # instead of being bridged by a straight segment.
    full = pd.RangeIndex(linked["run_no"].min(), linked["run_no"].max() + 1, name="run_no")
    view = view.set_index("run_no").reindex(full).reset_index()
    by_wap = st.toggle("boat 위치별로 보기", value=False)
    slots = str(linked["wap_slots"].dropna().iloc[0]).split(",") if linked["wap_slots"].notna().any() else []
    wap_names = [f"slot {slots[w - 1].strip()}" if len(slots) >= w else f"WAP {w}" for w in range(1, 6)]
    fig = go.Figure()
    if by_wap:
        for w in range(1, 6):
            fig.add_trace(go.Scatter(x=view["run_no"], y=view[f"thk_mean_wap{w}"], mode="lines+markers",
                                     name=wap_names[w - 1], line=dict(color=t.series[w - 1]),
                                     marker=dict(size=6, color=t.series[w - 1]),
                                     hovertemplate=f"run %{{x}} · {wap_names[w - 1]}: %{{y:.2f}}Å<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=view["run_no"], y=view["thk_mean"], mode="lines+markers", name="평균 두께",
                                 line=dict(color=t.series[0]), marker=dict(size=7, color=t.series[0]),
                                 customdata=view[["title"]].to_numpy(),
                                 hovertemplate="run %{x}: %{y:.2f}Å<br>%{customdata[0]}<extra></extra>",
                                 showlegend=False))
    loop = linked["loop"]
    for i in linked.index[loop.diff().fillna(0) != 0]:
        run_no, before = linked.at[i, "run_no"], loop.shift().at[i]
        fig.add_vline(x=run_no - 0.5, line_width=1, line_color=t.axis)
        fig.add_annotation(x=run_no - 0.5, y=1, yref="paper", text=f"LOOP {before:g}→{loop.at[i]:g}", showarrow=False,
                           font=dict(color=t.muted, size=11), xanchor="left", yanchor="top")
    fig.update_layout(title="Run별 평균 두께 (49점)", xaxis_title="run 번호", yaxis_title="두께 (Å)")
    chart(fig, height=360)

    fig = go.Figure(go.Scatter(x=view["run_no"], y=view["uf_pct_mean"], mode="lines+markers",
                               line=dict(color=t.series[0]), marker=dict(size=7, color=t.series[0]),
                               hovertemplate="run %{x}: U/F %{y:.2f}%<extra></extra>", showlegend=False))
    fig.update_layout(title="Run별 웨이퍼 내 균일도 U/F (49점, boat 5위치 평균)", xaxis_title="run 번호",
                      yaxis_title="U/F (%)")
    chart(fig, height=300)

    c1, c2 = st.columns(2)
    run_pick = c1.selectbox("Run", linked["run_id"].tolist(), index=min(18, len(linked) - 1), key="map_run")
    block = linked[linked["run_id"] == run_pick].iloc[0]
    wap = c2.selectbox("Boat 위치 (웨이퍼 맵)", [1, 2, 3, 4, 5], format_func=lambda w: wap_names[w - 1])
    left, right = st.columns(2)
    if radial is not None:
        prof = radial[radial["block_id"] == block["block_id"]]
        fig = go.Figure()
        for w in range(1, 6):
            part = prof[prof["wap_index"] == w].sort_values("radius_mm")
            fig.add_trace(go.Scatter(x=part["radius_mm"], y=part["thk_mean"], mode="lines+markers",
                                     name=wap_names[w - 1], line=dict(color=t.series[w - 1]),
                                     marker=dict(size=7, color=t.series[w - 1]),
                                     hovertemplate=f"{wap_names[w - 1]} · r=%{{x}}mm: %{{y:.2f}}Å<extra></extra>"))
        fig.update_layout(title=f"{run_pick} · 반경별 평균 두께 (중심 → 가장자리)", xaxis_title="반경 (mm)",
                          yaxis_title="두께 (Å)")
        with left:
            chart(fig, height=420)
    if points is not None and len(points):
        pts = points[(points["block_id"] == block["block_id"]) & (points["wap_index"] == wap)]
        theta = np.linspace(0, 2 * np.pi, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=150 * np.cos(theta), y=150 * np.sin(theta), mode="lines",
                                 line=dict(color=t.axis, width=1), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=pts["x_mm"], y=pts["y_mm"], mode="markers", showlegend=False,
            marker=dict(size=16, color=pts["thickness_a"], colorscale=[[0, t.sequential[0]], [1, t.sequential[-1]]],
                        colorbar=dict(title="Å"), line=dict(width=1, color=t.surface)),
            customdata=pts[["point"]].to_numpy(),
            hovertemplate="point %{customdata[0]} (%{x}, %{y})<br>%{marker.color:.2f}Å<extra></extra>"))
        fig.update_layout(title=f"웨이퍼 맵 · {wap_names[wap - 1]}", xaxis=dict(scaleanchor="y", title="x (mm)"),
                          yaxis=dict(title="y (mm)"))
        with right:
            chart(fig, height=420)

    st.subheader("DOE · 9점 (trace 파일 없음)")
    doe = blocks[(blocks["layout"] == "9pt") & blocks["thk_mean"].notna()].copy()
    if len(doe):
        st.caption(
            "다른 조건이 표준(블록에서 가장 많이 쓰인 값)일 때 한 조건만 바꾼 블록입니다. "
            "9점 배치라 위의 49점 값과 직접 비교하지 않습니다."
        )
        # 표준 조건: 블록마다 한 조건만 바꾸므로 조건별로 가장 많이 쓰인 값
        standard = {f: float(doe[f].astype(float).mode().iloc[0]) for f in [*(f for f, _ in DOE_FACTORS), "loop"]}
        doe["sheet_label"] = np.where(doe["sheet"].str.startswith("1."), "DOE1", "DOE2 (ToF 설치 후)")
        fig = make_subplots(rows=1, cols=4, subplot_titles=[label for _, label in DOE_FACTORS], shared_yaxes=True)
        for c, (factor, _label) in enumerate(DOE_FACTORS, start=1):
            others = [f for f, _ in DOE_FACTORS if f != factor]
            mask = (doe["loop"] == standard["loop"]) & doe["rf_forward_power_w"].notna()
            for other in others:
                mask &= np.isclose(doe[other].astype(float), standard[other])
            sub = doe[mask].sort_values(factor)
            for k, (sheet_label, part) in enumerate(sub.groupby("sheet_label")):
                fig.add_trace(go.Scatter(x=part[factor], y=part["thk_mean"], mode="lines+markers", name=sheet_label,
                                         legendgroup=sheet_label, showlegend=c == 1,
                                         line=dict(color=t.series[k]), marker=dict(size=8, color=t.series[k]),
                                         hovertemplate=f"{sheet_label}<br>%{{x}} → %{{y:.1f}}Å<extra></extra>"),
                              row=1, col=c)
        fig.update_yaxes(title_text="평균 두께 (Å, 9점)", row=1, col=1)
        chart(fig, height=340)

    st.subheader("배치별 요약")
    measured = blocks[blocks["layout"].isin(["9pt", "49pt"])]
    summary = (
        measured.groupby("layout")
        .agg(blocks=("block_id", "size"), thk_mean=("thk_mean", "median"), thk_center=("thk_center", "median"),
             thk_edge=("thk_edge", "median"), edge_radius=("edge_radius_mm", "first"),
             edge_vs_center=("edge_vs_center_pct", "median"), uf=("uf_pct_mean", "median"),
             wtow=("wtow_pct", "median"))
        .reset_index()
    )
    table(summary, column_config={
        "layout": "배치", "blocks": "블록 수",
        "thk_mean": st.column_config.NumberColumn("평균 두께 중앙값 (Å)", format="%.2f"),
        "thk_center": st.column_config.NumberColumn("중심 (Å)", format="%.2f"),
        "thk_edge": st.column_config.NumberColumn("가장자리 링 (Å)", format="%.2f"),
        "edge_radius": st.column_config.NumberColumn("가장자리 반경 (mm)", format="%.0f"),
        "edge_vs_center": st.column_config.NumberColumn("가장자리−중심 (%)", format="%+.2f"),
        "uf": st.column_config.NumberColumn("U/F 중앙값 (%)", format="%.2f"),
        "wtow": st.column_config.NumberColumn("WtoW 중앙값 (%)", format="%.2f"),
    })
    st.caption("조건(LOOP·온도 등)도 배치마다 달라서 이 표는 배치 간 차이를 보여 줄 뿐, 측정 방식의 효과만 분리한 것은 아닙니다.")


def _non_overlapping(frame: pd.DataFrame, x: str, y: str, min_dx: float = 0.18, min_dy: float = 0.06):
    """Rows whose labels would not collide (x on a log axis), most informative first."""

    placed: list[tuple[float, float]] = []
    for row in frame.sort_values(y, ascending=False).itertuples():
        lx, ly = float(np.log10(getattr(row, x))), float(getattr(row, y))
        if all(abs(lx - px) > min_dx or abs(ly - py) > min_dy for px, py in placed):
            placed.append((lx, ly))
            yield row


def _virtual_sensor_targets(tables) -> None:
    result = analysis()
    var = result.get("variability")
    summary = result.get("variability_summary")
    if var is None or summary is None:
        empty_state("아직 분석 없음", "서버에서 `ald-analyze-params`를 실행하면 이 탭이 채워집니다.")
        return
    st.markdown(
        "ALD 사이클 동안 각 신호를 **레시피 템플릿**(학습 run에서 step·step 내 위치별 평균)과 비교했습니다. "
        f"평가는 검증 run {summary['n_validation_runs']}개로 했고 시험 run은 열지 않았습니다.\n\n"
        "- **레시피 설명력**: 템플릿이 설명하는 변동 비율. 1에 가까우면 레시피를 그대로 따르는 신호라 템플릿 비교만으로 이상을 잡을 수 있습니다.\n"
        "- **다른 센서로 추정**: 템플릿으로 설명되지 않는 잔차를, 대상과 **같은 계열 센서를 모두 뺀 상태**에서 "
        "다른 센서의 잔차로 얼마나 설명하는지(선형). 센서를 달지 않았다고 가정한 값입니다."
    )
    counts = summary["class_counts"]
    kpi_row([(name.replace("공정 응답 · ", ""), f"{counts.get(name, 0)}", None) for name in CLASSES])
    measured = var[var["role"].fillna("measurement") == "measurement"]
    measured = measured[measured["class"].isin(CLASSES[:3])]
    t = tokens(theme_mode())
    # x: share of variation the recipe does NOT explain, on a log axis so the
    # many near-recipe signals (R² 0.94–0.99) spread out instead of piling up.
    measured = measured.assign(unexplained=(1 - measured["r2_recipe"]).clip(0.004, 1.5),
                               cross=measured["r2_cross"].clip(lower=-0.1))
    short = measured[measured["shortlist"]]
    rest = measured[~measured["shortlist"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rest["unexplained"], y=rest["cross"], mode="markers", name="기타 측정 신호",
        marker=dict(size=7, color=CONTEXT_GRAY[theme_mode()]), customdata=rest[["column", "r2_recipe"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>레시피 R² %{customdata[1]:.3f} · 다른 센서 %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=short["unexplained"], y=short["cross"], mode="markers", name="1차 후보",
        marker=dict(size=10, color=t.series[0]), customdata=short[["column", "name_ko", "r2_recipe"]].to_numpy(),
        hovertemplate="%{customdata[0]} · %{customdata[1]}<br>레시피 R² %{customdata[2]:.3f} · 다른 센서 %{y:.2f}"
                      "<extra></extra>"))
    for row in _non_overlapping(short, "unexplained", "cross"):
        fig.add_annotation(x=np.log10(row.unexplained), y=row.cross, text=row.column, showarrow=False, yshift=12,
                           font=dict(size=11, color=t.text_secondary))
    fig.add_vline(x=0.01, line_width=1, line_color=t.axis)
    fig.add_hline(y=CROSS_FEASIBLE_R2, line_width=1, line_color=t.axis)
    fig.add_annotation(x=np.log10(0.01), y=1.04, text="← 레시피 추종 (R² ≥ 0.99)", showarrow=False, xanchor="right",
                       font=dict(size=11, color=t.muted))
    fig.update_layout(title="레시피로 설명되지 않는 변동 vs 같은 계열 없이 다른 센서로 설명되는 잔차 (검증 run)",
                      xaxis=dict(type="log", title="레시피로 설명되지 않는 변동 비율 1 − R² (로그 축, 오른쪽일수록 레시피와 무관)"),
                      yaxis=dict(title="다른 센서로 잔차 설명 R²", range=[-0.12, 1.08]))
    chart(fig, height=540)
    st.caption("이름이 겹치는 점은 생략했습니다. 마우스를 올리거나 아래 표에서 확인하세요.")

    st.subheader(f"1차 가상 센서 후보 {len(short)}개 (계열별 대표)")
    st.caption(
        "후보 = 설명서 중요도 A·B 파라미터와 압력(VG11)·APC 밸브·RF 매칭의 대표 컬럼 중, 사이클 동안 레시피만으로 설명되지 않는 신호. "
        "'근사 복사만 제외'는 같은 계열 센서가 남아 있을 때의 값입니다 (예: VG11은 다른 압력계가 있으면 쉽게 추정됨)."
    )
    view = short[["column", "name_ko", "importance", "family", "class", "r2_recipe", "resid_rel", "run_drift",
                  "r2_cross", "r2_cross_copy", "top_partner", "top_partner_r"]]
    table(view, column_config={
        "column": "컬럼", "name_ko": "설명", "importance": "중요도", "family": "계열", "class": "분류",
        "r2_recipe": st.column_config.NumberColumn("레시피 설명력", format="%.3f"),
        "resid_rel": st.column_config.NumberColumn("잔차/신호 폭", format="%.2f"),
        "run_drift": st.column_config.NumberColumn("run 간 변동", format="%.3f",
                                                   help="run별 잔차 평균의 표준편차 ÷ 신호 표준편차"),
        "r2_cross": st.column_config.NumberColumn("다른 센서 (계열 제외)", format="%.3f"),
        "r2_cross_copy": st.column_config.NumberColumn("다른 센서 (근사 복사만 제외)", format="%.3f"),
        "top_partner": "가장 관련 큰 신호",
        "top_partner_r": st.column_config.NumberColumn("상관", format="%+.2f"),
    })
    with st.expander(f"전체 {len(var)}개 컬럼"):
        pick = st.multiselect("분류", list(CLASSES), placeholder="전체", key="var_class")
        full = var if not pick else var[var["class"].isin(pick)]
        table(full[["column", "name_ko", "family", "importance", "role", "class", "r2_recipe", "resid_rel",
                    "run_drift", "r2_cross", "r2_cross_copy"]], height=420)


def _issues(tables) -> None:
    issues = tables["issues"]
    if issues is None or issues.empty:
        empty_state("이슈 없음", "점검 결과 문제가 발견되지 않았습니다.")
        return
    counts = issues["severity"].value_counts()
    kpi_row([(SEVERITY_LABEL[k], f"{counts.get(k, 0)}건", None) for k in ("critical", "warning", "info")])
    for severity in ("critical", "warning", "info"):
        part = issues[issues["severity"] == severity]
        if part.empty:
            continue
        st.markdown(f"**{severity_text(severity)}**")
        table(part[["run_id", "kind", "detail"]],
              column_config={"run_id": "Run", "kind": "종류", "detail": "내용"})
    st.caption(
        "치명: 학습 데이터에서 제외하거나 보정해야 함 · 경고: 사용 전 확인 필요 · 정보: 알고 있으면 되는 사항"
    )
