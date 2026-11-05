"""Small UI helpers shared by pages (Streamlit-version tolerant)."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .state import theme_mode
from .theme import FONT, SEVERITY_LABEL, SEVERITY_STATUS, STATUS, template_name, tokens

_MODERN_WIDTH = inspect.signature(st.plotly_chart).parameters.get("width") is not None and (
    inspect.signature(st.plotly_chart).parameters["width"].default == "stretch"
)


def chart(fig: go.Figure, *, height: int | None = None, key: str | None = None) -> None:
    mode = theme_mode()
    t = tokens(mode)
    # Streamlit's frontend injects its own theme colors even with theme=None,
    # so set the chart chrome explicitly; explicit properties beat any template.
    fig.update_layout(
        template=template_name(mode),
        paper_bgcolor=t.surface,
        plot_bgcolor=t.surface,
        font=dict(family=FONT, color=t.text_primary),
    )
    fig.update_xaxes(gridcolor=t.grid, linecolor=t.axis, tickcolor=t.axis, zeroline=False,
                     tickfont=dict(color=t.muted), title_font=dict(color=t.text_secondary), automargin=True)
    fig.update_yaxes(gridcolor=t.grid, linecolor=t.axis, tickcolor=t.axis, zeroline=False,
                     tickfont=dict(color=t.muted), title_font=dict(color=t.text_secondary), automargin=True)
    if height is not None:
        fig.update_layout(height=height)
    config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    if _MODERN_WIDTH:
        st.plotly_chart(fig, theme=None, config=config, key=key)
    else:  # Streamlit < 1.45
        st.plotly_chart(fig, use_container_width=True, theme=None, config=config, key=key)


def table(frame: pd.DataFrame, *, height: int | None = None, column_config: dict | None = None) -> None:
    kwargs = {"hide_index": True, "column_config": column_config or {}}
    if height is not None:
        kwargs["height"] = height
    if not _MODERN_WIDTH:
        kwargs["use_container_width"] = True
    st.dataframe(frame, **kwargs)


_METRIC_HAS_BORDER = "border" in inspect.signature(st.metric).parameters


def kpi_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Row of stat tiles: (label, value, help)."""

    extra = {"border": True} if _METRIC_HAS_BORDER else {}
    for col, (label, value, help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text, **extra)


def severity_text(severity: str) -> str:
    """Severity as text plus a colored glyph; the label carries meaning, not the color."""

    status = SEVERITY_STATUS.get(severity, "good")
    return f":{_streamlit_color(status)}[■] {SEVERITY_LABEL.get(severity, severity)}"


def _streamlit_color(status: str) -> str:
    return {"critical": "red", "warning": "orange", "serious": "orange", "good": "green"}[status]


def status_color(severity: str) -> str:
    return STATUS[SEVERITY_STATUS.get(severity, "good")]


def empty_state(title: str, body: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)


def missing_overview_notice() -> None:
    st.warning(
        "아직 데이터 개요가 없습니다. 서버에서 먼저 실행하세요:\n\n"
        "```bash\nald-build-overview --data-root <CSV 폴더> --exsitu <Ex-situ 워크북.xlsx>\n```"
    )
