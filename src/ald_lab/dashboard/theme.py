"""Chart tokens and the Plotly template (validated reference palette, light + dark).

Categorical order is fixed (never cycled or re-ranked); status colors are
reserved for state and always paired with a text label.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from .tokens import (  # noqa: F401  (re-exported for the views)
    CONTEXT_GRAY,
    DARK,
    FONT,
    LIGHT,
    SEVERITY_LABEL,
    SEVERITY_STATUS,
    STATUS,
    Tokens,
    tokens,
)


def build_template(mode: str) -> go.layout.Template:
    t = tokens(mode)
    axis = dict(
        gridcolor=t.grid,
        gridwidth=1,
        linecolor=t.axis,
        linewidth=1,
        zeroline=False,
        ticks="outside",
        tickcolor=t.axis,
        tickfont=dict(color=t.muted, size=12),
        title=dict(font=dict(color=t.text_secondary, size=12)),
        showspikes=False,
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, color=t.text_primary, size=13),
            paper_bgcolor=t.surface,
            plot_bgcolor=t.surface,
            colorway=list(t.series),
            xaxis=axis,
            yaxis=axis,
            # Title pinned to the container top; legend sits between it and the plot.
            legend=dict(font=dict(color=t.text_secondary, size=12), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=t.surface,
                            bordercolor=t.axis, font_color=t.text_primary),
            margin=dict(l=56, r=24, t=76, b=48),
            title=dict(font=dict(size=15, color=t.text_primary), x=0, xanchor="left",
                       y=0.985, yref="container", yanchor="top"),
            bargap=0.35,
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=2, color=t.surface)))],
            scattergl=[go.Scattergl(line=dict(width=2), marker=dict(size=8))],
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
        ),
    )


def register_templates() -> None:
    pio.templates["ald_light"] = build_template("light")
    pio.templates["ald_dark"] = build_template("dark")


def template_name(mode: str) -> str:
    return "ald_dark" if mode == "dark" else "ald_light"
