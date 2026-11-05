"""Color and type tokens shared by the dashboard (Plotly) and the HTML report (no Plotly import).

Validated reference palette, light + dark. Categorical order is fixed (never cycled or re-ranked);
status colors are reserved for state and always paired with a text label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokens:
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]


LIGHT = Tokens(
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Tokens(
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#f0efec",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)
# De-emphasis gray for context series in emphasis charts.
CONTEXT_GRAY = {"light": "#c3c2b7", "dark": "#52514e"}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
SEVERITY_STATUS = {"critical": "critical", "warning": "warning", "info": "good"}
SEVERITY_LABEL = {"critical": "치명", "warning": "경고", "info": "정보"}

FONT = 'system-ui, -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif'


def tokens(mode: str) -> Tokens:
    return DARK if mode == "dark" else LIGHT
