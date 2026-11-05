"""Small inline-SVG charts for the HTML report (no JavaScript library, works offline).

Colors come from CSS variables defined by the page (light and dark), so the
charts follow the page theme. One value axis per chart.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np


def _num(value: float) -> str:
    return f"{value:.1f}"


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _tick_label(value: float, step: float | None = None) -> str:
    """Enough decimals to tell neighbouring ticks apart (99.95 vs 100.00 on a narrow axis)."""

    if value == 0:
        return "0"
    decimals = max(0, -math.floor(math.log10(step)) + (1 if step and f"{step:g}".endswith("5") else 0)) if step else 0
    if not step and abs(value) < 1:
        return f"{value:.3g}"
    return f"{value:,.{min(decimals, 6)}f}"


def _step(ticks: list[float]) -> float | None:
    return ticks[1] - ticks[0] if len(ticks) > 1 else None


def improvement_bars(rows: list[tuple], *, clip: float = 100.0, title: str = "", unit: str = "%") -> str:
    """Horizontal bars of improvement over a reference in % (right = better, left = worse; ``unit`` "%p" for
    differences of improvements).

    ``rows`` are ``(label, value)`` or ``(label, value, muted)``; muted bars (a difference within the
    seed spread) are drawn gray and say so — ``muted`` may also be the word to print instead of
    "불분명" (e.g. "반복 없음"). Values beyond ``±clip`` are drawn to the edge with their exact value printed.
    """

    rows = [(r[0], r[1], (r[2] if isinstance(r[2], str) else bool(r[2])) if len(r) > 2 else False) for r in rows
            if r[1] is not None and math.isfinite(r[1])]
    if not rows:
        return ""
    label_w = int(min(max(max(text_width(r[0]) for r in rows) + 16, 160), 400))
    value_w, plot_w, row_h, top = 110, 380, 22, 24
    width = label_w + plot_w + value_w
    height = top + row_h * len(rows) + 8
    lo = max(min(0.0, min(r[1] for r in rows)), -clip)
    hi = min(max(0.0, max(r[1] for r in rows)), clip)
    if hi - lo < 10:
        hi = lo + 10 if hi <= 0 else hi
        lo = min(lo, hi - 10)
    scale = plot_w / (hi - lo)
    zero = label_w + (0 - lo) * scale
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(title or "개선율")}">']
    ticks = nice_ticks(lo, hi, 5)
    for tick in ticks:
        if lo - 1e-9 <= tick <= hi + 1e-9:
            x = label_w + (tick - lo) * scale
            parts.append(f'<line class="grid" x1="{_num(x)}" y1="{top - 6}" x2="{_num(x)}" y2="{height - 6}"/>')
            parts.append(f'<text class="tick" x="{_num(x)}" y="{top - 10}" text-anchor="middle">'
                         f'{_tick_label(tick, _step(ticks))} {unit}</text>')
    for i, (label, value, muted) in enumerate(rows):
        y = top + i * row_h
        shown = max(min(value, clip), -clip)
        x0, x1 = sorted((zero, label_w + (shown - lo) * scale))
        cls = "bar-muted" if muted else ("bar-good" if value >= 0 else "bar-bad")
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 15}" text-anchor="end">{escape(_clip_text(label, 52))}</text>')
        parts.append(f'<rect class="{cls}" x="{_num(x0)}" y="{y + 4}" width="{_num(max(x1 - x0, 1))}" height="{row_h - 8}" rx="2"/>')
        word = (muted if isinstance(muted, str) else "불분명") if muted else ("개선" if value >= 0 else "악화")
        text = f"{value:+.1f} {unit} {word}" + (" ◀" if value < -clip else (" ▶" if value > clip else ""))
        parts.append(f'<text class="value" x="{label_w + plot_w + 6}" y="{y + 15}">{escape(text)}</text>')
    parts.append(f'<line class="axis" x1="{_num(zero)}" y1="{top - 6}" x2="{_num(zero)}" y2="{height - 6}"/>')
    parts.append("</svg>")
    return "".join(parts)


def value_bars(rows: list[tuple[str, float, str]], *, unit: str = "", title: str = "") -> str:
    """Horizontal bars of values on one linear axis from 0 (smaller is better, e.g. MAE).

    ``rows`` are ``(label, value, kind)``; kind ``base`` is drawn gray, ``prev`` in the second series
    colour and ``now`` in the first, so a page can say "기준선 · 이전 · 이번" in its legend.
    """

    rows = [r for r in rows if r[1] is not None and math.isfinite(r[1])]
    if not rows:
        return ""
    label_w = int(min(max(max(text_width(r[0]) for r in rows) + 16, 160), 420))
    value_w, plot_w, row_h, top = 90, 380, 24, 24
    width = label_w + plot_w + value_w
    height = top + row_h * len(rows) + 8
    hi = max(r[1] for r in rows) or 1.0
    ticks = [t for t in nice_ticks(0.0, hi, 5) if 0 <= t <= hi * 1.0001]
    scale = plot_w / (ticks[-1] if ticks and ticks[-1] >= hi else hi)
    parts = [(f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
              f'aria-label="{escape(title or "값")}">')]
    for tick in ticks:
        x = label_w + tick * scale
        parts.append(f'<line class="grid" x1="{_num(x)}" y1="{top - 6}" x2="{_num(x)}" y2="{height - 6}"/>')
        parts.append(f'<text class="tick" x="{_num(x)}" y="{top - 10}" text-anchor="middle">'
                     f'{_tick_label(tick, _step(ticks))}</text>')
    cls = {"base": "bar-muted", "prev": "bar-prev", "now": "bar-good"}
    for i, (label, value, kind) in enumerate(rows):
        y = top + i * row_h
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 16}" text-anchor="end">{escape(_clip_text(label, 56))}</text>')
        parts.append(f'<rect class="{cls.get(kind, "bar-good")}" x="{label_w}" y="{y + 4}" '
                     f'width="{_num(max(value * scale, 1))}" height="{row_h - 8}" rx="2"/>')
        text = f"{value:.3g} {unit}".strip()
        parts.append(f'<text class="value" x="{_num(label_w + value * scale + 6)}" y="{y + 16}">{escape(text)}</text>')
    parts.append(f'<line class="axis" x1="{label_w}" y1="{top - 6}" x2="{label_w}" y2="{height - 6}"/>')
    parts.append("</svg>")
    return "".join(parts)


def text_width(text: str, size: float = 12) -> float:
    """Rough rendered width: Hangul and other wide glyphs take about 1 em, the rest about 0.58 em."""

    return sum(size if ord(ch) >= 0x2E80 else size * 0.58 for ch in text)


def _clip_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def line_chart(t_minutes: np.ndarray, series: list[tuple[str, np.ndarray, str]], *, unit: str = "",
               shade: np.ndarray | None = None, shade_label: str = "ALD 사이클 구간", title: str = "") -> str:
    """Lines over time (minutes) on one value axis; ``shade`` marks rows to tint (e.g. ALD cycles)."""

    t = np.asarray(t_minutes, dtype=float)
    if len(t) < 2:
        return ""
    left, right, top, bottom, width, height = 64, 12, 30, 34, 960, 280
    plot_w, plot_h = width - left - right, height - top - bottom
    values = np.concatenate([np.asarray(v, dtype=float) for _, v, _ in series])
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    lo, hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    t0, t1 = float(t.min()), float(t.max())
    sx = plot_w / max(t1 - t0, 1e-9)
    sy = plot_h / (hi - lo)

    def px(tv: float) -> float:
        return left + (tv - t0) * sx

    def py(v: float) -> float:
        return top + plot_h - (min(max(v, lo), hi) - lo) * sy

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(title)}">']
    if shade is not None and np.asarray(shade).any():
        mask = np.asarray(shade, dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            x0, x1 = px(t[a]), px(t[min(b, len(t)) - 1])
            parts.append(f'<rect class="shade" x="{_num(x0)}" y="{top}" width="{_num(max(x1 - x0, 1))}" height="{plot_h}"/>')
    yticks = nice_ticks(lo, hi, 5)
    for tick in yticks:
        if lo <= tick <= hi:
            y = py(tick)
            parts.append(f'<line class="grid" x1="{left}" y1="{_num(y)}" x2="{width - right}" y2="{_num(y)}"/>')
            parts.append(f'<text class="tick" x="{left - 6}" y="{_num(y + 4)}" text-anchor="end">'
                         f'{_tick_label(tick, _step(yticks))}</text>')
    xticks = nice_ticks(t0, t1, 7)
    for tick in xticks:
        if t0 <= tick <= t1:
            x = px(tick)
            parts.append(f'<text class="tick" x="{_num(x)}" y="{height - bottom + 16}" text-anchor="middle">'
                         f'{_tick_label(tick, _step(xticks))}</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 4}" text-anchor="end">시간 (분)</text>')
    if unit:
        parts.append(f'<text class="tick" x="{left - 6}" y="{top - 12}" text-anchor="end">{escape(unit)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>')
    for name, v, cls in series:
        v = np.asarray(v, dtype=float)
        points = " ".join(f"{_num(px(a))},{_num(py(b))}" for a, b in zip(t, v) if math.isfinite(b))
        parts.append(f'<polyline class="{cls}" points="{points}"/>')
    x = left
    for name, _, cls in series:
        parts.append(f'<line class="{cls}" x1="{x}" y1="14" x2="{x + 18}" y2="14"/>')
        parts.append(f'<text class="label" x="{x + 24}" y="18">{escape(name)}</text>')
        x += 24 + 14 * len(name) + 20
    if shade is not None and np.asarray(shade).any():
        parts.append(f'<rect class="shade" x="{x}" y="7" width="18" height="12"/>')
        parts.append(f'<text class="label" x="{x + 24}" y="18">{escape(shade_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def loss_curve(history: list[dict], best_epoch: int | None, *, title: str = "") -> str:
    """Train and validation loss per epoch on one axis (log scale when they span > 20×), with the best epoch."""

    rows = [h for h in history if h.get("epoch") is not None]
    if len(rows) < 2:
        return ""
    epochs = np.array([float(h["epoch"]) for h in rows])
    series = [("학습 손실", np.array([float(h.get("train_loss", np.nan)) for h in rows]), "ln-true"),
              ("검증 손실", np.array([float(h.get("val_loss", np.nan)) for h in rows]), "ln-pred")]
    values = np.concatenate([v for _, v, _ in series])
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return ""
    log = values.max() / values.min() > 20
    tf = np.log10 if log else (lambda v: v)
    lo, hi = float(tf(values.min())), float(tf(values.max()))
    pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 or 1.0)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom, width, height = 64, 16, 30, 34, 640, 230
    plot_w, plot_h = width - left - right, height - top - bottom
    e0, e1 = float(epochs.min()), float(epochs.max())

    def px(ep: float) -> float:
        return left + (ep - e0) / max(e1 - e0, 1e-9) * plot_w

    def py(v: float) -> float:
        return top + plot_h - (min(max(float(tf(v)), lo), hi) - lo) / (hi - lo) * plot_h

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(title or "학습 곡선")}">']
    if log:
        ticks = [10.0 ** k for k in range(math.floor(lo), math.ceil(hi) + 1)]
        labels = [f"{t:g}" for t in ticks]
    else:
        ticks = nice_ticks(lo, hi, 5)
        labels = [_tick_label(t, _step(ticks)) for t in ticks]
    for t, text in zip(ticks, labels):
        if lo <= float(tf(t)) <= hi:
            y = py(t)
            parts.append(f'<line class="grid" x1="{left}" y1="{_num(y)}" x2="{width - right}" y2="{_num(y)}"/>')
            parts.append(f'<text class="tick" x="{left - 6}" y="{_num(y + 4)}" text-anchor="end">{escape(text)}</text>')
    xticks = [t for t in nice_ticks(e0, e1, 6) if e0 <= t <= e1 and float(t).is_integer()]
    for t in xticks:
        parts.append(f'<text class="tick" x="{_num(px(t))}" y="{height - bottom + 16}" text-anchor="middle">{t:g}</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 4}" text-anchor="end">epoch'
                 + (" · 세로축 로그" if log else "") + "</text>")
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>')
    if best_epoch:
        x = px(float(best_epoch))
        parts.append(f'<line class="mark" x1="{_num(x)}" y1="{top}" x2="{_num(x)}" y2="{top + plot_h}"/>')
        anchor = "end" if x > left + plot_w * 0.7 else "start"
        parts.append(f'<text class="marklabel" x="{_num(x + (-4 if anchor == "end" else 4))}" y="{top + 12}" '
                     f'text-anchor="{anchor}">최고 epoch {best_epoch}</text>')
    for _, v, cls in series:
        points = " ".join(f"{_num(px(a))},{_num(py(b))}" for a, b in zip(epochs, v) if math.isfinite(b) and b > 0)
        parts.append(f'<polyline class="{cls}" points="{points}"/>')
    x = left
    for name, _, cls in series:
        parts.append(f'<line class="{cls}" x1="{x}" y1="14" x2="{x + 18}" y2="14"/>')
        parts.append(f'<text class="label" x="{x + 24}" y="18">{escape(name)}</text>')
        x += 24 + 14 * len(name) + 20
    parts.append("</svg>")
    return "".join(parts)


def timeline_chart(bars: list[tuple[str, float, float, str, str]], lanes: list[tuple[str, str]], *,
                   title: str = "") -> str:
    """Jobs over time, one lane per device (a Gantt chart). ``bars`` are ``(lane, start_min, end_min, color,
    tooltip)``; ``lanes`` are ``(lane, label)`` from top to bottom. One axis: minutes since the first job started."""

    if not bars or not lanes:
        return ""
    label_w, right, top, lane_h, bottom = 64, 16, 10, 30, 34
    width = 900
    plot_w = width - label_w - right
    height = top + lane_h * len(lanes) + bottom
    end = max(b[2] for b in bars)
    ticks = nice_ticks(0, end, 7)
    hi = max(end, ticks[-1] if ticks else end) or 1.0
    row = {lane: i for i, (lane, _) in enumerate(lanes)}

    def px(t: float) -> float:
        return label_w + t / hi * plot_w

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(title or "작업 타임라인")}">']
    for t in ticks:
        if 0 <= t <= hi:
            parts.append(f'<line class="grid" x1="{_num(px(t))}" y1="{top}" x2="{_num(px(t))}" '
                         f'y2="{top + lane_h * len(lanes)}"/>')
            parts.append(f'<text class="tick" x="{_num(px(t))}" y="{top + lane_h * len(lanes) + 16}" '
                         f'text-anchor="middle">{_tick_label(t, _step(ticks))}</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 4}" text-anchor="end">분 (첫 작업 시작부터)</text>')
    for lane, label in lanes:
        y = top + row[lane] * lane_h
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + lane_h / 2 + 4:.1f}" text-anchor="end">'
                     f"{escape(label)}</text>")
    for lane, start, stop, color, tip in bars:
        if lane not in row:
            continue
        y = top + row[lane] * lane_h + 5
        x0, x1 = px(start), px(stop)
        parts.append(f'<rect class="tl" x="{_num(x0)}" y="{y}" width="{_num(max(x1 - x0, 1.5))}" height="{lane_h - 10}" '
                     f'rx="2" style="fill:{color}"><title>{escape(tip)}</title></rect>')
    parts.append(f'<line class="axis" x1="{label_w}" y1="{top + lane_h * len(lanes)}" x2="{width - right}" '
                 f'y2="{top + lane_h * len(lanes)}"/>')
    parts.append("</svg>")
    return "".join(parts)


def series_chart(series: list[dict], *, bands: list[tuple[float, float, str]] = (),
                 band_legend: list[tuple[str, str]] = (), marks: list[tuple[float, str]] = (), log: bool = False,
                 xlabel: str = "시간 (분)", unit: str = "", title: str = "", height: int = 260) -> str:
    """Series over a shared x on one value axis: each ``{"name", "x", "y", "cls", "kind"}`` with kind ``line`` or
    ``dots``. ``bands`` tint x-ranges ``(x0, x1, css class)``; ``marks`` are labelled dashed verticals. ``log``
    draws the value axis in powers of ten (a pressure that spans 1 to 760 Torr)."""

    xs = np.concatenate([np.asarray(s["x"], dtype=float) for s in series]) if series else np.array([])
    ys = np.concatenate([np.asarray(s["y"], dtype=float) for s in series]) if series else np.array([])
    ok = np.isfinite(ys) & (ys > 0 if log else True)
    if len(xs) < 2 or not ok.any():
        return ""
    left, right, top, bottom, width = 64, 12, 34, 34, 960
    plot_w, plot_h = width - left - right, height - top - bottom
    x0, x1 = float(xs.min()), float(xs.max())
    if log:
        lo, hi = math.floor(math.log10(float(ys[ok].min()))), math.ceil(math.log10(float(ys[ok].max())))
        hi = max(hi, lo + 1)
        tf = math.log10
    else:
        lo, hi = float(ys[ok].min()), float(ys[ok].max())
        pad = (hi - lo) * 0.08 or 1.0
        lo, hi = lo - pad, hi + pad
        tf = float

    def px(v: float) -> float:
        return left + (v - x0) / max(x1 - x0, 1e-9) * plot_w

    def py(v: float) -> float:
        value = tf(max(v, 10.0 ** lo)) if log else v
        return top + plot_h - (min(max(value, lo), hi) - lo) / (hi - lo) * plot_h

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
             f'aria-label="{escape(title)}">']
    for b0, b1, cls in bands:
        a, b = px(max(b0, x0)), px(min(b1, x1))
        parts.append(f'<rect class="{cls}" x="{_num(a)}" y="{top}" width="{_num(max(b - a, 1))}" height="{plot_h}"/>')
    ticks = [10.0 ** k for k in range(int(lo), int(hi) + 1)] if log else nice_ticks(lo, hi, 5)
    for tick in ticks:
        if (lo <= math.log10(tick) <= hi) if log else (lo <= tick <= hi):
            y = py(tick)
            parts.append(f'<line class="grid" x1="{left}" y1="{_num(y)}" x2="{width - right}" y2="{_num(y)}"/>')
            text = f"{tick:g}" if log else _tick_label(tick, _step(ticks))
            parts.append(f'<text class="tick" x="{left - 6}" y="{_num(y + 4)}" text-anchor="end">{text}</text>')
    xticks = nice_ticks(x0, x1, 8)
    for tick in xticks:
        if x0 <= tick <= x1:
            parts.append(f'<text class="tick" x="{_num(px(tick))}" y="{height - bottom + 16}" text-anchor="middle">'
                         f'{_tick_label(tick, _step(xticks))}</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 4}" text-anchor="end">{escape(xlabel)}'
                 + (" · 세로축 로그" if log else "") + "</text>")
    if unit:
        parts.append(f'<text class="tick" x="{left - 6}" y="{top - 8}" text-anchor="end">{escape(unit)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>')
    for at, label in marks:
        if x0 <= at <= x1:
            x = px(at)
            parts.append(f'<line class="mark" x1="{_num(x)}" y1="{top}" x2="{_num(x)}" y2="{top + plot_h}"/>')
            parts.append(f'<text class="marklabel" x="{_num(x + 4)}" y="{top + 12}">{escape(label)}</text>')
    for s in series:
        pts = [(a, b) for a, b in zip(np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float))
               if math.isfinite(b) and (b > 0 or not log)]
        if s.get("kind") == "dots":
            parts.append("".join(f'<circle class="{s["cls"]}" cx="{_num(px(a))}" cy="{_num(py(b))}" r="1.8"/>'
                                 for a, b in pts))
        else:
            parts.append(f'<polyline class="{s["cls"]}" points="'
                         + " ".join(f"{_num(px(a))},{_num(py(b))}" for a, b in pts) + '"/>')
    x = left
    for s in series:
        if s.get("kind") == "dots":
            parts.append(f'<circle class="{s["cls"]}" cx="{x + 9}" cy="14" r="2.5"/>')
        else:
            parts.append(f'<line class="{s["cls"]}" x1="{x}" y1="14" x2="{x + 18}" y2="14"/>')
        parts.append(f'<text class="label" x="{x + 24}" y="18">{escape(s["name"])}</text>')
        x += 24 + int(text_width(s["name"])) + 20
    for cls, label in band_legend:
        parts.append(f'<rect class="{cls}" x="{x}" y="7" width="14" height="14"/>')
        parts.append(f'<text class="label" x="{x + 20}" y="18">{escape(label)}</text>')
        x += 20 + int(text_width(label)) + 20
    parts.append("</svg>")
    return "".join(parts)
