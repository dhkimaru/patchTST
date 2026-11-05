"""Flow diagrams as inline SVG, no library: a call tree drawn left to right (caller → callee, the calls one
function makes stacked top to bottom in the order the code makes them) and a small graph on a fixed grid.

Colours come from CSS classes (``g-<group>``) so light and dark themes both work; every colour is named in a
text legend next to the diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Callable

from .charts import text_width
from .codemap import CallTree

BOX_W, BOX_H, GAP_X, GAP_Y, PAD = 220, 36, 48, 8, 6


@dataclass
class Box:
    title: str
    sub: str
    group: str
    tip: str
    root: bool = False
    badge: str = ""


def _fit(text: str, limit: int, *, keep_end: bool = False) -> str:
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):] if keep_end else text[: limit - 1] + "…"


def _box(x: float, y: float, box: Box, width: int = BOX_W) -> str:
    cls = f"node g-{box.group}" + (" root" if box.root else "")
    chars = (width - 18) // 7 - (2 if box.badge else 0)
    badge = (f'<text class="nx" x="{x + width - 7:.0f}" y="{y + 15:.0f}" text-anchor="end">{escape(box.badge)}</text>'
             if box.badge else "")
    return (f'<g class="{cls}"><title>{escape(box.tip)}</title>'
            f'<rect class="nb" x="{x:.0f}" y="{y:.0f}" width="{width}" height="{BOX_H}" rx="6"/>'
            f'<rect class="ns" x="{x:.0f}" y="{y:.0f}" width="5" height="{BOX_H}" rx="2"/>'
            f'<text class="nt" x="{x + 12:.0f}" y="{y + 15:.0f}">{escape(_fit(box.title, chars))}</text>'
            f'<text class="nm" x="{x + 12:.0f}" y="{y + 29:.0f}">{escape(_fit(box.sub, chars + 4, keep_end=True))}</text>'
            f"{badge}</g>")


def _head(x: float, y: float, direction: str = "right", cls: str = "ah") -> str:
    shape = {"right": "l-7,-4 v8 z", "left": "l7,-4 v8 z", "down": "l-4,-7 h8 z", "up": "l-4,7 h8 z"}[direction]
    return f'<path class="{cls}" d="M{x:.1f},{y:.1f} {shape}"/>'


def _svg(width: float, height: float, body: list[str], label: str) -> str:
    return (f'<div class="diagram"><svg role="img" aria-label="{escape(label)}" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">{"".join(body)}</svg></div>')


def call_tree(tree: CallTree, describe: Callable[[str], Box], *, label: str = "") -> str:
    """Draw ``tree``: a parent sits level with its first callee, the rest follow underneath. Calls to a function
    already drawn elsewhere (``tree.extra``) are not drawn as lines; the caller gets a ↪ badge instead."""

    pos: dict[str, tuple[int, int]] = {}
    rows = 0

    def place(key: str, depth: int) -> None:
        nonlocal rows
        pos[key] = (depth, rows)
        kids = tree.children.get(key, [])
        if not kids:
            rows += 1
        for kid in kids:
            place(kid, depth + 1)

    place(tree.root, 0)
    columns = max(d for d, _ in pos.values()) + 1

    def x(depth: int) -> float:
        return PAD + depth * (BOX_W + GAP_X)

    def y(row: int) -> float:
        return PAD + row * (BOX_H + GAP_Y)

    body = []
    for key, kids in tree.children.items():
        if not kids or key not in pos:
            continue
        d, r = pos[key]
        x1, y1 = x(d) + BOX_W, y(r) + BOX_H / 2
        xm = x1 + GAP_X / 2
        for kid in kids:
            kd, kr = pos[kid]
            x2, y2 = x(kd), y(kr) + BOX_H / 2
            body.append(f'<path class="eg" d="M{x1:.0f},{y1:.0f}H{xm:.0f}V{y2:.0f}H{x2 - 6:.0f}"/>')
            body.append(_head(x2, y2))
    again: dict[str, int] = {}
    for caller, _ in tree.extra:
        again[caller] = again.get(caller, 0) + 1
    for key, (d, r) in pos.items():
        box = describe(key)
        if key in again:
            box.badge = "↪" if again[key] == 1 else f"↪{again[key]}"
        body.append(_box(x(d), y(r), box))
    width = 2 * PAD + columns * BOX_W + (columns - 1) * GAP_X
    height = 2 * PAD + rows * (BOX_H + GAP_Y) - GAP_Y
    return _svg(width, height, body, label)


def grid_graph(places: dict[str, tuple[float, float]], edges: dict[tuple[str, str], int],
               describe: Callable[[str], Box], *, titles: list[str] | None = None, footer: Box | None = None,
               label: str = "", box_w: int = 200, gap_x: int = 110, gap_y: int = 30) -> str:
    """Nodes at fixed (column, row) places; edges to the right are solid, edges back to the left dashed, edges
    within a column vertical. The number on an edge is ``edges[(a, b)]``."""

    top = 22 if titles else 0
    n_cols = int(max(c for c, _ in places.values())) + 1
    n_rows = max(r for _, r in places.values()) + 1

    def at(n: str) -> tuple[float, float]:
        c, r = places[n]
        return PAD + c * (box_w + gap_x), PAD + top + r * (BOX_H + gap_y)

    body, labels = [], []
    for i, title in enumerate(titles or []):
        labels.append(f'<text class="ct" x="{PAD + i * (box_w + gap_x) + box_w / 2:.0f}" y="{PAD + 12}" '
                      f'text-anchor="middle">{escape(title)}</text>')
    for (a, b), count in edges.items():
        if a not in places or b not in places or a == b:
            continue
        (xa, ya), (xb, yb) = at(a), at(b)
        ca, cb = places[a][0], places[b][0]
        if ca == cb:  # same column: straight down (or up)
            down = yb > ya
            x1 = x2 = xa + box_w / 2
            y1, y2 = (ya + BOX_H, yb) if down else (ya, yb + BOX_H)
            body.append(f'<path class="eg" d="M{x1:.0f},{y1:.0f}V{y2 - 6 if down else y2 + 6:.0f}"/>')
            body.append(_head(x2, y2, "down" if down else "up"))
            mx, my, t = x1 + 4, (y1 + y2) / 2, "start"
        else:
            back = cb < ca
            if back:
                x1, y1, x2, y2 = xa, ya + BOX_H / 2 + 10, xb + box_w, yb + BOX_H / 2 + 10
            else:
                x1, y1, x2, y2 = xa + box_w, ya + BOX_H / 2, xb, yb + BOX_H / 2
            c1, c2 = (x1 + x2) / 2, (x1 + x2) / 2
            tip = x2 + 6 if back else x2 - 6
            body.append(f'<path class="{"ex" if back else "eg"}" d="M{x1:.0f},{y1:.0f}C{c1:.0f},{y1:.0f} '
                        f'{c2:.0f},{y2:.0f} {tip:.0f},{y2:.0f}"/>')
            body.append(_head(x2, y2, "left" if back else "right", "ah ahx" if back else "ah"))
            # label 60 % of the way along, where edges leaving the same box have spread apart
            s = 0.6
            mx = (1 - s) ** 3 * x1 + 3 * (1 - s) ** 2 * s * c1 + 3 * (1 - s) * s * s * c2 + s ** 3 * x2
            my = (1 - s) ** 3 * y1 + 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s * s * y2 + s ** 3 * y2
            t = "middle"
        labels.append(f'<text class="el" x="{mx:.0f}" y="{my + 4:.0f}" text-anchor="{t}">{count}</text>')
    for n in places:
        x, y = at(n)
        body.append(_box(x, y, describe(n), box_w))
    width = 2 * PAD + n_cols * box_w + (n_cols - 1) * gap_x
    height = PAD + top + n_rows * (BOX_H + gap_y) - gap_y + PAD
    if footer:
        y = height + gap_y / 2 - PAD
        body.append(_box(PAD, y, footer, int(width - 2 * PAD)))
        height = y + BOX_H + PAD
    return _svg(width, height, body + labels, label)


# ----------------------------------------------------------------------------- illustrations (특이사항 tab)

def _card(x: float, y: float, w: float, h: float, head: str, sub: str = "", cls: str = "now") -> str:
    lines = [f'<g class="node g-{cls}"><rect class="nb" x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="6"/>',
             f'<rect class="ns" x="{x:.0f}" y="{y:.0f}" width="5" height="{h:.0f}" rx="2"/>']
    if sub:
        lines.append(f'<text class="nt" x="{x + 12:.0f}" y="{y + h / 2 - 3:.0f}">{escape(head)}</text>')
        lines.append(f'<text class="nm" x="{x + 12:.0f}" y="{y + h / 2 + 12:.0f}">{escape(sub)}</text>')
    else:
        lines.append(f'<text class="nt" x="{x + 12:.0f}" y="{y + h / 2 + 4:.0f}">{escape(head)}</text>')
    return "".join(lines) + "</g>"


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    if abs(y1 - y2) < 0.5:
        return f'<path class="eg" d="M{x1:.0f},{y1:.0f}H{x2 - 6:.0f}"/>' + _head(x2, y2)
    xm = (x1 + x2) / 2
    return (f'<path class="eg" d="M{x1:.0f},{y1:.0f}C{xm:.0f},{y1:.0f} {xm:.0f},{y2:.0f} {x2 - 6:.0f},{y2:.0f}"/>'
            + _head(x2, y2))


def step_lanes(lanes: list[tuple[str, str, list[tuple[str, str]], str]], *, label: str = "") -> str:
    """Rows of steps joined by arrows: ``(title, css group, [(head, sub)], note)`` per row."""

    w = max(max(text_width(h, 12.5), text_width(s, 11)) for _, _, steps, _ in lanes for h, s in steps) + 26
    gap, h, row = 30, 46, 98
    n = max(len(steps) for _, _, steps, _ in lanes)
    body = []
    for i, (title, cls, steps, note) in enumerate(lanes):
        y = PAD + i * row
        body.append(f'<text class="ct" x="{PAD}" y="{y + 12}">{escape(title)}</text>')
        for j, (head, sub) in enumerate(steps):
            x = PAD + j * (w + gap)
            body.append(_card(x, y + 20, w, h, head, sub, cls))
            if j:
                body.append(_arrow(x - gap + 2, y + 20 + h / 2, x, y + 20 + h / 2))
        if note:
            body.append(f'<text class="el" x="{PAD}" y="{y + 20 + h + 18}">{escape(note)}</text>')
    width = 2 * PAD + n * w + (n - 1) * gap
    return _svg(width, PAD + len(lanes) * row - 6, body, label)


def gather_figure(*, rows: int, channels: int, context: int, batch: int, patch: int, stride: int,
                  sample: float = 1.0, label: str = "") -> str:
    """How one batch is made on the GPU: the whole series → random window ends → one index gather → patches."""

    pw, gap, height = 196, 44, 232
    mb = rows * channels * 4 / 2**20
    body = []

    def panel(i: int, title: str, captions: tuple[str, str]) -> float:
        x = PAD + i * (pw + gap)
        body.append(f'<text class="ct" x="{x}" y="{PAD + 12}">{escape(title)}</text>')
        for k, text in enumerate(captions):
            body.append(f'<text class="nm" x="{x}" y="{height - 30 + 15 * k}">{escape(text)}</text>')
        if i:
            body.append(_arrow(x - gap + 6, 112, x - 6, 112))
        return x

    # ① the standardized series of every train run, stacked: rows = seconds, columns = signals
    x = panel(0, "① 시계열 전체를 GPU에", (f"{rows:,}행 × {channels}신호", f"float32 {mb:,.0f} MB · 처음에 한 번"))
    tx, ty, tw, th = x + 8, 30, 168, 150
    bands = 7
    for b in range(bands):
        body.append(f'<rect class="{"fa" if b % 2 else "fb"}" x="{tx}" y="{ty + b * th / bands:.1f}" width="{tw}" '
                    f'height="{th / bands:.1f}"/>')
    for c in range(1, 12):
        body.append(f'<line class="grid" x1="{tx + c * tw / 12:.1f}" y1="{ty}" x2="{tx + c * tw / 12:.1f}" y2="{ty + th}"/>')
    for b, frac in ((0, 0.25), (3, 0.45), (5, 0.2)):
        body.append(f'<rect class="win" x="{tx}" y="{ty + (b + frac) * th / bands:.1f}" width="{tw}" height="9"/>')
    body.append(f'<rect class="frame" x="{tx}" y="{ty}" width="{tw}" height="{th}"/>')
    body.append(f'<text class="el" x="{tx + tw - 4}" y="{ty + th / bands * 2 - 3:.0f}" text-anchor="end">run 경계</text>')
    # ② window ends drawn at random on the GPU
    x = panel(1, "② window 끝 위치 추첨", (f"{batch:,}개 · torch.randperm (GPU)", "run 경계를 넘는 위치는 미리 제외"))
    for k, text in enumerate(("52 310", "8 121", "640 022", "⋮", "113 905")):
        y = 36 + k * 26
        if text == "⋮":
            body.append(f'<text class="nt" x="{x + 60}" y="{y + 15}" text-anchor="middle">⋮</text>')
            continue
        body.append(f'<rect class="cell" x="{x + 8}" y="{y}" width="104" height="20" rx="3"/>'
                    f'<text class="nt" x="{x + 60}" y="{y + 14}" text-anchor="middle">{text}</text>')
    # ③ one gather: ends[:, None] + offsets → (batch, context, channels)
    x = panel(2, "③ 인덱스 한 번으로 batch", ("x[끝 + (−%d…0)] — GPU 연산 1번" % (context - 1),
                                              f"→ ({batch:,}, {context}, {channels}) 한 덩어리"))
    gx, gy = x + 8, 44
    for r in range(9):
        body.append(f'<line class="grid" x1="{gx}" y1="{gy + r * 14}" x2="{gx + 72}" y2="{gy + r * 14}"/>')
    for c in range(7):
        body.append(f'<line class="grid" x1="{gx + c * 12}" y1="{gy}" x2="{gx + c * 12}" y2="{gy + 112}"/>')
    body.append(f'<rect class="frame" x="{gx}" y="{gy}" width="72" height="112"/>')
    body.append(f'<text class="el" x="{gx + 36}" y="{gy - 6}" text-anchor="middle">인덱스 행렬</text>')
    body.append(_arrow(gx + 78, gy + 56, gx + 104, gy + 56))
    for k in range(3):
        body.append(f'<rect class="cube" x="{gx + 110 + 8 * (2 - k)}" y="{gy + 8 * k}" width="60" height="96" rx="2"/>')
    body.append(f'<text class="el" x="{gx + 148}" y="{gy + 124}" text-anchor="middle">batch</text>')
    # ④ inside the model: each window cut into patches (a view, no copy)
    n = (context + stride - patch) // stride + 1
    x = panel(3, "④ 모델 안에서 patch로", (f"{patch * sample:g}초 patch, {stride * sample:g}초 간격 → 토큰 {n}개",
                                         "unfold: 복사 없이 나눠 보기"))
    bx, by, bw = x + 8, 52, 176
    body.append(f'<rect class="win" x="{bx}" y="{by}" width="{bw}" height="14"/>')
    body.append(f'<text class="el" x="{bx + bw / 2}" y="{by - 6}" text-anchor="middle">window {context * sample:g}초</text>')
    scale = bw / (context + stride)
    shown = min(n, 12)
    for k in range(shown):
        px0 = bx + k * stride * scale
        body.append(f'<rect class="patch" x="{px0:.1f}" y="{by + 22 + (k % 2) * 9}" width="{patch * scale:.1f}" '
                    f'height="6" rx="1"/>')
    for k in range(min(n, 10)):
        body.append(f'<rect class="cell" x="{bx + k * 17.5:.1f}" y="{by + 66}" width="13" height="13" rx="2"/>')
    if n > 10:
        body.append(f'<text class="nt" x="{bx + 176}" y="{by + 77}" text-anchor="end">…</text>')
    body.append(f'<text class="el" x="{bx + bw / 2}" y="{by + 100}" text-anchor="middle">→ Transformer</text>')
    width = 2 * PAD + 4 * pw + 3 * gap
    return _svg(width, height, body, label)


def queue_figure(devices: list[str], *, n_jobs: int, label: str = "") -> str:
    """Jobs in one queue; one worker process per GPU takes the next job as soon as it is free."""

    gpus = devices or ["cuda:0"]
    h, row = 46, 56
    height = 2 * PAD + max(len(gpus), 3) * row
    mid = height / 2
    body = [_card(PAD, mid - h / 2, 170, h, "실험 YAML", f"→ 작업 {n_jobs:,}개 (보고서 전체)", "prev"),
            _card(PAD + 210, mid - h / 2, 170, h, "대기열 (queue)", "작업을 하나씩 꺼내 감", "now")]
    body.append(_arrow(PAD + 172, mid, PAD + 210, mid))
    top = mid - len(gpus) * row / 2
    for i, device in enumerate(gpus):
        y = top + i * row + (row - h) / 2
        name = device.replace("cuda:", "GPU ") if device.startswith("cuda") else device.upper()
        body.append(_card(PAD + 440, y, 240, h, f"{name} 작업자", "데이터 한 번 올림 → 작업 학습 → 다음 작업", "now"))
        body.append(_arrow(PAD + 382, mid, PAD + 440, y + h / 2))
        body.append(_arrow(PAD + 682, y + h / 2, PAD + 740, mid))
    body.append(_card(PAD + 740, mid - h / 2, 150, h, "결과 모음", "요약·진행 기록", "prev"))
    return _svg(2 * PAD + 890, height, body, label)


# ----------------------------------------------------------------------------- In tab

def count_flow(lanes: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]], *, label: str = "") -> str:
    """Rows of counts joined by arrows, what was removed written over each arrow: ``(title, [(value, caption)],
    [(drop line, drop note)])`` per row; the last box of a row (what is used) is outlined in the accent."""

    n = max(len(boxes) for _, boxes, _ in lanes)
    # each column (and each gap) as wide as its longest text, so a long last box does not widen the others
    box_w = [max(max(text_width(v, 15) * 1.05, text_width(c, 11.5)) for _, boxes, _ in lanes
                 for v, c in boxes[j:j + 1]) + 28 for j in range(n)]
    gap = [max([max(text_width(a, 11.5), text_width(b, 11)) for _, _, drops in lanes for a, b in drops[j:j + 1]]
               + [60]) + 28 for j in range(n - 1)]
    lefts = [PAD + sum(box_w[:j]) + sum(gap[:j]) for j in range(n)]
    h, row = 48, 104
    body = []
    for i, (title, boxes, drops) in enumerate(lanes):
        y = PAD + i * row
        body.append(f'<text class="ct" x="{PAD}" y="{y + 12}">{escape(title)}</text>')
        for j, (value, caption) in enumerate(boxes):
            x, w = lefts[j], box_w[j]
            cls = "nb nb-final" if j == len(boxes) - 1 else "nb"
            body.append(f'<g class="node"><rect class="{cls}" x="{x:.0f}" y="{y + 40:.0f}" width="{w:.0f}" '
                        f'height="{h}" rx="6"/><text class="cv" x="{x + w / 2:.0f}" y="{y + 61:.0f}" '
                        f'text-anchor="middle">{escape(value)}</text><text class="nm" x="{x + w / 2:.0f}" '
                        f'y="{y + 79:.0f}" text-anchor="middle">{escape(caption)}</text></g>')
            if j:
                x1, x2, ym = x - gap[j - 1] + 4, x - 4, y + 40 + h / 2
                body.append(f'<path class="eg" d="M{x1:.0f},{ym:.0f}H{x2 - 6:.0f}"/>' + _head(x2, ym))
                drop, note = drops[j - 1] if j - 1 < len(drops) else ("", "")
                body.append(f'<text class="dl" x="{(x1 + x2) / 2:.0f}" y="{ym - 8:.0f}" text-anchor="middle">'
                            f'{escape(drop)}</text>')
                if note:
                    body.append(f'<text class="nm" x="{(x1 + x2) / 2:.0f}" y="{ym + 18:.0f}" text-anchor="middle">'
                                f'{escape(note)}</text>')
    width = lefts[-1] + box_w[-1] + PAD
    return _svg(width, PAD + len(lanes) * row - 8, body, label)


def split_strip(cells: list[tuple[int, str, bool]], *, per_row: int = 20, label: str = "") -> str:
    """One square per run number: its split (css class ``cell-<status>``), a corner mark when ``flag``."""

    w, h, gap = 36, 28, 4
    rows = (len(cells) + per_row - 1) // per_row
    body = []
    for i, (number, status, flag) in enumerate(cells):
        x, y = PAD + (i % per_row) * (w + gap), PAD + (i // per_row) * (h + gap)
        tip = {"train": "학습", "validation": "검증", "test": "시험", "excluded": "제외", "missing": "파일 없음"}.get(status, status)
        body.append(f'<g><title>run {number:03d}: {escape(tip)}{" · 두께 예측에서 제외" if flag else ""}</title>'
                    f'<rect class="cell-{status}" x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>'
                    f'<text class="{"cnum-muted" if status == "missing" else "cnum"}" x="{x + w / 2}" y="{y + 18}" '
                    f'text-anchor="middle">{number:03d}</text>'
                    + (f'<path class="cflag" d="M{x + w - 10},{y} h10 v10 z"/>' if flag else "") + "</g>")
    width = 2 * PAD + per_row * (w + gap) - gap
    return _svg(width, 2 * PAD + rows * (h + gap) - gap, body, label)


def window_rule(context: float, horizon: float, *, label: str = "") -> str:
    """A window and its farthest target must sit inside one continuous segment (no gap, no run boundary)."""

    seg_a, gap_x, seg_b = (40, 430), (430, 482), (482, 740)
    scale = 220 / max(context + horizon, 1e-9)  # window + horizon always 220 px wide
    y0 = 70
    body = [f'<text class="ct" x="{PAD}" y="18">window와 예측 시점이 모두 한 구간 안에 있을 때만 학습·평가에 씁니다</text>',
            f'<rect class="seg" x="{seg_a[0]}" y="{y0}" width="{seg_a[1] - seg_a[0]}" height="26" rx="4"/>',
            f'<text class="nm" x="{(seg_a[0] + seg_a[1]) / 2}" y="{y0 + 17}" text-anchor="middle">구간 1: 끊김 없는 연속 기록</text>',
            f'<rect class="seg-gap" x="{gap_x[0] + 3}" y="{y0}" width="{gap_x[1] - gap_x[0] - 6}" height="26" rx="4"/>',
            f'<text class="nm" x="{(gap_x[0] + gap_x[1]) / 2}" y="{y0 - 6}" text-anchor="middle">끊김·run 경계</text>',
            f'<rect class="seg" x="{seg_b[0]}" y="{y0}" width="{seg_b[1] - seg_b[0]}" height="26" rx="4"/>',
            f'<text class="nm" x="{(seg_b[0] + seg_b[1]) / 2}" y="{y0 + 17}" text-anchor="middle">구간 2</text>']
    for top_side, start, ok in ((True, 80.0, True), (False, seg_a[1] - (context + horizon) * scale + 18, False)):
        end = start + context * scale
        target = end + horizon * scale
        y = y0 - 22 if top_side else y0 + 38
        cls = "win-ok" if ok else "win-bad"
        body.append(f'<rect class="{cls}" x="{start:.0f}" y="{y}" width="{end - start:.0f}" height="12" rx="3"/>')
        body.append(f'<path class="eg" d="M{end:.0f},{y + 6}H{target - 5:.0f}"/>' + _head(target, y + 6))
        body.append(f'<circle class="{"tgt-ok" if ok else "tgt-bad"}" cx="{target:.0f}" cy="{y + 6}" r="4"/>')
        text = (f"사용: 최근 {context:g}초 window와 {horizon:g}초 뒤 정답이 모두 구간 1 안" if ok
                else f"제외: {horizon:g}초 뒤 정답이 끊긴 곳 너머에 있음 (끝 위치 목록에서 미리 제외)")
        body.append(f'<text class="{"st-ok" if ok else "st-bad"}" x="{start:.0f}" y="{y - 6 if top_side else y + 28}">'
                    f"{escape(text)}</text>")
    return _svg(760, y0 + 84, body, label)
