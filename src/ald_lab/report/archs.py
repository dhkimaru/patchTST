"""Model architecture figures for the report's Function tab, as inline SVG (no library).

Each figure reads left to right: what goes in, what the model does to it stage by stage, what comes out.
The numbers on the figures (patch length, widths, layer counts) come from :mod:`ald_lab.models.defaults`,
so a changed default shows up here. Colours come from CSS classes, so both themes work.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..models import defaults
from .charts import text_width
from .diagram import PAD, _arrow, _head, _svg, step_lanes

Draw = Callable[[float, float, float, float], str]
TOP = 26  # drawings start under the stage titles
H = 128  # drawing height


def _t(x: float, y: float, text: str, cls: str = "nm", anchor: str = "middle") -> str:
    from html import escape

    return f'<text class="{cls}" x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}">{escape(text)}</text>'


def _wave(x: float, y: float, w: float, amp: float, kind: int, cls: str, n: int = 60) -> str:
    """A made-up signal: 0 pulses (ALD cycles), 1 a slow drift with ripple, 2 a ramp and plateau, 3 smooth."""

    pts = []
    for i in range(n + 1):
        u = i / n
        if kind == 0:
            v = 0.8 if (u * 6) % 1 < 0.35 else -0.5
        elif kind == 1:
            v = 0.6 * math.sin(u * 5.5) + 0.25 * math.sin(u * 31)
        elif kind == 2:
            v = min(1.0, u * 2.2) * 1.4 - 0.7 + 0.08 * math.sin(u * 40)
        else:
            v = 0.7 * math.sin(u * 5.5 + 0.4)
        pts.append(f"{x + u * w:.1f},{y - v * amp:.1f}")
    return f'<polyline class="{cls}" points="{" ".join(pts)}"/>'


def _frame(x: float, y: float, w: float, h: float, cls: str = "abox") -> str:
    return f'<rect class="{cls}" x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="6"/>'


def _tok(x: float, y: float, s: float, k: int) -> str:
    return f'<rect class="tok tk{k % 4 + 1}" x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" rx="2"/>'


def _signals(rows: int = 3) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        lane = (h - 8) / rows
        body = [_frame(x, y, w, h)]
        for i in range(rows):
            cy = y + 4 + lane * (i + 0.5)
            body.append(_wave(x + 8, cy, w - 16, lane * 0.32, (0, 1, 2)[i % 3], f"wv wv{i % 4 + 1}"))
        return "".join(body)

    return draw


def _block(lines: list[str], badge: str = "", hi: bool = False) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h, "abox hi" if hi else "abox")]
        top = y + h / 2 - (len(lines) - 1) * 9
        for i, line in enumerate(lines):
            body.append(_t(x + w / 2, top + i * 18 + 4, line, "nt" if i == 0 else "nm"))
        if badge:
            body.append(f'<rect class="badge" x="{x + w - 34:.0f}" y="{y - 9:.0f}" width="40" height="18" rx="9"/>'
                        + _t(x + w - 14, y + 4, badge, "bt"))
        return "".join(body)

    return draw


def _stack(blocks: list[list[str]], badge: str = "") -> Draw:
    """A block made of sub-blocks (one encoder layer), repeated ``badge`` times."""

    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h, "abox hi")]
        inner = (h - 12 - 4 * (len(blocks) - 1)) / len(blocks)
        for i, lines in enumerate(blocks):
            by = y + 6 + i * (inner + 4)
            body.append(f'<rect class="sub{i % 2}" x="{x + 8:.0f}" y="{by:.0f}" width="{w - 16:.0f}" '
                        f'height="{inner:.0f}" rx="4"/>')
            body.append(_t(x + w / 2, by + inner / 2 + (-2 if len(lines) > 1 else 4), lines[0], "nt"))
            if len(lines) > 1:
                body.append(_t(x + w / 2, by + inner / 2 + 12, lines[1], "nm"))
        if badge:
            body.append(f'<rect class="badge" x="{x + w - 34:.0f}" y="{y - 9:.0f}" width="40" height="18" rx="9"/>'
                        + _t(x + w - 14, y + 4, badge, "bt"))
        return "".join(body)

    return draw


def _outputs(labels: list[str], *, now: bool = False) -> Draw:
    """A short past line and the predicted points ahead of it."""

    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        mid = y + h / 2 + 6
        cut = x + (w * 0.62 if now else w * 0.36)
        body.append(_wave(x + 8, mid, cut - x - 8, 18, 3, "wv wv0", 30))
        body.append(f'<line class="nowline" x1="{cut:.0f}" y1="{y + 10:.0f}" x2="{cut:.0f}" y2="{y + h - 20:.0f}"/>')
        body.append(_t(cut, y + h - 7, "지금", "tiny"))
        end = mid - 0.7 * math.sin(5.9) * 18  # where the past line stops
        if now:
            body.append(f'<circle class="pdot" cx="{cut:.0f}" cy="{end:.0f}" r="5"/>')
            body.append(_t(cut + 10, end - 10, labels[0], "nt", "start"))
            return "".join(body)
        span = (x + w - 14 - cut) / max(len(labels), 1)
        for i, label in enumerate(labels):
            px, py = cut + span * (i + 0.75), mid - 4 - 12 * (i % 2)
            body.append(f'<line class="fan" x1="{cut:.0f}" y1="{end:.0f}" x2="{px:.0f}" y2="{py:.0f}"/>')
            body.append(f'<circle class="pdot" cx="{px:.0f}" cy="{py:.0f}" r="4.5"/>')
            body.append(_t(px, mid - 30, label, "tiny"))
        return "".join(body)

    return draw


def _figure(stages: list[tuple[str, float, Draw, tuple[str, ...]]], *, label: str, gap: float = 34,
            h: float = H, extra: str = "") -> str:
    """Stages left to right: (title, width, drawing, notes under the drawing), joined by arrows."""

    body, x = [], PAD
    for i, (title, w, draw, notes) in enumerate(stages):
        body.append(_t(x, PAD + 12, title, "ct", "start"))
        body.append(draw(x, TOP + PAD, w, h))
        for k, note in enumerate(notes):
            body.append(_t(x + w / 2, TOP + PAD + h + 16 + 14 * k, note))
        if i:
            body.append(_arrow(x - gap + 3, TOP + PAD + h / 2, x - 3, TOP + PAD + h / 2))
        x += w + gap
    lines = max((len(s[3]) for s in stages), default=0)
    return _svg(x - gap + PAD, TOP + PAD + h + 14 * lines + 12, body + [extra], label)


def _width(lines: list[str], minimum: float = 110) -> float:
    return max([minimum] + [text_width(s, 12) + 24 for s in lines])


# ----------------------------------------------------------------------------- classic models

def _feature_grid(names: tuple[str, ...], rows: tuple[str, ...]) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        head, left = 22, 44
        cw = (w - left - 6) / len(names)
        rh = (h - head - 8) / len(rows)
        body = [_frame(x, y, w, h)]
        for j, name in enumerate(names):
            body.append(_t(x + left + cw * (j + 0.5), y + 15, name, "tiny"))
        for i, row in enumerate(rows):
            ry = y + head + i * rh
            body.append(_t(x + left - 6, ry + rh / 2 + 4, row, "tiny", "end"))
            for j in range(len(names)):
                if row == "⋮":
                    continue
                body.append(f'<rect class="tok tk{i % 4 + 1}" x="{x + left + cw * j + 2:.1f}" y="{ry + 3:.1f}" '
                            f'width="{cw - 4:.1f}" height="{rh - 6:.1f}" rx="2"/>')
        return "".join(body)

    return draw


def _two_models() -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        half = (h - 8) / 2
        body = [_frame(x, y, w, half, "abox"), _frame(x, y + half + 8, w, half, "abox hi")]
        body.append(_t(x + w / 2, y + half / 2 - 2, "Ridge", "nt"))
        body.append(_t(x + w / 2, y + half / 2 + 13, "가중합 Σ w·x (선형)", "nm"))
        ty = y + half + 8
        body.append(_t(x + w / 2, ty + 17, "LightGBM", "nt"))
        gap = (w - 40) / 4
        for k in range(4):  # little trees joined by +
            cx, cy = x + 20 + gap / 2 + k * gap, ty + 29
            body.append(f'<path class="tree" d="M{cx:.0f},{cy:.0f} l-8,10 M{cx:.0f},{cy:.0f} l8,10 '
                        f'M{cx - 8:.0f},{cy + 10:.0f} l-4,7 M{cx - 8:.0f},{cy + 10:.0f} l4,7"/>')
            body.append(f'<circle class="pdot" cx="{cx:.0f}" cy="{cy:.0f}" r="2.5"/>')
            if k < 3:
                body.append(_t(cx + gap / 2, cy + 13, "+", "nm"))
        return "".join(body)

    return draw


def classic_figure() -> str:
    names = ("마지막", "평균", "표준편차", "최소", "최대", "변화")
    base = ["기준선 + 보정량", "예측: 현재값 + 변화량", "가상 센서: 레시피 평균 + 보정량"]
    return _figure([
        ("① 입력 window", 130, _signals(), ("최근 L초 × 신호 C개",)),
        ("② 신호마다 요약값 6개", 300, _feature_grid(names, ("신호 1", "신호 2", "⋮", "신호 C")),
         ("시간 순서 대신 요약값으로 줄임", "→ 특징 6 × C개 (한 줄)")),
        ("③ 표 형식 모델", 170, _two_models(), ("LightGBM = 결정나무 수천 개의 합", "둘 다 CPU로 학습")),
        ("④ 보정량 → 최종값", _width(base, 200), _block(base, hi=True), ("트리는 학습 범위 밖 값을 내지 못하므로", "기준선과의 차이만 학습")),
        ("⑤ 출력", 150, _outputs(["+10", "+20", "+30"]), ("예측: 10·20·30 s 뒤", "가상 센서: 지금 값")),
    ], label="고전 ML 구조: window 요약 특징과 Ridge·LightGBM")


def _run_table() -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        cols = ("사이클 평균", "사이클 편차", "공정 평균")
        cw = (w - 58) / 3
        for j, name in enumerate(cols):
            body.append(_t(x + 52 + cw * (j + 0.5), y + 15, name, "tiny"))
        for i, run in enumerate(("run 3", "run 4", "run 5", "⋮", "run 80")):
            ry = y + 22 + i * 20
            body.append(_t(x + 44, ry + 14, run, "tiny", "end"))
            if run != "⋮":
                for j in range(3):
                    body.append(f'<rect class="tok tk{j + 1}" x="{x + 52 + cw * j + 2:.1f}" y="{ry + 3:.1f}" '
                                f'width="{cw - 4:.1f}" height="14" rx="2"/>')
        return "".join(body)

    return draw


def _loo() -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        n, s = 8, 14
        step = (w - 24) / n
        for r in range(3):
            ry = y + 12 + r * 39
            for k in range(n):
                cls = "tok tkout" if k == r + 1 else "tok tk1"
                body.append(f'<rect class="{cls}" x="{x + 12 + k * step + (step - s) / 2:.1f}" y="{ry:.1f}" '
                            f'width="{s}" height="{s}" rx="2"/>')
            body.append(_t(x + w / 2, ry + s + 13, f"{r + 2}번째 run을 빼고 학습 → 그 run 예측", "tiny"))
        return "".join(body)

    return draw


def vm_figure() -> str:
    lines = ["Ridge · PLS · LightGBM", "run 약 60개 → 작은 모델", "(LightGBM 잎 7개)"]
    return _figure([
        ("① run 하나의 trace", 140, _signals(), ("약 4시간 · 신호 C개",)),
        ("② run 요약", 240, _run_table(), ("신호마다 3개 → run 1개가 1행",)),
        ("③ 모델", _width(lines, 170), _block(lines, hi=True), ("",)),
        ("④ 출력", 150, _block(["두께 평균 (Å)", "U/F (%)", "WtoW (%)"]), ("run 하나에 값 하나",)),
        ("⑤ 평가", 210, _loo(), ("run 하나씩 빼며 반복 (LOO)",)),
    ], label="두께 예측 구조: run 요약과 run 단위 모델")


# ----------------------------------------------------------------------------- deep models

def _patching(patch: int, stride: int) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        body.append(_wave(x + 10, y + 26, w - 20, 12, 0, "wv wv1", 60))
        n = 7
        step = (w - 20) / (n + 1)
        for k in range(n):
            px = x + 10 + k * step
            body.append(f'<rect class="patch" x="{px:.1f}" y="{y + 46 + (k % 2) * 8:.1f}" width="{2 * step:.1f}" '
                        f'height="6" rx="2"/>')
        s = 16
        for k in range(n):
            body.append(_tok(x + 12 + k * (s + 4), y + 82, s, 0))
        body.append(_t(x + 12 + n * (s + 4) + 6, y + 95, "…", "nt", "start"))
        body.append(_t(x + w / 2, y + h - 8, f"조각 {patch} s · 간격 {stride} s (절반 겹침)", "tiny"))
        return "".join(body)

    return draw


def _embed(d_model: int) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        for k in range(5):
            bx = x + 16 + k * (w - 32) / 5
            body.append(f'<rect class="tok tk1" x="{bx:.1f}" y="{y + 22:.1f}" width="12" height="{h - 58:.0f}" rx="2"/>')
        body.append(_t(x + w / 2, y + h - 22, f"토큰마다 숫자 {d_model}개", "tiny"))
        body.append(_t(x + w / 2, y + h - 8, "+ 위치 정보", "tiny"))
        return "".join(body)

    return draw


def patchtst_figure() -> str:
    d = defaults.DEEP["patchtst"]
    return _figure([
        ("① 입력 window", 120, _signals(), ("최근 L초 × 신호 C개",)),
        ("② RevIN", 112, _block(["정규화", "window마다", "평균 0 · 편차 1"]), ("출력 때 되돌림",)),
        ("③ patch로 자르기", 190, _patching(d["patch_len"], d["stride"]), ("신호 하나 → 토큰 N개",)),
        ("④ 임베딩", 116, _embed(d["d_model"]), (f"Linear {d['patch_len']}→{d['d_model']}",)),
        ("⑤ Transformer encoder", 200, _stack([["Multi-head attention", f"{d['n_heads']} head → 더하기 · BatchNorm"],
                                                ["Feed-forward", f"{d['d_ff']} · GELU → 더하기 · BatchNorm"]],
                                               f"×{d['n_layers']}"),
         ("앞 층의 attention 점수를 다음 층에 더함",)),
        ("⑥ 출력 head", 150, _outputs(["+10", "+20", "+30"]), ("토큰을 펼쳐 Linear 1개", "→ 10·20·30 s 뒤 값")),
    ], label="PatchTST 구조")


def channel_modes_figure() -> str:
    """Channel-independent vs mixing tokens."""

    s, gap = 18, 4
    body = []

    def panel(x: float, title: str, notes: tuple[str, str], mixing: bool) -> None:
        body.append(_t(x, PAD + 12, title, "ct", "start"))
        y0 = TOP + PAD
        if not mixing:
            for c in range(3):
                for k in range(6):
                    body.append(_tok(x + k * (s + gap), y0 + c * (s + 10), s, c))
                body.append(_arrow(x + 6 * (s + gap) + 4, y0 + c * (s + 10) + s / 2, x + 6 * (s + gap) + 34,
                                   y0 + c * (s + 10) + s / 2))
                body.append(_frame(x + 6 * (s + gap) + 36, y0 + c * (s + 10) - 2, 120, s + 4))
                body.append(_t(x + 6 * (s + gap) + 96, y0 + c * (s + 10) + s / 2 + 4, "encoder (공유)", "nm"))
        else:
            for k in range(6):
                for c in range(3):
                    body.append(f'<rect class="tok tk{c + 1}" x="{x + k * (s + gap):.1f}" '
                                f'y="{y0 + c * (s * 0.62) + 10:.1f}" width="{s:.1f}" height="{s * 0.62:.1f}"/>')
                body.append(f'<rect class="tokframe" x="{x + k * (s + gap):.1f}" y="{y0 + 10:.1f}" width="{s:.1f}" '
                            f'height="{s * 1.86:.1f}" rx="2"/>')
            mid = y0 + 10 + s * 0.93
            body.append(_arrow(x + 6 * (s + gap) + 4, mid, x + 6 * (s + gap) + 34, mid))
            body.append(_frame(x + 6 * (s + gap) + 36, mid - (s + 4) / 2, 120, s + 4))
            body.append(_t(x + 6 * (s + gap) + 96, mid + 4, "encoder 1번", "nm"))
        for k, note in enumerate(notes):
            body.append(_t(x, TOP + PAD + 3 * (s + 10) + 14 + 14 * k, note, "nm", "start"))

    panel(PAD, "신호별 독립 (ci, 원래 PatchTST)", ("토큰 = 신호 하나의 patch. 신호마다 따로 encoder를 지남",
                                                  "→ 신호 사이 관계는 출력 head에서만 섞임"), False)
    panel(PAD + 340, "신호 혼합 (mix, 이전 실험 최고 설정)", ("토큰 = 같은 시각 patch의 모든 신호를 이어 붙인 것",
                                                          "→ attention 전에 신호를 함께 봄"), True)
    legend = "".join(f'<rect class="tok tk{c + 1}" x="{PAD + 680}" y="{TOP + PAD + 6 + c * 22}" width="12" height="12" rx="2"/>'
                     + _t(PAD + 698, TOP + PAD + 16 + c * 22, f"신호 {c + 1}", "nm", "start") for c in range(3))
    return _svg(PAD + 760, TOP + PAD + 3 * (s + 10) + 42, body + [legend], "PatchTST 신호 처리 방식: 독립과 혼합")


def _channel_tokens(attention: bool) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h, "abox hi" if attention else "abox")]
        s, n = 18, 4
        top = y + (h - n * (s + 8)) / 2 + 4
        cx = x + 26 if attention else x + w / 2 - s / 2
        for k in range(n):
            ty = top + k * (s + 8)
            if k == 2:
                body.append(_t(cx + s / 2, ty + 13, "⋮", "nt"))
                continue
            body.append(_tok(cx, ty, s, 3 if k == 3 else k))
            body.append(_t(cx - 4, ty + 13, "C" if k == 3 else str(k + 1), "tiny", "end"))
        if attention:
            centers = [top + k * (s + 8) + s / 2 for k in (0, 1, 3)]
            for i, a in enumerate(centers):
                for b in centers[i + 1:]:
                    r = (b - a) / 2
                    body.append(f'<path class="arc" d="M{cx + s + 2:.0f},{a:.0f} A{r * 1.6:.0f},{r:.0f} 0 0 1 '
                                f'{cx + s + 2:.0f},{b:.0f}"/>')
            body.append(_t(x + w - 10, y + h - 10, "모든 신호 쌍", "tiny", "end"))
        return "".join(body)

    return draw


def _rows_to_tokens() -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        s, n = 18, 4
        top = y + (h - n * (s + 8)) / 2 + 4
        for k in range(n):
            ty = top + k * (s + 8)
            if k == 2:
                body.append(_t(x + w / 2, ty + 13, "⋮", "nt"))
                continue
            body.append(_wave(x + 10, ty + s / 2, w - 64, 6, (0, 1, 2, 3)[k], f"wv wv{(3 if k == 3 else k) % 4 + 1}", 40))
            body.append(_arrow(x + w - 52, ty + s / 2, x + w - 30, ty + s / 2))
            body.append(_tok(x + w - 26, ty, s, 3 if k == 3 else k))
        return "".join(body)

    return draw


def itransformer_figure() -> str:
    d = defaults.DEEP["itransformer"]
    return _figure([
        ("① 입력 window", 120, _signals(), ("최근 L초 × 신호 C개",)),
        ("② 신호마다 토큰 1개", 170, _rows_to_tokens(), (f"window 전체 → Linear L→{d['d_model']}",)),
        ("③ 신호 사이 attention", 170, _channel_tokens(True), (f"{d['n_heads']} head · LayerNorm · ×{d['n_layers']}",
                                                         "신호 간 참조 관계 학습")),
        ("④ 출력 head", 150, _outputs(["지금 값"], now=True), ("모든 신호 토큰을 펼쳐 Linear", "(예측이면 10·20·30 s 뒤)")),
    ], label="iTransformer 구조")


def _decompose(kernel: int, d_rep: int) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = []
        bh = (h - 10) / 2
        for i, (title, kind, cls) in enumerate(((f"추세 (이동평균 {kernel} s)", 3, "wv wv2"),
                                                ("나머지 (원본 − 추세)", 0, "wv wv1"))):
            by = y + i * (bh + 10)
            body.append(_frame(x, by, 150, bh))
            body.append(_t(x + 75, by + 15, title, "tiny"))
            body.append(_wave(x + 10, by + bh / 2 + 8, 130, 10 if kind == 3 else 7, kind, cls, 40))
            body.append(_arrow(x + 152, by + bh / 2, x + 184, by + bh / 2))
            body.append(_frame(x + 186, by + 6, 104, bh - 12, "abox hi"))
            body.append(_t(x + 238, by + bh / 2 + 4, f"Linear L→{d_rep}", "nt"))
            body.append(_arrow(x + 292, by + bh / 2, x + 330, y + h / 2 + (-6 if i == 0 else 6)))
        body.append(f'<circle class="plus" cx="{x + 344:.0f}" cy="{y + h / 2:.0f}" r="13"/>'
                    + _t(x + 344, y + h / 2 + 5, "+", "nt"))
        return "".join(body)

    return draw


def dlinear_figure() -> str:
    d = defaults.DEEP["dlinear"]
    return _figure([
        ("① 입력 (신호 하나)", 120, _signals(1), ("신호마다 같은 가중치로 따로",)),
        ("② 둘로 나눈 뒤 각각 선형층 → ③ 더하기", 358, _decompose(d["kernel"], d["d_rep"]),
         ("Transformer·순환 없이 선형층 2개뿐",)),
        ("④ 출력 head", 150, _outputs(["+10", "+20", "+30"]), ("신호별 표현을 모아 Linear",)),
    ], label="DLinear 구조")


def _gru_chain(hidden: int, layers: int) -> Draw:
    def draw(x: float, y: float, w: float, h: float) -> str:
        body = [_frame(x, y, w, h)]
        n, cw, ch = 5, 40, 24
        step = (w - 30) / n
        rows = [y + h - 50 - r * 38 for r in range(layers)]
        for k in range(n):
            cx = x + 16 + k * step
            label = ("t1", "t2", "t3", "…", "tL")[k]
            body.append(_t(cx + cw / 2, y + h - 5, label, "tiny"))
            if label == "…":
                for ry in rows:
                    body.append(_t(cx + cw / 2, ry + 16, "…", "nt"))
                continue
            for r, ry in enumerate(rows):
                last = k == n - 1 and r == layers - 1
                body.append(f'<rect class="{"tok tk1" if last else "cellbox"}" x="{cx:.0f}" y="{ry:.0f}" width="{cw}" '
                            f'height="{ch}" rx="4"/>')
                body.append(_t(cx + cw / 2, ry + 16, "GRU", "tiny"))
                below = y + h - 17 if r == 0 else rows[r - 1]
                body.append(f'<path class="eg" d="M{cx + cw / 2:.0f},{below - (2 if r else 0):.0f}V{ry + ch + 6:.0f}"/>'
                            + _head(cx + cw / 2, ry + ch, "up"))
            if k:
                for ry in rows:
                    body.append(f'<path class="eg" d="M{cx - step + cw:.0f},{ry + ch / 2:.0f}H{cx - 6:.0f}"/>'
                                + _head(cx, ry + ch / 2))
        body.append(_t(x + 10, rows[-1] - 7, f"상태 {hidden}개를 1초씩 넘김 · {layers}층", "tiny", "start"))
        return "".join(body)

    return draw


def gru_figure() -> str:
    d = defaults.DEEP["gru"]
    return _figure([
        ("① 입력 window", 120, _signals(), ("최근 L초 × 신호 C개",)),
        ("② 시점마다 변환", 112, _block(["Linear", f"신호 C개 → {d['hidden']}", "1초 단위"]), ("",)),
        ("③ 순서대로 읽기", 270, _gru_chain(d["hidden"], d["n_layers"]), ("앞에서부터 상태를 갱신",)),
        ("④ 출력 head", 150, _outputs(["+10", "+20", "+30"]), ("마지막 상태 → Linear",)),
    ], label="GRU 구조")


def wrapper_figure() -> str:
    """What every deep model shares: RevIN around the backbone and the output head, per task."""

    return step_lanes([
        ("예측 — 대상 신호가 입력에 있음 (값을 직접 학습)", "now",
         [("window", "최근 L초 × C"), ("RevIN 정규화", "window 평균·편차 빼기"), ("backbone", "PatchTST·DLinear·…"),
          ("출력 head", "Linear → 10·20·30 s"), ("되돌리기", "대상 신호의 평균·편차"), ("예측값", "물리 단위")],
         "정규화로 수준을 제거해 형태만 학습한 뒤, 대상 신호의 수준을 복원"),
        ("가상 센서 — 대상 신호가 입력에 없음 (레시피 평균 대비 보정량 학습)", "prev",
         [("window", "대상·같은 계열 제외"), ("RevIN 정규화", "window 평균·편차 빼기"), ("backbone", "PatchTST·DLinear·…"),
          ("+ 입력 통계", "신호별 평균·편차 → 64"), ("출력 head", "보정량 (편차 단위)"), ("레시피 평균 +", "→ 지금 값")],
         "정규화로 제거된 수준을 통계값으로 다시 넣고, 레시피 평균 대비 편차만 학습 (손실 MAE)"),
    ], label="딥러닝 공통 틀: RevIN과 출력 head")


# ----------------------------------------------------------------------------- Out tab: what one prediction is

def task_figure(task: str, *, context: float, horizons: list[float]) -> str:
    """One prediction of ``task`` on a time line: what the model sees and what it answers."""

    w, h = 420, 150
    body = []
    if task == "thickness":
        body.append(_t(PAD + 4, 22, "run 하나 (약 4시간)", "ct", "start"))
        body.append(f'<rect class="pl-win" x="{PAD + 4}" y="34" width="250" height="72" rx="4"/>')
        for k, (kind, cls) in enumerate(((0, "wv wv1"), (1, "wv wv2"), (2, "wv wv3"))):
            body.append(_wave(PAD + 12, 50 + k * 22, 234, 7, kind, cls, 70))
        body.append(_t(PAD + 129, 124, "입력: run 전체를 요약", "tiny"))
        body.append(_arrow(PAD + 262, 70, PAD + 300, 70))
        cx, cy = PAD + 350, 70
        body.append(f'<circle class="wafer" cx="{cx}" cy="{cy}" r="34"/>')
        for r in (11, 22):
            body.append(f'<circle class="ring" cx="{cx}" cy="{cy}" r="{r}"/>')
        body.append(_t(cx, cy + 4, "Å", "nt"))
        body.append(_t(cx, 124, "출력: 값 하나", "tiny"))
        return _svg(w, h, body, "두께 예측: run 1개로 값 1개 예측")
    now = PAD + 250
    left = PAD + 40
    scale = (now - left) / max(context, 1)
    win = max(left, now - context * scale)
    if task == "virtual_sensor":
        body.append(_t(PAD + 4, 22, "다른 신호 (입력)", "ct", "start"))
        body.append(f'<rect class="pl-win" x="{win:.0f}" y="30" width="{now - win:.0f}" height="58" rx="4"/>')
        for k, (kind, cls) in enumerate(((0, "wv wv1"), (1, "wv wv2"))):
            body.append(_wave(PAD + 8, 46 + k * 26, now - PAD - 8, 7, kind, cls, 50))
        body.append(_t((win + now) / 2, 102, f"최근 {context:g}초", "tiny"))
        body.append(_t(PAD + 4, 122, "대상 센서", "ct", "start"))
        body.append(f'<line class="pl-future" x1="{PAD + 64}" y1="128" x2="{now:.0f}" y2="128"/>')
        body.append(f'<line class="nowline" x1="{now}" y1="26" x2="{now}" y2="140"/>')
        body.append(f'<circle class="pdot" cx="{now}" cy="128" r="5"/>')
        body.append(_t(now + 10, 132, "지금 값 추정", "nt", "start"))
        body.append(_t(now + 10, 48, "센서가 없어도", "tiny", "start"))
        body.append(_t(now + 10, 62, "다른 신호로 계산", "tiny", "start"))
        return _svg(w, h, body, "가상 센서: 다른 신호로 지금 값 추정")
    body.append(_t(PAD + 4, 22, "대상 신호 (+ 다른 입력 신호)", "ct", "start"))
    body.append(f'<rect class="pl-win" x="{win:.0f}" y="34" width="{now - win:.0f}" height="80" rx="4"/>')
    body.append(_wave(PAD + 8, 82, now - PAD - 8, 14, 0, "wv wv0", 60))
    body.append(_t((win + now) / 2, 130, f"입력: 최근 {context:g}초", "tiny"))
    body.append(f'<line class="nowline" x1="{now}" y1="30" x2="{now}" y2="118"/>')
    body.append(_t(now, 144, "지금", "tiny"))
    far = max(horizons or [1.0])
    for k, hz in enumerate(horizons or []):
        x = now + 14 + (w - now - 40) * hz / far
        y = 70 - 10 * (k % 2)
        body.append(f'<line class="fan" x1="{now}" y1="88" x2="{x:.0f}" y2="{y}"/>')
        body.append(f'<circle class="pdot" cx="{x:.0f}" cy="{y}" r="4.5"/>')
        body.append(_t(x, y - 10, f"+{hz:g}초", "tiny"))
    body.append(_t(now + (w - now) / 2, 130, "출력: 이후 값", "tiny"))
    return _svg(w, h, body, "미래값 예측: 최근 구간으로 이후 값 예측")
