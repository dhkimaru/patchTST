"""README 시스템 구성도를 밝게/어둡게 SVG로 만든다 (외부 라이브러리 없음).

    python scripts/readme_diagrams.py
    → docs/images/architecture_{light,dark}.svg   층(칸)마다 모듈, 오른쪽에 서버 저장소

글을 고칠 때는 아래 TASKS · LAYERS · MODELS · DATA · STORAGE 목록만 바꾸면 된다. 설명은 줄마다 하나씩 적고,
줄 수가 늘면 칸 높이가 따라 늘어난다. 글이 칸을 넘으면 경고가 나온다 (글자 폭은 어림값이라 화면에서 한 번 더 볼 것).
색은 README 결과 그림(docs/images/results_*.svg)과 같다.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

# ---------------------------------------------------------------------------------------------------------
# 내용 (여기만 고치면 된다)
# ---------------------------------------------------------------------------------------------------------
TASKS = [  # (과제, 대상, 입력 → 출력)
    ("미래값 예측", "8개 신호", "최근 192초 → 10·20·30초 뒤 값"),
    ("가상 센서", "16개 센서", "다른 센서 신호 → 대상 센서의 현재 값"),
    ("두께 예측 (VM)", "run 단위", "run 하나의 trace → 두께 · U/F"),
]
LAYERS = [  # (번호, 이름, 설명, [(모듈, 파일, [하는 일 — 줄마다 하나])])
    ("1", "사용자 화면", "결과를 보고 실험을 넣는 곳", [
        ("실험실 대시보드", "server.py · report/", ["결과 확인 · 조합 입력 · 진행 상황 · 공개 주소는 보기 전용"]),
        ("터미널 명령 (CLI)", "cli.py · scripts/", ["설치 · 기준 실험 일괄 실행 · 대시보드 띄우기"]),
    ]),
    ("2", "실험 운영", "무엇을 어떤 순서로 돌릴지", [
        ("대기열 · 작업자", "launch.py", ["요청을 하나씩 차례로", "GPU 4장에 나눠 학습"]),
        ("모델 선정 단계", "stages.py", ["PatchTST 조합 탐색 →", "전 대상 적용 → 대안 모델"]),
        ("차수별 실험 일괄 등록", "planfile.py", ["계획서(.md)에 적은 차수를", "대기열에 순서대로"]),
        ("누적 기록", "ledger.py", ["모든 실험의 job을", "한 표로 모음"]),
    ]),
    ("3", "학습 파이프라인", "training/ · runner.py가 job마다 순서대로 실행", [
        ("실험 펼치기", "config.py", ["실험 YAML →", "job 목록"]),
        ("데이터 준비", "prepare.py", ["원본 → 학습용 배열", "구간 분할 · 캐시"]),
        ("window 자르기", "windows.py", ["정규화한 데이터를", "GPU에 올려 자름"]),
        ("학습", "trainer.py", ["검증 오차 기준", "조기 종료"]),
        ("평가", "evaluate.py", ["기준선 대비 개선율", "MAE · RMSE · R²", "사이클 안/밖 따로"]),
    ]),
]
MODELS = [  # (묶음, [모델]) — 첫 묶음의 첫 모델이 주 모델로 강조된다
    ("딥러닝", ["PatchTST · 주 모델", "iTransformer", "DLinear", "GRU"]),
    ("고전 ML", ["LightGBM", "Ridge", "PLS"]),
    ("기준선", ["현재값 유지", "레시피 평균", "학습 평균"]),
]
DATA = [  # (이름, 위치, [내용])
    ("원본 데이터 읽기", "data/", ["trace CSV · Ex-situ 두께"]),
    ("신호 200개 정의", "configs/columns/", ["단위 · 중요도 · 센서 그룹"]),
    ("run 분할", "configs/splits/", ["학습 51 · 검증 11 · 시험 11"]),
    ("기준 실험 설정 (YAML)", "configs/experiments/", ["대상 · 모델 · 학습 설정"]),
]
STORAGE = [("prepared/", "배열 캐시"), ("experiments/", "결과 · 가중치"), ("queue/", "실행 요청"),
           ("packages/", "모델 패키지"), ("reports/", "HTML 보고서"), ("journal/", "단계 기록")]

# ---------------------------------------------------------------------------------------------------------
# 그리기 도구
# ---------------------------------------------------------------------------------------------------------
FONT = "'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',system-ui,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'D2Coding',monospace"
THEMES = {
    "light": {"text": "#1f2328", "text2": "#59636e", "muted": "#8b949e", "line": "#8b949e", "boxline": "#d0d7de",
              "band": "#f6f8fa", "bg": "#ffffff", "blue": "#2a78d6", "blue_fill": "#e8f1fc", "on_blue": "#ffffff",
              "orange": "#eb6834", "orange_fill": "#fdf3ee"},
    "dark": {"text": "#e6edf3", "text2": "#b1bac4", "muted": "#8b949e", "line": "#6e7681", "boxline": "#30363d",
             "band": "#161b22", "bg": "#0d1117", "blue": "#3987e5", "blue_fill": "#10233a", "on_blue": "#0d1117",
             "orange": "#d95926", "orange_fill": "#24150f"},
}
LINE = 17  # description line height


def text_width(s: str, size: float, mono: bool = False, bold: bool = False) -> float:
    """Rough advance width in px; errs on the wide side (Windows/macOS Korean fonts)."""

    if mono:
        return sum((1.05 if ord(c) > 0x2E80 else 0.62) for c in s) * size
    total = 0.0
    for c in s:
        o = ord(c)
        if 0xAC00 <= o <= 0xD7A3 or 0x2460 <= o <= 0x24FF or 0x2190 <= o <= 0x2193:
            total += 1.0
        elif c == " ":
            total += 0.28
        elif c in "·.,:;()[]/|'!":
            total += 0.32
        elif c.isupper() or c in "×=+−~_²":
            total += 0.66
        else:
            total += 0.56
    return total * size * (1.04 if bold else 1.0)


class Svg:
    def __init__(self, width: int, height: int, title: str, theme: str) -> None:
        self.w, self.h, self.title, self.c = width, height, title, THEMES[theme]
        self.parts: list[str] = []
        self.warnings: list[str] = []

    def text(self, x: float, y: float, s: str, *, size: float = 12, weight: int = 400, color: str = "text",
             anchor: str = "start", mono: bool = False, limit: float | None = None) -> None:
        if limit is not None and text_width(s, size, mono, weight >= 600) > limit:
            self.warnings.append(f"{text_width(s, size, mono, weight >= 600):.0f}px > {limit:.0f}px: {s}")
        family = f' font-family="{MONO}"' if mono else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{self.c[color]}" text-anchor="{anchor}"{family}>{escape(s)}</text>')

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str = "band", stroke: str | None = "boxline",
             dashed: bool = False, rx: float = 8, width: float = 1.2) -> None:
        line = f' stroke="{self.c[stroke]}" stroke-width="{width}"' if stroke else ""
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{self.c[fill]}"{line}{dash}/>')

    def arrow(self, points: list[tuple[float, float]], *, color: str = "line", width: float = 1.6) -> None:
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.c[color]}" stroke-width="{width}" '
                          f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#ah-{color})"/>')

    def badge(self, cx: float, cy: float, label: str, r: float = 11) -> None:
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{self.c["blue"]}"/>')
        self.text(cx, cy + 4.3, label, size=12.5, weight=700, color="on_blue", anchor="middle")

    def raw(self, fragment: str) -> None:
        self.parts.append(fragment)

    def render(self) -> str:
        markers = "".join(
            f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{self.c[k]}"/></marker>'
            for k in ("line", "blue"))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.title)}" font-family="{FONT}">'
                f'<title>{escape(self.title)}</title><defs>{markers}</defs>' + "".join(self.parts) + "</svg>\n")


def module_height(lines: list[str]) -> float:
    return 64 + LINE * (len(lines) - 1)


def module(s: Svg, x: float, y: float, w: float, h: float, name: str, path: str, lines: list[str]) -> None:
    """A module box: name, file, then what it does (one or more lines)."""

    s.rect(x, y, w, h, fill="bg", stroke="boxline", rx=7, width=1.1)
    s.text(x + 11, y + 20, name, size=13.5, weight=700, limit=w - 20)
    s.text(x + 11, y + 37, path, size=11.5, mono=True, color="blue", limit=w - 20)
    for i, line in enumerate(lines):
        s.text(x + 11, y + 54 + i * LINE, line, size=12, color="text2", limit=w - 20)


def band(s: Svg, x: float, y: float, w: float, h: float, num: str, name: str, note: str) -> None:
    s.rect(x, y, w, h, fill="band", stroke="boxline", rx=10)
    s.badge(x + 22, y + 20, num)
    s.text(x + 40, y + 25, name, size=14.5, weight=700)
    s.text(x + 46 + text_width(name, 14.5, bold=True), y + 25, note, size=12, color="muted",
           limit=w - 60 - text_width(name, 14.5, bold=True))


# ---------------------------------------------------------------------------------------------------------
# 시스템 구성도
# ---------------------------------------------------------------------------------------------------------
def architecture(theme: str) -> Svg:
    W = 960
    sw, sgap = 112, 52                 # storage column and the gap that holds its arrows
    lx, lw = 24, W - 48 - sw - sgap   # layer bands 1-3
    sx = lx + lw + sgap
    head, pad, between = 38, 12, 30   # band title, bottom padding, gap between bands

    # heights first: every band is as tall as its tallest module
    tops, heights, box_h = [], [], []
    y = 146
    for _, _, _, mods in LAYERS:
        bh = max(module_height(m[2]) for m in mods)
        tops.append(y)
        box_h.append(bh)
        heights.append(head + bh + pad)
        y += heights[-1] + between
    p_bot = tops[-1] + heights[-1]
    b4 = p_bot + 38
    data_h = max(module_height(d[2]) for d in DATA)
    inner = 2 * data_h + 10
    b4h = 40 + inner + pad
    legend_y = b4 + b4h + 30
    H = int(legend_y + 18)

    s = Svg(W, H, "ALD Lab 시스템 구성도: 사용자 화면, 실험 운영, 학습 파이프라인, 모델, 데이터·설정, 서버 저장소", theme)
    s.text(24, 34, "ALD Lab 시스템 구성도", size=18, weight=700)
    s.text(24, 56, "PEALD SiN 장비 trace로 미래값 · 가상 센서 · 두께를 예측하는 PatchTST 중심 실험 환경",
           size=13, color="text2")

    # 과제 띠
    tw = (W - 48 - 24) / 3
    for i, (task, scope, io) in enumerate(TASKS):
        x = 24 + i * (tw + 12)
        s.rect(x, 72, tw, 56, fill="blue_fill", stroke=None, rx=8)
        s.raw(f'<rect x="{x:.1f}" y="72" width="4" height="56" rx="2" fill="{s.c["blue"]}"/>')
        s.text(x + 16, 95, task, size=14, weight=700)
        s.text(x + tw - 12, 95, scope, size=12, color="muted", anchor="end")
        s.text(x + 16, 116, io, size=12.5, color="text2", limit=tw - 28)

    # 층 1~3
    for (num, name, note, mods), top, h, bh in zip(LAYERS, tops, heights, box_h):
        band(s, lx, top, lw, h, num, name, note)
        pipeline = num == LAYERS[-1][0]
        gap = 16 if pipeline else 12
        n = len(mods)
        mw = (lw - 24 - gap * (n - 1)) / n
        for i, (m_name, m_path, m_lines) in enumerate(mods):
            mx = lx + 12 + i * (mw + gap)
            module(s, mx, top + head, mw, bh, m_name, m_path, m_lines)
            if pipeline and i < n - 1:  # order of the pipeline
                s.arrow([(mx + mw + 2, top + head + bh / 2), (mx + mw + gap - 2, top + head + bh / 2)],
                        color="blue", width=1.8)

    def down(x: float, y0: float, y1: float, label: str) -> None:
        s.arrow([(x, y0 + 2), (x, y1 - 3)])
        s.text(x + 10, (y0 + y1) / 2 + 4.5, label, size=12, color="text2")

    down(130, tops[0] + heights[0], tops[1], "실험 요청")
    down(130, tops[1] + heights[1], tops[2], "요청 하나씩 ald-train 실행")

    # 서버 저장소
    st_top, st_bot = tops[0], p_bot
    s.rect(sx, st_top, sw, st_bot - st_top, fill="orange_fill", stroke="orange", dashed=True, rx=10)
    s.text(sx + 14, st_top + 25, "work/", size=14.5, weight=700)
    s.text(sx + 14, st_top + 43, "서버에만 · Git 밖", size=11, color="muted", limit=sw - 24)
    step = (st_bot - st_top - 66) / len(STORAGE)
    for i, (name, what) in enumerate(STORAGE):
        y = st_top + 66 + i * step + step / 2 - 6
        if i:
            s.raw(f'<line x1="{sx + 12}" y1="{y - step / 2 + 2:.1f}" x2="{sx + sw - 12}" y2="{y - step / 2 + 2:.1f}" '
                  f'stroke="{s.c["orange"]}" stroke-opacity="0.3"/>')
        s.text(sx + 14, y, name, size=11.5, mono=True, limit=sw - 24)
        s.text(sx + 14, y + 18, what, size=12, color="text2", limit=sw - 24)
    mid = lx + lw + sgap / 2
    for top, h, label, to_storage in ((tops[0], heights[0], "누적 기록", False),
                                      (tops[1], heights[1], "요청 저장", True),
                                      (tops[2], heights[2], "결과 저장", True)):
        y = top + h / 2 + 8
        a, b = lx + lw + 3, sx - 3
        s.arrow([(a, y), (b, y)] if to_storage else [(b, y), (a, y)])
        s.text(mid, y - 7, label, size=11, color="text2", anchor="middle", limit=sgap - 2)

    # 층 4 · 5
    mw4 = 480
    band(s, 24, b4, mw4, b4h, "4", "모델", "models/ · deep.py · classic.py")
    gx, gy = 36, b4 + 40
    for (group, names), gw in zip(MODELS, (190, 124, 118)):  # + gaps 12 → 12px inside the band
        s.rect(gx, gy, gw, inner, fill="bg", stroke="boxline", rx=7)
        s.text(gx + 11, gy + 20, group, size=13.5, weight=700)
        for j, name in enumerate(names):
            cy = gy + 32 + j * 25
            strong = group == MODELS[0][0] and j == 0
            s.rect(gx + 9, cy, gw - 18, 20, fill="blue" if strong else "band", stroke=None if strong else "boxline",
                   rx=5)
            s.text(gx + 18, cy + 14.5, name, size=12.5, weight=700 if strong else 400,
                   color="on_blue" if strong else "text", limit=gw - 36)
        gx += gw + 12

    dx = 24 + mw4 + 12
    dw = W - 24 - dx
    band(s, dx, b4, dw, b4h, "5", "데이터 · 설정", "data/ · configs/")
    cw = (dw - 24 - 12) / 2
    for i, (name, where, lines) in enumerate(DATA):
        module(s, dx + 12 + (i % 2) * (cw + 12), b4 + 40 + (i // 2) * (data_h + 10), cw, data_h, name, where, lines)

    s.arrow([(264, b4 - 2), (264, p_bot + 3)])
    s.text(274, (p_bot + b4) / 2 + 4.5, "모델 구현", size=12, color="text2")
    s.arrow([(700, b4 - 2), (700, p_bot + 3)])
    s.text(710, (p_bot + b4) / 2 + 4.5, "trace · 분할 · 설정", size=12, color="text2")

    # 범례
    ly = legend_y
    s.rect(24, ly - 11, 14, 14, fill="blue", stroke=None, rx=3)
    s.text(44, ly, "주 모델", size=12, color="text2")
    s.rect(112, ly - 11, 14, 14, fill="orange_fill", stroke="orange", dashed=True, rx=3)
    s.text(132, ly, "서버에만 있는 데이터 (.gitignore)", size=12, color="text2")
    s.arrow([(344, ly - 4), (372, ly - 4)])
    s.text(380, ly, "데이터 · 요청이 가는 방향", size=12, color="text2")
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        svg = architecture(theme)
        path = OUT / f"architecture_{theme}.svg"
        path.write_text(svg.render(), encoding="utf-8")
        print(path.relative_to(OUT.parents[1]))
        for warning in dict.fromkeys(svg.warnings):
            print("  ! " + warning)


if __name__ == "__main__":
    main()
