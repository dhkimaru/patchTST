"""README와 대시보드 요약 탭의 실험 로드맵 그림을 밝게/어둡게 SVG로 만든다 (외부 라이브러리 없음).

    python scripts/readme_roadmap.py            → docs/images/results_flow_{light,dark}.svg
    python scripts/readme_roadmap.py <폴더>     → 다른 폴더에 저장

카드 하나 = 한 일(제목) · 잰 것(설명 줄) · 값(큰 숫자) · 의미(아래 줄), 판정은 카드 위 배지. 선 위 문구는 '판정 → 다음 실험'.
차수가 진행되면 CARDS · DEAD와 머리글(차수 범위, 날짜)만 고친다. 같은 파일을 README(두 저장소)와
대시보드 요약 탭(report/overview.py roadmap_figure)이 함께 쓴다. 글이 칸을 넘는지는 화면에서 한 번 더 볼 것.
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',system-ui,sans-serif"
TH = {
    "light": dict(ink="#1f2328", ink2="#59636e", muted="#8b949e", card="#ffffff", band="#f6f8fa", tag="#f0f2f5",
                  shadow="#1f2328", tint=0.045, explore="#2a78d6", problem="#e16f24", cause="#8250df", solve="#1a7f37",
                  next="#3987e5", fc="#1a9e6e", good="#1a7f37", warn="#bf8700", fact="#6e7781", on="#ffffff",
                  dead="#cf222e"),
    "dark": dict(ink="#e6edf3", ink2="#b1bac4", muted="#8b949e", card="#161b22", band="#11161d", tag="#1c2128",
                 shadow="#000000", tint=0.09, explore="#4493f8", problem="#f0883e", cause="#a371f7", solve="#3fb950",
                 next="#6da7ec", fc="#2fbf8a", good="#3fb950", warn="#d29922", fact="#8b949e", on="#0d1117",
                 dead="#f85149"),
}
W = 980
CW = 224
COL = {"L": 40, "M": 378, "R": 716}       # gaps of 114 px hold the ribbon labels


def tw(s, size, bold=False):
    """Rough text width (Hangul full em, the rest ~0.6 em)."""
    return (1.05 if bold else 1.0) * sum(size if "가" <= c <= "힣" or c in "①②✕" else size * (0.62 if c in "~%·×÷→−" else 0.57)
                                         for c in s)


def T(x, y, s, t, size=12, fill="ink", weight=400, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{t[fill]}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


# key: (column, y, phase, number, title, date, label, big, captions, verdict, sticker on the left)
CARDS = {
    1: ("L", 256, "explore", "1", "설정 탐색 · 최종 설정", "10-27 ~ 11-03", "가상 센서 최종 설정", "16개 대상",
        ["온도 · 압력 · 가스 · 밸브 · RF", "LightGBM보다 나음 10개 (참고)"], "채택", None),
    2: ("M", 256, "problem", "2", "시험 run 평가", "11-04", "검증 대비 시험 run 오차 (가스 · 밸브)", "1.0~2.2배",
        ["온도 · RF 반사 전력만 양호", "APC · VG11은 기준선 수준 이하"], "문제 발견", None),
    3: ("R", 256, "cause", "3", "교차검증 5묶음", "11-06 ~ 11-07", "기준선보다 나은 대상 (5묶음 전부)", "8개 모두",
        ["가스 · 밸브 · APC · VG11", "어려운 run: 검증 4 · 시험 7"], "채택 유지", None),
    4: ("R", 520, "cause", "4", "원인 분석", "11-08 ~ 11-11", "가상 센서 오차의 원인", "2가지",
        ["① 1초 묶음 × step 전환 시점 흔들림", "② 전체 학습 시 사이클 구간 희생", "장비 드리프트는 아님 (날짜와 무관)"],
        "확인", None),
    "5a": ("M", 452, "cause", "5", "step 맞춤 1초 묶음", "11-12 ~ 11-14", "개선율 상승 (기존 1초 묶음 대비)", "17~42 %p",
           ["5개 대상, 9개 분할 모두", "→ 원인 ① 확인"], "채택", None),
    "6a": ("L", 452, "cause", "6", "원본 0.5초 값으로 재채점", "11-14 · split_v2 · 재학습 없음", "같은 정답으로 비교 시 가장 정확",
           "0.5초 학습", ["→ 1초 묶음 자체가 오차 원인", "→ '1초 묶음 유지' 판단 정정"], "정정", None),
    "5b": ("M", 606, "cause", "7", "사이클 전용 모델", "11-10 ~ 11-11", "개선율 상승 (공정 전체 학습 대비)", "10~17 %p",
           ["5개 대상, 교차검증 3묶음", "→ 원인 ② 뒷받침"], "유력", None),
    "6b": ("L", 606, "cause", "8", "step 맞춤 + 사이클 전용", "11-13 ~ 11-14", "개선율 (레시피 평균 대비, 5묶음)", "69~87 %",
           ["기존 1초는 25~52 % (가스 · 밸브 6개)", "→ 두 해법을 합치면 효과"], "일부 채택", None),
    7: ("L", 824, "solve", "9", "0.5초 + 사이클 전용 조합", "11-15 ~ 11-16", "오차 감소 (step 맞춤 조합 대비)", "16~57 %",
        ["교차검증 5묶음 · VG11은 5 %", "→ 8개 대상 최종 구성"], "채택", None),
    8: ("M", 824, "solve", "10", "시험 run 다시 확인", "11-16 · 재사용, 참고", "검증 대비 시험 run 오차 (가스 · 밸브)", "0.4~1.2배",
        ["시험 오차도 1초보다 9~47 % 낮음", "VG11만 레시피 평균보다 나쁨"], "확인", None),
    "9a": ("R", 748, "solve", "11", "나머지 가상 센서 8개", "11-16 ~ 11-17", "0.5초 조합 오차 감소 (1초 대비, 5묶음)", "35 · 18 %",
           ["R.PWR · P.POS, 5묶음 모두 감소", "온도 5 · L.POS는 1초 유지"], "채택", None),
    "9b": ("R", 902, "next", "12", "VG11 사이클 밖 오차", "11-20 · 학습 없음", "사이클 밖 오차 중 대기압 근처 구간", "93~96%",
           ["모델이 이 구간 압력을 낮게 예측", "→ 이 구간은 따로 평가 검토"], "확인", "지금"),
}
VERDICT_COLOR = {"채택": "good", "채택 유지": "good", "일부 채택": "good", "유력": "good", "확인": "fact", "정정": "warn",
                 "문제 발견": "problem", "계획": "next"}
DEAD = [("최근 run 가중", "드리프트 아님 · 온도는 악화"), ("쉬운 run 비율", "사후 상관 · 재현 안 됨"),
        ("센서 하나만 고장", "온도 외 결과 거의 같음"), ("LightGBM과 섞기", "PatchTST 단독보다 못함"),
        ("patch · window 탐색", "확실히 나은 설정 없음")]


def card_h(k):
    return 126 + 17 * (len(CARDS[k][8]) - 1)


def box(k):
    col, y, *_ = CARDS[k]
    return COL[col], y, CW, card_h(k)


def sticker(p, t, x, y, text, color, anchor):
    w = tw(text, 10.5, True) + 16
    sx = x - w if anchor == "end" else x
    p.append(f'<rect x="{sx:.1f}" y="{y - 10}" width="{w:.1f}" height="20" rx="10" fill="{t[color]}"/>')
    p.append(T(sx + w / 2, y + 3.8, text, t, 10.5, "on", 700, "middle"))


def draw_card(p, t, k):
    col, y, phase, num, title, date, label, big, cap, verdict, left = CARDS[k]
    x, _, w, h = box(k)
    p.append(f'<rect x="{x + 1.5}" y="{y + 3}" width="{w}" height="{h}" rx="14" fill="{t["shadow"]}" fill-opacity="0.06"/>')
    dash = ' stroke-dasharray="5 4"' if verdict == "계획" else ""
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{t["card"]}" stroke="{t[phase]}" '
             f'stroke-opacity="0.6" stroke-width="1.4"{dash}/>')
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{t[phase]}" fill-opacity="{t["tint"]}"/>')
    r = 12 if len(num) == 1 else 13
    p.append(f'<circle cx="{x + 24}" cy="{y + 26}" r="{r}" fill="{t[phase]}"/>')
    p.append(T(x + 24, y + 30.5, num, t, 12 if len(num) == 1 else 11, "on", 800, "middle"))
    p.append(T(x + 44, y + 25, title, t, 13.5, "ink", 700))
    p.append(T(x + 44, y + 41, date, t, 11, "muted"))
    p.append(T(x + 16, y + 63, label, t, 11, phase, 700))
    p.append(T(x + 15, y + 90, big, t, 25, phase, 800))
    for i, s in enumerate(cap):
        p.append(T(x + 16, y + 110 + 17 * i, s, t, 11.5, "ink2"))
    sticker(p, t, x + w - 14, y, verdict, VERDICT_COLOR[verdict], "end")
    if left:
        sticker(p, t, x + 14, y, "● " + left, "dead" if False else phase, "start")


SHAFT, HEAD_L, HEAD_W, GAP = 6, 15, 19, 4   # arrow: shaft width, head length and width, gap before the card
DIRS = {"r": (1, 0), "l": (-1, 0), "d": (0, 1)}


def shaft(p, t, d, phase, dashed=False):
    style = ' stroke-dasharray="0.1 10" stroke-linecap="round"' if dashed else ' stroke-linecap="butt"'
    p.append(f'<path d="{d}" fill="none" stroke="{t[phase]}" stroke-width="{SHAFT}" stroke-linejoin="round"{style}/>')


def arrow(p, t, start, tip, direction, phase, via=None, dashed=False):
    """One arrow: the shaft stops at the base of the head and the head's point stops just before the card.
    ``via`` holds the two control points of a curve whose last stretch already runs in the arrow's direction;
    the second one moves back with the end so the curve meets the head straight."""

    dx, dy = DIRS[direction]
    bx, by = tip[0] - dx * HEAD_L, tip[1] - dy * HEAD_L          # base of the head
    ex, ey = bx + dx, by + dy                                     # shaft runs 1 px under the head: no seam
    if via:
        (c1x, c1y), (c2x, c2y) = via
        d = f"M{start[0]},{start[1]} C{c1x},{c1y} {c2x - dx * HEAD_L},{c2y - dy * HEAD_L} {ex},{ey}"
    else:
        d = f"M{start[0]},{start[1]} L{ex},{ey}"
    shaft(p, t, d, phase, dashed)
    nx, ny = -dy * HEAD_W / 2, dx * HEAD_W / 2
    pts = f"{tip[0]},{tip[1]} {bx + nx:.1f},{by + ny:.1f} {bx - nx:.1f},{by - ny:.1f}"
    p.append(f'<polygon points="{pts}" fill="{t[phase]}" stroke="{t[phase]}" stroke-width="1.5" stroke-linejoin="round"/>')


def pill(p, t, cx, cy, text, phase, anchor="middle"):
    w = tw(text, 11.5, True) + 15
    x0 = cx - w / 2 if anchor == "middle" else cx
    p.append(f'<rect x="{x0:.1f}" y="{cy - 11.5}" width="{w:.1f}" height="23" rx="11.5" fill="{t["card"]}" '
             f'stroke="{t[phase]}" stroke-width="1.5"/>')
    p.append(T(x0 + w / 2, cy + 4, text, t, 11.5, phase, 700, "middle"))


def section(p, t, x, y, text, phase):
    p.append(f'<rect x="{x}" y="{y - 11}" width="4" height="14" rx="2" fill="{t[phase]}"/>')
    p.append(T(x + 11, y + 1, text, t, 12.5, phase, 800))


def figure(t):
    p = []
    b = {k: box(k) for k in CARDS}
    rgt = lambda k: b[k][0] + b[k][2]
    lft = lambda k: b[k][0]
    cx = lambda k: b[k][0] + b[k][2] / 2
    top = lambda k: b[k][1]
    bot = lambda k: b[k][1] + b[k][3]
    my = lambda k: b[k][1] + b[k][3] / 2

    # header + hero number
    p.append(T(40, 46, "실험 로드맵", t, 26, "ink", 800))
    p.append(T(40, 73, "가상 센서의 새 run 오차 원인 추적과 해결 과정 · 1~648차, 11-20 기준", t, 13.5, "ink2"))
    v1, v2 = "1.0~2.2배", "0.4~1.2배"
    w2 = tw(v2, 22, True); w1 = tw(v1, 22, True)
    x2 = 940 - w2; xa = x2 - 30; x1 = xa - 6 - w1
    p.append(T(940, 26, "검증 대비 시험 run 오차 (가스 · 밸브)", t, 11.5, "muted", 700, "end"))
    p.append(T(x1, 56, v1, t, 22, "problem", 800)); p.append(T(xa + 12, 55, "→", t, 18, "muted", 700, "middle"))
    p.append(T(x2, 56, v2, t, 22, "solve", 800))
    p.append(T(x1 + w1 / 2, 76, "1초 최종 설정", t, 11, "muted", 400, "middle"))
    p.append(T(x2 + w2 / 2, 76, "0.5초 조합", t, 11, "muted", 400, "middle"))

    # forecast strip
    y0 = 100
    p.append(f'<rect x="20" y="{y0}" width="{W - 40}" height="104" rx="16" fill="{t["band"]}"/>')
    p.append(T(42, y0 + 32, "미래값 예측", t, 14, "fc", 800))
    p.append(T(42, y0 + 51, "10 · 20 · 30초 뒤 값", t, 11, "muted"))
    p.append(T(42, y0 + 76, "새 run에서도 검증 수준", t, 11.5, "fc", 700))
    xs = [286, 482, 678, 862]
    p.append(f'<line x1="{xs[0]}" y1="{y0 + 48}" x2="{xs[-1]}" y2="{y0 + 48}" stroke="{t["fc"]}" stroke-width="6" '
             f'stroke-linecap="round"/>')
    fc = [("설정 탐색", "19.5 → 7.9", "남은 오차, 현재값 유지 = 100"), ("시험 run", "1.0배", "남은 오차 검증 7.9 · 시험 7.9"),
          ("예측 거리 확장", "30초~10분", "거리별로 나은 모델 비교"), ("장기 예측 보완", "68 · 83 % 감소", "5~10분 APC · VG11 (RevIN 끔)")]
    for x, (a, v, c) in zip(xs, fc):
        p.append(f'<circle cx="{x}" cy="{y0 + 48}" r="8" fill="{t["card"]}" stroke="{t["fc"]}" stroke-width="4"/>')
        p.append(T(x, y0 + 30, a, t, 12.5, "ink", 700, "middle"))
        p.append(T(x, y0 + 77, v, t, 15, "fc", 800, "middle"))
        p.append(T(x, y0 + 94, c, t, 11, "muted", 400, "middle"))

    # section labels
    section(p, t, 40, 240, "탐색 → 문제 발견", "explore")
    section(p, t, 40, 436, "원인 추적 · 두 갈래", "cause")
    section(p, t, 378, 810, "해결 → 확인 → 다음", "solve")

    # arrows (drawn under the cards)
    y1 = my(1)
    arrow(p, t, (rgt(1), y1), (lft(2) - GAP, y1), "r", "problem")
    arrow(p, t, (rgt(2), y1), (lft(3) - GAP, y1), "r", "cause")
    arrow(p, t, (cx(3), bot(3)), (cx(4), top(4) - GAP), "d", "cause")
    y4 = my(4)
    for k5, k6 in (("5a", "6a"), ("5b", "6b")):
        yk = my(k5)
        arrow(p, t, (lft(4), y4), (rgt(k5) + GAP, yk), "l", "cause", via=((lft(4) - 52, y4), (rgt(k5) + 52, yk)))
        arrow(p, t, (lft(k5), yk), (rgt(k6) + GAP, yk), "l", "cause")
    # the two branches merge at a junction above card 9
    jx, jy = cx("6b"), (bot("6b") + top(7)) / 2
    ya = my("6a")
    shaft(p, t, f"M{lft('6a')},{ya} C{lft('6a') - 22},{ya} 17,{ya + 14} 17,{ya + 40} L17,{jy - 22} "
                f"Q17,{jy} 40,{jy} L{jx},{jy}", "solve")
    arrow(p, t, (jx, bot("6b")), (jx, top(7) - GAP), "d", "solve")
    p.append(f'<circle cx="{jx}" cy="{jy}" r="8" fill="{t["card"]}" stroke="{t["solve"]}" stroke-width="4.5"/>')
    y7 = my(7)
    arrow(p, t, (rgt(7), y7), (lft(8) - GAP, y7), "r", "solve")
    for k in ("9a", "9b"):
        yk = my(k)
        arrow(p, t, (rgt(8), y7), (lft(k) - GAP, yk), "r", CARDS[k][2], via=((rgt(8) + 52, y7), (lft(k) - 52, yk)))

    for k in CARDS:
        draw_card(p, t, k)

    # ribbon labels: verdict → next step
    g1, g2 = (rgt(1) + lft(2)) / 2, (rgt(2) + lft(3)) / 2
    pill(p, t, g1, y1 - 27, "채택 → 시험 run", "problem")
    pill(p, t, g2, y1 - 27, "2.2배 → 과적합?", "cause")
    pill(p, t, cx(3), (bot(3) + top(4)) / 2, "과적합 아님 → 어려운 run 원인 추적", "cause")
    pill(p, t, g2, my("5a") - 31, "① → 묶음 변경", "cause")
    pill(p, t, g2, my("5b") + 31, "② → 구간 분리", "cause")
    pill(p, t, g1, my("5a") - 27, "채점 기준 통일", "cause")
    pill(p, t, g1, my("5b") - 27, "① 해법과 결합", "cause")
    pill(p, t, jx + 18, jy, "둘 다 효과 → 합침", "solve", "start")
    pill(p, t, g1, y7 - 27, "채택 → 시험 run", "solve")
    pill(p, t, g2, my("9a") - 31, "해결 → 나머지", "solve")
    pill(p, t, g2, my("9b") + 31, "VG11만 남음", "next")

    # experiments that did not lead on
    fy = bot("9b") + 36
    section(p, t, 40, fy, "이어지지 않은 실험과 이유", "fact")
    gap = 12; cw = (W - 80 - gap * (len(DEAD) - 1)) / len(DEAD)
    for i, (name, why) in enumerate(DEAD):
        x = 40 + i * (cw + gap)
        p.append(f'<rect x="{x:.1f}" y="{fy + 14}" width="{cw:.1f}" height="50" rx="10" fill="{t["tag"]}"/>')
        p.append(T(x + 12, fy + 34, "✕", t, 11.5, "dead", 800))
        p.append(T(x + 27, fy + 34, name, t, 12, "ink2", 700))
        p.append(T(x + 12, fy + 53, why, t, 11, "muted"))
    ny = fy + 92
    for i, line in enumerate([
            "화살표 위 글자는 판정 → 다음 실험. 개선율은 기준선(미래값: 현재값 유지, 가상 센서: 레시피 평균) 대비, 비교 대상이 다른 숫자는 카드에 표시.",
            "오차는 모두 MAE(평균 절대 오차). 개선율 = 1 - MAE ÷ 기준선 MAE, 오차 감소 = 1 - MAE ÷ 비교 대상 MAE, 배 = MAE끼리의 비, %p = 개선율끼리의 차이.",
            "판정: 채택 = 구성에 반영 · 일부 채택 = 일부 대상만 반영 · 유력 = 일부 분할에서 확인 · 확인 = 사실 확인 · 정정 = 앞선 결론 수정",
            "시험 run은 설정 선택에 쓰지 않은 마지막 11개 run, 11-04 · 11-16(재사용, 참고)에서만 사용."]):
        p.append(T(40, ny + 18 * i, line, t, 11.5, "muted"))
    H = ny + 70
    title = "실험 로드맵: 가상 센서의 검증 대비 시험 run 오차를 1.0~2.2배에서 0.4~1.2배로 줄이기까지 (1~648차)"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{escape(title)}" font-family="{FONT}"><title>{escape(title)}</title>{"".join(p)}</svg>')


for mode, th in TH.items():
    (OUT / f"results_flow_{mode}.svg").write_text(figure(th), encoding="utf-8")
print("ok")
