"""요약 탭: 현재 결론과 그림 두 개 (대상별 상대 오차, 기준선 = 100 · 일별 기록 추이).

Numbers come from the ledger (every job of every run, dashboard searches included) through
:func:`ald_lab.stages.patchtst_first`, the same per-target decision the 하이퍼파라미터 탐색 tab shows.
"Remaining error" is a model's MAE as a share of the baseline's (baseline = 100, lower is better); averages over
targets are geometric means so that one target with a large error does not hide the others.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

import pandas as pd

from .. import stages
from ..ledger import main_records

TASKS = {"forecast": ("미래값 예측", "현재값 유지"), "virtual_sensor": ("가상 센서", "레시피 평균")}
DECISION_TEXT = {"keep": ("채택", "good"), "fallback": ("대안 필요", "bad"), "missing": ("미학습", "muted")}


def _ok(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and pd.notna(value)


def geo_mean(values: list[float]) -> float:
    values = [v for v in values if _ok(v) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def decisions(ledger: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Per task (forecast, virtual sensor) the PatchTST-first rows; tasks without records are left out."""

    out: dict[str, pd.DataFrame] = {}
    if ledger is None or not len(ledger) or "task" not in ledger:
        return out
    for task in TASKS:
        if (ledger["task"] == task).any():
            frame = stages.patchtst_first(ledger, task)
            if len(frame):
                out[task] = frame
    return out


def _scored(frame: pd.DataFrame) -> list[Any]:
    """Rows with a baseline and a PatchTST record, best remaining error first."""

    rows = [r for r in frame.itertuples() if _ok(r.baseline_mae) and r.baseline_mae > 0 and _ok(r.patch_mae)]
    return sorted(rows, key=lambda r: r.patch_mae / r.baseline_mae)


def remaining(mae: Any, baseline: Any) -> float:
    return 100.0 * float(mae) / float(baseline) if _ok(mae) and _ok(baseline) and baseline else float("nan")


def trend(ledger: pd.DataFrame, task: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per day a run started: the PatchTST record so far (seeds of one configuration averaged), as the geometric mean
    of remaining error over the task's targets; starts on the first day every target has a record.

    Days with only side studies of the task (other splits, regions, horizons) are listed too, flagged ``side`` with
    the number of runs: the record is carried over unchanged, since those runs do not count toward it."""

    base = {r.target: float(r.baseline_mae) for r in frame.itertuples() if _ok(r.baseline_mae) and r.baseline_mae > 0}
    if not base or ledger is None or not len(ledger) or "when" not in ledger:
        return []
    every_run = ledger[(ledger["task"] == task) & ledger["when"].notna()]
    if "smoke" in every_run:
        every_run = every_run[~every_run["smoke"].astype(bool)]
    ledger = main_records(ledger)
    rows = ledger[(ledger["task"] == task) & (ledger["model"] == "patchtst") & ledger["mae"].notna()
                  & ledger["target"].isin(list(base))]
    if "smoke" in rows:
        rows = rows[~rows["smoke"].astype(bool)]
    rows = rows[rows["when"].notna()]
    if not len(rows):
        return []
    cfg = (rows.groupby(["target", "exp_dir", "label"], dropna=False)
           .agg(mae=("mae", "mean"), when=("when", "first")).reset_index())
    cfg["day"] = pd.to_datetime(cfg["when"]).dt.normalize()
    lgbm = [remaining(r.lgbm_mae, base[r.target]) for r in frame.itertuples() if r.target in base]
    lgbm_mean = geo_mean(lgbm) if lgbm and all(_ok(v) for v in lgbm) else None
    main_days = set(cfg["day"].unique())
    run_days = pd.to_datetime(every_run["when"]).dt.normalize()
    side_runs = every_run.assign(day=run_days)[~run_days.isin(main_days)].groupby("day")["exp_dir"].nunique()
    out = []
    for day in sorted(main_days | set(side_runs.index)):
        best = cfg[cfg["day"] <= day].groupby("target")["mae"].min()
        if set(best.index) != set(base):
            continue
        side = day not in main_days
        out.append({"day": pd.Timestamp(day), "mean": geo_mean([100 * best[t] / base[t] for t in base]),
                    "lgbm": lgbm_mean, "targets": len(base), "side": side,
                    "side_runs": int(side_runs.get(day, 0)) if side else 0})
    return out


# ----------------------------------------------------------------------------- conclusion sentences

def _names(rows: list[Any], limit: int = 4) -> str:
    names = [r.target for r in rows[:limit]]
    return ", ".join(names) + (f" 외 {len(rows) - limit}개" if len(rows) > limit else "")


def conclusion_points(dec: dict[str, pd.DataFrame], trends: dict[str, list[dict[str, Any]]],
                      thickness: list[tuple[str, str, float]] = ()) -> list[str]:
    """Plain sentences (HTML) for the top of the 요약 tab: what is decided, how large the gain is, where PatchTST
    still trails LightGBM, and what changed on the last day."""

    out = []
    for task, frame in dec.items():
        title, base_name = TASKS[task]
        n = len(frame)
        keep = frame[frame["decision"] == "keep"]
        unsure = int((keep["patch_seeds"] < 2).sum())
        fallback, missing = int((frame["decision"] == "fallback").sum()), int((frame["decision"] == "missing").sum())
        head = f"<b>{title}</b>: {n}개 대상 중 {len(keep)}개 PatchTST 채택"
        tail = []
        if unsure:
            tail.append("모두 seed 1개로 편차 미확인" if unsure == len(keep) else f"{unsure}개는 seed 1개로 편차 미확인")
        if fallback:
            tail.append(f"다른 모델 검토 {fallback}개")
        if missing:
            tail.append(f"미학습 {missing}개")
        text = head + (f" ({' · '.join(tail)})" if tail else "")
        scored = _scored(frame)
        gains = [r.patch_gain_pct for r in scored if _ok(r.patch_gain_pct)]
        if gains:
            weakest = min(scored, key=lambda r: r.patch_gain_pct if _ok(r.patch_gain_pct) else math.inf)
            text += (f"<br>기준선({base_name}) 대비 오차 {min(gains):.0f}~{max(gains):.0f} % 감소 "
                     f"(최소: {escape(weakest.target)} {weakest.patch_gain_pct:.0f} %)")
        with_lgbm = [r for r in scored if _ok(r.lgbm_mae)]
        behind = sorted((r for r in with_lgbm if r.patch_mae > r.lgbm_mae), key=lambda r: -r.patch_mae / r.lgbm_mae)
        if with_lgbm:
            text += f"<br>LightGBM(참고) 대비 우위 {len(with_lgbm) - len(behind)}/{len(with_lgbm)}개"
            text += f" (열세: {escape(_names(behind))})" if behind else ""
        out.append(text)
    for task, points in trends.items():
        main = [pt for pt in points if not pt.get("side")]
        if points and points[-1].get("side") and main and _ok(points[-1]["mean"]):
            last = points[-1]
            out.append(f"최근 실험일({last['day']:%m-%d}) {TASKS[task][0]}: 부가 실험만 진행, 기본 기록 {last['mean']:.1f} 유지 "
                       f"(마지막 기본 실험 {main[-1]['day']:%m-%d})")
        elif len(main) >= 2 and _ok(main[-1]["mean"]) and _ok(main[-2]["mean"]):
            last, prev = main[-1], main[-2]
            change = 100 * (last["mean"] / prev["mean"] - 1)
            if abs(change) > 0.05:
                out.append(f"최근 실험일({last['day']:%m-%d}) {TASKS[task][0]} 상대 오차(대상 평균) "
                           f"{prev['mean']:.1f} → {last['mean']:.1f} ({change:+.0f} %)".replace("+-", "−").replace("(-", "(−"))
            else:
                out.append(f"최근 실험일({last['day']:%m-%d}) {TASKS[task][0]} 기록 변동 없음 ({last['mean']:.1f})")
    if thickness:
        best = max(thickness, key=lambda t: t[2])
        low = min(t[2] for t in thickness)
        out.append(f"<b>두께 예측(VM)</b>: PatchTST 적용 대상 아님. 최고 모델 {escape(best[1])}, 학습 평균 대비 "
                   f"{low:.0f}~{best[2]:.0f} % 개선 (leave-one-run-out)")
    if dec:
        out.append("모든 수치는 검증 run 기준이며, 시험 run은 최종 후보 확정 후 1회만 평가")
    return out


STUDY_NAMES = (  # study prefix → what the side study checks (요약 tab line)
    ("cv", "교차검증"), ("time", "시간순 검증"), ("single_sensor", "센서 하나 고장"), ("long_horizon", "예측 거리 확장"),
    ("lc", "학습 run 수"), ("recent", "최근 run 학습"), ("res_05s", "0.5초 간격"), ("region_", "구간 전용 학습"),
    ("fc_", "미래값 구간 전용 학습"),
)


def side_study_note(ledger: pd.DataFrame) -> str | None:
    """One line for the 요약 tab: side studies run since (date, count, kinds), which the decisions above leave out."""

    if not len(ledger) or "study" not in ledger:
        return None
    side = ledger[ledger["study"].fillna("").astype(str) != ""]
    if not len(side) or "when" not in side or side["when"].isna().all():
        return None
    kinds: list[str] = []
    for study in side["study"].astype(str).unique():
        name = next((n for prefix, n in STUDY_NAMES if study.startswith(prefix)), study)
        if name not in kinds:
            kinds.append(name)
    last = side["when"].max()
    return (f"부가 실험(채택 판정 · 기록 갱신에서 제외): 최근 실행 {last:%m-%d} · 실행 {side['exp_dir'].nunique()}회 "
            f"({escape(' · '.join(kinds))}). 결과는 Hyper Parameter ⑦ 누적 기록과 하이퍼파라미터 탐색 탭의 탐색 결과에 있음")


# ----------------------------------------------------------------------------- charts

def _t(x: float, y: float, text: str, cls: str = "", anchor: str = "start", extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"' + (f' class="{cls}"' if cls else "")
            + (f" {extra}" if extra else "") + f">{escape(text)}</text>")


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for decade in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** decade
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def target_chart(dec: dict[str, pd.DataFrame]) -> str:
    """Per target: PatchTST best (dot) and LightGBM (diamond) as remaining error on a log axis, grouped by task, with
    the reduction and the decision written at the right."""

    groups = [(task, _scored(frame)) for task, frame in dec.items()]
    groups = [(task, rows) for task, rows in groups if rows]
    if not groups:
        return ""
    values = [remaining(r.patch_mae, r.baseline_mae) for _, rows in groups for r in rows]
    values += [remaining(r.lgbm_mae, r.baseline_mae) for _, rows in groups for r in rows if _ok(r.lgbm_mae)]
    lo = 10 ** math.floor(math.log10(max(min(values), 0.01)))
    hi = max(100.0, max(values) * 1.15)
    width, left, right, row_h, head_h, top = 960, 220, 200, 24, 30, 36
    x0, x1 = left, width - right
    n_rows = sum(len(rows) for _, rows in groups)
    height = top + head_h * len(groups) + row_h * n_rows + 44

    def px(v: float) -> float:
        v = min(max(v, lo), hi)
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    p = [(f'<svg class="chart ov" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
          'aria-label="대상별 상대 오차: PatchTST와 LightGBM, 기준선 = 100">')]
    p.append(f'<circle class="ov-pt" cx="{left + 6}" cy="16" r="5"/>' + _t(left + 16, 20, "PatchTST 최고 (seed 평균)"))
    lx = left + 200
    p.append(f'<path class="ov-lgb" d="M{lx + 6},10 L{lx + 12},16 L{lx + 6},22 L{lx},16 Z"/>'
             + _t(lx + 18, 20, "LightGBM (참고)"))
    p.append(_t(width - 8, 20, "감소율 · 판정", "tick", "end"))
    bottom = height - 44
    for v in _log_ticks(lo, hi):
        x = px(v)
        p.append(f'<line class="{"ov-ref" if v == 100 else "grid"}" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
        p.append(_t(x, bottom + 16, f"{v:g}", "tick", "middle"))
    p.append(_t((x0 + x1) / 2, bottom + 34, "상대 오차 (기준선 = 100, 로그 눈금, 작을수록 좋음)", "tick", "middle"))
    y = top
    for task, rows in groups:
        title, base_name = TASKS[task]
        p.append(_t(8, y + 20, f"{title} (기준선: {base_name})", "ov-gh"))
        y += head_h
        for r in rows:
            cy = y + row_h / 2
            pt = remaining(r.patch_mae, r.baseline_mae)
            tip = (f"{r.target} · PatchTST {r.patch_mae:.4g} (상대 오차 {pt:.1f}) · 기준선 {r.baseline_mae:.4g}"
                   + (f" · LightGBM {r.lgbm_mae:.4g} ({remaining(r.lgbm_mae, r.baseline_mae):.1f})" if _ok(r.lgbm_mae) else "")
                   + f" · seed {int(r.patch_seeds)}개")
            p.append(f'<g><title>{escape(tip)}</title>')
            p.append(f'<line class="grid" x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke-dasharray="2 4"/>')
            p.append(_t(left - 12, cy + 4, str(r.target), "", "end"))
            p.append(f'<circle class="ov-pt" cx="{px(pt):.1f}" cy="{cy:.1f}" r="5.5"/>')
            if _ok(r.lgbm_mae):  # drawn over the dot so that an equal LightGBM stays visible
                lx = px(remaining(r.lgbm_mae, r.baseline_mae))
                p.append(f'<path class="ov-lgb" opacity=".9" d="M{lx:.1f},{cy - 6:.1f} L{lx + 6:.1f},{cy:.1f} '
                         f'L{lx:.1f},{cy + 6:.1f} L{lx - 6:.1f},{cy:.1f} Z"/>')
            gain = f"−{r.patch_gain_pct:.0f} %" if _ok(r.patch_gain_pct) and r.patch_gain_pct >= 0 else (
                f"+{-r.patch_gain_pct:.0f} %" if _ok(r.patch_gain_pct) else "-")
            word, cls = DECISION_TEXT.get(r.decision, ("", "muted"))
            p.append(_t(x1 + 16, cy + 4, gain, "ov-val"))
            p.append(_t(x1 + 72, cy + 4, word, f"ov-{cls}"))
            if r.decision == "keep" and int(r.patch_seeds) < 2:
                p.append(_t(x1 + 108, cy + 4, "seed 1개", "tick"))
            p.append("</g>")
            y += row_h
    p.append("</svg>")
    return "".join(p)


def trend_chart(trends: dict[str, list[dict[str, Any]]]) -> str:
    """One panel per task: the PatchTST record (remaining error, target mean) on each day a run started, LightGBM as a
    dashed reference line."""

    panels = [(task, pts) for task, pts in trends.items() if pts]
    if not panels:
        return ""
    width, left, right, ph, gap, top = 960, 64, 112, 170, 54, 40
    height = top + len(panels) * (ph + gap)
    p = [(f'<svg class="chart ov" viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" '
          'aria-label="날짜별 PatchTST 최고 기록 (기준선 = 100, 대상 평균)">')]
    p.append(f'<line class="ov-ptl" x1="{left}" y1="16" x2="{left + 22}" y2="16"/><circle class="ov-pt" cx="{left + 11}" cy="16" r="4"/>'
             + _t(left + 30, 20, "PatchTST 누적 최고 기록"))
    p.append(f'<line class="ov-lgbl" x1="{left + 260}" y1="16" x2="{left + 282}" y2="16"/>' + _t(left + 290, 20, "LightGBM (참고)"))
    if any(pt.get("side") for _, pts in panels for pt in pts):
        p.append(f'<circle class="ov-pt ov-side" cx="{left + 431}" cy="16" r="4"/>'
                 + _t(left + 442, 20, "부가 실험만 한 날 (기록 유지)"))
    y0 = top
    for task, pts in panels:
        title = TASKS[task][0]
        vals = [pt["mean"] for pt in pts] + [pts[0]["lgbm"]] if _ok(pts[0]["lgbm"]) else [pt["mean"] for pt in pts]
        hi = max(vals) * 1.25
        step = next(s for s in (1, 2, 5, 10, 20, 25, 50) if hi / s <= 5)
        hi = math.ceil(hi / step) * step
        ytop, ybot = y0 + 26, y0 + ph
        n = len(pts)
        xs = [left + 40 + (i * (width - left - right - 80) / (n - 1) if n > 1 else (width - left - right - 80) / 2)
              for i in range(n)]

        def py(v: float, ybot: float = ybot, ytop: float = ytop, hi: float = hi) -> float:
            return ybot - v / hi * (ybot - ytop)

        p.append(_t(8, y0 + 12, f"{title} · 대상 {pts[0]['targets']}개", "ov-gh"))
        v = 0
        while v <= hi + 1e-9:
            p.append(f'<line class="grid" x1="{left}" y1="{py(v):.1f}" x2="{width - right}" y2="{py(v):.1f}"/>')
            p.append(_t(left - 8, py(v) + 4, f"{v:g}", "tick", "end"))
            v += step
        if _ok(pts[0]["lgbm"]):
            yl = py(pts[0]["lgbm"])
            p.append(f'<line class="ov-lgbl" x1="{left}" y1="{yl:.1f}" x2="{width - right}" y2="{yl:.1f}"/>')
            p.append(_t(width - right + 8, yl + 4, f"LightGBM {pts[0]['lgbm']:.1f}", "tick"))
        for i in range(1, n):
            dash = " ov-sidel" if pts[i].get("side") else ""
            p.append(f'<line class="ov-ptl{dash}" x1="{xs[i - 1]:.1f}" y1="{py(pts[i - 1]["mean"]):.1f}" '
                     f'x2="{xs[i]:.1f}" y2="{py(pts[i]["mean"]):.1f}"/>')
        every = max(1, math.ceil(n / 12))
        for i, (x, pt) in enumerate(zip(xs, pts)):
            side = pt.get("side")
            tip = (f"{pt['day']:%Y-%m-%d} · 부가 실험 {pt.get('side_runs', 0)}회 · 기본 기록 유지 {pt['mean']:.2f}" if side
                   else f"{pt['day']:%Y-%m-%d} · 상대 오차 {pt['mean']:.2f}")
            p.append(f'<g><title>{tip}</title><circle class="ov-pt{" ov-side" if side else ""}" cx="{x:.1f}" '
                     f'cy="{py(pt["mean"]):.1f}" r="4.5"/></g>')
            if i % every == 0 or i == n - 1:
                if not side or i == n - 1:
                    p.append(_t(x, py(pt["mean"]) - 10, f"{pt['mean']:.1f}", "ov-val", "middle"))
                p.append(_t(x, ybot + 16, f"{pt['day']:%m-%d}", "tick", "middle"))
        y0 += ph + gap
    p.append(_t(width - 8, height - 6, "상대 오차: 기준선 = 100, 대상 기하평균 · 날짜: 실험 시작일 (부가 실험일 포함)",
                "tick", "end"))
    p.append("</svg>")
    return "".join(p)
