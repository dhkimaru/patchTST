"""요약 탭: 현재 결론과 그림 세 개 (대상별 상대 오차, 기준선 = 100 · 일별 기록 추이 · 실험 로드맵).

Numbers come from the ledger (every job of every run, dashboard searches included) through
:func:`ald_lab.stages.patchtst_first`, the same per-target decision the 하이퍼파라미터 탐색 tab shows.
"Remaining error" is a model's MAE as a share of the baseline's (baseline = 100, lower is better); averages over
targets are geometric means so that one target with a large error does not hide the others.
"""

from __future__ import annotations

import math
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .. import stages
from ..ledger import main_records
from ..paths import PROJECT_ROOT

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


def conclusion_points(dec: dict[str, pd.DataFrame], thickness: list[tuple[str, str, float]] = ()) -> list[str]:
    """One short line per task for the top of the 요약 tab: how many targets PatchTST keeps, the average error cut
    against the baseline (geometric mean of remaining error over the targets) and, for reference, how many targets
    it beats LightGBM on. No target names; the per-target picture is the chart below."""

    out = []
    for task, frame in dec.items():
        title, base_name = TASKS[task]
        keep = frame[frame["decision"] == "keep"]
        notes = []
        unsure = int((keep["patch_seeds"] < 2).sum())
        fallback, missing = int((frame["decision"] == "fallback").sum()), int((frame["decision"] == "missing").sum())
        if unsure:
            notes.append(f"seed 반복 확인 필요 {unsure}개")
        if fallback:
            notes.append(f"다른 모델 검토 {fallback}개")
        if missing:
            notes.append(f"미학습 {missing}개")
        text = f"<b>{title}</b>: PatchTST {len(keep)}/{len(frame)}개 채택"
        scored = _scored(frame)
        left = [remaining(r.patch_mae, r.baseline_mae) for r in scored]
        if left:
            text += f", 기준선({base_name}) 대비 오차 평균 {100 - geo_mean(left):.0f}% 감소"
        with_lgbm = [r for r in scored if _ok(r.lgbm_mae)]
        if with_lgbm:
            ahead = sum(1 for r in with_lgbm if r.patch_mae <= r.lgbm_mae)
            text += f" (LightGBM보다 나은 대상 {ahead}/{len(with_lgbm)}개)"
        out.append(text + (f". {', '.join(notes)}" if notes else ""))
    if thickness:
        best = max(thickness, key=lambda t: t[2])
        low = min(t[2] for t in thickness)
        out.append(f"<b>두께 예측</b>: run 수가 적어 딥러닝 대신 고전 ML 사용. 최고 모델 {escape(best[1])}, "
                   f"학습 평균 대비 {low:.0f}~{best[2]:.0f}% 개선")
    return out


def lessons(root: Path = PROJECT_ROOT) -> tuple[str, list[tuple[str, str]]]:
    """'모델을 만들며 확인한 점' (``configs/plan/lessons.yaml``): hand-written takeaways, updated with the README each
    round. Returns (as of, [(title, text)]); empty when the file is missing."""

    path = root / "configs" / "plan" / "lessons.yaml"
    if not path.is_file():
        return "", []
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    points = [(str(p["title"]).strip(), str(p["text"]).strip()) for p in doc.get("points") or []
              if isinstance(p, dict) and p.get("title") and p.get("text")]
    return str(doc.get("as_of") or "").strip(), points


def side_study_note(ledger: pd.DataFrame) -> str | None:
    """One short line under 현재 결론: how many side-study runs there are and the latest one; they stay out of the
    decisions and the record above."""

    if not len(ledger) or "study" not in ledger:
        return None
    side = ledger[ledger["study"].fillna("").astype(str) != ""]
    if not len(side) or "when" not in side or side["when"].isna().all():
        return None
    return (f"부가 실험 {side['exp_dir'].nunique()}회(최근 {side['when'].max():%m-%d})는 채택 판정과 기록에서 빼고 "
            "누적 기록 탭에만 넣음")


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

        p.append(_t(8, y0 + 12, f"{title} · 대상 {pts[0]['targets']}개 · 기준선({TASKS[task][1]}) = 100", "ov-gh"))
        mid = (ytop + ybot) / 2
        p.append(_t(16, mid, "상대 오차 (기준선=100)", "tick ov-axis", "middle", f'transform="rotate(-90 16 {mid:.1f})"'))
        p.append(_t(width - right + 8, ybot + 16, "실험 시작일", "tick ov-axis"))
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
    p.append(_t(width - 8, height - 6, "세로: 그날까지의 PatchTST 최고 MAE ÷ 기준선 MAE × 100, 대상 기하평균 (낮을수록 좋음) · "
                "가로: 실험 시작일 (부가 실험일 포함)", "tick", "end"))
    p.append("</svg>")
    return "".join(p)


ROADMAP_CSS = (".roadmap svg{width:100%;height:auto;max-width:980px;display:block}"
               ".roadmap .rm-dark{display:none}"
               ':root[data-theme="dark"] .roadmap .rm-light{display:none}'
               ':root[data-theme="dark"] .roadmap .rm-dark{display:block}'
               '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .roadmap .rm-light{display:none}'
               ':root:not([data-theme="light"]) .roadmap .rm-dark{display:block}}'
               "@media print{.roadmap .rm-light{display:block!important}.roadmap .rm-dark{display:none!important}}")


_TEXT_TAG = re.compile(r"<text ([^>]*)>")


def _pin_text_style(svg: str) -> str:
    """The report styles every ``svg text`` (12 px, muted colour), which beats SVG presentation attributes; repeat
    each label's size, weight and colour as an inline style so the figure looks the same as in the README."""

    def fix(m: re.Match) -> str:
        attrs = m.group(1)
        found = {k: re.search(rf'{k}="([^"]+)"', attrs) for k in ("font-size", "font-weight", "fill")}
        style = ";".join(f"{k}:{v.group(1)}{'px' if k == 'font-size' else ''}" for k, v in found.items() if v)
        return f'<text {attrs} style="{style}">' if style else m.group(0)

    return _TEXT_TAG.sub(fix, svg)


def roadmap_figure(root: Path = PROJECT_ROOT) -> str:
    """The experiment roadmap that the README shows (``scripts/readme_roadmap.py`` →
    ``docs/images/results_flow_{light,dark}.svg``), inlined so the offline report needs no file next to it.
    Both themes are inlined and CSS shows the one that matches the page; empty when the files are missing."""

    files = {m: root / "docs" / "images" / f"results_flow_{m}.svg" for m in ("light", "dark")}
    if not all(p.is_file() for p in files.values()):
        return ""
    # inline SVG needs no namespace, and the report keeps no external URL in it
    svgs = {m: p.read_text(encoding="utf-8").strip().replace(' xmlns="http://www.w3.org/2000/svg"', "", 1)
            for m, p in files.items()}
    if not all(v.startswith("<svg") for v in svgs.values()):
        return ""
    body = "".join(f'<div class="rm-{m}">{_pin_text_style(v)}</div>' for m, v in svgs.items())
    return f'<style>{ROADMAP_CSS}</style><div class="roadmap">{body}</div>'

