"""Render :class:`ReportData` as one self-contained HTML file (no network, no server).

Tabs follow the reporting outline: 요약 → In → Out → Function(모델) → Code
Flow Chart → Hyper Parameter → 하이퍼파라미터 탐색 → 성능표 → 가중치·실행 파일 → 추론 시연 → 특이사항 → 이전 실험 →
다음 실험. Opened through the lab server (``ald-lab-server``) the 하이퍼파라미터 탐색 tab also queues experiments.
Printing (or "PDF로 저장") lays every tab out one after another.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dashboard.tokens import DARK, FONT, LIGHT, STATUS
from ..data.catalog import CYCLE_STEPS
from ..data.raw import DATE_COLUMN, META_COLUMNS, TIME_COLUMN
from ..journal import STAGE_LABELS
from ..models import defaults as model_defaults
from ..models.cards import model_card, task_card
from ..paths import PROJECT_ROOT
from . import archs, diagram, lab, overview, player
from . import design as design_notes
from .analysis import (
    CLEAR_SE,
    FACTOR_NAMES,
    SECONDS,
    config_table,
    device_order,
    effect_summary,
    factor_effects,
    horizon_columns,
    horizon_label,
    job_timeline,
    plain,
    setting_value,
    speed_table,
)
from .charts import (
    improvement_bars,
    line_chart,
    loss_curve,
    series_chart,
    timeline_chart,
    value_bars,
)
from .codemap import HUB_CALLERS, MODULE_CODE, PACKAGE_DIR, CodeMap
from .codemap import scan as scan_code
from .collect import Experiment, Pick, ReportData
from .dataview import INFO_COLUMNS, ISSUE_NAMES
from .diagram import (
    Box,
    count_flow,
    gather_figure,
    queue_figure,
    split_strip,
    step_lanes,
    window_rule,
)

TABS = (
    ("summary", "요약"),
    ("inputs", "In (입력)"),
    ("outputs", "Out (출력)"),
    ("models", "Function (모델)"),
    ("flow", "Code Flow Chart"),
    ("hparams", "Hyper Parameter"),
    ("lab", "하이퍼파라미터 탐색"),
    ("results", "성능표"),
    ("weights", "가중치·실행 파일"),
    ("demo", "추론 시연"),
    ("notes", "특이사항"),
    ("history", "이전 실험"),
    ("plan", "다음 실험"),
)
# Left sidebar: tabs grouped by the reporting order (data → model → hyperparameters → results → records).
NAV_GROUPS = (
    ("요약", "home", ("summary",)),
    ("데이터", "data", ("inputs", "outputs")),
    ("모델", "model", ("models", "flow")),
    ("하이퍼파라미터", "tune", ("hparams", "lab")),
    ("결과", "chart", ("results", "weights", "demo")),
    ("기록", "notes", ("notes", "history", "plan")),
)
NAV_ICONS = {
    "home": "M3 11 12 4l9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
    "data": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
    "model": "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4",
    "tune": "M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1M15 4v4M9 10v4M17 16v4",
    "chart": "M3 21h18M6 17v-6M11 17V5M16 17v-9M21 17v-4",
    "notes": "M6 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6zM6 3v18M9.5 8h6M9.5 12h6M9.5 16h4",
    "chev": "M7 10l5 5 5-5",
    "menu": "M4 7h16M4 12h16M4 17h16",
}
BASELINE_NAMES = {"persistence": "현재값 유지", "template": "레시피 평균", "mean": "학습 평균"}
REASONS = {"BACKWARD_TIMESTAMP_JUMP": "시간 역행, 중복 블록 의심"}
TARGET_NAMES = {"thk_mean": "두께 평균", "uf_pct_mean": "웨이퍼 내 균일도 (U/F)", "wtow_pct": "웨이퍼 간 균일도 (WtoW)",
                "thk_center": "중심 두께", "thk_edge": "가장자리 두께", "edge_vs_center_pct": "가장자리/중심 차이"}


# ----------------------------------------------------------------------------- helpers

def e(value: Any) -> str:
    return escape("" if value is None else str(value))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return e(value)
    if not math.isfinite(number):
        return "-"
    if number != 0 and (abs(number) >= 10 ** digits or abs(number) < 10 ** -(digits - 1)):
        return f"{number:.{digits - 1}e}"
    return f"{number:.{digits}g}"


def spread(value: Any) -> str:
    """A seed standard deviation: two significant digits, no exponent for ordinary sizes."""

    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2g}"


def pct(value: Any) -> str:
    """Improvement with its word, so the sign never carries the meaning alone."""

    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    value = float(value)
    cls = "good" if value >= 0 else "bad"
    word = "개선" if value >= 0 else "악화"
    return f'<span class="st {cls}">{value:+.1f} % {word}</span>'


def minutes(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    value = float(value)
    if value < 1:
        return f"{value * 60:.0f}초"
    if value < 120:
        return f"{value:.1f}분"
    return f"{value / 60:.1f}시간"


def table(header: list[str], rows: list[list[str]], *, cls: str = "") -> str:
    """``rows`` hold ready HTML cells (escape text before passing it)."""

    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def kv(rows: list[tuple[str, str]]) -> str:
    return table(["항목", "내용"], [[e(k), v] for k, v in rows], cls="kv")


def tile(label: str, value: str, note: str = "") -> str:
    return f'<div class="tile"><div class="tl">{e(label)}</div><div class="tv">{value}</div><div class="tn">{note}</div></div>'


def target_name(target: str, unit: str = "") -> str:
    name = TARGET_NAMES.get(target)
    text = f"{target} ({name})" if name else target
    return f"{text} [{unit}]" if unit else text


LABEL_BASE = {"patchtst": "PatchTST", "patchtst_mix": "PatchTST 혼합", "lightgbm": "LightGBM", "ridge": "Ridge",
              "dlinear": "DLinear", "itransformer": "iTransformer", "gru": "GRU", "pls": "PLS", "persistence": "현재값 유지",
              "template": "레시피 평균", "mean": "학습 평균"}
LABEL_KEYS = {"d_model": "d_model {}", "patch_seconds": "patch {}초", "stride_seconds": "간격 {}초", "n_heads": "head {}",
              "n_layers": "층 {}", "d_ff": "FFN {}", "dropout": "dropout {}", "revin_center": "RevIN {}", "revin": "RevIN {}",
              "train.lr": "lr {}", "channel_mode": "{}", "num_leaves": "잎 {}", "learning_rate": "학습률 {}",
              "n_components": "성분 {}", "kernel_seconds": "이동평균 {}초", "d_rep": "표현 {}", "hidden": "hidden {}",
              "min_child_samples": "잎 최소 {}", "objective": "손실 {}"}
LABEL_VALUES = {"mixing": "혼합", "independent": "독립"}
_LABEL = re.compile(r"^(?P<base>[^\[@#]+?)(?:#(?P<trial>\d+))?(?:\[(?P<grid>[^\]]*)\])?(?:@(?P<inp>.+))?$")
_TOP = re.compile(r"^(?P<model>[a-z_]+?)_top(?P<rank>\d+)$")


def nice_label(label: str) -> str:
    """A job label as people read it: ``patchtst_mix[d_model=32,patch_seconds=16]`` → "PatchTST 혼합 · d_model 32 ·
    patch 16초", ``patchtst#03`` → "PatchTST #03", ``lightgbm@all`` → "LightGBM · 입력 all"."""

    match = _LABEL.match(str(label))
    if not match:
        return str(label).replace("@", " · 입력 ")
    raw = match["base"]
    top = _TOP.match(raw)
    base = (f"{LABEL_BASE.get(top['model'], top['model'])} 상위 {top['rank']}" if top
            else LABEL_BASE.get(raw, raw))
    parts = [base + (f" #{match['trial']}" if match["trial"] else "")]
    for item in (match["grid"] or "").split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        value = LABEL_VALUES.get(value.strip(), value.strip())
        shape = LABEL_KEYS.get(key.strip())
        parts.append(shape.format(value) if shape else f"{key.strip()} {value}")
    if match["inp"]:
        parts.append(f"입력 {match['inp']}")
    return " · ".join(parts)


def _git_commit() -> str:
    """The code version: from git in a checkout, else from ``COMMIT``, which ``git archive`` fills in (export-subst)."""

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=PROJECT_ROOT)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        stamp = (Path(__file__).with_name("COMMIT")).read_text(encoding="utf-8").strip()
    except OSError:
        return "-"
    return stamp if stamp and not stamp.startswith("$Format") else "-"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _horizons(exp: Experiment) -> list[float]:
    if exp.task == "virtual_sensor":
        return [0.0]
    return [float(h) for h in (exp.config.get("window") or {}).get("horizons_seconds", [10, 20, 30])]


def _context(exp: Experiment) -> float:
    return float((exp.config.get("window") or {}).get("context_seconds", 192))


def _baseline(exp: Experiment) -> str:
    return BASELINE_NAMES.get(str(exp.summary.get("baseline")), "기준선")


def _exp_title(exp: Experiment) -> str:
    card = task_card(exp.task)
    smoke = " · 축소 확인(smoke)" if exp.smoke else ""
    return f"{exp.name} ({card['name']}){smoke}"


def _eval_word(exp: Experiment) -> str:
    return "leave-one-run-out (학습+검증 run)" if exp.task == "thickness" else "검증 run"


# ----------------------------------------------------------------------------- tabs

def _summary(data: ReportData) -> str:
    exps = data.experiments
    ledger = data.ledger if data.ledger is not None else pd.DataFrame()
    dec = overview.decisions(ledger)
    trends = {task: overview.trend(ledger, task, frame) for task, frame in dec.items()}
    thickness = [(p.target, model_card(p.model)["name"], float(p.improvement_pct)) for x in exps if x.task == "thickness"
                 for p in x.picks if p.improvement_pct is not None]
    out = []
    if len(ledger) and "exp_dir" in ledger:
        hours = float(ledger["fit_minutes"].fillna(0).sum()) / 60 if "fit_minutes" in ledger else 0.0
        out.append(f'<p class="sub">누적 실험 {ledger["exp_dir"].nunique()}회 · 학습 모델 {len(ledger):,}개 · '
                   f'학습 시간 {hours:.0f}시간 (대시보드 탐색 포함) · 실험이 끝날 때마다 갱신</p>')
    if data.notes:
        out.append('<h3>오늘 메모</h3><ul>' + "".join(f"<li>{e(n)}</li>" for n in data.notes) + "</ul>")
    points = overview.conclusion_points(dec, trends, thickness)
    if points:
        out.append('<h3>현재 결론</h3><div class="concl"><ul>' + "".join(f"<li>{p}</li>" for p in points)
                   + "</ul></div>")
    chart = overview.target_chart(dec)
    if chart:
        out.append("<h3>대상별 성능 - 기준선 대비 상대 오차</h3>"
                   '<p class="note">기준선(미래값 예측: 현재값 유지, 가상 센서: 레시피 평균)의 오차를 100으로 환산한 값입니다. '
                   "작을수록 좋으며, 파란 점이 주황 마름모보다 왼쪽에 있으면 PatchTST가 LightGBM보다 우수합니다. "
                   "점에 마우스를 올리면 실제 MAE가 표시됩니다.</p>" + chart)
    trend = overview.trend_chart(trends)
    if trend:
        out.append("<h3>PatchTST 최고 기록 추이 (일별)</h3>"
                   '<p class="note">날짜별로 그날까지의 PatchTST 최고 기록(같은 설정의 seed 평균)을 대상 평균으로 나타냈습니다. '
                   "낮을수록 좋습니다.</p>" + trend)
    status = _patchtst_status(data)
    if status:
        out.append("<details><summary>PatchTST 현황 - 대상별 성능 및 채택 여부 (실험 완료 시 갱신)</summary>"
                   + status + "</details>")
    out.append("<details><summary>과제별 최고 성능 - 정규 실험 기준 (대시보드 탐색 제외)</summary>")
    out.append('<p class="note">개선율 = (기준선 오차 − 모델 오차) ÷ 기준선 오차. 오차는 검증 run의 MAE(평균 절대 오차, 물리 단위)입니다. '
               "모델 선택은 검증 run으로만 했고, 시험 run은 최종 후보에서만 엽니다. 예측 거리별 오차와 모델별 순위는 "
               '<a href="#results">성능표</a>에 있습니다. 대시보드에서 실행한 탐색은 포함하지 않습니다.</p>')
    out.append(_purpose_table(exps, compact=True) + "</details>")
    if data.running:
        rows = [[e(r["name"]), e(r["stamp"]), f'{r["done"]} / {r["jobs"]}'] for r in data.running]
        out.append("<h3>진행 중인 실험</h3>" + table(["실험", "시작", "완료 작업 / 전체"], rows)
                   + '<p class="note">진행 중이거나 중단된 실험입니다. 완료되면 결과가 반영됩니다.</p>')
    out.append(_today(data))
    out.append('<h3>보고 순서 (4단계)</h3><ol class="steps">'
               '<li><b>원본 데이터 설명</b> → <a href="#inputs">In</a></li>'
               '<li><b>전처리</b> → <a href="#inputs">In</a> · <a href="#flow">Code Flow Chart</a></li>'
               '<li><b>하이퍼파라미터·학습</b> → <a href="#models">Function</a> · <a href="#hparams">Hyper Parameter</a></li>'
               '<li><b>소요 시간·강점·약점</b> → <a href="#results">성능표</a> · <a href="#notes">특이사항</a></li></ol>')
    return "".join(out)


def _patchtst_status(data: ReportData) -> str:
    """Per forecast / virtual-sensor target: the best PatchTST so far (every run, dashboard runs included), its gain
    over the baseline and the PatchTST-first decision."""

    from .. import stages

    ledger = data.ledger
    if ledger is None or not len(ledger):
        return ""
    rows = []
    for task in ("forecast", "virtual_sensor"):
        frame = stages.patchtst_first(ledger, task)
        for r in frame.itertuples():
            has = r.patch_mae is not None and r.patch_mae == r.patch_mae
            final = model_card(r.final)["name"] if r.final else "-"
            rows.append([e(TASK_SHORT.get(task, task)), e(target_name(r.target, r.unit)), fmt(r.baseline_mae),
                         fmt(r.patch_mae) if has else "-",
                         pct(r.patch_gain_pct) if has and r.patch_gain_pct == r.patch_gain_pct else "-",
                         lab._decision_chip(r), f"{e(final)} <span class='sub'>{fmt(r.final_mae)}</span>"])
    if not rows:
        return ""
    return (table(["과제", "대상 [단위]", "기준선 오차", "PatchTST 최고", "기준선 대비", "판정", "최종 모델"], rows)
            + f'<p class="sub">{lab.decision_rule_text()} 자세한 내용은 '
            '<a href="#lab">하이퍼파라미터 탐색</a> 탭에 있습니다.</p>')


def _horizon_set(exps: list[Experiment]) -> list[str]:
    """Horizon columns shown in cross-experiment tables (10/20/30 s unless the experiments use others)."""

    columns = {c for exp in exps if exp.task == "forecast" for c in horizon_columns(exp.table.columns)}
    return horizon_columns(columns) or ["mae_10s", "mae_20s", "mae_30s"]


def _row_of(exp: Experiment, target: str, label: str) -> dict[str, Any]:
    rows = exp.table[(exp.table["target"] == target) & (exp.table["label"] == label)]
    return rows.iloc[0].to_dict() if len(rows) else {}


def _hcells(values: dict[str, Any], columns: list[str], digits: int = 3) -> list[str]:
    return [fmt(values.get(c), digits) if values.get(c) is not None and not pd.isna(values.get(c)) else "-"
            for c in columns]


def _test_horizon_values(exp: Experiment, target: str, label: str) -> dict[str, float]:
    """``mae_10s``-style values of the test run: from test_summary.csv when written, else metrics_test.json."""

    if exp.test is not None:
        rows = exp.test[(exp.test["target"] == target) & (exp.test["label"] == label)]
        cols = horizon_columns(exp.test.columns)
        if len(rows) and cols:
            return {c: rows.iloc[0][c] for c in cols}
    by_h = exp.test_horizons.get((target, label)) or {}
    return {f"mae_{k}": v for k, v in by_h.items()}


def _purpose_table(exps: list[Experiment], *, compact: bool = False) -> str:
    hcols = [] if compact else _horizon_set(exps)
    rows = []
    for exp in exps:
        card = task_card(exp.task, horizons=_horizons(exp), context=_context(exp))
        tested = _test_lookup(exp)
        for p in exp.picks:
            test = tested.get((p.target, p.label))
            val = _row_of(exp, p.target, p.label) if exp.task == "forecast" else {}
            test_cell = '<span class="sub">미실시</span>'
            if test:
                th = _test_horizon_values(exp, p.target, p.label)
                test_cell = fmt(test["mae"]) + " " + pct(test["improvement_pct"])
                if any(c in th for c in hcols):
                    test_cell += ('<div class="sub">' + " / ".join(_hcells(th, hcols)) + f" ({'·'.join(horizon_label(c) for c in hcols)})</div>")
            rows.append([
                e(card["name"]) + f'<div class="sub">{e(exp.name)}</div>',
                e(target_name(p.target, p.unit)),
                f"{e(_baseline(exp))}<div class='sub'>{fmt(p.baseline_mae)}</div>",
                e(nice_label(p.label)) + f'<div class="sub">{e(model_card(p.model)["family"])}</div>',
                fmt(p.mae), *_hcells(val, hcols),
                pct(p.improvement_pct),
                test_cell,
            ])
    if not rows:
        return '<p class="empty">아직 끝난 실험이 없습니다.</p>'
    return table(["과제", "대상 [단위]", "기준선 오차", "최고 모델", "모델 오차", *[horizon_label(c) for c in hcols],
                  "개선율 (검증)", "시험 run"], rows)


def _test_lookup(exp: Experiment) -> dict[tuple[str, str], dict[str, Any]]:
    if exp.test is None:
        return {}
    out = {}
    for row in exp.test.to_dict(orient="records"):
        imp = row.get("improvement_pct")
        out[(row["target"], row["label"])] = {"mae": row.get("mae"),
                                              "improvement_pct": None if pd.isna(imp) else imp}
    return out


def _today(data: ReportData) -> str:
    journal = data.journal
    if journal is None or not len(journal):
        return ""
    local = journal["started_at"].dt.tz_convert(data.created.astimezone().tzinfo)
    last_day = local.dt.date.max()
    day = journal[local.dt.date == last_day]
    counts = day.groupby("stage").size()
    hours = float(day["duration_s"].fillna(0).sum()) / 3600
    parts = [f"{STAGE_LABELS.get(s, s)} {int(n)}건" for s, n in counts.items()]
    failed = int((day["status"] == "failed").sum())
    return (f"<h3>최근 작업일 ({last_day:%Y-%m-%d}) 기록</h3>"
            f"<p>{e(' · '.join(parts))} · 기록된 작업 시간 합계 {hours:.1f}시간"
            + (f' · <span class="st bad">실패 {failed}건</span>' if failed else "") + "</p>")


ISSUE_ACTIONS = {"MISSING_RUN_FILE": "해당 없음", "EXSITU_SHEET_FORMULA": "원시 측정점에서 다시 계산한 값을 씀",
                 "EXSITU_DATE_MISMATCH": "제목 번호로 연결 (STANDARD 연속진행 N = run N)"}
SEVERITY_TEXT = {"critical": ("치명", "bad"), "warning": ("경고", "warn"), "info": ("정보", "")}
NORMALIZE_EXAMPLES = ("VG11", "TempAct_C", "APCValveMon", "MFCMon_DCS", "MFCMon_NH3", "AUXMon_R.PWR")
COPY_THRESHOLD = 0.9999  # training.windows.COPY_THRESHOLD (that module needs torch; a test keeps the two equal)


def raw_meta_columns() -> list[str]:
    return [*META_COLUMNS, DATE_COLUMN, TIME_COLUMN]


def _run_no(run_id: str) -> str:
    return run_id.rsplit("_", 1)[-1]


def _thickness_excluded(data: ReportData) -> list[str]:
    exp = next((x for x in data.experiments if x.task == "thickness"), None)
    return [str(r) for r in ((exp.config.get("vm") or {}).get("exclude_runs") or [])] if exp else []


def _issue_action(kind: str, run_id: str, excluded: set[str], thick: set[str]) -> str:
    if kind in ISSUE_ACTIONS:
        return ISSUE_ACTIONS[kind]
    if run_id in excluded:
        return "run 제외 (모든 과제)"
    if kind in ("SAMPLING_GAP", "BACKWARD_TIMESTAMP_JUMP"):
        return "끊긴 곳에서 구간을 나눔 (window가 넘지 않음)"
    if run_id in thick:
        return "두께 예측에서 제외"
    return "그대로 사용 (차이 작음)" if kind == "THICKNESS_OUTLIER" else "그대로 사용 (원인 확인 중)"


def _count_flow(data: ReportData) -> str:
    ov, sp, dv = data.overview or {}, data.split or {}, data.data_view
    if not ov or not sp:
        return ""
    missing = [int(n) for n in ov.get("missing_run_numbers") or []]
    excluded = [x["run_id"] for x in sp.get("excluded") or []]
    parts = [("train", "학습"), ("validation", "검증"), ("test", "시험")]
    counts = [len(sp.get(k) or []) for k, _ in parts]
    n_runs = int(ov.get("n_runs") or 0)
    lanes = [("run (배치 1회 = CSV 1개)", [
        (f"{n_runs + len(missing)}", f"run 번호 001~{n_runs + len(missing):03d}"), (f"{n_runs}", "CSV 파일"),
        (f"{n_runs - len(excluded)}", "사용한 run"), (" / ".join(str(c) for c in counts), "학습 / 검증 / 시험")],
        [(f"−{len(missing)} 파일 없음", "·".join(f"{n:03d}" for n in missing)),
         (f"−{len(excluded)} 시간 역행", "·".join(_run_no(r) for r in excluded) + " (중복 블록)"),
         ("run 단위로 나눔", "시간 순서 (split_v2)")])]
    runs = dv.runs if dv is not None else None
    bundles = dv.bundles if dv is not None else {}
    n_rows = int(ov.get("n_rows") or 0)
    if runs is not None and n_rows:
        dropped = int(runs.loc[runs["run_id"].isin(excluded), "rows"].sum())
        boxes = [(f"{n_rows:,}", "원본 행 (0.5초)"), (f"{n_rows - dropped:,}", f"{'·'.join(_run_no(r) for r in excluded)} 제외 후")]
        drops = [(f"−{dropped:,}행", "·".join(_run_no(r) for r in excluded))]
        if all(k in bundles for k, _ in parts):
            rows_1s = [int(bundles[k]["rows"]) for k, _ in parts]
            boxes += [(f"{sum(rows_1s):,}", "1초 행"), (" / ".join(f"{r:,}" for r in rows_1s), "학습 / 검증 / 시험")]
            drops += [("0.5초 2개 → 1초 평균", "끊긴 구간 안에서만"), ("run 단위로 나눔", "")]
        lanes.append(("행 (시점)", boxes, drops))
    cols = dv.columns if dv is not None else None
    if cols is not None:
        status = cols["status"].value_counts()
        n_sig = len(cols)
        inputs = [int(info.get("inputs")) for x in data.experiments for info in ((x.prep or {}).get("targets") or {}).values()
                  if info.get("inputs")]
        lanes.append(("신호 (컬럼)", [
            (f"{n_sig + INFO_COLUMNS}", "CSV 컬럼"), (f"{n_sig}", "신호 컬럼"), (f"{int(status.get('OK', 0))}", "사용 가능한 신호"),
            (f"{min(inputs)} ~ {max(inputs)}" if inputs else "실험별", "실험별 입력 수")],
            [(f"−{INFO_COLUMNS} 정보 컬럼", "시각·step 이름 등"),
             (f"−{int(status.get('CONSTANT', 0))} 상수 · −{int(status.get('DUPLICATE', 0))} 중복", "값이 안 변함 / 복사"),
             ("실험마다 선택", "아래 ⑥")]))
    thick = _thickness_excluded(data)
    if thick:
        pool = set(sp.get("train") or []) | set(sp.get("validation") or [])
        out_of_pool = [r for r in thick if r in pool]
        lanes.append(("두께 예측 run", [(f"{len(pool)}", "학습 + 검증 run"), (f"{len(pool) - len(out_of_pool)}", "두께 학습·평가 run")],
                      [(f"−{len(out_of_pool)} 두께 이상", "·".join(_run_no(r) for r in out_of_pool))]))
    return count_flow(lanes, label="단계별 데이터 수")


def _split_strip(data: ReportData) -> str:
    sp, ov = data.split or {}, data.overview or {}
    if not sp:
        return ""
    status = {r: k for k in ("train", "validation", "test") for r in sp.get(k) or []}
    status.update({x["run_id"]: "excluded" for x in sp.get("excluded") or []})
    numbers = [int(_run_no(r)) for r in status] + [int(n) for n in ov.get("missing_run_numbers") or []]
    thick = set(_thickness_excluded(data))
    cells = []
    for n in range(1, max(numbers) + 1):
        run_id = f"standard_trace_{n:03d}"
        cells.append((n, status.get(run_id, "missing"), run_id in thick))
    counts = {k: sum(1 for _, s, _ in cells if s == k) for k in ("train", "validation", "test", "excluded", "missing")}
    legend = ('<div class="legend">'
              f'<span><i class="sw-train"></i>학습 {counts["train"]}</span>'
              f'<span><i class="sw-validation"></i>검증 {counts["validation"]}</span>'
              f'<span><i class="sw-test"></i>시험 {counts["test"]}</span>'
              f'<span><i class="sw-excluded"></i>제외 {counts["excluded"]}</span>'
              f'<span><i class="sw-missing"></i>파일 없음 {counts["missing"]}</span>'
              + (f'<span><i class="sw-flag"></i>오른쪽 위 표시: 두께 예측에서만 제외 {len(thick)}</span>' if thick else "")
              + "</div>")
    return legend + split_strip(cells, label="run 번호별 분할")


def _excluded_runs(data: ReportData) -> str:
    dv, sp, ov = data.data_view, data.split or {}, data.overview or {}
    issues = dv.issues if dv is not None and dv.issues is not None else pd.DataFrame(columns=["run_id", "kind", "detail"])
    split_of = {r: name for k, name in (("train", "학습"), ("validation", "검증"), ("test", "시험")) for r in sp.get(k) or []}
    excluded = {x["run_id"]: x.get("reason") for x in sp.get("excluded") or []}
    thick = _thickness_excluded(data)

    def detail(run_id: str, kinds: tuple[str, ...]) -> str:
        found = issues[(issues["run_id"] == run_id) & issues["kind"].isin(kinds)]["detail"]
        return e(" · ".join(found)) or "-"

    rows = []
    missing = [int(n) for n in ov.get("missing_run_numbers") or []]
    if missing:
        rows.append([e(" · ".join(f"{n:03d}" for n in missing)), '<span class="st">파일 없음</span>',
                     "번호 순서상 비어 있는 run (CSV가 없음)", "-", "해당 없음"])
    for run_id, reason in excluded.items():
        rows.append([f"<b>{_run_no(run_id)}</b>", '<span class="st bad">모든 과제에서 제외</span>',
                     e(REASONS.get(reason, reason)), detail(run_id, ("BACKWARD_TIMESTAMP_JUMP", "SAMPLING_GAP")),
                     "학습·검증·시험 어디에도 쓰지 않음"])
    gaps = sorted(set(issues.loc[issues["kind"] == "SAMPLING_GAP", "run_id"]) - set(excluded))
    for run_id in gaps:
        rows.append([f"<b>{_run_no(run_id)}</b>", '<span class="st good">사용 (구간 2개로 분할)</span>',
                     "0.5초 간격 끊김", detail(run_id, ("SAMPLING_GAP",)),
                     f"끊긴 곳을 넘는 window는 만들지 않음 ({split_of.get(run_id, '-')} run)"])
    for run_id in thick:
        rows.append([f"<b>{_run_no(run_id)}</b>", '<span class="st warn">두께 예측에서만 제외</span>', "두께 이상",
                     detail(run_id, ("THICKNESS_OUTLIER", "ABNORMAL_END")),
                     f"미래값 예측·가상 센서에는 사용 ({split_of.get(run_id, '-')} run)"])
    return table(["run", "처리", "이유", "근거 (데이터 개요의 품질 점검)", "영향"], rows) if rows else ""


def _excluded_columns(data: ReportData) -> str:
    dv = data.data_view
    if dv is None or dv.columns is None:
        return ""
    cols = dv.columns
    const = cols.loc[cols["status"] == "CONSTANT", "column"].tolist()
    dup = cols.loc[cols["status"] == "DUPLICATE", ["column", "duplicate_of"]]
    meta = [*raw_meta_columns(), "빈 마지막 열"]
    out = [f'<p class="note"><b>정보 컬럼 {INFO_COLUMNS}개</b>: {e(" · ".join(meta))}. 입력으로 쓰지 않고, 시각과 '
           "step 구간(공정·ALD 사이클) 판별에만 씁니다.</p>",
           f'<p class="note"><b>상수 {len(const)}개</b>: 74개 run 전체에서 값이 변하지 않는 컬럼입니다.</p>',
           '<div class="chips">' + "".join(f"<code>{e(c)}</code>" for c in const) + "</div>"]
    if len(dup):
        out.append(f'<p class="note"><b>중복 {len(dup)}개</b>: 모든 행에서 다른 컬럼과 값이 같아(완전 복사) 원본만 남겼습니다.</p>')
        groups: dict[str, list[str]] = {}
        for r in dup.itertuples():
            groups.setdefault(str(r.duplicate_of), []).append(str(r.column))
        out.append(table(["남긴 원본", "같은 값이라 뺀 컬럼"],
                         [[f"<code>{e(k)}</code>", " ".join(f"<code>{e(c)}</code>" for c in v)] for k, v in groups.items()]))
    copies = int((data.overview or {}).get("n_near_copy_pairs") or 0)
    if copies:
        out.append(f'<p class="note">이 밖에 거의 같은 신호 쌍 {copies}개(예: TempAct_C ≈ TempAct_HT.C)는 입력에 남겨 두되, '
                   f"가상 센서에서는 학습 run 기준 상관 |r| ≥ {COPY_THRESHOLD:g}인 복사본을 대상과 함께 뺍니다 (⑥ 표).</p>")
    return "".join(out)


def _issue_table(data: ReportData) -> str:
    dv, sp = data.data_view, data.split or {}
    if dv is None or dv.issues is None or not len(dv.issues):
        return ""
    excluded = {x["run_id"] for x in sp.get("excluded") or []}
    thick = set(_thickness_excluded(data))
    order = {"critical": 0, "warning": 1, "info": 2}
    issues = dv.issues.assign(_o=dv.issues["severity"].map(order)).sort_values(["_o", "run_id"], kind="stable")
    rows = []
    for r in issues.itertuples():
        word, cls = SEVERITY_TEXT.get(r.severity, (r.severity, ""))
        run = _run_no(r.run_id) if str(r.run_id).startswith("standard_trace_") else str(r.run_id)
        rows.append([f'<span class="st {cls}">{e(word)}</span>', e(run), e(ISSUE_NAMES.get(r.kind, r.kind)), e(r.detail),
                     e(_issue_action(r.kind, str(r.run_id), excluded, thick))])
    counts = issues["severity"].value_counts()
    head = " · ".join(f"{SEVERITY_TEXT[k][0]} {int(counts[k])}" for k in ("critical", "warning", "info") if k in counts)
    return (f"<details><summary>품질 점검 결과 {len(issues)}건 ({e(head)}) - 항목별 처리 내용</summary>"
            + table(["심각도", "run", "종류", "내용", "처리"], rows) + "</details>")


def _pipeline(data: ReportData) -> str:
    dv, sp = data.data_view, data.split or {}
    bundles = dv.bundles if dv is not None else {}
    parts = ("train", "validation", "test")
    full = all(k in bundles for k in parts)
    seg = sum(int(bundles[k].get("segments", 0)) for k in parts) if full else None
    n_runs = sum(len(sp.get(k) or []) for k in parts)
    nan = sum(int(bundles[k].get("nan_filled", 0)) for k in parts) if full else None
    rows_1s = sum(int(bundles[k]["rows"]) for k in parts) if full else None
    process = sum(int(bundles[k].get("process_rows", 0)) for k in parts) if full else None
    cycle = sum(int(bundles[k].get("cycle_rows", 0)) for k in parts) if full else None
    n_ok = int((dv.columns["status"] == "OK").sum()) if dv is not None and dv.columns is not None else None
    ov = data.overview or {}
    figure = step_lanes([
        ("run마다 (학습·검증·시험, 새 CSV 예측까지 같은 함수)", "now", [
            ("CSV 읽기", "0.5초 · 207컬럼"),
            ("끊긴 곳에서 나누기", f"run {n_runs} → 구간 {seg}" if seg else "간격이 깨지는 곳"),
            ("빈 값 채우기", f"앞 값으로 · 이번 {nan}개" if nan is not None else "앞 값으로"),
            ("1초 평균", "0.5초 2개씩, 구간 안에서")], ""),
        ("모델에 넣기 전", "now", [
            ("신호 고르기", f"{n_ok}개 → 실험별" if n_ok else "상수·중복 제외"),
            ("구간 표시", "공정 / ALD 사이클"),
            ("정규화", "학습 run 평균·표준편차"),
            ("window 위치", "구간을 넘지 않는 끝만")], ""),
    ], label="전처리 순서")
    raw_after = int(ov.get("n_rows") or 0) - (int(dv.runs.loc[dv.runs["run_id"].isin(
        [x["run_id"] for x in sp.get("excluded") or []]), "rows"].sum()) if dv is not None and dv.runs is not None else 0)
    rows = [
        ["① 시간 끊김 분할", "0.5초 간격이 깨지는 곳(끊김·역행)에서 구간을 나눔", "window가 끊긴 곳을 건너 학습되지 않도록",
         f"run {n_runs}개 → 구간 {seg}개" if seg else "-"],
        ["② 빈 값 채우기", "빈 값은 바로 앞 값(없으면 뒤 값)으로", "계산이 멈추지 않도록",
         f"{nan:,}개" + (" (이번 데이터에는 빈 값 없음)" if nan == 0 else "") if nan is not None else "-"],
        ["③ 1초 평균", "0.5초 값 2개를 평균 (구간 안에서만)", "학습 속도 2배, 측정 잡음 감소. 이전 실험에서 0.5초가 더 나빴음 (GM1)",
         f"{raw_after:,}행 → {rows_1s:,}행" if rows_1s and raw_after else "-"],
        ["④ 신호 고르기", "상수·중복 컬럼 제외, 실험마다 입력 세트 선택", "정보가 없는 입력 제거",
         f"200 → {n_ok}개 → 실험별 (⑥)" if n_ok else "-"],
        ["⑤ 가상 센서 입력 제외", f"대상과 같은 계열 센서 + 상관 |r| ≥ {COPY_THRESHOLD:g} 복사본을 입력에서 뺌",
         "대상의 복사본으로 추정하는 것을 방지", "대상마다 다름 (⑥ 표)"],
        ["⑥ 구간 표시", "공정 구간 = 첫 START ~ 마지막 END · ALD 사이클 = " + " · ".join(CYCLE_STEPS) + " step",
         "구간별로 평가하고, 가상 센서는 공정 구간만 씀",
         f"1초 행 중 공정 {process:,} · ALD 사이클 {cycle:,}" if process else "-"],
        ["⑦ 정규화", "신호마다 (값 − 평균) ÷ 표준편차. 평균·표준편차는 학습 run에서만 계산", "검증·시험 정보가 학습에 새지 않도록",
         "아래 표"],
        ["⑧ window 끝 위치", "window와 예측 시점이 모두 한 구간 안에 있는 위치만 목록으로", "run 경계·끊김을 넘는 window 방지",
         "아래 그림"],
    ]
    return figure + table(["단계", "한 일", "이유", "결과"], rows, cls="prep")


def _example_section(data: ReportData) -> str:
    dv = data.data_view
    ex = dv.example if dv is not None else None
    out = []
    if ex:
        unit = next((str(u) for u in (dv.columns.loc[dv.columns["column"] == ex["column"], "unit"] if dv.columns is not None
                                      else []) if u), "")
        run = _run_no(ex["run_id"])
        out.append(f'<h4>실제 run 예시 - 검증 run {e(run)} ({e(ex["column"])})</h4>'
                   f'<p class="note">학습과 같은 전처리 함수로 처리한 결과입니다. 0.5초 {ex["raw_rows"]:,}행 → 1초 '
                   f'{ex["rows_1s"]:,}행 · 구간 {ex["segments"]}개 · 빈 값 {ex["nan_filled"]}개. 1초 행 중 공정 구간 '
                   f'{ex["process_rows"]:,}행, ALD 사이클 {ex["cycle_rows"]:,}행.</p>')
        bands = [(a, b, "band-process") for a, b in ex["process"]] + [(a, b, "band-cycle") for a, b in ex["cycle"]]
        chart = series_chart([{"name": f"{ex['column']} (1초 평균)", "x": ex["t"], "y": ex["y"], "cls": "ln-pred", "kind": "line"}],
                             bands=bands, band_legend=[("band-process", "공정 구간 (START~END)"), ("band-cycle", "ALD 사이클")],
                             log=True, unit=unit, title=f"run {run} 구간 표시")
        out.append(f'<div class="chartbox"><div class="ctitle">① 구간 표시: 대기(STANDBY)는 흰 배경, 공정 구간은 연한 색, '
                   f"ALD 사이클은 진한 색. 압력이 대기압(약 760 Torr)과 공정 압력(1~4 Torr) 사이를 오가므로 세로축은 로그 눈금입니다."
                   f"</div>{chart}</div>")
        if ex.get("zoom_raw_t"):
            marks = [(t, "DCS 방출 (SI-FL1)") for t, name in ex["zoom_steps"] if name == "SI-FL1"]
            chart = series_chart([
                {"name": "원본 0.5초", "x": ex["zoom_raw_t"], "y": ex["zoom_raw_y"], "cls": "dot-raw", "kind": "dots"},
                {"name": "1초 평균 (학습에 쓴 값)", "x": ex["zoom_t"], "y": ex["zoom_y"], "cls": "ln-pred", "kind": "line"},
            ], marks=marks[:4], xlabel="시간 (초)", unit=unit, title=f"run {run} 1초 평균 확대")
            raw_max, avg_max = max(ex["zoom_raw_y"]), max(ex["zoom_y"])
            out.append(f'<div class="chartbox"><div class="ctitle">② 1초 평균 (ALD 사이클 2분 확대): 점은 0.5초 원본, 선은 2개씩 '
                       f"평균한 1초 값입니다. 최댓값이 원본 {fmt(raw_max)} → 평균 {fmt(avg_max)} {e(unit)}로, 짧은 튐은 "
                       f"줄고 형태는 유지됩니다.</div>{chart}</div>")
    exp = next((x for x in data.experiments if x.task == "forecast"), None)
    if exp is not None:
        out.append('<div class="chartbox"><div class="ctitle">③ window 규칙: 이 규칙을 만족하는 위치만 window 끝 위치 목록에 '
                   "넣습니다. 따라서 학습 중 무작위로 뽑아도 run 경계나 끊긴 곳을 넘는 window는 생기지 않습니다.</div>"
                   + window_rule(_context(exp), max(_horizons(exp) or [0]), label="window 규칙") + "</div>")
    return "".join(out)


def _scaling_table(data: ReportData) -> str:
    dv = data.data_view
    exp = next((x for x in data.experiments if x.task == "forecast" and x.picks), None)
    scaler = (exp.picks[0].doc.get("scaler") or {}) if exp is not None else {}
    if not scaler.get("columns"):
        return ""
    info = dv.columns.set_index("column") if dv is not None and dv.columns is not None else pd.DataFrame()
    index = {c: i for i, c in enumerate(scaler["columns"])}
    rows = []
    for col in [c for c in NORMALIZE_EXAMPLES if c in index]:
        i = index[col]
        unit = str(info.loc[col, "unit"]) if col in info.index and info.loc[col, "unit"] else ""
        name = str(info.loc[col, "name_ko"]) if col in info.index and info.loc[col, "name_ko"] else ""
        mean, std = float(scaler["mean"][i]), float(scaler["std"][i])
        rows.append([f"<code>{e(col)}</code>", e(name), e(unit), fmt(mean), fmt(std)])
    if not rows:
        return ""
    train_rows = (dv.bundles.get("train") or {}).get("rows") if dv is not None else None
    vg = index.get("VG11")
    example = ""
    if vg is not None:
        m, s_ = float(scaler["mean"][vg]), float(scaler["std"][vg])
        example = (f" 예) VG11 = 2.0 Torr → (2.0 − {fmt(m)}) ÷ {fmt(s_)} = {fmt((2.0 - m) / s_)}. 대기압 구간 때문에 평균·표준편차가 "
                   "커서 공정 압력(1~4 Torr) 차이는 작은 수로 바뀝니다. 그래서 고전 모델은 현재값 대비 변화를, 가상 센서는 레시피 평균 대비 "
                   "보정량을 따로 스케일해 학습합니다.")
    return ('<h4>정규화 - 학습 run 기준 평균·표준편차 (예시 신호)</h4>'
            f'<p class="note">{e(exp.name)} 실험의 값입니다. 학습 run {len((data.split or {}).get("train") or [])}개'
            + (f"(1초 행 {int(train_rows):,}개)" if train_rows else "")
            + f"로만 계산했고, 검증·시험 run과 새 CSV에도 이 값을 그대로 씁니다.{e(example)}</p>"
            + table(["컬럼", "이름", "단위", "평균 (학습 run)", "표준편차 (학습 run)"], rows))


def _experiment_inputs(data: ReportData) -> str:
    out = []
    for exp in data.experiments:
        out.append(f"<h4>{e(_exp_title(exp))}</h4>")
        data_cfg = exp.config.get("data") or {}
        rows = [
            ("샘플 간격 · 영역", f"{data_cfg.get('sample_seconds', 1.0)}초 · "
                              + {"all": "run 전체", "process": "공정 구간(START~END)", "cycle": "ALD 사이클만"}.get(
                                  str(data_cfg.get("region", "all")), e(data_cfg.get("region")))),
        ]
        if exp.task != "thickness":
            rows.append(("window", f"최근 {_context(exp):g}초"))
        sets = data_cfg.get("input_sets") or {"default": data_cfg.get("inputs", "all")}
        rows.append(("입력 세트", e(" / ".join(f"{k}: {v if isinstance(v, str) else f'{len(v)}개 지정'}"
                                           for k, v in sets.items()))))
        out.append(kv(rows))
        targets = (exp.prep or {}).get("targets") or {}
        if targets:
            trows = []
            for key, info in targets.items():
                removed = info.get("removed") or {}
                text = "<br>".join(f"{e(reason)}: {len(cols)}개 <span class='sub'>({e(', '.join(cols[:6]))}"
                                   f"{' …' if len(cols) > 6 else ''})</span>" for reason, cols in removed.items())
                trows.append([e(key), str(info.get("inputs")), text or "-"])
            out.append(table(["대상 · 입력 세트", "입력 신호 수", "뺀 신호"], trows))
        elif exp.task == "thickness" and exp.picks:
            doc = exp.picks[0].doc
            out.append(f"<p>입력 신호 {len(doc.get('inputs', []))}개 → run마다 요약한 특징 {len(doc.get('features', []))}개"
                       " (신호별 ALD 사이클 평균·표준편차, 공정 평균)</p>")
    return "".join(out)


def _inputs(data: ReportData) -> str:
    out = ['<p class="note">원본 데이터에서 제외한 항목과 모델 입력까지의 전처리 과정입니다. '
           "① 단계별 데이터 수 → ② 원본 데이터 → ③ 제외한 데이터 → ④ run 단위 분할 → ⑤ 전처리 → ⑥ 실험별 입력 순서입니다.</p>"]
    flow = _count_flow(data)
    if flow:
        out.append("<h3>① 단계별 데이터 수</h3>")
        out.append('<p class="note">화살표 위는 각 단계에서 제외한 항목이고, 오른쪽 끝 파란 테두리가 최종 사용 데이터입니다.</p>'
                   + flow)
    out.append("<h3>② 원본 데이터</h3>")
    ov = data.overview
    if ov:
        status = ov.get("column_status", {})
        out.append(kv([
            ("run (배치 1회 = 파일 1개)", f"{ov.get('n_runs')}개 · {e(str(ov.get('first_start', ''))[:10])} ~ "
                                         f"{e(str(ov.get('last_end', ''))[:10])} · 총 {float(ov.get('total_hours', 0)):.0f}시간"),
            ("행 수 · 샘플 간격", f"{int(ov.get('n_rows', 0)):,}행 · {ov.get('sampling_interval_s')}초"),
            ("신호 컬럼", f"{ov.get('n_signal_columns')}개 → 사용 {status.get('OK', 0)}개 "
                         f"(상수 {status.get('CONSTANT', 0)} · 중복 {status.get('DUPLICATE', 0)} 제외)"),
            ("레시피 · step", f"레시피 {ov.get('n_recipes')}종 · step {ov.get('n_unique_steps')}종"),
            ("빠진 run 번호", e(", ".join(str(n) for n in ov.get("missing_run_numbers", [])) or "-")),
            ("Ex-situ 두께", f"측정 블록 {ov.get('exsitu_blocks')}개 · run과 연결 {ov.get('exsitu_linked_runs')}개 "
                            f"(49점 {ov.get('exsitu_layouts', {}).get('49pt', 0)} · 9점 {ov.get('exsitu_layouts', {}).get('9pt', 0)})"),
            ("품질 이슈", e(", ".join(f"{SEVERITY_TEXT.get(k, (k,))[0]} {v}" for k, v in (ov.get("issues_by_severity") or {}).items()))),
        ]))
    else:
        out.append('<p class="empty">데이터 개요가 없습니다 (ald-build-overview를 먼저 실행).</p>')
    runs_table, columns_html, issues_html = _excluded_runs(data), _excluded_columns(data), _issue_table(data)
    if runs_table or columns_html:
        out.append("<h3>③ 제외한 데이터</h3>")
        if runs_table:
            out.append("<h4>제외하거나 따로 처리한 run</h4>" + runs_table)
        if columns_html:
            out.append("<h4>제외한 컬럼</h4>" + columns_html)
        out.append(issues_html)
    if data.split:
        sp = data.split
        out.append("<h3>④ run 단위 분할 (split_v2)</h3>")
        out.append('<p class="note">칸 하나가 run 하나입니다. 시간 순으로 앞쪽 run은 학습, 뒤쪽 run은 검증·시험에 씁니다. '
                   "한 run이 학습과 평가에 함께 쓰이지 않으며, 시험 run은 최종 후보를 정한 뒤 한 번만 평가합니다. "
                   "칸에 마우스를 올리면 run 번호와 용도가 표시됩니다.</p>")
        out.append(_split_strip(data))
        out.append(kv([
            ("학습 / 검증 / 시험", f"{len(sp.get('train', []))} / {len(sp.get('validation', []))} / {len(sp.get('test', []))} run"),
            ("제외", e(", ".join(f"{x['run_id']} ({REASONS.get(x.get('reason'), x.get('reason'))})"
                                for x in sp.get("excluded") or []) or "-")),
        ]))
    out.append("<h3>⑤ 전처리</h3>")
    out.append('<p class="note">모든 run에 같은 함수를 적용합니다. 윗줄은 run별 처리, 아랫줄은 모델 입력 직전 처리입니다. '
               "새 CSV를 예측할 때도 같은 함수를 씁니다.</p>")
    out.append(_pipeline(data))
    out.append(_example_section(data))
    out.append(_scaling_table(data))
    out.append("<h3>⑥ 실험별 입력</h3>")
    out.append(_experiment_inputs(data))
    return "".join(out)


def _outputs(data: ReportData) -> str:
    cards = []
    for exp in data.experiments:
        card = task_card(exp.task, horizons=_horizons(exp), context=_context(exp))
        units = {p.target: p.unit for p in exp.picks}
        targets = exp.config.get("targets") or []
        chips = "".join(f'<span class="chip">{e(target_name(t, units.get(t, "")))}</span> ' for t in targets)
        region = "" if exp.task == "thickness" else "ALD 사이클 / 그 외 / step 전환 직후 10초를 따로"
        rows = [("대상", chips), ("기준선", e(card["baseline"])),
                ("평가", e(_eval_word(exp)) + " · MAE(물리 단위) · 기준선 대비 개선율"
                 + (f'<div class="sub">{e(region)}</div>' if region else ""))]
        cards.append(f'<div class="ocard"><div class="oh"><span class="chip">{e(card["name"])}</span> '
                     f"<b>{e(_short(exp))}</b> <code>{e(exp.name)}</code></div>"
                     f'<div class="oo">{e(card["out"])}</div>'
                     + archs.task_figure(exp.task, context=_context(exp), horizons=_horizons(exp))
                     + '<div class="orows">' + "".join(f'<div><span class="ml">{k}</span>{v}</div>' for k, v in rows)
                     + "</div>"
                     + (f'<p class="sub">{e(exp.config["description"])}</p>' if exp.config.get("description") else "")
                     + "</div>")
    if not cards:
        return '<p class="empty">아직 끝난 실험이 없습니다.</p>'
    return ('<p class="note">실험별 모델 출력입니다. 그림의 점선 상자는 입력 구간, 점은 출력입니다.</p>'
            '<div class="ocards">' + "".join(cards) + "</div>")


MODEL_GROUPS = (  # (category, one line, models in display order, colour)
    ("기준선", "학습 없음 · 성능 비교의 기준", ("persistence", "template", "mean"), "var(--muted)"),
    ("고전 ML", "window를 요약 특징으로 바꿔 학습 · CPU 수 분", ("ridge", "pls", "lightgbm"), "var(--s2)"),
    ("딥러닝", "시계열을 직접 입력하는 신경망 · GPU", ("dlinear", "gru", "patchtst", "itransformer"), "var(--s3)"),
)
DEEP_KIND = {"dlinear": "선형", "gru": "순환", "patchtst": "Transformer", "itransformer": "Transformer"}
DEEP_TRAITS = {  # what a token is, how signals meet, how time is read
    "dlinear": ("없음 (신호마다 선형층)", "출력 head에서만", "추세·나머지에 선형 가중치"),
    "gru": ("1초 시점 (모든 신호)", "시점마다 처음부터 섞음", "순서대로 상태 갱신"),
    "patchtst": ("신호별 시간 조각 (patch)", "독립: head에서만 · 혼합: 처음부터", "patch끼리 attention"),
    "itransformer": ("신호 하나의 window 전체", "신호끼리 attention", "선형층 하나로 요약"),
}
BASELINE_ROWS = {  # how it predicts, the task, what beating it means
    "persistence": ("직전 값이 유지된다고 가정", "예측", "값이 변하는 시점을 예측함"),
    "template": ("같은 레시피 step·위치의 학습 run 평균", "가상 센서", "레시피 외 장비 상태를 반영함"),
    "mean": ("학습 run 값의 평균", "두께", "run 간 차이를 설명함"),
}
TASK_LEARNING = (  # task, output, baseline, classic, deep, evaluation
    ("forecast", "10·20·30초 뒤 값", "현재값 유지", "현재값 + <b>변화량</b>", "<b>값 직접</b> (RevIN으로 수준 되돌림)",
     "검증 run MAE"),
    ("virtual_sensor", "대상 센서의 지금 값", "레시피 평균", "레시피 평균 + <b>보정량</b>",
     "레시피 평균 + <b>보정량</b> (편차 단위 · MAE 손실)", "검증 run MAE"),
    ("thickness", "run 하나의 두께·U/F·WtoW", "학습 평균", "run 요약값 → <b>값 직접</b>", "-", "run 하나씩 빼기 (LOO)"),
)
DEEP_FIGURES = {"patchtst": "patchtst_figure", "itransformer": "itransformer_figure", "dlinear": "dlinear_figure",
                "gru": "gru_figure"}


def _model_use(data: ReportData) -> dict[str, dict[str, Any]]:
    """Per model: the experiments it was trained in, the targets it won, and its deep parameter counts."""

    use: dict[str, dict[str, Any]] = {}
    for exp in data.experiments:
        for model in exp.table["model"].unique():
            use.setdefault(str(model), {"exps": [], "best": {}, "params": []})["exps"].append(exp)
        for p in exp.picks:
            use.setdefault(p.model, {"exps": [exp], "best": {}, "params": []})["best"].setdefault(exp.name, []).append(p.target)
        if "n_params" in exp.results:
            for model, n in zip(exp.results["model"], exp.results["n_params"]):
                if pd.notna(n) and n:
                    use.setdefault(str(model), {"exps": [], "best": {}, "params": []})["params"].append(float(n))
    return use


def _count(n: float) -> str:
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.0f} k" if n >= 1e3 else f"{n:.0f}"


def _param_range(values: list[float]) -> str:
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    return _count(lo) if _count(lo) == _count(hi) else f"{_count(lo)} ~ {_count(hi)}"


def _model_map(use: dict[str, dict[str, Any]]) -> str:
    cols = []
    for title, line, names, color in MODEL_GROUPS:
        chips = []
        for name in names:
            card, u = model_card(name), use.get(name)
            wins = sum(len(v) for v in u["best"].values()) if u else 0
            kind = f'<span class="mk">{e(DEEP_KIND[name])}</span>' if name in DEEP_KIND else ""
            state = (f'<span class="mu">실험 {len(u["exps"])}개</span>' if u else '<span class="mu off">이번 미사용</span>')
            star = f'<span class="mw">최고 {wins}</span>' if wins else ""
            chips.append(f'<a class="mchip{"" if u else " off"}" href="#m-{name}"><b>{e(card["name"])}</b>{kind}'
                         f'<span class="mrow">{state}{star}</span></a>')
        cols.append(f'<div class="mcol" style="--c:{color}"><div class="mh">{e(title)}</div>'
                    f'<div class="ml2">{e(line)}</div>{"".join(chips)}</div>')
    return '<div class="mmap">' + '<div class="marrow">→</div>'.join(cols) + "</div>"


def _task_learning(data: ReportData) -> str:
    tasks = {exp.task for exp in data.experiments}
    rows = [[e(task_card(task)["name"]) + ("" if task in tasks else ' <span class="sub">(이번 결과 없음)</span>'),
             e(out), e(base), classic, deep, e(how)] for task, out, base, classic, deep, how in TASK_LEARNING]
    return table(["과제", "출력", "기준선", "고전 ML이 배우는 것", "딥러닝이 배우는 것", "평가"], rows, cls="tasks")


def _selection_matrix(data: ReportData, use: dict[str, dict[str, Any]]) -> str:
    exps = data.experiments
    if not exps:
        return '<p class="empty">아직 끝난 실험이 없습니다.</p>'
    rows = []
    for title, _, names, color in MODEL_GROUPS:
        for k, name in enumerate([n for n in names if n in use]):
            u = use[name]
            cells = []
            for exp in exps:
                won = u["best"].get(exp.name, [])
                n_targets = len(exp.picks) or len(exp.config.get("targets") or [])
                if won:
                    names_ = ", ".join(won[:4]) + (f" 외 {len(won) - 4}" if len(won) > 4 else "")
                    cells.append(f'<span class="win">★ {len(won)} / {n_targets}</span><div class="sub">{e(names_)}</div>')
                elif exp.name in {x.name for x in u["exps"]}:
                    cells.append('<span class="sub">학습함 (최고 아님)</span>')
                else:
                    cells.append('<span class="dot">·</span>')
            group = f'<i class="cdot" style="background:{color}"></i>{e(title)}' if k == 0 else ""
            rows.append([group, e(model_card(name)["name"]), *cells])
    return table(["묶음", "모델", *[e(_short(x)) for x in exps]], rows, cls="matrix sel")


def _classic_table(use: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name in ("ridge", "pls", "lightgbm"):
        card = model_card(name)
        task = next((x.task for x in use[name]["exps"]), "forecast") if name in use else "forecast"
        rows.append([f'<b id="m-{name}">{e(card["name"])}</b>' + ("" if name in use else '<div class="sub">이번 미사용</div>'),
                     e(card["what"]), e(card["good"]), e(card["bad"]),
                     e(_defaults_text(name, task)) if name != "ridge" else "-", e(_auto_text(name, task, "-"))])
    return table(["모델", "방식", "강점", "약점", "기본 설정", "자동 결정 항목"], rows, cls="mtab")


def _deep_table(use: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name in ("dlinear", "gru", "patchtst", "itransformer"):
        card, (token, mix, time_) = model_card(name), DEEP_TRAITS[name]
        u = use.get(name)
        rows.append([f'<a href="#m-{name}"><b>{e(card["name"])}</b></a>', e(DEEP_KIND[name]), e(token), e(mix), e(time_),
                     _param_range(u["params"]) if u else '<span class="sub">이번 미사용</span>',
                     e(card["good"]), e(card["bad"])])
    return table(["모델", "계열", "토큰 (attention 단위)", "신호 간 관계", "시간 처리 방식", "파라미터 수 (이번)", "강점", "약점"],
                 rows, cls="mtab")


def _models(data: ReportData) -> str:
    use = _model_use(data)
    out = [('<p class="note">모델은 세 분류입니다. 기준선은 비교 기준, 고전 ML은 요약 특징을 쓰는 모델, 딥러닝은 '
            "시계열을 직접 입력하는 신경망입니다. 이름을 누르면 해당 구조 그림으로 이동합니다.</p>"),
           "<h3>모델 구성</h3>", _model_map(use),
           "<h3>과제별 학습 대상</h3>", _task_learning(data),
           "<h3>모델별 선정 현황</h3>",
           ('<p class="note">★ = 검증 MAE가 가장 낮아 대표 모델로 선정된 대상 수 / 전체 대상 수. '
            "대상별 수치는 성능표 탭에 있습니다.</p>"), _selection_matrix(data, use)]
    # ① baselines
    out.append('<div class="catsec" style="--c:var(--muted)"><div class="cath"><i></i><b>① 기준선</b> '
               '<span class="sub">학습 없음</span></div>')
    out.append(table(["모델", "예측 방법", "과제", "이 기준보다 나으면"],
                     [[f'<b id="m-{n}">{e(model_card(n)["name"])}</b>', e(how), e(task), e(meaning)]
                      for n, (how, task, meaning) in BASELINE_ROWS.items()], cls="mtab"))
    out.append("</div>")
    # ② classic
    out.append('<div class="catsec" style="--c:var(--s2)"><div class="cath"><i></i><b>② 고전 ML</b> '
               '<span class="sub">예측·가상 센서: window 요약 → Ridge·LightGBM · 두께: run 요약 → Ridge·PLS·LightGBM</span></div>')
    out.append("<h4>예측·가상 센서</h4>" + archs.classic_figure())
    out.append("<h4>두께 (run 단위)</h4>" + archs.vm_figure())
    out.append(_classic_table(use) + "</div>")
    # ③ deep
    out.append('<div class="catsec" style="--c:var(--s3)"><div class="cath"><i></i><b>③ 딥러닝</b> '
               '<span class="sub">선형 DLinear · 순환 GRU · Transformer PatchTST·iTransformer</span></div>')
    out.append(_deep_table(use))
    out.append("<h4>딥러닝 모델 공통 구조</h4>"
               '<p class="note">backbone(아래 4종)만 바꾸고, 정규화와 출력 head는 같은 코드(SeriesModel)를 씁니다.</p>'
               + archs.wrapper_figure())
    for name in ("patchtst", "itransformer", "dlinear", "gru"):
        card = model_card(name)
        tag = "" if name in use else ' <span class="chip">이번 보고서에 결과 없음</span>'
        out.append(f'<h4 id="m-{name}">{e(card["name"])} <span class="chip">{e(DEEP_KIND[name])}</span>{tag}</h4>'
                   f'<p class="note">{e(card["what"])}</p>' + getattr(archs, DEEP_FIGURES[name])())
        if name == "patchtst":
            out.append(archs.channel_modes_figure())
    out.append('<p class="sub">그림의 숫자는 기본값입니다 (models/defaults.py). 실험마다 바꾼 값은 Hyper Parameter 탭에 있습니다.</p>')
    out.append("</div>")
    return "".join(out)


DEMO_FIRST = "patchtst"  # the model the 추론 시연 opens on when it was trained for the target


def _model_jobs(exp: Experiment, target: str) -> list[dict[str, Any]]:
    """Per trained model (baselines aside) its best configuration on validation and that configuration's best
    seed: the job to replay. Ordered by validation MAE (1st first)."""

    res = exp.results
    ok = res[(res["status"] == "ok") & (res["target"] == target)] if "status" in res else res[res["target"] == target]
    ok = ok[~ok["model"].isin(BASELINE_MODELS) & ok["mae"].notna()]
    out = []
    for model, group in ok.groupby("model"):
        means = group.groupby("label")["mae"].mean()
        label = str(means.idxmin())
        job = group[group["label"] == label].sort_values("mae").iloc[0]
        out.append({"model": str(model), "label": label, "mae": float(means.min()), "job_dir": exp.directory / str(job["job_dir"])})
    return sorted(out, key=lambda r: r["mae"])


def _demo_items(data: ReportData) -> list[dict[str, Any]]:
    """One player entry per target: the best job of every trained model on the same validation run."""

    from .collect import _example

    items = []
    for exp in data.experiments:
        if exp.task == "thickness":
            continue
        for pick in exp.picks:
            samples = []
            for rank, job in enumerate(_model_jobs(exp, pick.target), start=1):
                frame, zoom = _example(job["job_dir"])
                if frame is None:
                    continue
                name = model_card(job["model"])["name"]
                shown = name if job["label"] == job["model"] else f"{name} ({nice_label(job['label'])})"
                samples.append({"name": job["model"], "label": f"{rank}위 {name} · {fmt(job['mae'], 3)}",
                                "note": f"{shown} · 검증 MAE {fmt(job['mae'])} {pick.unit}".strip(),
                                "frame": frame, "zoom": zoom})
            item = player.sample_series(f"{exp.name}/{pick.target}", group=_short(exp),
                                        title=target_name(pick.target, pick.unit), target=pick.target, unit=pick.unit,
                                        context=_context(exp), samples=samples)
            if item:
                first = [k for k, m in enumerate(item["models"]) if m["name"] == DEMO_FIRST]
                item["default"] = first[0] if first else 0
                items.append(item)
    return items


def _demo(data: ReportData, items: list[dict[str, Any]], examples: bool = True) -> str:
    """추론 시연: replay the selected models' predictions on a validation run (``items`` from :func:`_demo_items`)."""

    steps = ('<div class="steps2">'
             '<div class="step2"><span class="sk">①</span> <span class="st2">입력</span>'
             '<div class="sr">점선 상자 = 모델이 지금 보는 최근 구간</div></div>'
             '<div class="step2"><span class="sk">②</span> <span class="st2">예측</span>'
             '<div class="sr">점 = 지금 낸 예측 (+10·20·30초 뒤, 가상 센서는 지금 값)</div></div>'
             '<div class="step2"><span class="sk">③</span> <span class="st2">확인</span>'
             '<div class="sr">시간이 지나 실측이 오면 오차와 누적 MAE가 바뀜</div></div></div>')
    out = [('<p class="note">학습한 모델이 <b>검증 run(학습에 쓰지 않은 run)</b>에서 낸 예측을 시간순으로 재생합니다. '
            "대상별로 모델마다 가장 좋은 설정을 넣었고(버튼: 순위 · 모델 · 검증 MAE), PatchTST가 있으면 PatchTST부터 "
            "표시합니다. 저장된 예측을 재생하는 것이며 모델을 다시 실행하지는 않습니다.</p>"), steps]
    if items:
        out.append(player.player(items))
    elif not examples:
        out.append('<p class="empty">이 보고서는 예측 예시 없이(--no-examples) 만들어 재생할 데이터가 없습니다. '
                   "<code>bash scripts/make_report.sh</code>로 다시 만들면 나타납니다.</p>")
    else:
        out.append('<p class="empty">재생할 예측이 없습니다 (끝난 예측·가상 센서 실험의 predictions_sample.parquet 없음).</p>')
    if any(x.task == "thickness" for x in data.experiments):
        out.append('<p class="sub">두께 예측은 run당 값이 하나라 재생하지 않습니다. 성능표 탭의 LOO 결과를 참고하세요.</p>')
    out.append("<h3>새 trace 추론 방법</h3>")
    out.append(table(["순서", "명령 (서버, ald-lab 폴더)", "결과"], [
        ["1", "<code>source .venv/bin/activate &amp;&amp; source scripts/env.sh</code>", "환경 준비"],
        ["2", "<code>ald-predict work/packages/&lt;실험&gt;_&lt;시각&gt;/&lt;대상&gt;__&lt;모델&gt; --csv &lt;CSV 파일 또는 폴더&gt; --html</code>",
         "predictions.csv + <b>predictions_player.html</b> (이 화면과 같은 재생기, run마다 하나)"],
        ["3", "predictions_player.html을 노트북으로 복사해 브라우저로 열기", "CSV에 대상 신호가 있으면 오차도 표시"],
    ]))
    return "".join(out)


FLOW = [
    ("data", "① 원본 데이터", [
        ("원본 데이터", "서버 폴더 (scripts/env.sh)", "data/raw.py · data/exsitu.py",
         "trace CSV 74개 (0.5초, 207컬럼) + Ex-situ 두께 엑셀"),
        ("데이터 개요", "ald-build-overview", "data/overview.py · catalog.py · profile.py",
         "work/data_overview/: run·컬럼·step·품질 이슈·두께 표"),
    ]),
    ("prep", "② 전처리", [
        ("정리된 데이터", "ald-train 시작 시 한 번 (자동)", "training/prepare.py",
         "work/prepared/: 1초 평균, 끊김에서 구간 분할, run 단위 학습/검증/시험 (시험 봉인)"),
        ("입력·정규화·window", "자동", "training/config.py (resolve_inputs) · windows.py",
         "대상 계열·복사본 제외, 학습 run 정규화, GPU에서 index로 window 자르기"),
    ]),
    ("train", "③ 하이퍼파라미터·학습", [
        ("작업 목록", "configs/experiments/*.yaml", "training/config.py (expand_jobs)",
         "대상 × 모델 × 설정 조합 × 입력 세트 × seed"),
        ("학습", "ald-train <yaml> --gpus 0,1,2,3", "training/runner.py · trainer.py · models/deep.py · classic.py · vm.py",
         "기준선 먼저 → GPU 4장 병렬, 조기 종료 → jobs/<id>/ (가중치, job.json, history.json)"),
    ]),
    ("eval", "④ 평가·소요 시간·강점·약점", [
        ("검증 비교", "자동 (학습 끝)", "training/evaluate.py · runner.summarize",
         "summary.md: MAE, 구간별 오차, 기준선 대비 개선율, 강점·약점 → 진행 기록(journal)"),
        ("시험 평가", "ald-final-test <실험 폴더>", "runner.run_final_test",
         "test_summary.md: 검증으로 고른 후보만 봉인된 시험 run에서 한 번"),
    ]),
    ("use", "⑤ 활용", [
        ("모델 패키지", "ald-export-model <실험 폴더>", "training/export.py",
         "work/packages/: 가중치 + job.json + README + predict.sh / predict.bat (+ zip)"),
        ("새 데이터 예측", "bash predict.sh <CSV> · ald-predict", "training/predict.py",
         "predictions.csv: 시점별 예측값 (실측이 있으면 오차)"),
        ("보고", "ald-report · scripts/lab.sh", "report/ · server.py",
         "work/reports/*.html (이 파일) · 실험실 대시보드 (본인용: 학습 실행 · 보기 전용: 공유 주소)"),
    ]),
]


COMMANDS = {  # in the order they are run
    "ald-build-overview": "원본 데이터 개요: run·컬럼·step·두께 표",
    "ald-analyze-params": "가상 센서 대상 분석",
    "ald-train": "학습: 실험 YAML → 작업 목록 → 모델·검증",
    "ald-final-test": "시험 run 평가: 검증으로 고른 후보만",
    "ald-export-model": "모델 패키지: 가중치 + 실행 파일",
    "ald-predict": "새 CSV 예측",
    "ald-report": "보고서 (이 파일)",
    "ald-queue": "실험 대기열: 대시보드에서 고른 설정을 차례로 학습",
    "대시보드": "화면 구성: streamlit run src/ald_lab/dashboard/app.py",
}
DASHBOARD_ROOT = "dashboard.app:" + MODULE_CODE
# Fixed categorical order (--s1 …); 공통 is grey. Each colour is named in the legend.
GROUPS = {"data": "data/ (원본 데이터)", "training": "training/ (학습·예측)", "models": "models/ (모델)",
          "analysis": "analysis/ (대상 분석)", "report": "report/ (보고서)", "dashboard": "dashboard/ (대시보드)",
          "cli": "cli.py (명령)", "common": "paths · journal (공통)"}
MODULE_ROLES = {
    "cli": "명령 진입점 (ald-build-overview · ald-train · ald-final-test · ald-report …)",
    "paths": "데이터·결과 폴더 경로 (환경변수)", "planfile": "실험 계획 파일(.md) 읽기: 차수별 양식으로 바꿔 대기열에", "journal": "진행 기록 (단계별 요약·지표·강점·약점)",
    "analysis.variability": "가상 센서 대상 분석: 레시피·다른 센서로 얼마나 설명되나",
    "data.raw": "trace CSV 읽기, 시각 파싱", "data.exsitu": "Ex-situ 엑셀 → 측정점·두께·U/F",
    "data.overview": "데이터 개요 표 만들기", "data.catalog": "step 구간·ALD 사이클 판별",
    "data.profile": "컬럼별 범위·상수·복사본 판정", "data.columns": "컬럼 사전 (의미·단위·계열)",
    "data.splits": "학습/검증/시험 run 분할",
    "training.config": "실험 YAML → 작업 목록, 입력 선택", "training.prepare": "1초 평균, 구간 분할, 분할별 묶음(bundle)",
    "training.windows": "정규화, GPU window, 복사본 탐지", "training.runner": "실험 실행·병렬 처리·요약·시험 평가",
    "training.trainer": "딥러닝 학습 루프 (조기 종료, bf16)", "training.evaluate": "오차 지표, 강점·약점 문장",
    "training.vm": "두께 예측 (run 단위)", "training.predict": "새 CSV 예측", "training.export": "모델 패키지",
    "training.selection": "대상별 대표 모델 고르기 (검증 기준)",
    "models.deep": "PatchTST · iTransformer · DLinear · GRU", "models.classic": "기준선 · Ridge · LightGBM",
    "models.layers": "딥러닝 공통 층", "models.defaults": "모델 기본 설정값", "models.cards": "모델·과제 설명 문구",
    "report.collect": "보고서에 넣을 결과 모으기", "report.render": "보고서 HTML 만들기",
    "report.charts": "보고서 차트 (SVG)", "report.analysis": "하이퍼파라미터 효과 분석",
    "report.codemap": "코드 호출 관계 읽기 (이 탭)", "report.diagram": "흐름도 그리기 (이 탭)",
    "report.dataview": "In 탭 데이터 (제외 항목·전처리 예시)",
    "report.design": "단계별 설계 설명: 입력·출력·결정·테스트·검토 포인트 (이 탭)",
    "report.plan": "다음 실험 후보 읽기 (configs/plan/)", "report.archs": "모델 구조 그림 (Function · Out 탭)",
    "report.player": "추론 재생기 (추론 시연 탭 · ald-predict --html)",
    "launch": "대기열·작업자: 대시보드에서 고른 설정으로 학습 (ald-queue)", "ledger": "누적 기록: 모든 실행의 설정과 결과",
    "stages": "PatchTST 우선: 탐색 순위 · 전체 적용 · 대안 판정", "server": "실험실 서버: 보고서 + 하이퍼파라미터 탐색 (ald-lab-server)",
    "report.lab": "하이퍼파라미터 탐색 탭", "report.overview": "요약 탭: 결론 문장 · 대상별 오차 · 기록 추이 그림", "models.constraints": "조합 검사 (head 수 · patch 길이)",
    "dashboard.views.launcher": "대시보드: 새 실험 실행 · 진행 상황", "dashboard.views.records": "대시보드: 누적 기록",
    "dashboard.app": "대시보드 시작·페이지 구성", "dashboard.state": "대시보드가 읽는 결과 파일",
    "dashboard.ui": "표·차트·KPI 공통 부품", "dashboard.theme": "차트 색·글꼴", "dashboard.tokens": "색 토큰 (보고서와 공유)",
    "dashboard.views.home": "대시보드: 홈", "dashboard.views.data_overview": "대시보드: 원본 데이터",
    "dashboard.views.preprocessing": "대시보드: 전처리", "dashboard.views.training": "대시보드: 학습",
    "dashboard.views.timeline": "대시보드: 타임라인",
}
GROUP_COLORS = {"data": "var(--s1)", "training": "var(--s2)", "models": "var(--s3)", "analysis": "var(--s4)",
                "report": "var(--s5)", "dashboard": "var(--s6)", "cli": "var(--s7)", "common": "var(--muted)"}
# where each folder sits in the overview: (column, row); 공통 (paths · journal) is the footer
PACKAGE_PLACES = {"cli": (0, 1.5), "report": (1, 0), "training": (1, 1), "analysis": (1, 2), "dashboard": (1, 3),
                  "models": (2, 0.5), "data": (2, 1.5)}
TREE_DEPTH, TREE_LIMIT, TREE_MIN_LINES = 3, 40, 10


def _group(module: str) -> str:
    head = module.split(".")[0]
    return head if head in GROUPS and head != "common" else "common"


def _legend(groups: list[str], marks: str) -> str:
    return ('<div class="legend">' + "".join(f'<span><i class="sw-{g}"></i>{e(GROUPS[g])}</span>' for g in groups)
            + marks + "</div>")


def _code_roots(cmap: CodeMap) -> dict[str, str]:
    roots = {cmd: cmap.entry_points[cmd] for cmd in COMMANDS if cmd in cmap.entry_points}
    roots.update({cmd: root for cmd, root in cmap.entry_points.items() if cmd not in roots})
    roots["대시보드"] = DASHBOARD_ROOT
    return {cmd: root for cmd, root in roots.items() if root in cmap.functions}


def _function_title(name: str) -> str:
    if name.endswith(".__init__"):
        return name[: -len(".__init__")] + "()"  # building an object
    return f"{name}()"


def _fn_code(cmap: CodeMap, key: str) -> str:
    return f"<code>{e(_function_title(cmap.functions[key].name))}</code>"


def _describer(cmap: CodeMap, roots: dict[str, str]) -> Any:
    entry = {root: cmd for cmd, root in roots.items()}

    def describe(key: str) -> Box:
        f = cmap.functions[key]
        path = cmap.files.get(f.module, f.module)
        tip = f"{path}:{f.lineno} · {f.lines}줄" + (f" — {f.doc}" if f.doc else "")
        if key in entry:
            name = "모듈 실행" if f.name == MODULE_CODE else _function_title(f.name)
            return Box(entry[key], f"{path} · {name}", _group(f.module), tip, root=True)
        title = path.rsplit("/", 1)[-1] if f.name == MODULE_CODE else _function_title(f.name)
        return Box(title, path, _group(f.module), tip)

    return describe


def _package_graph(cmap: CodeMap) -> str:
    counts: dict[tuple[str, str], int] = {}
    for f in cmap.functions.values():
        for callee in f.calls:
            a, b = _group(f.module), _group(callee.split(":")[0])
            if a != b:
                counts[a, b] = counts.get((a, b), 0) + 1
    files: dict[str, int] = {}
    lines: dict[str, int] = {}
    for module, n in cmap.module_lines.items():
        if n > 1:
            files[_group(module)] = files.get(_group(module), 0) + 1
            lines[_group(module)] = lines.get(_group(module), 0) + n

    def describe(g: str) -> Box:
        return Box(GROUPS[g], f"파일 {files.get(g, 0)}개 · {lines.get(g, 0):,}줄", g,
                   f"{GROUPS[g]} · 파일 {files.get(g, 0)}개, {lines.get(g, 0):,}줄")

    places = {g: xy for g, xy in PACKAGE_PLACES.items() if files.get(g)}
    to_common = sum(n for (a, b), n in counts.items() if b == "common")
    footer = Box(f"{GROUPS['common']} · 폴더 경로·진행 기록", f"거의 모든 폴더가 씀 (부르는 곳 {to_common}곳, 선은 생략)",
                 "common", "paths.py · journal.py")
    edges = {(a, b): n for (a, b), n in counts.items() if a in places and b in places}
    return diagram.grid_graph(places, edges, describe, titles=["명령", "기능", "모델 · 데이터"], footer=footer,
                              label="폴더끼리 부르는 관계")


def _shared_code(cmap: CodeMap, roots: dict[str, str]) -> str:
    reach = {cmd: cmap.reach(root) for cmd, root in roots.items()}
    hubs = cmap.hubs()
    count: dict[str, int] = {}
    for found in reach.values():
        for key in found:
            count[key] = count.get(key, 0) + 1
    shared = [k for k, n in count.items() if n >= 2 and k not in hubs and k not in roots.values()
              and cmap.functions[k].lines >= 15]
    shared.sort(key=lambda k: (-count[k], cmap.files.get(cmap.functions[k].module, ""), cmap.functions[k].lineno))
    if not shared:
        return '<p class="empty">여러 명령이 함께 쓰는 함수가 없습니다.</p>'
    rows = []
    for key in shared:
        f = cmap.functions[key]
        rows.append([f"<code>{e(_function_title(f.name))}</code>",
                     f"<code>{e(cmap.files.get(f.module, f.module))}:{f.lineno}</code>",
                     *('<span class="dot">●</span>' if key in reach[cmd] else "" for cmd in roots)])
    header = ["함수", "파일:줄", *(cmd.replace("ald-", "") for cmd in roots)]
    return table(header, rows, cls="matrix")


def _module_table(cmap: CodeMap) -> str:
    calls = cmap.module_calls()
    called_by: dict[str, set[str]] = {}
    for module, targets in calls.items():
        for target in targets:
            called_by.setdefault(target, set()).add(module)
    order = list(GROUPS)
    modules = [m for m in cmap.module_lines if cmap.module_lines[m] > 1 and m]
    modules.sort(key=lambda m: (order.index(_group(m)), m))

    def names(found: set[str]) -> str:
        return ", ".join(cmap.files.get(m, m).removesuffix(".py") for m in sorted(found)) or "-"

    rows = []
    for m in modules:
        n_funcs = sum(1 for f in cmap.functions.values() if f.module == m and f.name != MODULE_CODE)
        rows.append([f"<code>{e(cmap.files.get(m, m))}</code>", e(MODULE_ROLES.get(m) or cmap.module_docs.get(m, "")),
                     f"{cmap.module_lines[m]:,}", str(n_funcs), e(names(calls.get(m, set()))),
                     e(names(called_by.get(m, set())))])
    return table(["파일 (src/ald_lab/)", "역할", "줄 수", "함수 수", "이 파일이 부르는 파일", "이 파일을 부르는 파일"], rows)


def _hub_table(cmap: CodeMap) -> str:
    callers = cmap.callers()
    rows = []
    for key in sorted(cmap.hubs(), key=lambda k: (-len(callers[k]), k)):
        f = cmap.functions[key]
        rows.append([f"<code>{e(_function_title(f.name))}</code>",
                     f"<code>{e(cmap.files.get(f.module, f.module))}:{f.lineno}</code>",
                     str(len(callers[key])), e(f.doc)])
    return table(["함수", "파일:줄", "부르는 곳", "설명 (코드 주석)"], rows)


def _flow(data: ReportData | None = None) -> str:
    try:
        cmap = scan_code()
    except (OSError, SyntaxError, ValueError) as err:
        return f'<p class="empty">코드를 읽지 못해 Code Flow Chart를 그리지 못했습니다: {e(err)}</p>' + _flow_steps()
    roots = _code_roots(cmap)
    describe = _describer(cmap, roots)
    n_files = sum(1 for m, n in cmap.module_lines.items() if n > 1 and m)
    out = [
        '<p class="note">보고서를 만들 때 코드(<code>src/ald_lab/</code>)를 직접 읽어 구성한 탭입니다. 줄 번호·코드 '
        f"발췌·호출 그림은 이 결과를 만든 코드와 일치합니다. 파일 {n_files}개 · 함수 {len(cmap.functions)}개 · "
        f"{sum(cmap.module_lines.values()):,}줄.</p>",
        ('<p class="note"><b>검토 순서</b>: ① 작업 흐름으로 전체 구조를 보고, ② 단계별 상세 설계에서 입력·출력, '
         "처리 순서(코드 위치), 핵심 코드, 설계 결정과 대안, 자동 테스트, 검토 포인트를 확인합니다. ③ 산출물 표는 "
         "단계 간에 전달되는 파일, ④ 점검표는 데이터 누설을 막고 재현성을 지키는 장치입니다. ⑤ 이후의 함수 호출 그림은 "
         "코드를 직접 확인할 때 참고하는 자료입니다.</p>"),
        "<h3>① 작업 흐름 요약 (단계 · 명령 · 결과물)</h3>",
        _flow_steps(),
        "<h3>② 단계별 상세 설계 (검토용)</h3>",
        _design_review(cmap, data),
        "<h3>③ 산출물 (단계 간 전달 파일)</h3>",
        _artifact_table(),
        "<h3>④ 데이터 누설 · 재현성 점검표</h3>",
        _guard_table(cmap),
        "<h3>⑤ 명령별 Code Flow Chart (함수 호출)</h3>",
        '<p class="note"><b>보는 법</b>: 왼쪽에서 오른쪽으로 호출하는 함수 → 호출되는 함수 순입니다. 한 함수가 호출하는 함수는 코드의 '
        "호출 순서대로 위에서 아래로 배치했습니다. 검은 상자가 명령(시작점)입니다. ↪ 표시는 그림에 이미 있는 함수를 "
        "다시 호출하는 곳이며, 호출 대상은 그림 아래에 적었습니다. 상자에 마우스를 올리면 파일·줄 번호와 설명이 "
        "표시됩니다. 그림을 단순하게 하기 위해 여러 곳에서 쓰는 공통 함수(아래 표)와 "
        f"{TREE_MIN_LINES}줄이 안 되는 작은 함수는 뺐습니다.</p>",
        _legend(list(GROUPS), '<span><b class="ln"></b>부름</span><span><b>↪</b> 이미 그린 함수를 또 부름</span>'),
    ]
    for cmd, root in roots.items():
        tree = cmap.tree(root, depth=TREE_DEPTH, limit=TREE_LIMIT, min_lines=TREE_MIN_LINES)
        modules = {cmap.functions[k].module for k in tree.level}
        hidden = f" · 그림에서 뺀 함수 {tree.hidden}개" if tree.hidden else ""
        title = e(cmd) if cmd == "대시보드" else f"<code>{e(cmd)}</code>"
        again: dict[str, list[str]] = {}
        for caller, callee in tree.extra:
            again.setdefault(caller, []).append(callee)
        notes = (('<p class="sub">↪ 그림에 이미 있는 함수를 또 부르는 곳: '
                  + " · ".join(f"{_fn_code(cmap, a)} → {', '.join(_fn_code(cmap, b) for b in bs)}"
                               for a, bs in again.items()) + "</p>")
                 if again else "")
        out.append(f'<details{" open" if cmd == "ald-train" else ""}><summary>{title} · {e(COMMANDS.get(cmd, ""))}'
                   f'<span class="sub"> · 함수 {len(tree.level)}개 · 파일 {len(modules)}개{hidden}</span></summary>'
                   + diagram.call_tree(tree, describe, label=f"{cmd} 함수 호출 흐름") + notes + "</details>")
    out.append("<h3>⑥ 명령 간 공통 코드</h3>")
    out.append('<p class="note">여러 명령이 같은 함수를 공유합니다. 예를 들어 학습·시험 평가·새 CSV '
               "예측이 같은 전처리와 window 코드를 쓰므로, 새 데이터도 학습 때와 동일하게 처리됩니다. "
               "(● = 해당 명령에서 직접 또는 간접 호출, 15줄 이상 함수만 표시)</p>")
    out.append(_shared_code(cmap, roots))
    out.append("<h3>⑦ 전체 구조 (패키지 간 호출 관계)</h3>")
    out.append('<p class="note">왼쪽에서 오른쪽 방향으로 호출합니다. 선 위 숫자는 한 폴더의 함수가 다른 폴더의 함수를 호출하는 '
               "지점 수입니다. 점선은 반대 방향 호출입니다 (예: LightGBM·Ridge가 학습 쪽 "
               "window 코드를 사용).</p>")
    out.append(_package_graph(cmap))
    out.append(f"<details><summary>공통 함수 {len(cmap.hubs())}개 ({HUB_CALLERS}곳 이상에서 호출)"
               "</summary>" + _hub_table(cmap) + "</details>")
    out.append(f"<details><summary>파일별 역할과 호출 관계 ({n_files}개 파일)</summary>" + _module_table(cmap)
               + "</details>")
    return "".join(out)


CODE_LINES = 45  # longest code excerpt shown in the design notes


def _fn_ref(cmap: CodeMap, key: str) -> str:
    fn = cmap.functions.get(key)
    where = f' <span class="sub">{e(cmap.files.get(fn.module, fn.module))}:{fn.lineno}</span>' if fn else ""
    return f"<code>{e(design_notes.short_name(key))}</code>{where}"


def _code_excerpt(cmap: CodeMap, key: str, sources: dict[str, str]) -> str:
    fn = cmap.functions.get(key)
    path = cmap.files.get(fn.module) if fn else None
    if fn is None or path is None:
        return ""
    if path not in sources:
        sources[path] = (PACKAGE_DIR / path).read_text(encoding="utf-8")
    body, cut = design_notes.excerpt(sources[path], fn.lineno, fn.lines, limit=CODE_LINES)
    more = f'<p class="sub">… 이하 {cut}줄 생략 (전체: {e(path)}:{fn.lineno})</p>' if cut else ""
    return (f'<details class="code"><summary><code>{e(design_notes.short_name(key))}</code> '
            f'<span class="sub">{e(path)}:{fn.lineno} · {fn.lines}줄</span></summary>'
            f'<pre class="code">{e(body)}</pre>{more}</details>')


def _stage_facts(key: str, data: ReportData | None) -> str:
    """'이번 데이터에서' — the stage's numbers from this report's results, where they exist."""

    if data is None:
        return ""
    ov, dv = data.overview or {}, data.data_view
    exps = [x for x in data.experiments if not x.smoke]
    facts = ""
    if key == "overview" and ov:
        status = ov.get("column_status") or {}
        issues = ov.get("issues_by_severity") or {}
        facts = (f"run {ov.get('n_runs')}개 · {int(ov.get('n_rows') or 0):,}행 · 신호 {ov.get('n_signal_columns')}개 → "
                 f"OK {status.get('OK', 0)} · 상수 {status.get('CONSTANT', 0)} · 중복 {status.get('DUPLICATE', 0)} · 품질 점검 "
                 f"{sum(issues.values())}건 (치명 {issues.get('critical', 0)} · 경고 {issues.get('warning', 0)} · "
                 f"정보 {issues.get('info', 0)})")
    elif key == "prepare" and dv is not None and dv.bundles:
        parts = [(k, name) for k, name in (("train", "학습"), ("validation", "검증"), ("test", "시험")) if k in dv.bundles]
        rows = " · ".join(f"{name} {int(dv.bundles[k]['rows']):,}행" for k, name in parts)
        train = dv.bundles.get("train") or {}
        cols = int(train.get("columns") or 0)
        size = f" (학습 묶음 float32 약 {int(train['rows']) * cols * 4 / 1e6:,.0f} MB)" if cols and train.get("rows") else ""
        segs = sum(int(dv.bundles[k].get("segments") or 0) for k, _ in parts)
        nan = sum(int(dv.bundles[k].get("nan_filled") or 0) for k, _ in parts)
        facts = f"{rows} × 신호 {cols}개{size} · 끊김 없는 구간 {segs}개 · 채운 빈 값 {nan:,}개"
    elif key == "inputs":
        counts = {x.name: sorted({int(v["inputs"]) for v in ((x.prep or {}).get("targets") or {}).values()
                                  if v.get("inputs")}) for x in exps}
        counts = {k: v for k, v in counts.items() if v}
        facts = " · ".join(f"{k} {v[0]}" + (f"~{v[-1]}" if len(v) > 1 else "") + "개" for k, v in counts.items())
        facts = f"입력 신호 수: {facts}" if facts else ""
    elif key == "train" and exps:
        jobs = sum(int(x.summary.get("jobs") or 0) for x in exps)
        minutes_total = sum(float(x.summary.get("wall_minutes") or 0) for x in exps)
        failed = sum(int(x.summary.get("failed") or 0) for x in exps)
        facts = f"실험 {len(exps)}개 · 작업 {jobs}개 (실패 {failed}) · 실험 시간 합계 {minutes_total / 60:.1f}시간"
    elif key == "evaluate" and exps:
        opened = [x.name for x in exps if x.test is not None and len(x.test)]
        facts = ("시험 run을 연 실험: " + (", ".join(opened) if opened else "없음") + " · 나머지는 검증 결과만")
    elif key == "use" and exps:
        made = [x.name for x in exps if x.package]
        facts = f"모델 패키지 {len(made)}개 실험: " + (", ".join(made) if made else "없음")
    return f'<p class="facts"><b>이번 데이터</b>: {e(facts)}</p>' if facts else ""


def _design_review(cmap: CodeMap, data: ReportData | None) -> str:
    rows = [[f'<a href="#design-{s.key}">{e(s.title)}</a>', f"<code>{e(s.command)}</code>", e(s.purpose.split(". ")[0]),
             str(len(s.steps)), str(len(s.checks)), str(len(s.review))] for s in design_notes.STAGES]
    out = [('<p class="note">단계마다 <b>입력 → 출력</b>, <b>처리 순서</b>(각 줄 옆이 그 일을 하는 함수와 파일:줄), '
            "<b>핵심 코드</b>(지금 코드에서 그대로 발췌), <b>설계 결정</b>(고른 방법 · 고르지 않은 방법 · 이유), "
            "<b>자동 테스트</b>, <b>검토 포인트</b>(알고 있는 한계와 확인이 필요한 가정)를 적었습니다. 검토 의견은 검토 포인트와 "
            "설계 결정 표를 중심으로 주시면 됩니다.</p>"),
           table(["단계", "명령", "하는 일", "처리 단계", "테스트", "검토 포인트"], rows, cls="stages")]
    sources: dict[str, str] = {}
    for stage in design_notes.STAGES:
        steps = "".join(f"<li>{e(step.text)}" + ("<br>" + " · ".join(_fn_ref(cmap, f) for f in step.functions)
                                                  if step.functions else "") + "</li>" for step in stage.steps)
        io = ('<div class="sw"><div><b>입력</b>' + table(["파일·데이터", "내용"], [[f"<code>{e(a)}</code>", e(b)]
                                                                         for a, b in stage.inputs])
              + '</div><div><b>출력</b>' + table(["파일·데이터", "내용"], [[f"<code>{e(a)}</code>", e(b)]
                                                                     for a, b in stage.outputs]) + "</div></div>")
        decisions = table(["고른 방법", "고르지 않은 방법", "이유"],
                          [[e(d.choice), e(d.alternative), e(d.reason)] for d in stage.decisions], cls="design")
        checks = table(["테스트 (tests/)", "확인하는 것"],
                       [[f"<code>{e(c.test)}</code>", e(c.what)] for c in stage.checks], cls="design")
        code = "".join(_code_excerpt(cmap, key, sources) for key in stage.code)
        review = "".join(f"<li>{e(r)}</li>" for r in stage.review)
        out.append(
            f'<details class="stage" id="design-{stage.key}"><summary>{e(stage.title)} '
            f'<span class="sub"><code>{e(stage.command)}</code></span><div class="note">{e(stage.purpose)}</div></summary>'
            + _stage_facts(stage.key, data)
            + "<h4>입력 → 출력</h4>" + io
            + f'<h4>처리 순서</h4><ol class="steps">{steps}</ol>'
            + "<h4>핵심 코드 (지금 코드에서 발췌)</h4>" + code
            + "<h4>설계 결정</h4>" + decisions
            + "<h4>검증 (자동 테스트)</h4>" + checks
            + f'<h4>검토 포인트</h4><ul class="review">{review}</ul></details>')
    return "".join(out)


def _artifact_table() -> str:
    return ('<p class="note">회사 데이터(CSV·엑셀·가중치·시점별 예측값)는 모두 서버의 <code>work/</code> 아래에만 있고 '
            "코드 저장소에는 들어가지 않습니다.</p>"
            + table(["위치", "만드는 명령", "내용", "읽는 곳"],
                    [[f"<code>{e(a)}</code>", e(b), e(c), e(d)] for a, b, c, d in design_notes.ARTIFACTS]))


def _guard_table(cmap: CodeMap) -> str:
    rows = [[e(g.risk), e(g.guard), "<br>".join(_fn_ref(cmap, f) for f in g.functions),
             "<br>".join(f"<code>{e(t.split('::')[-1])}</code>" for t in g.tests)] for g in design_notes.GUARDS]
    return ('<p class="note">결과가 실제보다 좋게 나오는 대표적인 실수와 이를 막는 장치(코드 위치·자동 테스트)입니다.</p>'
            + table(["위험", "막는 방법", "코드", "테스트"], rows, cls="design"))


def _flow_steps() -> str:
    out = ['<p class="note">위에서 아래로 진행합니다. 각 상자: 단계 · 실행 명령 · 코드 파일 · 결과물.</p><div class="flow">']
    for lane_id, lane, steps in FLOW:
        out.append(f'<div class="lane {lane_id}"><div class="lane-h">{e(lane)}</div>')
        for i, (name, command, code, result) in enumerate(steps):
            out.append(f'<div class="fbox"><div class="fn">{e(name)}</div>'
                       f'<div class="fc"><code>{e(command)}</code></div>'
                       f'<div class="fcode">{e(code)}</div><div class="fr">{e(result)}</div></div>')
            if i < len(steps) - 1:
                out.append('<div class="arrow">▼</div>')
        out.append("</div>")
        if lane_id != FLOW[-1][0]:
            out.append('<div class="arrow lane-arrow">▼</div>')
    out.append("</div>")
    return "".join(out)


STATUS_TEXT = {"EARLY_STOPPED": "조기 종료", "MAX_EPOCHS": "최대 epoch 도달 (추가 학습 시 개선 가능)",
               "TIME_LIMIT": "시간 제한으로 멈춤", "NAN_LOSS": "손실 발산으로 멈춤"}
EXP_SHORT = {"g1_forecast_vg11": "VG11 예측", "g1_forecast_params": "파라미터 예측", "g1_virtual_sensor": "가상 센서",
             "g2_thickness": "두께", "g3_core_vs_all": "핵심 vs 전체"}
NEXT_SEARCH = {"lightgbm": "잎 수 · 학습률 · 잎 최소 샘플", "ridge": "window 요약 특징의 종류",
               "patchtst": "d_model · patch 길이 · 층 수", "itransformer": "d_model · 층 수 · head 수",
               "dlinear": "이동평균 길이 · 표현 크기", "gru": "은닉 크기 · 층 수", "pls": "성분 수 범위 넓히기"}


def _short(exp: Experiment) -> str:
    return EXP_SHORT.get(exp.name.removesuffix("_smoke"), exp.name)


def _model_items(exp: Experiment) -> list[dict[str, Any]]:
    baseline = str(exp.summary.get("baseline"))
    items = [{"name": baseline}]
    for item in exp.config.get("models") or []:
        item = {"name": item} if isinstance(item, str) else dict(item)
        if item["name"] != baseline:
            items.append(item)
    return items


def _params_text(params: dict[str, Any]) -> str:
    return " · ".join(f"{k} {v}" for k, v in params.items()) or "-"


def _defaults_text(name: str, task: str) -> str:
    if name == "lightgbm":
        d = model_defaults.VM_LIGHTGBM if task == "thickness" else model_defaults.LIGHTGBM
        keys = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree")
        return _params_text({k: d[k] for k in keys if k in d})
    if name == "pls":
        return _params_text(model_defaults.VM_PLS)
    if name in model_defaults.DEEP:
        return _params_text(model_defaults.DEEP[name])
    return "-"


def _auto_text(name: str, task: str, patience: Any) -> str:
    if name == "ridge":
        return ("규제 강도 alpha: run 교차검증으로 자동 (" + model_defaults.VM_RIDGE_ALPHAS + ")" if task == "thickness"
                else "규제 강도 alpha: 검증 run으로 " + " · ".join(f"{a:g}" for a in model_defaults.RIDGE_ALPHAS) + " 중 선택")
    if name == "lightgbm" and task != "thickness":
        return f"반복 수: 검증 손실로 조기 종료 (최대 {model_defaults.LIGHTGBM['n_estimators']:,}회)"
    if name in model_defaults.DEEP:
        return f"epoch 수: 조기 종료 (patience {patience}) · 학습률: 검증 손실이 멈추면 절반으로"
    return "-"


def _search_table(exp: Experiment) -> str:
    sets = list(((exp.config.get("data") or {}).get("input_sets") or {}).keys())
    patience = (exp.config.get("train") or {}).get("patience", 8)
    rows = []
    for item in _model_items(exp):
        name = item["name"]
        label = item.get("label") or name
        title = e(model_card(name)["name"]) + (f' <span class="sub">{e(label)}</span>' if label != name else "")
        if name == str(exp.summary.get("baseline")):
            rows.append([title, "-", "-", "-", '<span class="chip">기준선</span> 비교 기준 (설정 없음)'])
            continue
        params, grid = item.get("params") or {}, item.get("grid") or {}
        fixed = e(_params_text(params)) + ("<div class='sub'>나머지 기본값</div>" if params else "")
        if not params:
            fixed = "기본값<div class='sub'>" + e(_defaults_text(name, exp.task)) + "</div>"
        searched = []
        if grid:
            n = 1
            for values in grid.values():
                n *= len(values)
            searched.append(e(" × ".join(f"{k} {v}" for k, v in grid.items())) + f" ({n}개 조합)")
        if len(sets) > 1:
            searched.append(f"입력 세트 {len(sets)}개 ({e(' / '.join(sets))})")
        if grid:
            status = '<span class="st good">탐색함</span> (결과: 아래 "설정별 결과")'
        elif len(sets) > 1:
            status = ('<span class="st warn">입력만 비교</span> (모델 설정 고정, 다음 실험에서 탐색 예정)'
                      f'<div class="sub">다음: {e(NEXT_SEARCH.get(name, "주요 설정"))}</div>')
        else:
            status = ('<span class="st warn">탐색 안 함</span> (고정·기본값, 다음 실험에서 탐색 예정)'
                      f'<div class="sub">다음: {e(NEXT_SEARCH.get(name, "주요 설정"))}</div>')
        rows.append([title, fixed, "<br>".join(searched) or "-", e(_auto_text(name, exp.task, patience)), status])
    return table(["모델", "고정 설정", "탐색한 설정", "자동으로 정해지는 것", "상태"], rows)


def _not_searched(exps: list[Experiment]) -> str:
    seen: dict[str, dict[str, list[str]]] = {}
    for exp in exps:
        baseline = str(exp.summary.get("baseline"))
        for item in _model_items(exp):
            if item["name"] == baseline:
                continue
            bucket = seen.setdefault(item["name"], {"searched": [], "fixed": []})
            bucket["searched" if item.get("grid") else "fixed"].append(
                _short(exp) + (f" ({item['label']})" if item.get("label") else ""))
    rows = []
    for name, b in seen.items():
        if not b["fixed"]:
            continue
        rows.append([e(model_card(name)["name"]), e(", ".join(dict.fromkeys(b["searched"]))) or "-",
                     e(", ".join(dict.fromkeys(b["fixed"]))), e(NEXT_SEARCH.get(name, "주요 설정"))])
    windows = sorted({f"{_context(x):g}초" for x in exps if x.task != "thickness"})
    rows.append(["공통 설정", "-", "window 길이 (" + e(", ".join(windows)) + "), 샘플링 1초, 손실 함수 (계획서 고정)",
                 "window 길이 · 샘플링 간격"])
    return ('<p class="note">아래 모델·설정은 이번에 기본값 또는 고정값으로만 학습했습니다. 성능 차이를 모델 구조 때문이라고 '
            "단정하지 않고, 다음 실험에서 탐색할 예정입니다.</p>"
            + table(["모델", "탐색한 실험", "기본값·고정으로 쓴 실험", "다음 실험에서 탐색할 것"], rows))


def _config_results(exp: Experiment, configs: pd.DataFrame) -> str:
    picks = {p.target: p.label for p in exp.picks}
    has_region = configs["mae_cycle"].notna().any()
    blocks = []
    targets = list(dict.fromkeys(configs["target"]))
    for target in targets:
        part = configs[configs["target"] == target]
        rows = []
        for r in part.itertuples():
            mark = ""
            if r.label == picks.get(target):
                mark = '<span class="chip">최종 선택</span>'
            elif r.chosen:
                mark = '<span class="sub">모델 안에서 최고</span>'
            rows.append([e(model_card(r.model)["name"]), e(r.setting), str(r.seeds),
                         fmt(r.mae) + (f' <span class="sub">± {spread(r.mae_std)}</span>' if r.mae_std is not None
                                       and not pd.isna(r.mae_std) else ""),
                         *([fmt(r.mae_cycle), fmt(r.mae_other)] if has_region else []),
                         minutes(r.fit_minutes), mark])
        header = ["모델", "설정", "seed", "MAE ± seed 편차", *(["ALD 사이클", "그 외"] if has_region else []),
                  "학습 시간", "선택"]
        body = table(header, rows)
        unit = next((p.unit for p in exp.picks if p.target == target), "")
        blocks.append(body if len(targets) == 1 else
                      f"<details><summary>{e(target_name(target, unit))} · 조합 {len(part)}개</summary>{body}</details>")
    return ('<p class="note">같은 모델에서 설정만 바꾼 조합을 모두 적었습니다. seed 편차는 같은 설정을 seed만 바꿔 돌렸을 때의 '
            "표준편차입니다. 조합 사이 차이가 이보다 작으면 우열을 가리기 어렵습니다.</p>" + "".join(blocks))


def _verdict_html(text: str, effect: bool | None) -> str:
    cls = {True: "bad" if "나빠짐" in text else "good", False: "warn"}.get(effect, "")
    return f'<span class="st {cls}">{e(text)}</span>' if cls else f'<span class="sub">{e(text)}</span>'


def _effects_table(exp: Experiment, effects: list[dict[str, Any]]) -> str:
    rows = []
    for ef in effects:
        factor = ef["factor"]
        levels = " · ".join(
            (f"<b>{e(setting_value(factor, lv))} {fmt(m)}</b>" if lv == ef["best"]
             else f"{e(setting_value(factor, lv))} {fmt(m)}") + f' <span class="sub">({n})</span>'
            + (' <span class="chip">기준</span>' if lv == ef["ref"] else "")
            for lv, m, n in ef["levels"])
        changes = "<br>".join(
            f"→ {e(setting_value(factor, lv))}: {pct(ch)}" + (' <span class="sub">(seed 흔들림 안)</span>'
                                                               if clear is False else "")
            for lv, ch, clear in ef["changes"])
        rows.append([e(ef["target"]), e(model_card(ef["model"])["name"]), e(ef["name"]), levels, changes,
                     spread(ef["seed_std"]), _verdict_html(ef["verdict"], ef["effect"])])
    return ('<p class="note">설정 하나의 값별로, 나머지 설정과 seed를 모두 평균한 MAE입니다(괄호는 작업 수, 굵은 글씨가 가장 '
            "좋은 값). 변화는 <b>기준</b> 값에서 바꿨을 때입니다. 가장 좋은 값과 다른 값의 차이가 seed 편차로 본 표준오차의 "
            f"{CLEAR_SE:g}배를 넘으면 '효과 있음', 아니면 '불분명'입니다. seed를 한 번만 돌린 설정은 다른 설정 조합에서도 같은 값이 "
            "이기는지(방향 일관)만 표시했습니다. 대상 순서대로, 같은 대상 안에서는 오차가 작은 모델부터입니다.</p>"
            + table(["대상", "모델", "바꾼 설정", "값별 평균 MAE", "기준 대비 변화", "seed 편차", "판단"], rows,
                    cls="effects"))


def _classic_note(curve: dict[str, Any]) -> str:
    info, model = curve.get("pick_training") or {}, curve.get("pick_model")
    if model == "lightgbm" and info.get("iterations"):
        its = ", ".join(str(i) for i in info["iterations"])
        return (f"선택 모델 LightGBM은 반복 {its}회에서 조기 종료했습니다 "
                f"(최대 {model_defaults.LIGHTGBM['n_estimators']:,}회). epoch 개념이 없어 아래는 딥러닝 중 최고 모델의 곡선입니다.")
    if model == "ridge" and info.get("alpha") is not None:
        return f"선택 모델 Ridge는 검증으로 alpha {info['alpha']:g}를 골랐습니다(반복 학습 없음). 아래는 딥러닝 중 최고 모델의 곡선입니다."
    return f"선택 모델 {model_card(model or '')['name']}은 반복 학습이 없습니다. 아래는 딥러닝 중 최고 모델의 곡선입니다."


def _curves_section(exp: Experiment) -> str:
    if exp.task == "thickness":
        return '<p class="sub">두께 예측은 run 하나씩 빼며 맞추는 run 단위 모델이라 epoch 학습 곡선이 없습니다.</p>'
    if not exp.curves:
        return ""
    blocks = []
    many = len(exp.curves) > 3
    for pick in exp.picks:
        curve = exp.curves.get(pick.target)
        if not curve:
            continue
        chart = loss_curve(curve["history"], curve.get("best_epoch"), title=f"{pick.target} 학습 곡선")
        status = STATUS_TEXT.get(str(curve.get("status")), str(curve.get("status") or "-"))
        caption = (f"{nice_label(curve['label'])}: 최고 epoch {curve.get('best_epoch') or '-'} / "
                   f"학습한 epoch {curve.get('epochs') or len(curve['history'])} · {status} · 검증 MAE {fmt(curve['mae'])}")
        note = "" if curve["is_pick"] else f'<p class="sub">{e(_classic_note(curve))}</p>'
        body = note + (f'<div class="chartbox"><div class="ctitle">{e(caption)}</div>{chart}</div>' if chart else "")
        unit = pick.unit
        head = e(target_name(pick.target, unit)) + (" (선택 모델)" if curve["is_pick"] else " (딥러닝 최고 모델)")
        blocks.append(f"<details{'' if many else ' open'}><summary>{head}</summary>{body}</details>")
    return ('<p class="note">epoch별 학습 손실(회색)과 검증 손실(파랑)입니다. 손실은 정규화 단위라 값의 크기보다 추이를 봅니다. '
            "검증 손실이 가장 낮은 epoch(점선)의 가중치를 저장하고, 이후 개선이 없으면 조기 종료합니다.</p>" + "".join(blocks))


def _prev_verdict(item: dict[str, Any], seed_std: float) -> tuple[str, bool | None, str]:
    delta = float(item["mae"]) - float(item["ref"])
    if item.get("caution"):
        return "해석 주의 (정규화가 무너진 뒤의 누적 변화)", None, "-"
    if item.get("p") is not None:
        clear, basis = float(item["p"]) < 0.05, f"run 11개 검정 p={item['p']:g}"
    else:
        clear, basis = abs(delta) > 2 * seed_std, f"seed 편차({seed_std:g})의 {abs(delta) / seed_std:.1f}배"
    if clear:
        return ("효과 있음: " + ("좋아짐" if delta < 0 else "나빠짐")), True, basis
    return "불분명", False, basis


# Hyper Parameter tab: plain names for setting values in the summary and the charts
VALUE_NAMES = {"patchtst_ci": "신호별 독립(ci)", "patchtst_mix": "신호 혼합(mix)", "all": "전체(all)",
               "core": "핵심 A 대표(core)", "core_ab": "핵심 A·B 대표(core_ab)"}
STATUS_WORDS = {"clear": False, "unclear": "불분명", "norepeat": "반복 없음"}
BIG_EFFECT = 20.0  # % — the summary splits clear effects into large and small here


def _value_name(factor: str, value: Any) -> str:
    value = plain(value)
    return f"{value}초" if factor in SECONDS else VALUE_NAMES.get(str(value), str(value))


def _change_name(row: dict[str, Any]) -> str:
    name = FACTOR_NAMES.get(row["factor"], row["factor"]).split(" (")[0]
    return f"{name}: {_value_name(row['factor'], row['ref'])} → {_value_name(row['factor'], row['level'])}"


def _prev_items(h: dict[str, Any]) -> list[dict[str, Any]]:
    """The previous experiment's one-setting changes with their verdict and change in %, in phase order."""

    seed_std = float(h.get("seed_std") or 0.0145)
    items = []
    for item in h.get("hparam_effects") or []:
        verdict, effect, basis = _prev_verdict(item, seed_std)
        change = (float(item["ref"]) - float(item["mae"])) / float(item["ref"]) * 100
        items.append({**item, "verdict": verdict, "effect": effect, "basis": basis, "delta_pct": change,
                      "short": item.get("short") or f"{item['factor']} {item['change']}"})
    return items


def _hp_guide() -> str:
    points = [
        ("<b>%</b>는 <b>기준값(화살표 왼쪽)에서 바꿨을 때 오차가 줄어든 비율</b>입니다. 파랑(+)은 개선, 빨강(−)은 악화. 기준값은 "
         "입력 세트는 전체 신호(all), 숫자 설정은 작은 값, 모델 구성은 실험 설정 파일의 순서입니다."),
        ("<b>seed</b>는 같은 설정을 무작위 시작점만 바꿔 다시 돌린 것입니다. 설정이 같아도 결과가 조금 흔들리므로, 차이가 이 흔들림"
         f"(seed 편차)의 표준오차 {CLEAR_SE:g}배를 넘어야 <b>확실</b>로 봅니다. 회색 막대는 흔들림 안이라 <b>불분명</b>이거나, "
         "반복하지 않아 <b>반복 없음</b>인 것입니다. 불분명은 '효과 없음'이 아니라 '이 반복 횟수로는 구분할 수 없음'입니다."),
        ("<b>이번 실험은 검증 run, 이전 실험(2026-08)은 시험 run</b> 기준이라 두 실험의 숫자 크기를 직접 비교하지 않습니다. "
         "방향과 크기 순서만 봅니다."),
        ("<b>N개 평균</b>은 같은 설정 변경을 대상·모델 여러 개에서 해 본 경우 % 변화의 평균입니다. 대상마다 다를 수 있으니 "
         "⑥ 실험별 상세의 '설정 하나를 바꾼 효과' 표를 함께 봅니다."),
    ]
    return '<div class="facts"><ul>' + "".join(f"<li>{t}</li>" for t in points) + "</ul></div>"


def _hp_summary(now: list[tuple[Experiment, dict[str, Any]]], prev: list[dict[str, Any]]) -> str:
    cats = [("big", f"확실, 영향 큼 ({BIG_EFFECT:g} % 이상)"), ("small", f"확실, 영향 작음 ({BIG_EFFECT:g} % 미만)"),
            ("unclear", "구분 불가 (seed 편차 이내)"), ("norepeat", "반복 없음 (방향만 확인)")]

    def category(status: str, change: float) -> str:
        return ("big" if abs(change) >= BIG_EFFECT else "small") if status == "clear" else status

    cells: dict[str, tuple[list[tuple[float, str]], list[tuple[float, str]]]] = {k: ([], []) for k, _ in cats}
    for exp, r in now:
        cells[category(r["status"], r["change"])][0].append(
            (abs(r["change"]), f"[{e(_short(exp))}] {e(_change_name(r))} {pct(r['change'])}"))
    for it in prev:
        if it.get("caution"):
            continue
        cells[category("clear" if it["effect"] else "unclear", it["delta_pct"])][1].append(
            (abs(it["delta_pct"]), f"<b>{e(it['id'])}</b> {e(it['short'])} {pct(it['delta_pct'])}"))

    def cell(items: list[tuple[float, str]]) -> str:
        if not items:
            return '<span class="sub">-</span>'
        return '<ul class="tight">' + "".join(f"<li>{t}</li>" for _, t in sorted(items, key=lambda x: -x[0])) + "</ul>"

    rows = [[f"<b>{e(title)}</b>", cell(cells[k][0]), cell(cells[k][1])] for k, title in cats if any(cells[k])]
    if not rows:
        return '<p class="empty">비교할 설정 변경이 없습니다.</p>'
    return ('<p class="note">설정 하나만 바꾼 비교를 효과의 크기와 확실성으로 나눴습니다. 칸 안은 효과가 큰 순서입니다. '
            "이전 실험의 코드(N02, S1 → S2 …)는 ④에서 풀어 설명합니다.</p>"
            + table(["구분", "이번 실험 (검증 run)", "이전 실험 2026-08 (시험 run)"], rows, cls="hpsum"))


def _now_section(exps: list[Experiment], effects: dict[str, list[dict[str, Any]]]) -> str:
    charts, rows = [], []
    for exp in exps:
        summary = effect_summary(effects.get(exp.name, []))
        if not summary:
            continue
        bars = [(_change_name(r) + (f" · {r['n']}개 평균" if r["n"] > 1 else ""), r["change"], STATUS_WORDS[r["status"]])
                for r in summary]
        charts.append(f'<div class="chartbox"><div class="ctitle">{e(_exp_title(exp))}</div>'
                      f'{improvement_bars(bars, title=f"{_short(exp)} 설정 변경 효과")}</div>')
        for r in summary:
            counts = " · ".join(f"{word} {n}" for word, n in (("확실", r["n_clear"]), ("불분명", r["n_unclear"]),
                                                            ("반복 없음", r["n_norepeat"])) if n)
            rows.append([e(_short(exp)), e(_change_name(r)), pct(r["change"]),
                         f"대상 {r['targets']} × 모델 {r['models']} → {r['n']}개", e(counts)])
    if not charts:
        return '<p class="empty">이번 실험에서 설정을 바꿔 비교한 것이 없습니다.</p>'
    return ('<p class="note">실험별 그림입니다. 막대는 "기준값 → 변경값"의 효과이며, 같은 변경을 여러 대상·모델에서 했으면 '
            "평균입니다. 효과가 큰 순서로 정렬했습니다.</p>" + "".join(charts)
            + "<details><summary>막대별 수치 (평균한 작업 수와 판단 내역)</summary>"
            + table(["실험", "바꾼 설정", "평균 변화", "평균한 작업", "판단 내역 (대상·모델별)"], rows) + "</details>")


def _prev_section(h: dict[str, Any]) -> str:
    items = _prev_items(h)
    seed_std = float(h.get("seed_std") or 0.0145)
    out = [('<p class="note">2026-08 실험은 VG11의 10·20·30초 뒤 값을 PatchTST로 예측하면서, 기준 모델에서 설정을 하나씩 '
            f"바꿔 시험 run MAE(Torr)를 쟀습니다. seed만 바꿔도 ±{seed_std:g} 흔들려서, 차이가 그 2배를 넘거나 run 11개 통계 "
            "검정에서 p &lt; 0.05일 때만 효과로 봤습니다. 아래 순서대로 읽으면 코드 이름이 무엇인지 따라갈 수 있습니다.</p>")]
    story = h.get("hparam_story") or []
    if story:
        out.append("<h4>단계별 진행 내용</h4>"
                   + table(["단계", "한 일", "코드", "결과 (시험 run MAE)"], [[f"<b>{e(a)}</b>", e(b), e(c), e(d)]
                                                                   for a, b, c, d in story]))
    bars = [(f"{str(it.get('phase', '')).split(' ')[0]} · {it['short']} ({it['id']})", it["delta_pct"],
             False if it["effect"] else "불분명") for it in items if not it.get("caution")]
    if bars:
        out.append("<h4>설정 하나를 바꿨을 때 오차 변화 (단계 순서)</h4>"
                   '<p class="sub">A04~A09는 A03에서 모델이 무너진 뒤에 쌓은 변화라 그림에서 뺐습니다 (아래 표에는 있음).</p>'
                   f'<div class="chartbox">{improvement_bars(bars, title="이전 실험 설정 변경 효과")}</div>')
    rows, last = [], None
    for it in items:
        phase = str(it.get("phase") or "")
        rows.append([f"<b>{e(phase)}</b>" if phase != last else "", f"<b>{e(it['id'])}</b>",
                     e(it["short"]) + f'<div class="sub">{e(it["change"])}</div>', e(it.get("explain") or ""),
                     f"{fmt(it['ref'])} → {fmt(it['mae'])}", pct(it["delta_pct"]),
                     _verdict_html(it["verdict"], it["effect"])
                     + (f'<div class="sub">{e(it["basis"])}</div>' if it["basis"] != "-" else "")])
        last = phase
    out.append("<h4>실험 코드별 변경 내용</h4>"
               + table(["단계", "코드", "변경 내용", "설명", "MAE 전 → 후", "변화", "판단 (근거)"], rows, cls="prevfx"))
    return "".join(out)


def _experiment_detail(exp: Experiment, effects: list[dict[str, Any]]) -> str:
    cfg = exp.config
    train = cfg.get("train") or {}
    rows = [("seed", e(", ".join(str(s) for s in cfg.get("seeds", [42]))))]
    if exp.task != "thickness":
        rows.insert(0, ("window", f"최근 {_context(exp):g}초 → "
                        + ("지금 값" if exp.task == "virtual_sensor" else
                           ", ".join(f"{h:g}" for h in _horizons(exp)) + "초 뒤")))
        if train:
            rows.append(("딥러닝 학습", e(" · ".join(f"{k} {v}" for k, v in train.items()))))
    if exp.smoke:
        rows.append(("smoke", "run 6개·epoch 2로 줄인 확인용 실행 (성능 판단에 쓰지 않음)"))
    out = [kv(rows), "<h4>모델별 탐색 여부</h4>", _search_table(exp)]
    if effects:
        out += ["<h4>설정 하나를 바꾼 효과</h4>", _effects_table(exp, effects)]
    configs = config_table(exp)
    if len(configs):
        out += ["<h4>설정별 결과 (시도한 조합 전부)</h4>", _config_results(exp, configs)]
    curves = _curves_section(exp)
    if curves:
        out += ["<h4>학습 곡선</h4>", curves]
    notes = []
    for p in exp.picks:
        training = p.validation.get("training") or {}
        used = training.get("batch_size")
        if used and train.get("batch_size") and int(used) != int(train["batch_size"]):
            notes.append(f"{p.target} · {p.label}: batch {train['batch_size']} → {used} (GPU 메모리에 맞춰 자동 조정)")
    if notes:
        out.append("<h4>자동 조정</h4><ul>" + "".join(f"<li>{e(n)}</li>" for n in notes) + "</ul>")
    searched = sum(1 for item in _model_items(exp) if item.get("grid"))
    sets = len((cfg.get("data") or {}).get("input_sets") or {})
    what = " · ".join(x for x in (f"설정 탐색 모델 {searched}개" if searched else "",
                                  f"입력 세트 {sets}개 비교" if sets > 1 else "") if x) or "설정 탐색 없음 (기본값)"
    return (f'<details class="stage" id="hp-{e(exp.name)}"><summary>{e(_exp_title(exp))}'
            f'<div class="note">{e(what)} · seed {len(cfg.get("seeds", [42]))}개</div></summary>'
            + "".join(out) + "</details>")


def _hparams(data: ReportData) -> str:
    exps = data.experiments
    history = next((h for h in data.histories if h.get("hparam_effects")), None)
    if not exps and not history:
        return '<p class="empty">아직 끝난 실험이 없습니다.</p>'
    effects = {exp.name: factor_effects(exp) for exp in exps}
    now = [(exp, r) for exp in exps for r in effect_summary(effects[exp.name])]
    prev = _prev_items(history) if history else []
    out = ["<h3>① 보는 법</h3>", _hp_guide(),
           "<h3>② 결론 (영향이 큰 설정)</h3>", _hp_summary(now, prev)]
    if now:
        out += ["<h3>③ 이번 실험 (설정 하나를 바꿨을 때 오차 변화)</h3>", _now_section(exps, effects)]
    if history:
        out += ["<h3>④ 이전 실험 (2026-08) 변경 내용과 결과</h3>", _prev_section(history)]
    if exps:
        out += ["<h3>⑤ 탐색하지 않은 것 (다음 실험에서 탐색 예정)</h3>", _not_searched(exps),
                "<h3>⑥ 실험별 상세 (설정 · 효과 · 조합별 결과 · 학습 곡선)</h3>",
                ('<p class="note">실험 이름을 누르면 펼쳐집니다. 순서: 실험 설정 → 모델별 탐색 여부 → 설정 하나를 바꾼 효과(결론) → '
                 "시도한 조합 전부 → 학습 곡선.</p>")]
        out += [_experiment_detail(exp, effects[exp.name]) for exp in exps]
    if len(data.ledger):
        out += ["<h3>⑦ 누적 기록 (전체 실행 조합)</h3>", _ledger_section(data.ledger)]
    return "".join(out)


LEDGER_ROWS = 300  # newest rows shown in the report; the CSV next to it and the dashboard have all of them
TASK_SHORT = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서", "thickness": "두께"}


def _ledger_section(ledger: pd.DataFrame) -> str:
    from ..ledger import best_by_target

    ok = ledger[ledger["mae"].notna()]
    runs = ok.groupby("exp_dir", sort=False)
    out = [f'<p class="note">위 ①~⑥은 실험마다 가장 최근 실행만 봅니다. 여기는 <code>work/experiments/</code>의 모든 실행 '
           f"{runs.ngroups}회, 학습 기록 {len(ok):,}개(대시보드에서 돌린 것 포함)를 모았습니다. 같은 내용을 대시보드 "
           "<b>3. 하이퍼파라미터·학습 → ③ 누적 기록</b>에서 거르고 정렬해 볼 수 있고, 보고서 폴더의 "
           "<code>experiment_records.csv</code>에도 저장됩니다.</p>"]
    best = best_by_target(ok)
    if len(best):
        out.append("<h4>대상별 최고 기록 (검증 MAE)</h4>")
        out.append(table(["과제", "대상", "검증 MAE", "개선율", "모델", "설정", "입력", "언제 · 실험"], [
            [e(TASK_SHORT.get(r.task, r.task)), e(r.target), fmt(r.mae), pct(r.improvement_pct), e(nice_label(r.label)),
             f'<span class="sub">{e(r.settings)}</span>', e(r.input_set),
             f"{r.when:%m-%d %H:%M}<div class='sub'>{e(r.experiment)}</div>" if r.when == r.when else e(r.experiment)]
            for r in best.itertuples()]))
    rows = []
    for exp_dir, group in list(runs)[:40]:
        first = group.iloc[0]
        tops = []
        for target, g in group[~group["model"].isin(BASELINE_MODELS)].groupby("target", sort=False):
            top = g.loc[g["mae"].idxmin()]
            tops.append(f"{e(target)} {fmt(top['mae'])} <span class='sub'>{e(nice_label(top['label']))} "
                        f"{pct(top['improvement_pct'])}</span>")
        rows.append([f"{first['when']:%m-%d %H:%M}" if first["when"] == first["when"] else "-",
                     e(first["experiment"]) + f"<div class='sub'>{e(first['memo'])}</div>",
                     e(first["source"]), str(len(group)), "<br>".join(tops) or "-"])
    out.append("<h4>실행별 요약 (최근 40회)</h4>")
    out.append(table(["언제", "실험 · 메모", "출처", "학습 수", "대상별 최고 (모델 · 기준선 대비)"], rows))
    recent = ok.sort_values("when", ascending=False).head(LEDGER_ROWS)
    detail = table(["언제", "실험", "대상", "모델", "설정", "입력", "seed", "검증 MAE", "개선율", "학습"], [
        [f"{r.when:%m-%d %H:%M}" if r.when == r.when else "-", e(r.experiment), e(r.target), e(nice_label(r.label)),
         f'<span class="sub">{e(r.settings)}</span>', e(r.input_set), str(r.seed), fmt(r.mae), pct(r.improvement_pct),
         minutes(r.fit_minutes)] for r in recent.itertuples()])
    out.append(f"<details><summary>학습 기록 전체 (최근 {len(recent):,}개)</summary>{detail}</details>")
    return "".join(out)


def _results(data: ReportData, examples: bool) -> str:
    out = ["<h3>과제별 최고 성능</h3>", _purpose_table(data.experiments)]
    for exp in data.experiments:
        s = exp.summary
        out.append(f'<h3 id="r-{e(exp.name)}">{e(_exp_title(exp))}</h3>')
        out.append(f"<p>작업 {s.get('jobs')}개 (실패 {s.get('failed')}) · 총 {minutes(s.get('wall_minutes'))} · "
                   f"평가: {e(_eval_word(exp))} · 기준선: {e(_baseline(exp))} · 결과 폴더 "
                   f"<code>{e(_rel(exp.directory, data.work_root))}</code></p>")
        table_ = exp.table
        picks = {p.target: p for p in exp.picks}
        for target, group in table_.groupby("target", sort=False):
            group = group.sort_values("mae")
            pick = picks.get(target)
            unit = pick.unit if pick else ""
            head = (f"{e(target_name(target, unit))} · 최고 {e(nice_label(pick.label))} {fmt(pick.mae)} "
                    f"{pct(pick.improvement_pct)}" if pick else e(target))
            body = [_results_table(exp, group)]
            bars = improvement_bars([(nice_label(r.label), None if pd.isna(r.improvement_pct) else float(r.improvement_pct))
                                     for r in group.itertuples() if not r.is_baseline], title=f"{target} 개선율")
            if bars:
                body.append(f'<div class="chartbox">{bars}</div>')
            if pick:
                body.append(_strengths(pick))
                if examples and pick.example is not None:
                    body.append(_example_chart(exp, pick))
            out.append(f"<details open><summary>{head}</summary>{''.join(body)}</details>"
                       if len(picks) <= 4 else f"<details><summary>{head}</summary>{''.join(body)}</details>")
        if exp.test is not None:
            out.append("<h4>시험 run 결과 (최종)</h4>")
            hcols = _horizon_set([exp]) if exp.task == "forecast" else []
            rows = []
            for r in exp.test.to_dict(orient="records"):
                th = _test_horizon_values(exp, r["target"], r["label"])
                std = r.get("mae_std")
                rows.append([e(r["target"]), e(nice_label(r["label"])),
                             fmt(r["mae"]) + (f' <span class="sub">± {spread(std)}</span>'
                                              if std is not None and not pd.isna(std) else ""),
                             pct(None if pd.isna(r.get("improvement_pct")) else r.get("improvement_pct")),
                             *_hcells(th, hcols),
                             *([fmt(r.get("mae_cycle")), fmt(r.get("mae_other"))] if "mae_cycle" in r else [])])
            out.append(table(["대상", "모델", "시험 MAE", "기준선 대비", *[horizon_label(c) for c in hcols],
                              *(["ALD 사이클", "그 외"] if "mae_cycle" in exp.test else [])], rows))
        failed = exp.results[exp.results["status"] != "ok"] if "status" in exp.results else exp.results.iloc[:0]
        if len(failed):
            out.append("<h4>실패한 작업</h4><ul>" + "".join(
                f"<li>{e(r.label)} ({e(r.target)}): {e(getattr(r, 'error', ''))}</li>" for r in failed.itertuples())
                + "</ul>")
    return "".join(out)


def _results_table(exp: Experiment, group: pd.DataFrame) -> str:
    has_region = "mae_cycle" in group and group["mae_cycle"].notna().any()
    hcols = horizon_columns(group.columns) if exp.task == "forecast" else []
    header = (["순위", "모델", "MAE", "개선율"] + [horizon_label(c) for c in hcols]
              + (["ALD 사이클", "그 외"] if has_region else []) + ["seed", "학습 시간", "처리 속도"])
    rows = []
    for r in group.itertuples():
        values = r._asdict()
        cells = [str(int(r.rank)) if not pd.isna(r.rank) else "-",
                 e(nice_label(r.label)) + (' <span class="chip">기준선</span>' if r.is_baseline else ""),
                 fmt(r.mae) + (f' <span class="sub">± {spread(r.mae_std)}</span>'
                               if not pd.isna(getattr(r, "mae_std", np.nan)) else ""),
                 "-" if r.is_baseline else pct(None if pd.isna(r.improvement_pct) else r.improvement_pct)]
        cells += _hcells({c: values.get(c) for c in hcols}, hcols)
        if has_region:
            cells += [fmt(getattr(r, "mae_cycle", None)), fmt(getattr(r, "mae_other", None))]
        speed = getattr(r, "windows_per_second", None)
        cells += [str(int(r.seeds)), minutes(r.fit_minutes),
                  f"{float(speed):,.0f} window/s" if speed is not None and not pd.isna(speed) else "-"]
        rows.append(cells)
    return table(header, rows)


def _strengths(pick: Pick) -> str:
    good = pick.validation.get("strengths") or []
    bad = pick.validation.get("weaknesses") or []
    if not good and not bad:
        return ""
    def items(lines: list[str]) -> str:
        return "".join(f"<li>{e(x)}</li>" for x in lines) or '<li class="sub">없음</li>'

    return ('<div class="sw"><div><b class="st good">강점</b><ul>' + items(good)
            + '</ul></div><div><b class="st bad">약점</b><ul>' + items(bad) + "</ul></div></div>")


def _example_chart(exp: Experiment, pick: Pick) -> str:
    out = []
    for frame, what in ((pick.example, "run 전체"), (pick.example_zoom, "ALD 사이클 중간 10분 확대")):
        if frame is None:
            continue
        true_cols = [c for c in frame.columns if c.startswith("true_")]
        if not true_cols:
            continue
        col = true_cols[0]
        h = col[len("true_"):]
        t = frame["t_s"].to_numpy(float) / 60
        shade = frame["in_cycle"].to_numpy(bool) if "in_cycle" in frame and what == "run 전체" else None
        title = (f"{pick.target} 예측 예시 ({what}), 검증 run {frame['run_id'].iloc[0]}"
                 + ("" if h == "0s" else f", {h} 뒤 값"))
        chart = line_chart(t, [("실측", frame[col].to_numpy(float), "ln-true"),
                               (f"예측 ({nice_label(pick.label)})", frame[f"pred_{h}"].to_numpy(float), "ln-pred")],
                           unit=pick.unit, shade=shade, title=title)
        if chart:
            out.append(f'<div class="chartbox"><div class="ctitle">{e(title)}</div>{chart}</div>')
    return "".join(out)


def _weights(data: ReportData) -> str:
    out = []
    rows = []
    for exp in data.experiments:
        for p in exp.picks:
            files = "<br>".join(f"<code>{e(w['name'])}</code> {_size(w['bytes'])} <span class='sub'>sha256 {e(w['sha256'])}</span>"
                                for w in p.weights) or '<span class="sub">LOO 평가만 함 → 패키지에서 다시 맞춤</span>'
            rows.append([e(exp.name), e(p.target), e(nice_label(p.label)) + f'<div class="sub">seed {p.seed}</div>', files,
                         f"<code>{e(_rel(p.job_dir, data.work_root))}</code>"])
    out.append("<h3>대상별 최고 모델의 가중치 파일</h3>")
    out.append(table(["실험", "대상", "모델", "가중치 파일", "폴더 (work/ 기준)"], rows)
               if rows else '<p class="empty">아직 끝난 실험이 없습니다.</p>')
    out.append("<h3>모델 패키지 (가중치 + 실행 파일)</h3>")
    prow = []
    for exp in data.experiments:
        exp_rel = _rel(exp.directory, data.work_root)
        if exp.package:
            pkg_rel = _rel(Path(exp.package["path"]), data.work_root)
            prow.append([e(exp.name), '<span class="st good">만들어짐</span>',
                         f"<code>{e(pkg_rel)}/</code><div class='sub'>zip: {e(pkg_rel)}.zip · "
                         f"{e(exp.package.get('created', ''))}</div>"])
        else:
            prow.append([e(exp.name), '<span class="st warn">아직 없음</span>',
                         f"<code>ald-export-model work/{e(exp_rel)}</code>"])
    if prow:
        out.append(table(["실험", "상태", "위치 / 만드는 명령"], prow))
    out.append("<h3>실행 방법</h3>")
    out.append(table(["어디서", "명령", "결과"], [
        ["서버", "<code>cd /data1/Agent/ald-lab &amp;&amp; source .venv/bin/activate &amp;&amp; source scripts/env.sh</code>", "환경 준비"],
        ["서버", "<code>ald-export-model work/experiments/&lt;실험&gt;/&lt;시각&gt;</code>", "work/packages/&lt;실험&gt;_&lt;시각&gt;/ + .zip"],
        ["서버", "<code>bash work/packages/&lt;…&gt;/&lt;대상&gt;__&lt;모델&gt;/predict.sh &lt;CSV 파일 또는 폴더&gt;</code>",
         "같은 폴더의 predictions.csv (실측이 있으면 MAE 출력)"],
        ["Windows PC", "<code>predict.bat &lt;CSV 파일 또는 폴더&gt; [출력 CSV]</code>", "ald-lab이 설치된 Python 필요"],
        ["서버", "<code>ald-final-test work/experiments/&lt;실험&gt;/&lt;시각&gt;</code>", "최종 후보의 시험 run 성능 (보고 직전 한 번)"],
        ["서버", "<code>bash scripts/make_report.sh</code>", "이 보고서를 새로 만듦 (work/reports/)"],
    ]))
    out.append("<details><summary>패키지 폴더 구성</summary>" + table(["파일", "내용"], [
        ["<code>model.pt</code>", "딥러닝 가중치 (PyTorch state_dict + 구조 설정)"],
        ["<code>lightgbm.joblib</code> · <code>ridge.npz</code>", "고전 모델"],
        ["<code>estimator.joblib</code>", "두께 모델 (학습+검증 run 전체로 다시 맞춘 것)"],
        ["<code>template*.parquet</code>", "레시피 평균 (가상 센서가 출발하는 기준값)"],
        ["<code>job.json</code>", "입력 목록, 학습 run 정규화 값, window 설정"],
        ["<code>README.txt</code>", "In/Out/모델/성능/실행 방법 요약"],
        ["<code>predict.sh</code> · <code>predict.bat</code>", "실행 파일 (서버 / Windows)"],
    ]) + "</details>")
    return "".join(out)


# ----------------------------------------------------------------------------- 특이사항: 학습 시간 단축

# Fixed categorical order for models in the GPU timeline; baselines are grey. Named in a text legend.
TIMELINE_COLORS = {"ridge": "var(--s1)", "lightgbm": "var(--s2)", "patchtst": "var(--s3)", "dlinear": "var(--s4)",
                   "itransformer": "var(--s5)", "gru": "var(--s6)", "pls": "var(--s7)"}
BASELINE_MODELS = {"persistence", "template", "mean"}


def _speed_reference(data: ReportData) -> dict[str, Any]:
    return next((h["speed"] for h in data.histories if h.get("speed")), {})


def _figure_numbers(data: ReportData) -> tuple[dict[str, Any], str]:
    """Rows, signals, window, batch and patch of the VG11 forecast (else the first windowed experiment)."""

    exps = [x for x in data.experiments if x.task != "thickness" and x.prep]
    exp = next((x for x in exps if x.name.startswith("g1_forecast_vg11")), exps[0] if exps else None)
    numbers = {"rows": 800_000, "channels": 38, "context": 192, "batch": 1024, "patch": 16, "stride": 8, "sample": 1.0}
    if exp is None:
        return numbers, "숫자는 예시입니다"
    cfg = exp.config or {}
    sample = float((cfg.get("data") or {}).get("sample_seconds", 1.0))
    numbers["sample"] = sample
    numbers["context"] = int(round(float((cfg.get("window") or {}).get("context_seconds", 192)) / sample))
    numbers["batch"] = int((cfg.get("train") or {}).get("batch_size", 1024))
    numbers["rows"] = int((((exp.prep or {}).get("bundles") or {}).get("train") or {}).get("rows") or numbers["rows"])
    deep = exp.results[exp.results["kind"] == "deep"] if "kind" in exp.results else exp.results
    if "n_inputs" in deep and deep["n_inputs"].notna().any():
        numbers["channels"] = int(deep["n_inputs"].median())
    specs = [m for m in cfg.get("models") or [] if isinstance(m, dict) and m.get("name") == "patchtst"]
    specs.sort(key=lambda m: (m.get("params") or {}).get("channel_mode") != "mixing")
    if specs:
        params, grid = specs[0].get("params") or {}, specs[0].get("grid") or {}
        patch = params.get("patch_seconds") or (grid.get("patch_seconds") or [16])[0]
        stride = params.get("stride_seconds") or (grid.get("stride_seconds") or [8])[0]
        numbers["patch"] = max(1, int(round(float(patch) / sample)))
        numbers["stride"] = max(1, int(round(float(stride) / sample)))
    return numbers, f"그림의 숫자는 {_short(exp)} 실험({e(exp.name)})의 실제 값입니다"


def _timeline_stats(frame: pd.DataFrame) -> dict[str, Any]:
    devices = sorted(frame["device"].unique(), key=device_order)
    span = float(frame["end_s"].max() - frame["start_s"].min()) / 60
    total = float(frame["seconds"].sum()) / 60
    busy = {d: float(frame.loc[frame["device"] == d, "seconds"].sum()) / 60 for d in devices}
    return {"devices": devices, "span": span, "total": total, "busy": busy, "jobs": len(frame),
            "speedup": total / span if span > 0 else None}


def _lane_name(device: str) -> str:
    return device.replace("cuda:", "GPU ") if device.startswith("cuda") else device.upper()


def _timelines(data: ReportData) -> tuple[str, dict[str, Any] | None, list[str]]:
    """Per experiment run: a Gantt chart of its jobs by device, plus a summary table; the run with the most jobs
    and every device seen."""

    blocks, best = [], None
    timelines = [(x, job_timeline(x, data.journal)) for x in data.experiments if x.task != "thickness"]
    timelines = [(x, t) for x, t in timelines if len(t)]
    if not timelines:
        return '<p class="empty">진행 기록에 장치별 작업 기록이 없습니다.</p>', None, []
    devices = sorted({d for _, t in timelines for d in t["device"]}, key=device_order)
    biggest = max(timelines, key=lambda xt: len(xt[1]))[0].name
    rows = []
    for exp, frame in timelines:
        st = _timeline_stats(frame)
        if exp.name == biggest:
            best = {"exp": exp, **st}
        order = [*BASELINE_MODELS, *TIMELINE_COLORS]
        models = sorted(dict.fromkeys(frame["model"]), key=lambda m: order.index(m) if m in order else len(order))
        bars = [(r.device, r.start_s / 60, r.end_s / 60,
                 "var(--muted)" if r.model in BASELINE_MODELS else TIMELINE_COLORS.get(r.model, "var(--s8)"),
                 f"{r.title} · {minutes(r.seconds / 60)}" + ("" if r.status == "ok" else f" · {r.status}"))
                for r in frame.itertuples()]
        legend = ('<div class="legend">'
                  + "".join(f'<span><i style="background:{"var(--muted)" if m in BASELINE_MODELS else TIMELINE_COLORS.get(m, "var(--s8)")}"></i>'
                            f'{e(model_card(m)["name"])}</span>' for m in models) + "</div>")
        chart = timeline_chart(bars, [(d, _lane_name(d)) for d in st["devices"]], title=f"{exp.name} 작업 타임라인")
        use = " · ".join(f"{_lane_name(d)} {st['busy'][d] / st['span'] * 100:.0f} %" for d in st["devices"]) \
            if st["span"] else "-"
        rows.append([f"{e(_short(exp))}<div class='sub'>{e(exp.name)}</div>", str(st["jobs"]),
                     e(", ".join(_lane_name(d) for d in st["devices"])), minutes(st["total"]), minutes(st["span"]),
                     f"×{st['speedup']:.1f}" if st["speedup"] else "-", e(use)])
        open_attr = " open" if exp.name == biggest else ""
        blocks.append(f"<details{open_attr}><summary>{e(_short(exp))} · 작업 {st['jobs']}개 · 장치 {len(st['devices'])}개 · "
                      f"실제 {minutes(st['span'])}</summary>{legend}<div class='chartbox'>{chart}</div></details>")
    summary = table(["실험", "작업 수", "장치", "순서대로 했다면", "실제 걸린 시간", "배속", "장치별 가동률"], rows)
    return summary + "".join(blocks), best, devices


def _notes(data: ReportData) -> str:
    ref = _speed_reference(data)
    ref_wps = float(ref.get("windows_per_second") or 3600)
    hours = ref.get("hours_per_model") or [2, 8]
    speeds = speed_table(data.experiments)
    timeline_html, best, devices = _timelines(data)
    numbers, numbers_note = _figure_numbers(data)
    out = ['<p class="note">이번 시스템에서 학습 시간을 크게 줄인 방법을 정리했습니다. 핵심은 두 가지입니다. '
           "<b>① 학습용 window와 patch를 CPU가 아니라 GPU에서 한꺼번에 자르는 것</b>, "
           "<b>② GPU 여러 장에 서로 다른 학습 작업을 동시에 맡기는 것</b>. 아래 숫자는 이 보고서에 들어간 실험 기록에서 "
           "계산합니다.</p>",
           "<h3>요약 (이전 2026-08 대비)</h3>"]
    tiles = []
    row = name = None
    if len(speeds):  # compare like with like: the previous numbers are PatchTST
        mix = speeds[speeds["spec"] == "patchtst_mix"]
        row = (mix if len(mix) else speeds[speeds["model"] == "patchtst"] if (speeds["model"] == "patchtst").any()
               else speeds).iloc[0]
        name = nice_label(row["spec"])
        tiles.append(tile(f"초당 학습 window ({name})", f"{row['wps']:,.0f}",
                          f"이전 약 {ref_wps:,.0f} → ×{row['wps'] / ref_wps:.1f}"))
        tiles.append(tile(f"모델 하나 학습 시간 ({name}, 중앙값)", minutes(row["minutes"]),
                          f"이전 {hours[0]}~{hours[1]}시간"))
        peak = speeds["peak_mb"].max()
        tiles.append(tile("최대 GPU 메모리", f"{peak / 1024:.1f} GB" if pd.notna(peak) and peak >= 1024
                          else (f"{peak:,.0f} MB" if pd.notna(peak) else "-"),
                          f"이전 {ref.get('gpu_memory_mb', 150)} MB (GPU 활용도 낮음)" if pd.notna(peak)
                          else "CPU 실행이라 기록 없음"))
    if best:
        tiles.append(tile("GPU 동시 사용", f"{len(best['devices'])}장 · ×{best['speedup']:.1f}" if best["speedup"] else "-",
                          f"{e(_short(best['exp']))}: 순서대로 {minutes(best['total'])} → 실제 {minutes(best['span'])}"))
    out.append(f'<div class="tiles">{"".join(tiles)}</div>' if tiles else '<p class="empty">아직 끝난 딥러닝 실험이 없습니다.</p>')
    train = next((x.config.get("train") or {} for x in data.experiments if x.task != "thickness" and x.config), {})
    out.append(table(["이전 코드에서 시간이 걸린 곳", "이번 조치", "코드"], cls="where", rows=[
        ["window를 CPU에서 하나씩 자름 (window마다 Python 함수 1번: "
         + e(ref.get("per_window_work", "배열 자르기·복사")) + ")",
         "학습 데이터 전체를 GPU에 한 번 올리고, batch는 GPU에서 인덱스 연산 한 번으로 자름", "<code>training/windows.py</code>"],
        [f"batch {ref.get('batch_size', 128)}개씩 CPU→GPU 복사", f"batch {train.get('batch_size', 1024):,}개, 복사 없음 "
         "(데이터가 이미 GPU에 있음)", "<code>training/trainer.py</code>"],
        ["epoch마다 겹치는 window를 모두 학습 (이웃 window는 1초만 다름)",
         f"epoch마다 무작위 {int(train.get('windows_per_epoch', 200_000)):,}개만", "<code>windows_per_epoch</code>"],
        ["epoch마다 검증 run 전체 평가", f"조기 종료는 고르게 뽑은 검증 window {int(train.get('val_windows', 50_000)):,}개로 "
         "판단, 전체 검증은 학습 후 한 번", "<code>val_windows</code>"],
        ["32비트 계산", "bf16 자동 혼합 정밀도 + TF32 행렬 곱 (RTX 4090 지원)", "<code>trainer._autocast</code>"],
        ["실험마다 CSV를 다시 읽고 정리", "1초 평균·구간 분할한 결과를 <code>work/prepared/</code>에 저장해 재사용",
         "<code>training/prepare.py</code>"],
        ["GPU 1장씩 손으로 실행", "실험 하나의 작업 전부를 GPU 여러 장에 자동 분배", "<code>runner._run_parallel</code>"],
    ]))

    # ① GPU window / patch cutting
    out.append("<h3>① window와 patch를 GPU에서 자르기</h3>")
    out.append(
        '<p class="note">모델은 "최근 '
        f"{numbers['context'] * numbers['sample']:g}초 동안의 신호"
        '"(window 하나)를 보고 미래 값을 예측합니다. 학습에는 이런 window가 수십만 개 필요하고, window는 1초씩 밀리며 '
        "겹칩니다. 이 window를 <b>어디서, 몇 개씩</b> 자르는지가 학습 속도를 좌우했습니다.</p>")
    prev_steps = [("시계열 배열", "CPU 메모리"), ("window 1개 자르기", "Python 함수 1번 = 1개"),
                  (f"{ref.get('batch_size', 128)}개 묶기", "DataLoader (CPU)"), ("GPU로 복사", "batch마다"),
                  ("모델 계산", "GPU는 대부분 대기")]
    now_steps = [("시계열 전체", "GPU에 처음 한 번"), (f"끝 위치 {numbers['batch']:,}개 추첨", "GPU 난수"),
                 ("인덱스 한 번으로 batch", "GPU 연산 1번"), ("patch로 나누기", "unfold, 복사 없음"),
                 ("모델 계산", "GPU 상시 가동")]
    best_wps = (f"초당 약 {row['wps']:,.0f} window ({name}) · 모델 하나 {minutes(row['minutes'])}"
                if row is not None else "")
    out.append(step_lanes([
        ("이전 (2026-08): CPU에서 window를 하나씩", "prev", prev_steps,
         f"초당 약 {ref_wps:,.0f} window · GPU 메모리 {ref.get('gpu_memory_mb', 150)} MB · 모델 하나 {hours[0]}~{hours[1]}시간"),
        ("이번: GPU에서 batch 단위로 한꺼번에", "now", now_steps, best_wps),
    ], label="이전과 이번의 batch 만드는 과정"))
    out.append(gather_figure(**numbers, label="GPU에서 batch를 만드는 과정"))
    out.append(f'<p class="sub">{numbers_note}.</p>')
    out.append("<h4>단계별 설명</h4><ol class='steps'>"
               f"<li><b>시계열 전체를 GPU에 한 번 올립니다.</b> 학습 run을 1초 평균하고 신호별로 정규화한(평균·표준편차는 학습 run에서만 계산) 표 "
               f"하나({numbers['rows']:,}행 × 신호 {numbers['channels']}개, 약 "
               f"{numbers['rows'] * numbers['channels'] * 4 / 2**20:,.0f} MB)를 학습 시작 때 GPU 메모리에 올립니다. "
               "RTX 4090(24 GB)에 충분히 들어갑니다. 이후 학습이 끝날 때까지 CPU와 데이터를 주고받지 않습니다.</li>"
               "<li><b>window가 끝날 수 있는 위치를 미리 정해 둡니다.</b> window와 예측 시점(10·20·30초 뒤)이 run 경계나 "
               "시간이 끊긴 곳을 넘지 않는 행 번호만 목록(ends)으로 만듭니다. 그래서 무작위로 뽑아도 run이 섞인 window는 "
               "나오지 않습니다.</li>"
               f"<li><b>batch마다 끝 위치 {numbers['batch']:,}개를 GPU에서 뽑습니다.</b> epoch 시작 때 "
               "<code>torch.randperm</code>으로 순서를 섞고 앞에서부터 잘라 씁니다.</li>"
               f"<li><b>인덱스 연산 한 번으로 batch 전체를 만듭니다.</b> 끝 위치 {numbers['batch']:,}개에 "
               f"(−{numbers['context'] - 1}, …, −1, 0)을 더한 인덱스 행렬({numbers['batch']:,} × {numbers['context']})로 "
               f"표를 한 번에 읽으면 ({numbers['batch']:,}, {numbers['context']}, {numbers['channels']}) 크기의 입력이 "
               "나옵니다. 정답(10·20·30초 뒤 값)도 같은 방식으로 한 번에 읽습니다. window 수만큼 반복하던 Python 코드가 "
               "batch당 한 줄이 됩니다.</li>"
               "<li><b>patch는 모델 안에서 나눕니다.</b> PatchTST는 window를 일정 길이(patch)로 나눠 토큰으로 씁니다. "
               "<code>unfold</code>는 같은 메모리를 patch 단위로 다시 보는 연산이라 복사가 없고, 이것도 GPU에서 합니다.</li>"
               "</ol>")
    prev_code = ("# 이전: window 하나마다 이 함수가 한 번 불림 (CPU, Python)\n"
                 "def __getitem__(self, index):\n"
                 "    segment, endpoint = self._locate(index)\n"
                 "    start = endpoint - self.context_steps + 1\n"
                 "    x = segment.values[start:endpoint + 1, features].T.copy()   # 자르기 + 복사\n"
                 "    y = segment.values[np.ix_(targets, target_cols)].T.copy()\n"
                 "    ...  # 시각 문자열 4개 만들기, recipe·step JSON 해석 2번\n"
                 "    return {\"x\": torch.from_numpy(x), \"y\": torch.from_numpy(y), ...}\n"
                 f"# DataLoader가 {ref.get('batch_size', 128)}개를 모아 batch로 묶은 뒤 GPU로 복사")
    now_code = ("# 이번: batch 하나에 아래 몇 줄, 모두 GPU에서 (training/trainer.py · windows.py)\n"
                f"order = torch.randperm(len(train), device=\"cuda\")[:{int(train.get('windows_per_epoch', 200_000)):_}]\n"
                f"e = ends[order[i:i + {numbers['batch']}]]       # window 끝 위치\n"
                f"x = series[e[:, None] + offsets]    # ({numbers['batch']}, {numbers['context']}, {numbers['channels']}) 인덱스 한 번\n"
                "y = targets[e[:, None] + horizons]  # 10·20·30초 뒤 정답\n"
                "\n# 모델 안 (models/deep.py)\n"
                "patches = z.unfold(-1, patch_len, stride)  # (batch, 신호, patch 수, patch 길이) 복사 없음")
    out.append(f'<div class="sw"><div><h4>이전 코드 (요약)</h4><pre class="code">{e(prev_code)}</pre>'
               f'<p class="sub">{e(ref.get("code", ""))}</p></div>'
               f'<div><h4>이번 코드</h4><pre class="code">{e(now_code)}</pre></div></div>')
    out.append("<h4>속도 향상 요인</h4><ul>"
               f"<li>Python 함수 호출이 window마다 1번 → batch마다 1번으로, {numbers['batch']:,}배 줄었습니다.</li>"
               "<li>batch마다 CPU→GPU 복사가 없어졌습니다. GPU는 데이터를 기다리지 않고 계산만 합니다.</li>"
               "<li>이전에는 GPU 메모리를 150 MB만 썼습니다. 계산할 것이 없어서 GPU가 대부분 쉬고 있었다는 뜻입니다.</li>"
               "<li>여기에 epoch당 학습 window 수 축소, 검증 부분집합, bf16이 더해져 모델 하나가 몇 시간에서 몇 분으로 "
               "줄었습니다 (위 표).</li></ul>")

    # ② several GPUs
    n_jobs = sum(len(x.jobs) for x in data.experiments)
    out.append("<h3>② GPU 여러 장을 동시에 쓰기</h3>")
    out.append('<p class="note">모델 하나를 GPU 4장에 나눠 학습하는 방식(분산 학습)이 아닙니다. '
               "<b>서로 다른 학습 작업을 GPU마다 하나씩 동시에</b> 돌립니다. 실험 하나에는 대상 × 모델 × 설정 조합 × seed만큼 "
               "작업이 있고, 작업끼리는 주고받을 것이 없어서 이 방식이 가장 단순하고 빠릅니다.</p>")
    out.append(queue_figure(devices if any(d.startswith("cuda") for d in devices)
                            else ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], n_jobs=max(n_jobs, 1),
                            label="작업 대기열과 GPU별 작업자"))
    out.append("<ol class='steps'>"
               "<li><b>작업 목록</b>: 실험 YAML을 작업 목록으로 펼칩니다 (예: VG11 예측 = 모델 7종 × 설정 조합 × seed 3개).</li>"
               "<li><b>기준선 먼저</b>: 현재값 유지·레시피 평균 같은 기준선을 먼저 돌려, 나머지 작업이 비교할 기준을 정합니다.</li>"
               "<li><b>대기열</b>: 나머지 작업을 대기열 하나에 넣고, <code>--gpus 0,1,2,3</code>이면 GPU마다 작업자 프로세스를 "
               "하나씩(4개) 띄웁니다. CPU 코어도 작업자 수로 나눠 줍니다 (LightGBM처럼 CPU를 쓰는 모델용).</li>"
               "<li><b>작업자</b>: 각 작업자는 정리된 데이터(<code>work/prepared/</code>)를 읽어 자기 GPU에 한 번 올리고, "
               "대기열에서 작업을 하나 꺼내 학습 → 결과 저장 → 다음 작업을 꺼냅니다. 대기열이 빌 때까지 반복합니다.</li>"
               "<li><b>먼저 끝난 GPU가 다음 작업을 가져갑니다</b>: 오래 걸리는 작업과 짧은 작업이 섞여도 GPU가 놀지 않습니다. "
               "미리 GPU별로 작업을 나눠 두는 방식보다 끝나는 시각이 고릅니다.</li>"
               "<li><b>결과 모음</b>: 모든 작업이 끝나면 요약(summary.md)과 진행 기록을 만듭니다. 작업 하나가 실패해도 "
               "그 작업만 실패로 기록되고 나머지는 계속됩니다.</li></ol>")
    out.append("<h4>모델 하나를 GPU 4장에 나누지 않은 이유</h4><ul>"
               "<li>모델이 작아(파라미터 수만~수백만 개) GPU 한 장으로 충분히 빠릅니다. 4장에 나누면 batch마다 GPU끼리 "
               "결과를 맞추는 통신이 생겨 오히려 손해입니다.</li>"
               "<li>작업이 서로 독립이라 GPU 4장이면 거의 4배가 됩니다. seed가 같으면 어느 GPU에서 돌아도 결과가 같습니다.</li>"
               "<li>코드가 단순하고, 한 작업의 문제가 다른 작업에 번지지 않습니다.</li></ul>")
    out.append("<h4>실제 실행 기록 (장치별 타임라인)</h4>"
               '<p class="note">진행 기록에 남은 작업별 시작·종료 시각과 장치로 그렸습니다. 막대 하나가 작업 하나이고, '
               "막대에 마우스를 올리면 작업 이름과 시간이 나옵니다. '순서대로 했다면'은 작업 시간의 합, '배속'은 그 합을 "
               "실제 걸린 시간으로 나눈 값입니다.</p>")
    out.append(timeline_html)

    out.append("<h3>모델별 처리 속도 (이번 실험)</h3>")
    if len(speeds):
        rows = [[e(nice_label(r.spec)) + f'<div class="sub">{e(r.experiments)}</div>', str(r.jobs), f"{r.wps:,.0f}",
                 f"×{r.wps / ref_wps:.1f}", minutes(r.minutes),
                 (f"{r.peak_mb:,.0f} MB" if r.peak_mb is not None and pd.notna(r.peak_mb) else "-"),
                 (f"{r.params:,.0f}" if r.params is not None and pd.notna(r.params) else "-")]
                for r in speeds.itertuples()]
        out.append(table(["모델 구성", "작업 수", "초당 학습 window (중앙값)", f"이전({ref_wps:,.0f}) 대비",
                          "작업 하나 학습 시간 (중앙값)", "최대 GPU 메모리", "파라미터 수 (중앙값)"], rows))
        out.append('<p class="sub">채널 독립 PatchTST(patchtst_ci)는 신호마다 따로 계산해서 batch × 신호 수만큼 계산량이 '
                   "늘어나 가장 느립니다. 메모리에 맞게 batch를 자동으로 줄입니다.</p>")
    else:
        out.append('<p class="empty">아직 끝난 딥러닝 작업이 없습니다.</p>')
    return "".join(out)


def _history(data: ReportData) -> str:
    if not data.histories:
        return '<p class="empty">이전 실험 기록 파일이 없습니다 (configs/history/).</p>'
    main = next((h for h in data.histories if h.get("runs")), data.histories[0])
    out = [('<p class="note">전체 진행 흐름과 주제별 이전·이번 비교입니다. '
            "시기별 상세 기록(설정 · 전체 테스트 목록 · 통계 검정 · 실행 명령)은 맨 아래에 접어 두었습니다.</p>"),
           "<h3>전체 흐름</h3>", _history_timeline(data)]
    journey = _vg11_journey(data, main)
    if journey:
        out += ["<h3>VG11 10·20·30초 예측 오차의 변화 (시험 run)</h3>", journey]
    out += ["<h3>주제별 비교 (이전 · 이번)</h3>", _topics(data, main), "<h3>시기별 상세 기록</h3>"]
    for i, h in enumerate(data.histories):
        out.append(f'<details class="stage"{" open" if i == 0 else ""}><summary>{e(h.get("title"))} '
                   f'<span class="sub">{e(h.get("period"))}</span></summary>{_history_block(h)}</details>')
    return "".join(out)


def _vg11_numbers(data: ReportData) -> tuple[Pick | None, float | None, Pick | None, float | None]:
    """This round's VG11 forecast picks with their test MAE: the 37-signal experiment and the all-signal one."""

    vg11, params = _experiment(data, "g1_forecast_vg11"), _experiment(data, "g1_forecast_params")
    pick, p_vg = _pick(vg11, "VG11"), _pick(params, "VG11")
    rows = vg11.test.to_dict(orient="records") if vg11 is not None and vg11.test is not None else []
    ranked = sorted((r for r in rows if not r.get("is_baseline")), key=lambda r: r["mae"])
    best = float(ranked[0]["mae"]) if ranked else None
    p_test = _test_lookup(params).get((p_vg.target, p_vg.label)) if p_vg else None
    return pick, best, p_vg, (float(p_test["mae"]) if p_test and p_test.get("mae") is not None else None)


def _history_timeline(data: ReportData) -> str:
    nodes = [dict(n, kind="prev") for h in data.histories for n in h.get("timeline") or []]
    nodes.sort(key=lambda n: str(n.get("when")))
    exps = [x for x in data.experiments if not x.smoke]
    if exps:
        pick, best, p_vg, p_best = _vg11_numbers(data)
        if best is not None:
            key = f"시험 {fmt(best)}" + (f" → {fmt(p_best)}" if p_best is not None else "") + " Torr"
        else:
            key = f"검증 {fmt(pick.mae)} Torr" if pick else "결과 정리 중"
        signals = f"신호 {len(pick.doc.get('inputs', []))}개" if pick else ""
        if p_vg is not None and p_best is not None:
            signals += f" → {len(p_vg.doc.get('inputs', []))}개"
        deep = pd.concat([x.results[x.results["kind"] == "deep"] for x in exps if "kind" in x.results] or [pd.DataFrame()])
        fit = (f"모델당 {minutes(deep['fit_seconds'].median() / 60)}"
               if len(deep) and "fit_seconds" in deep and deep["fit_seconds"].notna().any() else "")
        jobs = sum(int(x.summary.get("jobs") or 0) for x in exps)
        nodes.append({"when": data.created.strftime("%Y-%m"), "title": "이번 시스템", "key": key, "kind": "now",
                      "note": " · ".join(x for x in (signals, f"실험 {len(exps)}개 · 모델 {jobs}개", fit) if x)})
    if not nodes:
        return ""
    cells = []
    for i, n in enumerate(nodes):
        cells.append(f'<div class="tnode {e(n["kind"])}"><div class="tw2">{e(n.get("when"))}</div>'
                     f'<div class="tt">{e(n.get("title"))}</div><div class="tkey">{e(n.get("key"))}</div>'
                     f'<div class="tnote">{e(n.get("note"))}</div></div>')
        if i < len(nodes) - 1:
            cells.append('<div class="tarrow">→</div>')
    return '<div class="timeline">' + "".join(cells) + "</div>"


def _vg11_journey(data: ReportData, h: dict[str, Any]) -> str:
    rows = [(f"{r.get('name')} ({r.get('id')})", float(r["mae"]), "base" if r.get("step") == "기준선" else "prev")
            for r in h.get("runs") or [] if r.get("journey") and r.get("mae") is not None]
    rows.sort(key=lambda r: (r[2] != "base", -r[1]))  # baseline first, then the errors coming down
    pick, best, p_vg, p_best = _vg11_numbers(data)
    if best is not None and pick:
        rows.append((f"이번 {nice_label(pick.label)} · 신호 {len(pick.doc.get('inputs', []))}개", best, "now"))
    if p_best is not None and p_vg:
        rows.append((f"이번 {nice_label(p_vg.label)} · 신호 {len(p_vg.doc.get('inputs', []))}개", p_best, "now"))
    if not rows:
        return ""
    chart = value_bars(rows, unit="Torr", title="VG11 예측 오차의 변화")
    return (f'<div class="chartbox">{chart}</div>'
            '<div class="legend"><span><i style="background:var(--axis)"></i>기준선 (현재값 유지)</span>'
            '<span><i style="background:var(--s2)"></i>이전 (2026-08)</span>'
            '<span><i style="background:var(--s1)"></i>이번</span>'
            '<span class="sub">막대가 짧을수록 정확합니다. 모두 같은 시험 run 11개 · 10·20·30초 평균 MAE.</span></div>')


def _bullet_card(title: str, items: list[str], cls: str = "") -> str:
    return (f'<div class="card {cls}"><div class="ct">{e(title)}</div><ul>'
            + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul></div>")


def _history_block(h: dict[str, Any]) -> str:
    """One history file: headline numbers and pictures first, the full records folded away."""

    runs = h.get("runs") or []
    base = next((r for r in runs if r.get("step") == "기준선"), None)
    best = next((r for r in runs if r.get("best")), None)
    out = [f"<p>{e(h.get('summary'))}</p>"]
    if base and best:
        out.append('<div class="tiles">'
                   + tile("VG11 10·20·30초 예측 (시험 MAE)", f"{base['mae']:.3f} → {best['mae']:.3f}",
                          pct((1 - best["mae"] / base["mae"]) * 100) + f" · {e(best['id'])}")
                   + tile("시험한 설정", f"{sum(len(g.get('rows', [])) for g in h.get('tests', [])) or len(runs)}개",
                          f"GPU 약 {h.get('gpu_hours')}시간")
                   + tile("모델 하나 학습", "2~8시간", "초당 약 3,600 window") + "</div>")
    story = h.get("hparam_story") or []
    if story:
        out.append('<h4>단계별로 한 일</h4><div class="steps2">' + "".join(
            f'<div class="step2"><div class="sk">{e(a)}</div><div class="st2">{e(b)}</div>'
            f'<div class="sr">{e(d)}</div><div class="sc">{e(c)}</div></div>' for a, b, c, d in story) + "</div>")
    if runs and base:
        bars = improvement_bars([(f"{r['id']}", (1 - float(r["mae"]) / base["mae"]) * 100)
                                 for r in runs if r.get("step") != "기준선"], title="이전 실험 개선율")
        out.append(f'<div class="chartbox"><div class="ctitle">현재값 유지 대비 개선율 (시험 run)</div>{bars}</div>')
    if h.get("results"):
        out.append(f"<h4>{e(h.get('results_title', '결과'))}</h4>"
                   + table([e(c) for c in h.get("results_header", [])], [[e(c) for c in row] for row in h["results"]]))
    if h.get("problems"):
        out.append(f'<h4>코드 점검에서 찾은 문제 {len(h["problems"])}가지</h4><div class="pgrid">' + "".join(
            f'<div class="prob"><b>{e(a)}</b><span>{e(b)}</span></div>' for a, b in h["problems"]) + "</div>")
    cards = [_bullet_card(title, h[key], cls) for key, title, cls in (
        ("lessons", "배운 것", "good"), ("limits", "한계", "bad"), ("roadmap", "계획했지만 하지 못한 것", ""),
        ("caveats", "읽을 때 주의", "bad"), ("other", "그 밖의 작업", "")) if h.get(key)]
    if cards:
        out.append('<div class="cards">' + "".join(cards) + "</div>")
    out.append(_history_records(h, runs, base))
    return "".join(out)


def _history_records(h: dict[str, Any], runs: list[dict[str, Any]], base: dict[str, Any] | None) -> str:
    """The full records of one history file (setup, every test, statistics, commands), folded."""

    parts = []
    for key, title in (("setup", "설정 (In · Out · 모델 · 학습)"), ("inputs", "In (입력)"), ("outputs", "Out (출력)"),
                       ("defaults", "Function (모델) · Hyper Parameter")):
        if h.get(key):
            parts.append(f"<h4>{title}</h4>" + kv([(k, e(v)) for k, v in h[key]]))
            if key == "inputs" and h.get("input_sets"):
                parts.append(table(["입력 세트", "신호 수", "내용"],
                                   [[f"<code>{e(a)}</code>", str(b), e(c)] for a, b, c in h["input_sets"]]))
    if runs:
        rows = []
        for r in runs:
            by_h = r.get("by_h")
            rows.append([
                e(r.get("step")), f"<b>{e(r.get('id'))}</b>" + (' <span class="chip">최종</span>' if r.get("best") else "")
                + f"<div class='sub'>{e(r.get('name'))}</div>",
                e(r.get("inputs")), e(r.get("change")) + (f"<div class='sub'>{e(r['note'])}</div>" if r.get("note") else ""),
                fmt(r.get("mae")),
                "-" if r.get("step") == "기준선" or not base else pct((1 - float(r["mae"]) / base["mae"]) * 100),
                " / ".join(f"{v:.3f}" for v in by_h) if by_h else "-",
                f"{r['stable']:.3f} / {r['transition']:.3f}" if r.get("stable") is not None else "-",
                e(r.get("epochs")), f"{r['hours']:.1f}시간" if r.get("hours") else "-",
                f"{int(r['params']):,}" if r.get("params") else "-",
            ])
        parts.append("<h4>주요 모델 성능표 (시험 run 179,169 window)</h4>"
                     + table(["단계", "모델", "입력", "바꾼 점", "MAE", "현재값 유지 대비", "10 / 20 / 30초", "안정 / 전환",
                              "best/epoch", "학습 시간", "파라미터"], rows))
    if h.get("tests"):
        base_mae = base["mae"] if base else None
        rows = []
        for group in h["tests"]:
            for i, r in enumerate(group.get("rows", [])):
                mae = r.get("mae")
                rows.append([
                    e(group.get("phase")) if i == 0 else "",
                    f"<b>{e(r.get('id'))}</b>" + (' <span class="chip">최종</span>' if r.get("best") else ""),
                    e(r.get("change")), fmt(mae),
                    " / ".join(f"{v:.3f}" for v in r["by_h"]) if r.get("by_h") else "-",
                    pct((1 - float(mae) / base_mae) * 100) if mae is not None and base_mae and r.get("id") != "B_PERSISTENCE"
                    and "GR1" not in str(r.get("id")) else "-",
                    e(r.get("note")),
                ])
        parts.append("<h4>전체 테스트 목록</h4>"
                     '<p class="note">시험 run MAE(10·20·30초 평균, Torr). 개선율은 모두 현재값 유지(2.092) 대비이고, '
                     "결론 열은 각 단계의 기준(직전 단계·A02·FA1)과 비교한 것입니다. GR1은 급변 구간만 따로 잰 값이라 "
                     "개선율을 붙이지 않았습니다.</p>"
                     + table(["단계", "테스트", "바꾼 점", "MAE", "10 / 20 / 30초", "현재값 유지 대비", "결론"], rows))
    if h.get("ablations"):
        parts.append("<h4>원인 분석·추가 실험 요약</h4>" + kv([(k, e(v)) for k, v in h["ablations"]]))
    if h.get("stats"):
        parts.append("<h4>통계 검정</h4>"
                     '<p class="note">run(파일) 11개별 MAE로 한 Wilcoxon 부호순위 검정, seed 42. p=0.001은 run 11개로 '
                     "나올 수 있는 가장 작은 값.</p>"
                     + table(["비교", "MAE 차이", "p", "판단"], [[e(c) for c in row] for row in h["stats"]]))
    if h.get("slices"):
        parts.append("<h4>레시피별 오차 (O_UNIVARIATE)</h4>"
                     + table(["레시피", "시험 window", "MAE 10 / 20 / 30초"], [[e(c) for c in row] for row in h["slices"]]))
    if h.get("data_checks"):
        parts.append("<h4>데이터 점검</h4>" + kv([(k, e(v)) for k, v in h["data_checks"]]))
    w = h.get("weights") or {}
    if w:
        parts.append("<h4>가중치 파일</h4>" + kv([("형식", e(w.get("format"))), ("위치", f"<code>{e(w.get('path'))}</code>"),
                                              ("참고", e(w.get("note")))]))
    if h.get("commands"):
        parts.append("<h4>실행 명령</h4>" + table(["단계", "명령"], [[e(a), f"<code>{e(b)}</code>"] for a, b in h["commands"]]))
    if not parts:
        return ""
    names = [n for k, n in (("setup", "설정"), ("inputs", "설정"), ("tests", "전체 테스트 목록"), ("stats", "통계 검정"),
                            ("data_checks", "데이터 점검"), ("commands", "실행 명령")) if h.get(k)]
    return (f"<details><summary>상세 기록: {e(' · '.join(dict.fromkeys(names)))}</summary>"
            + "".join(parts) + "</details>")


def _experiment(data: ReportData, name: str) -> Experiment | None:
    return next((x for x in data.experiments if x.name == name), None)


def _pick(exp: Experiment | None, target: str) -> Pick | None:
    return next((p for p in exp.picks if p.target == target), None) if exp else None


def _label_mae(exp: Experiment | None, target: str, prefix: str) -> tuple[str, float] | None:
    """Best (validation) row of ``target`` whose label starts with ``prefix``."""

    if exp is None:
        return None
    rows = exp.table[(exp.table["target"] == target) & exp.table["label"].astype(str).str.startswith(prefix)]
    if not len(rows):
        return None
    row = rows.sort_values("mae").iloc[0]
    return str(row["label"]), float(row["mae"])


def _test_by_horizon(pick: Pick | None) -> dict[str, float] | None:
    path = pick.job_dir / "metrics_test.json" if pick else None
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("by_horizon")


def _topics(data: ReportData, h: dict[str, Any]) -> str:
    """Previous tests next to this system's results, topic by topic (numbers from work/ where they exist)."""

    vg11 = _experiment(data, "g1_forecast_vg11")
    params = _experiment(data, "g1_forecast_params")
    vsensor = _experiment(data, "g1_virtual_sensor")
    pick = _pick(vg11, "VG11")
    test = _test_lookup(vg11).get((pick.target, pick.label)) if pick else None
    test_rows = vg11.test.to_dict(orient="records") if vg11 is not None and vg11.test is not None else []

    def test_text(row: dict[str, Any]) -> str:
        std = row.get("mae_std")
        return f"{e(nice_label(row['label']))} {fmt(row['mae'])}" + (
            f" ± {spread(std)}" if std is not None and not pd.isna(std) else "")

    if test:
        ranked = sorted((r for r in test_rows if not r.get("is_baseline")), key=lambda r: r["mae"])
        now_main = " · ".join(test_text(r) for r in ranked[:2]) + " (시험 run)" if ranked else f"시험 run {fmt(test['mae'])}"
        best_row = ranked[0] if ranked else None
        verdict = (f"같은 시험 run에서 이전 최고 0.188 → {fmt(best_row['mae'])} "
                   f"({(1 - best_row['mae'] / 0.1876) * 100:.0f} % 더 정확)" if best_row else "-")
    elif pick:
        now_main, verdict = f"검증 run {fmt(pick.mae)} ({e(nice_label(pick.label))}), 시험 run 평가 전", "시험 run 평가 후 비교"
    else:
        now_main, verdict = "아직 결과 없음", "-"

    h10 = _test_by_horizon(pick)
    now_10 = (f"{e(nice_label(pick.label))} 10초 {fmt(h10.get('10s'))} (시험 run)" if h10 and pick
              else (f"{e(nice_label(pick.label))} 10초 {fmt(pick.validation.get('metrics', {}).get('by_horizon', {}).get('10s'))} (검증)"
                    if pick else "-"))

    ci = _label_mae(vg11, "VG11", "patchtst_ci")
    mix = _label_mae(vg11, "VG11", "patchtst_mix")
    now_mix = (f"독립 {fmt(ci[1])} vs 혼합 {fmt(mix[1])} (검증, PatchTST)" if ci and mix else "-")

    p_vg = _pick(params, "VG11")
    now_inputs = "-"
    verdict_inputs = "트리 모델은 신호가 많을수록 좋아짐 (같은 예측 거리로 재확인 예정)"
    if pick and p_vg:
        seconds = ["·".join(f"{h:g}" for h in _horizons(x)) for x in (vg11, params)]
        same = seconds[0] == seconds[1]
        reach = f"{seconds[0]}초" if same else f"{seconds[0]}초 vs {seconds[1]}초"
        p_test = _test_lookup(params).get((p_vg.target, p_vg.label))
        both_test = same and test and p_test and test.get("mae") is not None and p_test.get("mae") is not None
        before, after = (test["mae"], p_test["mae"]) if both_test else (pick.mae, p_vg.mae)
        now_inputs = (f"{e(nice_label(pick.label))} 신호 {len(pick.doc.get('inputs', []))}개 {fmt(before)} → "
                      f"{e(nice_label(p_vg.label))} {len(p_vg.doc.get('inputs', []))}개 {fmt(after)} "
                      f"({'시험 run' if both_test else '검증'}, 예측 거리 {reach})")
        if same and pick.model == p_vg.model:
            verdict_inputs = (f"같은 모델·같은 예측 거리에서 신호를 늘리니 오차 {(1 - after / before) * 100:.0f} % 감소"
                              if after < before else "같은 조건에서 신호를 늘려도 나아지지 않음")
        elif same:
            verdict_inputs = "고른 모델이 달라 신호 수만의 효과로 보기는 어려움"

    vs = _pick(vsensor, "VG11")
    now_vs = (f"압력계 9개 모두 없이 지금 값 추정: {e(nice_label(vs.label))} {fmt(vs.mae)} Torr "
              f"(레시피 평균 {fmt(vs.baseline_mae)} 대비 {vs.improvement_pct:+.0f} %). 대상 {len(vsensor.picks)}개로 확대"
              if vs and vs.improvement_pct is not None else "-")

    stds = [r for r in test_rows if not r.get("is_baseline") and r.get("mae_std") is not None and not pd.isna(r.get("mae_std"))]
    now_seed = " · ".join(f"{e(nice_label(r['label']))} ± {spread(r['mae_std'])}" for r in stds[:3]) or "-"

    deep = pd.concat([x.results[x.results["kind"] == "deep"] for x in data.experiments
                      if "kind" in x.results and not x.smoke] or [pd.DataFrame()])
    speed = fit = "-"
    if len(deep) and "windows_per_second" in deep and deep["windows_per_second"].notna().any():
        speed = f"초당 약 {deep['windows_per_second'].median():,.0f} window (딥러닝 중앙값)"
        fit = f"딥러닝 {minutes(deep['fit_seconds'].median() / 60)} (중앙값), 최대 {minutes(deep['fit_seconds'].max() / 60)}"
    n_targets = sum(len(x.picks) for x in data.experiments if not x.smoke)

    rows = [
        ["VG11 10·20·30초 예측", "PatchTST FA1 0.188 (seed 3개 0.199 ± 0.014), 현재값 유지 2.092 (시험 run)", now_main, verdict],
        ["10초 예측만 비교", "처음 Keras 코드 3.2~4.1 (run 3개, 학습 데이터와 겹침) · 처음 구조 재현 1.873 · 현재값 유지 1.107",
         now_10, "처음 코드는 현재값 유지보다도 나빴음"],
        ["신호를 섞을지 따로 볼지", "독립 1.117 → 혼합 0.773 → 혼합 + VG11만 손실 0.188", now_mix, "같은 결론 재확인: 신호 혼합이 우수"],
        ["입력 신호 수", "VG11만 0.193 · 37개 0.188 · 복사본 제외 35개 0.200 → 차이 작음 (PatchTST)", now_inputs,
         verdict_inputs],
        ["정규화 (RevIN)", "RevIN만 꺼도 0.191 → 2.032 (가장 큰 원인)", "딥러닝은 RevIN 유지, 고전 모델은 '현재값 대비 변화'를 학습",
         "같은 원리: 값의 수준이 아니라 변화를 예측"],
        ["처음 코드의 설정 (손실·크기)", "MAE·가중 MAE·모델 축소를 누적하면 0.214 → 8.54 (A01~A09)",
         "예측은 MSE, 가상 센서만 MAE + 보정량 스케일", "손실은 과제에 맞춰 고름 (가상 센서는 MSE가 승온 구간에 끌림)"],
        ["샘플링 간격", "0.5초 원본 0.252 vs 1초 0.188", "1초 평균 (끊김 구간 안에서만)", "1초 유지"],
        ["Step 정보 · 음수 방지", "Step ID 임베딩 0.212, Softplus 0.214, 개선 없음",
         "Step 정보는 가상 센서의 레시피 평균(출발점)으로 사용", "입력보다 기준선으로 쓸 때 효과적"],
        ["가상 센서 (VG11 없이 압력)", "VB1 16.8 · VB2 12.0 · Ridge 44.8, 현재값 유지(2.092)보다 나쁨 (10~30초 뒤 예측)", now_vs,
         "과제가 달라 숫자 직접 비교는 안 됨. 레시피 평균에서 출발해 실패를 해결"],
        ["seed 변동", "FA1 ± 0.0145 (seed 3개)", now_seed, "차이가 이보다 작으면 같은 수준으로 봄"],
        ["통계 검정", "run 11개 Wilcoxon (FA1 vs 현재값 유지 p=0.001 등)", "검증 run 순위 + 시험 run 확인 (검정은 아직 미적용)",
         "다음 보고서에 run별 검정 추가 예정"],
        ["계획만 하고 못 한 것", "트리 기반 기준선, 하이퍼파라미터 탐색, VG11 외 대상",
         f"LightGBM·Ridge 포함 10종 비교, 설정 탐색, 대상 {n_targets}개", "모두 수행"],
        ["모델 하나 학습 시간", "2~8시간 (GPU 1장)", e(fit), "실험 수를 크게 늘릴 수 있게 됨"],
        ["처리 속도", "초당 약 3,600 window, GPU 메모리 150 MB", e(speed), ""],
        ["결과 정리", "단계별 보고서(md), best.pt", "대시보드 · 이 보고서 · 모델 패키지(가중치 + 실행 파일)", ""],
    ]
    for row in rows:  # no result in this run → no interpretation either
        if row[2] in ("-", "아직 결과 없음"):
            row[3] = '<span class="sub">이번 실험에 해당 결과 없음</span>'
    return '<div class="tcards">' + "".join(
        f'<div class="tcard"><div class="tt">{e(topic)}</div>'
        f'<div class="trow"><span class="tlab prev">이전</span><div>{prev}</div></div>'
        f'<div class="trow"><span class="tlab now">이번</span><div>{now}</div></div>'
        + (f'<div class="tverd">{verdict}</div>' if verdict else "") + "</div>"
        for topic, prev, now, verdict in rows) + "</div>"


PRIORITY_NAMES = {1: "1 먼저", 2: "2 다음", 3: "3 여유 있을 때"}
PLAN_STATUS = {"완료": "good", "진행 중": "warn", "보류": "bad"}


def _cat_color(i: int) -> str:
    return f"var(--s{i % 8 + 1})"


def _plan_status(status: str) -> str:
    """Only a status other than the default 계획 is worth a label on a card."""

    return "" if status == "계획" else f' <span class="st {PLAN_STATUS.get(status, "")}">{e(status)}</span>'


def _plan_variants(item: dict[str, Any]) -> str:
    if not item["variants"]:
        return ""
    title = item["variants_title"] or "비교할 설정"
    if title == "탐색 공간":
        body = table(["설정", "범위", ""], [[f"<code>{e(a)}</code>", e(b), f'<span class="sub">{e(c)}</span>']
                                         for a, b, c in item["variants"]], cls="space")
    else:
        body = '<div class="vchips">' + "".join(
            f'<div class="vchip"><b>{e(a)}</b><span>{e(b)}</span>' + (f"<i>{e(c)}</i>" if c else "") + "</div>"
            for a, b, c in item["variants"]) + "</div>"
    return f'<div class="vt">{e(title)}</div>{body}'


def _plan_item(item: dict[str, Any], color: str) -> str:
    meta = "".join(f'<div><span class="ml">{label}</span>{e(item[key])}</div>'
                   for key, label in (("success", "채택 기준"), ("targets", "대상"), ("cost", "비용")) if item[key])
    more = [f"<p><b>묻는 것</b> {e(item['question'])}</p>"]
    if item["evidence"]:
        more.append("<b>근거 (지금 결과)</b><ul>" + "".join(f"<li>{e(x)}</li>" for x in item["evidence"]) + "</ul>")
    if item["method"]:
        more.append('<b>방법</b><ol class="steps">' + "".join(f"<li>{e(x)}</li>" for x in item["method"]) + "</ol>")
    for key, label in (("metric", "판단 지표"), ("prerequisite", "먼저 할 일"), ("risk", "주의")):
        if item[key]:
            more.append(f"<p><b>{label}</b> {e(item[key])}</p>")
    if item["result"]:
        more.append(f"<p><b>결과</b> {e(item['result'])}</p>")
    return (f'<div class="icard" id="plan-{e(item["id"])}" style="--c:{color}">'
            f'<div class="ih"><span class="idb">{e(item["id"])}</span><b>{e(item["title"])}</b>'
            f'<span class="prio p{item["priority"]}">{e(PRIORITY_NAMES[item["priority"]])}</span>'
            f'{_plan_status(item["status"])}</div>'
            f'<div class="ig">{e(item["goal"] or item["question"])}</div>'
            f'<div class="pmeta">{meta}</div>{_plan_variants(item)}'
            f'<details class="more"><summary>근거 · 방법 · 주의</summary>{"".join(more)}</details></div>')


def _plan(data: ReportData) -> str:
    plan = data.plan
    if not plan or not plan["categories"]:
        return '<p class="empty">다음 실험 계획이 없습니다 (configs/plan/next_experiments.yaml).</p>'
    colors = {cat["key"]: _cat_color(i) for i, cat in enumerate(plan["categories"])}
    items = [(cat, item) for cat in plan["categories"] for item in cat["items"]]
    active = sum(1 for _, i in items if i["status"] in ("진행 중", "완료"))
    out = [(f'<p class="note">지금까지의 결과를 근거로 고른 다음 실험 후보입니다. 근거 숫자는 {e(plan["basis"])} 기준이고, '
            "원본은 <code>configs/plan/next_experiments.yaml</code>입니다. 카드를 누르면 상세 내용으로 이동합니다.</p>"),
           '<div class="tiles">' + tile("후보", f"{len(items)}개", f"카테고리 {len(plan['categories'])}개")
           + tile("먼저 할 것", f"{sum(1 for _, i in items if i['priority'] == 1)}개", "우선순위 1")
           + (tile("진행 중 · 완료", f"{active}개", "상태는 YAML에서 고침") if active else "") + "</div>"]
    if plan["principles"]:
        out.append('<h3>진행 원칙</h3><div class="rules">' + "".join(
            f'<div class="rule"><span>{n}</span>{e(p)}</div>' for n, p in enumerate(plan["principles"], 1)) + "</div>")
    out.append("<h3>로드맵 (우선순위별)</h3>")
    out.append('<div class="legend">' + "".join(
        f'<span><i style="background:{colors[c["key"]]}"></i>{e(c["items"][0]["id"][0] if c["items"] else "")} · '
        f'{e(c["title"])}</span>' for c in plan["categories"]) + "</div>")
    columns = []
    for prio, name in PRIORITY_NAMES.items():
        chosen = [(c, i) for c, i in items if i["priority"] == prio]
        cards = "".join(
            f'<a class="pcard" href="#plan-{e(i["id"])}" style="--c:{colors[c["key"]]}"><span class="idb">{e(i["id"])}</span>'
            f'<b>{e(i["title"])}</b><span class="pg">{e(i["goal"])}</span><span class="pc">{e(i["cost"])}</span></a>'
            for c, i in chosen)
        columns.append(f'<div class="bcol"><div class="bh">{e(name)} <span class="sub">{len(chosen)}개</span></div>{cards}</div>')
    out.append('<div class="board">' + "".join(columns) + "</div>")
    out.append("<h3>카테고리별 상세</h3>")
    for cat in plan["categories"]:
        ids = [i["id"] for i in cat["items"]]
        span = f"{ids[0]}–{ids[-1]}" if len(ids) > 1 else (ids[0] if ids else "")
        out.append(f'<div class="catsec" id="plan-cat-{e(cat["key"])}" style="--c:{colors[cat["key"]]}">'
                   f'<div class="cath"><i></i><b>{e(cat["title"])}</b> <span class="sub">{e(span)}</span></div>'
                   + (f'<p class="note">{e(cat["why"])}</p>' if cat["why"] else "")
                   + '<div class="icards">' + "".join(_plan_item(i, colors[cat["key"]]) for i in cat["items"])
                   + "</div></div>")
    return "".join(out)


def _css() -> str:
    def block(t: Any) -> str:
        return (f"--page:{t.page};--surface:{t.surface};--text:{t.text_primary};--text2:{t.text_secondary};"
                f"--muted:{t.muted};--grid:{t.grid};--axis:{t.axis};"
                + "".join(f"--s{i + 1}:{c};" for i, c in enumerate(t.series)))

    status = "".join(f"--{k}:{v};" for k, v in STATUS.items())
    groups = "".join(f"svg .g-{g} .ns{{fill:{c}}} .sw-{g}{{background:{c}}}" for g, c in GROUP_COLORS.items())
    return f"""
:root{{{block(LIGHT)}{status}--shade:rgba(42,120,214,.08);--line:#e8e7e1;
  --side:#1d2330;--side-text:#c9cfda;--side-line:rgba(255,255,255,.08);--side-hover:rgba(255,255,255,.06);
  --side-on:rgba(57,135,229,.28)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--side:#13171e}}}}
:root[data-theme="dark"]{{--side:#13171e}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{{block(DARK)}--shade:rgba(57,135,229,.14);--line:#2c2c2a}}}}
:root[data-theme="dark"]{{{block(DARK)}--shade:rgba(57,135,229,.14);--line:#2c2c2a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--page);color:var(--text);font:14px/1.6 {FONT};}}
header{{padding:20px 24px 8px;max-width:1280px;margin:0 auto}}
header h1{{margin:0;font-size:22px}} header .meta{{color:var(--muted);font-size:13px}}
.side{{position:fixed;left:0;top:0;bottom:0;width:236px;z-index:30;display:flex;flex-direction:column;overflow-y:auto;
  background:var(--side);color:var(--side-text);border-right:1px solid var(--side-line)}}
.side .brand{{padding:20px 20px 14px;border-bottom:1px solid var(--side-line)}}
.side .brand b{{display:block;color:#fff;font-size:17px;letter-spacing:.2px}} .side .brand span{{font-size:12px;opacity:.75}}
.side .navlabel{{padding:16px 20px 6px;font-size:11px;font-weight:700;letter-spacing:.14em;opacity:.55}}
.side button{{display:flex;align-items:center;gap:12px;width:calc(100% - 16px);margin:2px 8px;padding:9px 12px;border:0;
  border-radius:8px;background:none;color:inherit;font:inherit;font-size:14px;text-align:left;cursor:pointer}}
.side button:hover{{background:var(--side-hover)}}
.side svg{{width:20px;height:20px;flex:none;fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}}
.side .chev{{width:16px;height:16px;margin-left:auto;transition:transform .15s;opacity:.7}}
.side .grp.open .chev{{transform:rotate(180deg)}}
.side .sub{{display:none;margin:2px 0 6px 30px;border-left:1px solid var(--side-line)}}
.side .grp.open .sub{{display:block}}
.side .sub button{{width:calc(100% - 14px);margin:1px 0 1px 6px;padding:7px 12px;font-size:13.5px}}
.side button.on{{background:var(--side-on);color:#fff;font-weight:600}}
.side .item.on,.side .grp.has-on>.grph{{color:#fff}} .side .grp.has-on>.grph{{background:var(--side-hover)}}
.side .sidefoot{{margin-top:auto;padding:14px 16px 18px;border-top:1px solid var(--side-line);font-size:12px}}
.side .sidefoot .themebtn{{float:none;width:100%;margin:0 0 10px;justify-content:center;border:1px solid var(--side-line);
  background:none;color:inherit}}
.side .sidefoot .meta{{opacity:.7;line-height:1.5;word-break:break-all}}
.content{{margin-left:236px;min-width:0}}
.navbtn{{display:none;border:1px solid var(--line);background:var(--surface);color:var(--text2);border-radius:8px;
  padding:4px 8px;cursor:pointer;vertical-align:middle;margin-right:8px}}
.navbtn svg{{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;display:block}}
.scrim{{display:none}}
main{{max-width:1280px;margin:0 auto;padding:8px 24px 48px}}
section{{display:none}} section.on{{display:block}}
section>h2{{font-size:20px;margin:18px 0 12px;padding-left:12px;border-left:4px solid var(--s1);line-height:1.3}}
@media (max-width:900px){{.side{{transform:translateX(-100%);transition:transform .2s;box-shadow:0 0 24px rgba(0,0,0,.35)}}
  body.navopen .side{{transform:none}} .content{{margin-left:0}} .navbtn{{display:inline-block}}
  body.navopen .scrim{{display:block;position:fixed;inset:0;z-index:25;background:rgba(0,0,0,.35)}}}}
h3{{font-size:16px;margin:24px 0 8px}} h4{{font-size:14px;margin:16px 0 6px}}
.tw{{overflow-x:auto;margin:6px 0 10px}}
table{{border-collapse:collapse;width:100%;background:var(--surface);font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--text2);font-weight:600;background:var(--page)}}
table.kv td:first-child{{width:220px;color:var(--text2)}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:12px;background:var(--page);padding:1px 4px;border-radius:4px;
  word-break:break-all}}
.sub{{color:var(--muted);font-size:12px}} .note{{color:var(--text2);font-size:13px}}
.empty{{color:var(--muted)}}
.st{{font-weight:600;white-space:nowrap}} .st.good{{color:var(--good)}} .st.bad{{color:var(--critical)}}
.st.warn{{color:var(--serious)}}
.chip{{display:inline-block;font-size:11px;padding:0 6px;border-radius:999px;border:1px solid var(--line);color:var(--text2)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:10px 0}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}}
.tl{{color:var(--text2);font-size:12px}} .tv{{font-size:22px;font-weight:700}} .tn{{color:var(--muted);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}}
.card p{{margin:4px 0}} .ct{{font-weight:700}}
details{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px 12px;margin:8px 0}}
.concl{{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--s1);border-radius:10px;padding:8px 16px 8px 4px}} .concl li{{margin:5px 0}}
svg .ov-pt{{fill:var(--s1)}} svg .ov-lgb{{fill:var(--s2)}} svg .ov-ptl{{fill:none;stroke:var(--s1);stroke-width:2.2}}
svg .ov-lgbl{{stroke:var(--s2);stroke-width:1.6;stroke-dasharray:5 4}} svg .ov-ref{{stroke:var(--text2);stroke-width:1.2}}
svg .ov-gh{{fill:var(--text);font-weight:700;font-size:13px}} svg .ov-val{{fill:var(--text);font-weight:600}}
svg .ov-good{{fill:var(--good);font-weight:600}} svg .ov-bad{{fill:var(--critical);font-weight:600}} svg .ov-muted{{fill:var(--muted)}}
summary{{cursor:pointer;font-weight:600;padding:4px 0}}
.sw{{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px}} .sw ul{{margin:4px 0;padding-left:18px}}
.sw.small{{font-size:12px}}
.chartbox{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px;margin:8px 0}}
.ctitle{{font-size:12px;color:var(--text2);margin:0 0 4px 4px}}
svg.chart{{width:100%;height:auto;display:block}}
svg .grid{{stroke:var(--grid);stroke-width:1}} svg .axis{{stroke:var(--axis);stroke-width:1}}
svg text{{font-family:{FONT};font-size:12px;fill:var(--text2)}} svg .tick{{fill:var(--muted);font-size:11px}}
svg .value{{fill:var(--text)}}
svg .bar-good{{fill:var(--s1)}} svg .bar-bad{{fill:var(--critical)}} svg .bar-muted{{fill:var(--axis)}}
svg .mark{{stroke:var(--s2);stroke-width:1.2;stroke-dasharray:4 3}} svg .marklabel{{fill:var(--s2);font-size:11px}}
svg .shade{{fill:var(--shade)}}
svg .ln-true{{fill:none;stroke:var(--text2);stroke-width:1.2}} svg .ln-pred{{fill:none;stroke:var(--s1);stroke-width:1.4}}
.flow{{display:flex;flex-direction:column;align-items:stretch;gap:0}}
.lane{{border:1px solid var(--line);border-radius:12px;padding:10px;background:var(--surface);border-left:6px solid var(--s1)}}
.lane.prep{{border-left-color:var(--s2)}} .lane.train{{border-left-color:var(--s3)}} .lane.eval{{border-left-color:var(--s4)}}
.lane.use{{border-left-color:var(--s7)}}
.lane-h{{font-weight:700;margin-bottom:6px}}
.fbox{{display:grid;grid-template-columns:150px 1.1fr 1fr 1.6fr;gap:10px;align-items:start;background:var(--page);
  border:1px solid var(--line);border-radius:8px;padding:8px}}
.fn{{font-weight:700}} .fcode{{color:var(--text2);font-size:12px}} .fr{{font-size:13px}}
.arrow{{text-align:center;color:var(--muted);line-height:1.2;font-size:12px}} .lane-arrow{{font-size:16px;padding:2px 0}}
ol.steps li{{margin:2px 0}}
.diagram{{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px;margin:8px 0}}
.diagram svg{{display:block;max-width:none}}
svg .nb{{fill:var(--page);stroke:var(--axis);stroke-width:1}} svg .node:hover .nb{{stroke:var(--text2)}}
svg .nt{{fill:var(--text);font-size:12px;font-weight:600}} svg .nm{{fill:var(--muted);font-size:10.5px}}
svg .root .nb{{fill:var(--text);stroke:var(--text)}} svg .root .nt,svg .root .nm{{fill:var(--page)}}
svg .eg{{fill:none;stroke:var(--muted);stroke-width:1.2}} svg .ah{{fill:var(--muted)}}
svg .ex{{fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3;opacity:.8}}
svg .el{{font-size:11px;fill:var(--text2);paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}}
svg .nx{{fill:var(--text2);font-size:12px;font-weight:700}} svg .root .nx{{fill:var(--page)}}
svg .ct{{fill:var(--text2);font-size:12px;font-weight:600}}
{groups}
.legend{{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--text2);margin:6px 0}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
.legend b.ln{{display:inline-block;width:22px;border-top:1.5px solid var(--muted);margin-right:5px;vertical-align:3px}}
.legend b.ln.dash{{border-top-style:dashed}}
table.matrix td:nth-child(n+3),table.matrix th:nth-child(n+3){{text-align:center;white-space:nowrap}}
.dot{{color:var(--text2)}}
svg .g-prev .ns{{fill:var(--muted)}} svg .g-now .ns{{fill:var(--s1)}}
svg .fa{{fill:var(--surface)}} svg .fb{{fill:var(--grid);opacity:.55}} svg .frame{{fill:none;stroke:var(--axis)}}
svg .win{{fill:var(--s1);opacity:.55}} svg .patch{{fill:var(--s3)}} svg .cube{{fill:var(--surface);stroke:var(--s1)}}
svg .cell{{fill:var(--surface);stroke:var(--axis)}} svg .tl{{stroke:var(--surface);stroke-width:.6}}
pre.code{{background:var(--page);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto;
  font:12px/1.5 ui-monospace,Consolas,monospace;margin:6px 0}}
table.effects td:nth-child(-n+3),table.effects th{{white-space:nowrap}}
table.where td:last-child code{{word-break:normal;white-space:nowrap}} .sw>div{{min-width:0}}
svg .cv{{fill:var(--text);font-size:15px;font-weight:700}} svg .nb-final{{stroke:var(--s1);stroke-width:2}}
svg .dl{{fill:var(--text2);font-size:11.5px;font-weight:600}}
svg .cell-train{{fill:var(--s1);fill-opacity:.22;stroke:var(--s1)}}
svg .cell-validation{{fill:var(--s2);fill-opacity:.22;stroke:var(--s2)}}
svg .cell-test{{fill:var(--s3);fill-opacity:.22;stroke:var(--s3)}}
svg .cell-excluded{{fill:var(--critical);fill-opacity:.22;stroke:var(--critical)}}
svg .cell-missing{{fill:none;stroke:var(--axis);stroke-dasharray:3 3}}
svg .cnum{{fill:var(--text);font-size:11px}} svg .cnum-muted{{fill:var(--muted);font-size:11px}} svg .cflag{{fill:var(--serious)}}
.legend i.sw-train,.legend i.sw-validation,.legend i.sw-test,.legend i.sw-excluded{{border:1px solid}}
.legend i.sw-train{{background:color-mix(in srgb,var(--s1) 22%,transparent);border-color:var(--s1)}}
.legend i.sw-validation{{background:color-mix(in srgb,var(--s2) 22%,transparent);border-color:var(--s2)}}
.legend i.sw-test{{background:color-mix(in srgb,var(--s3) 22%,transparent);border-color:var(--s3)}}
.legend i.sw-excluded{{background:color-mix(in srgb,var(--critical) 22%,transparent);border-color:var(--critical)}}
.legend i.sw-missing{{border:1px dashed var(--axis)}}
.legend i.sw-flag{{background:var(--serious);clip-path:polygon(0 0,100% 0,100% 100%)}}
svg .seg{{fill:var(--grid);stroke:var(--axis)}} svg .seg-gap{{fill:none;stroke:var(--critical);stroke-dasharray:3 3}}
svg .win-ok{{fill:var(--s1);opacity:.6}} svg .win-bad{{fill:var(--critical);opacity:.45}}
svg .tgt-ok{{fill:var(--s1)}} svg .tgt-bad{{fill:var(--critical)}}
svg .st-ok{{fill:var(--good);font-weight:600}} svg .st-bad{{fill:var(--critical);font-weight:600}}
svg .band-process{{fill:var(--shade)}} svg .band-cycle{{fill:var(--s1);fill-opacity:.16}} svg .dot-raw{{fill:var(--muted)}}
.chips{{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 10px}}
table.prep td:first-child{{white-space:nowrap}}
details.stage{{padding:8px 14px}} details.stage>summary{{font-size:15px}}
details.stage>summary .note{{font-weight:400;margin:2px 0 4px}} details.stage[open]>summary{{border-bottom:1px solid var(--line);margin-bottom:6px}}
details.code{{background:var(--page);padding:2px 10px;margin:6px 0}} details.code>summary{{font-weight:500}}
.facts{{background:var(--shade);border-radius:8px;padding:6px 10px;font-size:13px;margin:8px 0}}
ul.review li,table.kv ul li{{margin:3px 0}} table.kv ul{{margin:0;padding-left:18px}}
table.design td:first-child{{min-width:150px}} table.plan td:nth-child(-n+2),table.plan td:nth-child(4),table.plan td:nth-child(5){{white-space:nowrap}}
html{{scroll-padding-top:56px}}
ul.tight{{margin:0;padding-left:18px}} ul.tight li{{margin:2px 0}}
svg .bar-prev{{fill:var(--s2)}}
.rules{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:8px;margin:6px 0 10px}}
.rule{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:13px;display:flex;gap:8px}}
.rule span{{flex:none;width:22px;height:22px;border-radius:50%;background:var(--text);color:var(--page);display:grid;
  place-items:center;font-weight:700;font-size:12px}}
.board{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:8px 0}}
.bcol{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px}}
.bh{{font-weight:700;margin:2px 4px 6px}}
.pcard{{display:block;text-decoration:none;color:var(--text);background:var(--page);border:1px solid var(--line);
  border-left:5px solid var(--c);border-radius:8px;padding:7px 9px;margin:6px 0}}
.pcard:hover{{border-color:var(--c)}} .pcard b{{font-size:13px}}
.pg{{display:block;color:var(--text2);font-size:12px;margin-top:2px}} .pc{{display:block;color:var(--muted);font-size:11px;margin-top:2px}}
.idb{{display:inline-block;min-width:26px;text-align:center;font-weight:700;font-size:12px;padding:0 5px;margin-right:6px;
  border-radius:6px;background:var(--c);color:var(--page)}}
.catsec{{border-left:5px solid var(--c);padding-left:12px;margin:18px 0}}
.cath{{font-size:16px;margin-bottom:2px}} .cath i{{display:inline-block;width:12px;height:12px;border-radius:3px;
  background:var(--c);margin-right:6px}}
.icards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:10px}}
.icard{{background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--c);border-radius:10px;padding:10px 12px}}
.ih{{display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px}} .ih b{{font-size:14px}}
.prio{{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:0 7px;color:var(--text2)}}
.prio.p1{{border-color:var(--text);color:var(--text);font-weight:700}}
.ig{{font-size:14px;font-weight:600;margin:6px 0}}
.pmeta{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:12px;margin:6px 0}}
.pmeta>div{{background:var(--page);border-radius:6px;padding:5px 7px}} .ml{{display:block;color:var(--muted);font-size:11px}}
.vt{{font-size:12px;color:var(--text2);font-weight:600;margin:8px 0 4px}}
.vchips{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}}
.vchip{{background:var(--page);border:1px solid var(--line);border-radius:8px;padding:5px 7px;font-size:12px}}
.vchip b,.vchip span{{display:block}} .vchip i{{display:block;color:var(--muted);font-style:normal;font-size:11px}}
table.space td{{padding:3px 6px;font-size:12px}} table.space td:first-child{{white-space:nowrap}}
details.more{{background:var(--page);margin:8px 0 0;padding:2px 10px;font-size:13px}}
details.more>summary{{font-weight:500;color:var(--text2)}}
details.more ul,details.more ol{{margin:2px 0 6px;padding-left:18px}} details.more p{{margin:4px 0}}
.timeline{{display:flex;align-items:stretch;gap:6px;overflow-x:auto;padding:4px 0 8px}}
.tnode{{flex:1 1 0;min-width:170px;background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--s2);
  border-radius:10px;padding:8px 10px}}
.tnode.now{{border-top-color:var(--s1);background:var(--shade)}}
.tw2{{color:var(--muted);font-size:12px}} .tnode .tt{{font-weight:700}} .tkey{{font-size:18px;font-weight:700;margin:4px 0}}
.tnote{{color:var(--text2);font-size:12px}} .tarrow{{align-self:center;color:var(--muted);font-size:18px}}
.tcards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:10px}}
.tcard{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13px}}
.tcard .tt{{font-weight:700;margin-bottom:6px}}
.trow{{display:grid;grid-template-columns:38px 1fr;gap:6px;margin:4px 0}}
.tlab{{font-size:11px;font-weight:700;border-radius:6px;padding:1px 0;text-align:center;align-self:start;color:var(--page)}}
.tlab.prev{{background:var(--s2)}} .tlab.now{{background:var(--s1)}}
.tverd{{margin-top:6px;padding-top:6px;border-top:1px dashed var(--line);font-weight:600}}
.steps2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:6px 0}}
.step2{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px}}
.sk{{font-weight:800;font-size:16px;color:var(--s2)}} .st2{{font-weight:700;font-size:13px}} .sr{{font-size:12px;margin:4px 0}}
.sc{{color:var(--muted);font-size:11px}}
.pgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:8px}}
.prob{{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--critical);border-radius:8px;
  padding:6px 9px;font-size:12px}} .prob b{{display:block;font-size:13px}}
.card.good{{border-top:3px solid var(--good)}} .card.bad{{border-top:3px solid var(--serious)}} .card ul{{margin:4px 0;padding-left:18px}}
table.hpsum td:first-child{{width:190px}} table.prevfx td:first-child{{width:150px}}
table.prevfx td:nth-child(2),table.prevfx td:nth-child(5),table.prevfx td:nth-child(6){{white-space:nowrap}}
table.prevfx td:nth-child(4){{min-width:260px}} .facts ul{{margin:4px 0;padding-left:18px}} .facts li{{margin:3px 0}}
table.stages td:nth-child(n+4),table.stages th:nth-child(n+4){{white-space:nowrap;text-align:center}}
table.stages td:first-child{{min-width:170px}} table.stages td:nth-child(2) code{{word-break:normal}}
svg .abox{{fill:var(--page);stroke:var(--axis)}} svg .abox.hi{{stroke:var(--s1);stroke-width:1.5}}
svg .wv{{fill:none;stroke-width:1.6}} svg .wv0{{stroke:var(--text2)}} svg .wv1{{stroke:var(--s1)}} svg .wv2{{stroke:var(--s2)}}
svg .wv3{{stroke:var(--s3)}} svg .wv4{{stroke:var(--s4)}}
svg .tok{{stroke-width:1}} svg .tk1{{fill:color-mix(in srgb,var(--s1) 30%,transparent);stroke:var(--s1)}}
svg .tk2{{fill:color-mix(in srgb,var(--s2) 30%,transparent);stroke:var(--s2)}}
svg .tk3{{fill:color-mix(in srgb,var(--s3) 30%,transparent);stroke:var(--s3)}}
svg .tk4{{fill:color-mix(in srgb,var(--s4) 30%,transparent);stroke:var(--s4)}}
svg .tkout{{fill:color-mix(in srgb,var(--critical) 35%,transparent);stroke:var(--critical)}}
svg .tokframe{{fill:none;stroke:var(--text2)}}
svg .badge{{fill:var(--text)}} svg .bt{{fill:var(--page);font-size:11px;font-weight:700}}
svg .tiny{{fill:var(--text2);font-size:10.5px}} svg .nowline{{stroke:var(--text2);stroke-dasharray:3 3}}
svg .pdot{{fill:var(--s3)}} svg .fan{{stroke:var(--s3);stroke-dasharray:2 3}}
svg .sub0{{fill:color-mix(in srgb,var(--s1) 16%,var(--page));stroke:none}} svg .sub1{{fill:var(--surface);stroke:var(--line)}}
svg .tree{{fill:none;stroke:var(--s2);stroke-width:1.4}} svg .arc{{fill:none;stroke:var(--s3);stroke-width:1.3;opacity:.85}}
svg .plus{{fill:var(--surface);stroke:var(--text2)}} svg .cellbox{{fill:var(--surface);stroke:var(--axis)}}
.mmap{{display:grid;grid-template-columns:1fr 24px 1fr 24px 1.3fr;gap:6px;align-items:stretch;margin:8px 0}}
.marrow{{align-self:center;text-align:center;color:var(--muted);font-size:18px}}
.mcol{{background:var(--surface);border:1px solid var(--line);border-top:5px solid var(--c);border-radius:12px;padding:8px 10px}}
.mh{{font-weight:800;font-size:15px}} .ml2{{color:var(--muted);font-size:12px;margin-bottom:6px}}
.mchip{{display:block;text-decoration:none;color:var(--text);background:var(--page);border:1px solid var(--line);
  border-left:4px solid var(--c);border-radius:8px;padding:5px 8px;margin:5px 0}}
.mchip:hover{{border-color:var(--c)}} .mchip.off{{opacity:.55}} .mchip b{{font-size:13px}}
.mk{{font-size:11px;color:var(--text2);border:1px solid var(--line);border-radius:999px;padding:0 6px;margin-left:6px}}
.mrow{{display:flex;flex-wrap:wrap;gap:4px;margin-top:2px}} .mu{{font-size:11px;color:var(--text2)}} .mu.off{{color:var(--muted)}}
.mw{{font-size:11px;font-weight:700;color:var(--good)}} span.win{{font-weight:700;color:var(--good);white-space:nowrap}}
i.cdot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
table.sel td{{vertical-align:middle}} table.mtab td:first-child{{white-space:nowrap}} table.tasks td:first-child{{white-space:nowrap}}
.ocards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:10px;margin:8px 0}}
.ocard{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px}}
.oh{{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:center}} .oh b{{font-size:15px}}
.oo{{font-size:17px;font-weight:700;margin:6px 0 2px}} .ocard .diagram{{background:var(--page)}}
.orows{{display:grid;gap:6px;font-size:13px}} .orows>div{{background:var(--page);border-radius:6px;padding:5px 8px}}
svg .wafer{{fill:color-mix(in srgb,var(--s1) 18%,var(--page));stroke:var(--s1)}} svg .ring{{fill:none;stroke:var(--s1);opacity:.5}}
svg .pl-win{{fill:var(--s1);fill-opacity:.10;stroke:var(--s1);stroke-opacity:.5;stroke-dasharray:3 3}}
svg .pl-future{{fill:none;stroke:var(--text2);stroke-width:1.1;stroke-dasharray:3 4;opacity:.55}}
a{{color:var(--s1)}}
.themebtn{{float:right;border:1px solid var(--line);background:var(--surface);color:var(--text2);border-radius:8px;
  padding:4px 8px;font:inherit;cursor:pointer}}
@media (max-width:760px){{header,main{{padding-left:16px;padding-right:16px}}
  .fbox{{grid-template-columns:1fr}} .sw{{grid-template-columns:1fr}} table.kv td:first-child{{width:auto}}
  .tw table:not(.kv){{min-width:680px}} .board{{grid-template-columns:1fr}} .pmeta{{grid-template-columns:1fr}}
  .mmap{{grid-template-columns:1fr}} .marrow{{transform:rotate(90deg)}}}}
@media print{{:root:root:root{{{block(LIGHT)}--shade:rgba(42,120,214,.08);--line:#e8e7e1}}
  .side,.navbtn,.scrim,.themebtn{{display:none}} .content{{margin-left:0}} section{{display:block!important;break-before:page}}
  .diagram{{overflow:visible}} .diagram svg{{width:100%;height:auto}}
  details{{border:none}} details>*:not(summary){{display:block}} body{{background:#fff}}}}
"""


_JS = """
const tabs=[...document.querySelectorAll('nav button[data-tab]')];
function show(id){const el=id&&document.getElementById(id);const sec=(el&&el.closest('main>section'))||document.getElementById(tabs[0].dataset.tab);
  tabs.forEach(b=>b.classList.toggle('on',b.dataset.tab===sec.id));
  document.querySelectorAll('nav .grp').forEach(g=>{const has=!!g.querySelector('[data-tab="'+sec.id+'"]');
    g.classList.toggle('has-on',has);g.classList.toggle('open',has);});
  document.querySelectorAll('main>section').forEach(s=>s.classList.toggle('on',s===sec));
  document.body.classList.remove('navopen');
  return el&&el!==sec?el:null;}
tabs.forEach(b=>b.addEventListener('click',()=>{history.replaceState(null,'','#'+b.dataset.tab);show(b.dataset.tab);window.scrollTo(0,0);}));
document.querySelectorAll('nav .grph').forEach(h=>h.addEventListener('click',()=>h.parentElement.classList.toggle('open')));
document.querySelectorAll('.navbtn,.scrim').forEach(b=>b.addEventListener('click',()=>document.body.classList.toggle('navopen')));
function pickSearch(row){const box=row.closest('.lab-xbox');if(!box)return;
  box.querySelectorAll('tr[data-xsel]').forEach(r=>r.classList.toggle('on',r===row));
  box.querySelectorAll('.lab-xpanel').forEach(p=>{p.hidden=p.dataset.x!==row.dataset.xsel;});}
document.addEventListener('click',ev=>{const row=ev.target.closest&&ev.target.closest('tr[data-xsel]');if(row)pickSearch(row);});
document.addEventListener('keydown',ev=>{const row=ev.target.closest&&ev.target.closest('tr[data-xsel]');if(row&&ev.key==='Enter')pickSearch(row);});
window.addEventListener('hashchange',()=>{const id=location.hash.slice(1);if(document.getElementById(id)){show(id);window.scrollTo(0,0);}});
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',ev=>{const id=a.getAttribute('href').slice(1);
  const el=document.getElementById(id);if(el&&el.tagName==='SECTION'){ev.preventDefault();history.replaceState(null,'','#'+id);show(id);window.scrollTo(0,0);}
  else if(el&&el.tagName==='DETAILS'){ev.preventDefault();el.open=true;el.scrollIntoView({block:'start'});}}));
const inner=show(location.hash.slice(1));
if('scrollRestoration' in history)history.scrollRestoration='manual';
setTimeout(()=>{if(inner){if(inner.tagName==='DETAILS')inner.open=true;inner.scrollIntoView({block:'start'});}else window.scrollTo(0,0);},0);
window.addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
"""
_THEME_JS = """
const tb=document.querySelector('.themebtn');
tb.addEventListener('click',()=>{const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme=dark?'light':'dark';});
"""


DEFAULT_THEME = "dark"  # the page opens dark; the header button switches to light, and print is always light


def render(data: ReportData, *, examples: bool = True, public: bool = False) -> str:
    """The report page. ``public``: the view-only dashboard — no server paths or code version in the header.
    Without replay data (``examples=False`` or nothing to replay) the 추론 시연 tab is left out."""

    demo_items = _demo_items(data) if examples else []
    tabs = [t for t in TABS if t[0] != "demo" or demo_items]
    sections = {
        "summary": _summary(data), "inputs": _inputs(data), "outputs": _outputs(data), "models": _models(data),
        "flow": _flow(data), "hparams": _hparams(data), "lab": lab.lab_section(data.ledger), "results": _results(data, examples), "weights": _weights(data),
        "demo": _demo(data, demo_items, examples), "notes": _notes(data),
        "history": _history(data), "plan": _plan(data),
    }
    nav = _sidebar(tabs)
    body = "".join(f'<section id="{k}"><h2>{e(v)}</h2>{sections[k]}</section>' for k, v in tabs)
    smoke = any(x.smoke for x in data.experiments)
    meta = (f"{data.created:%Y-%m-%d %H:%M} 기준 · 보기 전용" if public else
            f"{data.created:%Y-%m-%d %H:%M} 생성 · 코드 {_git_commit()} · 결과 폴더 {e(data.work_root)}")
    meta += " · <b>smoke(축소 확인) 결과 포함</b>" if smoke else ""
    return (
        f'<!doctype html><html lang="ko" data-theme="{DEFAULT_THEME}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>ALD 실험 보고 {data.created:%Y-%m-%d}</title><style>{_css()}{player.PLAYER_CSS}{lab.LAB_CSS}</style></head><body>"
        f'<nav class="side" aria-label="보고서 목차">{nav}<div class="sidefoot">'
        f'<button class="themebtn" type="button">밝게 / 어둡게</button><div class="meta">{meta}</div></div></nav>'
        '<div class="scrim"></div><div class="content"><header>'
        f'<h1><button class="navbtn" type="button" aria-label="목차 열기">{_icon("menu")}</button>'
        "ALD 공정 시계열 모델 실험 보고</h1></header>"
        f"<main>{body}</main></div>"
        f"<script>{_JS}{_THEME_JS}{player.PLAYER_JS}{lab.LAB_JS}</script></body></html>"
    )


def _icon(name: str, cls: str = "") -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true"{f" class={chr(34)}{cls}{chr(34)}" if cls else ""}><path d="{NAV_ICONS[name]}"/></svg>'


def _sidebar(tabs: list[tuple[str, str]]) -> str:
    """The left navigation: one button per single-tab group, a folding group otherwise (tabs not present are left out)."""

    names = dict(tabs)
    out = ['<div class="brand"><b>ALD Lab</b><span>PEALD SiN 공정 trace · PatchTST</span></div>',
           '<div class="navlabel">NAVIGATION</div>']
    for title, icon, keys in NAV_GROUPS:
        keys = [k for k in keys if k in names]
        if not keys:
            continue
        if len(keys) == 1:
            out.append(f'<button class="item" type="button" data-tab="{keys[0]}">{_icon(icon)}<span>{e(names[keys[0]])}</span></button>')
            continue
        subs = "".join(f'<button type="button" data-tab="{k}"><span>{e(names[k])}</span></button>' for k in keys)
        out.append(f'<div class="grp"><button class="grph" type="button">{_icon(icon)}<span>{e(title)}</span>'
                   f'{_icon("chev", "chev")}</button><div class="sub">{subs}</div></div>')
    return "".join(out)


def player_page(items: list[dict[str, Any]], *, title: str, meta: str) -> str:
    """A page with only the inference player (``ald-predict --html``), styled like the report."""

    return (
        f'<!doctype html><html lang="ko" data-theme="{DEFAULT_THEME}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)}</title><style>{_css()}{player.PLAYER_CSS}</style></head><body>"
        f'<header><button class="themebtn" type="button">밝게/어둡게</button><h1>{e(title)}</h1>'
        f'<div class="meta">{e(meta)}</div></header><main><section class="on">'
        '<p class="note">학습된 모델이 이 trace에 대해 낸 예측을 시간 순서대로 재생합니다. ▶ 재생을 누르세요. '
        "점선 상자는 모델이 보는 최근 구간, 점은 그 순간 낸 예측입니다.</p>"
        f"{player.player(items)}</section></main><script>{_THEME_JS}{player.PLAYER_JS}</script></body></html>"
    )


def write_report(data: ReportData, out: Path | None = None, *, examples: bool = True) -> Path:
    out = out or data.work_root / "reports" / f"ald_report_{data.created:%Y%m%d_%H%M}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, examples=examples), encoding="utf-8")
    if len(data.ledger):  # aggregates only: settings and scores of every job
        data.ledger.drop(columns=["params"], errors="ignore").to_csv(out.parent / "experiment_records.csv",
                                                                    index=False, encoding="utf-8-sig")
    latest = out.parent / "ald_report_latest.html"
    if out != latest:
        latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return out
