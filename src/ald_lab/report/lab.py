"""The 하이퍼파라미터 탐색 tab: PatchTST first — search, carry to every target, fall back where needed.

In the offline file the tab shows the search results and where each step stands (from the cumulative records).
Opened through the lab server (``ald-lab-server``, ``scripts/lab.sh``) the same page turns on a form that queues
experiments with chosen hyperparameter combinations, the queue, and buttons for steps 2 and 3; :data:`LAB_JS` asks
the server for fragments rendered here (:func:`explore_html`, :func:`first_html`, :func:`queue_html`), so offline
and live look the same. A server started with ``ALD_DASHBOARD_RUN=0`` is view-only: no form, no buttons.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import stages
from ..launch import INPUT_SET_NAMES, STATUS_NAMES, list_requests, worker_running
from .charts import improvement_bars

TASKS = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서"}
STATUS_CLASS = {"queued": "", "running": "warn", "done": "good", "failed": "bad", "cancelled": "bad"}
RANKING_ROWS = 8
QUEUE_SHOWN = 4  # request cards shown before the list scrolls
EXPLORE_GROUPS = 40  # newest searches listed in the tab
FACTOR_NAMES = {"channel_mode": "신호 처리", "patch_seconds": "patch 길이(초)", "stride_seconds": "patch 간격(초)",
                "d_model": "d_model", "n_heads": "head 수", "n_layers": "층 수", "d_ff": "FFN 크기", "dropout": "dropout",
                "revin": "RevIN", "revin_center": "RevIN 기준", "train.lr": "학습률", "input_set": "입력 세트",
                "context_seconds": "window(초)", "head_dropout": "head dropout", "kernel_seconds": "이동평균(초)"}
VALUE_NAMES = {"mixing": "혼합", "independent": "독립"}
VERDICTS = {"clear": ('<span class="st good">확실</span>', False), "small": ('<span class="st">작음</span>', "작음"),
            "unclear": ('<span class="st">잡음 수준</span>', "잡음 수준"),
            "norepeat": ('<span class="st warn">반복 없음</span>', "반복 없음")}


def _h():
    from . import render  # helpers live with the rest of the report

    return render


def factor_name(names: list[str]) -> str:
    return " = ".join(FACTOR_NAMES.get(n, n) for n in names)


def value_text(name: str, value: Any) -> str:
    if name == "input_set":
        return INPUT_SET_NAMES.get(str(value), str(value))
    if isinstance(value, float):
        return f"{value:g}"
    return VALUE_NAMES.get(str(value), str(value))


def settings_text(settings: dict[str, Any]) -> str:
    return " · ".join(f"{FACTOR_NAMES.get(k, k)} {value_text(k, v)}" for k, v in settings.items()) or "기본값"


def _settings(text: str) -> str:
    r = _h()
    return f'<span class="sub">{r.e(text.replace("channel_mode=", "").replace("train.lr=", "lr="))}</span>'


# ----------------------------------------------------------------------------- search results (설정별 효과)

def _effects_block(g: dict[str, Any]) -> str:
    r = _h()
    bars, rows = [], []
    for ef in g["effects"]:
        key = ef["factors"][0]
        name = factor_name(ef["factors"])
        a, b = ef["pair"]
        word, muted = VERDICTS[ef["verdict"]]
        bars.append((f"{name}: {value_text(key, a)} → {value_text(key, b)}", ef["effect"], muted))
        levels = " · ".join(f"{value_text(key, lv)} {ef['means'][lv]:+.1f}" for lv in ef["levels"])
        per_target = " · ".join(f"{t} {v:+.1f}" for t, v in ef["per_target"].items())
        noise = f" <span class='sub'>± {2 * ef['se']:.1f}</span>" if ef["se"] else ""
        rows.append([r.e(name), f'<span class="sub">{r.e(levels)}</span>', f"{ef['effect']:+.1f}%p{noise}", word,
                     ("같음" if ef["same_direction"] else "<b>다름</b>") + f'<div class="sub">{r.e(per_target)}</div>',
                     r.e(ef["hint"] or "-")])
    if not bars:
        return '<p class="empty">조합 사이에 바뀐 설정이 없습니다.</p>'
    chart = improvement_bars(bars, title="설정별 효과", unit="%p")
    table = r.table(["설정", "값별 평균 (대상 평균 대비 %p)", "효과", "판정", "대상별 방향", "다음에"], rows, cls="lab-eff")
    return f'<div class="chartbox"><div class="ctitle">설정을 바꿨을 때 개선율 변화 (%p)</div>{chart}</div>{table}'


def _suggestion(g: dict[str, Any]) -> str:
    clear = [ef for ef in g["effects"] if ef["verdict"] == "clear"]
    unclear = [ef for ef in g["effects"] if ef["verdict"] in ("unclear", "small")]
    parts = []
    if clear:
        best = ", ".join(f"{factor_name(ef['factors'])} {value_text(ef['factors'][0], ef['best'])}" for ef in clear)
        parts.append(f"효과가 확실한 설정은 최적값({best}) 주변을 3개 수준으로 세분화")
        edge = [factor_name(ef["factors"]) + f" ({ef['hint']})" for ef in clear if ef["hint"]]
        if edge:
            parts.append("최적값이 범위 끝에 있는 설정은 범위 확장: " + ", ".join(edge))
    if unclear:
        parts.append("효과가 잡음 수준이거나 작은 설정(" + ", ".join(factor_name(ef["factors"]) for ef in unclear)
                     + ")은 학습이 빠른 값으로 고정")
    if not parts:
        return ""
    return '<p class="lab-next"><b>다음 실험 제안</b>: ' + _h().e("; ".join(parts)) + ".</p>"


def _ranking_block(g: dict[str, Any]) -> str:
    r = _h()
    rows = []
    for rank, c in enumerate(g["ranking"][:RANKING_ROWS], start=1):
        sd = f" <span class='sub'>± {c['sd']:.1f}</span>" if c["sd"] is not None else ""
        cycle = f"{c['cycle']:+.1f} %" if c["cycle"] is not None else "-"
        maes = " · ".join(f"{t} {r.fmt(v, 3)}" for t, v in c["mae_by_target"].items())
        size = f"{c['n_params'] / 1e6:.2f}M" if c["n_params"] else "-"
        rows.append([f"<b>{rank}</b>", r.e(settings_text(c["settings"])), f"<b>{c['improvement']:+.1f} %</b>{sd}", cycle,
                     f'<span class="sub">{r.e(maes)}</span>', r.minutes(c["fit_minutes"]) if c["fit_minutes"] else "-",
                     size])
    more = f'<p class="sub">조합 {g["combos"]}개 중 상위 {min(RANKING_ROWS, g["combos"])}개.</p>' if g["combos"] > RANKING_ROWS else ""
    return (r.table(["순위", "조합 (바꾼 설정)", "개선율 (평균 ± seed 편차)", "사이클 안", "대상별 MAE", "작업당 학습", "파라미터"],
                    rows, cls="lab-rank") + more)


def explore_html(ledger: pd.DataFrame) -> str:
    """Every PatchTST search (dashboard request or step-1 file family), newest first: settings' effects and the
    combinations ranked."""

    r = _h()
    groups = stages.explore_groups(ledger) if len(ledger) else []
    if not groups:
        return ('<p class="empty">아직 탐색 결과가 없습니다. 위 양식에서 "탐색용"으로 조합을 넣어 학습하면 '
                "완료된 작업부터 설정별 효과와 조합 순위가 표시됩니다.</p>")
    out = [('<p class="note"><b>개선율</b> = 기준선(현재값 유지·레시피 평균) 대비 검증 MAE 감소율. <b>효과</b> = 그 설정의 좋은 값 '
           "평균 − 다른 값 평균(대상별 평균을 빼고 비교, %p). 같은 조합·대상을 seed만 바꿔 학습한 편차로 잡음 크기를 추정하며, "
           "효과가 그 표준오차의 2배를 넘고 1 %p 이상이면 <b>확실</b>, 2배를 넘지만 1 %p 미만이면 <b>작음</b>, 2배 이내면 "
           "<b>잡음 수준</b>입니다. 탐색(대시보드 실행과 1단계 실험 파일)은 "
           "요청(메모)별로 따로 집계합니다.</p>")]
    rows, panels = [], []
    for i, g in enumerate(groups[:EXPLORE_GROUPS]):
        key = "x" + hashlib.sha1(f"{g['task']}|{g['title']}".encode()).hexdigest()[:10]
        when = f"{g['when']:%m-%d %H:%M}" if pd.notna(g["when"]) else ""
        running = ' <span class="st warn">진행 중</span>' if g["running"] else ""
        sigma = f"seed 잡음 σ = {g['sigma']:.2f}%p" if g["sigma"] is not None else "seed 1개 (잡음 크기 측정 불가)"
        best = f"{g['ranking'][0]['improvement']:+.1f} %" if g["ranking"] else "-"
        clear = sum(ef["verdict"] == "clear" for ef in g["effects"])
        on = ' class="on"' if i == 0 else ""
        rows.append(f'<tr data-xsel="{key}" tabindex="0"{on}><td><b>{r.e(g["title"])}</b>{running}</td>'
                    f'<td>{r.e(TASKS.get(g["task"], g["task"]))}</td><td>{r.e(when)}</td>'
                    f'<td>{g["combos"]} × {len(g["targets"])} × {len(g["seeds"])}</td><td>{best}</td><td>{clear}개</td></tr>')
        hidden = "" if i == 0 else " hidden"
        panels.append(f'<div class="lab-xpanel catsec lab-explore" data-x="{key}"{hidden} '
                      f'style="--c:var(--s{1 if g["task"] == "forecast" else 3})">'
                      f'<div class="cath"><i></i><b>{r.e(g["title"])}</b>{running} '
                      f'<span class="sub">{r.e(TASKS.get(g["task"], g["task"]))} · {r.e(when)} · 조합 {g["combos"]}개 × '
                      f'대상 {len(g["targets"])}개 × seed {len(g["seeds"])}개 · {r.e(sigma)}</span></div>'
                      + _suggestion(g) + _effects_block(g) + "<h4>조합 순위</h4>" + _ranking_block(g) + "</div>")
    more = (f'<p class="sub">최근 {EXPLORE_GROUPS}개만 표시합니다. 이전 탐색 {len(groups) - EXPLORE_GROUPS}개는 '
            "Hyper Parameter 탭 ⑦ 누적 기록에 있습니다.</p>" if len(groups) > EXPLORE_GROUPS else "")
    listing = r.table(["탐색 (메모)", "과제", "언제", "조합 × 대상 × seed", "최고 개선율", "확실한 효과"], [], cls="lab-xlist")
    listing = listing.replace("<tbody></tbody>", "<tbody>" + "".join(rows) + "</tbody>")
    out.append('<div class="lab-xbox"><p class="sub">목록에서 탐색을 선택하면 아래에 설정별 효과와 조합 순위가 표시됩니다.</p>'
               + listing + more + "".join(panels) + "</div>")
    return "".join(out)


# ----------------------------------------------------------------------------- PatchTST-first status

def _decision_chip(row: Any) -> str:
    cls = {"keep": "good", "fallback": "bad", "missing": "warn"}[row.decision]
    text = row.reason
    if row.decision == "keep":  # the gain has its own column; keep only the caveat
        text = text[text.find("(") + 1:-1] if "(" in text else ""
    reason = f'<div class="sub">{_h().e(text)}</div>' if text else ""
    return f'<span class="st {cls}">{stages.DECISIONS[row.decision]}</span>{reason}'


def _num(value: Any) -> bool:
    return value is not None and pd.notna(value)


def _decision_table(frame: pd.DataFrame) -> str:
    r = _h()
    rows = []
    for row in frame.itertuples():
        patch = "-"
        if _num(row.patch_mae):
            spread = f" ± {r.fmt(row.patch_std, 2)}" if _num(row.patch_std) else ""
            shown_input = "실험 기본 입력" if row.patch_input == "default" else f"입력 {row.patch_input}"
            patch = (f"<b>{r.fmt(row.patch_mae)}</b>{spread}<div>{_settings(row.patch_settings)}</div>"
                     f'<div class="sub">{r.e(shown_input)} · seed {row.patch_seeds}개 · 시도한 설정 {row.patch_configs}개 · '
                     f"{r.e(row.patch_experiment)}</div>")
        gain = r.pct(row.patch_gain_pct) if _num(row.patch_gain_pct) else "-"
        lgbm = r.fmt(row.lgbm_mae)
        if _num(row.lgbm_gap_pct):
            lgbm += f'<div class="sub">PatchTST {row.lgbm_gap_pct:+.0f} %</div>'
        other = (f"{r.e(r.model_card(row.other_model)['name'])} {r.fmt(row.other_mae)}"
                 if row.other_model else '<span class="sub">-</span>')
        final = r.model_card(row.final)["name"] if row.final else "-"
        rows.append([f"<b>{r.e(row.target)}</b> <span class='sub'>{r.e(row.unit)}</span>", patch, r.fmt(row.baseline_mae),
                     gain, _decision_chip(row), lgbm, other, f"{r.e(final)} <span class='sub'>{r.fmt(row.final_mae)}</span>"])
    return r.table(["대상", "PatchTST 최고 (검증 MAE · 설정)", "기준선", "기준선 대비", "판정", "LightGBM (참고)",
                    "다른 모델 최고", "최종"], rows, cls="lab-first")


def decision_rule_text() -> str:
    return (f"판정 기준: PatchTST 최고 기록(같은 설정의 seed 평균)이 기준선보다 seed 표준편차의 {stages.NOISE_SD:g}배를 넘게 "
            "좋으면 <b>PatchTST 채택</b>입니다. 기준선을 넘지 못한 대상만 3단계에서 다른 모델(iTransformer · DLinear · LightGBM)을 "
            "적용합니다. LightGBM은 참고용이며 판정에는 쓰지 않습니다.")


def first_html(ledger: pd.DataFrame, *, live: bool = False) -> str:
    """Where PatchTST-first stands for each task; with ``live`` the step 2/3 buttons too."""

    r = _h()
    out = []
    for task, title in TASKS.items():
        frame = stages.patchtst_first(ledger, task) if len(ledger) else pd.DataFrame()
        ranking = stages.search_ranking(ledger, task) if len(ledger) else pd.DataFrame()
        n = stages.summary_numbers(frame) if len(frame) else {"targets": 0, "keep": 0, "fallback": 0, "missing": 0}
        out.append(f'<div class="catsec lab-task" style="--c:var(--s{1 if task == "forecast" else 3})">'
                   f'<div class="cath"><i></i><b>{r.e(title)}</b> '
                   f'<span class="sub">기준 실험 {r.e(stages.BASE_FOR_TASK[task])}</span></div>')
        out.append('<div class="tiles">' + r.tile("대상", f"{n['targets']}개", "기록이 있는 대상")
                   + r.tile("PatchTST 채택", f"{n['keep']}개", "기준선보다 확실히 좋음")
                   + r.tile("대안 필요", f"{n['fallback']}개", "기준선을 못 넘음 → 3단계")
                   + r.tile("PatchTST 미학습", f"{n['missing']}개", "2단계에서 학습") + "</div>")
        steps = [
            ("①", "탐색", f"조합 {len(ranking)}개 시도" if len(ranking) else "아직 없음",
             "대시보드 '탐색용' 또는 p1_patchtst_search_* 파일"),
            ("②", "전체 적용", "상위 조합 → 대상 전부 (정식 학습)", f"{stages.STAGE2_PREFIX}_{task}"),
            ("③", "대안", f"대안 필요 {n['fallback']}개 대상", f"{stages.STAGE3_PREFIX}_{task}"),
        ]
        out.append('<div class="steps2">' + "".join(
            f'<div class="step2"><span class="sk">{k}</span> <span class="st2">{t}</span><div class="sr">{r.e(d)}</div>'
            f'<div class="sc"><code>{r.e(c)}</code></div></div>' for k, t, d, c in steps) + "</div>")
        if live:
            options = "".join(f"<option{' selected' if k == 3 else ''}>{k}</option>" for k in range(1, 6))
            seeds = ", ".join(str(s) for s in stages.FULL_SEEDS)
            out.append(f'<div class="lab-actions" data-task="{task}">'
                       f'<label>상위 <select class="lab-k">{options}</select>개 조합</label>'
                       f'<label>seed <input class="lab-stage-seeds" value="{seeds}" size="9"></label>'
                       f'<button type="button" class="pl-btn pl-main" data-stage="2" {"disabled" if not len(ranking) else ""}>'
                       "② 모든 대상에 적용</button>"
                       f'<button type="button" class="pl-btn" data-stage="3" {"disabled" if not n["fallback"] else ""}>'
                       f"③ 대안 실행 ({n['fallback']}개 대상)</button>"
                       f'<span class="lab-msg"></span></div>')
        out.append("<h4>대상별 판정</h4>" + (_decision_table(frame) if len(frame) else '<p class="empty">기록 없음</p>'))
        out.append("</div>")
    out.append(f'<p class="sub">{decision_rule_text()}</p>')
    return "".join(out)


# ----------------------------------------------------------------------------- queue

def _remaining(q: dict[str, Any]) -> str:
    """At the current pace: wall time per finished job and what is left (running requests)."""

    progress, started = q.get("progress"), q.get("started")
    if q.get("status") != "running" or not progress or not started or not progress[0]:
        return ""
    try:
        start = datetime.fromisoformat(started)
    except ValueError:
        return ""
    elapsed = (datetime.now(start.tzinfo) - start).total_seconds() / 60
    done, total = progress
    per_job = elapsed / done
    r = _h()
    return f" · 작업당 평균 {r.minutes(per_job)} · 예상 남은 시간 약 {r.minutes(per_job * max(0, total - done))}"


def queue_html(work_root: Path, limit: int = 60, *, live: bool = True) -> str:
    """The request cards for the live tab (``live=False``: view-only, no cancel buttons). Running and queued
    requests come first (queued in the order the worker takes them), then the rest newest first; past
    ``QUEUE_SHOWN`` cards the list scrolls inside its box."""

    r = _h()
    requests = list_requests(work_root)
    running = [q for q in requests if q.get("status") == "running"]
    queued = [q for q in requests if q.get("status") == "queued"][::-1]
    rest = [q for q in requests if q.get("status") not in ("running", "queued")]
    requests = running + queued + rest
    active = running + queued
    shown = requests[:limit]
    more = f" · 최근 {len(shown)}건 표시" if len(requests) > len(shown) else ""
    scroll = " · 목록 안에서 스크롤" if len(shown) > QUEUE_SHOWN else ""
    finished = len(requests) - len(active)
    bulk = (f' <button type="button" class="pl-btn lab-hide" data-hide-all="{finished}">끝난 요청 {finished}건 목록에서 지우기</button>'
            if live and finished else "")
    head = ('<p class="sub lab-qhead">' + ("작업자 실행 중" if worker_running(work_root) else "작업자 대기 중")
            + (f" · 대기·실행 {len(active)}건" if active else " · 실행 중인 요청 없음")
            + f" · 요청 {len(requests)}건{more}{scroll} · 5초마다 새로 고침{bulk}</p>")
    if not requests:
        return head + ('<p class="empty">요청이 없습니다. 위에서 설정을 고른 뒤 ▶ 대기열에 넣고 학습을 누르세요.</p>' if live
                       else '<p class="empty">요청이 없습니다.</p>')
    cards = []
    for q in shown:
        status = q.get("status", "")
        progress = q.get("progress")
        bar = ""
        if progress and progress[1]:
            done, total = progress
            bar = (f'<div class="lab-bar"><i style="width:{100 * min(1, done / total):.1f}%"></i></div>'
                   f'<div class="sub">작업 {done} / {total} 완료{_remaining(q)}</div>')
        times = [f"요청 {q['created'][5:16].replace('T', ' ')}"]
        if q.get("started"):
            times.append(f"시작 {q['started'][11:16]}")
        if q.get("finished"):
            times.append(f"끝 {q['finished'][11:16]}")
        if not live:
            cancel = ""
        elif status in ("queued", "running"):
            cancel = f'<button type="button" class="pl-btn" data-cancel="{r.e(q["id"])}">취소</button>'
        else:
            cancel = (f'<button type="button" class="pl-btn lab-hide" data-hide="{r.e(q["id"])}" '
                      'title="목록에서만 지웁니다. 결과는 누적 기록과 탐색 결과에 남습니다">목록에서 지우기</button>')
        log = (f'<details class="more" data-log="{r.e(q["id"])}"><summary>학습 로그</summary><pre class="code">…</pre></details>'
               if status != "queued" else "")
        cards.append(f'<div class="card lab-req"><div class="lab-req-h"><span class="st {STATUS_CLASS.get(status, "")}">'
                     f'{r.e(STATUS_NAMES.get(status, status))}</span> <b>{r.e(q.get("memo") or q["name"])}</b>{cancel}</div>'
                     f'<div class="sub">{r.e(q["name"])} · 대상 {r.e(", ".join(q.get("targets") or []))} · GPU {r.e(q.get("gpus"))}'
                     f'{" · smoke" if q.get("smoke") else ""} · {" · ".join(times)} · <code>{r.e(q["id"])}</code></div>'
                     f"{bar}{log}</div>")
    cls = "lab-reqs scroll" if len(cards) > QUEUE_SHOWN else "lab-reqs"
    return head + f'<div class="{cls}">' + "".join(cards) + "</div>"


# ----------------------------------------------------------------------------- the tab

def lab_section(ledger: pd.DataFrame) -> str:
    """The tab as written into the report file (live parts are filled in by the lab server)."""

    return (
        '<p class="note">주 모델은 PatchTST입니다. <b>① 대표 대상으로 PatchTST 하이퍼파라미터 조합을 탐색</b>하고, '
        "<b>② 상위 조합을 모든 대상에 정식 학습</b>한 뒤, <b>③ 기준선을 넘지 못한 대상에만 다른 모델</b>을 적용합니다. "
        "결과는 성능표와 Hyper Parameter ⑦ 누적 기록에도 반영됩니다.</p>"
        '<div class="lab-note" id="lab-offline"><b>실행 기능은 서버에서 열 때 켜집니다.</b> 서버에서 '
        "<code>bash scripts/lab.sh</code>, 노트북에서 SSH 터널을 연 뒤 브라우저로 접속하면 이 탭에서 바로 조합을 골라 학습하고 "
        "진행 상황을 볼 수 있습니다. 지금 보이는 표는 보고서 생성 시점의 기록입니다.</div>"
        '<div id="lab-live" hidden>'
        '<div class="lab-note" id="lab-readonly" hidden><b>보기 전용 화면입니다.</b> 학습은 담당자가 별도로 실행하며, '
        "이 화면에서는 진행 상황과 결과를 확인할 수 있습니다. 학습이 끝나면 화면 상단에 새 결과 알림이 표시됩니다.</div>"
        '<div id="lab-new"><h3>새 실험 등록 (하이퍼파라미터 조합)</h3>'
        '<p class="note">칸에 값을 쉼표로 여러 개 넣으면(예: <code>8, 32</code>) 가능한 조합을 모두 만듭니다. '
        "<b>빠른 설정</b>의 \"탐색용\"은 대표 대상 · seed 2개 · 짧은 학습, \"정식 학습\"은 모든 대상 · seed 3개 · 실험 기본 "
        "학습량으로 채웁니다. window 길이를 여러 개 넣으면 window별로 실험이 하나씩 대기열에 들어갑니다.</p>"
        '<div id="lab-form" class="lab-form"></div><div id="lab-preview"></div>'
        '<details class="more lab-file" id="lab-file"><summary><b>실험 계획 파일(.md)로 일괄 등록</b> (여러 차수를 '
        "한 번에 대기열에)</summary>"
        '<p class="note">차수마다 <code>### 15차: …</code> 제목 아래에 <code>| 순서 | 칸 | 입력 |</code> 표(위 양식의 칸 이름 그대로)를 '
        "적은 마크다운 파일을 올리거나 내용을 붙여 넣으세요. <b>기준 실험</b> 줄이 있는 표만 실험으로 읽고, 나머지 글과 표는 "
        "건너뜁니다. GPU 칸에 번호가 없으면 위 양식에서 고른 GPU를 씁니다. 미리보기에서 차수별 작업 수와 문제를 확인한 뒤 "
        "등록하면 파일에 적힌 순서대로 대기열에 들어갑니다.</p>"
        '<div class="lab-row"><input type="file" id="lab-file-input" accept=".md,.markdown,.txt">'
        '<button type="button" class="pl-btn" id="lab-file-plan">미리보기</button>'
        '<button type="button" class="pl-btn pl-main" id="lab-file-run" disabled>▶ 모두 대기열에 넣기</button></div>'
        '<textarea id="lab-file-text" rows="6" placeholder="### 15차: …&#10;| 순서 | 칸 | 입력 |&#10;|---|---|---|&#10;'
        '| 1 | 기준 실험 | g1_forecast_params |&#10;…"></textarea>'
        '<div id="lab-file-preview"></div></details></div>'
        '<h3>진행 상황</h3><div id="lab-queue"></div></div>'
        "<h3>탐색 결과 (설정별 영향)</h3>"
        f'<div id="lab-explore">{explore_html(ledger)}</div>'
        "<h3>모델 선정 단계별 현황</h3>"
        f'<div id="lab-first">{first_html(ledger)}</div>'
    )


LAB_CSS = """
.lab-note{background:var(--shade);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0;font-size:13px}
.lab-form{display:grid;gap:10px}
.lab-row{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:13px}
.lab-row>label{display:flex;gap:6px;align-items:center;color:var(--text2)}
.lab-form select,.lab-form input[type=text],.lab-form input[type=number],.lab-actions select,.lab-actions input{font:inherit;
  color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:4px 7px}
.lab-chips{display:flex;flex-wrap:wrap;gap:4px}
.lab-chip{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--surface);color:var(--text2);border-radius:999px;
  padding:3px 10px;cursor:pointer} .lab-chip.on{background:var(--s1);border-color:var(--s1);color:#fff}
.lab-preset{font:inherit;font-size:12px;border:1px solid var(--s1);background:transparent;color:var(--text);border-radius:8px;
  padding:4px 10px;cursor:pointer}
.lab-box{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.lab-box h4{margin:2px 0 6px}
table.lab-params td{vertical-align:middle} table.lab-params input{width:100%;min-width:140px}
table.lab-params td:first-child{white-space:nowrap;width:180px} table.lab-params td.lab-cnt{width:64px;white-space:nowrap}
.lab-n{font-size:11px;color:var(--s1);font-weight:600}
.lab-err{color:var(--critical);font-weight:600} .lab-warn{color:var(--serious);font-weight:600} .lab-msg{font-size:12px;color:var(--text2)}
.lab-next{background:var(--shade);border-left:3px solid var(--s1);border-radius:6px;padding:8px 10px;margin:8px 0;font-size:13px}
.lab-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0;font-size:13px}
.lab-actions label{display:flex;gap:4px;align-items:center;color:var(--text2)}
.lab-reqs{display:grid;gap:8px} .lab-req{padding:8px 10px} .lab-req-h{display:flex;gap:8px;align-items:center}
.lab-file{margin:14px 0 4px} .lab-file textarea{width:100%;font:12px/1.5 ui-monospace,Consolas,monospace;background:var(--surface);
  color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px;margin:6px 0}
.lab-file input[type=file]{font:inherit;font-size:13px;max-width:100%}
table.lab-filelist td:nth-child(n+3){white-space:nowrap}
.lab-qhead{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px} .lab-qhead .pl-btn{margin-left:auto;padding:2px 10px}
.lab-hide{opacity:.85} .lab-hide:hover{opacity:1}
table.lab-xlist tr[data-xsel]{cursor:pointer} table.lab-xlist tr[data-xsel]:hover td{background:var(--shade)}
table.lab-xlist tr.on td{background:var(--shade);font-weight:600} table.lab-xlist tr.on td:first-child{box-shadow:inset 3px 0 0 var(--s1)}
table.lab-xlist td:nth-child(n+3){white-space:nowrap} .lab-xpanel{margin-top:12px}
.lab-reqs.scroll{max-height:470px;overflow-y:auto;overscroll-behavior:contain;padding:8px;border:1px solid var(--grid);
  border-radius:8px}
.lab-req-h .pl-btn{margin-left:auto;padding:2px 10px}
.lab-bar{height:8px;background:var(--grid);border-radius:4px;margin:6px 0 2px;overflow:hidden}
.lab-bar i{display:block;height:100%;background:var(--s1)}
table.lab-rank td:nth-child(n+3),table.lab-rank th:nth-child(n+3){white-space:nowrap}
table.lab-first th{white-space:nowrap} table.lab-first td:nth-child(n+3){white-space:nowrap}
.step2 .sc code{overflow-wrap:anywhere;white-space:normal}
.pl-btn:disabled{opacity:.45;cursor:not-allowed}
.lab-box,.lab-wide{overflow-x:auto}
.newbar{position:sticky;top:0;z-index:20;display:flex;gap:10px;align-items:center;justify-content:center;padding:8px 12px;
  background:var(--s1);color:#fff;font-size:13px} .newbar button{font:inherit;border:0;border-radius:6px;padding:3px 10px;cursor:pointer}
@media (max-width:640px){
  .lab-row>label{flex-wrap:wrap;min-width:0;max-width:100%} .lab-form select,.lab-actions select{max-width:100%}
  .lab-chips{min-width:0} table.lab-params td:first-child{width:auto;white-space:normal}
  table.lab-params input{min-width:90px} table.lab-params td:last-child{font-size:11px}
}
"""

LAB_JS = r"""
(function(){
const sec=document.getElementById('lab'); if(!sec) return;
if(!/^https?:$/.test(location.protocol)) return;
const HDR={'X-Requested-With':'ald-lab'};
const get=p=>fetch(p,{headers:HDR}).then(r=>{if(!r.ok)throw r;return r.headers.get('content-type').includes('json')?r.json():r.text();});
const post=(p,b)=>fetch(p,{method:'POST',headers:Object.assign({'Content-Type':'application/json'},HDR),body:JSON.stringify(b)})
  .then(r=>r.json());
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let M=null,F=null,timer=null,built=null;
// a bar on top when the server rebuilt the report after this page was opened
function watchBuild(){get('/api/status').then(s=>{if(built===null){built=s.built_at;return;}
  if(s.built_at>built&&!$('newbar')){const bar=document.createElement('div');bar.id='newbar';bar.className='newbar';
    bar.innerHTML='새 결과가 반영됐습니다 <button type="button">새로고침</button>';bar.querySelector('button').onclick=()=>location.reload();
    (document.querySelector('.content')||document.body).prepend(bar);}}).catch(()=>{});}
watchBuild();setInterval(watchBuild,20000);
get('/api/meta').then(meta=>{M=meta;$('lab-offline').hidden=true;$('lab-live').hidden=false;
  if(!M.run_enabled){$('lab-readonly').hidden=false;$('lab-new').hidden=true;}
  else{pickBase(M.default_base);wireFile();
    if(!location.hash){const b=document.querySelector("nav button[data-tab='lab']");if(b)b.click();}}
  queue();setInterval(queue,5000);first();setInterval(first,20000);}).catch(()=>{});
// a whole plan file (markdown): preview every request, then queue them all in the order written
function planText(){return $('lab-file-text').value;}
function fileBody(){return {text:planText(),gpus:F?[...F.gpus].sort().join(',')||'cpu':'auto'};}
function wireFile(){if(!$('lab-file-input'))return;
  $('lab-file-input').onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{$('lab-file-text').value=t;planFile();});};
  $('lab-file-plan').onclick=planFile;$('lab-file-text').oninput=()=>{$('lab-file-run').disabled=true;};
  $('lab-file-run').onclick=()=>{const n=$('lab-file-run').dataset.n;if(!confirm('실험 '+n+'개를 파일 순서대로 대기열에 넣을까요?'))return;
    $('lab-file-run').disabled=true;post('/api/submit-file',fileBody()).then(r=>{
      $('lab-file-preview').insertAdjacentHTML('afterbegin',r.error?`<p class="lab-err">${esc(r.error)}</p>`:
        `<p class="st good">${r.requests}개 실험을 대기열에 넣었습니다 (요청 ${r.ids.length}건). 아래 진행 상황에서 확인하세요.</p>`);queue();});};}
function planFile(){if(!planText().trim())return;$('lab-file-preview').innerHTML='<p class="sub">읽는 중…</p>';
  post('/api/plan-file',fileBody()).then(r=>{const box=$('lab-file-preview');
    if(r.error){box.innerHTML=`<p class="lab-err">${esc(r.error)}</p>`;return;}
    const items=r.items||[],ok=items.length>0&&items.every(i=>i.ok);
    if(!items.length){box.innerHTML='<p class="lab-err">기준 실험 줄이 있는 표를 찾지 못했습니다.</p>';$('lab-file-run').disabled=true;return;}
    box.innerHTML=`<div class="tw"><table class="lab-filelist"><tr><th>#</th><th>실험 (메모)</th><th>기준 실험</th><th>대상</th><th>작업</th><th>조합</th><th>실험 수</th><th>예상 시간</th><th>확인</th></tr>`
      +items.map((i,n)=>`<tr><td>${n+1}</td><td><b>${esc(i.memo||i.title)}</b><div class="sub">${esc(i.title)}</div></td><td>${esc(i.base||'-')}</td>`
        +`<td>${i.targets?i.targets+'개':'-'}</td><td>${i.ok?i.jobs+'개':'-'}</td><td>${i.ok?i.combos+'개':'-'}</td><td>${i.ok?i.experiments+'개':'-'}</td>`
        +`<td>${i.ok?esc(i.estimate):'-'}</td><td>${i.ok?'<span class="st good">좋음</span>':'<span class="st bad">문제</span>'}`
        +[...(i.problems||[]).map(p=>`<div class="lab-err">${esc(p)}</div>`),...(i.warnings||[]).map(w=>`<div class="lab-warn">⚠ ${esc(w)}</div>`),
          ...(i.ignored&&i.ignored.length?[`<div class="sub">읽지 않은 칸: ${esc(i.ignored.join(', '))}</div>`]:[])].join('')+`</td></tr>`).join('')
      +`</table></div><p class="sub">모두 합쳐 작업 ${r.jobs}개 · GPU 칸이 비어 있으면 위 양식의 GPU(${esc(fileBody().gpus)})</p>`;
    $('lab-file-run').disabled=!ok;$('lab-file-run').dataset.n=items.length;
    $('lab-file-run').textContent=ok?'▶ '+items.length+'개 모두 대기열에 넣기':'▶ 문제를 수정한 뒤 등록할 수 있습니다';}).catch(()=>{});}
function base(){return M.bases.find(b=>b.name===F.base);}
function pickBase(name){const b=M.bases.find(x=>x.name===name)||M.bases[0];
  F={base:b.name,targets:new Set(b.targets),models:{},search:0,input_sets:new Set(b.input_sets),seeds:b.seeds.join(', '),
     windows:b.window?String(b.window):'',train:Object.assign({},b.train),gpus:new Set(M.gpus),smoke:false,report:true,memo:''};
  const first=M.models_by_task[b.task].includes('patchtst')?['patchtst']:[M.models_by_task[b.task][0]];
  first.forEach(m=>addModel(m));render();}
function preset(kind){const b=base(),ex=M.explore[b.task];
  if(kind==='explore'&&ex){F.targets=new Set(ex.targets.filter(t=>b.targets.includes(t)));F.seeds=M.explore_seeds.join(', ');
    F.train=Object.assign({},b.train,M.explore_train);}
  else{F.targets=new Set(b.targets);F.seeds=M.full_seeds.join(', ');F.train=Object.assign({},b.train);}
  F.search=0;render();}
function addModel(m){const b=base(),d=(b.defaults[m]||{});F.models[m]={};
  (M.space[b.task][m]||[]).forEach(p=>{F.models[m][p.key]=(d[p.key]||[p.default]).join(', ');});}
function chips(name,items,set,label){return `<div class="lab-chips" data-chips="${name}">`+items.map(([v,t])=>
  `<button type="button" class="lab-chip${set.has(v)?' on':''}" data-v="${esc(v)}">${esc(t||v)}</button>`).join('')+'</div>';}
function count(text){return String(text||'').replace(/;/g,',').split(',').filter(x=>x.trim()).length;}
function badge(n){return n>1?`<span class="lab-n">값 ${n}개</span>`:'';}
function render(){const b=base(),task=b.task;
  const bases=M.bases.map(x=>`<option value="${esc(x.name)}"${x.name===b.name?' selected':''}>${esc(x.name)} (${esc(M.task_names[x.task])}, 대상 ${x.targets.length}개)</option>`).join('');
  let h=`<div class="lab-row"><label>기준 실험 <select id="lab-base">${bases}</select></label><span class="sub">${esc(b.description)}</span></div>`;
  if(M.explore[task])h+=`<div class="lab-row"><label>빠른 설정</label><button type="button" class="lab-preset" data-preset="explore">탐색용: 대표 대상 ${M.explore[task].targets.length}개 · seed ${M.explore_seeds.length}개 · 짧은 학습</button>`
    +`<button type="button" class="lab-preset" data-preset="full">정식 학습: 대상 전체 · seed ${M.full_seeds.length}개 · 기본 학습량</button></div>`;
  h+=`<div class="lab-row"><label>대상</label>${chips('targets',b.targets.map(t=>[t,t]),F.targets)}`
    +`<button type="button" class="lab-chip" id="lab-all">전체 선택/해제</button></div>`;
  h+=`<div class="lab-row"><label>모델</label>${chips('models',M.models_by_task[task].map(m=>[m,M.model_names[m]]),new Set(Object.keys(F.models)))}`
    +`<span class="sub">기준선(${esc(M.baselines[task])})은 자동으로 함께 학습합니다</span></div>`;
  Object.keys(F.models).forEach(m=>{const space=M.space[task][m]||[];
    h+=`<div class="lab-box"><h4>${esc(M.model_names[m])} 하이퍼파라미터</h4>`+(space.length?
      `<div class="tw"><table class="lab-params"><tr><th>설정</th><th>값 (쉼표로 여러 개 = 조합)</th><th></th><th>기본값 · 설명</th></tr>`+
      space.map(p=>`<tr><td>${esc(p.label)}</td><td><input type="text" data-m="${m}" data-k="${esc(p.key)}" value="${esc(F.models[m][p.key])}" placeholder="${p.kind==='int'||p.kind==='float'?'예: 8, 32':'예: '+esc((p.choices||[]).slice(0,2).join(', '))}"></td>`
        +`<td class="lab-cnt">${badge(count(F.models[m][p.key]))}</td>`
        +`<td class="sub">기본 ${esc(p.default)}${p.choices.length?' · 고를 수 있는 값 '+esc(p.choices.join(', ')):''}${p.help?' · '+esc(p.help):''}</td></tr>`).join('')
      +'</table></div>':'<p class="sub">정할 하이퍼파라미터가 없습니다 (자동 선택).</p>')+'</div>';});
  h+=`<div class="lab-row"><label>조합 방식 <select id="lab-search"><option value="0"${F.search?'':' selected'}>모든 조합</option>`
    +[4,8,16,24,32,48].map(n=>`<option value="${n}"${F.search==n?' selected':''}>무작위 ${n}개</option>`).join('')+`</select></label>`
    +(task!=='thickness'?`<label>window 길이(초) <input type="text" id="lab-windows" value="${esc(F.windows)}" size="10"></label>`:'')
    +`<label>seed <input type="text" id="lab-seeds" value="${esc(F.seeds)}" size="10"></label></div>`;
  h+=`<div class="lab-row"><label>입력 세트</label>${chips('input_sets',Object.entries(M.input_sets),F.input_sets)}</div>`;
  if(task!=='thickness')h+=`<div class="lab-row">`+M.train_fields.map(p=>p.choices.length?
      `<label>${esc(p.label)} <select data-train="${p.key}">${p.choices.map(c=>`<option${String(F.train[p.key]??p.default)===c?' selected':''}>${c}</option>`).join('')}</select></label>`:
      `<label>${esc(p.label)} <input type="number" data-train="${p.key}" value="${esc(F.train[p.key]??p.default)}" style="width:96px"></label>`).join('')+'</div>';
  h+=`<div class="lab-row"><label>GPU</label>${chips('gpus',M.gpus.map(g=>[g,'GPU '+g]),F.gpus)}<span class="sub">비우면 CPU · 다른 계정이 쓰는 GPU는 빼세요(nvidia-smi)</span>`
    +`<label><input type="checkbox" id="lab-smoke"${F.smoke?' checked':''}> 빠른 확인(smoke)</label>`
    +`<label><input type="checkbox" id="lab-report"${F.report?' checked':''}> 완료 후 보고서 파일 갱신</label></div>`;
  h+=`<div class="lab-row"><label style="flex:1">메모 <input type="text" id="lab-memo" value="${esc(F.memo)}" style="flex:1;min-width:240px" placeholder="예: 1차 patch·d_model·학습률 (탐색 결과가 이 이름으로 묶임)"></label>`
    +`<button type="button" class="pl-btn pl-main" id="lab-run">▶ 대기열에 넣고 학습</button></div>`;
  $('lab-form').innerHTML=h;wire();plan();}
function wire(){const f=$('lab-form');
  $('lab-base').onchange=e=>pickBase(e.target.value);
  f.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>preset(b.dataset.preset));
  f.querySelectorAll('[data-chips]').forEach(box=>box.onclick=e=>{const v=e.target.dataset&&e.target.dataset.v;if(v===undefined)return;
    const name=box.dataset.chips;
    if(name==='models'){if(F.models[v])delete F.models[v];else addModel(v);render();return;}
    const s=F[name];s.has(v)?s.delete(v):s.add(v);e.target.classList.toggle('on');plan();});
  $('lab-all').onclick=()=>{const b=base();F.targets=F.targets.size===b.targets.length?new Set():new Set(b.targets);render();};
  f.querySelectorAll('input[data-m]').forEach(i=>i.oninput=()=>{F.models[i.dataset.m][i.dataset.k]=i.value;
    i.closest('tr').querySelector('.lab-cnt').innerHTML=badge(count(i.value));plan();});
  f.querySelectorAll('[data-train]').forEach(i=>i.oninput=i.onchange=()=>{F.train[i.dataset.train]=i.value;plan();});
  $('lab-search').onchange=e=>{F.search=+e.target.value;plan();};
  if($('lab-windows'))$('lab-windows').oninput=e=>{F.windows=e.target.value;plan();};
  $('lab-seeds').oninput=e=>{F.seeds=e.target.value;plan();};
  $('lab-smoke').onchange=e=>{F.smoke=e.target.checked;};$('lab-report').onchange=e=>{F.report=e.target.checked;};
  $('lab-memo').oninput=e=>{F.memo=e.target.value;};
  $('lab-run').onclick=run;}
function body(){return {base:F.base,targets:[...F.targets],models:F.models,search_trials:F.search,input_sets:[...F.input_sets],
  seeds:F.seeds,windows:F.windows,train:F.train,memo:F.memo,gpus:[...F.gpus].sort().join(',')||'cpu',smoke:F.smoke,report:F.report};}
function plan(){clearTimeout(timer);timer=setTimeout(()=>post('/api/plan',body()).then(showPlan),350);}
function showPlan(r){const p=$('lab-preview');F.ok=!r.error;$('lab-run').disabled=!F.ok;
  if(r.error){p.innerHTML=`<p class="lab-err">설정을 확인하세요: ${esc(r.error)}</p>`;return;}
  p.innerHTML=`<div class="tiles">`+[['작업 수',r.jobs+'개','대상 × (기준선 + 조합 × seed), 입력 세트·window별'],['조합',r.combos+'개',F.search?`모델마다 무작위 ${F.search}개까지 (가능한 조합이 적으면 전부)`:'가능한 조합 모두'],
    ['실험',r.experiments+'개','window마다 하나'],['예상 시간',r.estimate,r.estimate_basis||'']].map(t=>
    `<div class="tile"><div class="tl">${t[0]}</div><div class="tv">${esc(t[1])}</div><div class="tn">${esc(t[2])}</div></div>`).join('')+'</div>'
    +(r.warnings||[]).map(w=>`<p class="lab-warn">⚠ ${esc(w)}</p>`).join('')
    +`<details class="more"><summary>실험 YAML</summary><pre class="code">${esc(r.yaml)}</pre></details>`;}
function run(){if(!F.ok)return;$('lab-run').disabled=true;post('/api/submit',body()).then(r=>{
  $('lab-preview').insertAdjacentHTML('afterbegin',r.error?`<p class="lab-err">${esc(r.error)}</p>`:
    `<p class="st good">대기열에 넣었습니다: ${r.ids.map(esc).join(', ')} . 아래 진행 상황에서 확인하세요.</p>`);
  $('lab-run').disabled=false;queue();});}
function queue(){get('/api/fragment/queue').then(html=>{const q=$('lab-queue');const open=[...q.querySelectorAll('details[open]')].map(d=>d.dataset.log);
  const old=q.querySelector('.lab-reqs'),top=old?old.scrollTop:0;
  q.innerHTML=html;const box=q.querySelector('.lab-reqs');if(box)box.scrollTop=top;q.querySelectorAll('[data-cancel]').forEach(b=>b.onclick=()=>{if(confirm('이 요청을 취소할까요?'))post('/api/cancel',{id:b.dataset.cancel}).then(queue);});
  q.querySelectorAll('[data-hide]').forEach(b=>b.onclick=()=>{if(confirm('이 요청을 진행 상황 목록에서 지울까요? 결과는 누적 기록과 탐색 결과에 남습니다.'))
    post('/api/hide',{id:b.dataset.hide}).then(r=>{if(r.error)alert(r.error);queue();});});
  q.querySelectorAll('[data-hide-all]').forEach(b=>b.onclick=()=>{if(confirm('끝난 요청 '+b.dataset.hideAll+'건(완료·실패·취소)을 목록에서 지울까요? 결과는 누적 기록과 탐색 결과에 남습니다.'))
    post('/api/hide-finished',{}).then(r=>{if(r.error)alert(r.error);queue();});});
  q.querySelectorAll('details[data-log]').forEach(d=>{const load=()=>get('/api/log?id='+encodeURIComponent(d.dataset.log)).then(t=>{d.querySelector('pre').textContent=t||'(아직 없음)';});
    d.ontoggle=()=>{if(d.open)load();}; if(open.includes(d.dataset.log)){d.open=true;load();}});}).catch(()=>{});}
function first(){get('/api/fragment/explore').then(html=>{const x=$('lab-explore'),sel=x.querySelector('tr.on[data-xsel]');
  const key=sel&&sel.dataset.xsel;x.innerHTML=html;const row=key&&x.querySelector('tr[data-xsel="'+key+'"]');if(row)row.click();}).catch(()=>{});
  get('/api/fragment/first').then(html=>{const box=$('lab-first');box.innerHTML=html;
  box.querySelectorAll('.lab-actions').forEach(a=>a.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{
    const msg=a.querySelector('.lab-msg');b.disabled=true;msg.textContent='대기열에 넣는 중…';
    post('/api/stage'+b.dataset.stage,{task:a.dataset.task,k:+a.querySelector('.lab-k').value,seeds:a.querySelector('.lab-stage-seeds').value,
      gpus:F?[...F.gpus].sort().join(',')||'cpu':'auto',report:true}).then(r=>{msg.textContent=r.error?('오류: '+r.error):('대기열에 넣었습니다: '+r.ids.join(', '));b.disabled=false;queue();});}));}).catch(()=>{});}
})();
"""
