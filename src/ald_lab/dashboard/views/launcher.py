"""Page 3, tabs ① and ②: start an experiment with chosen hyperparameters, then follow the queue.

The page only writes a request into ``work/queue/`` and wakes the worker (:mod:`ald_lab.launch`); training
runs in that separate process, so closing the browser or restarting the dashboard does not stop it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from ...launch import (
    INPUT_SET_NAMES,
    MODELS_BY_TASK,
    STATUS_NAMES,
    TRAIN_FIELDS,
    LaunchError,
    base_configs,
    base_input_sets,
    base_model_values,
    build_config,
    cancel,
    ensure_worker,
    list_requests,
    log_tail,
    param_space,
    parse_values,
    plan_jobs,
    submit,
    values_text,
    worker_running,
)
from ...ledger import estimate_text
from ...models.cards import model_card
from ..state import lab_paths, ledger
from ..ui import empty_state, kpi_row, table

TASK_LABELS = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서", "thickness": "두께(VM)"}
BASELINE_LABELS = {"forecast": "현재값 유지", "virtual_sensor": "레시피 평균", "thickness": "학습 평균"}
DEEP = ("patchtst", "itransformer", "dlinear", "gru")  # models whose training settings the form shows
STATUS_MARK = {"queued": ":gray[■]", "running": ":blue[■]", "done": ":green[■]", "failed": ":red[■]",
               "cancelled": ":orange[■]"}


def run_enabled() -> bool:
    return os.environ.get("ALD_DASHBOARD_RUN", "1") != "0"


@st.cache_data(show_spinner=False, max_entries=64)
def _plan(yaml_text: str) -> tuple[list[dict], str | None]:
    try:
        jobs = plan_jobs(yaml.safe_load(yaml_text))
    except LaunchError as exc:
        return [], str(exc)
    return [j.to_dict() for j in jobs], None


def _estimate(jobs: list[dict], task: str, n_devices: int) -> str:
    return estimate_text(ledger(), jobs, task, n_devices)


def _busy_runs(work_root: Path) -> list[str]:
    """Experiments being written to right now that the queue did not start (e.g. run_all.sh in a terminal)."""

    queued = {r.get("exp_dir") for r in list_requests(work_root) if r.get("status") == "running"}
    out = []
    now = time.time()
    for run in (work_root / "experiments").glob("*/*"):
        fresh = run.is_dir() and (now - run.stat().st_mtime < 600
                                  or any(now - p.stat().st_mtime < 600 for p in run.glob("jobs/*")))
        if fresh and not (run / "summary.json").is_file() and str(run) not in queued:
            out.append(f"{run.parent.name}/{run.name}")
    return out


def render_launcher() -> None:
    paths = lab_paths()
    if not run_enabled():
        st.info("이 서버에서는 대시보드 실행이 꺼져 있습니다 (scripts/env.sh의 ALD_DASHBOARD_RUN=0).")
        return
    bases = base_configs()
    if not bases:
        empty_state("기준 실험 없음", "configs/experiments/*.yaml 이 없습니다.")
        return
    st.markdown("기준 실험의 데이터·대상 설정을 가져오고, **모델과 하이퍼파라미터만 바꿔** 학습합니다. "
                "칸에 값을 쉼표로 여러 개 넣으면 **모든 조합**을 돌리고, 검증 MAE로 비교합니다.")
    names = [b["name"] for b in bases]
    i = st.selectbox("기준 실험", range(len(bases)), index=names.index("g1_forecast_vg11") if "g1_forecast_vg11" in names else 0,
                     format_func=lambda k: f"{bases[k]['name']} — {TASK_LABELS.get(bases[k]['task'], '')}"
                                           f" · {bases[k]['description'][:50]}")
    base = bases[i]
    doc, task = base["doc"], base["task"]
    c1, c2 = st.columns(2)
    targets = c1.multiselect("대상", list(doc["targets"]), default=list(doc["targets"])[:1], key=f"t_{base['name']}")
    in_base = [m["name"] if isinstance(m, dict) else m for m in doc.get("models") or []]
    options = list(MODELS_BY_TASK[task])
    models = c2.multiselect("모델", options, default=[m for m in ("lightgbm",) if m in options],
                            format_func=lambda m: model_card(m)["name"] + ("" if m in in_base else " (기준 실험에 없음)"),
                            key=f"m_{base['name']}")
    st.caption(f"기준선({BASELINE_LABELS[task]})은 자동으로 함께 돌아 모든 모델의 개선율 기준이 됩니다.")

    raw: dict[str, dict[str, object]] = {}
    for model in models:
        space = param_space(task, model)
        defaults = base_model_values(doc, model)
        with st.expander(f"{model_card(model)['name']} 하이퍼파라미터", expanded=True):
            if not space:
                st.caption("정할 하이퍼파라미터가 없습니다 (규제 강도는 검증으로 자동 선택).")
            cols = st.columns(4)
            values: dict[str, object] = {}
            for k, p in enumerate(space):
                col, key = cols[k % 4], f"p_{base['name']}_{model}_{p.key}"
                start = defaults.get(p.key, [p.default])
                if p.kind in ("choice", "bool"):
                    opts = list(p.choices) if p.kind == "choice" else [True, False]
                    values[p.key] = col.multiselect(p.label, opts, default=[v for v in start if v in opts] or [p.default],
                                                    key=key, help=p.help or None)
                else:
                    values[p.key] = col.text_input(p.label, values_text(start), key=key,
                                                   help=(p.help + " · " if p.help else "") + "쉼표로 여러 값 → 모든 조합")
            raw[model] = values

    st.markdown("**공통 설정**")
    c = st.columns(4)
    set_options = list(INPUT_SET_NAMES)
    start_sets = [s for s in base_input_sets(doc) if s in set_options] or ["all"]
    input_sets = c[0].multiselect("입력 세트", set_options, default=start_sets,
                                  format_func=lambda s: f"{s} — {INPUT_SET_NAMES[s]}", key=f"s_{base['name']}")
    seeds_text = c[1].text_input("seed", values_text(doc.get("seeds") or [42]), key=f"seed_{base['name']}",
                                 help="여러 개면 seed마다 다시 학습해 흔들림을 봅니다")
    context = None
    train: dict[str, object] = {}
    if task != "thickness":
        start_context = float((doc.get("window") or {}).get("context_seconds", 192))
        context = c[2].number_input("window 길이(초)", min_value=8.0, max_value=3600.0, value=start_context, step=8.0,
                                    format="%.0f", key=f"w_{base['name']}", help="모델이 한 번에 보는 과거 길이")
        if any(m in DEEP for m in models):
            base_train = doc.get("train") or {}
            tc = st.columns(len(TRAIN_FIELDS))
            for col, p in zip(tc, TRAIN_FIELDS):
                value = base_train.get(p.key, p.default)
                key = f"tr_{base['name']}_{p.key}"
                if p.kind == "choice":
                    train[p.key] = col.selectbox(p.label, list(p.choices), index=list(p.choices).index(value)
                                                 if value in p.choices else 0, key=key)
                else:
                    train[p.key] = int(col.number_input(p.label, min_value=1, value=int(value), step=1, key=key))

    st.markdown("**실행**")
    r = st.columns([2, 1, 1, 3])
    gpu_choice = r[0].multiselect("GPU", ["0", "1", "2", "3"], default=["0", "1", "2", "3"], key="gpus",
                                  help="비우면 CPU로 학습합니다")
    smoke = r[1].checkbox("빠른 확인(smoke)", help="run 몇 개·epoch 2로 끝까지 도는지만 확인")
    report = r[2].checkbox("끝나면 보고서 갱신", value=True)
    memo = r[3].text_input("메모 (누적 기록에 남음)", placeholder="예: LightGBM 잎 수·학습률 조합")
    gpus = ",".join(gpu_choice) if gpu_choice else "cpu"

    try:
        model_values = {m: {p.key: parse_values(p, raw[m][p.key]) for p in param_space(task, m)} for m in models}
        seeds = [int(s) for s in seeds_text.replace(";", ",").split(",") if s.strip()]
        cfg = build_config(doc, targets=targets, models=model_values, input_sets=input_sets, seeds=seeds,
                           context_seconds=context, train=train or None, memo=memo)
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        jobs, error = _plan(text)
    except (LaunchError, ValueError) as exc:
        cfg, text, jobs, error = None, "", [], str(exc)
    if error:
        st.error(f"설정을 확인하세요: {error}")
    else:
        trained = [j for j in jobs if j["model"] in models]
        combos = len({(j["label"], j["input_set"]) for j in trained})
        kpi_row([
            ("작업 수", f"{len(jobs)}개", "대상 × 모델 × 조합 × 입력 세트 × seed (기준선 포함)"),
            ("조합", f"{combos}개", "모델·하이퍼파라미터·입력 세트 조합 (대상·seed 제외)"),
            ("예상 시간", _estimate(jobs, task, 1 if gpus == "cpu" else len(gpu_choice)),
             "같은 과제·모델의 지난 학습 시간 중앙값으로 어림"),
            ("저장 위치", f"{cfg['name']}{'_smoke' if smoke else ''}", "work/experiments/ 아래 (보고서 본문과 별도)"),
        ])
        with st.expander("돌릴 작업 목록 · 실험 YAML"):
            table(pd.DataFrame([{"대상": j["target"], "모델": j["label"], "입력 세트": j["input_set"], "seed": j["seed"]}
                                for j in jobs]), height=260)
            st.code(text, language="yaml")
    busy = _busy_runs(paths.work_root)
    if busy:
        st.warning("터미널에서 돌고 있는 실험이 있어 GPU를 함께 쓰게 됩니다: " + ", ".join(busy))
    if st.button("▶ 대기열에 넣고 학습 시작", type="primary", disabled=bool(error) or cfg is None):
        record = submit(paths.work_root, cfg, gpus=gpus, smoke=smoke, report=report, data_root=paths.data_root,
                        n_jobs=len(jobs))
        started = ensure_worker(paths.work_root)
        st.success(f"요청 {record['id']}를 대기열에 넣었습니다 (작업 {len(jobs)}개). "
                   + ("작업자를 시작했습니다. " if started else "앞 요청이 끝나면 이어서 돕니다. ")
                   + "② 진행 상황 탭에서 확인하세요.")


def _request_card(work_root: Path, r: dict) -> None:
    with st.container(border=True):
        top = st.columns([3, 2, 2, 1])
        status = r.get("status", "")
        top[0].markdown(f"{STATUS_MARK.get(status, '')} **{STATUS_NAMES.get(status, status)}** · `{r['id']}`  \n"
                        f"{r.get('memo') or r['name']}")
        top[1].caption(f"{r['name']}  \n대상 {', '.join(r.get('targets') or [])}")
        times = [f"요청 {r['created'][5:16].replace('T', ' ')}"]
        if r.get("started"):
            times.append(f"시작 {r['started'][11:16]}")
        if r.get("finished"):
            times.append(f"끝 {r['finished'][11:16]}")
        top[2].caption(" · ".join(times) + f"  \nGPU {r.get('gpus')}" + (" · smoke" if r.get("smoke") else ""))
        if status in ("queued", "running") and top[3].button("취소", key=f"cancel_{r['id']}"):
            st.toast(f"{r['id']}: {cancel(work_root, r['id'])}")
        progress = r.get("progress")
        if progress and progress[1]:
            done, total = progress
            st.progress(min(1.0, done / total), text=f"작업 {done} / {total} 끝남")
        if status != "queued":
            with st.expander("학습 로그 (마지막 40줄)", expanded=status == "failed"):
                st.code(log_tail(work_root, r["id"]) or "(아직 없음)", language=None)


def render_queue() -> None:
    work_root = lab_paths().work_root
    requests = list_requests(work_root)
    active = any(r.get("status") in ("queued", "running") for r in requests)

    def panel() -> None:
        reqs = list_requests(work_root)
        running = worker_running(work_root)
        st.caption(("작업자 실행 중" if running else "작업자 쉬는 중") + " · 5초마다 새로 고침"
                   if any(r.get("status") in ("queued", "running") for r in reqs) else
                   "실행 중인 요청 없음")
        if not reqs:
            empty_state("요청 없음", "① 새 실험 실행 탭에서 설정을 고르고 ▶ 버튼을 누르면 여기에 나타납니다.")
            return
        for r in reqs[:20]:
            _request_card(work_root, r)
        stalled = any(r.get("status") == "queued" for r in reqs) and not running
        if stalled and st.button("작업자 다시 시작"):
            ensure_worker(work_root)

    st.fragment(panel, run_every=5 if active else None)()
