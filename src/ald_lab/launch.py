"""Run experiments on request: a file queue under ``work/queue/`` and one worker process that trains them in order.

The dashboard (or ``ald-queue submit``) turns chosen hyperparameters into an ordinary experiment YAML, puts it
in the queue and makes sure a worker is running. The worker runs ``ald-train`` for one request at a time
(the GPUs are never shared by two requests), writes progress into the request and, when asked, rebuilds the
HTML report. Results land in ``work/experiments/<base>__hp/<stamp>/`` like any other run, so the cumulative
records (:mod:`ald_lab.ledger`) and the report pick them up.

Nothing here imports torch at module level; validating a request does (through the experiment loader).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .models import defaults
from .paths import PROJECT_ROOT, resolve_paths

EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
HP_SUFFIX = "__hp"  # experiments started from the dashboard; the report's main tabs leave them out
STATUSES = ("queued", "running", "done", "failed", "cancelled")
STATUS_NAMES = {"queued": "대기", "running": "실행 중", "done": "완료", "failed": "실패", "cancelled": "취소"}
INPUT_SET_NAMES = {
    "all": "전체 (상수·중복 제외)", "core": "중요도 A 대표", "core_ab": "중요도 A·B 대표",
    "legacy_primary": "이전 프로젝트 37개", "A_all": "중요도 A 전체", "A_plus": "A 전체 + 압력계·APC·RF 매칭",
}
_LOCK_NAME = "worker.lock"


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # int | float | choice | bool
    default: Any
    choices: tuple[Any, ...] = ()
    help: str = ""


def _deep(name: str, key: str) -> Any:
    return defaults.DEEP[name][key]


_LR = Param("train.lr", "학습률", "float", 1e-3, help="AdamW 학습률 (이 모델에만 적용)")
PARAM_SPACE: dict[str, list[Param]] = {
    "lightgbm": [
        Param("num_leaves", "잎 수", "int", defaults.LIGHTGBM["num_leaves"], help="나무 하나의 복잡도"),
        Param("learning_rate", "학습률", "float", defaults.LIGHTGBM["learning_rate"]),
        Param("min_child_samples", "잎 최소 샘플", "int", defaults.LIGHTGBM["min_child_samples"]),
        Param("n_estimators", "나무 수 상한", "int", defaults.LIGHTGBM["n_estimators"], help="검증 손실로 조기 종료"),
        Param("subsample", "행 샘플 비율", "float", defaults.LIGHTGBM["subsample"]),
        Param("colsample_bytree", "특징 샘플 비율", "float", defaults.LIGHTGBM["colsample_bytree"]),
        Param("objective", "손실", "choice", defaults.LIGHTGBM["objective"], ("l2", "l1", "huber")),
    ],
    "ridge": [],
    "patchtst": [
        Param("channel_mode", "신호 처리", "choice", _deep("patchtst", "channel_mode"), ("independent", "mixing"),
              help="mixing = 신호 혼합(입력 신호를 함께 봄) · independent = 대상 신호의 과거만 봄(다른 입력은 안 쓰고 "
                   "GPU 메모리를 많이 씀)"),
        Param("patch_seconds", "patch 길이(초)", "int", _deep("patchtst", "patch_len")),
        Param("stride_seconds", "patch 간격(초)", "int", _deep("patchtst", "stride")),
        Param("d_model", "표현 크기 d_model", "int", _deep("patchtst", "d_model")),
        Param("n_heads", "attention head 수", "int", _deep("patchtst", "n_heads"), help="d_model의 약수"),
        Param("n_layers", "층 수", "int", _deep("patchtst", "n_layers")),
        Param("d_ff", "feed-forward 크기", "int", _deep("patchtst", "d_ff")),
        Param("dropout", "dropout", "float", _deep("patchtst", "dropout")),
        Param("revin", "RevIN 정규화", "bool", True),
        Param("revin_center", "RevIN 기준", "choice", "mean", ("mean", "last")),
        _LR,
    ],
    "itransformer": [
        Param("d_model", "표현 크기 d_model", "int", _deep("itransformer", "d_model")),
        Param("n_heads", "attention head 수", "int", _deep("itransformer", "n_heads"), help="d_model의 약수"),
        Param("n_layers", "층 수", "int", _deep("itransformer", "n_layers")),
        Param("d_ff", "feed-forward 크기", "int", _deep("itransformer", "d_ff")),
        Param("dropout", "dropout", "float", _deep("itransformer", "dropout")),
        _LR,
    ],
    "dlinear": [
        Param("kernel_seconds", "이동평균 길이(초)", "int", _deep("dlinear", "kernel")),
        Param("d_rep", "표현 크기", "int", _deep("dlinear", "d_rep")),
        _LR,
    ],
    "gru": [
        Param("hidden", "은닉 크기", "int", _deep("gru", "hidden")),
        Param("n_layers", "층 수", "int", _deep("gru", "n_layers")),
        Param("dropout", "dropout", "float", _deep("gru", "dropout")),
        _LR,
    ],
}
# Run-level thickness models take different settings (about 60 runs).
VM_PARAM_SPACE: dict[str, list[Param]] = {
    "lightgbm": [
        Param("num_leaves", "잎 수", "int", defaults.VM_LIGHTGBM["num_leaves"]),
        Param("learning_rate", "학습률", "float", defaults.VM_LIGHTGBM["learning_rate"]),
        Param("min_child_samples", "잎 최소 샘플", "int", defaults.VM_LIGHTGBM["min_child_samples"]),
        Param("n_estimators", "나무 수", "int", defaults.VM_LIGHTGBM["n_estimators"]),
    ],
    "pls": [Param("n_components", "성분 수", "int", defaults.VM_PLS["n_components"])],
    "ridge": [],
}
MODELS_BY_TASK = {
    "forecast": ("ridge", "lightgbm", "patchtst", "itransformer", "dlinear", "gru"),
    "virtual_sensor": ("ridge", "lightgbm", "patchtst", "itransformer", "dlinear", "gru"),
    "thickness": ("ridge", "pls", "lightgbm"),
}
DEEP_MODELS = ("patchtst", "itransformer", "dlinear", "gru")
TRAIN_FIELDS = (  # experiment-wide settings shown in the form (deep models)
    Param("max_epochs", "최대 epoch", "int", 60),
    Param("patience", "조기 종료 patience", "int", 8),
    Param("windows_per_epoch", "epoch당 window", "int", 200_000),
    Param("batch_size", "batch 크기", "int", 1024),
    Param("loss", "손실 함수", "choice", "mse", ("mse", "mae", "huber")),
)


class LaunchError(ValueError):
    """Raised for a request that cannot be built or run."""


def param_space(task: str, model: str) -> list[Param]:
    return (VM_PARAM_SPACE if task == "thickness" else PARAM_SPACE).get(model, [])


def _cast(param: Param, text: Any) -> Any:
    if param.kind == "bool":
        if isinstance(text, bool):
            return text
        value = str(text).strip().lower()
        if value in ("true", "1", "yes", "on", "켬"):
            return True
        if value in ("false", "0", "no", "off", "끔"):
            return False
        raise LaunchError(f"{param.label}: true 또는 false만 입력할 수 있습니다: {text!r}")
    if param.kind == "choice":
        value = str(text).strip()
        if value not in [str(c) for c in param.choices]:
            raise LaunchError(f"{param.label}: {', '.join(map(str, param.choices))} 중 하나여야 합니다: {text!r}")
        return value
    try:
        number = float(str(text).strip())
    except ValueError as exc:
        raise LaunchError(f"{param.label}: 숫자가 아닙니다: {text!r}") from exc
    if param.kind == "int":
        if number != int(number):
            raise LaunchError(f"{param.label}: 정수여야 합니다: {text!r}")
        return int(number)
    return number


def parse_values(param: Param, text: Any) -> list[Any]:
    """Values the user typed for one setting: ``"64, 128"`` → ``[64, 128]`` (several values = every combination)."""

    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [t for t in str(text).replace(";", ",").split(",") if t.strip()]
    if not items:
        raise LaunchError(f"{param.label}: 값을 하나 이상 넣으세요")
    values = [_cast(param, t) for t in items]
    return list(dict.fromkeys(values))


def values_text(values: Any) -> str:
    return ", ".join(str(v) for v in (values if isinstance(values, list) else [values]))


def base_configs() -> list[dict[str, Any]]:
    """The experiment YAMLs in ``configs/experiments/`` a request can start from."""

    out = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if doc.get("task") and doc.get("name"):
            out.append({"path": path, "name": str(doc["name"]), "task": doc["task"], "doc": doc,
                        "description": str(doc.get("description") or "")})
    return out


def base_model_values(doc: dict[str, Any], model: str) -> dict[str, list[Any]]:
    """The first entry for ``model`` in a base YAML: its params and grid as value lists (form defaults)."""

    for item in doc.get("models") or []:
        item = {"name": item} if isinstance(item, str) else item
        if item.get("name") == model:
            values = {k: [v] for k, v in (item.get("params") or {}).items()}
            values.update({k: list(v) for k, v in (item.get("grid") or {}).items()})
            return values
    return {}


def base_input_sets(doc: dict[str, Any]) -> list[str]:
    data = doc.get("data") or {}
    if data.get("input_sets"):
        return [str(v) for v in data["input_sets"].values() if isinstance(v, str)]
    inputs = data.get("inputs", "all")
    return [inputs] if isinstance(inputs, str) else ["all"]


def build_config(base: dict[str, Any], *, targets: list[str], models: dict[str, dict[str, list[Any]]],
                 input_sets: list[str], seeds: list[int], context_seconds: float | None = None,
                 train: dict[str, Any] | None = None, memo: str = "", search_trials: int | None = None) -> dict[str, Any]:
    """An experiment YAML (as a dict) from a base experiment and the chosen settings.

    ``models`` maps a model name to ``{setting: [values]}``: one value is fixed, several become a grid — every
    combination, or ``search_trials`` combinations drawn at random when given.
    """

    doc = copy.deepcopy(base)
    task = doc["task"]
    if not targets:
        raise LaunchError("대상을 하나 이상 고르세요")
    if not models:
        raise LaunchError("모델을 하나 이상 고르세요")
    if not input_sets:
        raise LaunchError("입력 세트를 하나 이상 고르세요")
    if not seeds:
        raise LaunchError("seed를 하나 이상 넣으세요")
    doc["name"] = f"{str(base['name']).removesuffix(HP_SUFFIX)}{HP_SUFFIX}"
    doc["description"] = memo.strip() or f"대시보드 실행: {base['name']} 설정 변경"
    doc["targets"] = list(targets)
    doc["seeds"] = [int(s) for s in seeds]
    data = dict(doc.get("data") or {})
    data.pop("inputs", None)
    data.pop("input_sets", None)
    if len(input_sets) == 1:
        data["inputs"] = input_sets[0]
    else:
        data["input_sets"] = {s: s for s in input_sets}
    doc["data"] = data
    if context_seconds and task != "thickness":
        doc["window"] = {**(doc.get("window") or {}), "context_seconds": float(context_seconds)}
    if train and task != "thickness":
        doc["train"] = {**(doc.get("train") or {}), **train}
    entries = []
    for name, chosen in models.items():
        if name not in MODELS_BY_TASK[task]:
            raise LaunchError(f"{name}: 이 과제({task})에서 쓸 수 없는 모델입니다")
        params, grid = {}, {}
        for key, values in chosen.items():
            values = list(values)
            if len(values) == 1:
                params[key] = values[0]
            elif values:
                grid[key] = values
        entry: dict[str, Any] = {"name": name}
        if params:
            entry["params"] = params
        if grid:
            entry["grid"] = grid
            if search_trials:
                entry["search"] = {"trials": int(search_trials), "seed": 0}
        entries.append(entry)
    doc["models"] = entries
    return doc


def docs_from_form(form: dict[str, Any]) -> list[dict[str, Any]]:
    """Experiments for one request of the HTML dashboard: one per window length typed in.

    ``form``: ``base`` (experiment name), ``targets``, ``models`` (``{model: {setting: "64, 128" or [values]}}``),
    ``search_trials`` (0 = every combination), ``input_sets``, ``seeds`` ("42, 43"), ``windows`` ("64, 192"),
    ``train`` (``{setting: value}``), ``memo``.
    """

    bases = {b["name"]: b for b in base_configs()}
    if form.get("base") not in bases:
        raise LaunchError(f"없는 기준 실험: {form.get('base')}")
    base = bases[form["base"]]
    task = base["task"]
    models = {}
    for name, raw in (form.get("models") or {}).items():
        space = {p.key: p for p in param_space(task, name)}
        models[name] = {key: parse_values(space[key], text) for key, text in (raw or {}).items() if key in space}
    seeds = [int(x) for x in str(form.get("seeds") or "42").replace(";", ",").split(",") if x.strip()]
    train = {}
    for p in TRAIN_FIELDS:
        value = (form.get("train") or {}).get(p.key)
        if value not in (None, ""):
            train[p.key] = parse_values(p, [value])[0]
    windows: list[float | None] = [None]
    if task != "thickness":
        text = str(form.get("windows") or "").replace(";", ",")
        windows = [float(w) for w in text.split(",") if w.strip()] or [None]
        if any(w is not None and w <= 0 for w in windows):
            raise LaunchError("window 길이는 0보다 커야 합니다")
    trials = int(form.get("search_trials") or 0) or None
    memo = str(form.get("memo") or "")
    docs = []
    for window in windows:
        note = " · ".join(x for x in (memo, f"window {window:g}초" if window is not None and len(windows) > 1 else "") if x)
        docs.append(build_config(base["doc"], targets=list(form.get("targets") or []), models=models,
                                 input_sets=list(form.get("input_sets") or []), seeds=seeds, context_seconds=window,
                                 train=train or None, memo=note, search_trials=trials))
    return docs


def check_combinations(doc: dict[str, Any]) -> list[str]:
    """Settings the models would reject, found before anything is queued (a search drops them by itself)."""

    from .models.constraints import combo_problems

    problems = []
    context = float((doc.get("window") or {}).get("context_seconds", 192))
    for item in doc.get("models") or []:
        item = {"name": item} if isinstance(item, str) else item
        if item.get("search"):
            continue
        params, grid = item.get("params") or {}, item.get("grid") or {}
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)) or [()]:
            problems += combo_problems(item["name"], {**params, **dict(zip(keys, combo))}, context)
    return list(dict.fromkeys(problems))


def plan_jobs(doc: dict[str, Any]) -> list[Any]:
    """The jobs a request would run (same expansion as ``ald-train``); raises on an invalid YAML."""

    from .training.config import ConfigError, expand_jobs, load_experiment

    problems = check_combinations(doc)
    if problems:
        raise LaunchError(" · ".join(problems))
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"ald_plan_{uuid.uuid4().hex}.yaml"
    tmp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        return expand_jobs(load_experiment(tmp))
    except (ConfigError, ValueError, KeyError) as exc:
        raise LaunchError(str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- queue

def queue_dir(work_root: Path) -> Path:
    return Path(work_root) / "queue"


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_request(folder: Path) -> dict[str, Any] | None:
    path = folder / "request.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def update_request(folder: Path, **fields: Any) -> dict[str, Any]:
    doc = read_request(folder) or {}
    doc.update(fields)
    _write_json(folder / "request.json", doc)
    return doc


def submit(work_root: Path, doc: dict[str, Any], *, gpus: str = "auto", smoke: bool = False, report: bool = False,
           data_root: Path | None = None, n_jobs: int | None = None, source: str = "dashboard") -> dict[str, Any]:
    """Put an experiment in the queue; returns the request record."""

    work_root = Path(work_root)
    now = _now()  # ids sort in submission order (to the millisecond); the queue runs oldest first
    request_id = f"{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}{uuid.uuid4().hex[:2]}"
    time.sleep(0.002)
    folder = queue_dir(work_root) / request_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "experiment.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")
    record = {
        "id": request_id, "status": "queued", "created": _now().isoformat(timespec="seconds"),
        "name": doc["name"] + ("_smoke" if smoke else ""), "task": doc.get("task"), "memo": doc.get("description", ""),
        "targets": doc.get("targets"), "models": [m["name"] if isinstance(m, dict) else m for m in doc["models"]],
        "n_jobs": n_jobs, "gpus": gpus, "smoke": smoke, "report": report, "source": source,
        "work_root": str(work_root), "data_root": str(data_root) if data_root else None,
        "pid": None, "exp_dir": None, "returncode": None, "started": None, "finished": None,
    }
    _write_json(folder / "request.json", record)
    return record


def list_requests(work_root: Path, *, include_hidden: bool = False) -> list[dict[str, Any]]:
    """Every request, newest first, with progress (finished jobs / jobs) for started ones. Requests removed from the
    list (:func:`hide_request`) are left out unless ``include_hidden``."""

    out = []
    root = queue_dir(Path(work_root))
    if not root.is_dir():
        return out
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        doc = read_request(folder)
        if not doc or (doc.get("hidden") and not include_hidden):
            continue
        doc["folder"] = str(folder)
        exp_dir = Path(doc["exp_dir"]) if doc.get("exp_dir") else None
        if exp_dir and exp_dir.is_dir():
            total = len(json.loads((exp_dir / "jobs.json").read_text(encoding="utf-8"))) \
                if (exp_dir / "jobs.json").is_file() else doc.get("n_jobs")
            done = sum(1 for _ in (exp_dir / "jobs").glob("*/metrics_validation.json"))
            doc["progress"] = (done, total)
        out.append(doc)
    return out


FINISHED = ("done", "failed", "cancelled")


def hide_request(work_root: Path, request_id: str) -> str:
    """Take a finished request off the progress list. Its results stay (누적 기록, 탐색 결과, report); only the card goes."""

    folder = queue_dir(Path(work_root)) / request_id
    doc = read_request(folder) if folder.is_dir() and request_id and "/" not in request_id else None
    if not doc:
        raise LaunchError(f"없는 요청: {request_id}")
    if doc.get("status") not in FINISHED:
        raise LaunchError("대기 중이거나 실행 중인 요청은 먼저 취소하세요")
    update_request(folder, hidden=True)
    return "hidden"


def hide_finished(work_root: Path) -> int:
    """Take every finished request (완료 · 실패 · 취소) off the progress list; returns how many."""

    hidden = 0
    for doc in list_requests(work_root):
        if doc.get("status") in FINISHED:
            update_request(Path(doc["folder"]), hidden=True)
            hidden += 1
    return hidden


def cancel(work_root: Path, request_id: str) -> str:
    """Cancel a queued request, or stop a running one (its whole process group)."""

    folder = queue_dir(Path(work_root)) / request_id
    doc = read_request(folder)
    if doc is None:
        raise LaunchError(f"없는 요청: {request_id}")
    if doc["status"] == "queued":
        update_request(folder, status="cancelled", finished=_now().isoformat(timespec="seconds"))
        return "cancelled"
    if doc["status"] == "running":
        update_request(folder, cancel_requested=True)
        if doc.get("pid"):
            try:
                os.killpg(int(doc["pid"]), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        return "stopping"
    return doc["status"]


def log_tail(work_root: Path, request_id: str, lines: int = 40) -> str:
    path = queue_dir(Path(work_root)) / request_id / "train.log"
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 64_000))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.replace("\r", "\n").splitlines()[-lines:])


# ----------------------------------------------------------------------------- worker

def _now() -> datetime:
    return datetime.now().astimezone()


def _lock(work_root: Path):
    """An exclusive lock on the queue folder, or None when another worker holds it."""

    import fcntl  # Linux server only

    root = queue_dir(work_root)
    root.mkdir(parents=True, exist_ok=True)
    handle = open(root / _LOCK_NAME, "a+")  # noqa: SIM115 (held for the worker's lifetime)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def worker_running(work_root: Path) -> bool:
    try:
        handle = _lock(Path(work_root))
    except ImportError:
        return False
    if handle is None:
        return True
    handle.close()
    return False


def ensure_worker(work_root: Path) -> bool:
    """Start a worker in the background unless one is running; True when one was started."""

    work_root = Path(work_root)
    if worker_running(work_root):
        return False
    log = (queue_dir(work_root) / "worker.log").open("ab")
    subprocess.Popen([sys.executable, "-m", "ald_lab.launch", "worker", "--work-root", str(work_root)],
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(PROJECT_ROOT),
                     start_new_session=True, env=os.environ.copy())
    log.close()
    return True


def _next(work_root: Path) -> Path | None:
    root = queue_dir(work_root)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        doc = read_request(folder)
        if doc and doc.get("status") == "queued":
            return folder
    return None


def _new_run(experiments: Path, name: str, before: set[str]) -> Path | None:
    folder = experiments / name
    if not folder.is_dir():
        return None
    fresh = sorted(p for p in folder.iterdir() if p.is_dir() and p.name not in before)
    return fresh[-1] if fresh else None


_TRAIN = "import sys; from ald_lab.cli import train_main; sys.exit(train_main(sys.argv[1:]))"
_REPORT = "import sys; from ald_lab.cli import report_main; sys.exit(report_main(sys.argv[1:]))"


def run_request(folder: Path, *, poll: float = 3.0) -> dict[str, Any]:
    """Train one request (blocking) and record the outcome in its request.json."""

    doc = read_request(folder) or {}
    work_root = Path(doc["work_root"])
    experiments = work_root / "experiments"
    before = {p.name for p in (experiments / doc["name"]).iterdir()} if (experiments / doc["name"]).is_dir() else set()
    cmd = [sys.executable, "-c", _TRAIN, str(folder / "experiment.yaml"), "--gpus", str(doc.get("gpus") or "auto"),
           "--work-root", str(work_root)]
    if doc.get("data_root"):
        cmd += ["--data-root", doc["data_root"]]
    if doc.get("smoke"):
        cmd.append("--smoke")
    started = _now().isoformat(timespec="seconds")
    with (folder / "train.log").open("ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(PROJECT_ROOT), start_new_session=True, env=os.environ.copy())
        update_request(folder, status="running", started=started, pid=proc.pid)
        while proc.poll() is None:
            time.sleep(poll)
            current = read_request(folder) or {}
            if not current.get("exp_dir"):
                run = _new_run(experiments, doc["name"], before)
                if run is not None:
                    update_request(folder, exp_dir=str(run))
    run = _new_run(experiments, doc["name"], before)
    current = read_request(folder) or {}
    status = "cancelled" if current.get("cancel_requested") else ("done" if proc.returncode == 0 else "failed")
    record = update_request(folder, status=status, returncode=proc.returncode, pid=None,
                            exp_dir=str(run) if run else current.get("exp_dir"),
                            finished=_now().isoformat(timespec="seconds"))
    if status == "done" and doc.get("report"):
        note = f"대시보드 실행 완료: {doc.get('memo') or doc['name']}"
        with (folder / "report.log").open("ab") as log:
            code = subprocess.call([sys.executable, "-c", _REPORT, "--work-root", str(work_root), "--note", note],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), env=os.environ.copy())
        record = update_request(folder, report_returncode=code)
    return record


def run_worker(work_root: Path) -> int:
    """Run queued requests one after another until the queue is empty (one worker at a time)."""

    work_root = Path(work_root)
    handle = _lock(work_root)
    if handle is None:
        return 0
    try:
        for folder in sorted(p for p in queue_dir(work_root).iterdir() if p.is_dir()):
            doc = read_request(folder)  # a worker that died mid-request leaves it "running": mark it
            if doc and doc.get("status") == "running":
                update_request(folder, status="failed", pid=None, error="작업자가 중간에 멈춤")
        count = 0
        while (folder := _next(work_root)) is not None:
            run_request(folder)
            count += 1
        return count
    finally:
        handle.close()


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실험 대기열: 대시보드에서 넣은 실험을 차례로 학습합니다.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_submit = sub.add_parser("submit", help="실험 YAML을 대기열에 넣고 작업자를 깨운다")
    p_submit.add_argument("config")
    p_submit.add_argument("--gpus", default="auto")
    p_submit.add_argument("--smoke", action="store_true")
    p_submit.add_argument("--report", action="store_true", help="끝나면 보고서를 다시 만든다")
    sub.add_parser("list", help="요청 목록")
    p_cancel = sub.add_parser("cancel", help="요청 취소 (실행 중이면 멈춤)")
    p_cancel.add_argument("request_id")
    sub.add_parser("worker", help="대기열이 빌 때까지 실행 (보통 자동으로 뜬다)")
    p_first = sub.add_parser("first", help="모델 선정 단계: 대상별 PatchTST 최고 · 기준선 대비 판정 (LightGBM 참고)")
    p_stage2 = sub.add_parser("stage2", help="1단계 탐색 상위 조합을 모든 대상에 (대기열에 넣음)")
    p_stage3 = sub.add_parser("stage3", help="PatchTST가 기준을 못 넘은 대상에 다른 모델 (대기열에 넣음)")
    for p in (p_first, p_stage2, p_stage3):
        p.add_argument("--task", required=True, choices=("forecast", "virtual_sensor"))
    for p, seeds in ((p_stage2, "42,43,44"), (p_stage3, "42,43")):
        p.add_argument("--seeds", default=seeds)
        p.add_argument("--gpus", default="auto")
        p.add_argument("--report", action="store_true")
    p_stage2.add_argument("--k", type=int, default=3, help="상위 몇 개 조합 (기본 3)")
    for p in (p_submit, *sub.choices.values()):
        if "--work-root" not in p._option_string_actions:
            p.add_argument("--work-root")
    args = parser.parse_args(argv)
    paths = resolve_paths(work_root=args.work_root)
    if args.command == "submit":
        doc = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        jobs = plan_jobs(doc)
        record = submit(paths.work_root, doc, gpus=args.gpus, smoke=args.smoke, report=args.report,
                        data_root=paths.data_root, n_jobs=len(jobs), source="terminal")
        started = ensure_worker(paths.work_root)
        print(f"대기열에 넣음: {record['id']} (작업 {len(jobs)}개)" + (" · 작업자 시작" if started else ""))
    elif args.command == "list":
        for r in list_requests(paths.work_root):
            progress = r.get("progress")
            done = f" {progress[0]}/{progress[1]}" if progress else ""
            print(f"{r['id']}  {STATUS_NAMES.get(r['status'], r['status'])}{done}  {r['name']}  {r.get('memo', '')}")
    elif args.command == "cancel":
        print(cancel(paths.work_root, args.request_id))
    elif args.command == "worker":
        print(f"요청 {run_worker(paths.work_root)}개 처리")
    else:
        _stage_command(args, paths)
    return 0


def stage_base(task: str) -> dict[str, Any]:
    """The experiment whose targets and data settings steps 2 and 3 use (every target of the task)."""

    from .stages import BASE_FOR_TASK

    for base in base_configs():
        if base["name"] == BASE_FOR_TASK[task]:
            return base["doc"]
    raise LaunchError(f"{BASE_FOR_TASK[task]}.yaml이 configs/experiments/에 없습니다")


def submit_docs(work_root: Path, docs: list[dict[str, Any]], *, gpus: str, report: bool,
                data_root: Path | None, source: str, smoke: bool = False) -> list[dict[str, Any]]:
    """Validate and queue several experiments, then wake the worker."""

    planned = [(doc, plan_jobs(doc)) for doc in docs]  # every one valid before any is queued
    records = [submit(work_root, doc, gpus=gpus, report=report, data_root=data_root, n_jobs=len(jobs),
                      source=source, smoke=smoke) for doc, jobs in planned]
    if records:
        ensure_worker(work_root)
    return records


def _stage_command(args: argparse.Namespace, paths: Any) -> None:
    import pandas as pd

    from .ledger import collect_ledger
    from .stages import (
        DECISIONS,
        patchtst_first,
        search_ranking,
        stage2_configs,
        stage3_config,
    )

    records = collect_ledger(paths.work_root)
    if args.command == "first":
        frame = patchtst_first(records, args.task)
        def num(value: Any) -> str:
            return f"{value:.4g}" if value is not None and pd.notna(value) else "-"

        for r in frame.itertuples():
            print(f"{r.target:16s} PatchTST {num(r.patch_mae):>9s}  기준선 {num(r.baseline_mae):>9s}  "
                  f"{DECISIONS[r.decision]} {r.reason}  (LightGBM {num(r.lgbm_mae)} 참고)")
        return
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    base = stage_base(args.task)
    if args.command == "stage2":
        docs = stage2_configs(search_ranking(records, args.task), base, k=args.k, seeds=seeds)
        if not docs:
            raise LaunchError("1단계 탐색(p1_patchtst_search_*) 결과가 아직 없습니다")
    else:
        doc = stage3_config(patchtst_first(records, args.task), base, seeds=seeds)
        if doc is None:
            print("대안이 필요한 대상이 없습니다 (모든 대상에서 PatchTST 채택)")
            return
        docs = [doc]
    for r in submit_docs(paths.work_root, docs, gpus=args.gpus, report=args.report, data_root=paths.data_root,
                         source="terminal"):
        print(f"대기열에 넣음: {r['id']} {r['name']} (작업 {r['n_jobs']}개)")


if __name__ == "__main__":
    raise SystemExit(main())
