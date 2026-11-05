"""An experiment plan written as markdown, read back into dashboard requests.

The plan is the same text people type the form from: one heading per request (``### 15차: …``) followed by a table of
``| 순서 | 칸 | 입력 |`` rows (or ``- 칸: 입력`` lines) whose labels are the form's labels (기준 실험, 대상, 모델, the model's
hyperparameter names, 조합 방식, window 길이(초), seed, 입력 세트, the training fields, GPU, 빠른 확인(smoke),
끝나면 보고서 갱신, 메모). Only sections with a 기준 실험 row are requests; everything else (notes, result tables) is
ignored. Each request becomes the same form the 하이퍼파라미터 탐색 tab posts, so planning and queueing go through
:func:`ald_lab.launch.docs_from_form` unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from . import launch
from .models.cards import model_card

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_LIST_ROW = re.compile(r"^\s*[-*]\s+([^:：]+)[:：]\s*(.*)$")
_OFF_NOTE = re.compile(r"\([^()]*끔[^()]*\)")  # "(전체는 끔)": which chips stay off — not a value


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "").replace("__", "")).strip()


def _norm(label: str) -> str:
    return re.sub(r"[\s()]", "", label).lower()


def _no_note(text: str) -> str:
    """A value without trailing remarks in parentheses ("켬 (마지막이라)")."""

    return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()


def sections(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """``(heading, [(label, value), …])`` for every heading of the markdown, rows in order."""

    out: list[tuple[str, list[tuple[str, str]]]] = []
    title, rows = "", []
    for line in text.splitlines():
        head = _HEADING.match(line)
        if head:
            if rows:
                out.append((title, rows))
            title, rows = _clean(head.group(1)), []
            continue
        if line.strip().startswith("|"):
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # the |---|---| line
            label, value = (cells[1], cells[2]) if len(cells) >= 3 else (cells + [""])[:2]
            if _norm(label) in ("칸", "설정", "항목") or not label:
                continue
            rows.append((label, value))
            continue
        item = _LIST_ROW.match(line)
        if item:
            rows.append((_clean(item.group(1)), _clean(item.group(2))))
    if rows:
        out.append((title, rows))
    return out


def _base_form(base: dict[str, Any], gpus: str) -> dict[str, Any]:
    """The form as the tab starts it for ``base`` (same defaults as the page's pickBase)."""

    doc, task = base["doc"], base["task"]
    window = (doc.get("window") or {}).get("context_seconds") if task != "thickness" else None
    return {"base": base["name"], "targets": list(doc["targets"]), "models": {}, "search_trials": 0,
            "input_sets": [s for s in launch.base_input_sets(doc) if s in launch.INPUT_SET_NAMES],
            "seeds": ", ".join(str(s) for s in doc.get("seeds") or [42]), "windows": "" if window is None else f"{window:g}",
            "train": {k: str(v) for k, v in (doc.get("train") or {}).items()}, "memo": "", "gpus": gpus, "smoke": False,
            "report": True}


def _model_values(base: dict[str, Any], model: str) -> dict[str, str]:
    given = launch.base_model_values(base["doc"], model)
    return {p.key: ", ".join(str(v) for v in given.get(p.key, [p.default]))
            for p in launch.param_space(base["task"], model)}


def _find_models(value: str, task: str) -> list[str]:
    low = value.lower()
    return [m for m in launch.MODELS_BY_TASK[task] if model_card(m)["name"].lower() in low or re.search(rf"\b{m}\b", low)]


def _find_targets(value: str, targets: list[str]) -> list[str]:
    found = [t for t in targets if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(t)}(?![A-Za-z0-9_.-])", value)]  # "VG11만 켬"
    if not found and re.search(r"전부|전체|모두|all", value, re.IGNORECASE):
        return list(targets)
    return found


def _find_input_sets(value: str) -> list[str]:
    text = _OFF_NOTE.sub(" ", value)
    names = sorted(((name, key) for key, name in launch.INPUT_SET_NAMES.items()), key=lambda x: -len(x[0]))
    found: list[str] = []
    for name, key in names:
        if name in text:
            found.append(key)
            text = text.replace(name, " ")
    for key in launch.INPUT_SET_NAMES:
        if key not in found and re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", text):
            found.append(key)
    if "all" not in found and re.search(r"(?<![가-힣\w])전체(?![가-힣])", text):  # bare "전체" once the longer names are gone
        found.append("all")
    return [k for k in launch.INPUT_SET_NAMES if k in found]


def _switch(value: str) -> bool | None:
    text = _no_note(value)
    if "끔" in text or text.lower() in ("off", "false", "no"):
        return False
    if "켬" in text or text.lower() in ("on", "true", "yes"):
        return True
    return None


def form_from_rows(title: str, rows: list[tuple[str, str]], *, gpus: str = "auto") -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """One request: ``(form, problems, ignored labels)``. ``form`` is None when no 기준 실험 row names a known base."""

    bases = {b["name"]: b for b in launch.base_configs()}
    base_value = next((v for label, v in rows if _norm(label) == _norm("기준 실험")), None)
    if base_value is None:
        return None, [], []
    name = next((n for n in sorted(bases, key=len, reverse=True) if n in base_value), None)
    if name is None:
        return None, [f"없는 기준 실험: {_no_note(base_value)}"], []
    base = bases[name]
    task = base["task"]
    form = _base_form(base, gpus)
    form["memo"] = title
    models = ["patchtst"] if "patchtst" in launch.MODELS_BY_TASK[task] else [launch.MODELS_BY_TASK[task][0]]
    problems: list[str] = []
    ignored: list[str] = []
    params: dict[str, tuple[str, str]] = {}  # hyperparameter rows by label, given to the models chosen below
    train = {_norm(p.label): p.key for p in launch.TRAIN_FIELDS}
    labels = {_norm(p.label) for m in launch.MODELS_BY_TASK[task] for p in launch.param_space(task, m)}
    for label, value in rows:
        key = _norm(label)
        plain = _no_note(value)
        if key == _norm("기준 실험") or key in ("순서",):
            continue
        if key == _norm("대상"):
            form["targets"] = _find_targets(value, list(base["doc"]["targets"]))
            if not form["targets"]:
                problems.append(f"대상을 알아보지 못함: {value}")
        elif key == _norm("모델"):
            models = _find_models(value, task)
            if not models:
                problems.append(f"모델을 알아보지 못함: {value}")
        elif key in labels:
            params[key] = (label, plain)
        elif key in train:
            form["train"][train[key]] = plain
        elif "조합" in key:
            trials = re.search(r"무작위\s*(\d+)", value)
            form["search_trials"] = int(trials.group(1)) if trials else 0
        elif key.startswith("window"):
            form["windows"] = plain
        elif key == "seed":
            form["seeds"] = plain
        elif "입력" in key:
            form["input_sets"] = _find_input_sets(value)
            if not form["input_sets"]:
                problems.append(f"입력 세트를 알아보지 못함: {value}")
        elif key == "gpu":
            ids = re.findall(r"\d+", plain)
            form["gpus"] = ",".join(ids) if ids else ("cpu" if "cpu" in plain.lower() else gpus)
        elif "smoke" in key or "빠른확인" in key:
            form["smoke"] = bool(_switch(value))
        elif "보고서" in key:
            on = _switch(value)
            form["report"] = True if on is None else on
        elif key == "메모":
            form["memo"] = plain or title
        else:
            ignored.append(label)
    form["models"], used = {}, set()
    for m in models:  # a label may mean different settings per model (학습률: LightGBM learning_rate, PatchTST train.lr)
        values = _model_values(base, m)
        for p in launch.param_space(task, m):
            if _norm(p.label) in params:
                values[p.key] = params[_norm(p.label)][1]
                used.add(_norm(p.label))
        form["models"][m] = values
    ignored += [label for key, (label, _) in params.items() if key not in used]
    return form, problems, ignored


def read_plan(text: str, *, gpus: str = "auto") -> list[dict[str, Any]]:
    """Every request of a markdown plan: ``{"title", "form", "problems", "ignored"}`` in the order written."""

    out = []
    for title, rows in sections(text):
        form, problems, ignored = form_from_rows(title, rows, gpus=gpus)
        if form is None and not problems:
            continue  # a note or a result table, not a request
        out.append({"title": title, "form": form, "problems": problems, "ignored": ignored})
    return out
