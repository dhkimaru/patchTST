"""원본 0.5초 값으로 다시 채점: 묶음 방식이 다른 가상 센서 실험을 같은 기준으로 비교한다. 학습 없음, work/와 원본 trace만 읽는다.

기존 1초 묶음(run 시작부터 2행씩), step 맞춤 1초 묶음(step이 바뀌면 새 묶음), 0.5초(묶음 없음)는 맞히는 값 자체가 달라
각자의 MAE를 직접 비교할 수 없다. 여기서는 각 묶음의 예측을 그 묶음이 덮는 0.5초 행에 그대로 펼쳐, 원본 0.5초 값과의 MAE를
잰다. 비교하는 실험이 모두 덮는 0.5초 행만 쓴다(같은 행, 같은 정답). 학습 모델은 seed별 MAE의 평균이다.
뒤쪽 절은 구간 전용 조합(사이클 안은 사이클 전용 모델, 나머지는 공정 전체 모델)을 묶음 방식별로 같은 기준에서 잰다.
work/reports/run_types.json(run_drift.py)이 있으면 run 유형(쉬움 · 어려움)별로도 나눈다. 시험 run은 쓰지 않는다.

    python scripts/raw_scale.py      # 화면 출력 + work/reports/raw_scale.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.data.catalog import CYCLE_STEPS
from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.prepare import EXPECTED_INTERVAL_S, TraceSource, run_arrays

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led.get("job_status", "ok") == "ok") & (led["task"] == "virtual_sensor")]
st = led["study"].fillna("").astype(str) if len(led) else pd.Series(dtype=str)
# A slot lists candidate studies; the first one that holds the target is used. Cycle-only slots try the 64 s window
# first (the 0.5 s and step-aligned cycle models use 64 s, ValveAct_29 32 s), so the groupings share their window.
CYC_SPLIT = [("region_cycle_win", "region_cycle"), ("stepbin_cycle_win", "stepbin_cycle"), ("res05_cycle",)]
GROUPS: list[tuple[str, list[tuple[str, ...]]]] = [("split_v2", [("lc51",), ("stepbin",), ("res_05s",)]),
                                                   ("split_v2 · 사이클만", CYC_SPLIT)]
GROUPS += [(f"교차검증 {k}", [(f"cv{k}",), (f"stepbin_cv{k}",), (f"res05_cv{k}",)]) for k in range(1, 6)]
GROUPS += [(f"교차검증 {k} · 사이클만", [(f"region_cycle_win_cv{k}", f"region_cycle_cv{k}"), (f"stepbin_cycle_cv{k}",),
                                       (f"res05_cycle_cv{k}",)]) for k in range(1, 6)]
GROUPS += [(f"시간순 {k}", [(f"time{k}",), (f"stepbin_time{k}",)]) for k in range(1, 4)]
# Region pairs per grouping: (name, full-process slot, cycle-only slot).
HYBRIDS: list[tuple[str, list[tuple[str, tuple[str, ...], tuple[str, ...]]]]] = [
    ("split_v2", [("기존 1초", ("lc51",), CYC_SPLIT[0]), ("step 맞춤", ("stepbin",), CYC_SPLIT[1]),
                  ("0.5초", ("res_05s",), CYC_SPLIT[2])])]
HYBRIDS += [(f"교차검증 {k}", [("기존 1초", (f"cv{k}",), (f"region_cycle_win_cv{k}", f"region_cycle_cv{k}")),
                             ("step 맞춤", (f"stepbin_cv{k}",), (f"stepbin_cycle_cv{k}",)),
                             ("0.5초", (f"res05_cv{k}",), (f"res05_cycle_cv{k}",))]) for k in range(1, 6)]
NAMES = {"lc51": "기존 1초", "res_05s": "0.5초"}
_types = P.work_root / "reports" / "run_types.json"
RUN_TYPES: dict[str, str] = ({k: v["type"] for k, v in json.loads(_types.read_text(encoding="utf-8"))["runs"].items()}
                             if _types.is_file() else {})
source = TraceSource(P)
_frames: dict[str, pd.DataFrame] = {}
_blocks: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
out: list[str] = []
add = out.append


def frame_of(run_id: str) -> pd.DataFrame:
    if run_id not in _frames:
        _frames[run_id] = source.load(run_id)
    return _frames[run_id]


def blocks(run_id: str, target: str, factor: int, align: str) -> tuple[np.ndarray, np.ndarray]:
    """First and last raw row of every block of the run, in bundle order."""

    key = (run_id, factor, align)
    if key not in _blocks:
        part = run_arrays(frame_of(run_id), [target], factor, align)
        _blocks[key] = (np.asarray(part["source_starts"]), np.asarray(part["source_rows"]))
    return _blocks[key]


def label(study: str) -> str:
    if study in NAMES:
        return NAMES[study]
    if study.startswith("stepbin"):
        return "step 맞춤"
    return "0.5초" if study.startswith("res05") else "기존 1초"


def raw_predictions(row) -> dict[str, pd.Series] | None:
    """Per validation run: the job's predictions spread over the 0.5 s rows their blocks cover (index = raw row)."""

    job_dir = Path(row.exp_dir) / "jobs" / str(row.job_id)
    pred_file, job_file = job_dir / "predictions_val.parquet", job_dir / "job.json"
    if not pred_file.is_file() or not job_file.is_file():
        return None
    doc = json.loads(job_file.read_text(encoding="utf-8"))
    factor = max(1, round(float(doc.get("sample_seconds", 1.0)) / EXPECTED_INTERVAL_S))
    align = str(doc.get("bin_align", "run"))
    pred = pd.read_parquet(pred_file)
    out_: dict[str, pd.Series] = {}
    for run_id, g in pred.groupby(pred["run_id"].astype(str), observed=True):
        starts, lasts = blocks(run_id, row.target, factor, align)
        rows = g["row"].to_numpy()
        ok = rows < len(starts)
        rows, values = rows[ok], g["pred_0s"].to_numpy()[ok]
        lengths = lasts[rows] - starts[rows] + 1
        raw = np.concatenate([np.arange(a, b + 1) for a, b in zip(starts[rows], lasts[rows], strict=True)])
        out_[run_id] = pd.Series(np.repeat(values, lengths), index=raw)
    return out_


def truth(run_id: str, target: str) -> np.ndarray:
    return frame_of(run_id)[target].ffill().bfill().fillna(0.0).to_numpy(float)


def score(preds: dict[str, pd.Series], common: dict[str, np.ndarray], target: str, kind: str | None = None,
          cycle: bool | None = None) -> float:
    err, n = 0.0, 0
    for run_id, idx in common.items():
        if kind and RUN_TYPES.get(run_id) != kind:
            continue
        if run_id not in preds:
            return np.nan
        rows = idx
        if cycle is not None:
            in_cycle = frame_of(run_id)["step_name"].isin(CYCLE_STEPS).to_numpy()[rows]
            rows = rows[in_cycle if cycle else ~in_cycle]
        if not len(rows):
            continue
        err += float(np.abs(preds[run_id].loc[rows].to_numpy() - truth(run_id, target)[rows]).sum())
        n += len(rows)
    return err / n if n else np.nan


def main_model(rows: pd.DataFrame) -> str | None:
    trained = [m for m in rows["model"].unique() if m != "template"]
    return "patchtst" if "patchtst" in trained else (min(trained) if trained else None)


def one_setting(rows: pd.DataFrame) -> pd.DataFrame:
    """A study can hold several settings of a model; keep the one with the lowest validation MAE."""

    if rows.empty or rows["settings"].nunique() <= 1:
        return rows
    return rows[rows["settings"] == rows.groupby("settings")["mae"].mean().idxmin()]


def pick(slot: tuple[str, ...], target: str) -> str | None:
    """First study of the slot that holds the target."""

    return next((s for s in slot if ((st == s) & (led["target"] == target)).any()), None)


def load_study(study: str, target: str) -> dict | None:
    """Main-model jobs (one setting, every seed) and one recipe-mean job of a study, spread over raw rows."""

    g = led[(st == study) & (led["target"] == target)]
    model = main_model(g)
    if model is None:
        return None
    chosen = one_setting(g[g["model"] == model])
    jobs = {"model": [raw_predictions(r) for r in chosen.itertuples()],
            "base": [raw_predictions(r) for r in g[g["model"] == "template"].head(1).itertuples()]}
    jobs = {k: [j for j in v if j] for k, v in jobs.items()}
    if not jobs["model"] or not jobs["base"]:
        return None
    window = pd.to_numeric(chosen["context_seconds"], errors="coerce").dropna()
    return {**jobs, "window": int(window.iloc[0]) if len(window) else None, "study": study}


def common_rows(every: list[dict[str, pd.Series]]) -> dict[str, np.ndarray]:
    """Raw rows that every job covers, per validation run."""

    common = {}
    for run_id in sorted(set.intersection(*[set(job) for job in every])):
        idx = set(every[0][run_id].index)
        for job in every[1:]:
            idx &= set(job[run_id].index)
        if idx:
            common[run_id] = np.array(sorted(idx))
    return common


def hybrid(full: dict[str, pd.Series], cycle: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Cycle-only predictions where the cycle model has them, the full-process model elsewhere."""

    return {run_id: (cycle[run_id].combine_first(s) if run_id in cycle else s) for run_id, s in full.items()}


recs = []
for group, slots in GROUPS:
    for target in sorted(set(led.loc[st.isin([s for slot in slots for s in slot]), "target"])) if len(led) else []:
        loaded: dict[str, dict] = {}
        for slot in slots:
            study = pick(slot, target)
            item = load_study(study, target) if study else None
            if item:
                loaded[label(study)] = {**item, "choices": len(slot)}
        if len(loaded) < 2:
            continue
        common = common_rows([job for v in loaded.values() for key in ("model", "base") for job in v[key]])
        if not common:
            continue
        rec = {"묶음": group, "대상": target, "0.5초 행 수": sum(len(v) for v in common.values())}
        first_base = None
        for name, v in loaded.items():
            m = float(np.mean([score(j, common, target) for j in v["model"]]))
            b = score(v["base"][0], common, target)
            first_base = b if first_base is None else first_base
            rec[f"{name} 모델"] = m
            rec[f"{name} 레시피 평균"] = b
            if v["choices"] > 1 or group.endswith("사이클만"):
                rec[f"{name} window"] = v["window"]
            for kind, kname in (("easy", "쉬움"), ("hard", "어려움")):
                if RUN_TYPES:
                    rec[f"{name} 모델 {kname}"] = float(np.mean([score(j, common, target, kind) for j in v["model"]]))
            rec[f"{name} 모델 사이클 안"] = float(np.mean([score(j, common, target, cycle=True) for j in v["model"]]))
        rec["기준 레시피 평균"] = first_base
        recs.append(rec)

if recs:
    f = pd.DataFrame(recs)
    add("## 원본 0.5초 값으로 다시 채점 (같은 0.5초 행 · 같은 정답, 학습 모델은 seed 평균, 시험 run 미사용)")
    add("각 묶음의 예측을 그 묶음이 덮는 0.5초 행에 펼쳐 원본 값과의 MAE를 쟀다. "
        "'기준 대비 %'는 그 묶음의 첫 실험(보통 기존 1초)의 레시피 평균을 100으로 놓은 남은 오차다. "
        "사이클만 묶음은 window 64초 실험을 먼저 쓰고, 없으면 32초 실험을 쓴다(window 열).")
    for group, g in f.groupby("묶음", sort=False):
        names = [c[:-3] for c in g.columns if c.endswith(" 모델") and g[c].notna().any()]
        table = pd.DataFrame({"대상": g["대상"]})
        for name in names:
            table[f"{name} MAE"] = g[f"{name} 모델"].map(lambda x: f"{x:.4g}")
            table[f"{name} 기준 대비 %"] = (100 * g[f"{name} 모델"] / g["기준 레시피 평균"]).round(1)
            table[f"{name} 레시피 평균 %"] = (100 * g[f"{name} 레시피 평균"] / g["기준 레시피 평균"]).round(1)
            if f"{name} window" in g:
                table[f"{name} window"] = g[f"{name} window"].map(lambda x: "-" if pd.isna(x) else f"{int(x)}초")
        add(f"\n[{group}]  (0.5초 행 {int(g['0.5초 행 수'].iloc[0]):,}개 안팎)")
        add(table.to_string(index=False))
        pairs = [(names[0], b) for b in names[1:]] + ([(names[1], names[2])] if len(names) > 2 else [])
        for a, b in pairs:
            diff = pd.DataFrame({"대상": g["대상"],
                                 f"{b} ÷ {a} − 1 (%)": (100 * (g[f"{b} 모델"] / g[f"{a} 모델"] - 1)).round(1),
                                 "사이클 안 (%)": (100 * (g[f"{b} 모델 사이클 안"] / g[f"{a} 모델 사이클 안"] - 1)).round(1)})
            for kname in ("쉬움", "어려움"):
                if f"{a} 모델 {kname}" in g:
                    diff[f"{kname} run (%)"] = (100 * (g[f"{b} 모델 {kname}"] / g[f"{a} 모델 {kname}"] - 1)).round(1)
            diff = diff.dropna(subset=[f"{b} ÷ {a} − 1 (%)"])
            if len(diff):
                add(f"음수면 {b}{'이' if b.endswith('맞춤') else '가'} 원본 0.5초 기준으로도 {a}보다 오차가 작다")
                add(diff.to_string(index=False))

# Region pairs (cycle-only model inside cycles, full-process model elsewhere) per grouping, on the same raw rows.
hrecs = []
for split, entries in HYBRIDS:
    studies = [s for _, full, cyc in entries for s in (*full, *cyc)]
    for target in sorted(set(led.loc[st.isin(studies), "target"])) if len(led) else []:
        loaded = {}
        for name, full_slot, cyc_slot in entries:
            full_study, cyc_study = pick(full_slot, target), pick(cyc_slot, target)
            full_item = load_study(full_study, target) if full_study else None
            cyc_item = load_study(cyc_study, target) if cyc_study else None
            if full_item and cyc_item:
                loaded[name] = {"full": full_item, "cycle": cyc_item}
        if len(loaded) < 2:
            continue
        common = common_rows([job for v in loaded.values() for job in (*v["full"]["model"], *v["full"]["base"])])
        if not common:
            continue
        rec = {"분할": split, "대상": target}
        ref = None
        for name, v in loaded.items():
            fulls, cycles = v["full"]["model"], v["cycle"]["model"]
            pairs = [hybrid(fj, cycles[min(i, len(cycles) - 1)]) for i, fj in enumerate(fulls)]
            base = score(v["full"]["base"][0], common, target)
            ref = base if ref is None else ref
            rec[f"{name} 조합"] = float(np.mean([score(p, common, target) for p in pairs]))
            rec[f"{name} 공정 전체"] = float(np.mean([score(j, common, target) for j in fulls]))
            rec[f"{name} 레시피 평균"] = base
            rec[f"{name} seed"] = f"{len(fulls)}/{len(cycles)}"
            rec[f"{name} 사이클 window"] = v["cycle"]["window"]
            if RUN_TYPES:
                for kind, kname in (("easy", "쉬움"), ("hard", "어려움")):
                    rec[f"{name} 조합 {kname}"] = float(np.mean([score(p, common, target, kind) for p in pairs]))
        rec["기준"] = ref
        hrecs.append(rec)

if hrecs:
    h = pd.DataFrame(hrecs)
    add("\n## 구간 전용 조합 · 원본 0.5초 값 (사이클 안은 사이클 전용 모델, 나머지는 공정 전체 모델)")
    add("묶음 방식마다 공정 전체 모델과 사이클 전용 모델을 합쳐 같은 0.5초 행에서 잰 남은 오차(%)다. 100 = 첫 묶음 방식"
        "(보통 기존 1초)의 레시피 평균. seed는 공정 전체 / 사이클 전용 seed 수(seed끼리 짝지어 합친 뒤 평균).")
    names = [n for n in ("기존 1초", "step 맞춤", "0.5초") if f"{n} 조합" in h]
    for split, g in h.groupby("분할", sort=False):
        table = pd.DataFrame({"대상": g["대상"]})
        for name in names:
            table[f"{name} 조합 %"] = (100 * g[f"{name} 조합"] / g["기준"]).round(1)
            table[f"{name} 공정 전체 %"] = (100 * g[f"{name} 공정 전체"] / g["기준"]).round(1)
        for name in names:
            table[f"{name} 레시피 평균 %"] = (100 * g[f"{name} 레시피 평균"] / g["기준"]).round(1)
        for name in names:
            table[f"{name} seed"] = g[f"{name} seed"]
            table[f"{name} 사이클 window"] = g[f"{name} 사이클 window"].map(lambda x: "-" if pd.isna(x) else f"{int(x)}초")
        add(f"\n[{split}]")
        add(table.to_string(index=False))
        kinds = [c for c in g.columns if c.endswith((" 조합 쉬움", " 조합 어려움"))]
        if kinds:
            k = pd.DataFrame({"대상": g["대상"]})
            for c in kinds:
                k[c.replace(" 조합", "")] = (100 * g[c] / g["기준"]).round(1)
            add("run 유형별 (조합, 같은 기준 대비 %)")
            add(k.to_string(index=False))
    folds = h[h["분할"].str.startswith("교차검증")]
    if folds["분할"].nunique() >= 2:
        rows = []
        for target, g in folds.groupby("대상"):
            row = {"대상": target, "묶음 수": g["분할"].nunique()}
            for name in names:
                v = 100 * g[f"{name} 조합"] / g["기준"]
                row[f"{name} 조합 %"] = f"{v.mean():.1f} ± {v.std():.1f}" if v.notna().sum() > 1 else "-"
            if {"step 맞춤", "0.5초"} <= set(names):
                d = 100 * (g["0.5초 조합"] / g["step 맞춤 조합"] - 1)
                if d.notna().sum() > 1:
                    row["0.5초 ÷ step 맞춤 − 1 (%)"] = f"{d.mean():.1f} ± {d.std():.1f}"
                    row["가장 큰 값"] = round(d.max(), 1)
            rows.append(row)
        add("\n교차검증 묶음 요약 (조합의 남은 오차 %, 묶음 평균 ± 표준편차). 0.5초 ÷ step 맞춤 − 1이 음수이고 가장 큰 값도 "
            "0보다 작으면 모든 묶음에서 0.5초 조합이 원본 값 기준으로 더 정확하다.")
        add(pd.DataFrame(rows).to_string(index=False))

text = "\n".join(out).strip() if out else "비교할 실험(같은 분할의 기존 묶음 · step 맞춤 · 0.5초)이 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "raw_scale.txt").write_text(text, encoding="utf-8")
print(text)
