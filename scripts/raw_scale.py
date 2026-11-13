"""원본 0.5초 값으로 다시 채점: 묶음 방식이 다른 가상 센서 실험을 같은 기준으로 비교한다. 학습 없음, work/와 원본 trace만 읽는다.

기존 1초 묶음(run 시작부터 2행씩), step 맞춤 1초 묶음(step이 바뀌면 새 묶음), 0.5초(묶음 없음)는 맞히는 값 자체가 달라
각자의 MAE를 직접 비교할 수 없다. 여기서는 각 묶음의 예측을 그 묶음이 덮는 0.5초 행에 그대로 펼쳐, 원본 0.5초 값과의 MAE를
잰다. 비교하는 실험이 모두 덮는 0.5초 행만 쓴다(같은 행, 같은 정답). 학습 모델은 seed별 MAE의 평균이다.
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
GROUPS = [("split_v2", ["lc51", "stepbin", "res_05s"]), ("split_v2 · 사이클만", ["region_cycle", "stepbin_cycle"])]
GROUPS += [(f"교차검증 {k}", [f"cv{k}", f"stepbin_cv{k}"]) for k in range(1, 6)]
GROUPS += [(f"시간순 {k}", [f"time{k}", f"stepbin_time{k}"]) for k in range(1, 4)]
NAMES = {"lc51": "기존 1초", "stepbin": "step 맞춤", "res_05s": "0.5초", "region_cycle": "기존 1초", "stepbin_cycle": "step 맞춤"}
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
    return NAMES.get(study, "step 맞춤" if study.startswith("stepbin") else "기존 1초")


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


recs = []
for group, studies in GROUPS:
    have = [s for s in studies if (st == s).any()] if len(led) else []
    if len(have) < 2:
        continue
    for target in sorted(set.intersection(*[set(led.loc[st == s, "target"]) for s in have])):
        loaded: dict[str, dict[str, list]] = {}
        for s in have:
            g = led[(st == s) & (led["target"] == target)]
            model = main_model(g)
            jobs = {"model": [raw_predictions(r) for r in one_setting(g[g["model"] == model]).itertuples()],
                    "base": [raw_predictions(r) for r in g[g["model"] == "template"].head(1).itertuples()]}
            jobs = {k: [j for j in v if j] for k, v in jobs.items()}
            if jobs["model"] and jobs["base"]:
                loaded[s] = jobs
        if len(loaded) < 2:
            continue
        every = [job for v in loaded.values() for jobs in v.values() for job in jobs]
        common = {}
        for run_id in sorted(set.intersection(*[set(job) for job in every])):
            idx = set(every[0][run_id].index)
            for job in every[1:]:
                idx &= set(job[run_id].index)
            if idx:
                common[run_id] = np.array(sorted(idx))
        if not common:
            continue
        rec = {"묶음": group, "대상": target, "0.5초 행 수": sum(len(v) for v in common.values())}
        first_base = None
        for s, v in loaded.items():
            name = label(s)
            m = float(np.mean([score(j, common, target) for j in v["model"]]))
            b = score(v["base"][0], common, target)
            first_base = b if first_base is None else first_base
            rec[f"{name} 모델"] = m
            rec[f"{name} 레시피 평균"] = b
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
        "'기준 대비 %'는 그 묶음의 첫 실험(기존 1초)의 레시피 평균을 100으로 놓은 남은 오차다.")
    for group, g in f.groupby("묶음", sort=False):
        names = [c[:-3] for c in g.columns if c.endswith(" 모델") and g[c].notna().any()]
        table = pd.DataFrame({"대상": g["대상"]})
        for name in names:
            table[f"{name} MAE"] = g[f"{name} 모델"].map(lambda x: f"{x:.4g}")
            table[f"{name} 기준 대비 %"] = (100 * g[f"{name} 모델"] / g["기준 레시피 평균"]).round(1)
            table[f"{name} 레시피 평균 %"] = (100 * g[f"{name} 레시피 평균"] / g["기준 레시피 평균"]).round(1)
        add(f"\n[{group}]  (0.5초 행 {int(g['0.5초 행 수'].iloc[0]):,}개 안팎)")
        add(table.to_string(index=False))
        if len(names) >= 2:
            a, b = names[0], names[1]
            diff = pd.DataFrame({"대상": g["대상"],
                                 f"{b} ÷ {a} − 1 (%)": (100 * (g[f"{b} 모델"] / g[f"{a} 모델"] - 1)).round(1),
                                 "사이클 안 (%)": (100 * (g[f"{b} 모델 사이클 안"] / g[f"{a} 모델 사이클 안"] - 1)).round(1)})
            for kname in ("쉬움", "어려움"):
                if f"{a} 모델 {kname}" in g:
                    diff[f"{kname} run (%)"] = (100 * (g[f"{b} 모델 {kname}"] / g[f"{a} 모델 {kname}"] - 1)).round(1)
            add(f"음수면 {b}가 원본 0.5초 기준으로도 오차가 작다")
            add(diff.to_string(index=False))

text = "\n".join(out).strip() if out else "비교할 실험(같은 분할의 기존 묶음 · step 맞춤 · 0.5초)이 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "raw_scale.txt").write_text(text, encoding="utf-8")
print(text)
