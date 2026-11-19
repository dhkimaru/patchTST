"""step별 레시피 평균 사용: 0.5초 조합이 레시피 평균보다 못한 step에서 레시피 평균을 쓰면 나아지는지 교차검증으로 확인한다.

대상마다 교차검증 k묶음의 0.5초 조합(공정 전체 res05_cv{k} + 사이클 전용 res05_cycle_cv{k}, 사이클 안은 사이클 전용 예측)과
같은 실험의 레시피 평균 예측(predictions_val.parquet)을 step 이름별로 모은다. 묶음 k에서 어떤 step을 레시피 평균으로 바꿀지는
나머지 묶음에서만 정하고(그 step의 레시피 평균 오차 합이 조합보다 작으면 바꿈), 묶음 k에서 채점해 조합만 쓴 경우와 비교한다.
학습 모델은 seed별 오차의 평균이다. 학습 없음, 시험 run 미사용.
step_choice.json에는 대상마다 채택 여부(모든 묶음에서 나아짐: 변화 평균 < 0, 가장 큰 값 < 0)와 바꾼 step을 남긴다.
scripts/export_pair.py가 이 파일을 읽어 채택된 대상의 패키지에 recipe_steps.json을 넣는다(ald-predict --recipe-steps).

    python scripts/step_choice.py                 # → work/reports/step_choice.txt, step_choice.json
    python scripts/step_choice.py --model ridge   # 다른 모델로 확인할 때
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.prepare import TraceSource, run_arrays

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", default="patchtst", help="조합에 쓰는 모델 (기본 patchtst)")
ap.add_argument("--folds", type=int, default=5, help="교차검증 묶음 수 (기본 5)")
args = ap.parse_args()

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")]
st = led["study"].fillna("").astype(str) if len(led) else pd.Series(dtype=str)
source = TraceSource(P)
_steps: dict[tuple[str, str], np.ndarray] = {}


def step_names(run_id: str, target: str) -> np.ndarray:
    """Step name of every 0.5 s row the run's bundle keeps (row index = the job's ``row``)."""

    key = (run_id, target)
    if key not in _steps:
        frame = source.load(run_id)
        rows = np.asarray(run_arrays(frame, [target], 1, "run")["source_rows"])
        _steps[key] = frame["step_name"].astype(str).to_numpy()[rows]
    return _steps[key]


def one_setting(rows: pd.DataFrame) -> pd.DataFrame:
    """A study can hold several settings of a model; keep the one with the lowest validation MAE."""

    if rows.empty or rows["settings"].nunique() <= 1:
        return rows
    return rows[rows["settings"] == rows.groupby("settings")["mae"].mean().idxmin()]


def load(row) -> pd.DataFrame | None:
    path = Path(row.exp_dir) / "jobs" / str(row.job_id) / "predictions_val.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=["run_id", "row", "true_0s", "pred_0s"])
    frame["run_id"] = frame["run_id"].astype(str)
    return frame


def fold_errors(k: int, target: str) -> pd.DataFrame | None:
    """Per step of fold k: summed absolute error of the pair (mean over seeds) and of the recipe mean, and rows."""

    full = one_setting(led[(st == f"res05_cv{k}") & (led["target"] == target) & (led["model"] == args.model)])
    cyc = one_setting(led[(st == f"res05_cycle_cv{k}") & (led["target"] == target) & (led["model"] == args.model)])
    base = led[(st == f"res05_cv{k}") & (led["target"] == target) & (led["model"] == "template")]
    fulls = [f for f in (load(r) for r in full.sort_values("seed").itertuples()) if f is not None]
    cycles = [c for c in (load(r) for r in cyc.sort_values("seed").itertuples()) if c is not None]
    recipe = next((b for b in (load(r) for r in base.itertuples()) if b is not None), None)
    if not fulls or not cycles or recipe is None:
        return None
    recipe = recipe.rename(columns={"pred_0s": "rec"})[["run_id", "row", "rec"]]
    parts = []
    for i, f in enumerate(fulls):
        c = cycles[min(i, len(cycles) - 1)].rename(columns={"pred_0s": "cyc"})[["run_id", "row", "cyc"]]
        m = f.merge(recipe, on=["run_id", "row"]).merge(c, on=["run_id", "row"], how="left")
        m["pair"] = m["cyc"].fillna(m["pred_0s"])
        m["step"] = ""
        for run_id, g in m.groupby("run_id"):
            names = step_names(run_id, target)
            ok = g["row"].to_numpy() < len(names)
            m.loc[g.index[ok], "step"] = names[g["row"].to_numpy()[ok]]
        m = m[m["step"] != ""]
        m["e_pair"] = (m["pair"] - m["true_0s"]).abs()
        m["e_rec"] = (m["rec"] - m["true_0s"]).abs()
        parts.append(m.groupby("step").agg(e_pair=("e_pair", "sum"), e_rec=("e_rec", "sum"), n=("e_pair", "size")))
    out = sum(p[["e_pair"]] for p in parts) / len(parts)  # seeds averaged on the pair; the recipe mean is one
    return out.join(parts[0][["e_rec", "n"]])


lines: list[str] = ["## step별 레시피 평균 사용 · 0.5초 조합 · 교차검증 (학습 없음, 시험 run 미사용)",
                    "묶음 k의 step 선택은 나머지 묶음의 오차 합으로만 정한다. 변화는 '선택 MAE ÷ 조합 MAE − 1'(음수면 나아짐), "
                    "바꾼 step은 묶음의 절반을 넘게 레시피 평균으로 고른 step, 행 %는 그 step들이 차지하는 행의 몫이다."]
targets = sorted(set(led.loc[st.str.match(r"^res05_cv\d+$"), "target"])) if len(led) else []
summary, detail, decisions = [], [], {}
for target in targets:
    folds = {k: e for k in range(1, args.folds + 1) if (e := fold_errors(k, target)) is not None}
    if len(folds) < 2:
        continue
    changes, pair_mae, pick_mae, picked = [], [], [], {}
    for k, e in folds.items():
        rest = sum(f[["e_pair", "e_rec"]] for j, f in folds.items() if j != k)
        rest = rest.reindex(e.index).fillna(np.inf)  # a step the other folds never saw keeps the pair
        use_rec = rest["e_rec"] < rest["e_pair"]
        for step in use_rec[use_rec].index:
            picked[step] = picked.get(step, 0) + 1
        chosen = np.where(use_rec, e["e_rec"], e["e_pair"]).sum() / e["n"].sum()
        plain = e["e_pair"].sum() / e["n"].sum()
        pair_mae.append(plain)
        pick_mae.append(chosen)
        changes.append(100 * (chosen / plain - 1))
    pooled = sum(f[["e_pair", "e_rec", "n"]] for f in folds.values())
    steady = [s for s, c in sorted(picked.items(), key=lambda x: -pooled.loc[x[0], "n"]) if c > len(folds) / 2]
    share = 100 * pooled.loc[steady, "n"].sum() / pooled["n"].sum() if steady else 0.0
    ch = pd.Series(changes)
    summary.append({"대상": target, "묶음 수": len(folds), "조합 MAE": f"{np.mean(pair_mae):.4g}",
                    "step 선택 MAE": f"{np.mean(pick_mae):.4g}", "변화 (%)": f"{ch.mean():.1f} ± {ch.std():.1f}",
                    "가장 큰 값": round(ch.max(), 1), "바꾼 step 수": len(steady), "바꾼 step 행 %": round(share, 1)})
    decisions[target] = {"adopt": bool(ch.mean() < 0 and ch.max() < 0), "steps": steady, "folds": len(folds),
                         "change_mean": round(float(ch.mean()), 2), "change_sd": round(float(ch.std()), 2),
                         "change_max": round(float(ch.max()), 2), "rows_share": round(float(share), 1),
                         "model": args.model}
    for s in steady:
        r = pooled.loc[s]
        detail.append({"대상": target, "step": s, "행 %": round(100 * r["n"] / pooled["n"].sum(), 1),
                       "조합 MAE": f"{r['e_pair'] / r['n']:.4g}", "레시피 평균 MAE": f"{r['e_rec'] / r['n']:.4g}",
                       "고른 묶음 수": picked[s]})

if summary:
    lines.append("\n[대상별 요약] 변화의 평균이 음수이고 가장 큰 값도 0보다 작으면 모든 묶음에서 나아진 것이다.")
    lines.append(pd.DataFrame(summary).to_string(index=False))
if detail:
    lines.append("\n[레시피 평균으로 바꾼 step] 모든 묶음을 합친 MAE")
    lines.append(pd.DataFrame(detail).to_string(index=False))
text = "\n".join(lines) if summary else "비교할 실험(교차검증 묶음의 0.5초 공정 전체 · 사이클 전용 · 레시피 평균)이 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "step_choice.txt").write_text(text, encoding="utf-8")
(P.work_root / "reports" / "step_choice.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
print(text)
