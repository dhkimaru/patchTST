"""가상 센서 잔차(실측 - 추정)로 이상 run을 찾을 수 있는지 확인한다. 학습 없음, 시험 run 미사용.

가상 센서를 만든 목적은 센서 값을 추정하고, 실측이 추정에서 벗어나면 이상을 알아채는 것이다. 교차검증 5묶음에서 각 run은
한 묶음의 검증 run이므로, 그 묶음의 모델은 그 run을 보지 않았다. 저장된 검증 예측(predictions_val.parquet)을 모으면
학습 + 검증 run 62개 모두에 대해 '보지 않은 모델의 추정'이 생긴다(새 학습 · 새 예측 없음).

- 0.5초 조합 대상: 0.5초 공정 전체(res05_cv{k}) + 0.5초 사이클 전용(res05_cycle_cv{k}), 사이클 안은 사이클 전용 예측
- 1초 공정 전체 대상(온도, L.POS): 1초 교차검증(cv{k})
- 비교 기준: 같은 실험의 레시피 평균(template)에서 벗어난 정도. 가상 센서가 레시피 평균보다 이상을 더 잘 드러내야 쓸모가 있다.

run마다 대상별 MAE(잔차 크기)를 구하고, 62개 run 사이에서 robust z = (값 - 중앙값) ÷ (1.4826 × MAD)로 바꾼다.
z > 3이면 그 대상에서 '튐'. 이상 run(--runs, 기본 001 · 002 · 055 · 057)마다 튄 대상과 순위, 크게 벗어난 step을 적는다.
정상 run 중 튄 대상이 많은 run도 함께 적어, 잘못된 경보가 얼마나 되는지 본다.

    python scripts/vs_anomaly.py                 # → work/reports/vs_anomaly.txt
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
ap.add_argument("--runs", default="standard_trace_001,standard_trace_002,standard_trace_055,standard_trace_057",
                help="이상 run (쉼표 구분)")
ap.add_argument("--model", default="patchtst", help="모델 (기본 patchtst)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--z", type=float, default=3.0, help="'튐'으로 볼 robust z (기본 3)")
ap.add_argument("--min-runs", type=int, default=5, help="대상마다 필요한 최소 run 수")
args = ap.parse_args()
anomalies = [r.strip() for r in args.runs.split(",") if r.strip()]

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")]
st = led["study"].fillna("").astype(str) if len(led) else pd.Series(dtype=str)
source = TraceSource(P)
_steps: dict[tuple[str, str, int], np.ndarray] = {}


def step_names(run_id: str, target: str, factor: int) -> np.ndarray:
    key = (run_id, target, factor)
    if key not in _steps:
        frame = source.load(run_id)
        rows = np.asarray(run_arrays(frame, [target], factor, "run")["source_rows"])
        _steps[key] = frame["step_name"].astype(str).to_numpy()[rows]
    return _steps[key]


def one_setting(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or rows["settings"].nunique() <= 1:
        return rows
    return rows[rows["settings"] == rows.groupby("settings")["mae"].mean().idxmin()]


def load(row, pred: str) -> pd.DataFrame | None:
    path = Path(row.exp_dir) / "jobs" / str(row.job_id) / "predictions_val.parquet"
    if not path.is_file():
        return None
    cols = ["run_id", "row", "true_0s", "pred_0s"] + (["in_cycle"] if pred == "full" else [])
    frame = pd.read_parquet(path, columns=cols)
    frame["run_id"] = frame["run_id"].astype(str)
    return frame.rename(columns={"pred_0s": pred})


def seed_mean(rows: pd.DataFrame, pred: str) -> pd.DataFrame | None:
    frames = [f for f in (load(r, pred) for r in rows.sort_values("seed").itertuples()) if f is not None]
    if not frames:
        return None
    out = frames[0].copy()
    out[pred] = np.mean([f.set_index(["run_id", "row"])[pred].reindex(pd.MultiIndex.from_frame(out[["run_id", "row"]])).to_numpy()
                         for f in frames], axis=0)
    return out


def factor_of(row) -> int:
    doc = json.loads((Path(row.exp_dir) / "jobs" / str(row.job_id) / "job.json").read_text(encoding="utf-8"))
    return max(1, round(float(doc["sample_seconds"]) / 0.5))


def fold_rows(target: str) -> tuple[pd.DataFrame | None, str]:
    """Out-of-fold rows of every CV fold: true, model prediction, recipe mean, step, in_cycle."""

    parts, kind = [], ""
    for k in range(1, args.folds + 1):
        pick = lambda s: one_setting(led[(st == s) & (led["target"] == target) & (led["model"] == args.model)])
        full, cyc = pick(f"res05_cv{k}"), pick(f"res05_cycle_cv{k}")
        base_study = f"res05_cv{k}"
        if full.empty:
            full, cyc, base_study = pick(f"cv{k}"), pick("__none__"), f"cv{k}"
        if full.empty:
            continue
        kind = "0.5초 조합" if not cyc.empty else "1초 공정 전체"
        base = led[(st == base_study) & (led["target"] == target) & (led["model"] == "template")]
        f = seed_mean(full, "full")
        rec = next((b for b in (load(r, "rec") for r in base.itertuples()) if b is not None), None)
        if f is None or rec is None:
            continue
        m = f.merge(rec[["run_id", "row", "rec"]], on=["run_id", "row"])
        c = seed_mean(cyc, "cyc") if not cyc.empty else None
        m["model"] = m["full"] if c is None else m.merge(c[["run_id", "row", "cyc"]], on=["run_id", "row"], how="left")["cyc"] \
            .fillna(m["full"]).to_numpy()
        factor = factor_of(full.iloc[0])
        m["step"] = ""
        for run_id, g in m.groupby("run_id"):
            names = step_names(run_id, target, factor)
            ok = g["row"].to_numpy() < len(names)
            m.loc[g.index[ok], "step"] = names[g["row"].to_numpy()[ok]]
        m["fold"] = k
        parts.append(m)
    return (pd.concat(parts, ignore_index=True) if parts else None), kind


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / mad if mad > 0 else s * 0


targets = sorted(set(led.loc[st.str.match(r"^(res05_)?cv\d+$"), "target"])) if len(led) else []
per_run, steps_out, kinds = [], {}, {}
for target in targets:
    m, kind = fold_rows(target)
    if m is None or m["run_id"].nunique() < args.min_runs:
        continue
    kinds[target] = kind
    m["e_model"] = (m["true_0s"] - m["model"]).abs()
    m["e_rec"] = (m["true_0s"] - m["rec"]).abs()
    g = m.groupby("run_id").agg(e_model=("e_model", "mean"), e_rec=("e_rec", "mean"), rows=("e_model", "size"))
    g["e_model_cycle"] = m[m["in_cycle"].astype(bool)].groupby("run_id")["e_model"].mean()
    g["z_model"], g["z_rec"] = robust_z(g["e_model"]), robust_z(g["e_rec"])
    g["rank_model"] = g["e_model"].rank(ascending=False).astype(int)
    g["rank_rec"] = g["e_rec"].rank(ascending=False).astype(int)
    g["target"] = target
    per_run.append(g.reset_index())
    by_step = m.groupby(["run_id", "step"])["e_model"].mean().unstack()
    normal = by_step.drop(index=[r for r in anomalies if r in by_step.index]).median()
    for r in anomalies:
        if r in by_step.index:
            ratio = (by_step.loc[r] / normal.replace(0, np.nan)).dropna().sort_values(ascending=False)
            steps_out[(r, target)] = ratio.head(3)

lines = ["## 가상 센서 잔차로 이상 run 찾기 (교차검증에서 보지 않은 모델의 추정, 학습 없음, 시험 run 미사용)",
         (f"z = robust z(학습 + 검증 run 사이), '튐' = z > {args.z:g}. 순위는 잔차 MAE가 큰 순서(1 = 가장 큼). "
          "'레시피'는 실측이 레시피 평균에서 벗어난 정도(가장 단순한 감지 기준).")]
if per_run:
    pr = pd.concat(per_run, ignore_index=True)
    n_runs = pr["run_id"].nunique()
    rows = []
    for r in anomalies:
        g = pr[pr["run_id"] == r]
        if g.empty:
            rows.append({"run": r, "대상 수": 0})
            continue
        hit_m, hit_r = g[g["z_model"] > args.z], g[g["z_rec"] > args.z]
        top = g.sort_values("z_model", ascending=False).head(3)
        rows.append({"run": r, "대상 수": len(g), "가상 센서로 튄 대상": len(hit_m), "레시피로 튄 대상": len(hit_r),
                     "가장 크게 벗어난 대상 (z, 순위)": ", ".join(f"{t.target} ({t.z_model:.1f}, {t.rank_model}/{n_runs})"
                                                         for t in top.itertuples())})
    lines += ["", "[1. 이상 run별 요약]", pd.DataFrame(rows).to_string(index=False)]
    tab = pr[pr["run_id"].isin(anomalies)].pivot(index="target", columns="run_id", values="rank_model")
    tab_rec = pr[pr["run_id"].isin(anomalies)].pivot(index="target", columns="run_id", values="rank_rec")
    both = tab.astype(int).astype(str) + " / " + tab_rec.astype(int).astype(str)
    both.insert(0, "추정 방식", [kinds.get(t, "") for t in both.index])
    lines += ["", f"[2. 대상별 순위] 가상 센서 잔차 순위 / 레시피 평균 잔차 순위 (run {n_runs}개 중, 1 = 가장 큼)",
              both.to_string()]
    score = pr.assign(hit=pr["z_model"] > args.z).groupby("run_id").agg(튄_대상=("hit", "sum"), 최대_z=("z_model", "max"))
    score = score.sort_values(["튄_대상", "최대_z"], ascending=False)
    score["이상 run"] = ["예" if r in anomalies else "" for r in score.index]
    lines += ["", "[3. 튄 대상이 많은 run] 이상 run이 위에 몰리면 가상 센서 잔차로 이상을 가려낼 수 있다는 뜻",
              score.head(12).round(1).to_string()]
    if steps_out:
        lines += ["", "[4. 이상 run에서 잔차가 크게 벗어난 step] 정상 run의 같은 step 잔차 중앙값 대비 배수, 대상별 상위 3개"]
        for r in anomalies:
            items = [(t, s) for (rr, t), s in steps_out.items() if rr == r]
            hits = pr[(pr["run_id"] == r)].set_index("target")["z_model"]
            for t, s in sorted(items, key=lambda x: -hits.get(x[0], 0))[:4]:
                lines.append(f"- {r} · {t}: " + ", ".join(f"{step} {v:.1f}배" for step, v in s.items()))
text = "\n".join(lines) if per_run else "교차검증 묶음의 가상 센서 예측(res05_cv · cv)이 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "vs_anomaly.txt").write_text(text + "\n", encoding="utf-8")
print(text)
