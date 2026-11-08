"""부가 실험 결과 정리 (교차검증 · 시간순 검증 · 센서 하나만 고장 · 예측 거리 확장). 학습 없이 work/experiments만 읽는다.

    python scripts/side_studies.py            # 화면 출력 + work/reports/side_studies.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths

P = resolve_paths()
led = collect_ledger(P.work_root)
led = led[led["mae"].notna() & (led.get("job_status", "ok") == "ok")]
out: list[str] = []
add = out.append


def by_run(row) -> dict[str, float]:
    f = Path(row.exp_dir) / "jobs" / row.job_id / "metrics_validation.json"
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return (doc.get("metrics") or doc).get("by_run") or {}


# 1) 가상 센서 부가 실험: 교차검증(cv1~5) · 시간순 검증(time1~3) · 센서 하나만 고장(single_sensor)
def vs_study(prefix: str, title: str) -> None:
    rows_ = led[led["study"].astype(str).str.startswith(prefix)]
    if not len(rows_):
        return
    add(f"\n## {title} (기준선 = 레시피 평균, 값은 검증 MAE)")
    rows, runs = [], []
    for (target, fold), g in rows_.groupby(["target", "study"]):
        base, pt = g[g["model"] == "template"], g[g["model"] == "patchtst"]
        if base.empty or pt.empty:
            continue
        b, p = float(base["mae"].iloc[0]), float(pt["mae"].mean())
        rows.append({"target": target, "fold": fold, "base": b, "pt": p, "gain": 100 * (1 - p / b),
                     "cycle": 100 * (1 - float(pt["mae_cycle"].mean()) / float(base["mae_cycle"].iloc[0])),
                     "other": 100 * (1 - float(pt["mae_other"].mean()) / float(base["mae_other"].iloc[0]))})
        rb = by_run(next(base.itertuples()))
        rp = [by_run(r) for r in pt.itertuples()]
        for run, v in rb.items():
            vals = [d[run] for d in rp if run in d]
            if vals:
                runs.append({"target": target, "run": run[-3:], "base": v, "pt": float(np.mean(vals))})
    f = pd.DataFrame(rows)
    if f.empty:
        return
    add("묶음별 개선율 (%)")
    add(f.pivot_table(index="target", columns="fold", values="gain").round(1).to_string())
    add("사이클 안 / 밖 개선율 (%, 묶음 평균)")
    add(f.groupby("target")[["cycle", "other"]].mean().round(1).to_string())
    r = pd.DataFrame(runs)
    summ = []
    for target, g in f.groupby("target"):
        rr = r[r["target"] == target]
        no55 = rr[rr["run"] != "055"]
        summ.append({"대상": target, "묶음 수": len(g), "개선율 평균": round(g["gain"].mean(), 1),
                     "묶음 간 SD": round(g["gain"].std(ddof=1), 1) if len(g) > 1 else np.nan,
                     "기준선 못 넘은 묶음": int((g["gain"] <= 0).sum()),
                     "run 수": len(rr), "run 전체 개선율": round(100 * (1 - rr["pt"].sum() / rr["base"].sum()), 1),
                     "055 제외": round(100 * (1 - no55["pt"].sum() / no55["base"].sum()), 1) if len(no55) else np.nan,
                     "좋아진 run 비율": f"{(rr['pt'] < rr['base']).mean() * 100:.0f} %"})
    add("\n대상별 요약 (run 전체 개선율 = run 오차 합 기준, 좋아진 run 비율 = PatchTST가 기준선보다 나은 run)")
    add(pd.DataFrame(summ).to_string(index=False))
    add("\nrun별 남은 오차 % (PatchTST ÷ 기준선 × 100)")
    r["pct"] = 100 * r["pt"] / r["base"]
    add(r.pivot_table(index="target", columns="run", values="pct").round(0).to_string())


vs_study("cv", "1. 가상 센서 run 단위 교차검증")
vs_study("time", "1-2. 가상 센서 시간순 검증 (앞쪽 run으로 학습 → 바로 다음 run으로 검증)")
vs_study("single_sensor", "1-3. 가상 센서 · 센서 하나만 고장 (같은 계열 센서는 입력에 남김)")

# 2) 예측 거리 확장
lh = led[led["study"].astype(str) == "long_horizon"]
if len(lh):
    add("\n## 2. 미래값 예측 거리 확장 (검증 MAE, PatchTST는 seed 평균)")
    hcols = sorted([c for c in lh.columns if c.startswith("mae_") and c.endswith("s") and c[4:-1].isdigit()
                    and lh[c].notna().any()], key=lambda c: int(c[4:-1]))
    names = {"persistence": "현재값 유지", "template": "레시피 평균", "patchtst": "PatchTST"}
    for target, g in lh.groupby("target"):
        t = g.groupby("model")[hcols + ["mae_cycle", "mae_other"]].mean().reindex(list(names)).dropna(how="all")
        t.index = [names[m] for m in t.index]
        t.columns = [c[4:] for c in hcols] + ["사이클 안", "사이클 밖"]
        add(f"\n[{target}]")
        add(t.apply(lambda s: s.map(lambda x: f"{x:.4g}")).to_string())
        if {"PatchTST", "현재값 유지", "레시피 평균"} <= set(t.index):
            g1 = 100 * (1 - t.loc["PatchTST"] / t.loc["현재값 유지"])
            g2 = 100 * (1 - t.loc["PatchTST"] / t.loc["레시피 평균"])
            add("현재값 유지 대비 개선 %: " + "  ".join(f"{k} {v:+.0f}" for k, v in g1.items()))
            add("레시피 평균 대비 개선 %: " + "  ".join(f"{k} {v:+.0f}" for k, v in g2.items()))

text = "\n".join(out).strip() if out else "부가 실험 결과가 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "side_studies.txt").write_text(text, encoding="utf-8")
print(text)
