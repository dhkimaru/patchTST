"""0.5초 구간 전용 조합의 시험 run 평가 (시험 run 재사용, 참고). 최종 구성이 정해진 가상 센서 대상에만 쓴다.

split_v2 검증으로 고른 0.5초 공정 전체 모델(study res_05s)과 0.5초 사이클 전용 모델(study res05_cycle)을 봉인된 시험 run에서
평가하고(ald-final-test와 같은 run_final_test, 고른 대상만), 구간별 MAE로 조합을 추정한다.
사이클 안 비율 f는 같은 실험의 레시피 평균에서 구한다: f = (전체 − 사이클 밖) ÷ (사이클 안 − 사이클 밖).
조합 MAE ≈ f × 사이클 전용 MAE + (1 − f) × 공정 전체 모델의 사이클 밖 MAE. 같은 식을 검증 run에도 적용해 나란히 적는다.
학습 모델은 seed별 MAE의 평균이고, 시험 결과로 설정을 다시 고르지 않는다.

    python scripts/pair_test.py --targets MFCMon_DCS,MFCMon_N2-1,MFCMon_N2-4,MFCMon_NH3   # → work/reports/pair_test.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.runner import parse_devices, run_final_test

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--targets", required=True, help="최종 구성이 정해진 대상 (쉼표 구분)")
ap.add_argument("--full", default="res_05s", help="공정 전체 모델의 study (기본 res_05s)")
ap.add_argument("--cycle", default="res05_cycle", help="사이클 전용 모델의 study (기본 res05_cycle)")
ap.add_argument("--model", default="patchtst", help="조합에 쓰는 모델 (기본 patchtst)")
ap.add_argument("--gpu", default="auto", help="'0' 또는 'cpu'")
args = ap.parse_args()
targets = [t.strip() for t in args.targets.split(",") if t.strip()]
P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")
              & led["study"].isin([args.full, args.cycle]) & led["target"].isin(targets)]
if not len(led):
    raise SystemExit(f"{args.full} · {args.cycle} 실험에 {', '.join(targets)} 기록이 없습니다.")
device = parse_devices(args.gpu)[0]
rows = []
for exp_dir, group in led.groupby("exp_dir"):
    # Only the chosen targets of this experiment are opened on the test runs.
    run_final_test(Path(exp_dir), P, device=device, scope="per_model", targets=sorted(set(group["target"])))
    for job in group.itertuples():
        doc = Path(exp_dir) / "jobs" / str(job.job_id) / "metrics_test.json"
        if not doc.is_file():
            continue  # not a picked label of that experiment
        test = json.loads(doc.read_text(encoding="utf-8"))
        region = test.get("by_region", {})
        rows.append({"study": job.study, "target": job.target, "model": job.model, "seed": job.seed,
                     "val": job.mae, "val_cycle": job.mae_cycle, "val_other": job.mae_other,
                     "test": test["mae"], "test_cycle": region.get("cycle"), "test_other": region.get("other")})
jobs = pd.DataFrame(rows)
mean = jobs.groupby(["study", "target", "model"]).agg(
    seeds=("seed", "nunique"), **{c: (c, "mean") for c in ("val", "val_cycle", "val_other", "test", "test_cycle", "test_other")})


def pick(study: str, target: str, model: str) -> pd.Series | None:
    key = (study, target, model)
    return mean.loc[key] if key in mean.index else None


out: list[str] = ["## 0.5초 구간 전용 조합 · 시험 run (시험 run 재사용, 참고)", "",
                  f"공정 전체 = {args.full}, 사이클 전용 = {args.cycle}, 모델 = {args.model}. 개선율은 같은 실험의 레시피 평균 대비(%).",
                  "조합은 구간별 MAE로 계산한 추정이다(사이클 안 비율은 레시피 평균의 구간별 MAE에서 구함).", ""]
table = []
for target in targets:
    base, full = pick(args.full, target, "template"), pick(args.full, target, args.model)
    cyc = pick(args.cycle, target, args.model)
    if base is None or full is None or cyc is None:
        out.append(f"- {target}: 공정 전체 · 사이클 전용 · 레시피 평균 중 빠진 것이 있어 건너뜀")
        continue
    row = {"대상": target, "seed (전체 · 사이클)": f"{int(full.seeds)} · {int(cyc.seeds)}"}
    for part, label in (("val", "검증"), ("test", "시험")):
        b, bc, bo = base[part], base[f"{part}_cycle"], base[f"{part}_other"]
        frac = (b - bo) / (bc - bo) if bc != bo else np.nan
        pair = frac * cyc[part] + (1 - frac) * full[f"{part}_other"]
        row[f"{label} 사이클 비율"] = round(frac, 2)
        row[f"{label} 레시피 평균 MAE"] = f"{b:.4g}"
        row[f"{label} 공정 전체 %"] = round(100 * (1 - full[part] / b), 1)
        row[f"{label} 조합 %"] = round(100 * (1 - pair / b), 1)
        row[f"{label} 사이클 안: 전용 ÷ 전체 − 1 (%)"] = round(100 * (cyc[part] / full[f"{part}_cycle"] - 1), 1)
    table.append(row)
if table:
    frame = pd.DataFrame(table)
    for label in ("검증", "시험"):
        out += [f"[{label} run]", frame[["대상", "seed (전체 · 사이클)", *[c for c in frame.columns if c.startswith(label)]]]
                .to_string(index=False), ""]
    gap = frame[["대상"]].assign(**{"조합 개선율 시험 − 검증 (%p)": frame["시험 조합 %"] - frame["검증 조합 %"],
                                   "공정 전체 개선율 시험 − 검증 (%p)": frame["시험 공정 전체 %"] - frame["검증 공정 전체 %"]})
    out += ["[시험 − 검증] 음수면 시험 run에서 개선 폭이 줄었다.", gap.to_string(index=False), ""]
text = "\n".join(out)
print(text)
report = P.work_root / "reports" / "pair_test.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(text + "\n", encoding="utf-8")
print(f"\n저장: {report}")
