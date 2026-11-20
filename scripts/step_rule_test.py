"""step별 레시피 평균 사용 규칙의 시험 run 확인 (학습 없음, 시험 run 재사용 · 참고).

work/reports/step_choice.json(scripts/step_choice.py)에서 채택된 대상마다, export_pair.py가 패키지로 만드는 것과 같은 job
(split_v2의 study res_05s · res05_cycle에서 검증 MAE가 가장 낮은 job)을 검증 run과 시험 run의 trace에 다시 돌려
(ald-predict --cycle-model --recipe-steps와 같은 경로) 같은 0.5초 행에서 세 가지 MAE를 잰다.
조합(0.5초 공정 전체 + 사이클 전용), 조합 + step 규칙(고른 step만 레시피 평균), 레시피 평균.
step은 교차검증 5묶음(학습 · 검증 run 62개)에서 골랐다. split_v2 검증 run도 그 62개에 들어 있어 고르는 데 쓰인 run이고,
처음 보는 run에서의 확인은 시험 run뿐이다. 시험 run 결과로 규칙이나 설정을 바꾸지 않는다.
VG11(training.evaluate.LEVEL_SPLITS)은 '공정 압력'과 '대기압 근처'(학습 run 레시피 평균 기준) 행으로 나눈 MAE도 적는다.

    python scripts/step_rule_test.py                         # → work/reports/step_rule_test.txt
    python scripts/step_rule_test.py --targets VG11 --no-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ald_lab.data.splits import load_split
from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.evaluate import LEVEL_SPLITS, level_ranges
from ald_lab.training.predict import experiment_for, predict_pair, use_recipe_steps
from ald_lab.training.prepare import TraceSource
from ald_lab.training.runner import parse_devices

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--targets", help="대상 (쉼표 구분, 기본: step_choice.json에서 채택된 대상)")
ap.add_argument("--full", default="res_05s", help="공정 전체 모델의 study (기본 res_05s)")
ap.add_argument("--cycle", default="res05_cycle", help="사이클 전용 모델의 study (기본 res05_cycle)")
ap.add_argument("--model", default="patchtst", help="모델 (기본 patchtst)")
ap.add_argument("--gpu", default="auto", help="'0' 또는 'cpu'")
ap.add_argument("--no-test", action="store_true", help="시험 run을 쓰지 않는다")
args = ap.parse_args()

P = resolve_paths()
choice_file = P.work_root / "reports" / "step_choice.json"
if not choice_file.is_file():
    raise SystemExit("work/reports/step_choice.json이 없습니다. 먼저 python scripts/step_choice.py를 실행하세요.")
choice = json.loads(choice_file.read_text(encoding="utf-8"))
targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else \
    [t for t, r in choice.items() if r.get("adopt") and r.get("steps")]
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")
              & (led["model"] == args.model)]
csv = TraceSource(P).csv
device = parse_devices(args.gpu)[0]


def best_job(study: str, target: str) -> Path | None:
    rows = led[(led["study"] == study) & (led["target"] == target)] if len(led) else led
    if not len(rows):
        return None
    row = rows.sort_values("mae").iloc[0]
    return Path(row["exp_dir"]) / "jobs" / str(row["job_id"])


out = ["## step별 레시피 평균 사용 · 시험 run 확인 (학습 없음, 시험 run 재사용 · 참고)",
       "조합 = 0.5초 공정 전체 + 사이클 전용, 규칙 = 조합에서 고른 step만 레시피 평균. 변화 = 규칙 MAE ÷ 조합 MAE - 1 (음수면 나아짐).",
       "검증 run은 step을 고른 교차검증 run에 들어 있어 참고용이고, 처음 보는 run은 시험 run이다.", ""]
table, skipped, levels = [], [], []
for target in targets:
    steps = list((choice.get(target) or {}).get("steps") or [])
    full_dir, cyc_dir = best_job(args.full, target), best_job(args.cycle, target)
    if not steps or full_dir is None or cyc_dir is None:
        skipped.append(f"- {target}: " + ("고른 step이 없음" if not steps else f"study {args.full} · {args.cycle} 기록 없음"))
        continue
    split = load_split(experiment_for(full_dir).split_path())
    parts = [("검증", split.runs("validation"))]
    if not args.no_test:
        parts.append(("시험", split.runs("test", include_test=True)))  # 재사용, 참고
    for label, runs in parts:
        files = [csv[r] for r in runs if r in csv]
        if not files:
            skipped.append(f"- {target} · {label}: trace CSV를 찾지 못함 (ALD_DATA_ROOT 확인)")
            continue
        pair = predict_pair(full_dir, cyc_dir, files, device=device)
        if "actual" not in pair or "recipe" not in pair:
            skipped.append(f"- {target} · {label}: 실측 또는 레시피 평균 열이 없음")
            continue
        rule = use_recipe_steps(pair, steps)
        e_pair, e_rule = (pair["pred"] - pair["actual"]).abs(), (rule["pred"] - rule["actual"]).abs()
        e_rec = (pair["recipe"] - pair["actual"]).abs()
        inside = rule["model_used"].eq("레시피 평균")
        unit = pair.attrs.get("unit") or "단위 미확인"
        table.append({"대상": target, "구분": label, "단위": unit, "run": len(files), "행": len(pair),
                      "바꾼 step 행 %": round(100 * inside.mean(), 1), "조합 MAE": f"{e_pair.mean():.4g}",
                      "규칙 MAE": f"{e_rule.mean():.4g}", "변화 (%)": round(100 * (e_rule.mean() / e_pair.mean() - 1), 1),
                      "레시피 평균 MAE": f"{e_rec.mean():.4g}",
                      "바꾼 step 안 조합 MAE": f"{e_pair[inside].mean():.4g}" if inside.any() else "-",
                      "바꾼 step 안 레시피 평균 MAE": f"{e_rec[inside].mean():.4g}" if inside.any() else "-"})
        ranges = level_ranges(target, pair["recipe"])
        if ranges is not None:
            unit = LEVEL_SPLITS[target][1]
            for name in pd.unique(ranges):
                sel = ranges == name
                levels.append({"대상": target, "구분": label, "구간": name, "행 %": round(100 * sel.mean(), 1),
                               f"레시피 평균 MAE ({unit})": f"{e_rec[sel].mean():.4g}",
                               f"조합 MAE ({unit})": f"{e_pair[sel].mean():.4g}", f"규칙 MAE ({unit})": f"{e_rule[sel].mean():.4g}",
                               "조합 개선율 (%)": round(100 * (1 - e_pair[sel].mean() / e_rec[sel].mean()), 1),
                               "규칙 개선율 (%)": round(100 * (1 - e_rule[sel].mean() / e_rec[sel].mean()), 1)})
if table:
    frame = pd.DataFrame(table)
    for label in ("검증", "시험"):
        part = frame[frame["구분"] == label].drop(columns="구분")
        if len(part):
            out += [f"[{label} run]", part.to_string(index=False), ""]
    if {"검증", "시험"} <= set(frame["구분"]):
        gap = frame.pivot(index="대상", columns="구분", values="변화 (%)")
        out += ["[변화 (%): 검증 · 시험] 시험 run에서도 음수면 처음 보는 run에서도 규칙이 오차를 줄인 것이다.",
                gap[["검증", "시험"]].to_string(), ""]
if levels:
    out += ["[압력 구간별] 대기압 근처 = 학습 run 레시피 평균이 " + ", ".join(f"{t} {v[0]:g} {v[1]}" for t, v in LEVEL_SPLITS.items() if t in {r["대상"] for r in levels})
            + "를 넘는 행. 개선율은 같은 구간의 레시피 평균 대비. 판정은 공정 압력 구간으로 한다.",
            pd.DataFrame(levels).to_string(index=False), ""]
out += skipped
text = "\n".join(out)
print(text)
report = P.work_root / "reports" / "step_rule_test.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(text + "\n", encoding="utf-8")
print(f"\n저장: {report}")
