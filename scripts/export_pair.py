"""가상 센서 구간 전용 조합(0.5초 공정 전체 + 0.5초 사이클 전용)을 대상별 패키지로 내보낸다. 학습 없음.

대상마다 split_v2의 공정 전체 study(기본 res_05s)와 사이클 전용 study(기본 res05_cycle)에서 검증 MAE가 가장 낮은 job을
하나씩 골라 work/packages/pair_<시각>/<대상>/ 아래 full/ · cycle/ 로 복사하고, 두 모델을 구간별로 합쳐 예측하는
predict.sh · predict.bat(ald-predict --cycle-model)과 README.txt를 만든다. 패키지에는 회사 데이터로 학습한 가중치가
들어 있으니 서버나 회사 PC에만 두고 Git에는 넣지 않는다.
work/reports/step_choice.json(scripts/step_choice.py)에서 step별 레시피 평균 사용이 채택된 대상은 그 step 목록을
recipe_steps.json으로 넣고 predict.sh · predict.bat에 --recipe-steps를 붙인다(--no-recipe-steps로 끔).

    python scripts/export_pair.py --targets MFCMon_DCS,MFCMon_N2-1,...   # → work/packages/pair_<시각>/ (+ zip)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.export import export_pair

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--targets", required=True, help="내보낼 대상 (쉼표 구분)")
ap.add_argument("--full", default="res_05s", help="공정 전체 모델의 study (기본 res_05s)")
ap.add_argument("--cycle", default="res05_cycle", help="사이클 전용 모델의 study (기본 res05_cycle)")
ap.add_argument("--model", default="patchtst", help="모델 (기본 patchtst)")
ap.add_argument("--no-zip", action="store_true", help="zip을 만들지 않는다")
ap.add_argument("--no-recipe-steps", action="store_true", help="step별 레시피 평균 사용(step_choice.json)을 넣지 않는다")
args = ap.parse_args()
targets = [t.strip() for t in args.targets.split(",") if t.strip()]
P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")
              & (led["model"] == args.model)]
root = P.work_root / "packages" / f"pair_{datetime.now():%Y%m%d-%H%M}"
choice_file = P.work_root / "reports" / "step_choice.json"
choice = {} if args.no_recipe_steps or not choice_file.is_file() else json.loads(choice_file.read_text(encoding="utf-8"))
lines = []
for target in targets:
    jobs = {}
    for part, study in (("full", args.full), ("cycle", args.cycle)):
        rows = led[(led["study"] == study) & (led["target"] == target)] if len(led) else led
        if not len(rows):
            raise SystemExit(f"{target}: study {study}에 {args.model} 기록이 없습니다.")
        best = rows.sort_values("mae").iloc[0]
        jobs[part] = Path(best["exp_dir"]) / "jobs" / str(best["job_id"])
        lines.append(f"{target} · {part}: {best['experiment']}/{best['stamp']} seed {best['seed']} 검증 MAE {best['mae']:.4g}")
    rule = choice.get(target) or {}
    rule = rule if rule.get("adopt") and (rule.get("steps") or rule.get("cells")) else None
    size = (f"step {len(rule['steps'])}개" if rule.get("steps") else f"step × 압력 구간 {len(rule['cells'])}칸") if rule else ""
    lines.append(f"{target} · step별 레시피 평균: " + (f"{size} (교차검증 변화 {rule['change_mean']:.1f}%)" if rule else "쓰지 않음"))
    export_pair(jobs["full"], jobs["cycle"], root / target, recipe_steps=rule,
                note=f"고른 기준: 대상별로 study {args.full} · {args.cycle}에서 검증 MAE가 가장 낮은 {args.model} job (시험 run은 쓰지 않음).")
(root / "README.txt").write_text("ALD 가상 센서 구간 전용 조합 패키지\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"패키지: {root}")
if not args.no_zip:
    print(f"zip:    {shutil.make_archive(str(root), 'zip', root_dir=root.parent, base_dir=root.name)}")
