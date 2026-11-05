"""Package the validation-selected model of each target: weights, settings and run scripts.

``ald-export-model work/experiments/<name>/<stamp>`` writes
``work/packages/<name>_<stamp>/<target>__<label>/`` holding the weight file(s),
``job.json`` (inputs, train-fitted scaler, window shape), the recipe template
when the model starts from it, a README, ``predict.sh`` (server) and
``predict.bat`` (Windows), plus a zip of the whole package. Thickness models
are refitted on every train + validation run (leave-one-run-out keeps no
single model). Packages hold weights derived from company data: they stay on
the server or company PCs and never go into Git.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..data.splits import load_split
from ..models.cards import model_card, task_card
from ..paths import PROJECT_ROOT, LabPaths, resolve_paths
from .config import BASELINE_LABELS, BASELINES
from .predict import experiment_for
from .selection import selected_jobs, weight_files

PACKAGE_FILES = (
    "job.json", "metrics_validation.json", "metrics_test.json", "history.json", "train.log",
    "model.pt", "classic.json", "ridge.npz", "lightgbm.joblib",
    "template.parquet", "template_step.parquet", "template_overall.json",
)


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in text)


def export_models(exp_dir: str | Path, paths: LabPaths, *, scope: str = "best", targets: list[str] | None = None,
                  out_root: Path | None = None, make_zip: bool = True) -> Path:
    exp_dir = Path(exp_dir).resolve()
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    picks = [p for p in selected_jobs(exp_dir, scope) if targets is None or p["target"] in targets]
    if not picks:
        raise ValueError(f"no finished candidates to export in {exp_dir}")
    package = out_root or paths.work_root / "packages" / f"{summary['experiment']}_{exp_dir.name}"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copyfile(exp_dir / "experiment.yaml", package / "experiment.yaml")
    cfg = experiment_for(picks[0]["job_dir"])
    index = []
    for pick in picks:
        folder = package / _safe(f"{pick['target']}__{pick['label']}")
        folder.mkdir()
        for name in PACKAGE_FILES:
            if (pick["job_dir"] / name).is_file():
                shutil.copyfile(pick["job_dir"] / name, folder / name)
        shutil.copyfile(exp_dir / "experiment.yaml", folder / "experiment.yaml")
        doc = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        if cfg.task == "thickness":
            _export_thickness(cfg, paths, doc, folder, smoke=bool(summary.get("smoke")))
        inputs = doc.get("inputs", [])
        (folder / "inputs.txt").write_text("\n".join(inputs) + "\n", encoding="utf-8")
        (folder / "README.txt").write_text(_readme(cfg, summary, exp_dir, pick, doc), encoding="utf-8")
        (folder / "predict.sh").write_text(_PREDICT_SH.replace("__LAB__", str(PROJECT_ROOT)), encoding="utf-8")
        (folder / "predict.sh").chmod(0o755)
        (folder / "predict.bat").write_text(_PREDICT_BAT, encoding="ascii", newline="\r\n")
        index.append({**{k: v for k, v in pick.items() if k != "job_dir"}, "folder": folder.name,
                      "weights": weight_files(folder)})
    (package / "package.json").write_text(json.dumps({
        "experiment": summary["experiment"], "stamp": exp_dir.name, "task": cfg.task, "scope": scope,
        "created": datetime.now().isoformat(timespec="seconds"), "models": index,
    }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [f"ALD 모델 패키지 — {summary['experiment']} ({exp_dir.name})", "",
             "대상별로 검증 run에서 가장 좋았던 모델입니다. 각 폴더의 README.txt를 보세요.", ""]
    lines += [f"- {m['folder']}: MAE {m['mae']:.4g} {m['unit']}".rstrip()
              + (f" (기준선 대비 {m['improvement_pct']:+.1f} %)" if m["improvement_pct"] is not None else "")
              for m in index]
    (package / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if make_zip:
        shutil.make_archive(str(package), "zip", root_dir=package.parent, base_dir=package.name)
    return package


def _export_thickness(cfg: Any, paths: LabPaths, doc: dict[str, Any], folder: Path, *, smoke: bool) -> None:
    import joblib

    from .runner import smoke_split
    from .vm import fit_final_model

    split = load_split(cfg.split_path())
    if smoke:
        split = smoke_split(split)
    spec = doc["job"]
    state = fit_final_model(cfg, paths, split, spec)
    joblib.dump(state, folder / "estimator.joblib")


def _readme(cfg: Any, summary: dict[str, Any], exp_dir: Path, pick: dict[str, Any], doc: dict[str, Any]) -> str:
    task = task_card(cfg.task, horizons=cfg.horizon_labels, context=cfg.context_seconds)
    card = model_card(pick["model"])
    unit = pick["unit"]
    base = BASELINE_LABELS.get(BASELINES[cfg.task], "기준선")
    lines = [
        f"ALD 모델 — {pick['target']} · {pick['label']}",
        f"실험 {summary['experiment']} ({exp_dir.name}) · 과제: {task['name']} · seed {pick['seed']}",
        "",
        "[Out: 무엇을 예측하나]",
        f"- {pick['target']} {unit}: {task['out']}".rstrip(),
        f"- {task['how']}",
        "",
        "[In: 입력]",
    ]
    if cfg.task == "thickness":
        lines += [f"- 신호 {len(doc.get('inputs', []))}개를 run마다 요약한 특징 {len(doc.get('features', []))}개",
                  "- 목록: inputs.txt"]
    else:
        lines += [f"- 신호 {len(doc['inputs'])}개, {doc['sample_seconds']:g}초 평균, 최근 {cfg.context_seconds:g}초 window"
                  " (목록: inputs.txt)",
                  "- 정규화 값은 학습 run에서만 계산해 job.json에 들어 있음"]
        removed = doc.get("removed_inputs") or {}
        for reason, cols in removed.items():
            lines.append(f"- 뺀 신호 ({reason}): {', '.join(cols[:12])}" + (" …" if len(cols) > 12 else ""))
    lines += [
        "",
        f"[Function: 모델 — {card['name']}]",
        f"- {card['what']}",
        f"- 강점: {card['good']}",
        f"- 약점: {card['bad']}",
        f"- 설정: {json.dumps(doc['job'].get('params') or {}, ensure_ascii=False) if doc['job'].get('params') else '기본값'}",
        "",
        "[검증 성능]",
        f"- MAE {pick['mae']:.4g} {unit}".rstrip()
        + (f" — {base} {pick['baseline_mae']:.4g} 대비 {pick['improvement_pct']:+.1f} %"
           if pick["improvement_pct"] is not None and pick["baseline_mae"] is not None else ""),
    ]
    if pick["mae_cycle"] == pick["mae_cycle"]:  # not NaN
        lines.append(f"- ALD 사이클 구간 {pick['mae_cycle']:.4g}, 그 외 {pick['mae_other']:.4g}")
    lines += [
        "- 모델 선택은 검증 run으로만 했다. 시험 run 결과는 ald-final-test가 metrics_test.json에 남긴다.",
        "",
        "[실행]",
        "- 서버:    bash predict.sh <trace CSV 파일 또는 폴더> [출력 CSV] [cpu|cuda:0]",
        "- Windows: predict.bat <trace CSV 파일 또는 폴더> [출력 CSV]   (ald-lab이 설치된 Python 필요)",
        "- 출력: 시점별 예측값 CSV. 같은 CSV에 실측값이 있으면 오차(MAE)도 화면에 출력",
        "- 재생 화면: <출력 CSV 이름>_player.html — 브라우저로 열고 ▶ 재생 (두께 모델은 없음)",
        "",
        "[파일]",
        "- model.pt: 딥러닝 가중치 (PyTorch state_dict + 구조 설정)",
        "- lightgbm.joblib / ridge.npz / classic.json: 고전 모델",
        "- estimator.joblib: 두께 모델 (학습+검증 run 전체로 다시 맞춘 것)",
        "- template*.parquet: 레시피 평균 (가상 센서가 출발하는 기준값)",
        "- job.json: 입력 목록, 정규화 값, window 설정 · history.json: epoch별 손실 · train.log: 학습 로그",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


_PREDICT_SH = """#!/usr/bin/env bash
# 이 폴더의 모델로 trace CSV를 예측한다 (예측 CSV + 재생 화면 <출력 이름>_player.html).
#   bash predict.sh <trace CSV 파일 또는 폴더> [출력 CSV] [cpu|cuda:0]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="${ALD_LAB:-__LAB__}"
if [[ -f "$LAB/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$LAB/.venv/bin/activate"
fi
if [[ $# -lt 1 ]]; then
  echo "사용법: bash predict.sh <trace CSV 파일 또는 폴더> [출력 CSV] [cpu|cuda:0]"
  exit 1
fi
python -m ald_lab.training.predict "$HERE" --csv "$1" --out "${2:-$HERE/predictions.csv}" --device "${3:-cpu}" --html
"""

_PREDICT_BAT = """@echo off
rem Predict with the model in this folder (prediction CSV + replay page <output name>_player.html).
rem   predict.bat <trace CSV file or folder> [output CSV]
rem Needs Python with ald-lab installed:  pip install -e "<ald-lab folder>[train]"
setlocal
set "HERE=%~dp0"
if "%~1"=="" (
  echo usage: predict.bat ^<trace CSV file or folder^> [output CSV]
  exit /b 1
)
set "OUT=%~2"
if "%OUT%"=="" set "OUT=%HERE%predictions.csv"
python -m ald_lab.training.predict "%HERE%." --csv "%~1" --out "%OUT%" --html
endlocal
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="실험에서 대상별 최고 모델을 가중치·설정·실행 파일(predict.sh/.bat)과 함께 패키지로 내보냅니다."
    )
    parser.add_argument("experiment_dir", help="work/experiments/<이름>/<시각>")
    parser.add_argument("--scope", choices=("best", "per_model"), default="best",
                        help="best: 대상별 최고 1개, per_model: 모델 종류별 최고 1개씩")
    parser.add_argument("--targets", help="이 대상만 (쉼표 구분)")
    parser.add_argument("--no-zip", action="store_true", help="zip을 만들지 않는다")
    parser.add_argument("--data-root", help="CSV 폴더 (두께 모델을 다시 맞출 때만 필요)")
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, work_root=args.work_root)
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    package = export_models(args.experiment_dir, paths, scope=args.scope, targets=targets, make_zip=not args.no_zip)
    print(f"패키지: {package}" + ("" if args.no_zip else f"\nzip:    {package}.zip"))
    print((package / "README.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
