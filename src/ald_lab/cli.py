"""Command-line entry points."""

from __future__ import annotations

import argparse
import json

from .data.overview import build_data_overview
from .data.splits import DEFAULT_SPLIT, load_split
from .paths import resolve_paths


def build_overview_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="원본 trace 데이터 개요를 생성합니다 (대시보드 1단계).")
    parser.add_argument("--data-root", help="standard_trace_NNN.csv 폴더 (기본: $ALD_DATA_ROOT)")
    parser.add_argument("--exsitu", help="Ex-situ 워크북 경로 (기본: $ALD_EXSITU_XLSX)")
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    parser.add_argument("--workers", type=int, default=4, help="CSV 파싱 프로세스 수")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, exsitu_xlsx=args.exsitu, work_root=args.work_root)
    result = build_data_overview(paths, workers=args.workers)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
    return 0


def analyze_params_main(argv: list[str] | None = None) -> int:
    from .analysis.variability import run_variability_analysis

    parser = argparse.ArgumentParser(
        description="가상 센서 대상 분석: ALD 사이클 구간에서 레시피로 설명되지 않는 변동을 컬럼별로 측정합니다."
    )
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT), help="run 분할 YAML")
    args = parser.parse_args(argv)
    paths = resolve_paths(work_root=args.work_root)
    result = run_variability_analysis(paths, load_split(args.split))
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
    return 0


def train_main(argv: list[str] | None = None) -> int:
    from .training.runner import parse_devices, run_experiment

    parser = argparse.ArgumentParser(
        description="실험 계획서(YAML) 하나로 전처리 → 기준선 → 모델 학습 → 검증 비교까지 실행합니다."
    )
    parser.add_argument("config", help="configs/experiments/*.yaml")
    parser.add_argument("--gpus", default="auto", help="'0,1,2,3', 'cpu', 기본: 보이는 GPU 전부")
    parser.add_argument("--smoke", action="store_true", help="run 몇 개·epoch 2로 끝까지 도는지만 확인")
    parser.add_argument("--models", help="이 모델만: 이름 또는 label (쉼표 구분, 기준선은 항상 포함)")
    parser.add_argument("--targets", help="이 대상만 (쉼표 구분)")
    parser.add_argument("--dry-run", action="store_true", help="작업 목록만 출력")
    parser.add_argument("--rebuild-data", action="store_true",
                        help="work/prepared/의 정리된 데이터를 다시 만든다 (원본 CSV가 바뀌었을 때)")
    parser.add_argument("--data-root", help="CSV 폴더 (기본: $ALD_DATA_ROOT)")
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, work_root=args.work_root)
    split_csv = lambda v: [s.strip() for s in v.split(",") if s.strip()] if v else None  # noqa: E731
    exp_dir = run_experiment(
        args.config,
        paths,
        devices=parse_devices(args.gpus),
        smoke=args.smoke,
        only_models=split_csv(args.models),
        only_targets=split_csv(args.targets),
        dry_run=args.dry_run,
        rebuild=args.rebuild_data,
    )
    if exp_dir is not None:
        print(f"\n결과: {exp_dir}/summary.md")
    return 0


def final_test_main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from .training.runner import parse_devices, run_final_test

    parser = argparse.ArgumentParser(
        description="검증으로 고른 후보만 봉인된 시험 run에서 한 번 평가합니다 (최종 보고용)."
    )
    parser.add_argument("experiment_dir", help="work/experiments/<이름>/<시각>")
    parser.add_argument("--scope", choices=("per_model", "best"), default="per_model",
                        help="per_model: 모델 종류별 최고 1개씩, best: 대상별 최고 1개")
    parser.add_argument("--gpu", default="auto", help="'0' 또는 'cpu'")
    parser.add_argument("--data-root", help="CSV 폴더 (기본: $ALD_DATA_ROOT)")
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    args = parser.parse_args(argv)
    paths = resolve_paths(data_root=args.data_root, work_root=args.work_root)
    run_final_test(Path(args.experiment_dir), paths, device=parse_devices(args.gpu)[0], scope=args.scope)
    return 0


def report_main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from .report.collect import collect
    from .report.render import write_report

    parser = argparse.ArgumentParser(
        description="work/ 산출물로 노트북에서 바로 여는 HTML 보고서(파일 하나)를 만듭니다."
    )
    parser.add_argument("--out", help="출력 HTML (기본: work/reports/ald_report_<날짜_시각>.html)")
    parser.add_argument("--note", action="append", default=[], help="요약 탭에 넣을 오늘 메모 (여러 번 가능)")
    parser.add_argument("--experiments", help="이 실험만 (쉼표 구분, 기본: 끝난 실험 전부)")
    parser.add_argument("--include-smoke", action="store_true", help="smoke(축소 확인) 실험도 넣는다")
    parser.add_argument("--no-examples", action="store_true",
                        help="예측 예시 그래프(검증 run의 실측·예측 시계열)를 빼고 집계 지표만 넣는다")
    parser.add_argument("--work-root", help="출력 폴더 (기본: $ALD_WORK_ROOT 또는 ald-lab/work)")
    args = parser.parse_args(argv)
    paths = resolve_paths(work_root=args.work_root)
    only = [s.strip() for s in args.experiments.split(",") if s.strip()] if args.experiments else None
    data = collect(paths, include_smoke=args.include_smoke, only=only, examples=not args.no_examples,
                   notes=args.note)
    out = write_report(data, Path(args.out) if args.out else None, examples=not args.no_examples)
    print(f"보고서: {out}")
    print(f"최신본: {out.parent / 'ald_report_latest.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_overview_main())
