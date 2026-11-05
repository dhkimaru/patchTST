#!/usr/bin/env bash
# 노트북에서 바로 여는 HTML 보고서(파일 하나)를 만든다.
#   → work/reports/ald_report_<날짜_시각>.html  (+ 같은 내용의 ald_report_latest.html)
#   bash scripts/make_report.sh
#   bash scripts/make_report.sh --note "오늘: 가상 센서 16개 학습 완료" --note "내일: 시험 run 평가"
#   bash scripts/make_report.sh --no-examples     # 예측 예시 그래프 없이 집계 지표만
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env.sh
if ! command -v ald-report >/dev/null; then  # 새 명령이 추가된 코드로 바꾼 직후
  python -m pip install -q --no-deps -e .
fi
ald-report "$@"
