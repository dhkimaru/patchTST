#!/usr/bin/env bash
# 데이터 개요(처음 한 번) → 실험들을 차례로 실행 → 모델 패키지 → HTML 보고서. 로그는 work/logs/.
#   bash scripts/run_all.sh --smoke                       # 작게 끝까지 도는지 확인 (수 분)
#   nohup bash scripts/run_all.sh > run_all.out 2>&1 &    # 본 실행 (SSH가 끊겨도 계속)
#   bash scripts/run_all.sh configs/experiments/g2_thickness.yaml   # 특정 실험만
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env.sh

SMOKE=()
CONFIGS=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then SMOKE=(--smoke); else CONFIGS+=("$arg"); fi
done
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(
    configs/experiments/g1_forecast_vg11.yaml     # VG11 예측 (이전 과제 재확인)
    configs/experiments/g2_thickness.yaml         # 두께 예측
    configs/experiments/g1_virtual_sensor.yaml    # 가상 센서
    configs/experiments/g1_forecast_params.yaml   # 중요 파라미터 예측
    configs/experiments/g3_core_vs_all.yaml       # 핵심 파라미터 vs 전체
  )
fi

mkdir -p "$ALD_WORK_ROOT/logs"
stamp=$(date +%Y%m%d-%H%M%S)
if [[ ! -f "$ALD_WORK_ROOT/data_overview/summary.json" ]]; then
  echo "== 원본 데이터 개요 (처음 한 번) =="
  if ! ald-build-overview 2>&1 | tee "$ALD_WORK_ROOT/logs/overview_$stamp.log"; then
    echo "!! 데이터 개요 실패 — scripts/env.sh 의 ALD_DATA_ROOT / ALD_EXSITU_XLSX 경로를 확인하세요"
  fi
fi
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "== $name $(date '+%F %T') =="
  if ! ald-train "$cfg" --gpus "$GPUS" ${SMOKE[@]+"${SMOKE[@]}"} 2>&1 | tee "$ALD_WORK_ROOT/logs/${name}_$stamp.log"; then
    echo "!! $name 실패 — 로그: $ALD_WORK_ROOT/logs/${name}_$stamp.log (다음 실험으로 넘어감)"
  fi
done
if [[ ${#SMOKE[@]} -eq 0 ]]; then
  echo "== 모델 패키지·보고서 =="
  bash scripts/export_models.sh || echo "!! 모델 패키지 실패 — bash scripts/export_models.sh 로 다시"
  bash scripts/make_report.sh || echo "!! 보고서 실패 — bash scripts/make_report.sh 로 다시"
fi
echo "== 끝 $(date '+%F %T') — 결과: $ALD_WORK_ROOT/experiments/*/*/summary.md · 보고서: $ALD_WORK_ROOT/reports/ald_report_latest.html =="
