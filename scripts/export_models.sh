#!/usr/bin/env bash
# 끝난 실험마다 (smoke 제외, 가장 최근 것) 대상별 최고 모델을 패키지로 내보낸다.
#   → work/packages/<실험>_<시각>/<대상>__<모델>/  (가중치 + README + predict.sh / predict.bat) + .zip
#   bash scripts/export_models.sh                 # 대상별 최고 1개
#   bash scripts/export_models.sh --scope per_model   # 모델 종류별 최고 1개씩
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env.sh
if ! command -v ald-export-model >/dev/null; then
  python -m pip install -q --no-deps -e .
fi
for exp in "$ALD_WORK_ROOT"/experiments/*/; do
  name=$(basename "$exp")
  [[ "$name" == *_smoke ]] && continue
  latest=""
  for stamp in $(ls -1 "$exp" | sort -r); do
    if [[ -f "$exp$stamp/summary.json" ]]; then latest="$exp$stamp"; break; fi
  done
  [[ -z "$latest" ]] && continue
  echo "== $name ($(basename "$latest")) =="
  if ! ald-export-model "$latest" "$@"; then
    echo "!! $name 내보내기 실패 (다음 실험으로 넘어감)"
  fi
done
