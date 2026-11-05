#!/usr/bin/env bash
# 실험실 대시보드: 보고서(HTML)를 실시간으로 띄우고, "하이퍼파라미터 탐색" 탭에서 조합을 골라 학습을 돌린다.
#   bash scripts/lab.sh                 # 포트: LAB_PORT 또는 PORT(scripts/env.sh), 없으면 5273
# 노트북에서: ssh -L 8501:localhost:<포트> <계정>@<서버> -p <SSH 포트> 를 연 채로 브라우저 http://localhost:8501
# 학습은 대기열 작업자(ald-queue worker)가 돌리므로, 이 창을 닫아도 학습은 계속된다.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env.sh
if ! command -v ald-lab-server >/dev/null; then  # 새 명령이 들어간 코드로 바꾼 직후
  python -m pip install -q --no-deps -e .
fi
exec ald-lab-server "$@"
