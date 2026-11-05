#!/usr/bin/env bash
# 대시보드: 사내망에서 http://<서버 IP>:8501 (밖에서는 SSH 터널: ssh -L 8501:localhost:8501 <계정>@<서버> -p <포트>)
# 3. 하이퍼파라미터·학습 → ① 새 실험 실행에서 넣은 실험은 별도 작업자 프로세스(ald-queue worker)가 돌린다.
# 대시보드를 꺼도 학습은 계속되고, 진행은 ② 진행 상황 또는 `ald-queue list`로 본다.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
source scripts/env.sh
if ! command -v ald-queue >/dev/null; then  # 실험 실행(대기열) 기능이 들어간 코드로 바꾼 직후
  python -m pip install -q --no-deps -e .
fi
exec streamlit run src/ald_lab/dashboard/app.py --server.port "${PORT:-8501}"
