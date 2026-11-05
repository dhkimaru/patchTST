#!/usr/bin/env bash
# 처음 한 번: 가상환경 만들기 → PyTorch(CUDA 12.4) → ald-lab 설치 → GPU 확인 → 테스트.
# 사용: cd ald-lab && bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"   # 드라이버 555 / CUDA 12.5 서버 기준

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
if ! python -c "import torch" 2>/dev/null; then
  pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
fi
pip install -e '.[train,dashboard,dev]'

python - <<'PY'
import torch
print(f"torch {torch.__version__} · CUDA 사용 가능 {torch.cuda.is_available()} · GPU {torch.cuda.device_count()}장")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
PY
PYTHONPATH=src python -m pytest -q

if [[ ! -f scripts/env.sh ]]; then
  cp scripts/env.example.sh scripts/env.sh
  echo "scripts/env.sh 를 만들었습니다. 데이터 경로를 확인하세요."
fi
echo "설치 완료. 다음: bash scripts/run_all.sh --smoke  (끝까지 도는지 확인)"
