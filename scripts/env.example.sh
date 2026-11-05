# 서버 경로 설정. 처음 한 번: cp scripts/env.example.sh scripts/env.sh 후 아래 경로만 고친다.
# (env.sh는 setup_server.sh가 없으면 자동으로 복사한다)

# 원본 trace CSV 74개가 있는 폴더 (standard_trace_001.csv ...)
export ALD_DATA_ROOT=/data1/Agent/patchtst-agent-lab/data/raw/standard_trace
# Ex-situ 워크북 (두께 과제와 대시보드 1페이지에 필요). 없으면 비워 둔다: export ALD_EXSITU_XLSX=
export ALD_EXSITU_XLSX=/data1/Agent/patchtst-agent-lab/data/<Ex-situ 워크북>.xlsx
# 산출물 폴더 (정리된 데이터·모델·결과·실행 기록). 기본: ald-lab/work
export ALD_WORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/work"
# 학습에 쓸 GPU 번호
export GPUS=0,1,2,3
# 대시보드에서 학습 실행(3. 하이퍼파라미터·학습 → ① 새 실험 실행)을 막으려면 주석을 푼다
# export ALD_DASHBOARD_RUN=0
