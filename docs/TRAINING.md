# 학습 실행 안내 (서버용)

`ald-train` 하나로 **전처리 → 기준선 → 모델 학습·하이퍼파라미터 탐색 → 검증 비교**가 실행된다.
각 단계는 실행 기록(journal)에 남아서 대시보드 2~4페이지에 시간순으로 표시된다.

## 1. 지원하는 과제와 모델

| 과제 (`task`) | 질문 | 기준선 | 평가 |
| --- | --- | --- | --- |
| `forecast` | 이 신호의 10·20·30초 뒤 값은? | 현재값 유지 | 검증 run의 물리 단위 MAE |
| `virtual_sensor` | 이 센서를 (같은 계열째) 달지 않았다면, 나머지 신호로 지금 값을 알 수 있나? | 레시피 평균 | 검증 run의 물리 단위 MAE |
| `thickness` | run 하나의 trace로 두께·균일도를 맞출 수 있나? | 학습 평균 | run 단위 leave-one-out |

| 모델 | 종류 | 메모 |
| --- | --- | --- |
| `persistence`, `template`, `mean` | 기준선 | 과제에 맞는 기준선이 자동으로 들어간다 |
| `ridge` | 선형 | window 요약 특성(마지막·평균·표준편차·최소·최대·최근 변화). 기준선과의 차이를 학습 |
| `lightgbm` | 트리 | 위와 같은 특성. LightGBM이 없으면 sklearn으로 대체 |
| `patchtst` | Transformer | `channel_mode: independent`(원형) / `mixing`(이전 최고 구조) |
| `itransformer` | Transformer | 신호 하나가 토큰 하나. 신호 사이 관계를 직접 학습 (가상 센서에 적합) |
| `dlinear` | 선형 시계열 | 추세/계절 분해 + 선형 |
| `gru` | RNN | 순환 신경망 기준 모델 |
| `pls` | 선형 (두께) | 부분최소제곱 |

## 2. 학습 시간 문제와 조치

이전 프로젝트는 RTX 4090에서 모델 하나에 **2~8시간** 걸렸다
(PatchTST 원형 3.2시간, FA1 2.3~3.6시간, S1 7.7시간). 처리 속도는 초당 약 3,600 window였고, GPU 메모리는 150 MB만 썼다.
GPU가 거의 놀고 있었다는 뜻이다. 원인과 조치는 다음과 같다.

| 원인 (이전 코드) | 조치 (이번 코드) |
| --- | --- |
| window를 CPU에서 한 개씩 잘라 `DataLoader`로 모음 (Python 호출이 window마다 한 번) | 데이터 전체를 GPU에 한 번 올리고, batch는 GPU에서 인덱스 한 번으로 자름 (`training/windows.py`) |
| batch 128 | batch 1024 (채널 독립 PatchTST는 메모리에 맞게 자동 축소) |
| epoch마다 겹치는 window 약 140만 개를 전부 학습 (이웃 window는 1초만 다름) | epoch마다 무작위 20만 개 (`windows_per_epoch`) |
| 매 epoch 검증 전체 평가 | 조기 종료는 고정된 검증 부분집합 5만 개로 판단하고, 전체 검증은 학습 후 한 번 |
| 실험마다 CSV를 다시 읽음 | run을 한 번 정리해 `work/prepared/`에 저장하고 재사용 |
| GPU 1장씩 수동 실행 | 실험 하나의 작업들을 GPU 4장에 자동 분배 (`--gpus 0,1,2,3`) |

GPU 속도는 서버에서 확인해야 한다. 학습 로그와 대시보드에 **초당 window 수, epoch 시간, 최대 GPU 메모리**가 기록된다.
이 세션의 CPU(4코어)로 실제 데이터 smoke를 돌려 동작은 확인했다.

### 이 세션에서 확인한 것 (CPU 4코어, GPU 없음 — 참고용)

- 합성 데이터 테스트 32개 통과 (Python 3.10, PyTorch 2.6.0 — 서버와 같은 버전).
- 실제 74 run으로 `g1_forecast_vg11.yaml --smoke`: 작업 10개 모두 성공.
- VG11 10·20·30초 예측, 학습량을 줄인 실행 (검증 run 11개, 177,575 window):

| 모델 | 검증 MAE (Torr) | 현재값 유지 대비 | ALD 사이클 구간 MAE | 사이클 외 MAE | 학습 시간 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 현재값 유지 | 2.016 | — | 0.364 | 2.704 | — |
| Ridge | 0.875 | +56.6 % | 0.896 | 0.866 | 0.3분 |
| LightGBM | **0.147** | **+92.7 %** | 0.261 | 0.100 | 1.6분 |
| PatchTST 혼합 (12 epoch × 4만 window, 서버 설정의 1/25) | 0.513 | +74.6 % | 0.262 | 0.618 | 6.4분 |
| DLinear | 0.566 | +71.9 % | 0.462 | 0.609 | 1.2분 |

  개선 대부분은 사이클 밖(배기·대기압 복귀) 압력 전환에서 나온다. ALD 사이클 안에서는 LightGBM·PatchTST가
  현재값 유지보다 약 28 % 낫다. 보고할 때 두 구간을 나눠서 보여 준다.
- PatchTST 혼합(48만 파라미터)은 CPU 4코어에서 초당 1,653 window를 처리했다. 이전 GPU 실행(초당 3,620)의 절반 수준이다.
- 가상 센서 (같은 계열 센서를 모두 뺌, 공정 구간, 검증 run 기준):

| 대상 | 레시피 평균 | Ridge | LightGBM | DLinear (L1, 8 epoch × 4만) |
| --- | ---: | ---: | ---: | ---: |
| TempAct_C (C zone 열전대 5개 제외) | 1.911 ℃ | 2.140 (−12 %) | **0.141 (+92.6 %)** | 1.190 (+37.7 %) |
| VG11 (압력계 9개 제외) | 2.097 Torr | 2.689 (−28 %) | 2.080 (+0.8 %) | **1.610 (+23.2 %)** |

  온도는 인접 zone·히터 신호로 거의 그대로 추정된다. 압력계가 하나도 없으면 압력은 레시피 이상으로 맞추기 어렵고,
  시간 패턴을 보는 DLinear만 23 % 나아졌다. iTransformer는 CPU에서 epoch당 4분이라 학습되는 것(검증 손실 감소)만 확인했다.
- 두께 예측 (`g2_thickness.yaml`, 58 run leave-one-out, 41초): LightGBM이 학습 평균 대비 두께 +17 %(0.200 → 0.165 Å),
  U/F +17 %, WtoW +29 %. Ridge·PLS는 평균보다 나쁘다(run 수 대비 특성이 많음).

## 3. 서버에 넣는 방법 (폴더 구조)

원본 데이터는 지금 있는 자리 그대로 두고, `ald-lab` 폴더 하나만 옆에 놓는다.

```
/data1/Agent/
├── patchtst-agent-lab/                 이전 프로젝트 (그대로 둠)
│   └── data/raw/standard_trace/        원본 CSV 74개 → ALD_DATA_ROOT
├── data/<Ex-situ 워크북>.xlsx           Ex-situ 워크북 (위치는 자유) → ALD_EXSITU_XLSX
└── ald-lab/                            ← 새로 넣는 폴더 (zip 압축 해제 또는 git)
    ├── configs/  src/  tests/  docs/  pyproject.toml
    ├── scripts/
    │   ├── setup_server.sh             처음 한 번: 가상환경·설치·GPU 확인·테스트
    │   ├── env.example.sh → env.sh     데이터 경로 (env.sh만 고치면 됨)
    │   ├── run_all.sh                  개요 → 실험들을 차례로 실행
    │   └── dashboard.sh                대시보드 (:8501)
    ├── .venv/                          setup_server.sh가 만듦
    └── work/                           실행하면 생김 (회사 데이터 — 외부 반출 금지)
        ├── cache/  prepared/           정리된 데이터 (재사용)
        ├── data_overview/  journal/    데이터 개요, 실행 기록
        ├── experiments/<실험>/<시각>/  결과 (summary.md 등)
        └── logs/                       실행 로그
```

넣는 방법은 둘 중 하나:

- **zip**: `ald-lab_<버전>.zip`을 `/data1/Agent/`에 올리고 `unzip ald-lab_<버전>.zip` → `/data1/Agent/ald-lab/`
- **git**: `git clone https://github.com/dhkimaru/patchtst-agent.git` 후 `git checkout <작업 브랜치>`
  (이 경우 `ald-lab`은 `patchtst-agent/ald-lab/`에 생긴다. 아래 명령은 그 폴더에서 실행)

## 4. 실행 순서

```bash
cd /data1/Agent/ald-lab

# 1) 처음 한 번: 설치 (PyTorch 2.6.0+cu124, 이전 .venv-gpu와 같은 버전) → GPU 확인 → 테스트
bash scripts/setup_server.sh
#    끝에 "torch 2.6.0+cu124 · CUDA 사용 가능 True · GPU 4장"이 나오면 정상
vi scripts/env.sh          # ALD_DATA_ROOT, ALD_EXSITU_XLSX 경로 확인

# 2) smoke: run 6개·epoch 2로 모든 실험이 끝까지 도는지 확인 (GPU에서 10분 안팎)
bash scripts/run_all.sh --smoke

# 3) 본 실행: SSH가 끊겨도 계속 돈다. 진행은 tail로 확인
nohup bash scripts/run_all.sh > run_all.out 2>&1 &
tail -f run_all.out

# 4) 대시보드 (사내망 http://<서버 IP>:8501, 밖에서는 SSH 터널 — 8장 참고) → 3. 하이퍼파라미터·학습
#    여기서 하이퍼파라미터를 골라 학습을 돌리고 누적 기록을 본다 (8장)
bash scripts/dashboard.sh

# 5) 최종 후보가 정해진 뒤에만: 봉인된 시험 run 평가 (모델 종류별 최고 1개씩)
source .venv/bin/activate && source scripts/env.sh
ald-final-test work/experiments/g1_forecast_vg11/<시각 폴더>

# 6) 모델 패키지와 보고서 (run_all.sh 끝에 자동으로도 만든다) — 7장 참고
bash scripts/export_models.sh
bash scripts/make_report.sh --note "오늘 한 일"
```

특정 실험만: `bash scripts/run_all.sh configs/experiments/g2_thickness.yaml`.
GPU를 일부만 쓰려면 `scripts/env.sh`의 `GPUS=0,1`. 원본 CSV가 바뀌면 `ald-train ... --rebuild-data`.
명령을 직접 쓸 때: `ald-train <계획서> --gpus 0,1,2,3 [--smoke] [--models patchtst,lightgbm] [--targets VG11] [--dry-run]`.

준비된 실험 계획서:

| 파일 | 내용 | 작업 수 |
| --- | --- | --- |
| `g1_forecast_vg11.yaml` | VG11 10·20·30초 예측. 이전 프로젝트와 같은 과제·입력으로 8개 모델 비교, PatchTST 혼합은 4개 조합 탐색, seed 3개 | 29 |
| `g1_forecast_params.yaml` | 중요 파라미터 8개의 10·20·30초 예측 | 48 |
| `g1_virtual_sensor.yaml` | 가상 센서 후보 16개(N2-3은 역할 확인 후 추가). 같은 계열 센서를 모두 뺀 상태에서 추정 | 96 |
| `g3_core_vs_all.yaml` | 같은 가상 센서 과제를 입력 세트(전체/중요도 A·B/A)만 바꿔 비교 | 64 |
| `g2_thickness.yaml` | 두께·U/F·WtoW, leave-one-run-out | 33 |

`run_all.sh`는 위 순서(VG11 예측 → 두께 → 가상 센서 → 중요 파라미터 예측 → 핵심 파라미터 비교)로 실행한다.

## 5. 결과물

```
work/
  prepared/                       정리된 run 배열 (재사용 캐시)
  experiments/<이름>/<시각>/
    summary.md / summary.csv      검증 비교표 (집계값만 — 공유해도 되는 파일)
    test_summary.md               ald-final-test 결과
    results.csv                   작업별 지표
    jobs/<작업>/
      model.pt                    딥러닝 모델 (PyTorch state_dict + 구조 설정)
      ridge.npz / lightgbm.joblib 고전 모델
      job.json                    입력 신호, 정규화 통계, 설정 (재현·추론에 필요)
      history.json, train.log     epoch별 손실, 학습 로그
      metrics_validation.json     세부 지표 + 강점·약점
      predictions_sample.parquet  검증 run 하나의 예측 (대시보드 그림용)
  journal/journal.jsonl           모든 단계의 시간순 기록
  packages/<실험>_<시각>/          모델 패키지 (ald-export-model) + 같은 이름의 .zip
    <대상>__<모델>/                가중치, job.json, README.txt, predict.sh, predict.bat
  reports/ald_report_<날짜_시각>.html   보고서 (ald-report), 최신본은 ald_report_latest.html
```

**저에게 결과를 공유할 때는 `summary.md`(그리고 `test_summary.md`)의 내용만 붙여 주면 된다.** 집계 수치뿐이라
원본 데이터나 샘플 단위 예측이 들어 있지 않다. `work/`는 Git에 올라가지 않는다.

### 모델 파일 형식 (Keras와의 차이)

처음의 `patchTST_v2.py`는 TensorFlow/Keras였다. 대상 컬럼마다 `patchtst_model_v5/<loss>_<window>_patchtst_<컬럼>_main.keras`를,
스케일러는 `scaler_v5/*.pkl`(joblib)로 따로 저장했고, 정규화 범위(VG11 0.7~771.6 등)는 코드에 직접 적혀 있었다.
`patchtst-agent-lab`부터는 PyTorch(`.pt` 체크포인트)를 쓴다.
이번 코드도 PyTorch이고, `model.pt` 하나에 가중치와 모델 구조 설정이 함께 들어 있다.
정규화 통계와 입력 신호 목록은 `job.json`에 있다. 다시 불러오는 방법:

```python
from pathlib import Path
from ald_lab.training.runner import load_predictor

doc, model = load_predictor(Path("work/experiments/g1_forecast_vg11/<시각>/jobs/<작업>"), device="cuda:0")
print(doc["inputs"], doc["scaler"]["mean"][:3])   # 입력 신호와 학습 run 기준 정규화 통계
```

## 6. 매일 보고: HTML 보고서

`bash scripts/make_report.sh`가 `work/` 결과로 **HTML 파일 하나**를 만든다
(`work/reports/ald_report_<날짜_시각>.html`, 같은 내용의 `ald_report_latest.html`). 그림·표·스크립트가 모두 파일 안에 있어서
노트북에 복사하면 인터넷·서버 연결 없이 브라우저로 열린다. 인쇄(PDF로 저장)하면 모든 탭이 차례로 나온다.
화면은 어둡게 열리고 오른쪽 위 "밝게/어둡게" 버튼으로 바꾼다(기본값은 `report/render.py`의 `DEFAULT_THEME`). 인쇄는 항상 밝게 나온다.

| 탭 | 내용 |
| --- | --- |
| 요약 | 끝난 실험 수, 목적별 결과(기준선 대비 개선율, 미래 값 예측은 10·20·30초 열과 시험 run 값), 끝나지 않은 실험, 최근 작업일 기록, 메모(`--note`) |
| In (입력) | 데이터가 줄어든 흐름(run·행·신호·두께 run), 원본 데이터, 제외한 run·컬럼과 이유, 품질 점검 항목별 처리, run 번호별 분할 그림, 전처리 단계(순서도·표), 검증 run 하나를 실제 전처리 함수로 통과시킨 그림(구간 표시, 0.5초→1초 평균), window 규칙, 학습 run 정규화 값, 실험·대상별 입력 수와 뺀 신호 |
| Out (출력) | 실험마다 카드 하나: 출력, 예측 하나를 시간축에 그린 그림(모델이 보는 입력 구간 → 출력), 대상, 기준선, 평가 방법 |
| Function (모델) | 모델 지도(기준선 · 고전 ML · 딥러닝, 이번 사용 여부와 최고로 뽑힌 수), 과제별로 무엇을 학습하나(변화량·보정량·값 직접), 모델 × 실험 선택 표(★ 뽑힌 대상), 묶음별 비교 표와 구조 그림(고전 ML 요약 특징 파이프라인, 두께 run 요약, 딥러닝 공통 틀, PatchTST·독립/혼합, iTransformer, DLinear, GRU). 그림의 숫자는 `models/defaults.py`(`report/archs.py`) |
| Code Flow Chart | 검토용 설계 자료와 코드 지도. ① 단계·명령·결과물 요약 → ② 단계별 상세 설계 6단계(원본 개요 · 학습 데이터 준비 · 입력·정규화·window · 학습 · 평가·선택·시험 run · 활용): 단계마다 이번 데이터의 숫자, 입력→출력 표, 처리 순서와 그 일을 하는 함수(파일:줄), 핵심 코드 발췌, 설계 결정(고른 방법·고르지 않은 방법·이유), 자동 테스트, 검토 포인트 → ③ 단계 사이의 산출물 → ④ 데이터 누설·재현성 점검표(위험·막는 방법·코드·테스트) → ⑤ 명령별 함수 호출 그림(`src/ald_lab/`을 AST로 읽음) → ⑥ 여러 명령이 함께 쓰는 함수 → ⑦ 폴더끼리 부르는 관계, 공통 도우미, 파일별 역할 |
| Hyper Parameter | ① 읽는 법(%·seed·확실/불분명/반복 없음) → ② 결론 표(크게 영향 · 작게 영향 · 차이 확인 안 됨 · 반복 없음 × 이번/이전 실험) → ③ 이번 실험의 설정 변경 효과(실험마다 그림 하나, 기준값 → 바꾼 값, 여러 대상·모델은 평균) → ④ 이전 실험(2026-08): 단계별 흐름, 단계 순서의 막대, 코드 풀이 표(N02·S1→S2 등을 쉬운 말로) → ⑤ 탐색하지 않은 것 → ⑥ 실험별 상세(접힘: 설정, 모델별 탐색 여부, 설정 하나를 바꾼 효과, 시도한 조합 전부, 학습 곡선). 비교의 기준값은 고정이다: 입력 세트는 all, 숫자 설정은 작은 값, 모델 구성은 실험 YAML 순서 |
| 하이퍼파라미터 탐색 | 모델 선정 단계(탐색 → 전체 적용 → 대안)의 현황: 탐색 조합 순위, 대상별 판정(PatchTST 최고 · LightGBM 참고 · 기준선 · 최종). `scripts/lab.sh`로 서버에서 열면 조합을 골라 학습을 돌리는 화면과 진행 상황, 2·3단계 버튼이 켜진다 (8장) |
| 성능표 | 대상별 전체 순위(MAE, 10·20·30초, 개선율, ALD 사이클/그 외, 시간, 속도), 개선율 막대, 강점·약점, 예측 예시 그래프, 시험 run 결과(예측 거리별) |
| 가중치·실행 파일 | 대상별 최고 모델의 가중치 파일(크기·sha256), 패키지 상태, 실행 명령 |
| 추론 시연 | 대상마다 학습한 모델별 최고 설정(버튼: 순위 · 모델 · 검증 MAE, PatchTST로 열림)이 검증 run에서 낸 예측을 ▶ 재생 버튼으로 시간 순서대로 다시 보여 주는 재생기(`report/player.py`): 모델 입력 구간, 그 순간의 예측(+10·20·30초 / 지금 값), 실측이 오면 오차와 누적 MAE. 사이클 10분 확대(1초)와 run 전체, 속도·위치 조절, 예측 거리 선택. 새 CSV는 `ald-predict … --html`로 같은 재생기를 만든다 |
| 특이사항 | 학습 시간을 줄인 방법: 이전 대비 숫자(초당 window, 모델당 시간, GPU 메모리, 동시 사용 배속), window·patch를 GPU에서 자르는 과정(그림·단계 설명·이전/이번 코드), GPU 여러 장에 작업을 나누는 방식(대기열 그림), 진행 기록으로 그린 장치별 실제 타임라인, 모델별 처리 속도 |
| 이전 실험 | 한눈에 보는 흐름(2026-06 처음 Keras 코드 → 2026-07 파이프라인 재구축 → 2026-08 PatchTST 단계 실험 → 이번), VG11 예측 오차의 변화 막대(같은 시험 run), 주제별 이전/이번 비교 카드, 시기별 기록(단계 카드 · 개선율 · 배운 것/한계/못 한 것, 설정·전체 테스트 목록·통계 검정·실행 명령은 접힘) |
| 다음 실험 | 진행 원칙, 우선순위별 로드맵 보드(1 먼저 · 2 다음 · 3 여유 있을 때, 카테고리 색), 카테고리별 카드(한 줄 목표 · 채택 기준 · 대상 · 비용 · 비교할 설정 또는 탐색 공간, 근거·방법·주의는 접힘). 카테고리: 입력·특징, 중요도 A 등급 활용, 하이퍼파라미터 조합 최적화, 과제 정의, 이상 감지, 두께 예측, 신뢰도·검증 방법 |

```bash
bash scripts/make_report.sh                                  # 끝난 실험 전부 (smoke 제외, 실험마다 가장 최근 것)
bash scripts/make_report.sh --note "가상 센서 16개 완료" --note "내일: 시험 run 평가"
bash scripts/make_report.sh --no-examples                    # 예측 예시 그래프(검증 run 시계열)를 빼고 집계 지표만
ald-report --experiments g1_forecast_vg11,g2_thickness       # 일부 실험만
```

노트북으로 가져오기: WinSCP·MobaXterm으로 `work/reports/ald_report_latest.html`을 내려받거나,
대시보드(SSH 터널 `http://localhost:8501`) 첫 화면의 **보고서 내려받기** 버튼을 누른다.
예측 예시 그래프와 추론 시연 탭에는 검증 run 하나의 실측 시계열이 들어 있으니, 회사 밖으로 보낼 때는 `--no-examples`로 만든다
(두 곳 모두 빠지고 집계 지표만 남는다).

이전 실험 탭은 `configs/history/` 안의 YAML 파일(집계 지표만)을 모두 읽어 `order` 순으로 보여 준다. 지금은 2026-08 PatchTST 단계 실험(`patchtst_2026_08.yaml`, 전체 테스트 목록·통계 검정 포함)과 2026-06~07 초기 Keras 작업(`legacy_2026_06.yaml`)이 있고, 맨 위에 이번 결과와의 주제별 비교표가 자동으로 붙는다. 새 단계가 끝나면 같은 형식으로 파일을 추가하면 된다.

Code Flow Chart의 단계별 설명은 `src/ald_lab/report/design.py`에 있다. 설명이 가리키는 함수(`module:qualname`)와 테스트 이름은
테스트(`test_design_notes_point_at_existing_code_and_tests`)가 실제 코드와 대조하므로, 함수 이름을 바꾸거나 지우면 설명도 함께
고쳐야 한다. 줄 번호와 코드 발췌는 보고서를 만들 때 소스에서 읽는다.

다음 실험 탭은 `configs/plan/next_experiments.yaml`(집계 수치와 설명만)을 읽는다. 후보마다 `id`·`title`·`priority`(1~3)·`status`(계획 | 진행 중 | 완료 | 보류)·`question`이 필요하고, 카드에는 `goal`(한 줄 목표)이 보인다. 하이퍼파라미터 후보는 `variants_title: 탐색 공간`으로 범위를 적는다. 실험이 끝나면 `status`와 `result`를 고치면 된다. 이전 실험 탭의 흐름 칸은 history 파일의 `timeline`, 오차 변화 막대는 `runs`에서 `journey: true`인 행을 쓴다.

## 7. 모델 패키지와 새 데이터 예측

`bash scripts/export_models.sh`는 끝난 실험마다 대상별 최고 모델(검증 기준)을 `work/packages/<실험>_<시각>/`로 내보낸다.

```
work/packages/g1_virtual_sensor_<시각>/
  README.txt                    대상별 성능 목록
  TempAct_C__lightgbm/
    lightgbm.joblib (또는 model.pt, ridge.npz, estimator.joblib)   가중치
    template*.parquet           레시피 평균 (가상 센서의 출발점)
    job.json                    입력 목록, 학습 run 정규화 값, window 설정
    README.txt                  Out / In / 모델 / 검증 성능 / 실행 방법
    predict.sh  predict.bat     실행 파일 (서버 / Windows)
    inputs.txt, history.json, train.log, metrics_validation.json, experiment.yaml
g1_virtual_sensor_<시각>.zip
```

```bash
# 서버: CSV 파일 하나 또는 standard_trace_NNN.csv가 있는 폴더
bash work/packages/<…>/TempAct_C__lightgbm/predict.sh /data1/Agent/new_runs/ [출력.csv] [cpu|cuda:0]
ald-predict work/experiments/<실험>/<시각>/jobs/<작업> --csv <CSV> --out pred.csv   # job 폴더에서 바로
ald-predict <패키지>/TempAct_C__lightgbm --csv <CSV 또는 폴더> --html          # + pred_player.html (재생 화면)
ald-export-model work/experiments/<실험>/<시각> --scope per_model                    # 모델 종류별 최고 1개씩
```

- 출력 CSV: `run_id, timestamp(예측 시점), step_name, in_cycle, pred…` (CSV에 실측값이 있으면 `actual…`과 MAE 출력).
  미래 값 예측은 `pred_+10s`처럼 horizon마다 한 열이다.
- `--html`을 붙이면 `<출력 이름>_player.html`도 만든다: 보고서 추론 시연 탭과 같은 재생기(run마다 하나, 파일 하나로 오프라인에서 열림).
  CSV에 대상 신호가 있으면 실측과 오차도 보인다. 두께 모델은 run 하나에 값 하나라 만들지 않는다.
- 가상 센서는 CSV에 그 센서 열이 **없어도** 예측한다 (입력으로 쓴 신호만 있으면 됨).
- 검증 run CSV를 넣으면 검증 때와 같은 오차가 나온다 (같은 window·정규화·레시피 평균을 쓴다).
- 두께 모델은 leave-one-run-out으로 평가만 했으므로, 내보낼 때 학습+검증 run 전체로 다시 맞춰 `estimator.joblib`로 저장한다.
- Windows에서 `predict.bat`을 쓰려면 PC에 Python과 ald-lab(`pip install -e "ald-lab[train]"`)이 있어야 한다.
- 가중치는 회사 데이터로 학습한 결과물이다. 서버·사내 PC에만 두고 Git이나 외부로 보내지 않는다.

**코드를 새 버전으로 바꿀 때**: 학습(`run_all.sh`)이 돌고 있는 동안에는 zip을 덮어쓰지 않는다
(실행 중인 스크립트와 다음 실험이 새 코드를 섞어 읽는다). `grep "^== " run_all.out`의 마지막 줄이 `== 끝`이 된 뒤
`cd /data1/Agent && unzip -o ald-lab_<버전>.zip`. `scripts/env.sh`, `.venv`, `work/`는 zip에 없어서 그대로 남는다.
새 명령(`ald-report` 등)은 스크립트가 처음 실행될 때 자동으로 등록된다.

## 8. 실험실 대시보드 (보고서 HTML + 하이퍼파라미터 탐색)

`bash scripts/lab.sh`가 보고서와 **같은 HTML 화면**을 서버에서 실시간으로 띄운다 (`ald-lab-server`, 포트는
`LAB_PORT` 또는 `scripts/env.sh`의 `PORT`, 없으면 5273). 실험이 끝날 때마다 화면이 다시 만들어지고, 새 탭
**하이퍼파라미터 탐색**에서 조합을 골라 학습을 돌린다. 노트북에서는 SSH 터널로 연다:

```bash
# 서버
cd /data1/Agent/ald-lab && bash scripts/lab.sh          # tmux 안에서 띄워 두면 창을 닫아도 계속 뜬다
# 노트북 (창을 열어 둔 채)
ssh -L 8501:localhost:5273 <계정>@<서버 주소> -p <SSH 포트>
# 브라우저: http://localhost:8501  → 하이퍼파라미터 탐색 탭
```

하이퍼파라미터 탐색 탭:

| 부분 | 하는 일 |
| --- | --- |
| 새 실험 | 기준 실험(`configs/experiments/*.yaml`)의 데이터·대상 설정을 가져온다. **탐색용**(대표 대상 3개 · seed 42, 43 · 짧은 학습) / **정식**(모든 대상 · seed 3개 · 기준 실험의 학습 설정) 버튼이 대상·seed·학습 설정을 한 번에 채운다. 대상·모델 칩을 고르고(기본 PatchTST), 모델마다 하이퍼파라미터 칸에 값을 넣는다. **쉼표로 여러 값 = 조합**이고, "조합 방식"은 기본이 모든 조합이며 **무작위 N개**(가능한 조합 중 N개만 뽑음)도 고를 수 있다. window 길이를 여러 개 넣으면 window마다 실험이 하나씩 들어간다. 입력 세트 · seed · 학습 설정(epoch · patience · epoch당 window · batch · 손실) · GPU · 메모. 입력할 때마다 작업 수 · 조합 수 · 예상 시간과 그 근거(모델별 작업 수 × 지난 학습 시간 중앙값, 학습만)와 실험 YAML이 바로 나온다. 신호 처리 independent를 고르면 GPU 메모리 경고가 나온다. **▶ 대기열에 넣고 학습** |
| 실험 계획 파일로 한꺼번에 넣기 | 차수마다 `### 15차: …` 제목과 `\| 순서 \| 칸 \| 입력 \|` 표(양식의 칸 이름 그대로)를 적은 마크다운 파일을 올리거나 붙여 넣으면, 차수마다 작업 수 · 조합 · 실험 수 · 예상 시간 · 문제를 보여 주고 **▶ 모두 대기열에 넣기**로 적힌 순서대로 넣는다. 기준 실험 줄이 있는 표만 실험으로 읽는다. 하나라도 문제가 있으면 아무것도 넣지 않는다 |
| 진행 상황 | 요청마다 상태(대기 · 실행 중 · 완료 · 실패 · 취소), 끝난 작업 수 막대, 작업당 평균 시간과 남은 시간, 학습 로그, 취소. 실행 중 · 대기가 위에 오고 4건이 넘으면 목록 안에서 스크롤. 끝난 요청은 **목록에서 지우기**(하나씩 또는 한꺼번에; 결과는 누적 기록 · 탐색 결과에 남음). 5초마다 새로 고친다 |
| 탐색 결과 | 탐색 목록(메모 · 과제 · 언제 · 크기 · 최고 개선율 · 확실한 효과 수)에서 하나를 고르면 그 탐색만 아래에 보인다. 탐색(대시보드 요청은 메모별, 1단계 파일은 이름별)마다 조합 순위(개선율 평균 ± seed 편차 · 사이클 안 개선 · 작업당 시간)와 **설정별 효과**: 설정을 바꿨을 때 개선율이 몇 %p 바뀌는지(대상마다 평균을 빼고 비교), 같은 조합을 seed만 바꿔 잰 잡음과 비교한 판정(확실 · 작음 · 잡음 안 · 반복 없음), 대상별 방향, 다음에 할 일(범위 넓히기 · 고정) |
| 모델 선정 단계별 현황 | 과제(예측 · 가상 센서)마다 대상 수 · 기준선보다 확실히 좋음 · 기준선을 못 넘음, 대상별 판정 표(PatchTST 최고 · 기준선 · 기준선 대비 · 판정 · LightGBM 참고 · 다른 모델 최고 · 최종). 버튼: **② 상위 k개 조합을 모든 대상에 적용**, **③ 대안 실행** |

**모델 선정 단계** (`ald_lab/stages.py`): PatchTST를 먼저 충분히 탐색해 모든 대상에 쓰고, 기준선을 확실히 넘지 못한 대상에만 다른 모델을 쓴다.
1. 탐색 — 대시보드 **탐색용**으로 넣은 요청(`<기준 실험>__hp`, 메모가 같으면 한 탐색) 또는 `configs/experiments/p1_patchtst_search_forecast.yaml`, `p1_patchtst_search_vs.yaml`: 대표 대상 3개씩, PatchTST 조합 16개 × 입력 세트(all · A_plus), 짧은 학습(epoch 15 · window 10만). 실험 YAML의 `search: {trials: 16, seed: 0}`이 `grid`에서 조합을 뽑는다 (head 수가 d_model을 못 나누거나 patch 간격이 patch보다 긴 조합은 뺌).
2. 전체 적용 — 대상마다 가장 좋은 조합을 1.0으로 둔 상대 MAE를 대상 평균해 순위를 매기고, 상위 k개 조합을 기준 실험(`g1_forecast_params` 8개 · `g1_virtual_sensor` 16개)의 모든 대상에 정식 학습(seed 3개) → `p2_patchtst_all_<과제>`. 탐색 순위는 대시보드 탐색(`__hp`)과 1단계 파일을 모두 합쳐 매긴다.
3. 대안 — PatchTST 최고(같은 설정의 seed 평균)가 기준선보다 seed 표준편차의 2배를 넘게 좋으면 **PatchTST 채택**, 아니면(기준선보다 나쁘거나 차이가 seed 흔들림 안) "대안 필요". 그 대상만 iTransformer · DLinear · LightGBM → `p3_fallback_<과제>`. LightGBM과의 차이는 표에 참고로만 나오고 판정에는 쓰지 않는다 (2026-10-29 확정: 이 연구는 PatchTST를 검증하는 것이 요지).

2·3단계 결과는 정식 실험 이름이라 보고서의 성능표 · 추론 시연에도 들어간다. 탭에서 직접 고른 실험은
`<기준 실험>__hp`로 저장되어 보고서 본문에서는 빠지고, Hyper Parameter ⑦ 누적 기록과 `work/reports/experiment_records.csv`에 들어간다.

동작 방식:
- 화면은 요청을 `work/queue/<요청 id>/`(실험 YAML + `request.json`)에 쓰고 **작업자 프로세스**를 깨우기만 한다.
  작업자(`ald-queue worker`)가 `ald-train`을 요청 하나씩 차례로 돌리므로 GPU를 두 요청이 나눠 쓰지 않는다.
  브라우저나 `lab.sh`를 꺼도 학습은 계속된다.
- 터미널에서 `run_all.sh`가 돌고 있으면 경고가 나온다 (GPU를 함께 쓰게 됨).
- 서버는 기본으로 127.0.0.1에서만 받는다 (SSH 터널로 접속). 사내망에 열려면 `ALD_LAB_HOST=0.0.0.0`.
  실행을 막으려면 `scripts/env.sh`에 `export ALD_DASHBOARD_RUN=0` — **보기 전용**: 양식·취소 버튼이 사라지고 실행 요청은 거절되며,
  화면에 서버 경로를 내보이지 않는다. 여러 사람에게 보여 주는 주소는 이렇게 띄운다.
- 새 결과가 반영되면 열린 화면 위에 "새 결과가 반영됐습니다" 막대가 뜬다 (20초마다 확인).
- 파일로 연 보고서(`ald_report_latest.html`)에서는 이 탭이 기록만 보여 주고 실행 기능은 꺼져 있다.

터미널에서도 같은 대기열과 단계를 쓸 수 있다:

```bash
ald-queue submit configs/experiments/p1_patchtst_search_forecast.yaml --gpus 0,1,2,3 --report   # 1단계
ald-queue first  --task forecast                    # 대상별 PatchTST 최고 · 기준선 대비 · 판정 (LightGBM 참고)
ald-queue stage2 --task forecast --k 3 --report     # 2단계: 상위 3개 조합을 모든 대상에
ald-queue stage3 --task forecast --report           # 3단계: 대안 필요 대상에 다른 모델
ald-queue list                                      # 요청 목록과 진행
ald-queue cancel <요청 id>                          # 대기 취소 / 실행 중이면 멈춤
```

Streamlit 대시보드(`bash scripts/dashboard.sh`)도 그대로 있다. 3페이지에 같은 대기열을 쓰는 실행 화면과 누적 기록 그림이 있다.

## 9. 실험 계획서 직접 쓰기

```yaml
name: my_experiment            # 영문·숫자·_-. 만
task: forecast                 # forecast | virtual_sensor | thickness
targets: [VG11]
data:
  split: split_v2              # configs/splits/<이름>.yaml
  sample_seconds: 1.0          # 0.5(원본) 또는 그 배수 (배수면 평균)
  bin_align: run               # 묶음 시작: run(구간 시작부터 2행씩, 기본) | step(step이 바뀌면 새 묶음, 두 step이 섞이지 않음)
  region: all                  # window 끝 위치: all | process(START~END) | cycle(ALD 사이클) | noncycle(공정 중 사이클 밖)
  inputs: all                  # all | core | core_ab | A_all | A_plus | legacy_primary | [컬럼 목록]
                               # core = 중요도 A 대표 14개, A_all = A 등급 전체 62개,
                               # A_plus = A 전체 + 등급 없는 압력계·APC·RF 매칭 위치 (74개)
  # input_sets: {all: all, core: core}   # 여러 입력 세트를 한 번에 비교 (G3)
  exclude: []                  # 추가로 뺄 컬럼
  exclude_target_family: true  # 가상 센서: 대상과 같은 계열 센서를 모두 뺀다
window:
  context_seconds: 192
  horizons_seconds: [10, 20, 30]   # forecast만
train:                         # 딥러닝 공통 설정
  batch_size: 1024
  max_epochs: 60
  windows_per_epoch: 200000
  patience: 8
  lr: 0.001
  loss: mse                    # mse | huber | mae
seeds: [42, 43, 44]
models:
  - ridge
  - name: patchtst
    params: {channel_mode: mixing, patch_seconds: 16, stride_seconds: 8, d_model: 128}
    grid:                      # 모든 조합을 학습하고 검증 MAE로 고른다
      d_model: [64, 128]
      train.lr: [0.001, 0.0003] # "train."으로 시작하면 이 모델의 학습 설정만 바꾼다
```

## 10. 지켜야 할 것

- 모델·하이퍼파라미터 선택은 검증 run으로만 한다. `ald-final-test`는 최종 후보가 정해진 뒤 한 번만 쓴다.
- 분할은 run 단위(`split_v2`: 학습 51 / 검증 11 / 시험 11, 071 제외)이고 정규화 통계는 학습 run으로만 계산한다.
- window는 시간 끊김(079의 4초 등)을 넘지 않는다.
- `work/` 아래 파일(정리된 데이터, 모델, 예측)은 회사 데이터이므로 외부로 보내지 않는다. 공유는 `summary.md`만.
