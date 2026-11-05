"""Stage-by-stage design notes for the Code Flow Chart tab: what a reviewer needs to check the pipeline.

Each stage states its purpose, the files it reads and writes, the processing steps (each tied to the
function that does it), the design choices with the alternative that was not taken, the tests that check
it and the open points a reviewer should question. Function keys (``module:qualname``) and test names are
checked against the code by a test, so the notes cannot point at code that no longer exists; the line
numbers and code excerpts in the report are read from the source when the report is built.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"


@dataclass(frozen=True)
class Step:
    text: str
    functions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    choice: str
    alternative: str
    reason: str


@dataclass(frozen=True)
class Check:
    test: str  # "test_training.py::test_windows_never_cross_segments"
    what: str


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    command: str
    purpose: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]
    steps: tuple[Step, ...]
    code: tuple[str, ...]
    decisions: tuple[Decision, ...]
    checks: tuple[Check, ...]
    review: tuple[str, ...]
    facts: tuple[str, ...] = field(default=())  # bundle stats keys shown as "이번 데이터에서" (see render)


STAGES: tuple[Stage, ...] = (
    Stage(
        key="overview",
        title="① 원본 데이터 개요",
        command="ald-build-overview",
        purpose="trace CSV 74개와 두께 엑셀을 한 번 읽어 run·컬럼·step·두께 표와 품질 점검 목록을 만듭니다. 이후 "
                "단계는 이 표와 분할 파일을 기준으로 데이터를 고르고, 원본은 고치지 않습니다.",
        inputs=(
            ("ALD_DATA_ROOT/standard_trace_NNN.csv", "0.5초 간격, 207컬럼(정보 7 + 신호 200), run 74개"),
            ("ALD_EXSITU_XLSX (Ex-situ 워크북)", "Ex-situ 측정 블록 134개 (49점·9점 배치)"),
            ("configs/columns/ald_columns_v1.yaml", "컬럼 의미·단위·계열(family)·중요도"),
        ),
        outputs=(
            ("work/cache/traces/*.parquet", "CSV를 그대로 옮긴 캐시. 파일 크기·수정 시각이 바뀌면 다시 만듦"),
            ("work/data_overview/runs.parquet", "run별 행 수·시작/끝·공정 구간·사이클 수·끊김·역행"),
            ("work/data_overview/columns.parquet", "컬럼별 범위·평균·상태(OK / CONSTANT / DUPLICATE) + 사전 정보"),
            ("work/data_overview/near_copies.parquet", "|r| ≥ 0.999인 신호 쌍 (참고용)"),
            ("work/data_overview/exsitu_*.parquet", "두께 블록 통계(원시 점에서 재계산)와 run 연결"),
            ("work/data_overview/issues.parquet · summary.json", "품질 점검 목록, 요약 수치. 진행 기록 1건"),
        ),
        steps=(
            Step("CSV 읽기: Date+Time을 시각으로 합치고 메타 4개를 분리, 나머지 200개를 float32 신호로 바꿈 "
                 "(숫자가 아닌 칸은 빈 값). 다음부터는 parquet 캐시를 읽음",
                 ("data.raw:read_trace_csv", "data.raw:load_trace")),
            Step("step 구간: 같은 (레시피, Step ID)가 이어지는 행을 한 구간으로 묶음", ("data.catalog:step_segments",)),
            Step("공정 구간 = 첫 START step 시작 ~ 그 뒤 마지막 END step 시작", ("data.catalog:process_window",)),
            Step("시간 점검: 간격이 0.5±0.01초 밖이면 끊김, 음수면 역행. 사이클 수 = SI-FL1 구간 수",
                 ("data.catalog:catalog_run",)),
            Step("컬럼 판정: 74 run 모두에서 한 값만 나오면 CONSTANT, 모든 run에서 값이 바이트 단위로 같으면 "
                 "DUPLICATE(파일에서 먼저 나온 컬럼을 원본으로)", ("data.profile:ColumnProfiler.result",)),
            Step("두께: 블록마다 원시 측정점에서 평균·U/F·WtoW를 다시 계산해 워크북 수식 값과 대조",
                 ("data.exsitu:_block_stats",)),
            Step("두께 ↔ run 연결: 블록 시작·끝과 run 경계가 5분 안이면 연결, 안 되면 제목의 '연속진행 N'",
                 ("data.exsitu:link_to_traces",)),
            Step("품질 점검: 시간 역행 = 치명, 끊김·두께 이상(|robust z| ≥ 5)·비정상 종료 = 경고, 사이클 수 차이·"
                 "수식 차이 = 정보", ("data.overview:_issues",)),
            Step("표 저장과 진행 기록", ("data.overview:build_data_overview",)),
        ),
        code=("data.catalog:process_window", "data.catalog:catalog_run"),
        decisions=(
            Decision("개요는 표시만 하고 데이터를 고치거나 지우지 않음. 빼는 결정은 분할 파일·실험 설정에서",
                     "개요 단계에서 이상 run을 삭제",
                     "판단 근거가 표로 남고, 기준이 바뀌어도 원본을 다시 읽지 않음"),
            Decision("중복 = 모든 run에서 값이 완전히 같을 때만", "상관 0.999 이상이면 중복",
                     "거의 같은 신호(이웃 zone 온도 등)는 다른 센서일 수 있어 지우지 않고 84쌍을 따로 표시"),
            Decision("두께 통계를 원시 측정점에서 다시 계산", "워크북 요약값 사용",
                     "일부 블록 수식에 범위 오류가 있음 (최대 0.113 차이)"),
            Decision("두께 이상치는 측정 배치(49점·9점)별로 판정", "전체를 한꺼번에",
                     "배치가 다르면 평균 위치가 달라 직접 비교할 수 없음"),
        ),
        checks=(
            Check("test_data_layer.py::test_read_trace_types_and_dropped_column", "메타·신호 타입, 빈 마지막 열 제거"),
            Check("test_data_layer.py::test_catalog_counts_cycles_gaps_and_process_window", "사이클 수·끊김·공정 구간"),
            Check("test_data_layer.py::test_profiler_flags_constant_duplicate_and_near_copy", "상수·중복·거의 같은 쌍 판정"),
            Check("test_data_layer.py::test_exsitu_parse_and_link", "두께 블록 읽기와 run 연결"),
            Check("test_exsitu_layouts.py::test_9pt_layout_detected_and_sheet_mismatch_reported", "9점 배치 인식, 수식 차이 보고"),
            Check("test_journal_overview.py::test_build_overview_end_to_end", "개요 전체 흐름과 진행 기록"),
        ),
        review=(
            "상수 판정은 이번 74 run 기준입니다. 다른 기간 데이터에서 값이 변하는 컬럼이 생겨도 자동으로 입력에 들어가지 않습니다.",
            "공정 구간은 '첫 START ~ 마지막 END'입니다. 중간에 멈췄다 다시 시작한 run(예: 055 통신 끊김)의 구간이 의도대로 잡히는지 확인이 필요합니다.",
            "두께 이상치 기준(|robust z| ≥ 5, MAD × 1.4826)에서 051·069도 경고(z = +6·+5)가 됩니다. 기준이 너무 민감한지 검토할 수 있습니다.",
            "숫자가 아닌 칸은 빈 값으로 바뀝니다. 이번 데이터에서 빈 값은 0개였지만, 새 장비 내보내기 형식에서는 확인이 필요합니다.",
        ),
    ),
    Stage(
        key="prepare",
        title="② 학습 데이터 준비 (전처리)",
        command="ald-train (첫 단계) · 결과는 work/prepared/에 캐시",
        purpose="split_v2의 학습·검증(최종 평가 때만 시험) run을 1초 간격 배열 묶음(bundle)으로 한 번 만들어 저장합니다. "
                "모든 실험과 새 CSV 예측이 같은 함수로 같은 형태의 묶음을 씁니다.",
        inputs=(
            ("work/cache/traces 또는 CSV", "run별 trace"),
            ("configs/splits/split_v2.yaml", "학습 51 / 검증 11 / 시험 11 run, 071 제외"),
            ("컬럼 사전", "사용 가능 신호 164개 (상수·중복 제외)"),
        ),
        outputs=(
            ("work/prepared/split_v2_1s_<분할>_<키>/values.npy", "행 × 신호 164개, float32. 메모리 매핑으로 읽음"),
            ("… /run_index · seg_id · seg_bounds", "행이 속한 run과 끊김 없는 구간의 경계"),
            ("… /in_cycle · in_process", "ALD 사이클 / 공정 구간 여부"),
            ("… /step_key · step_pos", "레시피|Step ID의 해시, step 시작 후 행 수(상한 있음)"),
            ("… /meta.json", "run 목록과 통계(행·구간·빈 값·사이클 행 등)"),
        ),
        steps=(
            Step("분할 읽기: 시험 run은 include_test=True일 때만 꺼낼 수 있음 (학습 단계에서는 오류)",
                 ("data.splits:load_split", "data.splits:RunSplit.runs")),
            Step("빈 값을 앞 값 → 뒤 값 → 0 순서로 채움", ("training.prepare:run_arrays",)),
            Step("0.5초 간격이 깨진 곳(끊김·역행)에서 구간을 나눔", ("training.prepare:run_arrays",)),
            Step("구간 안에서만 2행씩 평균해 1초 행을 만듦 (구간 끝에 짝이 없는 1행은 버림)", ("training.prepare:run_arrays",)),
            Step("1초 행마다 블록 마지막 원본 행 기준으로 step_key·step_pos·in_cycle·in_process를 붙임",
                 ("training.prepare:run_arrays",)),
            Step("run들을 이어 붙이고 구간 경계 표를 만듦", ("training.prepare:assemble_bundle",)),
            Step("저장: 분할·run 목록·간격·컬럼으로 만든 키가 같으면 다시 만들지 않음. 동시에 만들면 먼저 끝난 쪽을 씀",
                 ("training.prepare:build_bundle", "training.prepare:save_bundle")),
            Step("새 CSV 예측도 같은 run_arrays로 묶음을 만듦 (캐시하지 않음)", ("training.prepare:bundle_from_traces",)),
        ),
        code=("training.prepare:run_arrays",),
        decisions=(
            Decision("1초 평균", "0.5초 원본",
                     "이전 실험 GM1에서 0.5초가 더 나빴고(0.188 → 0.252), 학습이 2배 빠름. 대가로 짧은 튐이 약간 깎임 "
                     "(검증 run 053의 DCS 방출 최고값 4.3 → 4.15 Torr)"),
            Decision("끊긴 곳에서 구간을 나눔", "끊긴 곳을 보간으로 메움",
                     "없는 4초(079)를 지어내지 않음. 역행(071)은 나눠도 해결되지 않아 run을 제외"),
            Decision(".npy로 저장해 메모리 매핑", "실험마다 CSV를 다시 읽음",
                     "2.4 GB CSV를 매번 읽지 않고, GPU 작업자 4개가 같은 파일을 공유"),
            Decision("step_key = 레시피 + Step ID", "Step 이름",
                     "같은 이름이 다른 레시피에도 있을 수 있음"),
        ),
        checks=(
            Check("test_training.py::test_block_average_respects_time_breaks", "1초 평균이 끊김을 넘지 않음"),
            Check("test_splits.py::test_split_v2_is_disjoint_and_seals_test", "분할이 겹치지 않고 시험 run이 봉인됨"),
            Check("test_splits.py::test_overlapping_split_is_rejected", "겹치는 분할 파일은 거부"),
            Check("test_report_analysis.py::test_example_run_goes_through_the_training_preprocessing", "In 탭 예시가 같은 함수를 씀"),
        ),
        review=(
            "빈 값을 앞 값으로 채우면 센서가 오래 끊긴 구간에서 마지막 값이 계속 이어집니다. 이번 데이터는 0개였지만 새 데이터에는 경고가 필요합니다.",
            "ALD 사이클은 step 이름 9개로 판별합니다. 레시피가 바뀌어 step 이름이 달라지면 사이클로 잡히지 않습니다.",
            "1초 행의 step 정보는 블록 마지막 원본 행 기준이라, step이 바뀌는 순간에는 0.5초 어긋날 수 있습니다.",
            "step_pos 상한(원본 2,400행 = 20분) 뒤의 위치는 구분하지 않습니다. 20분이 넘는 step의 레시피 평균은 끝부분이 평평해집니다.",
        ),
        facts=("rows", "columns", "segments", "nan_filled"),
    ),
    Stage(
        key="inputs",
        title="③ 입력 선택 · 정규화 · window",
        command="ald-train (대상마다)",
        purpose="대상 하나마다 입력 신호를 정하고, 학습 run으로만 정규화 값을 재고, run 경계와 시간 끊김을 넘지 않는 "
                "window 끝 위치 목록을 만든 뒤 GPU에 올립니다.",
        inputs=(
            ("bundle (학습·검증)", "②의 결과"),
            ("실험 YAML", "inputs / input_sets, context_seconds, horizons_seconds, region"),
            ("컬럼 사전", "계열(family), 중요도(core·core_ab 세트)"),
        ),
        outputs=(
            ("job.json의 inputs · removed_inputs", "쓴 입력과 뺀 신호(이유별)"),
            ("job.json의 scaler", "신호별 평균·표준편차 (학습 run)"),
            ("WindowSet (GPU 메모리)", "x: 행 × 입력, y: 행 × 1, ends: window 끝 행 번호"),
        ),
        steps=(
            Step("입력 세트: all(164) · core(중요도 A 대표) · core_ab(A·B 대표) · legacy_primary(이전 프로젝트 38개) · 직접 목록",
                 ("training.config:resolve_inputs",)),
            Step("미래값 예측은 대상을 입력에 넣음. 가상 센서는 대상 + 같은 계열 + 학습 run에서 |r| ≥ 0.9999인 복사본을 뺌",
                 ("training.config:resolve_inputs", "training.windows:near_copies")),
            Step("학습 run에서 값이 전혀 변하지 않는 입력을 뺌", ("training.windows:constant_columns",)),
            Step("정규화: 학습 run 전체 행으로 신호별 평균·표준편차를 잼 (검증·시험·새 CSV에도 같은 값)",
                 ("training.windows:fit_scaler",)),
            Step("window 끝 위치: 앞으로 context−1행과 가장 먼 예측 시점이 모두 같은 구간 안인 행만. region = process이면 "
                 "끝 행이 공정 구간, cycle이면 사이클 안", ("training.windows:window_ends",)),
            Step("(행 × 채널) 표를 GPU에 한 번 올리고, batch는 인덱스 연산 한 번으로 자름",
                 ("training.runner:make_window_set", "training.windows:WindowSet.gather")),
            Step("대상·입력 세트별로 위 결과를 묶어 작업자 안에서 재사용 (최근 2개)", ("training.runner:DataCache.target_data",)),
        ),
        code=("training.windows:window_ends", "training.windows:WindowSet.gather", "training.windows:fit_scaler",
              "training.runner:DataCache.target_data"),
        decisions=(
            Decision("정규화 통계는 학습 run에서만", "학습+검증+시험 전체",
                     "검증·시험 run의 정보가 학습에 새지 않게 함"),
            Decision("데이터 전체에 정규화 값 하나", "run마다 따로 정규화",
                     "run마다 하면 새 run을 예측할 때 그 run 전체의 평균을 미리 알아야 함(미래 정보)"),
            Decision("window 끝을 미리 목록으로 만듦", "batch를 만들 때마다 경계 검사",
                     "무작위로 뽑아도 경계를 넘는 window가 구조적으로 나올 수 없음"),
            Decision("가상 센서 복사본 기준 |r| ≥ 0.9999", "0.999",
                     "이웃 zone 온도(r ≈ 0.9998)는 다른 센서라 입력에 남겨야 함"),
            Decision("대상의 계열을 통째로 뺌", "대상 컬럼만 뺌",
                     "PressAct·AUXMon_VG11처럼 대상의 복사본이 남으면 베끼기만 해도 맞힘"),
        ),
        checks=(
            Check("test_training.py::test_windows_never_cross_segments", "window가 구간을 넘지 않음"),
            Check("test_training.py::test_scaler_and_template_use_train_rows_only", "정규화 값이 학습 run 행에서만 나옴"),
            Check("test_training.py::test_virtual_sensor_drops_target_family", "가상 센서가 대상 계열을 뺌"),
            Check("test_training.py::test_near_copies_found_on_train_rows", "복사본을 학습 run에서 찾음"),
            Check("test_report.py::test_virtual_sensor_predicts_without_its_sensor", "대상 컬럼이 없는 CSV로도 예측"),
        ),
        review=(
            "정규화에 대기(STANDBY) 행도 들어갑니다. VG11 평균이 354 Torr라 공정 압력 차이가 작은 수가 되고, 이것을 모델별 처리(변화량·보정량 학습)로 보완합니다. 공정 구간만으로 재는 방법과 비교한 적은 없습니다.",
            "가상 센서(region = process)는 window의 '끝'만 공정 구간이면 됩니다. window 앞부분이 대기 구간을 포함할 수 있습니다.",
            "계열(family)은 컬럼 사전에서 사람이 정한 값입니다. 빠진 복사본은 상관 검사가 보완하지만, 지연·비선형 관계의 복사본은 잡지 못합니다.",
            "컬럼 사전에서 압력계·APC 밸브·RF 매칭 위치의 중요도가 비어 있어, core 세트에 압력 정보가 없습니다. 현업 확인이 필요합니다.",
        ),
    ),
    Stage(
        key="train",
        title="④ 학습",
        command="ald-train configs/experiments/<실험>.yaml --gpus 0,1,2,3",
        purpose="실험 YAML을 작업 목록으로 펼치고, 기준선을 먼저 돌린 뒤 나머지 작업을 GPU 4장에 나눠 학습합니다. "
                "작업마다 모델·설정·검증 지표를 폴더에 남깁니다.",
        inputs=(
            ("configs/experiments/*.yaml", "대상 × 모델 × 설정 조합 × 입력 세트 × seed"),
            ("②·③의 결과", "bundle, 입력, 정규화 값, window"),
        ),
        outputs=(
            ("work/experiments/<실험>/<시각>/jobs/<작업>/job.json", "입력·뺀 신호·정규화 값·모델 설정·window 설정"),
            ("… /model.pt · lightgbm.joblib · ridge.npz", "가중치 (모델 종류별)"),
            ("… /template.parquet", "레시피 평균 (가상 센서, 기준선)"),
            ("… /history.json · train.log", "epoch별 학습·검증 손실, 학습 로그"),
            ("… /metrics_validation.json · predictions_sample.parquet", "검증 지표, 예시 그래프용 예측"),
        ),
        steps=(
            Step("YAML 검증과 작업 펼치기", ("training.config:load_experiment", "training.config:expand_jobs")),
            Step("데이터 준비를 진행 기록에 남김", ("training.runner:_prepare_entry",)),
            Step("기준선(현재값 유지 / 레시피 평균)을 먼저 돌려 다른 작업의 개선율 기준을 정함",
                 ("training.runner:run_experiment",)),
            Step("나머지 작업을 대기열 하나에 넣고 GPU마다 작업자 프로세스 1개(spawn). 먼저 끝난 GPU가 다음 작업을 가져감",
                 ("training.runner:_run_parallel", "training.runner:_worker")),
            Step("작업 하나: 입력·window 준비 → (딥러닝) seed 고정, 모델 생성. 가상 센서는 정답을 (실측 − 레시피 평균) ÷ 그 "
                 "표준편차로 바꿈 → 학습 → 검증 예측 → 물리 단위로 되돌려 채점 → 파일 저장",
                 ("training.runner:run_job", "training.windows:offset_targets", "training.windows:residual_scale")),
            Step("딥러닝 학습: AdamW(lr 1e-3, weight decay 1e-4), epoch마다 무작위 200,000 window, batch 1,024, bf16, "
                 "기울기 자르기 1.0, 고르게 뽑은 검증 50,000 window로 조기 종료, 검증 손실이 멈추면 학습률 ×0.5, 최고 epoch 가중치 복원",
                 ("training.trainer:train_model",)),
            Step("고전 모델: window를 신호마다 요약 6개(마지막·평균·표준편차·최소·최대·마지막−중간)로 줄임. Ridge는 검증으로 α, "
                 "LightGBM은 학습 window 30만 개를 무작위로 뽑아 학습하고 검증 10만 개로 조기 종료(대상·예측 거리마다 모델 1개). "
                 "둘 다 기준선 대비 차이를 학습",
                 ("models.classic:window_features", "models.classic:Ridge.fit", "models.classic:LightGBM.fit")),
            Step("딥러닝 공통 틀: RevIN(window별 정규화) → backbone(PatchTST·iTransformer·DLinear·GRU) → 출력층 → 되돌리기",
                 ("models.deep:SeriesModel.forward", "models.deep:build_deep_model")),
            Step("두께는 별도 경로: run마다 요약 특징 → run 하나씩 빼며 학습·예측",
                 ("training.vm:run_features", "training.vm:run_thickness")),
        ),
        code=("training.runner:_worker", "models.classic:window_features", "models.deep:SeriesModel.forward",
              "training.trainer:train_model"),
        decisions=(
            Decision("GPU마다 서로 다른 작업 하나", "모델 하나를 GPU 4장에 나눠 학습(분산 학습)",
                     "모델이 작아 GPU 사이 통신 비용이 더 큼. 실험별 3.1~3.8배 빨라짐"),
            Decision("epoch마다 무작위 200,000 window", "매 epoch 전체 window",
                     "이웃 window는 1초만 달라 전체를 도는 것은 대부분 반복"),
            Decision("기준선 대비 차이(변화량·보정량)를 학습", "값을 직접 학습",
                     "나무 모델은 학습 범위 밖 값을 만들지 못하고, VG11은 1~760 Torr를 오감"),
            Decision("가상 센서 손실 = MAE", "MSE",
                     "MSE는 승온 구간의 큰 오차에 끌려 사이클 구간 예측이 망가짐 (사전 확인)"),
            Decision("실패한 작업은 기록하고 계속", "실험 전체 중단", "작업 하나의 문제가 나머지를 막지 않음"),
        ),
        checks=(
            Check("test_training.py::test_forecast_experiment_end_to_end", "예측 실험 전체 흐름"),
            Check("test_training.py::test_virtual_sensor_experiment", "가상 센서 실험 전체 흐름"),
            Check("test_training.py::test_parallel_workers", "작업자 프로세스 대기열"),
            Check("test_training.py::test_seed_reproducible_and_experiment_relocatable", "같은 seed면 같은 결과"),
            Check("test_training.py::test_revin_keeps_level_for_virtual_sensor", "가상 센서에서 RevIN이 절대 수준을 지움"),
            Check("test_training.py::test_deep_virtual_sensor_learns_template_correction", "딥러닝 가상 센서가 보정량을 학습"),
            Check("test_training.py::test_thickness_experiment", "두께 실험 전체 흐름"),
            Check("test_report_analysis.py::test_deep_defaults_match_the_constructors", "보고서의 기본값 = 모델 코드의 기본값"),
        ),
        review=(
            "검증 run이 조기 종료(딥러닝·LightGBM), Ridge α 선택, 모델 선택에 모두 쓰입니다. 그래서 검증 점수는 조금 낙관적이고, 공정한 비교는 시험 run 숫자로 해야 합니다.",
            "LightGBM은 학습 window 약 80만 개 중 30만 개만 무작위로 씁니다. 표본을 늘리면 나아지는지는 확인하지 않았습니다.",
            "LightGBM의 요약 특징 6개는 window 안의 시간 순서를 거의 버립니다(마지막 − 중간 하나만 남김).",
            "딥러닝은 PatchTST 일부 설정만 탐색했고, 나머지 모델은 기본값입니다 (Hyper Parameter 탭 참고).",
            "bf16·TF32 계산을 써서, 같은 seed라도 GPU 종류가 다르면 결과가 아주 조금 다를 수 있습니다.",
        ),
    ),
    Stage(
        key="evaluate",
        title="⑤ 평가 · 모델 선택 · 시험 run",
        command="ald-train (마지막 단계) · ald-final-test <실험 폴더>",
        purpose="모든 작업을 같은 지표로 채점해 대상마다 검증 run 기준 최고 모델을 고르고, 고른 후보만 시험 run에서 한 번 "
                "채점합니다.",
        inputs=(
            ("작업별 검증 예측", "물리 단위로 되돌린 값"),
            ("검증 bundle의 in_cycle · step_pos · run_index", "구간별·run별 분해에 사용"),
        ),
        outputs=(
            ("results.csv", "작업(seed)별 지표"),
            ("summary.csv · summary.json · summary.md", "설정별 seed 평균과 순위, 대상별 선택"),
            ("jobs/<작업>/metrics_test.json", "시험 run 지표 (ald-final-test 후)"),
        ),
        steps=(
            Step("지표: MAE(주지표, 물리 단위)·RMSE·R²·95 % 절대 오차, 예측 거리별, ALD 사이클 안/밖, step 전환 직후 10초 / "
                 "안정 구간(예측 시점 행이 step 시작 10초 안인지), run별", ("training.evaluate:regression_metrics",)),
            Step("개선율 = (1 − 모델 MAE ÷ 기준선 MAE) × 100", ("training.evaluate:improvement",)),
            Step("강점·약점 문장: 규칙 기반 (예측 거리별 배수, 전환 직후 배수, run별 편차, 학습 상태)",
                 ("training.evaluate:describe",)),
            Step("같은 설정의 seed들을 평균 → 대상마다 MAE 순 → 기준선이 아닌 1등을 선택", ("training.runner:summarize",)),
            Step("패키지·예시용 대표 작업: 고른 설정 중 검증 MAE가 가장 낮은 seed", ("training.selection:selected_jobs",)),
            Step("시험 run: 모델 종류별 최고 설정의 모든 seed를 한 번 적용. 학습 때 정규화 값·입력·window를 그대로 씀",
                 ("training.runner:run_final_test", "training.runner:predict_job")),
            Step("두께: 58 run을 하나씩 빼며 예측한 오차가 검증 지표. 시험 run 11개는 별도로 봉인",
                 ("training.vm:run_thickness", "training.vm:final_test_thickness")),
        ),
        code=("training.evaluate:regression_metrics", "training.selection:selected_jobs"),
        decisions=(
            Decision("모델 선택은 검증 run seed 평균 MAE로만", "시험 run 결과로 선택",
                     "시험 숫자를 '처음 보는 run에서의 성능'으로 말할 수 있게 함"),
            Decision("시험 run에서 모델 종류별 최고 후보를 모두 채점", "1등 하나만 채점",
                     "모델 비교를 시험 run에서도 보여 줌. 단 선택은 바꾸지 않음"),
            Decision("주지표 = 물리 단위 MAE", "정규화 단위 손실이나 RMSE",
                     "'평균 몇 Torr 틀린다'로 현업이 바로 이해"),
            Decision("구간별 분해 (사이클 / 그 외 / 전환 직후)", "전체 평균만",
                     "전체 평균은 배기·대기압 복귀 같은 큰 전환에 좌우됨"),
        ),
        checks=(
            Check("test_training.py::test_forecast_experiment_end_to_end", "요약·선택·시험 run 채점"),
            Check("test_report.py::test_history_topics_use_final_test", "보고서가 시험 run 숫자를 씀"),
            Check("test_report_analysis.py::test_factor_effects_against_seed_spread", "seed 편차로 효과 판단"),
            Check("test_splits.py::test_split_v2_is_disjoint_and_seals_test", "시험 run 봉인"),
        ),
        review=(
            "시험 run 결과를 본 뒤 모델이나 설정을 바꾸면 시험 run이 검증용이 됩니다. 지금까지는 바꾸지 않았고, 앞으로도 이 규칙을 지켜야 합니다.",
            "'step 전환 직후'는 예측 시점 행이 step 시작 10초 안인 경우입니다. 곧 step이 바뀌기 직전의 오차는 따로 보지 않습니다.",
            "run 단위 통계 검정(Wilcoxon)은 아직 없습니다. 차이가 작은 모델끼리는 우열을 단정하지 않습니다.",
            "시험 run이 11개라 run 하나가 숫자를 크게 움직일 수 있습니다. run별 오차를 함께 봐야 합니다.",
            "패키지의 대표 seed를 검증 최저로 고르는 것은 약한 선택 편향입니다(seed 사이 차이는 작음).",
        ),
    ),
    Stage(
        key="use",
        title="⑥ 활용 — 모델 패키지 · 새 CSV 예측 · 보고서",
        command="ald-export-model · ald-predict · ald-report · 대시보드",
        purpose="고른 모델을 실행 파일과 함께 묶어 새 CSV에 그대로 적용하고, 모든 결과를 인터넷 없이 열리는 HTML 하나로 정리합니다.",
        inputs=(
            ("실험 결과 폴더", "summary.csv, jobs/"),
            ("새 trace CSV", "학습 때와 같은 컬럼 형식"),
            ("work/ 전체", "보고서·대시보드는 결과 파일만 읽음"),
        ),
        outputs=(
            ("work/packages/<실험>_<시각>/<대상>__<모델>/", "가중치, job.json, 레시피 평균, README.txt, inputs.txt, predict.sh / .bat, zip"),
            ("predictions.csv", "시점별 run·시각·step·사이클 여부·예측값(실측이 있으면 함께)"),
            ("work/reports/ald_report_*.html", "이 보고서"),
        ),
        steps=(
            Step("대상별 최고 모델 폴더를 복사하고 README·실행 파일을 씀. 두께는 학습+검증 run 전체로 다시 맞춤",
                 ("training.export:export_models", "training.vm:fit_final_model")),
            Step("새 CSV: 읽기 → 같은 run_arrays로 묶음 → 학습 때 정규화 값·입력·window → 예측",
                 ("training.predict:predict_traces", "training.prepare:bundle_from_traces")),
            Step("모델 입력 신호가 CSV에 없으면 멈춤. 가상 센서는 대상 컬럼이 없어도 됨. 실측이 있으면 오차를 출력",
                 ("training.predict:predict_traces", "training.predict:describe_errors")),
            Step("보고서: work/의 결과·진행 기록·데이터 개요를 모으고 탭별 HTML을 만듦. 이 탭은 코드를 그때 다시 읽음",
                 ("report.collect:collect", "report.render:render")),
        ),
        code=("training.predict:predict_traces",),
        decisions=(
            Decision("패키지에 학습 때 정규화 값·입력 목록·레시피 평균을 함께 저장", "예측할 때 다시 계산",
                     "같은 모델이 같은 전처리로 돌아가는 것을 보장"),
            Decision("보고서는 결과 파일만 읽고 학습하지 않음", "보고서에서 필요한 계산을 다시 함",
                     "보고서를 만들어도 결과가 바뀌지 않고 몇 초면 끝남"),
        ),
        checks=(
            Check("test_report.py::test_package_predictions_match_validation", "패키지 예측 = 검증 결과"),
            Check("test_report.py::test_forecast_package_matches_validation", "예측 패키지의 예측 거리별 오차 재현"),
            Check("test_report.py::test_thickness_package", "두께 패키지"),
            Check("test_report.py::test_report_is_self_contained", "보고서가 외부 파일 없이 열림"),
        ),
        review=(
            "실시간 연결은 아직 없습니다 (CSV 파일 단위).",
            "학습 범위(레시피 8종, 2025-04~05) 밖의 데이터를 넣어도 경고하지 않습니다. 입력 값 범위 점검을 추가할 수 있습니다.",
            "predict.sh는 서버의 ald-lab 위치를 기본값으로 씁니다. 다른 곳으로 옮기면 ALD_LAB 환경변수로 알려 줘야 합니다.",
        ),
    ),
)


ARTIFACTS: tuple[tuple[str, str, str, str], ...] = (
    # (path, made by, content, read by)
    ("work/cache/traces/*.parquet", "모든 명령 (처음 읽을 때)", "CSV 내용 그대로 + 시각", "개요 · 학습 · 보고서"),
    ("work/data_overview/", "ald-build-overview", "run·컬럼·step·두께 표, 품질 점검", "학습(두께 정답) · 보고서 · 대시보드"),
    ("work/prepared/<분할>_<간격>_<파티션>_<키>/", "ald-train · ald-final-test", "1초 배열 묶음(.npy) + meta.json",
     "학습 · 시험 평가 · 보고서 In 탭"),
    ("work/experiments/<실험>/<시각>/", "ald-train", "experiment.yaml 사본, jobs.json, prep.json, results·summary",
     "시험 평가 · 패키지 · 보고서"),
    ("… /jobs/<작업>/", "ald-train · ald-final-test", "job.json, 가중치, history, 검증·시험 지표, 예측 예시",
     "패키지 · 보고서"),
    ("work/packages/<실험>_<시각>/", "ald-export-model", "대상별 모델 폴더 + zip", "ald-predict · 현업 PC"),
    ("work/journal/journal.jsonl", "모든 단계", "단계·시각·소요 시간·지표·강점·약점 (한 줄에 한 기록)", "보고서 · 대시보드"),
    ("work/reports/ald_report_*.html", "ald-report", "이 보고서", "노트북"),
)


@dataclass(frozen=True)
class Guard:
    risk: str
    guard: str
    functions: tuple[str, ...]
    tests: tuple[str, ...]


GUARDS: tuple[Guard, ...] = (
    Guard("같은 run이 학습과 평가에 함께 들어감", "run(파일) 단위 분할 split_v2. 겹치는 분할 파일은 읽을 때 거부",
          ("data.splits:load_split",), ("test_splits.py::test_overlapping_split_is_rejected",)),
    Guard("시험 run을 미리 봄", "학습 단계는 학습·검증 묶음만 만듦. 시험 run은 include_test=True(ald-final-test)에서만 열림",
          ("data.splits:RunSplit.runs", "training.runner:run_final_test"),
          ("test_splits.py::test_split_v2_is_disjoint_and_seals_test",)),
    Guard("정규화에 검증·시험 정보가 섞임", "평균·표준편차를 학습 묶음에서만 계산하고 job.json에 저장해 그대로 씀",
          ("training.windows:fit_scaler", "training.runner:DataCache.target_data"),
          ("test_training.py::test_scaler_and_template_use_train_rows_only",)),
    Guard("window가 run 경계·시간 끊김을 넘음", "끊김에서 구간을 나누고, 구간 안에 다 들어가는 끝 위치만 목록으로",
          ("training.prepare:run_arrays", "training.windows:window_ends"),
          ("test_training.py::test_windows_never_cross_segments", "test_training.py::test_block_average_respects_time_breaks")),
    Guard("가상 센서가 대상의 복사본으로 '추정'함", "대상 계열 전체와 학습 run에서 |r| ≥ 0.9999인 복사본을 입력에서 뺌",
          ("training.config:resolve_inputs", "training.windows:near_copies"),
          ("test_training.py::test_virtual_sensor_drops_target_family", "test_report.py::test_virtual_sensor_predicts_without_its_sensor")),
    Guard("레시피 평균(기준선)에 검증 정보가 섞임", "레시피 평균은 학습 run 행으로만 계산",
          ("models.classic:RecipeTemplate.fit", "training.runner:TargetData.attach_template"),
          ("test_training.py::test_scaler_and_template_use_train_rows_only",)),
    Guard("학습과 예측의 전처리가 다름", "새 CSV도 같은 run_arrays와 학습 때 저장한 정규화 값·입력·window를 씀",
          ("training.predict:predict_traces", "training.prepare:bundle_from_traces"),
          ("test_report.py::test_package_predictions_match_validation", "test_report.py::test_forecast_package_matches_validation")),
    Guard("두께 교차검증에서 빠진 run 정보가 샘", "run 하나를 뺄 때마다 표준화를 포함한 모델 전체를 나머지 run으로 다시 맞춤",
          ("training.vm:run_thickness", "training.vm:_estimator"), ("test_training.py::test_thickness_experiment",)),
    Guard("결과를 다시 낼 수 없음", "seed 고정, 실험 설정 사본, 분할 파일, 코드 버전(보고서 맨 위), 가중치 sha256",
          ("training.runner:run_job",), ("test_training.py::test_seed_reproducible_and_experiment_relocatable",)),
)


def function_keys() -> set[str]:
    keys = {f for s in STAGES for step in s.steps for f in step.functions} | {c for s in STAGES for c in s.code}
    return keys | {f for g in GUARDS for f in g.functions}


def test_names() -> set[str]:
    return {c.test for s in STAGES for c in s.checks} | {t for g in GUARDS for t in g.tests}


def missing_tests(tests_dir: Path = TESTS_DIR) -> list[str]:
    """Test references that do not name a test function in ``tests/``."""

    found: dict[str, set[str]] = {}
    for path in tests_dir.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found[path.name] = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    out = []
    for ref in sorted(test_names()):
        file, _, name = ref.partition("::")
        if name not in found.get(file, set()):
            out.append(ref)
    return out


def excerpt(source: str, lineno: int, lines: int, limit: int = 40) -> tuple[str, int]:
    """The function's source (dedented), cut to ``limit`` lines; returns the text and the lines left out."""

    body = source.splitlines()[lineno - 1 : lineno - 1 + lines]
    indent = min((len(x) - len(x.lstrip()) for x in body if x.strip()), default=0)
    body = [x[indent:] for x in body]
    while body and not body[-1].strip():
        body.pop()
    cut = max(len(body) - limit, 0)
    return "\n".join(body[:limit]), cut


def short_name(key: str) -> str:
    return re.sub(r"^.*:", "", key) + "()"
