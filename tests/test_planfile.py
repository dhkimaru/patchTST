"""A markdown experiment plan read back into dashboard forms (the tables handed out for typing the form)."""

from __future__ import annotations

from ald_lab import launch, planfile

PLAN = """
# 추가 학습 계획

## 결과 요약

| 발견 | 크기 | 다음 실험 |
|---|---|---|
| 손실 함수 mae | −16 % | 기본으로 |

### 15차: 층 · FFN · dropout (mae)

| 순서 | 칸 | 입력 |
|---|---|---|
| 1 | 기준 실험 | g1_forecast_params |
| 2 | 대상 | 8개 전부 켬 |
| 3 | 모델 | PatchTST만 |
| 9 | 층 수 | **4, 6** |
| 10 | feed-forward 크기 | **512, 1024** |
| 14 | 학습률 | 0.0003 |
| 15 | 조합 방식 | 모든 조합 |
| 16 | window 길이(초) | 96 |
| 17 | seed | 42, 43 |
| 18 | 입력 세트 | **A 전체 + 압력계·APC·RF 매칭**만 켬 (전체는 끔) |
| 19 | 최대 epoch | 40 |
| 23 | 손실 함수 | **mae** |
| 24 | GPU | 비어 있는 GPU만 |
| 26 | 끝나면 보고서 갱신 | **켬** (마지막이라) |
| 27 | 메모 | 15차 층·FFN |

미리보기 확인: **작업 136개**

### 11차: 입력 세트

- 기준 실험: g1_forecast_params
- 대상: MFCMon_P.POS, VG11만 켬
- 모델: PatchTST, LightGBM
- 학습률: 0.001
- 입력 세트: **5개 켬:** 전체 · 중요도 A 대표 · 중요도 A·B 대표 · 중요도 A 전체 · A 전체 + 압력계·APC·RF 매칭 (이전 프로젝트 37개만 끔)
- GPU: 0, 2
- 조합 방식: 무작위 8개
- 없는 칸: 아무거나

### 잘못된 기준

| 순서 | 칸 | 입력 |
|---|---|---|
| 1 | 기준 실험 | g9_nothing |
"""


def test_plan_tables_become_forms():
    items = planfile.read_plan(PLAN, gpus="1,3")
    assert [i["title"] for i in items] == ["15차: 층 · FFN · dropout (mae)", "11차: 입력 세트", "잘못된 기준"]  # 요약 table skipped
    first, second, bad = (i["form"] for i in items)
    assert first["base"] == "g1_forecast_params" and len(first["targets"]) == 8 and list(first["models"]) == ["patchtst"]
    assert first["models"]["patchtst"]["n_layers"] == "4, 6" and first["models"]["patchtst"]["d_ff"] == "512, 1024"
    assert first["models"]["patchtst"]["d_model"]  # untouched settings keep the base defaults
    assert first["input_sets"] == ["A_plus"] and first["train"]["loss"] == "mae" and first["train"]["max_epochs"] == "40"
    assert first["windows"] == "96" and first["seeds"] == "42, 43" and first["search_trials"] == 0
    assert first["gpus"] == "1,3" and first["report"] is True and first["memo"] == "15차 층·FFN"
    assert items[0]["problems"] == [] and items[0]["ignored"] == []
    assert second["targets"] == ["VG11", "MFCMon_P.POS"] and set(second["models"]) == {"patchtst", "lightgbm"}  # "VG11만"
    assert second["models"]["patchtst"]["train.lr"] == "0.001"
    assert second["models"]["lightgbm"]["learning_rate"] == "0.001"  # one 학습률 row: each model's own learning-rate setting
    assert set(second["input_sets"]) == {"all", "core", "core_ab", "A_all", "A_plus"}
    assert second["gpus"] == "0,2" and second["search_trials"] == 8 and second["memo"] == "11차: 입력 세트"
    assert items[1]["ignored"] == ["없는 칸"]
    assert bad is None and "g9_nothing" in items[2]["problems"][0]
    docs = launch.docs_from_form(first)  # the same path the form takes
    assert len(docs) == 1 and docs[0]["train"]["loss"] == "mae"
    jobs = [j for d in docs for j in launch.plan_jobs(d)]
    assert len(jobs) == 8 * (1 + 4 * 2)  # targets × (baseline + 4 combos × 2 seeds)
