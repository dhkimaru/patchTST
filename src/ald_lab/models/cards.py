"""Plain-language descriptions of tasks and models, shared by model packages and the report."""

from __future__ import annotations

TASK_CARDS = {
    "forecast": {
        "name": "미래값 예측",
        "out": "대상 신호의 {horizons}초 뒤 값",
        "how": "최근 {context}초의 신호로 이후 값을 예측한다. 이상 징후를 미리 파악하는 용도.",
        "baseline": "현재값 유지: 현재 값이 그대로 유지된다고 가정한 예측",
    },
    "virtual_sensor": {
        "name": "가상 센서",
        "out": "대상 센서의 지금 값",
        "how": "대상 센서와 같은 계열·복사본 신호를 모두 제외하고, 나머지 신호의 최근 {context}초로 현재 값을 추정한다. "
               "센서를 추가하지 않거나 센서가 고장 난 경우에도 값을 얻기 위한 용도.",
        "baseline": "레시피 평균: 학습 run에서 같은 레시피 step·같은 위치의 평균 (레시피만 반영, 장비 상태는 반영하지 않음)",
    },
    "thickness": {
        "name": "두께 예측 (VM)",
        "out": "run 하나의 Ex-situ 두께·균일도",
        "how": "run의 trace를 신호별 요약값(ALD 사이클 평균·표준편차, 공정 평균)으로 줄여 run 단위로 예측한다. "
               "run 수가 적어 run 하나씩 빼며 평가한다 (leave-one-run-out).",
        "baseline": "학습 평균: 나머지 run 두께의 평균",
    },
}

MODEL_CARDS = {
    "persistence": {"name": "현재값 유지", "family": "기준선",
                    "what": "마지막으로 관측된 값이 그대로 유지된다고 본다.",
                    "good": "계산이 필요 없고, 안정 구간에서는 정확하다.",
                    "bad": "압력 전환처럼 값이 바뀌는 순간을 전혀 예측하지 못한다."},
    "template": {"name": "레시피 평균", "family": "기준선",
                 "what": "학습 run에서 같은 레시피 step, step 안 같은 위치의 평균값을 쓴다.",
                 "good": "레시피가 정하는 모양(승온, 사이클 패턴)은 그대로 따라간다.",
                 "bad": "run마다 다른 장비 상태(드리프트, 이상)는 모른다."},
    "mean": {"name": "학습 평균", "family": "기준선",
             "what": "학습 run 값의 평균을 모든 run에 쓴다.",
             "good": "가장 단순한 비교 기준.", "bad": "run 사이 차이를 전혀 설명하지 못한다."},
    "ridge": {"name": "Ridge 회귀", "family": "고전 ML",
              "what": "window를 신호별 요약값(마지막·평균·표준편차·최소·최대·최근 변화)으로 줄여 선형 회귀한다. "
                      "기준선 대비 보정량을 학습한다.",
              "good": "몇 초 만에 학습되고 결과를 해석하기 쉽다.",
              "bad": "신호 사이 비선형 관계를 잡지 못한다."},
    "pls": {"name": "PLS 회귀", "family": "고전 ML",
            "what": "많은 입력을 소수의 잠재 성분으로 압축한 뒤 회귀한다.",
            "good": "표본보다 특징이 훨씬 많을 때도 안정적이다.", "bad": "비선형 관계를 잡지 못한다."},
    "lightgbm": {"name": "LightGBM", "family": "고전 ML",
                 "what": "같은 요약값으로 결정나무 수천 개를 순차 학습(boosting)해 기준선 대비 보정량을 학습한다.",
                 "good": "비선형 관계와 신호 간 상호작용을 반영하며, CPU로 수 분 안에 학습된다.",
                 "bad": "window 안의 시간 순서는 요약값으로만 본다."},
    "patchtst": {"name": "PatchTST", "family": "딥러닝",
                 "what": "시계열을 짧은 조각(patch)으로 잘라 Transformer로 조각 사이 관계를 학습한다. "
                         "신호별로 따로 보는 방식(독립)과 섞어 보는 방식(혼합)이 있다.",
                 "good": "긴 시간 패턴을 직접 학습한다. 이전 실험의 주력 모델.",
                 "bad": "GPU와 학습 시간이 필요하고 정규화(RevIN) 설정에 민감하다."},
    "itransformer": {"name": "iTransformer", "family": "딥러닝",
                     "what": "신호 하나의 window 전체를 토큰 하나로 보고, 신호 사이 관계를 attention으로 학습한다.",
                     "good": "신호 간 관계가 중요한 가상 센서에 적합하다.",
                     "bad": "신호 수의 제곱으로 메모리가 늘어 batch가 작아진다."},
    "dlinear": {"name": "DLinear", "family": "딥러닝",
                "what": "추세(이동평균)와 나머지로 나눈 뒤 선형층 하나로 예측한다.",
                "good": "가볍고 안정적이며 학습이 빠르다.", "bad": "표현력이 제한적이다."},
    "gru": {"name": "GRU", "family": "딥러닝",
            "what": "순환 신경망: 시간 순서대로 상태를 갱신하며 읽는다.",
            "good": "순서 정보를 자연스럽게 다룬다.", "bad": "긴 window에서 느리고, 먼 과거 정보를 잃기 쉽다."},
}


def model_card(name: str) -> dict[str, str]:
    return MODEL_CARDS.get(name, {"name": name, "family": "", "what": "", "good": "", "bad": ""})


def task_card(task: str, *, horizons: list[float] | None = None, context: float | None = None) -> dict[str, str]:
    card = dict(TASK_CARDS.get(task, {"name": task, "out": "", "how": "", "baseline": ""}))
    fill = {"horizons": "·".join(f"{h:g}" for h in horizons or []), "context": f"{context:g}" if context else "?"}
    return {k: v.format(**fill) for k, v in card.items()}
