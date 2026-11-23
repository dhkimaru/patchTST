"""Write the 1단계 (hyperparameter 극한) study files: ``configs/hpo/x1_*.yaml``. No training.

One study per representative target and model. The first trial of each study is the current best setting from the
cumulative records (``enqueue_from``), so the saturation curve starts where the project stands.

- 미래값 예측 (1 s, base g1_forecast_params): VG11, APC, 온도 C, 매처 L.POS. Objective: validation MAE.
- 가상 센서 0.5 s 조합: the process-wide model (base g1_virtual_sensor_res05) is searched on the MAE outside the
  ALD cycles and the cycle-only model (g1_virtual_sensor_res05_cycleonly) on the MAE inside them. The pair's MAE is
  the row-weighted sum of the two parts, so the two searches together optimize the pair.
  Targets: NH3, 밸브 29, DCS, APC, VG11, P.POS, R.PWR.
- 가상 센서 1 s (base g1_virtual_sensor): 온도 C.

    python scripts/hpo_studies.py            # → configs/hpo/x1_*.yaml (overwrites)
    ald-queue submit configs/hpo/x1_vs_nh3_cycle.yaml
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "hpo"
TRIALS = 150

# Model size and regularization (both tasks). PatchTST with every signal of a patch in one token (channel mixing):
# channel-independent was 6x worse in the previous project and 164 times slower here, so it is not searched.
MODEL_SPACE = {
    "n_layers": {"type": "int", "low": 2, "high": 10},
    "d_model": [64, 128, 192, 256, 384],
    "n_heads": [4, 8, 16],
    "d_ff": [128, 256, 512, 1024],
    "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
    "attn_dropout": [0.0, 0.1],
    "head_dropout": [0.0, 0.1],
    "norm": ["batch", "layer"],
    "stride_ratio": [0.25, 0.5, 1.0],
}
TRAIN_SPACE = {
    "train.lr": {"type": "float", "low": 3.0e-5, "high": 3.0e-3, "log": True},
    "train.lr_schedule": ["plateau", "cosine"],
    "train.warmup_epochs": [0, 5],
    "train.weight_decay": {"type": "float", "low": 1.0e-6, "high": 0.1, "log": True},
    "train.batch_size": [256, 512, 1024, 2048],
}
FORECAST = {
    "base": "g1_forecast_params",
    "objective": "mae",
    "train": {"min_epochs": 40, "patience": 15},
    "pruner": {"kind": "hyperband", "min_resource": 40, "reduction_factor": 3},
    "space": {
        **TRAIN_SPACE,
        "train.max_epochs": [80, 120, 200, 300],
        "train.loss": ["mae", "huber"],
        **MODEL_SPACE,
        # 1 s samples: 13 · 19.5 · 39 s are a third, half and one ALD cycle.
        "patch_seconds": [8, 13, 16, 19.5, 32, 39],
        "revin": [True, False],
        "revin_center": ["last", "mean"],
        "window.context_seconds": [64, 96, 128, 192, 256, 384, 512, 768, 1024],
    },
    "targets": {"vg11": "VG11", "apc": "APCValveMon", "tempc": "TempAct_C", "lpos": "MFCMon_L.POS"},
}
VS_SPACE = {
    **TRAIN_SPACE,
    "train.max_epochs": [40, 80, 120, 200],
    **MODEL_SPACE,
    # 0.5 s samples: 6.5 · 13 s are a sixth and a third of an ALD cycle.
    "patch_seconds": [2, 4, 6.5, 8, 13, 16],
    "revin": [True, False],
    "revin_center": ["last", "mean"],
    "window.context_seconds": [32, 48, 64, 96, 128, 192, 256],
}
VS_COMMON = {
    "train": {"min_epochs": 20, "patience": 10, "loss": "mae"},
    "pruner": {"kind": "hyperband", "min_resource": 20, "reduction_factor": 3},
    "space": VS_SPACE,
}
VS_PAIR = {"nh3": "MFCMon_NH3", "v29": "ValveAct_29", "dcs": "MFCMon_DCS", "apc": "APCValveMon", "vg11": "VG11",
           "ppos": "MFCMon_P.POS", "rpwr": "AUXMon_R.PWR"}
VS_1S = {"tempc": "TempAct_C"}


def study(name: str, target: str, base: str, objective: str, common: dict, ref_experiment: str,
          description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "base": base,
        "target": target,
        "model": "patchtst",
        "objective": objective,
        "trials": TRIALS,
        "seed": 42,
        "sampler": {"kind": "tpe", "multivariate": True, "n_startup_trials": 20, "constant_liar": True, "seed": 0},
        "pruner": common["pruner"],
        "train": common["train"],
        "fixed": {"channel_mode": "mixing"},
        "space": common["space"],
        "enqueue_from": {"experiment": ref_experiment, "metric": objective, "min_seeds": 2},
        "saturation": {"window": 50, "min_gain_pct": 1.0},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = []
    for key, target in FORECAST["targets"].items():
        docs.append(study(f"x1_fc_{key}", target, FORECAST["base"], FORECAST["objective"], FORECAST,
                          FORECAST["base"], f"미래값 예측 {target}: PatchTST 하이퍼파라미터 극한 탐색 (검증 MAE)"))
    for key, target in VS_PAIR.items():
        docs.append(study(f"x1_vs_{key}_full", target, "g1_virtual_sensor_res05", "mae_other", VS_COMMON,
                          "g1_virtual_sensor_res05",
                          f"가상 센서 {target} 0.5초 공정 전체 모델: 사이클 밖 MAE 기준 극한 탐색"))
        docs.append(study(f"x1_vs_{key}_cycle", target, "g1_virtual_sensor_res05_cycleonly", "mae", VS_COMMON,
                          "g1_virtual_sensor_res05_cycleonly",
                          f"가상 센서 {target} 0.5초 사이클 전용 모델: 사이클 안 MAE 기준 극한 탐색"))
    for key, target in VS_1S.items():
        docs.append(study(f"x1_vs_{key}", target, "g1_virtual_sensor", "mae", VS_COMMON, "g1_virtual_sensor",
                          f"가상 센서 {target} 1초 공정 전체 모델: 극한 탐색 (검증 MAE)"))
    for doc in docs:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", doc["name"])
        head = (f"# {doc['description']}\n# scripts/hpo_studies.py가 만든 파일. 고칠 때는 스크립트를 고치고 다시 만든다.\n"
                f"# 실행: ald-queue submit configs/hpo/{doc['name']}.yaml  (또는 ald-hpo run ... --gpus 0,1,2,3)\n")
        (OUT / f"{doc['name']}.yaml").write_text(head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                                                 encoding="utf-8")
    print(f"{len(docs)}개 → {OUT}")


if __name__ == "__main__":
    main()
