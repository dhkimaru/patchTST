"""Default hyperparameters, importable without torch (the report lists what was not searched).

``classic.py`` and ``training/vm.py`` read their defaults from here; the deep backbones keep theirs
in their constructors and a test checks they match :data:`DEEP`.
"""

from __future__ import annotations

# Window-feature LightGBM (forecast / virtual sensor). n_estimators is a cap: early stopping picks the count.
LIGHTGBM = {
    "objective": "l2",  # l2 | l1 (matches an MAE metric) | huber
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "early_stopping_rounds": 50,
    "max_train_windows": 300_000,
    "max_val_windows": 100_000,
}
# Ridge on window features: the strength is chosen on validation among these.
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

# Run-level thickness models (about 60 runs, so small trees).
VM_LIGHTGBM = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 5}
VM_RIDGE_ALPHAS = "10^-2 ~ 10^5 (29개, run 교차검증으로 자동)"
VM_PLS = {"n_components": 2}

# Deep backbones (constructor defaults; patch/stride/kernel in steps).
DEEP = {
    "patchtst": {"patch_len": 16, "stride": 8, "d_model": 128, "n_heads": 16, "n_layers": 3, "d_ff": 256,
                 "dropout": 0.05, "attn_dropout": 0.0, "norm": "batch", "residual_attention": True,
                 "channel_mode": "independent"},
    "dlinear": {"kernel": 25, "d_rep": 64},
    "itransformer": {"d_model": 128, "n_heads": 8, "n_layers": 2, "d_ff": 256, "dropout": 0.1},
    "gru": {"hidden": 128, "n_layers": 2, "dropout": 0.1},
}
