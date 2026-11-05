from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ald_lab.analysis.variability import (
    TemplateBuilder,
    _classify,
    _family_groups,
    _ridge_cross_r2,
    _with_keys,
    residual_frame,
)
from tests.conftest import make_trace


def _keyed(seed: int) -> pd.DataFrame:
    raw = make_trace("2025-04-15 10:00:00", seed=seed, n_cycles=4)
    frame = pd.DataFrame(
        {
            "recipe": raw["Recipe(Table) Name"],
            "step_id": raw["Step ID"],
            "step_name": raw["Step Name"],
            "VG11": raw["VG11"].astype(float),
            "ValveAct_29": raw["ValveAct_29"].astype(float),
        }
    )
    return _with_keys(frame)


def test_template_removes_recipe_shape():
    runs = [_keyed(s) for s in range(4)]
    builder = TemplateBuilder(["ValveAct_29", "VG11"])
    for keyed in runs[:3]:
        builder.add(keyed)
    resid = residual_frame(runs[3], builder.mean(), ["ValveAct_29", "VG11"])
    # The valve opens exactly on SI-FL1 in every run: the template explains it fully.
    assert np.allclose(resid["ValveAct_29"], 0.0)
    # VG11 is noise around 1.0: residuals keep its spread.
    assert resid["VG11"].std() == pytest.approx(0.1, rel=0.3)
    assert resid["in_cycle"].any() and not resid["in_cycle"].all()


def test_family_exclusion_hides_redundant_sensor():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(4000, 1))
    twin = signal + rng.normal(scale=0.01, size=(4000, 1))  # same-family sensor
    weak = 0.3 * signal + rng.normal(size=(4000, 1))         # unrelated family, weakly coupled
    data = np.hstack([signal, twin, weak])
    names = ["target", "twin", "weak"]
    copies = {n: {n} for n in names}
    family = pd.Series({"target": "fam", "twin": "fam", "weak": "other"})
    r2_copy, _, _ = _ridge_cross_r2(data[:3000], data[3000:], names, copies)
    r2_family, partner, _ = _ridge_cross_r2(data[:3000], data[3000:], names, _family_groups(copies, family))
    assert r2_copy[0] > 0.99
    assert r2_family[0] < 0.2 and partner[0] == "weak"


def test_classification_order():
    base = {"cycle_std": 1.0, "r2_recipe": 0.5, "r2_cross": 0.8, "binary": False, "max_unique_per_run": 500,
            "role": "measurement"}
    assert _classify(pd.Series(base)) == "공정 응답 · 다른 센서로 추정 가능"
    assert _classify(pd.Series({**base, "r2_cross": 0.1})) == "공정 응답 · 추정 어려움"
    assert _classify(pd.Series({**base, "r2_recipe": 0.995})) == "레시피 추종"
    assert _classify(pd.Series({**base, "max_unique_per_run": 3})) == "저해상도 신호"
    assert _classify(pd.Series({**base, "role": "setpoint"})) == "설정값·제어 명령"
    assert _classify(pd.Series({**base, "cycle_std": 0.0})) == "사이클 중 일정"
