# 두께 값은 실제와 무관한 임의 값이다 (공개본).
from __future__ import annotations

import numpy as np
import pytest

from ald_lab.data.exsitu import LAYOUTS, _block_stats, radial_profile


def _points(rings: dict[float, int], value_at: callable) -> list[dict[str, object]]:
    rows = []
    point = 0
    for radius, count in rings.items():
        for k in range(count):
            angle = 2 * np.pi * k / count
            point += 1
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            for wap in range(1, 6):
                rows.append({"point": point, "x_mm": round(x, 1), "y_mm": round(y, 1), "wap_index": wap,
                             "thickness_a": value_at(radius, wap)})
    return rows


def test_49pt_layout_center_edge_and_wtow():
    rows = _points({0.0: 1, 46.7: 8, 93.3: 16, 140.0: 24}, lambda r, w: 100.0 - r / 140 + 0.5 * w)
    stats = _block_stats(rows, summary={})
    assert stats["layout"] == "49pt" and stats["n_points"] == 49
    assert stats["thk_center"] == pytest.approx(100.0 + 0.5 * 3)
    assert stats["thk_edge"] == pytest.approx(99.0 + 0.5 * 3)
    assert stats["edge_radius_mm"] == 140.0
    means = [100.0 - np.mean([0] + [46.7] * 8 + [93.3] * 16 + [140.0] * 24) / 140 + 0.5 * w for w in range(1, 6)]
    expected_wtow = (max(means) - min(means)) / (2 * np.mean(means)) * 100
    assert stats["wtow_pct"] == pytest.approx(expected_wtow)


def test_9pt_layout_detected_and_sheet_mismatch_reported():
    rows = _points({0.0: 1, 70.0: 4, 145.0: 4}, lambda r, w: 95.0 - r / 145)
    # A workbook summary computed over the wrong range (e.g. MIN missing a row).
    stats = _block_stats(rows, summary={"U/F": [0.1, None, None, None, None]})
    assert stats["layout"] == "9pt"
    assert stats["ring_radii_mm"] == "0/70/145"
    assert stats["sheet_check_max_abs_diff"] == pytest.approx(abs(0.1 - stats["uf_pct_wap1"]))


def test_empty_block_and_radial_profile():
    assert _block_stats([], summary={})["layout"] == "none"
    import pandas as pd

    rows = _points({0.0: 1, 70.0: 4, 145.0: 4}, lambda r, w: 95.0 - r / 145)
    frame = pd.DataFrame(rows).assign(block_id="B1")
    profile = radial_profile(frame)
    wap1 = profile[profile["wap_index"] == 1].set_index("radius_mm")
    assert list(wap1.index) == list(LAYOUTS["9pt"])
    assert wap1.loc[145.0, "thk_mean"] == pytest.approx(94.0)
    assert wap1.loc[70.0, "n"] == 4
