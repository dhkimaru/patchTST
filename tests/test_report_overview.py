"""요약 탭: conclusion sentences and the two charts, from a ledger like the dashboard's."""

from __future__ import annotations

import pandas as pd

from ald_lab.report import overview


def _row(target, model, mae, day, *, exp="r1", label=None, seed=42, task="forecast"):
    return {"task": task, "target": target, "model": model, "label": label or model, "mae": mae, "exp_dir": exp,
            "when": pd.Timestamp(day), "smoke": False, "unit": "", "settings": "", "input_set": "all",
            "experiment": "g1__hp", "seed": seed, "fit_minutes": 1.0}


def _ledger():
    rows = [_row(t, "persistence", 1.0, "2026-10-27 09:00", exp="b") for t in ("A", "B", "C")]
    rows += [_row("A", "lightgbm", 0.2, "2026-10-27 09:00", exp="b"), _row("B", "lightgbm", 0.5, "2026-10-27 09:00", exp="b")]
    rows += [_row("A", "patchtst", m, "2026-10-28 20:00", exp="d1", seed=s) for m, s in ((0.49, 42), (0.51, 43))]
    rows += [_row("B", "patchtst", 0.4, "2026-10-28 20:00", exp="d1"),  # one seed only
             _row("C", "patchtst", 1.1, "2026-10-28 20:00", exp="d1")]  # worse than the baseline
    rows += [_row("A", "patchtst", m, "2026-10-29 21:00", exp="d2", seed=s) for m, s in ((0.29, 42), (0.31, 43))]
    return pd.DataFrame(rows)


def test_conclusion_and_charts():
    ledger = _ledger()
    dec = overview.decisions(ledger)
    assert list(dec) == ["forecast"]  # no virtual-sensor records: no virtual-sensor lines
    frame = dec["forecast"].set_index("target")
    assert frame.loc["A", "decision"] == "keep" and frame.loc["C", "decision"] == "fallback"
    trends = {"forecast": overview.trend(ledger, "forecast", dec["forecast"])}
    days = [p["day"].strftime("%m-%d") for p in trends["forecast"]]
    assert days == ["10-28", "10-29"]  # 10-27 has only baselines; the line starts once every target has a record
    assert trends["forecast"][0]["mean"] > trends["forecast"][1]["mean"]
    assert abs(trends["forecast"][1]["mean"] - overview.geo_mean([30, 40, 110])) < 1e-9
    points = overview.conclusion_points(dec, trends, [("thk_mean", "LightGBM", 17.2), ("wtow_pct", "LightGBM", 28.9)])
    text = " ".join(points)
    assert "3개 대상 중 2개 PatchTST 채택" in text and "seed 1개로 편차 미확인" in text and "다른 모델 검토 1개" in text
    assert "LightGBM(참고) 대비 우위 1/2개 (열세: A)" in text
    assert "최근 실험일(10-29)" in text and "(−" in text and "17~29 %" in text
    chart = overview.target_chart(dec)
    assert 'aria-label="대상별 상대 오차' in chart and "대안 필요" in chart and "seed 1개" in chart
    assert chart.count('class="ov-pt"') == 3 + 1 and chart.count('class="ov-lgb"') == 2 + 1  # + legend marks
    trend = overview.trend_chart(trends)
    assert "LightGBM" in trend and "10-28" in trend and "10-29" in trend


def test_empty_ledger_draws_nothing():
    empty = pd.DataFrame()
    assert overview.decisions(empty) == {} and overview.target_chart({}) == "" and overview.trend_chart({}) == ""
    assert overview.conclusion_points({}, {}) == []
