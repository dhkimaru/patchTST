"""PatchTST first: random search in the experiment config, ranking, step-2/3 experiments and the per-target decision."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from ald_lab import stages
from ald_lab.launch import plan_jobs
from ald_lab.paths import PROJECT_ROOT
from ald_lab.training.config import apply_smoke, expand_jobs, load_experiment

SEARCHES = ("p1_patchtst_search_forecast", "p1_patchtst_search_vs")


@pytest.mark.parametrize("name", SEARCHES)
def test_search_configs_draw_valid_distinct_combinations(name):
    cfg = load_experiment(PROJECT_ROOT / "configs" / "experiments" / f"{name}.yaml")
    jobs = [j for j in expand_jobs(cfg) if j.model == "patchtst"]
    per_set = {}
    for j in jobs:
        per_set.setdefault(j.input_set, set()).add(tuple(sorted(j.params.items())))
        assert int(j.params["d_model"]) % int(j.params["n_heads"]) == 0
        assert j.params["stride_seconds"] <= j.params["patch_seconds"] <= cfg.context_seconds
    assert set(per_set) == {"all", "A_plus"} and all(len(v) == 16 for v in per_set.values())
    assert per_set["all"] == per_set["A_plus"]  # the same combinations on every input set and target
    assert [j.job_id for j in expand_jobs(load_experiment(PROJECT_ROOT / "configs" / "experiments" / f"{name}.yaml"))] \
        == [j.job_id for j in expand_jobs(cfg)]  # seeded: the same draw every time
    smoke = [j for j in expand_jobs(apply_smoke(cfg)) if j.model == "patchtst"]
    assert len({j.label for j in smoke}) <= 2 * 2


def _row(task, target, model, mae, *, base="g1", label=None, params=None, input_set="all", exp="r1", seed=42):
    return {"task": task, "target": target, "model": model, "label": label or model, "mae": mae, "base": base,
            "params": params or {}, "input_set": input_set, "exp_dir": exp, "seed": seed, "smoke": False,
            "unit": "Torr", "settings": str(params or {}), "experiment": base}


def test_ranking_and_stage2():
    a, b, c = {"d_model": 64}, {"d_model": 128}, {"d_model": 256}
    rows = [_row("forecast", t, "patchtst", m, base=stages.SEARCH_PREFIX + "_forecast", label=l, params=p, input_set=s)
            for t, scale in (("VG11", 1.0), ("APCValveMon", 10.0))
            for l, p, s, m in (("a", a, "all", 1.0), ("b", b, "all", 1.2), ("c", c, "A_plus", 1.1))
            for m in [m * scale]]
    rows.append(_row("forecast", "VG11", "lightgbm", 0.5))  # not part of the search
    ranking = stages.search_ranking(pd.DataFrame(rows), "forecast")
    assert [p["d_model"] for p in ranking["params"]] == [64, 256, 128]
    assert ranking.iloc[0]["score"] == pytest.approx(1.0) and ranking.iloc[1]["score"] == pytest.approx(1.1)
    base = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "g1_forecast_params.yaml").read_text())
    docs = stages.stage2_configs(ranking, base, k=2, seeds=[1, 2])
    assert [d["name"] for d in docs] == ["p2_patchtst_all_forecast_all", "p2_patchtst_all_forecast_A_plus"]
    assert docs[0]["targets"] == base["targets"] and docs[0]["data"]["inputs"] == "all"
    assert docs[0]["models"] == [{"name": "patchtst", "label": "patchtst_top1", "params": a}]
    assert docs[1]["models"][0]["label"] == "patchtst_top2" and docs[1]["seeds"] == [1, 2]
    assert len(plan_jobs(docs[0])) == len(base["targets"]) * (1 + 2)  # baseline + top-1 × 2 seeds per target
    single = stages.stage2_configs(ranking, base, k=1)
    assert [d["name"] for d in single] == ["p2_patchtst_all_forecast"]


def test_patchtst_first_decision_and_stage3():
    rows = [
        # A: PatchTST well ahead of the baseline, LightGBM a bit better → still PatchTST (LightGBM only for reference)
        _row("virtual_sensor", "A", "template", 2.0), _row("virtual_sensor", "A", "lightgbm", 1.00),
        _row("virtual_sensor", "A", "patchtst", 1.03, seed=1), _row("virtual_sensor", "A", "patchtst", 1.05, seed=2),
        # B: PatchTST 50 % behind LightGBM but clearly better than the baseline → kept
        _row("virtual_sensor", "B", "template", 2.0), _row("virtual_sensor", "B", "lightgbm", 1.0),
        _row("virtual_sensor", "B", "patchtst", 1.5), _row("virtual_sensor", "B", "itransformer", 1.2),
        # C: PatchTST worse than the baseline → fallback
        _row("virtual_sensor", "C", "template", 1.0), _row("virtual_sensor", "C", "patchtst", 1.2),
        # D: no PatchTST yet
        _row("virtual_sensor", "D", "template", 1.0), _row("virtual_sensor", "D", "lightgbm", 0.8),
        # E: PatchTST 2 % better than the baseline, within 2 seed SDs → fallback; iTransformer already better
        _row("virtual_sensor", "E", "template", 1.0), _row("virtual_sensor", "E", "itransformer", 0.9),
        _row("virtual_sensor", "E", "patchtst", 0.97, seed=1), _row("virtual_sensor", "E", "patchtst", 0.99, seed=2),
    ]
    frame = stages.patchtst_first(pd.DataFrame(rows), "virtual_sensor").set_index("target")
    assert frame.loc["A", "decision"] == "keep" and frame.loc["A", "patch_seeds"] == 2
    assert frame.loc["A", "patch_mae"] == pytest.approx(1.04) and frame.loc["A", "final"] == "patchtst"
    assert frame.loc["A", "patch_gain_pct"] == pytest.approx(48.0)
    assert frame.loc["B", "decision"] == "keep" and frame.loc["B", "final"] == "patchtst"
    assert frame.loc["B", "lgbm_gap_pct"] == pytest.approx(50.0)  # shown, never decides
    assert frame.loc["C", "decision"] == "fallback" and frame.loc["C", "reason"] == "기준선보다 나쁨"
    assert frame.loc["D", "decision"] == "missing"
    assert frame.loc["E", "decision"] == "fallback" and "seed 편차" in frame.loc["E", "reason"]
    assert frame.loc["E", "final"] == "itransformer" and frame.loc["E", "final_mae"] == pytest.approx(0.9)
    assert stages.summary_numbers(frame.reset_index()) == {"keep": 2, "fallback": 2, "missing": 1, "targets": 5}
    base = {"name": "g1_virtual_sensor", "task": "virtual_sensor", "targets": ["A", "B", "C", "D", "E"], "models": []}
    doc = stages.stage3_config(frame.reset_index(), base, seeds=[7])
    assert doc["name"] == "p3_fallback_virtual_sensor" and doc["targets"] == ["C", "E"] and doc["seeds"] == [7]
    assert [m["name"] for m in doc["models"]] == list(stages.FALLBACK_MODELS)
    assert stages.stage3_config(frame.reset_index().assign(decision="keep"), base) is None
    assert stages.to_records(frame.reset_index())[3]["patch_mae"] is None


def _design_rows(seed_noise: float = 0.3):
    """A 2 × 2 × 2 dashboard search on two targets, two seeds: patch helps by 5 %p, lr hurts by 3, d_model nothing."""

    rng = np.random.default_rng(0)
    rows = []
    for patch, d_model, lr in itertools.product((8, 32), (64, 256), (0.0003, 0.002)):
        for target, level in (("VG11", 60.0), ("APC", 20.0)):
            for seed in (42, 43):
                y = level + (5 if patch == 32 else 0) - (3 if lr == 0.002 else 0) + rng.normal(0, seed_noise)
                row = _row("forecast", target, "patchtst", 2.0 * (1 - y / 100), base="g1_forecast_params",
                           params={"patch_seconds": patch, "stride_seconds": 4, "d_model": d_model, "train.lr": lr},
                           exp="r1", seed=seed)
                rows.append({**row, "experiment": "g1_forecast_params__hp", "memo": "1차 탐색 · window 192초",
                             "improvement_pct": y, "context_seconds": 192.0, "mae_cycle": 0.3, "fit_minutes": 2.0,
                             "n_params": 1e5, "when": pd.Timestamp("2026-10-28"), "status": "완료"})
    for target in ("VG11", "APC"):
        rows.append({**_row("forecast", target, "persistence", 2.0, base="g1_forecast_params", exp="r1"),
                     "experiment": "g1_forecast_params__hp", "memo": "1차 탐색 · window 192초", "improvement_pct": None,
                     "context_seconds": 192.0, "mae_cycle": 0.4, "when": pd.Timestamp("2026-10-28"), "status": "완료"})
    return pd.DataFrame(rows)


def test_explore_groups_read_a_search_as_a_design():
    groups = stages.explore_groups(_design_rows())
    assert len(groups) == 1
    g = groups[0]
    assert g["title"] == "1차 탐색" and g["combos"] == 8 and g["seeds"] == [42, 43] and g["sigma"] < 1
    effects = {tuple(ef["factors"]): ef for ef in g["effects"]}
    assert set(effects) == {("patch_seconds",), ("d_model",), ("train.lr",)}  # stride never changed
    assert effects[("patch_seconds",)]["effect"] == pytest.approx(5, abs=0.5)
    assert effects[("patch_seconds",)]["verdict"] == "clear" and effects[("patch_seconds",)]["best"] == 32
    assert effects[("patch_seconds",)]["hint"] == "더 큰 값도 시도" and effects[("patch_seconds",)]["same_direction"]
    assert effects[("train.lr",)]["effect"] == pytest.approx(-3, abs=0.5) and effects[("train.lr",)]["best"] == 0.0003
    assert effects[("d_model",)]["verdict"] in ("unclear", "small") and effects[("d_model",)]["hint"].startswith("고정")
    top = g["ranking"][0]
    assert top["settings"]["patch_seconds"] == 32 and top["settings"]["train.lr"] == 0.0003 and top["seeds"] == 2
    assert top["cycle"] == pytest.approx(25.0)  # 1 − 0.3 / 0.4
    # the same search reaches step 2 with its window
    docs = stages.stage2_configs(stages.search_ranking(_design_rows(), "forecast"),
                                 {"name": "g1_forecast_params", "task": "forecast", "targets": ["VG11", "APC"],
                                  "window": {"context_seconds": 64}, "data": {"inputs": "all"}, "models": []}, k=2)
    assert len(docs) == 1 and docs[0]["window"]["context_seconds"] == 192.0
    assert docs[0]["models"][0]["params"]["patch_seconds"] == 32


def test_nice_labels():
    from ald_lab.report.render import nice_label

    assert nice_label("patchtst_mix[d_model=32,patch_seconds=16]") == "PatchTST 혼합 · d_model 32 · patch 16초"
    assert nice_label("patchtst#03") == "PatchTST #03"
    assert nice_label("pls[n_components=1]@core_ab") == "PLS · 성분 1 · 입력 core_ab"
    assert nice_label("patchtst_top2") == "PatchTST 상위 2"
    assert nice_label("patchtst[channel_mode=mixing,train.lr=0.002]") == "PatchTST · 혼합 · lr 0.002"


def test_queue_puts_running_first_and_scrolls(monkeypatch, tmp_path):
    from ald_lab.report import lab

    def request(i, status):
        return {"id": f"20261029-1{i:02d}000-000aa", "status": status, "name": "g1_forecast_params__hp", "memo": f"요청 {i}",
                "created": f"2026-10-29T1{i % 10}:00:00", "targets": ["VG11"], "gpus": "0"}

    # newest first, as launch.list_requests returns them
    requests = [request(9, "queued"), request(8, "queued"), request(7, "running")] + [request(i, "done") for i in range(6, 0, -1)]
    monkeypatch.setattr(lab, "list_requests", lambda root: list(requests))
    monkeypatch.setattr(lab, "worker_running", lambda root: True)
    page = lab.queue_html(tmp_path)
    order = [page.index(f"<b>요청 {i}</b>") for i in (7, 8, 9, 6, 1)]
    assert order == sorted(order)  # running, then queued in worker order, then the rest newest first
    assert 'class="lab-reqs scroll"' in page and "요청 9건" in page
    monkeypatch.setattr(lab, "list_requests", lambda root: requests[:lab.QUEUE_SHOWN])
    assert 'class="lab-reqs"' in lab.queue_html(tmp_path)


def test_stage_bases_exist():
    for name in stages.BASE_FOR_TASK.values():
        assert (Path(PROJECT_ROOT) / "configs" / "experiments" / f"{name}.yaml").is_file()
