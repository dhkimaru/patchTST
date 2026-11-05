"""Experiments started from the dashboard: request → queue → worker → cumulative records (synthetic data, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from ald_lab import launch
from ald_lab.data.columns import load_column_dictionary
from ald_lab.ledger import best_by_target, best_over_time, collect_ledger
from ald_lab.report.collect import collect
from ald_lab.report.render import render
from ald_lab.training.config import load_experiment, resolve_inputs

from .test_training import _config, lab  # noqa: F401  (fixture)


def _base(tmp_path: Path, split_path: Path) -> dict:
    return yaml.safe_load(_config(tmp_path, split_path).read_text(encoding="utf-8"))


def _space(model: str) -> dict:
    return {p.key: p for p in launch.param_space("forecast", model)}


def test_values_and_combinations():
    leaves = _space("lightgbm")["num_leaves"]
    assert launch.parse_values(leaves, "15, 31;31") == [15, 31]
    assert launch.parse_values(_space("patchtst")["revin"], ["true", False]) == [True, False]
    with pytest.raises(launch.LaunchError):
        launch.parse_values(leaves, "many")
    with pytest.raises(launch.LaunchError):
        launch.parse_values(leaves, "1.5")
    with pytest.raises(launch.LaunchError):
        launch.parse_values(_space("patchtst")["channel_mode"], "both")
    bad = {"window": {"context_seconds": 16}, "models": [
        {"name": "patchtst", "params": {"n_heads": 4}, "grid": {"d_model": [16, 30], "patch_seconds": [8, 32]}}]}
    problems = launch.check_combinations(bad)
    assert any("d_model 30" in p for p in problems) and any("patch 32초" in p for p in problems)
    # every model's defaults are valid values of its own settings
    for task in ("forecast", "thickness"):
        for model in launch.MODELS_BY_TASK[task]:
            for p in launch.param_space(task, model):
                assert launch.parse_values(p, [p.default]) == [p.default]


def test_build_config_expands_like_ald_train(lab, tmp_path):  # noqa: F811
    _, split_path = lab
    base = _base(tmp_path, split_path)
    doc = launch.build_config(
        base, targets=["VG11"], seeds=[1], input_sets=["all", "A_all"], context_seconds=16, memo="잎 수 비교",
        models={"lightgbm": {"num_leaves": [15, 31], "n_estimators": [20]},
                "patchtst": {"channel_mode": ["mixing"], "patch_seconds": [4], "stride_seconds": [2], "d_model": [16],
                             "n_heads": [2], "n_layers": [1], "d_ff": [32]}},
        train={"max_epochs": 2})
    assert doc["name"] == "synthetic__hp" and doc["description"] == "잎 수 비교"
    assert doc["data"]["input_sets"] == {"all": "all", "A_all": "A_all"} and "inputs" not in doc["data"]
    lgb = next(m for m in doc["models"] if m["name"] == "lightgbm")
    assert lgb["grid"] == {"num_leaves": [15, 31]} and lgb["params"] == {"n_estimators": 20}
    jobs = launch.plan_jobs(doc)
    # baseline once + LightGBM 2 settings × 2 input sets + PatchTST 1 × 2 input sets
    assert len(jobs) == 1 + 4 + 2 and jobs[0].model == "persistence"
    assert launch.build_config(doc, targets=["VG11"], models={"ridge": {}}, input_sets=["all"],
                               seeds=[1])["name"] == "synthetic__hp"  # rebuilding never stacks the suffix
    with pytest.raises(launch.LaunchError):
        launch.plan_jobs({**doc, "models": [{"name": "itransformer", "params": {"d_model": 30, "n_heads": 4}}]})
    with pytest.raises(launch.LaunchError):
        launch.build_config(base, targets=["VG11"], models={"pls": {}}, input_sets=["all"], seeds=[1])


def test_grade_input_sets(lab, tmp_path):  # noqa: F811
    _, split_path = lab
    columns, _ = load_column_dictionary()
    # 공개본 사전에는 현업 중요도가 없어 시험용 등급을 붙인다.
    graded = ["TempAct_U", "TempAct_L", "MFCMon_DCS", "MFCMon_NH3"]
    columns.loc[columns["column"].isin(graded), "importance"] = "A"
    columns.loc[columns["column"].isin(graded[::2]), "primary"] = True
    usable = columns.loc[columns["status"] == "OK", "column"].tolist()
    path = _config(tmp_path, split_path, task="virtual_sensor", targets=["TempAct_C"], models=["ridge"],
                   data={"split": str(split_path), "input_sets": {"A_all": "A_all", "A_plus": "A_plus", "core": "core"}})
    cfg = load_experiment(path)
    sets = {name: resolve_inputs(cfg, "TempAct_C", usable=usable, dictionary=columns, input_set=name)[0]
            for name in ("A_all", "A_plus", "core")}
    assert set(sets["core"]) < set(sets["A_all"]) < set(sets["A_plus"])
    assert "VG11" in sets["A_plus"] and "VG11" not in sets["A_all"] and "TempAct_C" not in sets["A_plus"]


def test_queue_worker_and_records(lab, tmp_path, monkeypatch):  # noqa: F811
    paths, split_path = lab
    base = _base(tmp_path, split_path)
    doc = launch.build_config(base, targets=["VG11"], seeds=[1], input_sets=["all"], memo="LightGBM 잎 수",
                              models={"ridge": {}, "lightgbm": {"num_leaves": [7, 15], "n_estimators": [20]}})
    jobs = launch.plan_jobs(doc)
    first = launch.submit(paths.work_root, doc, gpus="cpu", data_root=paths.data_root, n_jobs=len(jobs))
    second = launch.submit(paths.work_root, doc, gpus="cpu", data_root=paths.data_root, n_jobs=len(jobs))
    assert launch.cancel(paths.work_root, second["id"]) == "cancelled"
    assert [r["status"] for r in launch.list_requests(paths.work_root)] == ["cancelled", "queued"]
    assert not launch.worker_running(paths.work_root)

    assert launch.run_worker(paths.work_root) == 1  # runs the queued request in a child process, then stops
    done = {r["id"]: r for r in launch.list_requests(paths.work_root)}[first["id"]]
    assert done["status"] == "done" and done["returncode"] == 0, launch.log_tail(paths.work_root, first["id"])
    assert done["progress"] == (len(jobs), len(jobs)) and Path(done["exp_dir"]).parent.name == "synthetic__hp"

    records = collect_ledger(paths.work_root)
    assert len(records) == len(jobs) and set(records["source"]) == {"대시보드"}
    assert set(records["memo"]) == {"LightGBM 잎 수"} and records["settings"].str.contains("num_leaves=15").any()
    best = best_by_target(records)
    assert list(best["target"]) == ["VG11"] and best.iloc[0]["model"] in ("ridge", "lightgbm")
    assert len(best_over_time(records, "forecast", "VG11")) == 1

    # The report keeps dashboard runs out of its main tabs and lists them under Hyper Parameter ⑦.
    data = collect(paths)
    assert data.experiments == [] and len(data.ledger) == len(jobs)
    html = render(data, examples=False)
    assert "⑦ 누적 기록" in html and "LightGBM 잎 수" in html and "num_leaves=15" in html
