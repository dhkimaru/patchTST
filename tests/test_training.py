"""End-to-end training on synthetic traces (CPU): prepare → baselines → models → summary → final test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from ald_lab.journal import Journal  # noqa: E402
from ald_lab.models.deep import build_deep_model  # noqa: E402
from ald_lab.paths import resolve_paths  # noqa: E402
from ald_lab.training.config import expand_jobs, load_experiment, resolve_inputs  # noqa: E402
from ald_lab.training.prepare import build_bundle, run_arrays  # noqa: E402
from ald_lab.training.runner import run_experiment, run_final_test  # noqa: E402
from ald_lab.training.windows import WindowSet, window_ends  # noqa: E402
from ald_lab.data.splits import load_split  # noqa: E402
from ald_lab.data.raw import read_trace_csv  # noqa: E402

from .conftest import make_trace  # noqa: E402

RUNS = {"standard_trace_001": "train", "standard_trace_002": "train", "standard_trace_003": "train",
        "standard_trace_004": "validation", "standard_trace_005": "validation", "standard_trace_006": "test"}


@pytest.fixture
def lab(tmp_path: Path):
    data = tmp_path / "traces"
    data.mkdir()
    for i, run_id in enumerate(RUNS, start=1):
        frame = make_trace(f"2025-04-{10 + i} 10:00:00", seed=i, n_cycles=25,
                           gap_after_row=300 if run_id == "standard_trace_002" else None)
        frame.to_csv(data / f"{run_id}.csv", index=False)
    split = {"split_id": "synthetic", "excluded": [],
             **{p: [r for r, q in RUNS.items() if q == p] for p in ("train", "validation", "test")}}
    split_path = tmp_path / "split.yaml"
    split_path.write_text(yaml.safe_dump(split))
    paths = resolve_paths(data_root=data, work_root=tmp_path / "work")
    return paths, split_path


def _config(tmp_path: Path, split_path: Path, **overrides) -> Path:
    doc = {
        "name": "synthetic",
        "task": "forecast",
        "targets": ["VG11"],
        "data": {"split": str(split_path), "sample_seconds": 1.0, "region": "all", "inputs": "all"},
        "window": {"context_seconds": 16, "horizons_seconds": [2, 4]},
        "train": {"batch_size": 64, "max_epochs": 2, "windows_per_epoch": 256, "val_windows": 128, "patience": 2},
        "seeds": [1],
        "models": [
            "ridge",
            {"name": "lightgbm", "params": {"n_estimators": 20}},
            {"name": "patchtst", "params": {"patch_seconds": 4, "stride_seconds": 2, "d_model": 16, "n_heads": 2,
                                            "n_layers": 1, "d_ff": 32}, "grid": {"channel_mode": ["independent", "mixing"]}},
            {"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}},
            {"name": "itransformer", "params": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32}},
            {"name": "gru", "params": {"hidden": 16, "n_layers": 1, "train.lr": 0.003}},
        ],
    }
    for key, value in overrides.items():
        doc[key] = value
    path = tmp_path / f"{doc['name']}.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True))
    return path


def test_block_average_respects_time_breaks(lab):
    paths, _ = lab
    frame = read_trace_csv(paths.data_root / "standard_trace_002.csv")
    arrays = run_arrays(frame, ["VG11", "TempAct_C"], factor=2)
    # The 4 s gap after row 300 splits the run into two segments; no block straddles it.
    assert len(arrays["segment_lengths"]) == 2
    assert arrays["segment_lengths"][0] == 301 // 2
    raw = frame["VG11"].to_numpy()
    assert arrays["values"][0, 0] == pytest.approx(raw[:2].mean(), rel=1e-6)


def test_windows_never_cross_segments(lab):
    paths, split_path = lab
    split = load_split(split_path)
    bundle = build_bundle(paths, split, "train", sample_seconds=1.0, columns=["VG11", "TempAct_C"])
    ends = window_ends(bundle, context=16, horizons=(4,))
    seg = np.asarray(bundle.seg_id)
    assert (seg[ends - 15] == seg[ends]).all() and (seg[ends + 4] == seg[ends]).all()
    ws = WindowSet(np.asarray(bundle.values), np.asarray(bundle.values[:, :1]), ends, context=16, horizons=(2, 4))
    x, y = ws.gather(torch.arange(5))
    assert x.shape == (5, 16, 2) and y.shape == (5, 2, 1)
    e = ends[3]
    assert torch.allclose(x[3, -1], torch.as_tensor(np.asarray(bundle.values[e])))
    assert float(y[3, 1, 0]) == pytest.approx(float(bundle.values[e + 4, 0]))
    with pytest.raises(Exception):
        build_bundle(paths, split, "test", sample_seconds=1.0, columns=["VG11"])  # sealed


def test_virtual_sensor_drops_target_family(lab, tmp_path):
    paths, split_path = lab
    cfg = load_experiment(_config(tmp_path, split_path, task="virtual_sensor"))
    from ald_lab.data.columns import load_column_dictionary

    dictionary, _ = load_column_dictionary()
    inputs, removed = resolve_inputs(cfg, "VG11", usable=["VG11", "PressAct", "AUXMon_VG11", "TempAct_C"],
                                     dictionary=dictionary)
    assert inputs == ["TempAct_C"]
    assert set(removed["대상과 같은 계열·복사본"]) == {"VG11", "PressAct", "AUXMon_VG11"}
    # persistence is not allowed for virtual sensors; the template baseline is added automatically
    assert [m.name for m in cfg.models][0] == "template"


def test_revin_keeps_level_for_virtual_sensor():
    model = build_deep_model("dlinear", {}, n_channels=2, context=8, n_horizons=1, n_targets=1, target_channels=None)
    assert model.inject and not model.denorm
    shifted = torch.ones(1, 8, 2)
    # Same shape, different level: with statistics injected, the output depends on the level.
    assert not torch.allclose(model(shifted), model(shifted * 5))


def test_forecast_experiment_end_to_end(lab, tmp_path):
    paths, split_path = lab
    config = _config(tmp_path, split_path)
    jobs = expand_jobs(load_experiment(config))
    assert jobs[0].model == "persistence"
    assert sum(j.model == "patchtst" for j in jobs) == 2 and all(j.grid_size == 2 for j in jobs if j.model == "patchtst")
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    assert set(results["model"]) == {"persistence", "ridge", "lightgbm", "patchtst", "dlinear", "itransformer", "gru"}
    summary = json.loads((exp_dir / "summary.json").read_text())
    assert "VG11" in summary["selected"]
    assert summary["grid_choices"][0]["tried"] == 2
    job_dir = exp_dir / results.loc[results["model"] == "patchtst", "job_dir"].iloc[0]
    assert {"model.pt", "job.json", "history.json", "metrics_validation.json"} <= {p.name for p in job_dir.iterdir()}
    stages = [e.stage for e in Journal(paths.journal_dir).entries()]
    assert stages[0] == "preprocess" and stages[-1] == "evaluate" and "hpo" in stages and "train" in stages
    metrics = json.loads((job_dir / "metrics_validation.json").read_text())["metrics"]
    assert set(metrics["by_horizon"]) == {"2s", "4s"} and metrics["by_region"]["cycle"] is not None
    assert set(metrics["by_horizon_region"]) == {"2s", "4s"} and metrics["by_horizon_region"]["2s"]["cycle"] is not None
    assert set(metrics["by_transition"]) == {"first_3s", "later"}
    steps = metrics["by_step"]  # per recipe step name: [mae, windows]
    assert steps and sum(n for _, n in steps.values()) == metrics["n_windows"] * 2
    saved = pd.read_parquet(job_dir / "predictions_val.parquet")
    assert len(saved) == metrics["n_windows"]
    assert {"run_id", "row", "in_cycle", "true_2s", "pred_2s", "true_4s", "pred_4s"} <= set(saved.columns)
    assert abs((saved["pred_2s"] - saved["true_2s"]).abs().mean() - metrics["by_horizon"]["2s"]) < 1e-3 * (1 + metrics["mae"])

    test = run_final_test(exp_dir, paths, device="cpu")
    assert (exp_dir / "test_summary.md").is_file()
    assert set(test["model"]) >= {"persistence", "patchtst"}
    assert Journal(paths.journal_dir).entries()[-1].tags[-1] == "test"


def test_virtual_sensor_experiment(lab, tmp_path):
    paths, split_path = lab
    config = _config(tmp_path, split_path, task="virtual_sensor", name="vs",
                     models=["ridge", {"name": "itransformer", "params": {"d_model": 16, "n_heads": 2, "n_layers": 1}}],
                     data={"split": str(split_path), "sample_seconds": 0.5, "region": "process",
                           "input_sets": {"all": "all", "two": ["TempAct_C", "ValveAct_29"]}})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    assert set(results["label"]) == {"template", "ridge@all", "ridge@two", "itransformer@all", "itransformer@two"}
    doc = json.loads((exp_dir / results["job_dir"].iloc[-1] / "job.json").read_text())
    assert "VG11" not in doc["inputs"] and "PressAct" not in doc["inputs"]


def test_thickness_experiment(lab, tmp_path):
    paths, split_path = lab
    rng = np.random.default_rng(0)
    paths.overview_dir.mkdir(parents=True, exist_ok=True)
    # 두께 · U/F는 실제와 무관한 임의 값 (공개본)
    pd.DataFrame({"run_id": list(RUNS), "layout": "49pt", "thk_mean": 100.0 + rng.normal(0, 0.3, len(RUNS)),
                  "uf_pct_mean": 0.5 + rng.normal(0, 0.05, len(RUNS))}).to_parquet(
        paths.overview_dir / "exsitu_blocks.parquet")
    config = _config(tmp_path, split_path, task="thickness", name="vm", targets=["thk_mean", "uf_pct_mean"],
                     models=["ridge", {"name": "pls", "grid": {"n_components": [1, 2]}}], vm={"exclude_runs": []})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    assert set(results["model"]) == {"mean", "ridge", "pls"}
    test = run_final_test(exp_dir, paths, device="cpu")
    assert len(test) >= 2


def test_parallel_workers(lab, tmp_path):
    """The per-GPU worker pool (spawned processes), exercised with two CPU workers."""

    paths, split_path = lab
    config = _config(tmp_path, split_path, name="par", models=[
        {"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}},
        {"name": "gru", "params": {"hidden": 8, "n_layers": 1}},
    ], seeds=[1, 2])
    exp_dir = run_experiment(config, paths, devices=["cpu", "cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert len(results) == 5 and (results["status"] == "ok").all(), results.get("error")


def test_revin_last_starts_from_persistence():
    model = build_deep_model("dlinear", {"revin_center": "last"}, n_channels=2, context=8, n_horizons=3, n_targets=1,
                             target_channels=[1])
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    x = torch.randn(4, 8, 2)
    out = model(x)
    assert torch.allclose(out, x[:, -1:, 1:2].expand(-1, 3, -1), atol=1e-6)


def test_near_copies_found_on_train_rows(lab):
    paths, split_path = lab
    from ald_lab.training.windows import near_copies

    bundle = build_bundle(paths, load_split(split_path), "train", sample_seconds=1.0,
                          columns=["VG11", "PressAct", "AUXMon_VG11", "TempAct_C"])
    pairs = near_copies(bundle, bundle.columns)
    found = {frozenset(p) for p in zip(pairs["column_a"], pairs["column_b"])}
    assert frozenset({"VG11", "PressAct"}) in found and frozenset({"VG11", "AUXMon_VG11"}) in found
    assert not any("TempAct_C" in p for p in found)


def test_seed_reproducible_and_experiment_relocatable(lab, tmp_path):
    """Same seed → same result; a moved experiment folder still opens for the final test."""

    import shutil

    paths, split_path = lab
    config = _config(tmp_path, split_path, name="rep", models=[{"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}}])
    first = pd.read_csv(run_experiment(config, paths, devices=["cpu"]) / "results.csv")
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    second = pd.read_csv(exp_dir / "results.csv")
    mae = lambda r: float(r.loc[r["model"] == "dlinear", "mae"].iloc[0])  # noqa: E731
    assert mae(first) == pytest.approx(mae(second), rel=1e-6)
    moved = tmp_path / "moved"
    shutil.copytree(exp_dir, moved)
    shutil.rmtree(exp_dir)
    test = run_final_test(moved, paths, device="cpu")
    assert set(test["model"]) == {"persistence", "dlinear"}


def test_residual_template_in_forecast(lab, tmp_path):
    paths, split_path = lab
    config = _config(tmp_path, split_path, name="res", models=[{"name": "ridge", "params": {"residual": "template"}}])
    results = pd.read_csv(run_experiment(config, paths, devices=["cpu"]) / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")


def test_recipe_mean_baseline_and_side_study_in_forecast(lab, tmp_path):
    """Longer horizons compare against the recipe mean too; a side study never decides PatchTST adoption."""
    from ald_lab import stages
    from ald_lab.ledger import collect_ledger, main_records

    paths, split_path = lab
    config = _config(tmp_path, split_path, name="long", study="long_horizon",
                     window={"context_seconds": 16, "horizons_seconds": [2, 8]},
                     models=["template", {"name": "patchtst", "params": {"patch_seconds": 4, "stride_seconds": 2,
                                                                         "d_model": 16, "n_heads": 2, "n_layers": 1,
                                                                         "d_ff": 32}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    assert set(results["model"]) == {"persistence", "template", "patchtst"}
    metrics = json.loads((exp_dir / results.loc[results["model"] == "template", "job_dir"].iloc[0]
                          / "metrics_validation.json").read_text())["metrics"]
    assert set(metrics["by_horizon"]) == {"2s", "8s"} and np.isfinite(metrics["mae"])
    ledger = collect_ledger(paths.work_root)
    assert set(ledger["study"]) == {"long_horizon"} and main_records(ledger).empty
    assert ledger.loc[ledger["model"] == "template", ["mae_2s", "mae_8s"]].notna().all(axis=None)
    assert stages.patchtst_first(ledger, "forecast").empty


def test_deep_virtual_sensor_learns_template_correction(lab, tmp_path):
    paths, split_path = lab
    config = _config(tmp_path, split_path, task="virtual_sensor", name="vsdeep",
                     models=[{"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    job = exp_dir / results.loc[results["model"] == "dlinear", "job_dir"].iloc[0]
    assert torch.load(job / "model.pt", weights_only=False)["model_config"]["residual"] == "template"
    assert (job / "template.parquet").is_file()
    test = run_final_test(exp_dir, paths, device="cpu")
    assert set(test["model"]) == {"template", "dlinear"}


def test_scaler_and_template_use_train_rows_only(lab, tmp_path):
    """Normalization and the recipe template come from the train partition; validation rows never enter them."""

    from ald_lab.training.runner import DataCache

    paths, split_path = lab
    cfg = load_experiment(_config(tmp_path, split_path, task="virtual_sensor", name="leak", models=["template"],
                                  targets=["TempAct_C"],
                                  data={"split": str(split_path), "sample_seconds": 1.0, "region": "process"}))
    cache = DataCache(cfg, paths, load_split(split_path), "cpu")
    data = cache.target_data("TempAct_C")
    cols = data.scaler.columns
    train = np.asarray(cache.train_bundle.values)[:, cache.train_bundle.column_index(cols)].astype(np.float64)
    val = np.asarray(cache.val_bundle.values)[:, cache.val_bundle.column_index(cols)].astype(np.float64)
    assert np.allclose(data.scaler.mean, train.mean(axis=0)) and np.allclose(data.scaler.std, train.std(axis=0))
    pooled = np.concatenate([train, val]).mean(axis=0)
    assert not np.allclose(data.scaler.mean, pooled)  # validation rows would have moved it

    template = data.attach_template(cache.train_bundle, cache.val_bundle)
    y = data.train.y.cpu().numpy()[:, 0]
    expected = pd.DataFrame({"key": np.asarray(cache.train_bundle.step_key), "pos": np.asarray(cache.train_bundle.step_pos),
                             "y": y}).groupby(["key", "pos"])["y"].mean()
    assert np.allclose(template.table.iloc[:, 0].to_numpy(), expected.to_numpy())


def test_side_study_summary_script(lab, tmp_path, monkeypatch, capsys):
    """scripts/side_studies.py reads the learning-curve, recent-window, cycle-only and per-step results."""
    import runpy

    paths, split_path = lab
    tiny = {"name": "patchtst", "params": {"patch_seconds": 4, "stride_seconds": 2, "d_model": 16, "n_heads": 2,
                                           "n_layers": 1, "d_ff": 32}}
    for study, region, extra, ctx in (("lc51", "process", {}, 16), ("lc20", "process", {}, 16), ("lc20b", "process", {}, 16),
                                      ("recent4", "process", {}, 16), ("recent4_10", "process", {}, 16),
                                      ("region_cycle", "cycle", {}, 16), ("rw20", "process", {"recency_halflife_runs": 2}, 16),
                                      ("early_recent", "process", {}, 16), ("recent4_30", "process", {}, 16),
                                      ("region_cycle_win", "cycle", {}, 8)):
        config = _config(tmp_path, split_path, task="virtual_sensor", name=f"vs_{study}", study=study, seeds=[1, 2],
                         targets=["TempAct_C"], models=["template", tiny], window={"context_seconds": ctx},
                         train={"batch_size": 64, "max_epochs": 2, "windows_per_epoch": 256, "val_windows": 128,
                                "patience": 2, **extra},
                         data={"split": str(split_path), "sample_seconds": 1.0, "region": region, "inputs": "all"})
        results = pd.read_csv(run_experiment(config, paths, devices=["cpu"]) / "results.csv")
        assert (results["status"] == "ok").all(), results.get("error")
    monkeypatch.setenv("ALD_WORK_ROOT", str(paths.work_root))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "side_studies.py"), run_name="__main__")
    text = (paths.work_root / "reports" / "side_studies.txt").read_text(encoding="utf-8")
    assert "1-4. 학습 run 수 곡선" in text and ("줄어드는 중" in text or "포화" in text or "커짐" in text)
    assert "1-5. 최근 run 20개만으로 학습" in text and "053~067" in text
    assert "1-7. 사이클 구간만 학습" in text and "lc51" in text
    assert "4. 레시피 단계별 남은 오차" in text and "TempAct_C · PatchTST" in text
    assert "1-8. 학습 기간 길이" in text and "1-9. 학습 run 수 곡선 · 두 번째 추첨" in text
    assert "1-11. 최근 run 가중 학습" in text and "반감기 20개" in text
    assert "1-12. 초기 run(001~011) 포함 효과" in text and "1-13. 사이클만 학습 · window 길이" in text and "window 8초" in text


def test_schedule_baseline_projects_steps_without_future_labels(lab, tmp_path):
    """The recipe-schedule baseline reads the template at the planned step, and survives the final test."""
    from ald_lab.ledger import collect_ledger, main_records
    from ald_lab.models.classic import StepSchedule

    paths, split_path = lab
    split = load_split(split_path)
    bundle = build_bundle(paths, split, "train", sample_seconds=1.0, columns=["VG11", "TempAct_C"])
    plan = StepSchedule(cap=1200).fit(bundle)
    assert plan.duration and set(plan.duration) <= set(plan.next) | set(plan.duration)
    assert 0 <= plan.info["fixed_duration_share"] <= 1
    keys, pos = plan.future(np.asarray(bundle.step_key[:50]), np.asarray(bundle.step_pos[:50]), np.array([0, 3, 30]))
    assert (keys[:, 0] == np.asarray(bundle.step_key[:50])).all() and (pos[:, 0] == np.asarray(bundle.step_pos[:50])).all()

    config = _config(tmp_path, split_path, name="sched", study="long_horizon",
                     window={"context_seconds": 16, "horizons_seconds": [2, 8]}, models=["template", "schedule"])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    assert set(results["model"]) == {"persistence", "template", "schedule"}
    job_dir = exp_dir / results.loc[results["model"] == "schedule", "job_dir"].iloc[0]
    assert (job_dir / "schedule.json").is_file() and (job_dir / "template.parquet").is_file()
    assert pd.read_parquet(job_dir / "predictions_val.parquet")["pred_8s"].notna().all()
    ledger = collect_ledger(paths.work_root)
    assert main_records(ledger).empty
    test = run_final_test(exp_dir, paths, device="cpu", scope="all")
    assert "schedule" in set(test["model"])


def test_noncycle_region_keeps_process_rows_outside_cycles(lab):
    paths, split_path = lab
    bundle = build_bundle(paths, load_split(split_path), "train", sample_seconds=1.0, columns=["VG11"])
    ends = window_ends(bundle, 1, (1,), "noncycle")  # synthetic runs have only a few START rows outside cycles
    assert len(ends) and np.asarray(bundle.in_process)[ends].all() and not np.asarray(bundle.in_cycle)[ends].any()
    both = set(window_ends(bundle, 1, (1,), "cycle")) | set(ends)
    assert both == set(window_ends(bundle, 1, (1,), "process"))


def test_combine_predictions_script(lab, tmp_path, monkeypatch):
    """scripts/combine_predictions.py pairs region-specialized models, averages models and picks per horizon."""
    import runpy

    paths, split_path = lab
    vs_models = ["template", "ridge", {"name": "lightgbm", "params": {"n_estimators": 10}}]
    for study, region in (("lc51", "process"), ("region_cycle", "cycle"), ("region_noncycle", "noncycle")):
        config = _config(tmp_path, split_path, task="virtual_sensor", name=f"vs_{study}", study=study, targets=["TempAct_C"],
                         models=vs_models if study == "lc51" else ["template", "ridge"], window={"context_seconds": 1},
                         data={"split": str(split_path), "sample_seconds": 1.0, "region": region, "inputs": "all"})
        results = pd.read_csv(run_experiment(config, paths, devices=["cpu"]) / "results.csv")
        assert (results["status"] == "ok").all(), results.get("error")
    config = _config(tmp_path, split_path, name="long", study="long_horizon",
                     window={"context_seconds": 16, "horizons_seconds": [2, 8]},
                     models=["template", "schedule", "ridge", {"name": "lightgbm", "params": {"n_estimators": 10}}])
    assert (pd.read_csv(run_experiment(config, paths, devices=["cpu"]) / "results.csv")["status"] == "ok").all()
    monkeypatch.setenv("ALD_WORK_ROOT", str(paths.work_root))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "combine_predictions.py"), run_name="__main__")
    text = (paths.work_root / "reports" / "combine_predictions.txt").read_text(encoding="utf-8")
    assert "1. 가상 센서 · 구간 전용 모델 합치기" in text and "사이클 전용 + 사이클 밖 전용" in text
    assert "2. 가상 센서 · PatchTST와 LightGBM 예측 평균" in text
    assert "3. 미래값 30초~10분 · 예측 거리별 모델 비교와 선택" in text and "레시피 일정 평균" in text


def test_recency_weights_favour_recent_runs(lab, tmp_path):
    """Windows of newer training runs are drawn more often; the cached window set stays unweighted."""
    from ald_lab.training.runner import make_window_set
    from ald_lab.training.windows import fit_scaler, with_recency_weights

    paths, split_path = lab
    bundle = build_bundle(paths, load_split(split_path), "train", sample_seconds=1.0, columns=["VG11", "TempAct_C"])
    ws = make_window_set(bundle, ["TempAct_C"], "VG11", fit_scaler(bundle, ["TempAct_C", "VG11"]), context=8,
                         horizons=(0,), region="process", device="cpu")
    weighted = with_recency_weights(ws, bundle, 1.0)
    assert getattr(ws, "window_weights", None) is None and abs(weighted.window_weights.sum() - 1) < 1e-9
    run = np.asarray(bundle.run_index)[ws.ends_np]
    newest, oldest = len(bundle.run_ids) - 1, 0
    ratio = weighted.window_weights[run == newest][0] / weighted.window_weights[run == oldest][0]
    assert abs(ratio - 2 ** (len(bundle.run_ids) - 1)) < 1e-6
    config = _config(tmp_path, split_path, task="virtual_sensor", name="rw", targets=["TempAct_C"],
                     train={"batch_size": 64, "max_epochs": 2, "windows_per_epoch": 256, "val_windows": 128,
                            "patience": 2, "recency_halflife_runs": 1},
                     models=["template", {"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}}],
                     data={"split": str(split_path), "sample_seconds": 1.0, "region": "process", "inputs": "all"})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    doc = json.loads((exp_dir / results.loc[results["model"] == "dlinear", "job_dir"].iloc[0] / "job.json").read_text())
    assert doc["job"]["model"] == "dlinear"
    entries = Journal(paths.journal_dir).entries()
    assert any(e.config and (e.config.get("train") or {}).get("recency_halflife_runs") == 1 for e in entries)
