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
