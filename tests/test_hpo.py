"""Hyperparameter search (ald_lab.hpo) on synthetic traces, CPU: study → trials → saturation report → follow-ups."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("optuna")

from ald_lab import hpo, launch
from ald_lab.ledger import collect_ledger, main_records
from ald_lab.models.deep import build_deep_model
from ald_lab.training.config import load_experiment
from ald_lab.training.runner import run_experiment
from ald_lab.training.trainer import TrainSettings, _lr_factor, predict, train_model
from ald_lab.training.windows import WindowSet

from .test_training import _config, lab  # noqa: F401  (fixture)

SMALL = {"channel_mode": "mixing", "patch_seconds": 4, "n_heads": 2, "n_layers": 1}


def _study(tmp_path: Path, base: Path, **overrides) -> Path:
    doc = {
        "name": "synthetic_search",
        "base": str(base),
        "target": "VG11",
        "model": "patchtst",
        "objective": "mae",
        "trials": 4,
        "sampler": {"kind": "tpe", "n_startup_trials": 2, "seed": 3},
        "pruner": {"kind": "halving", "min_resource": 1},
        "train": {"max_epochs": 3, "patience": 3},
        "fixed": SMALL,
        "space": {
            "train.lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "train.lr_schedule": ["plateau", "cosine"],
            "d_model": [8, 16],
            "d_ff_mult": [2, 4],
            "stride_ratio": [0.5, 1.0],
            "window.context_seconds": [12, 16],
        },
        "enqueue": [{"train.lr": 0.003, "train.lr_schedule": "plateau", "d_model": 16, "d_ff_mult": 2,
                     "stride_ratio": 0.5, "window.context_seconds": 16}],
        "saturation": {"window": 2, "min_gain_pct": 1.0},
    }
    doc.update(overrides)
    path = tmp_path / f"{doc['name']}.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True))
    return path


def test_study_runs_continues_and_reports(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    base = _config(tmp_path, split_path, models=["ridge"])
    study = _study(tmp_path, base)
    exp_dir = hpo.run_study(study, paths, devices=["cpu"])

    jobs = json.loads((exp_dir / "jobs.json").read_text())
    assert jobs[0]["model"] == "persistence" and len(jobs) == 5  # baseline + 4 trials
    results = pd.read_csv(exp_dir / "results.csv")
    assert (results["status"] == "ok").all(), results.get("error")
    trials = results[results["model"] == "patchtst"]
    assert len(trials) == 4 and trials["improvement_pct"].notna().all()
    assert not any((exp_dir / "jobs" / j / "model.pt").exists() for j in trials["job_id"])  # trials keep no weights
    assert not any((exp_dir / "jobs" / j / "predictions_val.parquet").exists() for j in trials["job_id"])
    first = json.loads((exp_dir / "jobs" / trials["job_id"].iloc[0] / "history.json").read_text())
    assert "val_mae" in first[0]

    summary = json.loads((paths.work_root / "hpo" / "synthetic_search" / "summary.json").read_text())
    counts = summary["counts"]
    assert counts["complete"] + counts["pruned"] == 4 and counts["running"] == 0
    assert summary["saturation"]["verdict"] in ("포화", "개선 중", "판단 보류")
    assert len(summary["expected_best"]) == counts["complete"]
    text = (paths.work_root / "reports" / "hpo_synthetic_search.txt").read_text(encoding="utf-8")
    assert "[포화 판정]" in text and "[포화 곡선]" in text
    assert "synthetic_search" in (paths.work_root / "reports" / "hpo_overview.txt").read_text(encoding="utf-8")
    assert (exp_dir / "hpo_summary.txt").is_file() and (exp_dir / "trials.csv").is_file()

    # The enqueued trial ran first with exactly its settings (derived keys resolved, window from the trial).
    enq = json.loads((exp_dir / "jobs" / trials["job_id"].iloc[0] / "job.json").read_text())
    assert enq["job"]["params"]["d_ff"] == 32 and enq["job"]["params"]["stride_seconds"] == 2.0
    assert enq["context_steps"] == 16

    # Trials stay out of the main records (validation-selected) but are in the ledger with their window length.
    led = collect_ledger(paths.work_root)
    rows = led[led["experiment"] == "synthetic_search"]
    assert len(rows) == 5 and set(rows["study"]) == {"hpo:synthetic_search"}
    assert not len(main_records(rows))
    assert set(rows[rows["model"] == "patchtst"]["context_seconds"].astype(float)) <= {12.0, 16.0}

    # A larger budget continues the same study instead of starting over.
    hpo.run_study(study, paths, devices=["cpu"], trials=6)
    again = json.loads((paths.work_root / "hpo" / "synthetic_search" / "summary.json").read_text())
    assert again["counts"]["complete"] + again["counts"]["pruned"] == 6

    # Follow-up: the best trials as an ordinary experiment with seeds (one document per window length).
    docs = hpo.top_configs(study, paths, k=2, seeds=[1, 2])
    assert sum(len(d["models"]) for d in docs) == 2 and all(d["seeds"] == [1, 2] for d in docs)
    assert all(d["study"] == "hpo_top:synthetic_search" for d in docs)
    for d in docs:
        out = tmp_path / f"{d['name']}.yaml"
        out.write_text(yaml.safe_dump(d, allow_unicode=True))
        cfg = load_experiment(out)
        assert cfg.models[0].name == "persistence" and all(m.params.get("d_ff") for m in cfg.models[1:])


def test_reference_from_the_ledger(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    base = _config(tmp_path, split_path, seeds=[1, 2],
                   models=[{"name": "patchtst", "params": {**SMALL, "d_model": 8, "d_ff": 16, "stride_seconds": 2}}])
    run_experiment(base, paths, devices=["cpu"])
    rows = hpo.reference_rows(paths.work_root, "synthetic", min_seeds=2)
    assert list(rows["target"]) == ["VG11"] and int(rows.iloc[0]["seeds"]) == 2
    assert rows.iloc[0]["sd"] >= 0 and rows.iloc[0]["params"]["d_model"] == 8
    # The CLI keeps one row per (experiment, model, target): two experiments of the same target do not overwrite.
    for experiment, metric in (("synthetic", "mae"), ("synthetic", "mae_cycle")):
        assert hpo.main(["reference", "--experiment", experiment, "--metric", metric, "--min-seeds", "2",
                         "--work-root", str(paths.work_root), "--data-root", str(paths.data_root)]) == 0
    ref = json.loads((paths.work_root / "hpo" / "reference.json").read_text())
    assert len(ref["rows"]) == 1 and ref["rows"][0]["metric"] == "mae_cycle"  # same key, the later call wins
    assert "VG11" in (paths.work_root / "reports" / "hpo_reference.txt").read_text(encoding="utf-8")
    assert hpo._seed_sd_pct(hpo.load_study(_study(tmp_path, base, name="sd_check", enqueue_from={
        "experiment": "synthetic"})), paths) == pytest.approx(ref["rows"][0]["sd_pct"])

    # enqueue_from turns the reference into the first trial; settings outside the space ride along as extras.
    study = _study(tmp_path, base, name="ref_search", trials=2, enqueue=[],
                   enqueue_from={"experiment": "synthetic", "min_seeds": 2},
                   space={"train.lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}, "d_model": [8, 16]})
    exp_dir = hpo.run_study(study, paths, devices=["cpu"])
    summary = json.loads((paths.work_root / "hpo" / "ref_search" / "summary.json").read_text())
    jobs = json.loads((exp_dir / "jobs.json").read_text())
    ref = next(j for j in jobs if j["label"] == "patchtst#ref")
    assert ref["params"]["d_ff"] == 16 and ref["params"]["d_model"] == 8
    assert ref["params"]["train.patience"] == 2 and ref["params"]["train.max_epochs"] == 2  # the run's own training
    if summary["counts"]["complete"] and summary["reference"]:
        assert summary["reference"]["number"] == 0


def test_study_validation(tmp_path, lab):  # noqa: F811
    paths, split_path = lab
    base = _config(tmp_path, split_path)
    with pytest.raises(hpo.StudyError):
        hpo.load_study(_study(tmp_path, base, objective="rmse"))
    with pytest.raises(hpo.StudyError):
        hpo.load_study(_study(tmp_path, base, space={"train.lr_typo": [0.1]}))
    with pytest.raises(hpo.StudyError):
        hpo.load_study(_study(tmp_path, base, enqueue=[{"d_model": 99}]))
    spec = hpo.load_study(_study(tmp_path, base))
    assert spec.space["d_model"] == {"type": "categorical", "choices": [8, 16]}
    assert hpo.is_study_file(spec.path) and not hpo.is_study_file(base)
    params, context = hpo.materialize({"patch_seconds": 8, "stride_ratio": 0.25, "d_model": 64, "d_ff_mult": 3,
                                       "window.context_seconds": 32}, 16)
    assert params == {"patch_seconds": 8, "stride_seconds": 2.0, "d_model": 64, "d_ff": 192} and context == 32
    explicit, _ = hpo.materialize({"d_model": 64, "d_ff_mult": 3, "d_ff": 512, "patch_seconds": 8,
                                   "stride_ratio": 0.5, "stride_seconds": 8}, 16)
    assert explicit["d_ff"] == 512 and explicit["stride_seconds"] == 8  # an explicit value beats a derived one
    with pytest.raises(hpo.StudyError):
        hpo.load_study(_study(tmp_path, base, space={"data.input_set": ["all", "nonsense"]}))
    with pytest.raises(hpo.StudyError):
        hpo.load_study(_study(tmp_path, base, fixed={**SMALL, "d_ff": 32}))  # d_ff_mult in the space contradicts
    assert hpo.trial_problems("patchtst", {"patch_seconds": 40, "attn_dropout": 0.0}, 32)  # patch longer than window
    assert not hpo.trial_problems("patchtst", {"patch_seconds": 8, "attn_dropout": 0.0, "head_dropout": 0.0}, 32)

    # The queue takes a study like an experiment and runs it with ald-hpo.
    doc = yaml.safe_load(spec.path.read_text(encoding="utf-8"))
    record = launch.submit(paths.work_root, doc, gpus="cpu", data_root=paths.data_root)
    assert record["kind"] == "hpo" and record["n_jobs"] == 4 and record["targets"] == ["VG11"]
    assert launch.is_study_doc(doc) and not launch.is_study_doc(yaml.safe_load(base.read_text()))


def test_saturation_and_expected_best():
    assert hpo.expected_best([3.0])[0] == 3.0
    curve = hpo.expected_best([5.0, 4.0, 3.0, 2.0])
    assert curve[0] == pytest.approx(3.5) and curve[-1] < curve[0] and curve == sorted(curve, reverse=True)
    still = hpo.saturation([10, 9, 8, 7, 6, 5], window=2, min_gain_pct=1.0, seed_sd_pct=None)
    assert still["verdict"] == "개선 중" and still["gain_pct"] == pytest.approx(100 * (7 - 5) / 7)
    flat = hpo.saturation([10, 5, 5.01, 5.02, 4.99], window=3, min_gain_pct=1.0, seed_sd_pct=0.5)
    assert flat["verdict"] == "포화" and flat["threshold_pct"] == 1.0
    noisy = hpo.saturation([10, 5, 5.01, 4.9], window=2, min_gain_pct=1.0, seed_sd_pct=3.0)
    assert noisy["verdict"] == "포화" and noisy["threshold_pct"] == 3.0  # a 2 % gain is inside the seed spread
    assert hpo.saturation([3, 2], window=5, min_gain_pct=1.0, seed_sd_pct=None)["verdict"] == "판단 보류"


def _tiny_sets():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3)).astype(np.float32)
    y = x[:, :1] * 0.5
    ends = np.arange(20, 390)
    make = lambda: WindowSet(x, y, ends, context=16, horizons=(2,))
    return make(), make()


class _SmallGpu(torch.nn.Module):
    """Runs out of "GPU memory" above 8 windows, like a 500-patch setting with batch 2048 on a 24 GB card."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > 8:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 34.00 GiB")
        return self.inner(x)


def test_out_of_memory_splits_the_batch_and_keeps_its_meaning():
    import copy

    train, val = _tiny_sets()
    plain = build_deep_model("dlinear", {"kernel": 5, "d_rep": 8}, n_channels=3, context=16, n_horizons=1,
                             n_targets=1, target_channels=[0])
    small = _SmallGpu(copy.deepcopy(plain))
    settings = TrainSettings(batch_size=32, max_epochs=1, windows_per_epoch=32, val_windows=16, eval_batch_size=64,
                             lr=1e-2, lr_schedule="constant", amp=False)
    logs: list[str] = []
    out = train_model(small, train, val, settings, log=logs.append)
    ref = train_model(plain, train, val, settings, log=lambda m: None)
    assert out["status"] == ref["status"] and out["batch_parts"] == 4 and out["eval_batch_size"] == 8
    assert any("4조각" in m for m in logs) and any("검증 batch 64 → 8" in m for m in logs)
    for a, b in zip(small.inner.parameters(), plain.parameters()):  # one step on 4 pieces = one step on the batch
        assert torch.allclose(a, b, atol=1e-6)
    pred = predict(small, val, np.arange(40), batch_size=64, amp=False)
    assert pred.shape == (40, 1, 1) and np.allclose(pred, predict(plain, val, np.arange(40), amp=False), atol=1e-6)

    with pytest.raises(RuntimeError):  # another error is not retried
        train_model(_Broken(), train, val, settings, log=lambda m: None)


class _Broken(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("shape mismatch")


def test_trainer_prunes_waits_and_schedules():
    train, val = _tiny_sets()
    model = build_deep_model("dlinear", {"kernel": 5, "d_rep": 8}, n_channels=3, context=16, n_horizons=1,
                             n_targets=1, target_channels=[0])
    settings = TrainSettings(batch_size=64, max_epochs=5, windows_per_epoch=256, val_windows=128, patience=1,
                             lr=1e-2)
    seen = []
    out = train_model(model, train, val, settings, log=lambda m: None,
                      on_epoch=lambda e, v: seen.append((e, v)) or e >= 2)
    assert out["status"] == "PRUNED" and out["epochs"] == 2 and [e for e, _ in seen] == [1, 2]
    assert out["best_val_mae"] == pytest.approx(min(v for _, v in seen))

    # min_epochs: early stopping (patience 1) cannot end training before epoch 5 even if validation stalls.
    long = TrainSettings(batch_size=64, max_epochs=5, windows_per_epoch=256, val_windows=128, patience=1, lr=1e-7,
                         min_epochs=5)
    out = train_model(build_deep_model("dlinear", {"kernel": 5, "d_rep": 8}, n_channels=3, context=16, n_horizons=1,
                                       n_targets=1, target_channels=[0]), train, val, long, log=lambda m: None)
    assert out["epochs"] == 5

    cosine = TrainSettings(max_epochs=10, lr_schedule="cosine", warmup_epochs=2)
    factor = _lr_factor(cosine)
    assert factor(0) < factor(1) < factor(2) == pytest.approx(1.0) and factor(9) < 0.1
    assert _lr_factor(TrainSettings(lr_schedule="constant"))(5) == 1.0
    with pytest.raises(ValueError):
        TrainSettings.from_dict({"lr_schedule": "onecycle"})


def test_parallel_workers_share_one_study(lab, tmp_path):  # noqa: F811
    """Two worker processes (one per device) ask and tell through the same journal file; the budget holds."""
    paths, split_path = lab
    base = _config(tmp_path, split_path, models=["ridge"])
    study = _study(tmp_path, base, name="parallel_search", trials=4, pruner={"kind": "none"})
    exp_dir = hpo.run_study(study, paths, devices=["cpu", "cpu"])
    results = pd.read_csv(exp_dir / "results.csv")
    trials = results[results["model"] == "patchtst"]
    assert (results["status"] == "ok").all(), results.get("error")
    assert 4 <= len(trials) <= 5 and trials["trial"].is_unique  # at most one extra trial from a race at the end
    summary = json.loads((paths.work_root / "hpo" / "parallel_search" / "summary.json").read_text())
    assert summary["counts"]["complete"] == len(trials) and summary["counts"]["running"] == 0
    assert len(json.loads((exp_dir / "jobs.json").read_text())) == len(trials) + 1


def _ledger_reference(tmp_path, split_path, paths):
    """A finished two-seed PatchTST run that ``enqueue_from`` can point at (patience 2, two epochs, plateau)."""
    base = _config(tmp_path, split_path, seeds=[1, 2],
                   models=[{"name": "patchtst", "params": {**SMALL, "d_model": 8, "d_ff": 16, "stride_seconds": 2}}])
    run_experiment(base, paths, devices=["cpu"])
    return base


def test_current_setting_trial_trains_exactly_as_recorded(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    base = _ledger_reference(tmp_path, split_path, paths)
    study = _study(tmp_path, base, name="exact_ref", trials=2, enqueue=[], train={"max_epochs": 3, "min_epochs": 3},
                   enqueue_from={"experiment": "synthetic", "min_seeds": 2},
                   space={"train.lr_schedule": ["plateau", "cosine"], "train.weight_decay": [0.0001, 0.01],
                          "norm": ["batch", "layer"], "head_dropout": [0.0, 0.1], "d_model": [8, 16],
                          "stride_ratio": [0.5, 1.0], "window.context_seconds": [12, 16]})
    exp_dir = hpo.run_study(study, paths, devices=["cpu"])
    ref = next(j for j in json.loads((exp_dir / "jobs.json").read_text()) if j["label"] == "patchtst#ref")
    p = ref["params"]
    # keys the record does not name take the run's values or the defaults, never the sampler's
    assert p["train.lr_schedule"] == "plateau" and p["train.weight_decay"] == 0.0001 and p["norm"] == "batch"
    assert p["head_dropout"] == 0.0 and p["stride_seconds"] == 2 and p["d_ff"] == 16
    assert p["train.max_epochs"] == 2 and p["train.patience"] == 2 and p["train.min_epochs"] == 0  # its own training
    job = json.loads((exp_dir / "jobs" / ref["job_id"] / "job.json").read_text())
    assert job["context_steps"] == 16
    summary = json.loads((paths.work_root / "hpo" / "exact_ref" / "summary.json").read_text())
    assert summary["reference"]["version"] == hpo.REF_VERSION and summary["reference"]["number"] == 0
    assert summary["progress"]["finished"] == 2 and summary["progress"]["left"] == 0
    text = (paths.work_root / "reports" / "hpo_exact_ref.txt").read_text(encoding="utf-8")
    assert "{'n'" not in text and "[진행]" in text  # the curve prints numbers, the pace line is there


def test_reference_keeps_runs_that_trained_differently_apart(lab, tmp_path):  # noqa: F811
    """Two runs with the same model settings but another train block are two references: the mean and the
    reproduced training come from the same runs (REF_VERSION 3). A trial far from its recorded mean is flagged."""

    from types import SimpleNamespace

    paths, split_path = lab
    base = _ledger_reference(tmp_path, split_path, paths)  # seeds 1, 2 with max_epochs 2
    model = [{"name": "patchtst", "params": {**SMALL, "d_model": 8, "d_ff": 16, "stride_seconds": 2}}]
    # a dashboard run of the same experiment (base "synthetic") that trained otherwise
    other = _config(tmp_path, split_path, name="synthetic__hp", seeds=[3], models=model,
                    train={"batch_size": 64, "max_epochs": 1, "windows_per_epoch": 128, "val_windows": 128,
                           "patience": 2, "lr_schedule": "cosine"})
    run_experiment(other, paths, devices=["cpu"])
    led = collect_ledger(paths.work_root)
    assert led["train"].nunique() == 2 and led["train"].str.contains("lr_schedule=cosine").any()
    rows = hpo.reference_rows(paths.work_root, "synthetic", min_seeds=2)
    assert int(rows.iloc[0]["seeds"]) == 2 and "max_epochs=2" in rows.iloc[0]["train"]  # not the 3-seed mix
    spec = hpo.load_study(_study(tmp_path, base, name="v3_ref", enqueue_from={"experiment": "synthetic", "min_seeds": 2},
                                 space={"d_model": [8, 16], "train.lr_schedule": ["plateau", "cosine"]}))
    in_space, extra, expected = hpo._reference_trial(spec, paths.work_root, load_experiment(base))
    assert in_space["train.lr_schedule"] == "plateau" and extra["train.max_epochs"] == 2
    assert expected["seeds"] == 2 and expected["mean"] == pytest.approx(float(rows.iloc[0]["mean"]))

    trial = lambda value, attrs: SimpleNamespace(value=value, user_attrs=attrs)
    ok = hpo.reference_check(spec, paths, trial(1.05, {"expected": {"mean": 1.0, "sd": 0.01, "seeds": 3}}))
    assert ok["ok"] and ok["limit_pct"] == hpo.REF_TOLERANCE_PCT and ok["gap_pct"] == pytest.approx(5.0)
    bad = hpo.reference_check(spec, paths, trial(1.425, {"expected": {"mean": 1.0, "sd": 0.064, "seeds": 3}}))
    assert not bad["ok"] and bad["limit_pct"] == pytest.approx(19.2)  # 3 seed SD is wider than 10 %
    assert hpo.reference_check(spec, paths, trial(1.0, {})) is None  # no record: nothing to compare
    hpo.main(["reference", "--experiment", "synthetic", "--min-seeds", "2", "--work-root", str(paths.work_root),
              "--data-root", str(paths.data_root)])  # a trial queued before the expectation was kept uses this table
    old = hpo.reference_check(spec, paths, trial(float(rows.iloc[0]["mean"]) * 2, {}))
    assert old is not None and not old["ok"] and old["gap_pct"] == pytest.approx(100.0)
    assert "train" in (paths.work_root / "reports" / "hpo_reference.txt").read_text(encoding="utf-8")


def test_dashboard_flags_a_reference_that_did_not_reproduce(monkeypatch):
    from ald_lab.report.lab import hpo_html

    row = {"name": "x1_demo", "file": "x1_demo.yaml", "description": "", "target": "MFCMon_NH3", "model": "patchtst",
           "objective": "mae", "unit": "slm", "budget": 150, "task": "virtual_sensor", "request": None,
           "report": "hpo_x1_demo.txt", "finished": 40, "state": "running", "progress": {"per_trial_min": 1.8},
           "counts": {"complete": 29, "pruned": 9, "fail": 2}, "reference": 0.07487, "ref_version": 2,
           "ref_check": {"mean": 0.05253, "sd": 0.003367, "seeds": 3, "gap_pct": 42.5, "limit_pct": 19.2, "ok": False},
           "best": 0.05345, "gain_pct": 28.6, "saturation": "판단 보류"}
    status = {"rows": [row], "states": {"done": 0, "running": 1, "queued": 0, "partial": 0, "new": 0},
              "finished": 40, "budget": 150, "queue_end": None}
    monkeypatch.setattr(hpo, "studies_status", lambda paths, folder=None: status)
    page = hpo_html(None, live=True)
    assert "⚠ 기록과 +42%" in page and "⚠ 이전 방식" in page and "data-hpo-requeue" in page  # v2: put v3 in
    assert '<div class="sub">참고</div>' in page  # 지금 대비 is only indicative
    row.update(ref_version=hpo.REF_VERSION, ref_check={**row["ref_check"], "gap_pct": 3.0, "ok": True})
    page = hpo_html(None, live=True)
    assert "⚠" not in page and "data-hpo-requeue" not in page


def test_an_old_style_reference_is_replaced(lab, tmp_path, monkeypatch):  # noqa: F811
    paths, split_path = lab
    base = _ledger_reference(tmp_path, split_path, paths)
    study = _study(tmp_path, base, name="old_ref", trials=3, enqueue=[], pruner={"kind": "none"},
                   enqueue_from={"experiment": "synthetic", "min_seeds": 2},
                   space={"d_model": [8, 16], "norm": ["batch", "layer"]})
    spec = hpo.load_study(study)
    hpo.open_study(spec, paths).enqueue_trial({"d_model": 16, "norm": "layer"},
                                              user_attrs={"role": "reference", "extra_params": {}})  # version 1
    hpo.run_study(study, paths, devices=["cpu"])
    summary = json.loads((paths.work_root / "hpo" / "old_ref" / "summary.json").read_text())
    assert summary["reference"]["version"] == hpo.REF_VERSION and summary["reference"]["number"] == 1
    assert not hpo.enqueue_reference(spec, paths)  # already there: nothing queued twice
    # both current-setting trials side by side: the keys they differ on, and the record they should reproduce
    refs = summary["references"]
    assert [t["version"] for t in refs["trials"]] == [1, hpo.REF_VERSION] and "train.max_epochs" in refs["differ"]
    assert refs["record"] and "d_model=8" in refs["record"]["settings"] and "max_epochs=2" in refs["record"]["train"]
    report = paths.work_root / "reports" / "hpo_old_ref.txt"
    assert "[지금 설정 trial 비교]" in report.read_text(encoding="utf-8")

    # the dashboard link writes a missing report from the study journal (a study run by older code)
    from ald_lab.server import Lab

    report.unlink()
    monkeypatch.setattr(hpo, "study_files", lambda folder_=None: [study])
    text = Lab(paths, examples=False).report_text("hpo_old_ref.txt")
    assert text and "[지금 설정 trial 비교]" in text and "trial 1 (3판" in text


def test_failed_trials_explain_themselves(lab, tmp_path, monkeypatch):  # noqa: F811
    """Why a trial failed: the error kept on the trial (new code), a broken combination, or only the pipeline
    journal (a study started by older code); the dashboard and the text report list each one with its settings."""

    from optuna.trial import TrialState

    from ald_lab.report.lab import hpo_html

    paths, split_path = lab
    base = _config(tmp_path, split_path, models=["ridge"])
    path = _study(tmp_path, base, name="fails", trials=5, enqueue=[])
    spec = hpo.load_study(path)
    study = hpo.open_study(spec, paths)
    settings = {"window.context_seconds": 16, "d_model": 16, "stride_ratio": 0.5}
    for attrs in ({"error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 34.00 GiB"},
                  {"problems": ["patchtst: patch 20초가 window 16초보다 깁니다"]}, {}, {}):
        trial = study.ask()
        for key, value in {**attrs, "actual": {**settings, "patch_seconds": 2, "train.batch_size": 2048}}.items():
            trial.set_user_attr(key, value)
        study.tell(trial, state=TrialState.FAIL)
    journal = Path(paths.journal_dir) / "journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    line = {"stage": "train", "status": "failed", "error": "torch.OutOfMemoryError: CUDA out of memory.",
            "config": {"experiment": "fails", "label": "patchtst#t0002"}}
    other = {**line, "config": {"experiment": "fails_other", "label": "patchtst#t0003"}}
    journal.write_text(json.dumps(line, ensure_ascii=False) + "\n" + json.dumps(other) + "\n", encoding="utf-8")

    found = hpo.failures(spec, study.get_trials(deepcopy=False), paths)
    assert [f["kind"] for f in found] == ["GPU 메모리 부족", "설정 조합 불가", "GPU 메모리 부족", "기록 없음"]
    assert "patch 16개" in found[0]["settings"] and "batch 2048" in found[0]["settings"]  # (16 - 2) // 1 + 2 (end pad)
    # the size the GPU reported for two real failures: batch × heads × patches² × 4 bytes
    big = hpo.brief_settings({"window.context_seconds": 256, "patch_seconds": 2, "stride_ratio": 0.5, "n_heads": 16,
                              "train.batch_size": 2048, "stride_seconds": 1})
    assert "patch 256개" in big and "attention 점수 한 층 8.00 GiB" in big
    assert "7.94 GiB" in hpo.brief_settings({"window.context_seconds": 256, "patch_seconds": 2, "stride_seconds": 0.5,
                                             "n_heads": 8, "train.batch_size": 1024})
    assert found[3]["error"] == hpo.FAIL_UNKNOWN  # the other study's journal line is not taken
    assert hpo.fail_kind("검증 mae가 NaN: 학습 발산 (NAN_LOSS)") == "학습 발산 (NaN)"

    with journal.open("a", encoding="utf-8") as handle:  # read incrementally: a later line is still picked up
        handle.write(json.dumps({**line, "config": {"experiment": "fails", "label": "patchtst#t0003"},
                                 "error": "RuntimeError: worker exited"}) + "\n")
    assert hpo.failures(spec, study.get_trials(deepcopy=False), paths)[3]["kind"] == "작업자 중단"

    text = (Path(paths.work_root) / "reports" / "hpo_fails.txt")
    hpo.report_study(spec, paths, importance=False)
    assert "[실패] 4개: GPU 메모리 부족 2" in text.read_text(encoding="utf-8")
    monkeypatch.setattr(hpo, "study_files", lambda folder_=None: [path])
    page = hpo_html(paths, live=True)
    assert 'class="tw lab-hpo-scroll"' in page and 'data-key="fail-fails"' in page
    assert "실패 원인: GPU 메모리 부족 2, 설정 조합 불가 1, 작업자 중단 1" in page and "trial 0" in page


def test_dashboard_lists_studies_with_progress_and_end_times(lab, tmp_path, monkeypatch):  # noqa: F811
    from ald_lab.report.lab import hpo_html
    from ald_lab.server import Lab

    paths, split_path = lab
    base = _config(tmp_path, split_path, models=["ridge"])
    folder = tmp_path / "hpo_cfg"
    folder.mkdir()
    done = _study(folder, base, name="study_done", trials=2, enqueue=[])
    _study(folder, base, name="study_next", trials=3, enqueue=[])
    _study(folder, base, name="study_new", trials=2, enqueue=[])
    monkeypatch.setattr(hpo, "study_files", lambda folder_=None: sorted(folder.glob("*.yaml")))
    monkeypatch.setattr(launch, "ensure_worker", lambda work_root: False)
    hpo.run_study(done, paths, devices=["cpu"])

    server = Lab(paths, examples=False)
    ids = server.hpo_submit({"names": ["study_next"], "gpus": "cpu"})["ids"]
    with pytest.raises(launch.LaunchError):
        server.hpo_submit({"names": ["study_next"], "gpus": "cpu"})  # already waiting
    with pytest.raises(launch.LaunchError):
        server.hpo_submit({"names": ["nope"], "gpus": "cpu"})
    status = hpo.studies_status(paths, folder)
    rows = {r["name"]: r for r in status["rows"]}
    assert rows["study_done"]["state"] == "done" and rows["study_done"]["finished"] == 2
    assert rows["study_next"]["state"] == "queued" and rows["study_next"]["request"] == ids[0]
    assert rows["study_next"]["end_at"] and status["queue_end"]  # from the pace of the finished study
    assert rows["study_next"]["end_basis"].startswith("같은 과제")  # same task: not marked as borrowed
    assert rows["study_new"]["state"] == "new" and status["states"]["done"] == 1
    doc = yaml.safe_load((paths.work_root / "queue" / ids[0] / "experiment.yaml").read_text(encoding="utf-8"))
    assert launch.is_study_doc(doc) and doc["name"] == "study_next"

    page = hpo_html(paths, live=True)
    assert "study_done" in page and "완료" in page and "대기" in page and "미실행" in page
    assert 'data-hpo-submit' in page and 'value="study_new"' in page and "hpo_study_done.txt" in page
    assert 'class="lab-hpo-pick" value="study_next" disabled' in page  # waiting: cannot be queued twice
    assert "data-hpo-submit" not in hpo_html(paths, live=False)  # view-only: no buttons

    assert server.hpo_requeue({"name": "study_done"}) == {"queued": False}  # no enqueue_from: nothing to add
    top = server.hpo_top({"name": "study_done", "k": 2, "seeds": "1"})  # the best settings again with seeds
    queued = [yaml.safe_load((paths.work_root / "queue" / i / "experiment.yaml").read_text(encoding="utf-8"))
              for i in top["ids"]]
    assert all(d["study"] == "hpo_top:study_done" and d["seeds"] == [1] for d in queued)


def test_queue_list_shows_the_expected_end():
    from datetime import datetime, timedelta

    started = (datetime.now().astimezone() - timedelta(minutes=30)).isoformat(timespec="seconds")
    text = launch.remaining_text({"status": "running", "progress": (10, 40), "started": started})
    assert "작업당 평균 3.0분" in text and "약 1.5시간" in text and "(끝 " in text
    assert launch.remaining_text({"status": "queued", "progress": None}) == ""
