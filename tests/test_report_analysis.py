"""Report helpers that need no training: the code map, the flow diagrams and the hyperparameter analysis."""

from __future__ import annotations

import inspect
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ald_lab.models import defaults
from ald_lab.report import archs, design, diagram, player
from ald_lab.report.analysis import (
    config_table,
    effect_summary,
    factor_effects,
    setting_value,
)
from ald_lab.report.codemap import MODULE_CODE, scan
from ald_lab.report.collect import Experiment
from ald_lab.report.plan import PlanError, load_plan

PACKAGE = {
    "__init__.py": "",
    "a.py": '''
"""Entry module."""
from . import b
from .b import Thing, helper


def main():
    helper()
    thing = Thing()
    thing.unique_method()
    b.other()
    items = [1]
    items.append(2)  # list method: never an edge, though Thing has an append too
    run_later(callback)  # a function passed on is a call too
    return thing.size  # a property read is not a call


def callback():
    pass


def run_later(fn):
    return fn
''',
    "b.py": '''
class Thing:
    def __init__(self):
        self.setup()

    def setup(self):
        pass

    def unique_method(self):
        pass

    def append(self, value):
        pass

    @property
    def size(self):
        return 1


def helper():
    pass


def other():
    helper()
''',
}


@pytest.fixture()
def toy(tmp_path: Path):
    root = tmp_path / "toy"
    root.mkdir()
    for name, text in PACKAGE.items():
        (root / name).write_text(text, encoding="utf-8")
    return scan(root)


def test_code_map_resolves_calls(toy):
    calls = toy.functions["a:main"].calls
    assert calls[:5] == ["b:helper", "b:Thing.__init__", "b:Thing.unique_method", "b:other", "a:run_later"]
    assert "a:callback" in calls
    assert "b:Thing.size" not in calls and "b:Thing.append" not in calls
    assert toy.functions["b:Thing.__init__"].calls == ["b:Thing.setup"]
    assert toy.imports["a"] == {"b"} and toy.files["b"] == "b.py"
    assert toy.module_calls()["a"] == {"b"}
    assert f"a:{MODULE_CODE}" not in toy.functions  # module code with no calls is not a step


def test_call_tree_places_each_function_once(toy):
    tree = toy.tree("a:main", depth=3, min_lines=1)
    assert tree.children["a:main"] == ["b:helper", "b:Thing.__init__", "b:Thing.unique_method", "b:other",
                                       "a:run_later", "a:callback"]
    assert tree.children["b:Thing.__init__"] == ["b:Thing.setup"]
    assert ("b:other", "b:helper") in tree.extra  # drawn under main, called again from other
    assert tree.preorder()[:3] == ["a:main", "b:helper", "b:Thing.__init__"]
    small = toy.tree("a:main", depth=3, min_lines=3)  # one- and two-line leaves are left out
    assert "b:helper" not in small.level and small.hidden > 0

    def describe(key: str) -> diagram.Box:
        return diagram.Box(key.split(":")[1], key.split(":")[0] + ".py", "training", key, root=key == "a:main")

    svg = diagram.call_tree(tree, describe, label="toy")
    assert svg.count('class="node') == len(tree.level) and "↪" in svg and 'aria-label="toy"' in svg


def test_grid_graph_draws_back_edges_dashed():
    places = {"cli": (0, 0), "training": (1, 0), "models": (2, 0)}
    edges = {("cli", "training"): 3, ("training", "models"): 8, ("models", "training"): 5}
    svg = diagram.grid_graph(places, edges, lambda g: diagram.Box(g, "", g, g),
                             footer=diagram.Box("공통", "", "common", ""))
    assert svg.count('class="eg"') == 2 and svg.count('class="ex"') == 1
    assert svg.count('class="node') == 4 and ">8<" in svg and ">5<" in svg


def test_this_package_flows():
    cmap = scan()
    assert {"ald-train", "ald-final-test", "ald-predict", "ald-report"} <= set(cmap.entry_points)
    assert all(root in cmap.functions for root in cmap.entry_points.values())
    assert "training.runner:run_experiment" in cmap.functions["cli:train_main"].calls
    assert "training.runner:run_job" in cmap.functions["training.runner:run_experiment"].calls
    assert "training.trainer:train_model" in cmap.functions["training.runner:run_job"].calls
    assert "dashboard.views.home:render" in cmap.functions[f"dashboard.app:{MODULE_CODE}"].calls
    # set.add() / list.append() in data code must not look like calls into the analysis package
    assert "analysis.variability" not in cmap.module_calls().get("data.exsitu", set())
    tree = cmap.tree("cli:train_main")
    assert "training.runner:run_job" in tree.level and len(tree.level) <= 40


def test_design_notes_point_at_existing_code_and_tests():
    """The Code Flow Chart review notes name functions and tests; both must still exist."""

    cmap = scan()
    missing = [k for k in design.function_keys() if k not in cmap.functions]
    assert missing == []
    assert design.missing_tests() == []
    assert {s.key for s in design.STAGES} == {"overview", "prepare", "inputs", "train", "evaluate", "use"}
    assert all(s.steps and s.decisions and s.checks and s.review for s in design.STAGES)
    text, cut = design.excerpt("x = 1\n\ndef f(a):\n    b = a\n    return b\n", 3, 3, limit=2)
    assert text == "def f(a):\n    b = a" and cut == 1


def test_next_experiment_plan_loads(tmp_path):
    plan = load_plan()
    assert plan is not None and plan["principles"]
    ids = [i["id"] for c in plan["categories"] for i in c["items"]]
    assert len(ids) == len(set(ids)) >= 10
    assert all(i["question"] and i["status"] for c in plan["categories"] for i in c["items"])
    bad = tmp_path / "plan.yaml"
    bad.write_text("categories:\n  - key: a\n    title: A\n    items:\n"
                   "      - {id: X1, title: t, priority: 1, status: 계획, question: q}\n"
                   "      - {id: X1, title: t, priority: 1, status: 계획, question: q}\n", encoding="utf-8")
    with pytest.raises(PlanError):
        load_plan(bad)
    assert load_plan(tmp_path / "absent.yaml") is None


def test_deep_defaults_match_the_constructors():
    deep = pytest.importorskip("ald_lab.models.deep")
    backbones = {"patchtst": deep.PatchTSTBackbone, "dlinear": deep.DLinearBackbone,
                 "itransformer": deep.ITransformerBackbone, "gru": deep.GRUBackbone}
    for name, cls in backbones.items():
        found = {k: p.default for k, p in inspect.signature(cls.__init__).parameters.items()
                 if p.default is not inspect.Parameter.empty}
        assert found == defaults.DEEP[name], name


def _experiment(tmp_path: Path, rows: list[dict], jobs: list[dict]) -> Experiment:
    results = pd.DataFrame(rows)
    return Experiment(name="grid", stamp="0", directory=tmp_path, summary={"task": "forecast", "baseline": "persistence"},
                      config={}, table=pd.DataFrame(), results=results, prep=None, test=None, picks=[],
                      package=None, jobs=jobs)


def test_factor_effects_against_seed_spread(tmp_path):
    rows, jobs = [], []
    # d_model matters (0.5 apart), patch length does not (0.01 apart); seeds differ by ±0.02
    for d_model in (32, 64):
        for patch in (8, 16):
            label = f"patchtst_mix[d_model={d_model},patch_seconds={patch}]"
            for seed, noise in ((1, -0.02), (2, 0.02)):
                job_id = f"{label}-s{seed}"
                mae = 2.0 - (0.5 if d_model == 64 else 0) + (0.01 if patch == 16 else 0) + noise
                rows.append({"job_id": job_id, "target": "VG11", "model": "patchtst", "label": label, "seed": seed,
                             "mae": mae, "mae_cycle": mae / 2, "mae_other": mae, "fit_seconds": 60.0, "status": "ok"})
                jobs.append({"job_id": job_id, "variant": {"d_model": d_model, "patch_seconds": patch}})
    rows.append({"job_id": "base", "target": "VG11", "model": "persistence", "label": "persistence", "seed": 1,
                 "mae": 3.0, "status": "ok"})
    jobs.append({"job_id": "base", "variant": {}})
    exp = _experiment(tmp_path, rows, jobs)
    table = config_table(exp)
    assert len(table) == 4 and set(table["seeds"]) == {2} and table["mae_std"].notna().all()
    assert table.iloc[0]["chosen"] and "d_model 64" in table.iloc[0]["setting"]
    effects = {ef["factor"]: ef for ef in factor_effects(exp)}
    assert effects["d_model"]["effect"] is True and effects["d_model"]["best"] == 64
    assert effects["patch_seconds"]["effect"] is False and "불분명" in effects["patch_seconds"]["verdict"]
    assert effects["d_model"]["seed_std"] == pytest.approx(0.02 * 2 ** 0.5, rel=1e-6)
    assert setting_value("d_model", 64.0) == "d_model 64"  # floats from a column with gaps print as ints


def test_single_seed_effects_report_consistency(tmp_path):
    rows, jobs = [], []
    for inputs in ("core", "all"):
        for k in (1, 2):
            job_id = f"pls-{inputs}-{k}"
            rows.append({"job_id": job_id, "target": "thk_mean", "model": "pls", "seed": 42, "status": "ok",
                         "label": f"pls[n_components={k}]@{inputs}", "mae": (0.2 if inputs == "all" else 0.3) + k * 0.01})
            jobs.append({"job_id": job_id, "variant": {"n_components": k}, "input_set": inputs})
    effects = {ef["factor"]: ef for ef in factor_effects(_experiment(tmp_path, rows, jobs))}
    assert effects["input_set"]["seed_std"] is None and effects["input_set"]["consistent"] is True
    assert effects["input_set"]["verdict"] == "반복(seed) 없음, 방향 일관" and effects["input_set"]["best"] == "all"


def test_effects_use_a_fixed_reference(tmp_path):
    """Every model compares input sets from the full set, whatever order its jobs ran in, so one change is one bar."""

    rows, jobs = [], []
    for model, order in (("ridge", ("core", "all")), ("lightgbm", ("all", "core"))):
        for inputs in order:
            for seed, noise in ((1, -0.01), (2, 0.01)):
                job_id = f"{model}-{inputs}-{seed}"
                rows.append({"job_id": job_id, "target": "VG11", "model": model, "label": f"{model}@{inputs}",
                             "seed": seed, "status": "ok", "mae": (1.0 if inputs == "all" else 1.5) + noise})
                jobs.append({"job_id": job_id, "variant": {}, "input_set": inputs})
    exp = _experiment(tmp_path, rows, jobs)
    exp.config = {"data": {"input_sets": {"core": "core", "all": "all"}}}
    effects = factor_effects(exp)
    assert [ef["ref"] for ef in effects] == ["all", "all"]
    assert all(ch[0] == "core" and ch[1] == pytest.approx(-50.0, abs=2) and ch[2] is True
               for ef in effects for ch in ef["changes"])
    summary = effect_summary(effects)
    assert len(summary) == 1 and summary[0]["n"] == 2 and summary[0]["status"] == "clear"
    assert summary[0]["ref"] == "all" and summary[0]["level"] == "core"


def test_job_timeline_and_speed_table(tmp_path):
    from ald_lab.report.analysis import job_timeline, speed_table
    from ald_lab.report.charts import timeline_chart

    exp = _experiment(tmp_path, [
        {"job_id": "a", "target": "VG11", "model": "patchtst", "label": "patchtst_mix[d_model=64]", "seed": 1,
         "status": "ok", "mae": 1.0, "windows_per_second": 50_000.0, "fit_seconds": 120.0, "peak_gpu_mb": 900.0,
         "n_params": 1e5},
        {"job_id": "b", "target": "VG11", "model": "patchtst", "label": "patchtst_mix[d_model=128]", "seed": 1,
         "status": "ok", "mae": 1.0, "windows_per_second": 30_000.0, "fit_seconds": 240.0, "peak_gpu_mb": 1500.0,
         "n_params": 3e5},
        {"job_id": "c", "target": "VG11", "model": "lightgbm", "label": "lightgbm", "seed": 1, "status": "ok",
         "mae": 0.5, "windows_per_second": None, "fit_seconds": 60.0}], [])
    speeds = speed_table([exp])
    assert list(speeds["spec"]) == ["patchtst_mix"]
    assert speeds.iloc[0]["wps"] == 40_000 and speeds.iloc[0]["minutes"] == 3 and speeds.iloc[0]["peak_mb"] == 1500

    start = pd.Timestamp("2026-10-27T00:00:00Z")
    journal = pd.DataFrame([
        {"stage": stage, "title": title, "started_at": start + pd.Timedelta(seconds=t), "duration_s": d, "status": "ok",
         "config": {"device": dev, "model": "patchtst", "label": "x"},
         "artifacts": {"job_dir": f"/w/experiments/grid/{stamp}/jobs/{title}"}}
        for stage, title, t, d, dev, stamp in [("train", "j1", 0, 60, "cuda:0", "0"), ("hpo", "j2", 10, 60, "cuda:1", "0"),
                                               ("train", "j3", 0, 30, "cuda:0", "other-run"),
                                               ("evaluate", "j4", 0, 30, "cuda:0", "0")]])
    timeline = job_timeline(exp, journal)
    assert list(timeline["device"]) == ["cuda:0", "cuda:1"] and list(timeline["start_s"]) == [0, 10]
    assert timeline["end_s"].max() == 70
    svg = timeline_chart([(r.device, r.start_s / 60, r.end_s / 60, "var(--s1)", r.title) for r in timeline.itertuples()],
                         [("cuda:0", "GPU 0"), ("cuda:1", "GPU 1")])
    assert svg.count('class="tl"') == 2 and "GPU 1" in svg


def test_in_tab_figures():
    flow = diagram.count_flow([("run", [("80", "번호"), ("74", "파일"), ("51 / 11 / 11", "학습 / 검증 / 시험")],
                                [("−6 파일 없음", "041·056"), ("run 단위로 나눔", "")]),
                               ("행", [("2,438,733", "원본"), ("1,191,731", "1초 행")], [("0.5초 2개 → 1초", "")])],
                              label="flow")
    assert flow.count('class="nb nb-final"') == 2 and flow.count('class="nb"') == 3 and "−6 파일 없음" in flow
    strip = diagram.split_strip([(1, "train", True), (2, "train", False), (3, "missing", False), (4, "validation", False),
                                 (5, "excluded", False)], per_row=3, label="strip")
    assert strip.count('class="cell-train"') == 2 and strip.count('class="cflag"') == 1
    assert "<title>run 003: 파일 없음</title>" in strip and 'class="cnum-muted"' in strip
    rule = diagram.window_rule(192, 30, label="rule")
    assert "win-ok" in rule and "win-bad" in rule and "최근 192초" in rule and "30초 뒤" in rule


def test_bundle_stats_use_whole_partitions(tmp_path):
    from ald_lab.paths import resolve_paths
    from ald_lab.report.dataview import bundle_stats

    paths = resolve_paths(work_root=tmp_path / "work")
    for name, runs, rows in (("full", ["a", "b"], 100), ("smoke", ["a"], 10), ("val", ["c"], 40)):
        folder = paths.work_root / "prepared" / name
        folder.mkdir(parents=True)
        part = "validation" if name == "val" else "train"
        (folder / "meta.json").write_text(json.dumps({"partition": part, "run_ids": runs, "sample_seconds": 1.0,
                                                      "stats": {"rows": rows, "segments": len(runs)}}))
    stats = bundle_stats(paths, {"train": ["a", "b"], "validation": ["c"], "test": ["d"]})
    assert stats["train"]["rows"] == 100 and stats["validation"]["rows"] == 40 and "test" not in stats
    assert bundle_stats(paths, None) == {}


def test_example_run_goes_through_the_training_preprocessing(trace_dir, tmp_path):
    from ald_lab.paths import resolve_paths
    from ald_lab.report.dataview import example_run

    paths = resolve_paths(data_root=trace_dir, work_root=tmp_path / "work")
    ex = example_run(paths, "standard_trace_004")  # 4 s gap in the standby before the process
    assert ex["segments"] == 2 and ex["rows_1s"] <= ex["raw_rows"] // 2 and ex["nan_filled"] == 0
    assert 0 < ex["cycle_rows"] < ex["process_rows"] < ex["rows_1s"] and len(ex["cycle"]) == 1
    assert ex["zoom_steps"][0][1] != "SI-FL1" and any(name == "SI-FL1" for _, name in ex["zoom_steps"])
    assert abs(len(ex["zoom_raw_t"]) - 2 * len(ex["zoom_t"])) <= 2 and max(ex["zoom_y"]) <= max(ex["zoom_raw_y"]) + 1e-9
    assert example_run(paths, "standard_trace_099") is None and example_run(paths, "standard_trace_004", "nope") is None


def test_architecture_figures_are_well_formed():
    figures = [archs.classic_figure(), archs.vm_figure(), archs.patchtst_figure(), archs.channel_modes_figure(),
               archs.itransformer_figure(), archs.dlinear_figure(), archs.gru_figure(), archs.wrapper_figure(),
               *(archs.task_figure(t, context=192, horizons=[10, 20, 30]) for t in ("forecast", "virtual_sensor",
                                                                                     "thickness"))]
    labels = []
    for fig in figures:
        svg = ET.fromstring(fig).find("svg")  # each one parses as XML
        labels.append(svg.get("aria-label"))
    assert len(set(labels)) == len(labels) and "PatchTST 구조" in labels
    d = defaults.DEEP["patchtst"]
    assert f"Linear {d['patch_len']}→{d['d_model']}" in figures[2] and f"×{d['n_layers']}" in figures[2]


def test_player_series_from_samples_and_predictions():
    t = np.arange(0.0, 900.0)
    sample = pd.DataFrame({"run_id": "standard_trace_004", "t_s": t, "in_cycle": t > 100,
                           "true_10s": np.sin(t / 7) + 753.123456, "pred_10s": np.sin(t / 7) + 753.1,
                           "true_20s": np.cos(t / 9), "pred_20s": np.nan})
    other = sample.assign(pred_10s=sample["pred_10s"] + 1)
    shifted = sample.assign(t_s=sample["t_s"] + 5)  # another run's windows: left out
    jobs = [{"name": n, "label": n, "frame": f.iloc[::3].reset_index(drop=True), "zoom": f.iloc[200:500]}
            for n, f in (("lightgbm", sample), ("patchtst", other), ("gru", shifted))]
    item = player.sample_series("x/VG11", group="g", title="VG11", target="VG11", unit="Torr", context=64, samples=jobs)
    assert item["h"] == [10.0, 20.0] and [v["name"] for v in item["views"]] == ["ALD 사이클 10분 확대", "run 전체"]
    assert [m["name"] for m in item["models"]] == ["lightgbm", "patchtst"] and item["run"] == "standard_trace_004"
    zoom, first, second = item["views"][0], item["models"][0]["p"][0], item["models"][1]["p"][0]
    assert len(zoom["t"]) == len(first[0]) == len(zoom["y"][1]) == len(zoom["cyc"]) == 300
    assert zoom["y"][0][0] == float(f"{np.sin(200 / 7) + 753.123456:.5g}") and first[1][0] is None  # NaN → null
    assert second[0][0] == pytest.approx(first[0][0] + 1, abs=1e-3)
    html = player.player([item])
    data = json.loads(html.split('id="pl-data">')[1].split("</script>")[0])
    assert data[0]["h"] == [10.0, 20.0] and data[0]["ctx"] == 64.0 and "y" not in data[0]["models"][0]
    stamps = pd.date_range("2026-10-01", periods=50, freq="s")
    frame = pd.DataFrame({"run_id": ["a"] * 25 + ["b"] * 25, "timestamp": stamps, "step_name": "CYCLE",
                          "in_cycle": True, "actual_+10s": 1.0, "pred_+10s": 1.1})
    frame.attrs.update(target="VG11", unit="Torr")
    items = player.prediction_series(frame, model="LightGBM", context=64)
    assert [s["run"] for s in items] == ["a", "b"] and items[1]["views"][0]["t"][0] == 0.0
    assert items[0]["h"] == [10.0] and items[0]["views"][0]["steps"] == [[0, "CYCLE"]]
    assert items[0]["models"][0]["p"][0][0][0] == 1.1
    assert player.series("k", group="g", title="t", target="", unit="", run="", context=1, horizons=[], views=[None],
                         models=[{"name": "m", "label": "m", "p": [[[1.0]]]}]) is None
