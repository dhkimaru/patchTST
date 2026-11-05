"""Model packages, predictions on new CSVs and the offline HTML report (synthetic data, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ald_lab.data.overview import build_data_overview  # noqa: E402
from ald_lab.journal import Journal  # noqa: E402
from ald_lab.report import render  # noqa: E402
from ald_lab.report.collect import collect  # noqa: E402
from ald_lab.report.render import TABS, write_report  # noqa: E402
from ald_lab.training.export import export_models  # noqa: E402
from ald_lab.training.predict import PredictError, predict_traces  # noqa: E402
from ald_lab.training.runner import run_experiment, run_final_test  # noqa: E402
from ald_lab.training.windows import COPY_THRESHOLD  # noqa: E402

from .test_training import _config, lab  # noqa: E402,F401  (fixture)

VALIDATION = ["standard_trace_004", "standard_trace_005"]


def _validation_files(paths) -> list[Path]:
    return [paths.data_root / f"{r}.csv" for r in VALIDATION]


def test_package_predictions_match_validation(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    config = _config(tmp_path, split_path, task="virtual_sensor", name="vspkg",
                     models=["ridge", {"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}}],
                     data={"split": str(split_path), "sample_seconds": 1.0, "region": "process"})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    package = export_models(exp_dir, paths, scope="per_model")
    index = json.loads((package / "package.json").read_text(encoding="utf-8"))
    assert {m["model"] for m in index["models"]} == {"ridge", "dlinear"}
    assert (package.parent / f"{package.name}.zip").is_file()
    for model in index["models"]:
        folder = package / model["folder"]
        assert {"job.json", "README.txt", "predict.sh", "predict.bat", "inputs.txt", "template.parquet"} <= {
            p.name for p in folder.iterdir()}
        assert (folder / "predict.sh").stat().st_mode & 0o111
        assert b"\r\n" in (folder / "predict.bat").read_bytes()
        (folder / "predict.bat").read_bytes().decode("ascii")
        frame = predict_traces(folder, _validation_files(paths))
        validation = json.loads((folder / "metrics_validation.json").read_text())["metrics"]
        mae = float((frame["pred"] - frame["actual"]).abs().mean())
        assert mae == pytest.approx(validation["mae"], rel=1e-4)
        assert len(frame) == validation["n_windows"]
        assert {"run_id", "timestamp", "step_name", "in_cycle"} <= set(frame.columns)


def test_virtual_sensor_predicts_without_its_sensor(lab, tmp_path):  # noqa: F811
    """The point of a virtual sensor: the CSV may lack the sensor (and its copies) entirely."""

    paths, split_path = lab
    config = _config(tmp_path, split_path, task="virtual_sensor", name="vsabsent", models=["ridge"],
                     data={"split": str(split_path), "sample_seconds": 1.0, "region": "process"})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    folder = next(p for p in export_models(exp_dir, paths, make_zip=False).iterdir() if p.is_dir())
    csv = tmp_path / "no_vg11" / "standard_trace_004.csv"
    csv.parent.mkdir()
    pd.read_csv(paths.data_root / "standard_trace_004.csv").drop(columns=["VG11", "PressAct", "AUXMon_VG11"]).to_csv(
        csv, index=False)
    frame = predict_traces(folder, [csv])
    assert "actual" not in frame and frame["pred"].notna().all() and len(frame) > 0
    broken = tmp_path / "no_temp.csv"
    pd.read_csv(paths.data_root / "standard_trace_004.csv").drop(columns=["TempAct_C"]).to_csv(broken, index=False)
    with pytest.raises(PredictError):
        predict_traces(folder, [broken])  # a model input is missing


def test_forecast_package_matches_validation(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    config = _config(tmp_path, split_path, name="fcpkg", models=[{"name": "lightgbm", "params": {"n_estimators": 20}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    folder = next(p for p in export_models(exp_dir, paths, make_zip=False).iterdir() if p.is_dir())
    frame = predict_traces(folder, _validation_files(paths))
    validation = json.loads((folder / "metrics_validation.json").read_text())["metrics"]
    for h in ("2", "4"):
        mae = float((frame[f"pred_+{h}s"] - frame[f"actual_+{h}s"]).abs().mean())
        assert mae == pytest.approx(validation["by_horizon"][f"{h}s"], rel=1e-4)
    # ald-predict --html: the same player, one series per run
    from ald_lab.training.predict import main as predict_main

    out = tmp_path / "pred.csv"
    assert predict_main([str(folder), "--csv", str(paths.data_root), "--out", str(out), "--html"]) == 0
    page = (tmp_path / "pred_player.html").read_text(encoding="utf-8")
    items = json.loads(page.split('id="pl-data">')[1].split("</script>")[0])
    assert len(items) == len(list(paths.data_root.glob("standard_trace_*.csv"))) and items[0]["h"] == [2.0, 4.0]
    assert "VG11 추론 재생 (LightGBM)" in page and "<script src" not in page and items[0]["views"][0]["y"]
    assert items[0]["models"][0]["p"][0][0]


def test_thickness_package(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    rng = np.random.default_rng(0)
    runs = [f"standard_trace_00{i}" for i in range(1, 7)]
    paths.overview_dir.mkdir(parents=True, exist_ok=True)
    # 두께는 실제와 무관한 임의 값 (공개본)
    pd.DataFrame({"run_id": runs, "layout": "49pt", "thk_mean": 100.0 + rng.normal(0, 0.3, len(runs))}).to_parquet(
        paths.overview_dir / "exsitu_blocks.parquet")
    config = _config(tmp_path, split_path, task="thickness", name="vmpkg", targets=["thk_mean"],
                     models=["ridge"], vm={"exclude_runs": []})
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    folder = next(p for p in export_models(exp_dir, paths, make_zip=False).iterdir() if p.is_dir())
    assert (folder / "estimator.joblib").is_file()
    frame = predict_traces(folder, _validation_files(paths))
    assert list(frame["run_id"]) == VALIDATION and frame["pred_thk_mean"].between(95, 105).all()


def test_report_is_self_contained(lab, tmp_path):  # noqa: F811
    paths, split_path = lab
    config = _config(tmp_path, split_path, name="rpt", models=[
        "ridge", {"name": "dlinear", "params": {"kernel": 5, "d_rep": 8}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    export_models(exp_dir, paths)
    build_data_overview(paths, workers=1)
    data = collect(paths, notes=["오늘 메모 <b>escaped</b>"])
    assert [x.name for x in data.experiments] == ["rpt"] and len(data.histories) >= 2
    out = write_report(data, tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    for key, _ in TABS:
        assert f'id="{key}"' in html
    assert "<script src" not in html and "<link" not in html and "http://" not in html.replace("http://서버", "")
    assert "VG11" in html and "만들어짐" in html and "&lt;b&gt;escaped&lt;/b&gt;" in html
    summary = html.split('<section id="summary"')[1].split("</section>")[0]  # 요약: conclusion first, tables folded
    assert "현재 결론" in summary and "미학습 1개" in summary and "<details><summary>과제별 최고 성능" in summary
    assert "S2 (=FA1)" in html  # previous experiments tab
    for text in ("단계별 데이터 수", "제외한 데이터", "제외한 컬럼", "run 단위 분할", 'aria-label="전처리 순서"',
                 'aria-label="window 규칙"', 'aria-label="run 번호별 분할"', "품질 점검 결과", "정규화 - 학습 run 기준"):
        assert text in html, text
    assert (out.parent / "ald_report_latest.html").read_text(encoding="utf-8") == html
    assert len(Journal(paths.journal_dir).entries()) > 0
    # Function tab: models by category, architecture figures, where each model won (no blank-heavy table)
    for text in ("모델 구성", "과제별 학습 대상", "모델별 선정 현황", "① 기준선", "② 고전 ML", "③ 딥러닝",
                 'aria-label="PatchTST 구조"', 'aria-label="iTransformer 구조"', 'aria-label="DLinear 구조"',
                 'aria-label="GRU 구조"', 'aria-label="딥러닝 공통 틀: RevIN과 출력 head"',
                 'aria-label="미래값 예측: 최근 구간으로 이후 값 예측"', "추론 시연", 'id="pl-data"'):
        assert text in html, text
    assert "대상별로 고른 모델의 구조" not in html
    items = json.loads(html.split('id="pl-data">')[1].split("</script>")[0])
    assert {i["key"] for i in items} == {"rpt/VG11"} and items[0]["h"] == [2.0, 4.0]
    assert {m["name"] for m in items[0]["models"]} == {"ridge", "dlinear"}  # every trained model, not only the 1st
    assert items[0]["models"][0]["label"].startswith("1위")
    for k, v in enumerate(items[0]["views"]):
        assert len(v["t"]) == len(items[0]["models"][1]["p"][k][0]) == len(v["y"][1]) > 0
    smoke_free = collect(paths, include_smoke=False, examples=False)
    assert all(p.example is None for x in smoke_free.experiments for p in x.picks)
    bare = render.render(smoke_free, examples=False)
    assert 'id="pl-data"' not in bare and 'data-tab="demo"' not in bare  # nothing to replay: no empty tab
    public = render.render(smoke_free, examples=False, public=True)  # the view-only dashboard
    assert "보기 전용" in public and str(paths.work_root) not in public and "코드 " not in public.split('<div class="content">')[0]
    side = bare.split('<div class="content">')[0]  # left sidebar: tabs grouped, every tab still reachable
    assert 'class="side"' in side and ">데이터<" in side and ">하이퍼파라미터<" in side
    assert side.count("data-tab=") == bare.count("<section id=")


def test_history_topics_use_final_test(lab, tmp_path):  # noqa: F811
    """The previous-experiments tab compares against this system's VG11 test-run numbers when they exist."""

    paths, split_path = lab
    config = _config(tmp_path, split_path, name="g1_forecast_vg11", models=[
        {"name": "lightgbm", "params": {"n_estimators": 20}},
        {"name": "patchtst", "label": "patchtst_mix", "params": {"channel_mode": "mixing", "patch_seconds": 4,
                                                                 "stride_seconds": 2, "d_model": 16, "n_heads": 2,
                                                                 "n_layers": 1, "d_ff": 32}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    run_final_test(exp_dir, paths, device="cpu")
    params = _config(tmp_path, split_path, name="g1_forecast_params", targets=["VG11", "TempAct_C"], data={
        "split": str(split_path), "sample_seconds": 1.0, "region": "all", "inputs": "all"},
        models=[{"name": "lightgbm", "params": {"n_estimators": 20}}])
    run_final_test(run_experiment(params, paths, devices=["cpu"]), paths, device="cpu")
    html = write_report(collect(paths), tmp_path / "topics.html").read_text(encoding="utf-8")
    assert "주제별 비교 (이전 · 이번)" in html and "(시험 run)" in html and "10초" in html
    assert "<h3>전체 흐름</h3>" in html and 'aria-label="VG11 예측 오차의 변화"' in html and "이번 시스템" in html
    assert "(시험 run, 예측 거리 2·4초)" in html and "재확인 예정" not in html  # same horizons: test numbers side by side
    assert "초기 작업" in html and "전체 테스트 목록" in html and "A07" in html


def test_report_hyperparameter_and_code_tabs(lab, tmp_path):  # noqa: F811
    """A small grid with two seeds fills every Hyper Parameter section; the Code Flow Chart reads the source."""

    paths, split_path = lab
    config = _config(tmp_path, split_path, name="hpgrid", seeds=[1, 2], models=[
        "ridge", {"name": "dlinear", "params": {"d_rep": 8}, "grid": {"kernel": [3, 5]}}])
    exp_dir = run_experiment(config, paths, devices=["cpu"])
    run_final_test(exp_dir, paths, device="cpu")
    package = export_models(exp_dir, paths, make_zip=False)
    readme = next(package.rglob("README.txt")).read_text(encoding="utf-8")
    html = write_report(collect(paths), tmp_path / "hp.html").read_text(encoding="utf-8")
    for text in ("결론 (영향이 큰 설정)", "탐색하지 않은 것", "모델별 탐색 여부", "설정별 결과", "설정 하나를 바꾼 효과",
                 "kernel 3", "학습 곡선", "최고 epoch", "이전 실험 (2026-08)", "N03", "① 보는 법", "실험 코드별 변경 내용",
                 "RevIN만 끄기", "기준 대비 변화",
                 "<th>2초</th>", "<th>4초</th>", "(2초·4초)",
                 'aria-label="ald-train 함수 호출 흐름"', "명령 간 공통 코드", "패키지 간 호출 관계",
                 "window와 patch를 GPU에서 자르기", "GPU 여러 장을 동시에 쓰기", 'aria-label="hpgrid 작업 타임라인"',
                 "모델별 처리 속도", "prepared_cache.py",
                 "② 단계별 상세 설계", 'id="design-prepare"', "데이터 누설 · 재현성 점검표", "검토 포인트",
                 "def run_arrays", "<b>이번 데이터</b>:", "다음 실험", 'id="plan-A1"', "먼저 할 것",
                 '<html lang="ko" data-theme="dark">'):
        assert text in html, text
    for banned in ("\uc13c\ud130\uc7a5", "반출"):  # a job title kept out of the report, the old weights note
        assert banned not in html and banned not in readme


def test_report_copy_threshold_matches_training():
    assert render.COPY_THRESHOLD == COPY_THRESHOLD  # the report states the threshold without importing torch
