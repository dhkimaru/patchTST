"""The lab server: the live report, the plan/submit API of the 하이퍼파라미터 탐색 tab, steps 2 and 3 (synthetic data)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from ald_lab import launch
from ald_lab.server import Lab, make_handler

from .test_training import _config, lab  # noqa: F401  (fixture)


@pytest.fixture
def bases(lab, tmp_path, monkeypatch):  # noqa: F811
    """configs/experiments/ replaced by synthetic experiments (a forecast search and its step-2 base)."""

    _, split_path = lab
    folder = tmp_path / "experiments_cfg"
    folder.mkdir()
    base = yaml.safe_load(_config(tmp_path, split_path).read_text(encoding="utf-8"))
    params = {**base, "name": "g1_forecast_params", "models": ["persistence", "ridge"]}
    search = {**base, "name": "p1_patchtst_search_forecast", "models": ["persistence", {
        "name": "patchtst", "search": {"trials": 2, "seed": 0},
        "params": {"stride_seconds": 2, "n_layers": 1, "d_ff": 16},
        "grid": {"d_model": [8, 16], "n_heads": [2], "patch_seconds": [4, 8]}}]}
    for doc in (params, search):
        (folder / f"{doc['name']}.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(launch, "EXPERIMENTS_DIR", folder)
    return folder


def _http(port: int, path: str, body: dict | None = None, *, header: bool = True):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None)
    if header:
        request.add_header("X-Requested-With", "ald-lab")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if response.headers.get_content_type() == "application/json" else text


def test_lab_server_end_to_end(lab, bases, tmp_path):  # noqa: F811
    paths, _ = lab
    server = Lab(paths, examples=True)
    meta = server.meta()
    assert {b["name"] for b in meta["bases"]} == {"g1_forecast_params", "p1_patchtst_search_forecast"}
    assert meta["default_base"] == "g1_forecast_params" and "patchtst" in meta["space"]["forecast"]
    assert meta["run_enabled"] and meta["explore"]["forecast"]["targets"] and meta["full_seeds"] == [42, 43, 44]
    form = {"base": "g1_forecast_params", "targets": ["VG11"], "seeds": "1", "windows": "16, 24", "input_sets": ["all"],
            "models": {"lightgbm": {"num_leaves": "7, 15", "n_estimators": "20"}}, "search_trials": 0,
            "train": {"max_epochs": "2"}, "gpus": "cpu", "memo": "잎 수"}
    plan = server.plan(form)
    assert plan["experiments"] == 2 and plan["jobs"] == 2 * (1 + 2) and plan["combos"] == 2
    assert "context_seconds: 24.0" in plan["yaml"] and "잎 수 · window 16초" in plan["yaml"]
    assert plan["warnings"] == [] and plan["estimate_basis"]
    mixed = server.plan({**form, "models": {"patchtst": {"channel_mode": "mixing, independent", "d_model": "16", "n_heads": "2"}},
                         "windows": "16"})
    assert any("independent" in w for w in mixed["warnings"])
    with pytest.raises(launch.LaunchError):
        server.plan({**form, "models": {"lightgbm": {"num_leaves": "many"}}})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert "보고서를 만드는 중" in _http(port, "/")  # before the first build
        with pytest.raises(urllib.error.HTTPError):
            _http(port, "/api/submit", form, header=False)  # no header: refused
        ids = _http(port, "/api/submit", {**form, "windows": "16"})["ids"]
        assert len(ids) == 1 and "대기" in _http(port, "/api/fragment/queue")
        assert launch.run_worker(paths.work_root) == 1
        assert [r["status"] for r in launch.list_requests(paths.work_root)] == ["done"]
        assert "완료" in _http(port, "/api/fragment/queue") and _http(port, "/api/log?id=" + ids[0])

        # step 1 search, then the step-2 button carries its best combination to every target of the base
        search = launch.base_configs()[1]["doc"]
        assert search["name"] == "p1_patchtst_search_forecast"
        launch.submit(paths.work_root, search, gpus="cpu", data_root=paths.data_root, n_jobs=None)
        assert launch.run_worker(paths.work_root) == 1
        first = _http(port, "/api/fragment/first")
        assert "② 모든 대상에 적용" in first and "기준선" in first
        explore = _http(port, "/api/fragment/explore")
        assert "p1_patchtst_search" in explore and "조합" in explore
        stage = _http(port, "/api/stage2", {"task": "forecast", "k": 1, "seeds": "1", "gpus": "cpu", "report": False})
        assert len(stage["ids"]) == 1, stage
        doc = yaml.safe_load((Path(paths.work_root) / "queue" / stage["ids"][0] / "experiment.yaml").read_text())
        assert doc["name"] == "p2_patchtst_all_forecast" and doc["models"][0]["label"] == "patchtst_top1"
        assert "error" in _http(port, "/api/stage3", {"task": "forecast", "gpus": "cpu"})  # nothing to fall back yet
        assert _http(port, "/api/cancel", {"id": stage["ids"][0]})["status"] == "cancelled"

        server.rebuild()
        page = _http(port, "/")
        assert "하이퍼파라미터 탐색" in page and 'id="lab-live"' in page and "모델 선정 단계" in page
        assert "<script src" not in page
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_view_only_server_refuses_runs(lab, bases, monkeypatch):  # noqa: F811
    paths, _ = lab
    monkeypatch.setenv("ALD_DASHBOARD_RUN", "0")
    server = Lab(paths, examples=False)
    assert server.meta()["run_enabled"] is False
    launch.submit(paths.work_root, launch.base_configs()[0]["doc"], gpus="cpu", data_root=paths.data_root, n_jobs=None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert "data-cancel" not in _http(port, "/api/fragment/queue")
        answer = _http(port, "/api/submit", {"base": "g1_forecast_params", "targets": ["VG11"], "gpus": "cpu"})
        assert "실행이 꺼져" in answer["error"]
        assert len(launch.list_requests(paths.work_root)) == 1  # nothing added
        server.rebuild()
        page = _http(port, "/")
        assert "보기 전용" in page and str(paths.work_root) not in page
    finally:
        httpd.shutdown()
        httpd.server_close()


PLAN_FILE = """
## 메모: 결과 표는 건너뛴다

| 발견 | 크기 |
|---|---|
| mae | −16 % |

### 1차 잎 수

| 순서 | 칸 | 입력 |
|---|---|---|
| 1 | 기준 실험 | g1_forecast_params |
| 2 | 대상 | VG11 하나만 켬 |
| 3 | 모델 | LightGBM만 |
| 4 | 잎 수 | 7, 15 |
| 5 | 나무 수 상한 | 20 |
| 6 | window 길이(초) | 16 |
| 7 | seed | 1 |
| 8 | 메모 | 1차 잎 수 |

### 2차 window

- 기준 실험: g1_forecast_params
- 대상: VG11 하나만 켬
- 모델: LightGBM만
- 나무 수 상한: 20
- window 길이(초): 16, 24
- 메모: 2차 window
"""


def test_plan_file_queues_every_request_and_finished_ones_leave_the_list(lab, bases, tmp_path):  # noqa: F811
    paths, _ = lab
    server = Lab(paths, examples=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(server))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        preview = _http(port, "/api/plan-file", {"text": PLAN_FILE, "gpus": "cpu"})
        assert [i["memo"] for i in preview["items"]] == ["1차 잎 수", "2차 window"] and all(i["ok"] for i in preview["items"])
        assert [i["experiments"] for i in preview["items"]] == [1, 2] and preview["jobs"] == (1 + 2) + 2 * (1 + 1)
        bad = _http(port, "/api/submit-file", {"text": PLAN_FILE.replace("VG11 하나만", "없는대상 하나만"), "gpus": "cpu"})
        assert "대기열에 넣지 않았습니다" in bad["error"] and not launch.list_requests(paths.work_root)
        done = _http(port, "/api/submit-file", {"text": PLAN_FILE, "gpus": "cpu"})
        assert done["requests"] == 2 and len(done["ids"]) == 3  # 1차 one experiment, 2차 one per window
        queued = launch.list_requests(paths.work_root)
        assert [r["memo"] for r in queued[::-1]] == ["1차 잎 수", "2차 window · window 16초", "2차 window · window 24초"]
        refused = _http(port, "/api/hide", {"id": done["ids"][0]})
        assert "먼저 취소" in refused["error"]  # still queued
        assert launch.run_worker(paths.work_root) == 3
        page = _http(port, "/api/fragment/queue")
        assert page.count("data-hide=") == 3 and 'data-hide-all="3"' in page
        assert _http(port, "/api/hide", {"id": done["ids"][0]})["status"] == "hidden"
        assert len(launch.list_requests(paths.work_root)) == 2 and len(launch.list_requests(paths.work_root, include_hidden=True)) == 3
        assert _http(port, "/api/hide-finished", {})["hidden"] == 2 and not launch.list_requests(paths.work_root)
        assert "1차 잎 수" in server.ledger()["memo"].tolist()  # results stay in the records
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_explore_results_are_a_list_to_pick_from(lab, bases):  # noqa: F811
    from ald_lab.report.lab import explore_html

    paths, _ = lab
    for memo in ("1차", "2차"):
        form = {"base": "g1_forecast_params", "targets": ["VG11"], "seeds": "1", "windows": "16", "input_sets": ["all"],
                "models": {"patchtst": {"d_model": "8", "n_heads": "2", "patch_seconds": "4, 8", "stride_seconds": "4",
                                        "n_layers": "1", "d_ff": "16"}},
                "train": {"max_epochs": "1"}, "gpus": "cpu", "memo": memo}
        launch.submit_docs(paths.work_root, launch.docs_from_form(form), gpus="cpu", report=False, data_root=paths.data_root,
                           source="dashboard")
    assert launch.run_worker(paths.work_root) == 2
    page = explore_html(Lab(paths, examples=False).ledger())
    rows, panels = page.count("<tr data-xsel="), page.count('class="lab-xpanel')
    assert rows == panels == 2 and page.count(" hidden ") + page.count(" hidden>") >= 1  # one panel shown at a time
    assert page.index("<b>2차</b>") < page.index("<b>1차</b>")  # newest first
