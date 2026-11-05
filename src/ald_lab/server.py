"""The lab server: the HTML report, live, with the 하이퍼파라미터 탐색 tab able to start experiments.

``ald-lab-server`` (``scripts/lab.sh``) serves the same page ``ald-report`` writes, rebuilt in the background
whenever an experiment finishes, plus a small JSON API the tab uses: plan a request (job count, time estimate,
YAML), queue it, follow the queue, cancel, and start PatchTST-first steps 2 and 3. Training itself runs in the
queue worker (:mod:`ald_lab.launch`), never in this process. Standard library only.

It listens on 127.0.0.1 by default (reach it through an SSH tunnel); ``ALD_LAB_HOST=0.0.0.0`` opens it to the
network. Requests that change something need the ``X-Requested-With: ald-lab`` header, which a page from another
site cannot send without permission.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from . import launch
from .paths import LabPaths, resolve_paths

REBUILD_CHECK_SECONDS = 15
TASK_NAMES = {"forecast": "미래값 예측", "virtual_sensor": "가상 센서", "thickness": "두께 예측"}
LOADING = ("<!doctype html><html lang='ko'><meta charset='utf-8'><meta http-equiv='refresh' content='5'>"
           "<body style='font:15px sans-serif;padding:40px;background:#111;color:#eee'>보고서를 만드는 중입니다… "
           "(실험이 많으면 1분 정도 걸립니다. 5초마다 새로 고칩니다)</body></html>")


class Lab:
    """Report cache, cumulative records cache and the API behind the handler."""

    def __init__(self, paths: LabPaths, *, examples: bool = True) -> None:
        self.paths = paths
        self.examples = examples
        self.html: str | None = None
        self.built_at = 0.0
        self.build_error = ""
        self._signature: float | None = None
        self._ledger: tuple[float, Any] | None = None
        self._lock = threading.Lock()
        self._building = False

    # -- report ---------------------------------------------------------------------------------------------
    def signature(self) -> float:
        root = self.paths.work_root
        patterns = [("experiments", "*/*/summary.json"), ("experiments", "*/*/test_summary.csv"),
                    ("packages", "*/package.json"), ("data_overview", "*.json")]
        times = [p.stat().st_mtime for sub, pattern in patterns for p in (root / sub).glob(pattern)]
        config = Path(launch.PROJECT_ROOT) / "configs"
        times += [p.stat().st_mtime for p in config.glob("*/*.yaml")]
        return max(times, default=0.0)

    def rebuild(self) -> None:
        with self._lock:
            if self._building:
                return
            self._building = True
        try:
            from .report.collect import collect
            from .report.render import render

            signature = self.signature()
            data = collect(self.paths, examples=self.examples)
            self.html = render(data, examples=self.examples, public=not launch_enabled())
            self.built_at, self._signature, self.build_error = time.time(), signature, ""
        except Exception as exc:  # noqa: BLE001 — keep serving the last good page
            self.build_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._building = False

    def watch(self) -> None:
        while True:
            try:
                if self._signature is None or self.signature() != self._signature:
                    self.rebuild()
            except OSError as exc:  # a file vanishing mid-scan must not stop the watcher
                print(f"보고서 확인 건너뜀: {exc}", flush=True)
            time.sleep(REBUILD_CHECK_SECONDS)

    def ledger(self):
        from .ledger import collect_ledger

        root = self.paths.work_root / "experiments"
        stamp = max([p.stat().st_mtime for p in root.glob("*/*/jobs/*")] + [p.stat().st_mtime for p in root.glob("*/*/*.csv")]
                    + [0.0])
        if self._ledger is None or self._ledger[0] != stamp:
            self._ledger = (stamp, collect_ledger(self.paths.work_root))
        return self._ledger[1]

    # -- API ------------------------------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        from .stages import EXPLORE_SEEDS, EXPLORE_TRAIN, FULL_SEEDS, REPRESENTATIVE

        bases = []
        for b in launch.base_configs():
            doc = b["doc"]
            names = [m["name"] if isinstance(m, dict) else m for m in doc.get("models") or []]
            bases.append({
                "name": b["name"], "task": b["task"], "description": b["description"], "targets": list(doc["targets"]),
                "models": names, "input_sets": [s for s in launch.base_input_sets(doc) if s in launch.INPUT_SET_NAMES],
                "window": (doc.get("window") or {}).get("context_seconds") if b["task"] != "thickness" else None,
                "seeds": list(doc.get("seeds") or [42]), "train": dict(doc.get("train") or {}),
                "defaults": {m: launch.base_model_values(doc, m) for m in launch.MODELS_BY_TASK[b["task"]]},
            })
        from .models.cards import model_card

        space = {task: {m: [{"key": p.key, "label": p.label, "kind": p.kind, "default": p.default,
                             "choices": [str(c) for c in p.choices], "help": p.help}
                            for p in launch.param_space(task, m)] for m in models}
                 for task, models in launch.MODELS_BY_TASK.items()}
        names = [b["name"] for b in bases]
        return {
            "bases": bases, "space": space, "models_by_task": {k: list(v) for k, v in launch.MODELS_BY_TASK.items()},
            "model_names": {m: model_card(m)["name"] for ms in launch.MODELS_BY_TASK.values() for m in ms},
            "input_sets": launch.INPUT_SET_NAMES, "task_names": TASK_NAMES,
            "baselines": {"forecast": "현재값 유지", "virtual_sensor": "레시피 평균", "thickness": "학습 평균"},
            "train_fields": [{"key": p.key, "label": p.label, "default": p.default, "choices": [str(c) for c in p.choices]}
                             for p in launch.TRAIN_FIELDS],
            "gpus": [g.strip() for g in os.environ.get("GPUS", "0,1,2,3").split(",") if g.strip()],
            "default_base": "g1_forecast_params" if "g1_forecast_params" in names else names[0],
            "run_enabled": launch_enabled(),
            "explore": {task: {"targets": list(targets)} for task, targets in REPRESENTATIVE.items()},
            "explore_seeds": list(EXPLORE_SEEDS), "explore_train": dict(EXPLORE_TRAIN), "full_seeds": list(FULL_SEEDS),
        }

    def plan(self, form: dict[str, Any]) -> dict[str, Any]:
        from .ledger import estimate_parts

        docs = launch.docs_from_form(form)
        jobs = [j.to_dict() for doc in docs for j in launch.plan_jobs(doc)]
        if not jobs:
            raise launch.LaunchError("돌릴 작업이 없습니다")
        trained = [j for j in jobs if j["model"] in (form.get("models") or {})]
        gpus = str(form.get("gpus") or "cpu")
        devices = 1 if gpus == "cpu" else len(gpus.split(","))
        warnings = []
        busy = self.busy_runs()
        if busy:
            warnings.append("터미널에서 실행 중인 실험이 있어 GPU를 함께 사용하게 됩니다: " + ", ".join(busy))
        modes = {str(j["params"].get("channel_mode")) for j in trained if j["model"] == "patchtst"}
        if "independent" in modes:
            warnings.append("신호 처리 independent(신호별 독립)는 대상 신호의 과거만 보고 다른 입력은 쓰지 않습니다. 신호 수 × batch "
                            "크기만큼 한 번에 계산해 GPU 메모리(24 GB)를 넘을 수 있습니다. 비교가 목적이 아니면 mixing만 쓰세요.")
        text, basis = estimate_parts(self.ledger(), jobs, docs[0]["task"], devices)
        return {"jobs": len(jobs), "combos": len({(j["label"], j["input_set"]) for j in trained}),
                "experiments": len(docs), "estimate": text, "estimate_basis": basis,
                "yaml": "\n---\n".join(yaml.safe_dump(d, allow_unicode=True, sort_keys=False) for d in docs),
                "warnings": warnings}

    def busy_runs(self) -> list[str]:
        """Experiments being written to now that the queue did not start (e.g. run_all.sh in a terminal)."""

        queued = {r.get("exp_dir") for r in launch.list_requests(self.paths.work_root) if r.get("status") == "running"}
        now, out = time.time(), []
        for run in (self.paths.work_root / "experiments").glob("*/*"):
            if not run.is_dir() or (run / "summary.json").is_file() or str(run) in queued:
                continue
            if now - max([run.stat().st_mtime] + [p.stat().st_mtime for p in run.glob("jobs/*")]) < 600:
                out.append(f"{run.parent.name}/{run.name}")
        return out

    def submit(self, form: dict[str, Any]) -> dict[str, Any]:
        docs = launch.docs_from_form(form)
        records = launch.submit_docs(self.paths.work_root, docs, gpus=str(form.get("gpus") or "cpu"),
                                     report=bool(form.get("report")), data_root=self.paths.data_root, source="dashboard",
                                     smoke=bool(form.get("smoke")))
        return {"ids": [r["id"] for r in records]}

    def plan_file(self, form: dict[str, Any]) -> dict[str, Any]:
        """Every request of a markdown plan (:mod:`ald_lab.planfile`), each planned like the form."""

        from .planfile import read_plan

        items = []
        for item in read_plan(str(form.get("text") or ""), gpus=str(form.get("gpus") or "auto")):
            row = {"title": item["title"], "problems": item["problems"], "ignored": item["ignored"], "ok": False}
            if item["form"] is not None and not item["problems"]:
                f = item["form"]
                row.update(memo=f["memo"], base=f["base"], targets=len(f["targets"]), gpus=f["gpus"])
                try:
                    row.update(self.plan(f), ok=True)
                    row.pop("yaml", None)
                except launch.LaunchError as exc:
                    row["problems"] = [str(exc)]
            items.append(row)
        return {"items": items, "jobs": sum(i.get("jobs", 0) for i in items if i["ok"])}

    def submit_file(self, form: dict[str, Any]) -> dict[str, Any]:
        """Queue every request of a markdown plan in the order written — none if any of them has a problem."""

        from .planfile import read_plan

        items = read_plan(str(form.get("text") or ""), gpus=str(form.get("gpus") or "auto"))
        if not items:
            raise launch.LaunchError("기준 실험 줄이 있는 실험을 찾지 못했습니다")
        bad = [f"{i['title']}: {' · '.join(i['problems'])}" for i in items if i["form"] is None or i["problems"]]
        if bad:
            raise launch.LaunchError("문제가 있는 실험이 있어 대기열에 넣지 않았습니다: " + " / ".join(bad))
        for item in items:  # check all before queueing any
            launch.docs_from_form(item["form"])
        ids = []
        for item in items:
            ids += self.submit(item["form"])["ids"]
        return {"ids": ids, "requests": len(items)}

    def stage(self, step: int, form: dict[str, Any]) -> dict[str, Any]:
        from .stages import (
            patchtst_first,
            search_ranking,
            stage2_configs,
            stage3_config,
        )

        task = form.get("task")
        if task not in ("forecast", "virtual_sensor"):
            raise launch.LaunchError(f"알 수 없는 과제입니다: {task}")
        seeds = [int(x) for x in str(form.get("seeds") or "42,43").replace(";", ",").split(",") if x.strip()]
        base, records = launch.stage_base(task), self.ledger()
        if step == 2:
            docs = stage2_configs(search_ranking(records, task), base, k=int(form.get("k") or 3), seeds=seeds)
            if not docs:
                raise launch.LaunchError("1단계 탐색 결과가 아직 없습니다")
        else:
            doc = stage3_config(patchtst_first(records, task), base, seeds=seeds)
            if doc is None:
                raise launch.LaunchError("대안이 필요한 대상이 없습니다")
            docs = [doc]
        records = launch.submit_docs(self.paths.work_root, docs, gpus=str(form.get("gpus") or "auto"),
                                     report=bool(form.get("report", True)), data_root=self.paths.data_root,
                                     source="dashboard")
        return {"ids": [r["id"] for r in records]}


def launch_enabled() -> bool:
    return os.environ.get("ALD_DASHBOARD_RUN", "1") != "0"


def make_handler(lab: Lab) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ald-lab"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per request is noise
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, doc: Any, status: int = 200) -> None:
            self._send(status, json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _html(self, text: str) -> None:
            self._send(200, text.encode("utf-8"), "text/html; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._html(lab.html or LOADING)
                elif url.path == "/api/meta":
                    self._json(lab.meta())
                elif url.path == "/api/status":
                    self._json({"built_at": lab.built_at, "error": lab.build_error,
                                "worker": launch.worker_running(lab.paths.work_root)})
                elif url.path == "/api/fragment/queue":
                    from .report.lab import queue_html

                    self._html(queue_html(lab.paths.work_root, live=launch_enabled()))
                elif url.path == "/api/fragment/explore":
                    from .report.lab import explore_html

                    self._html(explore_html(lab.ledger()))
                elif url.path == "/api/fragment/first":
                    from .report.lab import first_html

                    self._html(first_html(lab.ledger(), live=launch_enabled()))
                elif url.path == "/api/log":
                    request_id = (parse_qs(url.query).get("id") or [""])[0]
                    self._send(200, launch.log_tail(lab.paths.work_root, request_id, 80).encode("utf-8"),
                               "text/plain; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:  # noqa: BLE001 — report it to the page, keep serving
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if self.headers.get("X-Requested-With") != "ald-lab":
                self._json({"error": "허용되지 않은 요청"}, HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                form = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/plan":
                    self._json(lab.plan(form))
                    return
                if url.path == "/api/plan-file":
                    self._json(lab.plan_file(form))
                    return
                if not launch_enabled():
                    self._json({"error": "이 서버에서는 실행이 꺼져 있습니다 (ALD_DASHBOARD_RUN=0)"})
                    return
                if url.path == "/api/submit":
                    self._json(lab.submit(form))
                elif url.path in ("/api/stage2", "/api/stage3"):
                    self._json(lab.stage(int(url.path[-1]), form))
                elif url.path == "/api/submit-file":
                    self._json(lab.submit_file(form))
                elif url.path == "/api/cancel":
                    self._json({"status": launch.cancel(lab.paths.work_root, str(form.get("id")))})
                elif url.path == "/api/hide":
                    self._json({"status": launch.hide_request(lab.paths.work_root, str(form.get("id")))})
                elif url.path == "/api/hide-finished":
                    self._json({"hidden": launch.hide_finished(lab.paths.work_root)})
                elif url.path == "/api/rebuild":
                    threading.Thread(target=lab.rebuild, daemon=True).start()
                    self._json({"status": "rebuilding"})
                else:
                    self._send(404, b"not found", "text/plain")
            except (launch.LaunchError, ValueError, KeyError) as exc:
                self._json({"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(paths: LabPaths, *, host: str, port: int, examples: bool = True) -> None:
    lab = Lab(paths, examples=examples)
    threading.Thread(target=lab.watch, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(lab))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"실험실 대시보드: http://{shown}:{port}  (결과 폴더 {paths.work_root})", flush=True)
    print(f"  밖에서: ssh -L 8501:localhost:{port} <계정>@<서버> -p <SSH 포트> → 노트북 브라우저 http://localhost:8501",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="보고서(HTML)를 실시간으로 띄우고, 하이퍼파라미터 탐색 탭에서 실험을 실행합니다.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAB_PORT") or os.environ.get("PORT") or 5273))
    parser.add_argument("--host", default=os.environ.get("ALD_LAB_HOST", "127.0.0.1"))
    parser.add_argument("--no-examples", action="store_true", help="예측 예시·추론 시연 없이 (집계만)")
    parser.add_argument("--work-root")
    args = parser.parse_args(argv)
    serve(resolve_paths(work_root=args.work_root), host=args.host, port=args.port, examples=not args.no_examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
