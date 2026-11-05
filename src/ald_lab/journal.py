"""Append-only experiment journal: the chronological record behind the dashboard.

Every pipeline stage (data overview, preprocessing, hyperparameter search,
training, evaluation) appends one JSON line describing what ran, with which
settings, how long it took, what it produced, and the model's strengths and
weaknesses. The dashboard replays these lines in time order.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:  # POSIX only; the GPU server is Ubuntu.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback without locking
    fcntl = None  # type: ignore[assignment]

STAGES = ("data", "preprocess", "hpo", "train", "evaluate", "note")
STAGE_LABELS = {
    "data": "원본 데이터",
    "preprocess": "전처리",
    "hpo": "하이퍼파라미터 탐색",
    "train": "학습",
    "evaluate": "평가",
    "note": "메모",
}


@dataclass
class JournalEntry:
    stage: str
    title: str
    started_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    status: str = "running"
    summary: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    git_commit: str | None = None
    host: str = field(default_factory=platform.node)
    entry_id: str = field(default_factory=lambda: _new_id())

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {STAGES}")


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Journal:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "journal.jsonl"

    def append(self, entry: JournalEntry) -> JournalEntry:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return entry

    def entries(self, stage: str | None = None) -> list[JournalEntry]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = JournalEntry(**json.loads(line))
            if stage is None or entry.stage == stage:
                out.append(entry)
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = [asdict(e) for e in self.entries()]
        if not rows:
            return pd.DataFrame(columns=list(JournalEntry.__dataclass_fields__))
        frame = pd.DataFrame(rows)
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
        return frame.sort_values("started_at").reset_index(drop=True)

    @contextmanager
    def record(
        self,
        stage: str,
        title: str,
        *,
        config: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Iterator[JournalEntry]:
        """Time a block and append its entry, marking it failed on exceptions."""

        entry = JournalEntry(
            stage=stage,
            title=title,
            started_at=_now(),
            config=dict(config or {}),
            parent_ids=list(parent_ids or []),
            tags=list(tags or []),
            git_commit=_git_commit(),
        )
        started = time.perf_counter()
        try:
            yield entry
        except BaseException as exc:
            entry.status = "failed"
            entry.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise
        else:
            entry.status = "ok"
        finally:
            entry.finished_at = _now()
            entry.duration_s = round(time.perf_counter() - started, 3)
            self.append(entry)
