"""Run-level train/validation/test splits.

Splits are always by run (file). Test runs stay sealed: analysis and model
selection use train and validation only, and code that needs test runs must
ask for them explicitly with ``include_test=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..paths import PROJECT_ROOT

DEFAULT_SPLIT = PROJECT_ROOT / "configs" / "splits" / "split_v2.yaml"
PARTITIONS = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when a split file is inconsistent."""


@dataclass(frozen=True)
class RunSplit:
    split_id: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    excluded: tuple[str, ...]

    def partition_of(self, run_id: str) -> str | None:
        for name in PARTITIONS:
            if run_id in getattr(self, name):
                return name
        return "excluded" if run_id in self.excluded else None

    def runs(self, *partitions: str, include_test: bool = False) -> list[str]:
        if "test" in partitions and not include_test:
            raise SplitError("test runs are sealed; pass include_test=True for final evaluation only")
        return [r for p in partitions for r in getattr(self, p)]


def load_split(path: str | Path = DEFAULT_SPLIT) -> RunSplit:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    parts = {name: tuple(doc.get(name) or ()) for name in PARTITIONS}
    excluded = tuple(item["run_id"] for item in doc.get("excluded") or ())
    seen: dict[str, str] = {}
    for name, runs in [*parts.items(), ("excluded", excluded)]:
        for run_id in runs:
            if run_id in seen:
                raise SplitError(f"{run_id} appears in both {seen[run_id]} and {name}")
            seen[run_id] = name
    return RunSplit(split_id=str(doc["split_id"]), excluded=excluded, **parts)
