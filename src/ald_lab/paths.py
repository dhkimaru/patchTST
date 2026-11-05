"""Resolve where raw data lives and where generated outputs go.

Raw company data is never stored in the repository. Locations come from
explicit arguments first, then environment variables:

- ``ALD_DATA_ROOT``: directory holding ``standard_trace_NNN.csv`` files
- ``ALD_EXSITU_XLSX``: ex-situ metrology workbook (.xlsx)
- ``ALD_WORK_ROOT``: generated outputs (default ``<project>/work``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathConfigError(ValueError):
    """Raised when a required data location is missing."""


@dataclass(frozen=True)
class LabPaths:
    data_root: Path | None
    exsitu_xlsx: Path | None
    work_root: Path

    @property
    def trace_cache(self) -> Path:
        return self.work_root / "cache" / "traces"

    @property
    def overview_dir(self) -> Path:
        return self.work_root / "data_overview"

    @property
    def journal_dir(self) -> Path:
        return self.work_root / "journal"

    def require_data_root(self) -> Path:
        if self.data_root is None:
            raise PathConfigError("set ALD_DATA_ROOT or pass --data-root")
        if not self.data_root.is_dir():
            raise PathConfigError(f"data root does not exist: {self.data_root}")
        return self.data_root


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def resolve_paths(
    *,
    data_root: str | os.PathLike[str] | None = None,
    exsitu_xlsx: str | os.PathLike[str] | None = None,
    work_root: str | os.PathLike[str] | None = None,
) -> LabPaths:
    work = _optional_path(work_root) or _optional_path(os.environ.get("ALD_WORK_ROOT"))
    return LabPaths(
        data_root=_optional_path(data_root) or _optional_path(os.environ.get("ALD_DATA_ROOT")),
        exsitu_xlsx=_optional_path(exsitu_xlsx)
        or _optional_path(os.environ.get("ALD_EXSITU_XLSX")),
        work_root=work or PROJECT_ROOT / "work",
    )
