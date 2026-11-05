"""Next-experiment candidates for the report's 다음 실험 tab, read from ``configs/plan/next_experiments.yaml``.

The YAML holds only aggregate numbers and descriptions. ``load_plan`` checks the shape (unique ids,
known priority and status) so a typo shows up in the tests rather than as a broken tab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import PROJECT_ROOT

PLAN_PATH = PROJECT_ROOT / "configs" / "plan" / "next_experiments.yaml"
STATUSES = ("계획", "진행 중", "완료", "보류")
PRIORITIES = (1, 2, 3)
REQUIRED = ("id", "title", "priority", "status", "question")
TEXT_FIELDS = ("goal", "variants_title", "targets", "metric", "success", "cost", "prerequisite", "risk", "result")
LIST_FIELDS = ("evidence", "method")


class PlanError(ValueError):
    """Raised when the plan file does not have the expected shape."""


def _item(raw: dict[str, Any], category: str) -> dict[str, Any]:
    missing = [k for k in REQUIRED if not raw.get(k)]
    if missing:
        raise PlanError(f"{category}/{raw.get('id', '?')}: missing {', '.join(missing)}")
    if raw["priority"] not in PRIORITIES:
        raise PlanError(f"{raw['id']}: priority must be one of {PRIORITIES}")
    if raw["status"] not in STATUSES:
        raise PlanError(f"{raw['id']}: status must be one of {STATUSES}")
    item = {k: raw[k] for k in REQUIRED}
    item.update({k: str(raw.get(k) or "") for k in TEXT_FIELDS})
    item.update({k: [str(v) for v in raw.get(k) or []] for k in LIST_FIELDS})
    variants = []
    for row in raw.get("variants") or []:
        row = [str(v) for v in row]
        if not 2 <= len(row) <= 3:
            raise PlanError(f"{raw['id']}: a variant is [name, description] or [name, description, note]")
        variants.append((row + [""])[:3])
    item["variants"] = variants
    return item


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any] | None:
    """The plan as plain dicts, or None when the file is absent."""

    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories, seen = [], set()
    for cat in raw.get("categories") or []:
        if not cat.get("key") or not cat.get("title"):
            raise PlanError("each category needs a key and a title")
        items = [_item(x, cat["key"]) for x in cat.get("items") or []]
        for item in items:
            if item["id"] in seen:
                raise PlanError(f"duplicate id {item['id']}")
            seen.add(item["id"])
        categories.append({"key": cat["key"], "title": cat["title"], "why": str(cat.get("why") or ""), "items": items})
    return {"as_of": str(raw.get("as_of") or ""), "basis": str(raw.get("basis") or ""),
            "principles": [str(p) for p in raw.get("principles") or []], "categories": categories}
