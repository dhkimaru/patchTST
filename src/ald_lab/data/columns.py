"""Column dictionary: what each trace column means and how important it is."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..paths import PROJECT_ROOT

DEFAULT_COLUMN_DICTIONARY = PROJECT_ROOT / "configs" / "columns" / "ald_columns_v1.yaml"
COLUMN_FIELDS = (
    "column",
    "group",
    "family",
    "family_rep",
    "name_ko",
    "unit",
    "param_id",
    "importance",
    "primary",
    "mapping",
    "role",
    "status",
    "duplicate_of",
    "notes",
)


def load_column_dictionary(path: str | Path = DEFAULT_COLUMN_DICTIONARY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(columns, parameters)`` tables from the YAML dictionary."""

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    columns = pd.DataFrame(doc["columns"]).reindex(columns=list(COLUMN_FIELDS))
    columns["primary"] = columns["primary"].eq(True)
    columns["family_rep"] = columns["family_rep"].eq(True)
    columns["family"] = columns["family"].fillna(columns["column"])
    columns["status"] = columns["status"].fillna("OK")
    parameters = pd.DataFrame(doc["parameters"])
    duplicated = columns["column"][columns["column"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"column dictionary lists columns more than once: {duplicated}")
    return columns, parameters


def core_columns(columns: pd.DataFrame, importance: tuple[str, ...] = ("A",)) -> list[str]:
    """Representative (primary) columns of parameters with the given importance."""

    mask = columns["primary"] & columns["importance"].isin(importance) & (columns["status"] == "OK")
    return columns.loc[mask, "column"].tolist()


# Ungraded measurements that the A-grade sets may add: pressure gauges, the APC valve, RF matching positions.
UNGRADED_PLUS_FAMILIES = ("pressure", "exhaust_valve", "rf_match_load", "rf_match_phase")


def grade_columns(columns: pd.DataFrame, importance: tuple[str, ...] = ("A",), *, plus: bool = False) -> list[str]:
    """Every usable column (not only the representative one) of the given importance; with ``plus`` also the
    ungraded pressure gauges, APC valve and RF matching positions (measurements only)."""

    ok = columns["status"] == "OK"
    mask = ok & columns["importance"].isin(importance)
    if plus:
        mask |= ok & columns["importance"].isna() & columns["family"].isin(UNGRADED_PLUS_FAMILIES) \
            & (columns["role"] == "measurement")
    return columns.loc[mask, "column"].tolist()

