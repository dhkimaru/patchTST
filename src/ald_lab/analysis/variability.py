"""Which parameters are worth a virtual sensor? A recipe-template residual analysis.

During ALD cycles most signals follow the recipe: they repeat the same shape
every cycle. A *recipe template* (mean value per recipe, step and position
inside the step, fitted on train runs) captures that shape. What remains,
``value - template``, is the part that carries process information and
anomalies. For each column this module measures on validation runs:

- ``r2_recipe``: share of variance the template explains. Near 1 means the
  signal just follows the recipe; anomaly checks can compare it with the
  template directly.
- ``resid_rel``: residual size relative to the signal's spread.
- ``run_drift``: how much the residual's run mean moves between runs.
- ``r2_cross``: share of the residual that a ridge model on the other
  sensors' residuals explains when the target's whole *family* is removed
  (e.g. every thermocouple of the same zone, every pressure gauge). This is
  the honest "the sensor is not installed" case. ``r2_cross_copy`` removes
  only near-copies (|r| ≥ 0.999) for comparison. High means a virtual
  sensor is feasible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.catalog import CYCLE_STEPS
from ..data.splits import RunSplit
from ..journal import Journal
from ..paths import LabPaths

KEYS = ["recipe", "step_id", "pos"]
# Long steps (ramp, standby) share one template value past this many rows.
POSITION_CAP = 2400
# Recipe-driven: residual RMS at most 10 % of the signal's spread (R² ≥ 0.99).
RECIPE_DRIVEN_R2 = 0.99
CROSS_FEASIBLE_R2 = 0.5
# Non-binary columns with at most this many distinct values per run are
# quantized near-constant readings (e.g. line pressure switches).
LOW_RESOLUTION_UNIQUE = 20
COPY_THRESHOLD = 0.999
RIDGE_ALPHA = 1.0
IMPORTANCE_RANK = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class VariabilityResult:
    table: pd.DataFrame
    summary: dict[str, object]


def _with_keys(frame: pd.DataFrame) -> pd.DataFrame:
    step_key = frame["recipe"] + "\x1f" + frame["step_id"].astype(str)
    segment = step_key.ne(step_key.shift()).cumsum()
    out = frame.copy()
    out["pos"] = np.minimum(out.groupby(segment).cumcount().to_numpy(), POSITION_CAP)
    out["in_cycle"] = out["step_name"].isin(CYCLE_STEPS).to_numpy()
    return out


class TemplateBuilder:
    """Accumulate per-key sums over runs; ``mean()`` gives the recipe template."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._sum: pd.DataFrame | None = None
        self._count: pd.Series | None = None

    def add(self, keyed: pd.DataFrame) -> None:
        grouped = keyed.groupby(KEYS, sort=False)
        sums = grouped[self.columns].sum(min_count=1).astype(np.float64)
        counts = grouped.size()
        if self._sum is None:
            self._sum, self._count = sums, counts
        else:
            self._sum = self._sum.add(sums, fill_value=0.0)
            self._count = self._count.add(counts, fill_value=0)

    def mean(self) -> pd.DataFrame:
        if self._sum is None or self._count is None:
            raise ValueError("no runs added to the template")
        return self._sum.div(self._count, axis=0)


def residual_frame(keyed: pd.DataFrame, template: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows of ``keyed`` that have a template value, with ``value - template`` per column."""

    joined = keyed[KEYS + ["in_cycle"] + columns].join(template, on=KEYS, rsuffix="__tpl", how="inner")
    resid = pd.DataFrame(
        {c: joined[c].to_numpy(np.float64) - joined[f"{c}__tpl"].to_numpy(np.float64) for c in columns},
        index=joined.index,
    )
    resid["in_cycle"] = joined["in_cycle"].to_numpy()
    return resid


def _copy_groups(near_copies: pd.DataFrame, columns: list[str]) -> dict[str, set[str]]:
    groups = {c: {c} for c in columns}
    for row in near_copies.itertuples():
        if abs(row.pearson_r) >= COPY_THRESHOLD and row.column_a in groups and row.column_b in groups:
            groups[row.column_a].add(row.column_b)
            groups[row.column_b].add(row.column_a)
    return groups


def _family_groups(copies: dict[str, set[str]], family: pd.Series) -> dict[str, set[str]]:
    """Near-copies plus every column of the same family."""

    members: dict[str, set[str]] = {}
    for column, fam in family.items():
        members.setdefault(fam, set()).add(column)
    return {c: copies[c] | members.get(family.get(c, c), {c}) for c in copies}


def _ridge_cross_r2(
    train: np.ndarray, valid: np.ndarray, names: list[str], groups: dict[str, set[str]]
) -> tuple[np.ndarray, list[str | None], np.ndarray]:
    """For each column, R² of predicting its residual from the others' residuals."""

    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    xt, xv = (train - mean) / std, (valid - mean) / std
    gram = xt.T @ xt
    corr = gram / len(xt)
    r2 = np.full(len(names), np.nan)
    partner: list[str | None] = [None] * len(names)
    partner_r = np.full(len(names), np.nan)
    for t, name in enumerate(names):
        if np.allclose(xv[:, t].var(), 0.0):
            continue
        allowed = [i for i, n in enumerate(names) if n not in groups.get(name, {name})]
        if not allowed:
            continue
        a = np.asarray(allowed)
        w = np.linalg.solve(gram[np.ix_(a, a)] + RIDGE_ALPHA * np.eye(len(a)), gram[a, t])
        pred = xv[:, a] @ w
        sse = float(((xv[:, t] - pred) ** 2).sum())
        sst = float(((xv[:, t] - xv[:, t].mean()) ** 2).sum())
        r2[t] = 1 - sse / sst if sst > 0 else np.nan
        best = a[np.argmax(np.abs(corr[a, t]))]
        partner[t], partner_r[t] = names[best], float(corr[best, t])
    return r2, partner, partner_r


CLASSES = (
    "공정 응답 · 다른 센서로 추정 가능",
    "공정 응답 · 추정 어려움",
    "레시피 추종",
    "설정값·제어 명령",
    "저해상도 신호",
    "사이클 중 일정",
)


def _classify(row: pd.Series) -> str:
    if not np.isfinite(row["cycle_std"]) or row["cycle_std"] <= 1e-9:
        return "사이클 중 일정"
    if row.get("role") in ("setpoint", "command"):
        return "설정값·제어 명령"
    if row["r2_recipe"] >= RECIPE_DRIVEN_R2:
        return "레시피 추종"
    if not row.get("binary", False) and row.get("max_unique_per_run", np.inf) <= LOW_RESOLUTION_UNIQUE:
        return "저해상도 신호"
    if np.isfinite(row["r2_cross"]) and row["r2_cross"] >= CROSS_FEASIBLE_R2:
        return "공정 응답 · 다른 센서로 추정 가능"
    return "공정 응답 · 추정 어려움"


def analyze_variability(
    paths: LabPaths,
    split: RunSplit,
    columns_table: pd.DataFrame,
    near_copies: pd.DataFrame,
    *,
    subsample: int = 10,
) -> VariabilityResult:
    columns = columns_table.loc[columns_table["status"] == "OK", "column"].tolist()
    cache = paths.trace_cache

    def load(run_id: str) -> pd.DataFrame:
        matches = sorted(cache.glob(f"{run_id}.*.parquet"))
        if not matches:
            raise FileNotFoundError(f"{run_id} missing from trace cache; run ald-build-overview first")
        return _with_keys(pd.read_parquet(matches[-1]))

    builder = TemplateBuilder(columns)
    for run_id in split.train:
        builder.add(load(run_id))
    template = builder.mean()

    train_rows, valid_rows = [], []
    for run_id in split.train:
        resid = residual_frame(load(run_id), template, columns)
        train_rows.append(resid.loc[resid["in_cycle"], columns].iloc[::subsample])

    sse = np.zeros(len(columns))
    values_sum = np.zeros(len(columns))
    values_sq = np.zeros(len(columns))
    n_rows = 0
    run_means = []
    for run_id in split.validation:
        keyed = load(run_id)
        resid = residual_frame(keyed, template, columns)
        cyc = resid[resid["in_cycle"]]
        raw = keyed.loc[cyc.index, columns].to_numpy(np.float64)
        r = cyc[columns].to_numpy(np.float64)
        sse += np.nansum(r**2, axis=0)
        values_sum += np.nansum(raw, axis=0)
        values_sq += np.nansum(raw**2, axis=0)
        n_rows += len(raw)
        run_means.append(np.nanmean(r, axis=0))
        valid_rows.append(cyc[columns].iloc[::subsample])

    mean = values_sum / n_rows
    sst = values_sq - n_rows * mean**2
    cycle_std = np.sqrt(np.maximum(sst / n_rows, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_recipe = np.where(sst > 0, 1 - sse / sst, np.nan)
        resid_rmse = np.sqrt(sse / n_rows)
        resid_rel = np.where(cycle_std > 0, resid_rmse / cycle_std, np.nan)
        run_drift = np.where(cycle_std > 0, np.std(np.vstack(run_means), axis=0) / cycle_std, np.nan)

    train_mat = pd.concat(train_rows).to_numpy(np.float64)
    valid_mat = pd.concat(valid_rows).to_numpy(np.float64)
    finite = np.isfinite(train_mat).all(axis=1)
    finite_v = np.isfinite(valid_mat).all(axis=1)
    meta = columns_table.set_index("column")
    family = meta["family"] if "family" in meta.columns else pd.Series(dict(zip(columns, columns)))
    copies = _copy_groups(near_copies, columns)
    r2_cross, partner, partner_r = _ridge_cross_r2(
        train_mat[finite], valid_mat[finite_v], columns, _family_groups(copies, family)
    )
    r2_cross_copy, _, _ = _ridge_cross_r2(train_mat[finite], valid_mat[finite_v], columns, copies)

    table = pd.DataFrame(
        {
            "column": columns,
            "cycle_std": cycle_std,
            "r2_recipe": r2_recipe,
            "resid_rmse": resid_rmse,
            "resid_rel": resid_rel,
            "run_drift": run_drift,
            "r2_cross": r2_cross,
            "r2_cross_copy": r2_cross_copy,
            "top_partner": partner,
            "top_partner_r": partner_r,
        }
    )
    for field in ("name_ko", "group", "family", "family_rep", "unit", "param_id", "importance", "primary",
                  "role", "mapping", "binary", "max_unique_per_run"):
        if field in meta.columns:
            table[field] = table["column"].map(meta[field])
    table["class"] = table.apply(_classify, axis=1)
    table["importance_rank"] = table["importance"].map(IMPORTANCE_RANK).fillna(3)
    # First virtual-sensor candidates: one representative per family (the parameter
    # sheet's A/B items, plus chamber pressure, APC valve and RF matching, which
    # the sheet does not rate) whose ALD-cycle behaviour is a process response.
    is_response = table["class"].str.startswith("공정 응답")
    is_measured = table["role"].fillna("measurement") == "measurement"
    table["shortlist"] = is_response & is_measured & table["family_rep"].fillna(False).astype(bool)
    table = table.sort_values(["shortlist", "importance_rank", "r2_recipe"], ascending=[False, True, True])
    table = table.reset_index(drop=True)

    summary = {
        "split_id": split.split_id,
        "n_train_runs": len(split.train),
        "n_validation_runs": len(split.validation),
        "region": "ALD cycle steps " + ",".join(CYCLE_STEPS),
        "n_columns": len(columns),
        "class_counts": table["class"].value_counts().to_dict(),
        "shortlist": table.loc[table["shortlist"], "column"].tolist(),
        "thresholds": {"recipe_driven_r2": RECIPE_DRIVEN_R2, "cross_feasible_r2": CROSS_FEASIBLE_R2,
                       "copy_threshold": COPY_THRESHOLD, "low_resolution_unique": LOW_RESOLUTION_UNIQUE},
        "cross_exclusion": "target family + near-copies",
    }
    return VariabilityResult(table=table, summary=summary)


def run_variability_analysis(paths: LabPaths, split: RunSplit, journal: Journal | None = None) -> VariabilityResult:
    """CLI entry: read overview tables, analyze, write outputs and a journal entry."""

    overview = paths.overview_dir
    columns_table = pd.read_parquet(overview / "columns.parquet")
    near = pd.read_parquet(overview / "near_copies.parquet")
    journal = journal or Journal(paths.journal_dir)
    out = paths.work_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "split_id": split.split_id,
        "template_fit": "train runs",
        "evaluated_on": "validation runs (test sealed)",
        "region": "ALD cycle steps",
        "position_cap_rows": POSITION_CAP,
        "ridge_alpha": RIDGE_ALPHA,
        "thresholds": {"recipe_driven_r2": RECIPE_DRIVEN_R2, "cross_feasible_r2": CROSS_FEASIBLE_R2,
                       "low_resolution_unique": LOW_RESOLUTION_UNIQUE},
        "cross_exclusion": "target family (same zone thermocouples, all pressure gauges, same gas line) + near-copies",
    }
    with journal.record("data", "가상 센서 대상 분석 (레시피 대비 변동)", config=config,
                        tags=["virtual_sensor", "target_selection"]) as entry:
        result = analyze_variability(paths, split, columns_table, near)
        result.table.to_parquet(out / "param_variability.parquet", index=False)
        (out / "param_variability.json").write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        counts = result.summary["class_counts"]
        entry.summary = (
            f"ALD 사이클 구간에서 {result.summary['n_columns']}개 컬럼을 레시피 템플릿(학습 run)과 비교했습니다 "
            f"(검증 run {result.summary['n_validation_runs']}개로 평가). "
            + ", ".join(f"{k} {v}개" for k, v in counts.items())
            + f". 1차 가상 센서 후보 {len(result.summary['shortlist'])}개."
        )
        entry.metrics = {"class_counts": counts, "n_shortlist": len(result.summary["shortlist"])}
        entry.artifacts = {"table": str(out / "param_variability.parquet")}
        short = result.table[result.table["shortlist"]]
        entry.strengths = [
            f"{r.column}: 레시피 설명력 {r.r2_recipe:.2f}, 같은 계열 없이 다른 센서로 잔차 {r.r2_cross:.2f} 설명"
            for r in short.head(6).itertuples()
        ]
        entry.weaknesses = [
            "계열(family) 구분은 컬럼 이름 규칙으로 정했고, 실제 센서 위치·배관 구성은 확인 전",
            "선형(ridge) 기준의 1차 판정. 비선형·시계열 모델은 다음 단계에서 비교",
            "컬럼 대응(밸브·N2 MFC 역할)은 아직 추정이며 사용자 확인 대기",
        ]
    return result
