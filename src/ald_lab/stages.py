"""PatchTST first: search PatchTST settings, carry the best combinations to every target, other models only where
PatchTST does not work.

1. **탐색**: PatchTST settings tried on representative targets with short training — the dashboard's 하이퍼파라미터
   탐색 form (experiments ending in ``__hp``) or ``configs/experiments/p1_patchtst_search_*.yaml``.
   :func:`explore_groups` reads each search back as a design: which setting moved the error how much, against the
   seed noise.
2. **전체 적용** (:func:`stage2_configs`): the top-k combinations (ranked by MAE relative to the best combination of
   each target, averaged over targets) on every target of the task, full training, several seeds.
3. **대안** (:func:`stage3_config`): the other models, only for targets where the best PatchTST does not beat the
   baseline beyond the seed noise (:func:`patchtst_first`). LightGBM is shown for reference and never decides.

Everything here reads the cumulative records (:mod:`ald_lab.ledger`) and returns experiment YAMLs as dicts or
plain tables; the queue (:mod:`ald_lab.launch`) runs them. No torch.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .launch import HP_SUFFIX
from .ledger import main_records

SEARCH_PREFIX = "p1_patchtst_search"  # base name of step-1 experiments written as files
STAGE2_PREFIX = "p2_patchtst_all"
STAGE3_PREFIX = "p3_fallback"
BASE_FOR_TASK = {"forecast": "g1_forecast_params", "virtual_sensor": "g1_virtual_sensor"}
FALLBACK_MODELS = ("itransformer", "dlinear", "lightgbm")
NOISE_SD = 2.0  # PatchTST is kept when it beats the baseline by more than this many seed standard deviations
BASELINES = ("persistence", "template", "mean")
DECISIONS = {"keep": "PatchTST 채택", "fallback": "대안 필요", "missing": "PatchTST 미학습"}
# Step 1 runs on a few targets that stand for the rest (the dashboard's "탐색용" preset fills these in).
REPRESENTATIVE = {"forecast": ("VG11", "APCValveMon", "MFCMon_L.POS"),
                  "virtual_sensor": ("VG11", "TempAct_C", "MFCMon_L.POS")}
EXPLORE_TRAIN = {"max_epochs": 15, "patience": 4, "windows_per_epoch": 100_000, "batch_size": 1024}
EXPLORE_SEEDS = (42, 43)
FULL_SEEDS = (42, 43, 44)
EFFECT_SE = 2.0  # an effect counts as clear when it is larger than this many standard errors
MIN_EFFECT = 1.0  # … and at least this many %p of improvement; smaller ones are "small" (not worth tuning)


def _py(value: Any) -> Any:
    """numpy scalars → plain Python (YAML and JSON take only those)."""

    return value.item() if isinstance(value, np.generic) else value


def _combo_key(params: dict[str, Any], input_set: str, context: Any = None) -> str:
    ctx = None if context is None or (isinstance(context, float) and math.isnan(context)) else float(context)
    return json.dumps([sorted((k, v) for k, v in params.items()), input_set, ctx], default=str)


def _context_of(row: Any) -> float | None:
    value = row.get("context_seconds") if hasattr(row, "get") else None
    return None if value is None or pd.isna(value) else float(value)


def _search_rows(records: pd.DataFrame, task: str | None, model: str = "patchtst") -> pd.DataFrame:
    """Step-1 runs: experiment files named ``p1_patchtst_search*`` and every dashboard run (``…__hp``)."""

    rows = records[(records["model"] == model) & records["mae"].notna() & ~records["smoke"].astype(bool)]
    if task is not None:
        rows = rows[rows["task"] == task]
    experiment = rows["experiment"].astype(str) if "experiment" in rows else pd.Series("", index=rows.index)
    return rows[rows["base"].astype(str).str.startswith(SEARCH_PREFIX) | experiment.str.endswith(HP_SUFFIX)]


def search_ranking(records: pd.DataFrame, task: str, model: str = "patchtst") -> pd.DataFrame:
    """Every combination tried in step 1 for ``task``: relative MAE (1.0 = best combination of that target)
    averaged over targets, best first. A combination is its settings, input set and window length."""

    rows = _search_rows(records, task, model)
    columns = ["key", "params", "input_set", "context", "score", "targets", "complete", "mae_by_target", "runs"]
    if not len(rows):
        return pd.DataFrame(columns=columns)
    contexts = rows["context_seconds"] if "context_seconds" in rows else pd.Series(None, index=rows.index)
    rows = rows.assign(context=[None if c is None or pd.isna(c) else float(c) for c in contexts])
    rows = rows.assign(key=[_combo_key(p, s, c) for p, s, c in zip(rows["params"], rows["input_set"], rows["context"])])
    per = rows.groupby(["key", "target"])["mae"].mean().reset_index()
    per["rel"] = per["mae"] / per.groupby("target")["mae"].transform("min")
    n_targets = per["target"].nunique()
    out = []
    for key, group in per.groupby("key"):
        sample = rows[rows["key"] == key].iloc[0]
        out.append({"key": key, "params": sample["params"], "input_set": sample["input_set"], "context": sample["context"],
                    "score": float(group["rel"].mean()), "targets": len(group), "complete": len(group) == n_targets,
                    "mae_by_target": dict(zip(group["target"], group["mae"])),
                    "runs": rows.loc[rows["key"] == key, "exp_dir"].nunique()})
    frame = pd.DataFrame(out, columns=columns)
    return frame.sort_values(["complete", "score"], ascending=[False, True]).reset_index(drop=True)


def stage2_configs(ranking: pd.DataFrame, base: dict[str, Any], *, k: int = 3, seeds: list[int] | None = None,
                   memo: str = "") -> list[dict[str, Any]]:
    """Step 2: the top-``k`` combinations on every target of ``base`` (one YAML per input set and window they use)."""

    top = ranking.head(k)
    if not len(top):
        return []
    if "context" not in top:
        top = top.assign(context=None)
    top = top.assign(_ctx=[-1.0 if c is None or pd.isna(c) else float(c) for c in top["context"]])
    groups = list(top.groupby(["input_set", "_ctx"], sort=False))
    several_sets = top["input_set"].nunique() > 1
    several_windows = top["_ctx"].nunique() > 1
    docs = []
    for (input_set, ctx), group in groups:
        doc = copy.deepcopy(base)
        # a first-class experiment (not __hp): the report's main tabs and 추론 시연 show it
        doc["name"] = (f"{STAGE2_PREFIX}_{base['task']}" + (f"_{input_set}" if several_sets else "")
                       + (f"_w{ctx:g}" if several_windows and ctx >= 0 else ""))
        doc["description"] = memo or f"PatchTST 2단계: 탐색 상위 {len(top)}개 조합을 모든 대상에 ({input_set})"
        doc["seeds"] = list(seeds or FULL_SEEDS[:2])
        data = dict(doc.get("data") or {})
        data.pop("input_sets", None)
        data["inputs"] = input_set
        doc["data"] = data
        if ctx >= 0 and "window" in doc:
            doc["window"] = {**doc["window"], "context_seconds": float(ctx)}
        doc["models"] = [{"name": "patchtst", "label": f"patchtst_top{rank + 1}",
                          "params": {k: _py(v) for k, v in dict(params).items()}}
                         for rank, params in zip(group.index, group["params"])]
        docs.append(doc)
    return docs


def patchtst_first(records: pd.DataFrame, task: str, *, noise_sd: float = NOISE_SD) -> pd.DataFrame:
    """Per target of ``task``: the best PatchTST configuration (any run, seeds averaged), whether it beats the
    baseline beyond the seed noise (the decision), LightGBM for reference, and the best other model."""

    records = main_records(records)  # cross-validation folds and other side studies do not decide
    rows = records[(records["task"] == task) & records["mae"].notna() & ~records["smoke"].astype(bool)]
    out = []
    for target, group in rows.groupby("target", sort=True):
        patch = group[group["model"] == "patchtst"]
        lgbm = group[group["model"] == "lightgbm"]
        base = group[group["model"].isin(BASELINES)]
        others = group[~group["model"].isin(("patchtst", *BASELINES))]
        row: dict[str, Any] = {"target": target, "unit": group["unit"].dropna().iloc[0] if group["unit"].notna().any() else ""}
        row["baseline_mae"] = float(base["mae"].min()) if len(base) else None
        row["lgbm_mae"] = float(lgbm["mae"].min()) if len(lgbm) else None
        if len(patch):
            # a configuration = (run, label): average its seeds, keep the best configuration
            cfg = patch.groupby(["exp_dir", "label"]).agg(mae=("mae", "mean"), std=("mae", "std"), n=("mae", "size"),
                                                         settings=("settings", "first"),
                                                         input_set=("input_set", "first"),
                                                         experiment=("experiment", "first")).reset_index()
            best = cfg.sort_values("mae").iloc[0]
            row.update(patch_mae=float(best["mae"]), patch_std=None if pd.isna(best["std"]) else float(best["std"]),
                       patch_seeds=int(best["n"]), patch_settings=best["settings"], patch_input=best["input_set"],
                       patch_experiment=best["experiment"], patch_configs=len(cfg))
            margin = noise_sd * (row["patch_std"] or 0.0)
            if row["baseline_mae"] is None:
                row["decision"], row["reason"], row["patch_gain_pct"] = "keep", "기준선 기록 없음", None
            else:
                gain = row["baseline_mae"] - row["patch_mae"]
                row["patch_gain_pct"] = 100 * gain / row["baseline_mae"] if row["baseline_mae"] else None
                row["decision"] = "keep" if gain > margin else "fallback"
                if row["decision"] == "keep":
                    row["reason"] = f"기준선보다 {row['patch_gain_pct']:.1f} % 좋음" + (
                        " (seed 1개, 편차 미확인)" if row["patch_seeds"] < 2 else "")
                else:
                    row["reason"] = ("기준선보다 나쁨" if gain <= 0 else
                                     f"기준선 대비 {row['patch_gain_pct']:.1f} % 개선이나 seed 편차 {noise_sd:g}배 이내")
        else:
            row.update(patch_mae=None, patch_std=None, patch_seeds=0, patch_settings="", patch_input="",
                       patch_experiment="", patch_configs=0, patch_gain_pct=None, decision="missing",
                       reason="PatchTST 미학습")
        row["lgbm_gap_pct"] = (100 * (row["patch_mae"] / row["lgbm_mae"] - 1)
                               if row["patch_mae"] is not None and row["lgbm_mae"] else None)
        if len(others):
            best_other = others.loc[others["mae"].idxmin()]
            row.update(other_model=best_other["model"], other_label=best_other["label"],
                       other_mae=float(best_other["mae"]))
        else:
            row.update(other_model="", other_label="", other_mae=None)
        if row["decision"] == "keep":
            row["final"], row["final_mae"] = "patchtst", row["patch_mae"]
        elif row["other_mae"] is not None and (row["patch_mae"] is None or row["other_mae"] < row["patch_mae"]):
            row["final"], row["final_mae"] = row["other_model"], row["other_mae"]
        else:
            row["final"], row["final_mae"] = ("patchtst", row["patch_mae"]) if row["patch_mae"] is not None else ("", None)
        out.append(row)
    return pd.DataFrame(out)


def stage3_config(decisions: pd.DataFrame, base: dict[str, Any], *, models: tuple[str, ...] = FALLBACK_MODELS,
                  seeds: list[int] | None = None, memo: str = "") -> dict[str, Any] | None:
    """Step 3: the other models on the targets PatchTST did not settle."""

    targets = [t for t in decisions.loc[decisions["decision"] == "fallback", "target"] if t in base.get("targets", [])]
    if not targets:
        return None
    doc = copy.deepcopy(base)
    doc["name"] = f"{STAGE3_PREFIX}_{base['task']}"
    doc["description"] = memo or f"3단계 대안: PatchTST가 기준선을 넘지 못한 대상 {len(targets)}개에 다른 모델"
    doc["targets"] = targets
    doc["seeds"] = list(seeds or FULL_SEEDS[:2])
    doc["models"] = [{"name": m} for m in models]
    return doc


def summary_numbers(decisions: pd.DataFrame) -> dict[str, int]:
    counts = decisions["decision"].value_counts() if len(decisions) else pd.Series(dtype=int)
    return {k: int(counts.get(k, 0)) for k in DECISIONS} | {"targets": len(decisions)}


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows (NaN → None)."""

    return json.loads(frame.replace({np.nan: None}).to_json(orient="records", force_ascii=False, default_handler=str))


# ----------------------------------------------------------------------------- reading a search back as a design

_WINDOW_NOTE = re.compile(r"\s*·\s*window [0-9.]+초$")
_FILE_SUFFIX = re.compile(r"_w[0-9]+_[A-Za-z_]+$")


def _group_key(row: Any) -> tuple[str, str]:
    """Runs that belong to one search: a dashboard request (its memo, one experiment per window) or a family of
    ``p1_patchtst_search_*`` files (``…_w192_all``, ``…_w96_A`` share a name stem)."""

    if str(row["experiment"]).endswith(HP_SUFFIX):
        return row["task"], _WINDOW_NOTE.sub("", str(row.get("memo") or "")) or row["base"]
    return row["task"], _FILE_SUFFIX.sub("", str(row["base"]))


def _sort_levels(values: list[Any]) -> list[Any]:
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _factor_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One column per setting of the rows (params, input set, window), as strings for grouping."""

    params = pd.DataFrame([dict(p) for p in rows["params"]], index=rows.index)
    params["input_set"] = rows["input_set"]
    params["context_seconds"] = [None if c is None or pd.isna(c) else float(c) for c in rows.get("context_seconds",
                                                                                                    [None] * len(rows))]
    return params


def _effects(rows: pd.DataFrame, factors: pd.DataFrame, varying: list[list[str]], sigma: float | None) -> list[dict]:
    out = []
    centred = rows["y"] - rows.groupby("target")["y"].transform("mean")
    for names in varying:
        col = factors[names[0]]
        levels = _sort_levels(list(col.dropna().unique()))
        means = {lv: float(centred[col == lv].mean()) for lv in levels}
        counts = {lv: int((col == lv).sum()) for lv in levels}
        best = max(levels, key=lambda lv: means[lv])
        worst = min(levels, key=lambda lv: means[lv])
        if len(levels) == 2:
            lo, hi = levels
            effect = means[hi] - means[lo]
            pair = (lo, hi)
        else:
            effect = means[best] - means[worst]
            pair = (worst, best)
        se = sigma * math.sqrt(1 / counts[pair[0]] + 1 / counts[pair[1]]) if sigma else None
        per_target = {}
        for target, g in rows.groupby("target"):
            gc = g["y"]
            gcol = col.loc[g.index]
            a, b = gc[gcol == pair[0]], gc[gcol == pair[1]]
            if len(a) and len(b):
                per_target[target] = float(b.mean() - a.mean())
        signs = {np.sign(v) for v in per_target.values() if abs(v) > 1e-12}
        verdict = "norepeat" if se is None else ("unclear" if abs(effect) <= EFFECT_SE * se
                                                 else ("clear" if abs(effect) >= MIN_EFFECT else "small"))
        numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in levels)
        hint = ""
        if verdict == "clear" and numeric and len(levels) >= 2 and best in (levels[0], levels[-1]):
            hint = "더 큰 값도 시도" if best == levels[-1] else "더 작은 값도 시도"
        elif verdict in ("unclear", "small"):
            hint = "고정 가능 (학습이 빠른 값으로)"
        levels = [_py(v) for v in levels]
        means = {_py(k): v for k, v in means.items()}
        counts = {_py(k): v for k, v in counts.items()}
        out.append({"factors": names, "levels": levels, "means": means, "counts": counts,
                    "pair": (_py(pair[0]), _py(pair[1])), "effect": effect, "se": se, "verdict": verdict,
                    "best": _py(best), "per_target": per_target,
                    "same_direction": len(signs) <= 1, "hint": hint})
    return sorted(out, key=lambda r: -abs(r["effect"]))


def explore_summary(rows: pd.DataFrame, baselines: pd.DataFrame | None = None) -> dict[str, Any]:
    """One search read as a design: combinations ranked, each setting's effect on the improvement over the
    baseline (centred per target) with the seed noise, and per-target directions.

    ``rows``: PatchTST jobs of the search (ledger rows). ``baselines``: the baseline jobs of the same runs, for the
    improvement inside ALD cycles."""

    rows = rows[rows["improvement_pct"].notna()].copy()
    rows["y"] = rows["improvement_pct"].astype(float)
    if baselines is not None and len(baselines) and "mae_cycle" in rows:
        base_cycle = baselines.groupby(["exp_dir", "target"])["mae_cycle"].mean()
        ref = [base_cycle.get((d, t)) for d, t in zip(rows["exp_dir"], rows["target"])]
        rows["y_cycle"] = [100 * (1 - m / b) if b and pd.notna(m) and pd.notna(b) else None
                           for m, b in zip(rows["mae_cycle"], ref)]
    else:
        rows["y_cycle"] = None
    factors = _factor_table(rows)
    rows["combo"] = [json.dumps(sorted((k, str(v)) for k, v in r.items() if pd.notna(v)), default=str)
                     for r in factors.to_dict(orient="records")]
    # settings that change between combinations; settings that always change together are one factor
    groups: dict[tuple, list[str]] = {}
    combos = rows.drop_duplicates("combo")
    for name in factors.columns:
        col = factors.loc[combos.index, name]
        if col.nunique(dropna=True) < 2:
            continue
        codes = tuple(pd.factorize(col.astype(str))[0])
        groups.setdefault(codes, []).append(name)
    varying = list(groups.values())
    cell = rows.groupby(["combo", "target"])["y"]
    resid = rows["y"] - cell.transform("mean")
    dof = len(rows) - rows.groupby(["combo", "target"]).ngroups
    sigma = float(math.sqrt((resid ** 2).sum() / dof)) if dof > 0 else None
    shown = [n for names in varying for n in names]
    ranking = []
    for combo, g in rows.groupby("combo"):
        first = factors.loc[g.index[0]]
        per_seed = g.groupby("seed")["y"].mean()
        ranking.append({
            "combo": combo, "settings": {n: _py(first[n]) for n in shown},
            "all_settings": {k: _py(v) for k, v in first.dropna().to_dict().items()},
            "improvement": float(g["y"].mean()), "sd": float(per_seed.std()) if len(per_seed) > 1 else None,
            "cycle": float(pd.Series(g["y_cycle"]).dropna().mean()) if pd.Series(g["y_cycle"]).notna().any() else None,
            "seeds": int(g["seed"].nunique()), "targets": int(g["target"].nunique()),
            "mae_by_target": g.groupby("target")["mae"].mean().to_dict(),
            "fit_minutes": float(g["fit_minutes"].mean()) if "fit_minutes" in g and g["fit_minutes"].notna().any() else None,
            "n_params": float(g["n_params"].mean()) if "n_params" in g and g["n_params"].notna().any() else None,
        })
    ranking.sort(key=lambda r: -r["improvement"])
    return {"rows": len(rows), "combos": len(ranking), "targets": sorted(rows["target"].unique()),
            "seeds": sorted(int(s) for s in rows["seed"].dropna().unique()), "sigma": sigma,
            "ranking": ranking, "effects": _effects(rows, factors, varying, sigma) if varying else []}


def explore_groups(records: pd.DataFrame, task: str | None = None) -> list[dict[str, Any]]:
    """Every search in the records (newest first), each read back with :func:`explore_summary`."""

    rows = _search_rows(records, task)
    if not len(rows):
        return []
    keys = [_group_key(r) for r in rows.to_dict(orient="records")]
    rows = rows.assign(_task=[k[0] for k in keys], _group=[k[1] for k in keys])
    base_rows = records[records["model"].isin(BASELINES) & records["exp_dir"].isin(set(rows["exp_dir"]))]
    out = []
    for (task_name, name), g in rows.groupby(["_task", "_group"], sort=False):
        summary = explore_summary(g, base_rows[base_rows["exp_dir"].isin(set(g["exp_dir"]))])
        when = g["when"].max() if "when" in g else None
        out.append({"task": task_name, "title": name, "when": when, "experiments": sorted(g["experiment"].unique()),
                    "running": bool((g.get("status") == "실행 중").any()) if "status" in g else False, **summary})
    out.sort(key=lambda r: (r["when"] is not None and r["when"] == r["when"], r["when"]), reverse=True)
    return out
