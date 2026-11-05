"""What the Hyper Parameter tab derives from finished experiments: every configuration tried, and the
effect of changing one setting at a time, judged against the spread between seeds (no torch)."""

from __future__ import annotations

import math
import re
from itertools import product
from typing import Any

import pandas as pd

from .collect import Experiment

HORIZON_COLUMN = re.compile(r"^mae_(\d+(?:\.\d+)?)s$")
FACTOR_NAMES = {"spec": "모델 구성", "input_set": "입력 세트", "d_model": "모델 크기 (d_model)",
                "patch_seconds": "patch 길이", "stride_seconds": "stride", "n_components": "PLS 성분 수",
                "n_layers": "층 수", "n_heads": "head 수", "d_ff": "d_ff", "kernel": "이동평균 길이",
                "hidden": "은닉 크기"}
SECONDS = {"patch_seconds", "stride_seconds", "kernel_seconds", "context_seconds"}
# A difference counts as an effect when it exceeds this many standard errors of the seed spread.
CLEAR_SE = 2.0
INPUT_REFERENCE = "all"  # input-set effects are measured from the full signal set


def horizon_columns(columns: Any) -> list[str]:
    """``mae_10s``-style columns (horizon > 0), in horizon order."""

    found = []
    for column in columns:
        match = HORIZON_COLUMN.match(str(column))
        if match and float(match.group(1)) > 0:
            found.append((float(match.group(1)), str(column)))
    return [c for _, c in sorted(found)]


def horizon_label(column: str) -> str:
    match = HORIZON_COLUMN.match(column)
    return f"{float(match.group(1)):g}초" if match else column


def plain(value: Any) -> Any:
    """32.0 → 32: grid values come back as floats when a column also holds None."""

    return int(value) if isinstance(value, float) and value.is_integer() else value


def setting_value(key: str, value: Any) -> str:
    value = plain(value)
    if key == "spec":
        return str(value)
    if key == "input_set":
        return f"입력 {value}"
    if key in SECONDS:
        return f"{FACTOR_NAMES.get(key, key)} {value}초"
    return f"{key} {value}"


def job_table(exp: Experiment) -> pd.DataFrame:
    """One row per finished, non-baseline job, with its model configuration, grid values and input set."""

    results = exp.results
    if "status" in results:
        results = results[results["status"] == "ok"]
    results = results.drop(columns=[c for c in ("variant", "input_set") if c in results])
    meta = pd.DataFrame([{"job_id": j["job_id"], "variant": j.get("variant") or {},
                          "input_set": j.get("input_set", "default")} for j in exp.jobs],
                        columns=["job_id", "variant", "input_set"])
    frame = results.merge(meta, on="job_id", how="left")
    frame["variant"] = frame["variant"].apply(lambda v: v if isinstance(v, dict) else {})
    frame["input_set"] = frame["input_set"].fillna("default")
    frame["spec"] = frame["label"].astype(str).str.split("[").str[0].str.split("@").str[0]
    frame = frame[frame["model"] != str(exp.summary.get("baseline"))].reset_index(drop=True)
    for key in sorted({k for v in frame["variant"] for k in v}):
        frame[key] = frame["variant"].apply(lambda v, k=key: v.get(k))
    return frame


def _factors(group: pd.DataFrame) -> list[str]:
    """Settings that took more than one value within this (target, model) group."""

    keys = ["spec", "input_set", *sorted({k for v in group["variant"] for k in v})]
    return [k for k in keys if k in group and group[k].dropna().nunique() > 1]


def setting_text(row: pd.Series, factors: list[str]) -> str:
    parts = [str(row["spec"])]
    for key in factors:
        value = row.get(key)
        if key == "spec" or value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        parts.append(setting_value(key, value))
    return " · ".join(parts)


def _pooled_seed_std(group: pd.DataFrame) -> float | None:
    """Spread between seeds of one configuration, pooled over the configurations with ≥ 2 seeds."""

    num = den = 0.0
    for _, cfg in group.groupby("label"):
        if len(cfg) > 1:
            num += float(cfg["mae"].var(ddof=1)) * (len(cfg) - 1)
            den += len(cfg) - 1
    return math.sqrt(num / den) if den else None


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    return float(frame[column].mean()) if column in frame and frame[column].notna().any() else None


def level_order(exp: Experiment, factor: str, levels: list[Any]) -> list[Any]:
    """The values of one setting in a fixed order; the first is the reference the others are compared with.

    Model configurations and input sets follow the experiment YAML (the full set ``all`` first), numbers go
    up. Job order would flip the comparison from one model to the next."""

    config = exp.config or {}
    if factor == "spec":
        ref = [item if isinstance(item, str) else (item.get("label") or item.get("name"))
               for item in config.get("models") or []]
    elif factor == "input_set":
        ref = list(((config.get("data") or {}).get("input_sets") or {}).keys())
        if INPUT_REFERENCE in levels:
            ref = [INPUT_REFERENCE, *(r for r in ref if r != INPUT_REFERENCE)]
    else:
        try:
            return sorted(levels, key=float)
        except (TypeError, ValueError):
            ref = []
    rank = {v: i for i, v in enumerate(ref)}
    return sorted(levels, key=lambda lv: (rank.get(lv, len(rank)), levels.index(lv)))


def _orders(jobs: pd.DataFrame) -> tuple[dict[Any, int], dict[tuple[Any, Any], float]]:
    """Targets in job order, and each (target, model)'s best mean MAE (best model first in tables)."""

    targets = {t: i for i, t in enumerate(dict.fromkeys(jobs["target"]))}
    best = jobs.groupby(["target", "model", "label"])["mae"].mean().groupby(level=[0, 1]).min()
    return targets, {k: float(v) for k, v in best.items()}


def config_table(exp: Experiment) -> pd.DataFrame:
    """Every configuration of each (target, model) that tried more than one, with seed spread and regions:
    targets in job order, the model with the lowest MAE first, configurations by MAE."""

    jobs = job_table(exp)
    rows = []
    for (target, model), group in jobs.groupby(["target", "model"], sort=False):
        if group["label"].nunique() < 2:
            continue
        factors = _factors(group)
        best = group.groupby("label")["mae"].mean().idxmin()
        for label, cfg in group.groupby("label", sort=False):
            rows.append({
                "target": target, "model": model, "label": label, "setting": setting_text(cfg.iloc[0], factors),
                "seeds": len(cfg), "mae": float(cfg["mae"].mean()),
                "mae_std": float(cfg["mae"].std(ddof=1)) if len(cfg) > 1 else None,
                "mae_cycle": _mean(cfg, "mae_cycle"), "mae_other": _mean(cfg, "mae_other"),
                "fit_minutes": (_mean(cfg, "fit_seconds") or 0) / 60, "chosen": label == best,
            })
    table = pd.DataFrame(rows)
    if len(table):
        targets, best_mae = _orders(jobs)
        table["_t"] = table["target"].map(targets)
        table["_m"] = [best_mae[(t, m)] for t, m in zip(table["target"], table["model"])]
        table = (table.sort_values(["_t", "_m", "model", "mae"], kind="stable").drop(columns=["_t", "_m"])
                 .reset_index(drop=True))
    return table


def factor_effects(exp: Experiment) -> list[dict[str, Any]]:
    """For each (target, model) and each setting that took ≥ 2 values: the mean MAE per value over every other
    setting and seed, and whether the best value beats the others by more than the seed spread allows."""

    jobs = job_table(exp)
    out = []
    for (target, model), group in jobs.groupby(["target", "model"], sort=False):
        factors = _factors(group)
        for factor in factors:
            sub = group[group[factor].notna()]
            levels = level_order(exp, factor, list(dict.fromkeys(sub[factor].tolist())))
            if len(levels) < 2:
                continue
            stats = {lv: (float(sub.loc[sub[factor] == lv, "mae"].mean()), int((sub[factor] == lv).sum()))
                     for lv in levels}
            best = min(levels, key=lambda lv: stats[lv][0])
            sd = _pooled_seed_std(sub)
            consistent = _consistent(sub, factor, [f for f in factors if f != factor and sub[f].notna().all()],
                                     len(levels))
            if sd is None:
                verdict = "반복(seed) 없음, " + {True: "방향 일관", False: "방향 엇갈림", None: "판단 보류"}[consistent]
                effect = None
            else:
                clear = [lv for lv in levels if lv != best and _apart(stats, sd, lv, best)]
                if len(clear) == len(levels) - 1:
                    verdict, effect = f"효과 있음: {setting_value(factor, best)}가 가장 좋음", True
                elif clear:
                    verdict, effect = (f"일부 효과: {setting_value(factor, best)}가 "
                                       f"{', '.join(setting_value(factor, c) for c in clear)}보다 좋음"), True
                else:
                    verdict, effect = "불분명 (차이가 seed 편차 안)", False
            ref = levels[0]
            out.append({
                "target": target, "model": model, "factor": factor, "name": FACTOR_NAMES.get(factor, factor),
                "levels": [(lv, stats[lv][0], stats[lv][1]) for lv in levels], "ref": ref, "best": best,
                "seed_std": sd, "verdict": verdict, "effect": effect, "consistent": consistent,
                # improvement of each other value over the reference in %, and whether that pair is apart by more
                # than the seed spread allows (None without repeated seeds)
                "changes": [(lv, (stats[ref][0] - stats[lv][0]) / stats[ref][0] * 100,
                             None if sd is None else _apart(stats, sd, ref, lv)) for lv in levels[1:] if stats[ref][0]],
            })
    if out:
        targets, best_mae = _orders(jobs)
        factor_rank = {f: i for i, f in enumerate(FACTOR_NAMES)}
        out.sort(key=lambda ef: (targets.get(ef["target"], 0), factor_rank.get(ef["factor"], len(factor_rank)),
                                 best_mae.get((ef["target"], ef["model"]), 0.0)))
    return out


def effect_summary(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per setting change (reference → value) over every target and model of an experiment: the mean
    change in %, how many (target, model) pairs it averages, and how many of them were apart by more than the
    seed spread, within it, or not repeated. Largest change first."""

    groups: dict[tuple[str, Any, Any], list[tuple[str, str, float, bool | None]]] = {}
    for ef in effects:
        for lv, change, clear in ef["changes"]:
            groups.setdefault((ef["factor"], ef["ref"], lv), []).append((ef["target"], ef["model"], change, clear))
    rows = []
    for (factor, ref, lv), items in groups.items():
        n_clear = sum(c is True for *_, c in items)
        n_unclear = sum(c is False for *_, c in items)
        rows.append({
            "factor": factor, "ref": ref, "level": lv, "change": sum(c for _, _, c, _ in items) / len(items),
            "n": len(items), "targets": len({t for t, *_ in items}), "models": len({m for _, m, *_ in items}),
            "n_clear": n_clear, "n_unclear": n_unclear, "n_norepeat": len(items) - n_clear - n_unclear,
            "status": "clear" if n_clear > n_unclear else ("unclear" if n_clear + n_unclear else "norepeat"),
        })
    rows.sort(key=lambda r: -abs(r["change"]))
    return rows


def _apart(stats: dict[Any, tuple[float, int]], sd: float, a: Any, b: Any) -> bool:
    """Are the mean MAEs of two values further apart than CLEAR_SE standard errors of the seed spread?"""

    return abs(stats[a][0] - stats[b][0]) > CLEAR_SE * sd * math.sqrt(1 / stats[a][1] + 1 / stats[b][1])


def _consistent(sub: pd.DataFrame, factor: str, others: list[str], n_levels: int) -> bool | None:
    """Does the same value win under every combination of the other settings? (None: nothing to compare)"""

    if not others:
        return None
    winners = []
    for combo in product(*(list(dict.fromkeys(sub[f])) for f in others)):
        mask = pd.Series(True, index=sub.index)
        for f, value in zip(others, combo):
            mask &= sub[f] == value
        part = sub[mask].groupby(factor)["mae"].mean()
        if len(part) == n_levels:
            winners.append(part.idxmin())
    return len(set(winners)) == 1 if len(winners) > 1 else None


def speed_table(exps: list[Experiment]) -> pd.DataFrame:
    """Per deep model configuration family (``patchtst_mix``, ``dlinear``, …) over every experiment: how many jobs,
    windows per second, minutes per job and peak GPU memory (medians, memory as the maximum)."""

    frames = []
    for exp in exps:
        results = exp.results
        if "windows_per_second" not in results:
            continue
        ok = results[results["status"] == "ok"] if "status" in results else results
        ok = ok[ok["windows_per_second"].notna()]
        if len(ok):
            frames.append(ok.assign(experiment=exp.name))
    if not frames:
        return pd.DataFrame(columns=["spec", "model", "jobs", "experiments", "wps", "minutes", "peak_mb", "params"])
    frame = pd.concat(frames, ignore_index=True)
    frame["spec"] = frame["label"].astype(str).str.split("[").str[0].str.split("@").str[0]
    rows = []
    for spec, group in frame.groupby("spec", sort=False):
        rows.append({
            "spec": spec, "model": str(group["model"].iloc[0]), "jobs": len(group),
            "experiments": ", ".join(dict.fromkeys(group["experiment"])),
            "wps": float(group["windows_per_second"].median()),
            "minutes": float(group["fit_seconds"].median()) / 60 if "fit_seconds" in group else None,
            "peak_mb": float(group["peak_gpu_mb"].max()) if "peak_gpu_mb" in group and group["peak_gpu_mb"].notna().any()
            else None,
            "params": float(group["n_params"].median()) if "n_params" in group and group["n_params"].notna().any()
            else None,
        })
    return pd.DataFrame(rows).sort_values("wps", ascending=False).reset_index(drop=True)


def job_timeline(exp: Experiment, journal: pd.DataFrame) -> pd.DataFrame:
    """Every training job of this experiment run with its device and start/end in seconds from the first start
    (from the journal, which records each job with the device it ran on)."""

    columns = ["device", "model", "label", "title", "status", "start_s", "end_s"]
    if journal is None or not len(journal) or "artifacts" not in journal:
        return pd.DataFrame(columns=columns)
    marker = f"{exp.name}/{exp.stamp}/jobs"
    rows = []
    for entry in journal.itertuples():
        if entry.stage not in ("train", "hpo") or entry.duration_s is None or pd.isna(entry.duration_s):
            continue
        artifacts = entry.artifacts if isinstance(entry.artifacts, dict) else {}
        if marker not in str(artifacts.get("job_dir", "")).replace("\\", "/"):
            continue
        config = entry.config if isinstance(entry.config, dict) else {}
        rows.append({"device": str(config.get("device", "?")), "model": str(config.get("model", "")),
                     "label": str(config.get("label", "")), "title": str(entry.title), "status": str(entry.status),
                     "start": entry.started_at, "seconds": float(entry.duration_s)})
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame["start_s"] = (frame["start"] - frame["start"].min()).dt.total_seconds()
    frame["end_s"] = frame["start_s"] + frame["seconds"]
    return frame.sort_values("start_s").reset_index(drop=True)[columns + ["seconds"]]


def device_order(device: str) -> tuple[int, int]:
    """cuda:0 < cuda:1 < … < cpu."""

    match = re.match(r"cuda:(\d+)$", device)
    return (0, int(match.group(1))) if match else (1, 0)
