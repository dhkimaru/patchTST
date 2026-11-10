"""Baselines and classical models on the same windows as the deep models.

- ``persistence``: the target's last observed value (forecast only).
- ``template``: the recipe template — mean target per (recipe step, position
  in step) over train runs. It knows the recipe, not the equipment state,
  so beating it means a model uses real process information. For a forecast it
  reads the step that is actually running at the target time, i.e. it knows the
  future step timing.
- ``schedule``: the same template, but the step at the target time is projected
  from the current step with the typical step durations and step order of the
  train runs (a recipe plan known in advance), so it does not see future timing.
- ``ridge`` and ``lightgbm``: regressors on window summary features
  (last, mean, std, min, max, recent change of every input channel). By
  default they learn the *correction* to the task baseline (the change from
  the last value for forecasts, the deviation from the recipe template for
  virtual sensors), which trees cannot extrapolate on their own.

All predict standardized ``(n, H, T)`` arrays like the deep models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..training.prepare import Bundle
from ..training.windows import WindowSet, evenly_spaced
from . import defaults

FEATURES_PER_CHANNEL = ("last", "mean", "std", "min", "max", "delta")
CLASSIC_MODELS = ("persistence", "template", "schedule", "ridge", "lightgbm")


def window_features(x: torch.Tensor) -> torch.Tensor:
    """``(B, L, C)`` → ``(B, 6C)`` summary features."""

    last = x[:, -1]
    mid = x[:, x.shape[1] // 2]
    return torch.cat(
        [last, x.mean(1), x.std(1, unbiased=False), x.amin(1), x.amax(1), last - mid], dim=1
    )


def feature_matrix(ws: WindowSet, idx: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for _, tidx in ws.iter_batches(idx, batch_size):
            x, _ = ws.gather(tidx)
            parts.append(window_features(x).float().cpu().numpy())
    return np.concatenate(parts)


def target_matrix(ws: WindowSet, idx: np.ndarray) -> np.ndarray:
    rows = torch.as_tensor(ws.target_rows(idx), device=ws.device)
    return ws.y[rows].cpu().numpy()  # (n, H, T)


class RecipeTemplate:
    """Mean of each target per (step_key, step_pos) over train rows, with fallbacks."""

    def __init__(self) -> None:
        self.table: pd.DataFrame | None = None
        self.step_mean: pd.DataFrame | None = None
        self.overall: np.ndarray | None = None

    def fit(self, bundle: Bundle, y: np.ndarray) -> RecipeTemplate:
        frame = pd.DataFrame(y, columns=[f"t{i}" for i in range(y.shape[1])])
        frame["key"], frame["pos"] = np.asarray(bundle.step_key), np.asarray(bundle.step_pos)
        cols = [c for c in frame.columns if c.startswith("t")]
        self.table = frame.groupby(["key", "pos"])[cols].mean()
        self.step_mean = frame.groupby("key")[cols].mean()
        self.overall = frame[cols].mean().to_numpy()
        return self

    def values(self, bundle: Bundle) -> np.ndarray:
        return self.lookup(np.asarray(bundle.step_key), np.asarray(bundle.step_pos))

    def lookup(self, key: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Template values ``(n, T)`` at any (step key, position) pairs, with the same fallbacks."""

        assert self.table is not None and self.step_mean is not None and self.overall is not None
        keys = pd.DataFrame({"key": np.asarray(key, dtype=np.int64), "pos": np.asarray(pos).astype(np.int32)})
        exact = keys.join(self.table, on=["key", "pos"])
        cols = list(self.table.columns)
        out = exact[cols].to_numpy(np.float64)
        missing = np.isnan(out).any(axis=1)
        if missing.any():
            by_step = keys.loc[missing, ["key"]].join(self.step_mean, on="key")[cols].to_numpy(np.float64)
            out[missing] = by_step
        still = np.isnan(out)
        out[still] = np.broadcast_to(self.overall, out.shape)[still]
        return out.astype(np.float32)

    def save(self, directory: Path) -> None:
        assert self.table is not None and self.step_mean is not None and self.overall is not None
        self.table.reset_index().to_parquet(directory / "template.parquet", index=False)
        self.step_mean.reset_index().to_parquet(directory / "template_step.parquet", index=False)
        (directory / "template_overall.json").write_text(json.dumps(self.overall.tolist()))

    @classmethod
    def load(cls, directory: Path) -> RecipeTemplate:
        out = cls()
        out.table = pd.read_parquet(directory / "template.parquet").set_index(["key", "pos"])
        out.step_mean = pd.read_parquet(directory / "template_step.parquet").set_index("key")
        out.overall = np.asarray(json.loads((directory / "template_overall.json").read_text()))
        return out


class ClassicModel:
    name = "classic"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        self.info: dict[str, Any] = {}

    def fit(self, train: WindowSet, val: WindowSet, ctx: dict[str, Any]) -> dict[str, Any]:
        return {}

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        (directory / "classic.json").write_text(json.dumps({"name": self.name, "params": self.params,
                                                            "info": self.info}, default=str))

    def load_state(self, directory: Path) -> None:
        doc = json.loads((directory / "classic.json").read_text())
        self.info = doc.get("info", {})
        self.residual = self.info.get("residual", "none")


class Persistence(ClassicModel):
    name = "persistence"

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        channels = ctx.get("target_channels")
        if channels is None:
            raise ValueError("persistence needs the target among the inputs (forecast task)")
        last = ws.x[torch.as_tensor(ws.ends_np[idx], device=ws.device)][:, channels].cpu().numpy()
        return np.repeat(last[:, None, :], len(ws.horizons), axis=1)


class Template(ClassicModel):
    name = "template"

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        template = getattr(ws, "template", None)
        if template is None:
            raise ValueError("template values were not attached to this window set")
        return template[ws.target_rows(idx)]


class StepSchedule:
    """Typical duration and next step of every recipe step key, from the train runs.

    ``future`` projects (step key, position) forward by a number of samples: the current step lasts its typical
    duration, then the most frequent next step follows with its typical duration, and so on. A step that has
    already run longer than typical ends now; a step never seen ending lasts forever.
    """

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)  # step_pos is capped at this many samples in the bundles
        self.duration: dict[int, int] = {}
        self.next: dict[int, int] = {}
        self.info: dict[str, Any] = {}

    def fit(self, bundle: Bundle) -> StepSchedule:
        key = np.asarray(bundle.step_key, dtype=np.int64)
        seg = np.asarray(bundle.seg_id)
        change = np.r_[True, (key[1:] != key[:-1]) | (seg[1:] != seg[:-1])]
        starts = np.flatnonzero(change)
        lengths = np.diff(np.r_[starts, len(key)])
        rows = []
        for i in range(len(starts) - 1):
            a, b = starts[i], starts[i + 1]
            if seg[a] != seg[b] or i == 0 or seg[starts[i - 1]] != seg[a]:
                continue  # first or last step of a segment: its true length or successor is unknown
            rows.append((int(key[a]), int(lengths[i]), int(key[b])))
        frame = pd.DataFrame(rows, columns=["key", "length", "next"])
        if len(frame):
            self.duration = {int(k): int(round(v)) for k, v in frame.groupby("key")["length"].median().items()}
            self.next = {int(k): int(g["next"].mode().iloc[0]) for k, g in frame.groupby("key")}
            spread = frame.groupby("key")["length"].agg(lambda v: float(np.percentile(v, 90) - np.percentile(v, 10)))
            weight = frame.groupby("key")["length"].sum()
            fixed = spread[spread <= 1].index
            self.info = {"steps": len(self.duration), "fixed_duration_steps": int(len(fixed)),
                         "fixed_duration_share": round(float(weight[fixed].sum() / weight.sum()), 4)}
        return self

    def future(self, key: np.ndarray, pos: np.ndarray, ahead: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(key, pos) ``(n, H)`` after ``ahead`` ``(H,)`` samples from each (key, pos) ``(n,)``."""

        key = np.asarray(key, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        ahead = np.asarray(ahead, dtype=np.int64)
        span = int(ahead.max()) + 1 if len(ahead) else 1
        known = sorted(k for k in self.duration if k in self.next)
        index = {k: i for i, k in enumerate(known)}
        # after_key[i, t], after_pos[i, t]: where the plan is t samples after step known[i] ends
        after_key = np.empty((len(known), span), dtype=np.int64)
        after_pos = np.empty((len(known), span), dtype=np.int64)
        for i, k in enumerate(known):
            t, cur = 0, self.next.get(k)
            while t < span:
                if cur is None:
                    break
                d = self.duration.get(cur)
                n = span - t if d is None or cur not in self.next else min(max(d, 1), span - t)
                after_key[i, t:t + n] = cur
                after_pos[i, t:t + n] = np.arange(n)
                t += n
                cur = self.next.get(cur) if d is not None else None
            if t < span:  # chain ended on a step never seen ending: it lasts
                last = after_key[i, t - 1] if t else k
                base = after_pos[i, t - 1] + 1 if t else 0
                after_key[i, t:] = last
                after_pos[i, t:] = base + np.arange(span - t)
        row = pd.Series(key).map(index).to_numpy()
        has = ~pd.isna(row)
        d = np.full(len(key), -1, dtype=np.int64)
        d[has] = np.asarray([self.duration[k] for k in key[has]], dtype=np.int64)
        remaining = np.maximum(d - pos, 0)[:, None]
        h = ahead[None, :]
        stay = (~has)[:, None] | (h < remaining)
        out_key = np.broadcast_to(key[:, None], stay.shape).copy()
        out_pos = (pos[:, None] + h).copy()
        moved = ~stay
        if moved.any():
            r = np.broadcast_to(np.where(has, row, 0).astype(np.int64)[:, None], stay.shape)[moved]
            t = (h - remaining)[moved]
            out_key[moved] = after_key[r, t]
            out_pos[moved] = after_pos[r, t]
        return out_key, np.minimum(out_pos, self.cap)

    def save(self, directory: Path) -> None:
        (directory / "schedule.json").write_text(json.dumps(
            {"cap": self.cap, "duration": {str(k): v for k, v in self.duration.items()},
             "next": {str(k): v for k, v in self.next.items()}, "info": self.info}))

    @classmethod
    def load(cls, directory: Path) -> StepSchedule:
        doc = json.loads((directory / "schedule.json").read_text())
        out = cls(doc["cap"])
        out.duration = {int(k): int(v) for k, v in doc["duration"].items()}
        out.next = {int(k): int(v) for k, v in doc["next"].items()}
        out.info = doc.get("info", {})
        return out


class Schedule(ClassicModel):
    """Recipe template at the step the recipe plan puts at the target time (no future timing from the trace)."""

    name = "schedule"

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        recipe, plan = getattr(ws, "recipe", None), getattr(ws, "schedule", None)
        if recipe is None or plan is None:
            raise ValueError("the recipe template and step schedule were not attached to this window set")
        ends = ws.ends_np[idx]
        keys, pos = plan.future(np.asarray(ws.step_key)[ends], np.asarray(ws.step_pos)[ends], np.asarray(ws.horizons))
        values = recipe.lookup(keys.reshape(-1), pos.reshape(-1))
        return values.reshape(len(idx), len(ws.horizons), -1).astype(np.float32)


def residual_mode(params: dict[str, Any], ws: WindowSet, ctx: dict[str, Any]) -> str:
    mode = str(params.get("residual", "auto"))
    if mode != "auto":
        return mode
    if ctx.get("target_channels") is not None:
        return "persistence"
    if getattr(ws, "template", None) is not None:
        return "template"
    return "none"


def base_prediction(mode: str, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray | float:
    if mode == "persistence":
        return Persistence().predict(ws, idx, ctx)
    if mode == "template":
        return Template().predict(ws, idx, ctx)
    return 0.0


def _train_sample(ws: WindowSet, limit: int, seed: int) -> np.ndarray:
    if len(ws) <= limit:
        return np.arange(len(ws))
    return np.sort(np.random.default_rng(seed).choice(len(ws), size=limit, replace=False))


class Ridge(ClassicModel):
    """Closed-form ridge on standardized window features; alpha picked on validation.

    Solves ``(XᵀX/n + αI) w = Xᵀy/n``, so α is on the scale of feature correlations.
    """

    name = "ridge"
    ALPHAS = defaults.RIDGE_ALPHAS

    def fit(self, train: WindowSet, val: WindowSet, ctx: dict[str, Any]) -> dict[str, Any]:
        limit = int(self.params.get("max_train_windows", 300_000))
        tidx = _train_sample(train, limit, ctx.get("seed", 0))
        vidx = evenly_spaced(len(val), int(self.params.get("max_val_windows", 100_000)))
        self.residual = residual_mode(self.params, train, ctx)
        xt = feature_matrix(train, tidx).astype(np.float64)
        yt = (target_matrix(train, tidx) - base_prediction(self.residual, train, tidx, ctx)).reshape(len(tidx), -1)
        yt = yt.astype(np.float64)
        self.mu, self.sd = xt.mean(0), xt.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        xt = (xt - self.mu) / self.sd
        self.y_mu = yt.mean(0)
        gram, rhs = xt.T @ xt / len(xt), xt.T @ (yt - self.y_mu) / len(xt)
        xv = (feature_matrix(val, vidx).astype(np.float64) - self.mu) / self.sd
        yv = (target_matrix(val, vidx) - base_prediction(self.residual, val, vidx, ctx)).reshape(len(vidx), -1)
        alphas = [float(a) for a in self.params.get("alphas", self.ALPHAS)]
        scores = {}
        best = None
        for alpha in alphas:
            w = np.linalg.solve(gram + alpha * np.eye(len(gram)), rhs)
            mse = float(np.mean((xv @ w + self.y_mu - yv) ** 2))
            scores[alpha] = mse
            if best is None or mse < best[0]:
                best = (mse, alpha, w)
        assert best is not None
        self.w, self.alpha = best[2], best[1]
        self.info = {"alpha": self.alpha, "val_mse_by_alpha": scores, "train_windows": len(tidx),
                     "features": int(xt.shape[1]), "residual": self.residual}
        return self.info

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        out = []
        for start in range(0, len(idx), 65_536):
            part = idx[start : start + 65_536]
            x = (feature_matrix(ws, part).astype(np.float64) - self.mu) / self.sd
            out.append(x @ self.w + self.y_mu)
        pred = np.concatenate(out).reshape(len(idx), len(ws.horizons), -1)
        return (pred + base_prediction(self.residual, ws, idx, ctx)).astype(np.float32)

    def save(self, directory: Path) -> None:
        super().save(directory)
        np.savez(directory / "ridge.npz", w=self.w, mu=self.mu, sd=self.sd, y_mu=self.y_mu)

    def load_state(self, directory: Path) -> None:
        super().load_state(directory)
        state = np.load(directory / "ridge.npz")
        self.w, self.mu, self.sd, self.y_mu = state["w"], state["mu"], state["sd"], state["y_mu"]


class LightGBM(ClassicModel):
    """Gradient-boosted trees (LightGBM; sklearn HistGradientBoosting if LightGBM is missing)."""

    name = "lightgbm"
    DEFAULTS = dict(defaults.LIGHTGBM)

    def fit(self, train: WindowSet, val: WindowSet, ctx: dict[str, Any]) -> dict[str, Any]:
        p = {**self.DEFAULTS, **self.params}
        tidx = _train_sample(train, int(p["max_train_windows"]), ctx.get("seed", 0))
        vidx = evenly_spaced(len(val), int(p["max_val_windows"]))
        self.residual = residual_mode(self.params, train, ctx)
        xt, xv = feature_matrix(train, tidx), feature_matrix(val, vidx)
        yt = (target_matrix(train, tidx) - base_prediction(self.residual, train, tidx, ctx)).reshape(len(tidx), -1)
        yv = (target_matrix(val, vidx) - base_prediction(self.residual, val, vidx, ctx)).reshape(len(vidx), -1)
        n_jobs = int(ctx.get("n_jobs") or os.cpu_count() or 1)
        self.models: list[Any] = []
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None
        iterations = []
        for j in range(yt.shape[1]):
            if lgb is not None:
                model = lgb.LGBMRegressor(
                    objective=str(p["objective"]), n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]),
                    num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child_samples"]),
                    subsample=float(p["subsample"]), subsample_freq=int(p["subsample_freq"]),
                    colsample_bytree=float(p["colsample_bytree"]), n_jobs=n_jobs,
                    random_state=int(ctx.get("seed", 0)), verbose=-1,
                )
                model.fit(xt, np.ascontiguousarray(yt[:, j]), eval_set=[(xv, np.ascontiguousarray(yv[:, j]))],
                          callbacks=[lgb.early_stopping(int(p["early_stopping_rounds"]), verbose=False)])
                iterations.append(int(model.best_iteration_ or p["n_estimators"]))
                self.backend = "lightgbm"
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor

                loss = {"l1": "absolute_error", "huber": "squared_error"}.get(str(p["objective"]), "squared_error")
                model = HistGradientBoostingRegressor(
                    loss=loss, max_iter=min(int(p["n_estimators"]), 500), learning_rate=float(p["learning_rate"]),
                    max_leaf_nodes=int(p["num_leaves"]), min_samples_leaf=int(p["min_child_samples"]),
                    early_stopping=True, random_state=int(ctx.get("seed", 0)),
                )
                model.fit(xt, np.ascontiguousarray(yt[:, j]))
                iterations.append(int(model.n_iter_))
                self.backend = "sklearn_hgb"
            self.models.append(model)
        self.info = {"backend": self.backend, "iterations": iterations, "train_windows": len(tidx),
                     "features": int(xt.shape[1]), "residual": self.residual}
        return self.info

    def predict(self, ws: WindowSet, idx: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        out = []
        for start in range(0, len(idx), 65_536):
            x = feature_matrix(ws, idx[start : start + 65_536])
            out.append(np.stack([m.predict(x) for m in self.models], axis=1))
        pred = np.concatenate(out).reshape(len(idx), len(ws.horizons), -1)
        return (pred + base_prediction(self.residual, ws, idx, ctx)).astype(np.float32)

    def save(self, directory: Path) -> None:
        super().save(directory)
        import joblib

        joblib.dump(self.models, directory / "lightgbm.joblib")

    def load_state(self, directory: Path) -> None:
        super().load_state(directory)
        import joblib

        self.models = joblib.load(directory / "lightgbm.joblib")


CLASSIC_REGISTRY: dict[str, type[ClassicModel]] = {
    "persistence": Persistence,
    "template": Template,
    "schedule": Schedule,
    "ridge": Ridge,
    "lightgbm": LightGBM,
}
