"""두께 예측(VM) 설계 전 데이터 점검: 예측할 만한 run 간 차이가 있는지, 무엇과 함께 움직이는지 본다. 학습 없음.

학습 + 검증 run만 쓴다(시험 run 봉인). 이상 run은 g2_thickness.yaml의 vm.exclude_runs대로 뺀다.
1. 대상 요약: 49점 배치 run의 두께 평균 · U/F · WtoW · 중심 · 가장자리의 평균, 표준편차, 범위.
2. 측정 신뢰도(split-half): run마다 측정점을 둘로 나눠 run 평균을 두 번 내고, run 사이 상관 r에서 Spearman-Brown 신뢰도
   2r/(1+r)를 구한다. run 간 차이 중 측정점 표집 잡음이 아닌 몫이며, 어떤 모델의 R²도 이 값을 넘기 어렵다.
   측정 장비의 반복 오차처럼 한 run의 모든 점에 같이 들어가는 잡음은 이 값에 잡히지 않는다(상한을 높게 잡는 쪽).
   점 나누기(같은 WAP 안에서 반씩, 무작위 200번 평균)와 WAP 나누기(1 · 3 · 5 대 2 · 4) 두 가지로 본다.
3. 시간 구조: run 번호와의 순위 상관, 바로 앞 run과의 상관, 기준선별 MAE(학습 평균 leave-one-out, 이전 run들의 평균,
   바로 앞 run 값, EWMA 0.3). 앞 run의 측정값을 쓸 수 있다면 VM이 넘어야 할 기준선이 학습 평균이 아니게 된다.
4. 사이클 수: trace에서 센 사이클 수별 두께와 사이클당 두께.
5. run 사이 대기 시간: 앞 run(시간 순, 모든 run) 공정 끝부터 이번 run 공정 시작까지(시간)와 두께의 순위 상관.
6. run 요약 특성(g2_thickness와 같은 특성)과 대상의 순위 상관 상위 10개, 그리고 대상 순서를 섞었을 때 나오는 최대 |r|의
   95% 분위(우연 수준). 상위 특성이 우연 수준을 넘지 못하면 그 특성들로는 run 간 차이를 설명한다고 말할 수 없다.

    python scripts/vm_audit.py                  # → work/reports/vm_audit.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.data.splits import load_split
from ald_lab.paths import resolve_paths
from ald_lab.training.config import load_experiment
from ald_lab.training.vm import DEFAULT_EXCLUDED_RUNS, TARGET_UNITS, _dataset

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "g2_thickness.yaml"),
                help="두께 예측 실험 YAML (분할, 제외 run, 입력 세트)")
ap.add_argument("--input-set", default="all", help="6번 특성에 쓸 입력 세트 (기본 all)")
ap.add_argument("--reps", type=int, default=200, help="무작위 반복 수 (점 나누기, 순서 섞기)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

P = resolve_paths()
cfg = load_experiment(args.config)
split = load_split(cfg.split_path())
excluded = {str(r) for r in cfg.vm.get("exclude_runs", DEFAULT_EXCLUDED_RUNS)}
rng = np.random.default_rng(args.seed)
blocks = pd.read_parquet(P.overview_dir / "exsitu_blocks.parquet")
blocks = blocks.dropna(subset=["run_id"])
blocks = blocks[blocks["layout"] == "49pt"].drop_duplicates("run_id").set_index("run_id")
fit = [r for r in split.runs("train", "validation") if r in blocks.index and r not in excluded]
b = blocks.loc[fit]
out = ["## 두께 예측(VM) 데이터 점검 (학습 없음, 학습 + 검증 run, 시험 run 봉인)",
       f"run {len(fit)}개 (제외: {', '.join(sorted(excluded)) or '없음'}). 상관은 모두 순위 상관(Spearman)이다.", ""]
TARGETS = [t for t in ("thk_mean", "uf_pct_mean", "wtow_pct", "thk_center", "thk_edge", "edge_vs_center_pct") if t in b]


def spearman(a, c) -> float:
    frame = pd.DataFrame({"a": np.asarray(a, dtype=float), "c": np.asarray(c, dtype=float)}).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["c"].nunique() < 2:
        return float("nan")
    return float(frame["a"].rank().corr(frame["c"].rank()))


def brown(r: float) -> float:
    """Spearman-Brown reliability of the full mean from the correlation of two halves (0 when the halves disagree)."""

    return 2 * r / (1 + r) if r > 0 else 0.0


# 1. targets
rows = []
for t in TARGETS:
    v = b[t].astype(float).dropna()
    rows.append({"대상": t, "단위": TARGET_UNITS.get(t, ""), "run": len(v), "평균": f"{v.mean():.4g}", "표준편차": f"{v.std():.3g}",
                 "최소": f"{v.min():.4g}", "최대": f"{v.max():.4g}", "표준편차 ÷ 평균 (%)": round(100 * v.std() / abs(v.mean()), 2)})
out += ["[1. 대상 요약]", pd.DataFrame(rows).to_string(index=False), ""]

# 2. split-half reliability of the run mean (and U/F by WAP halves)
points_path = P.overview_dir / "exsitu_points.parquet"
if points_path.is_file() and "block_id" in b:
    pts = pd.read_parquet(points_path).dropna(subset=["thickness_a"])
    pts = pts.merge(b[["block_id"]].reset_index(), on="block_id")
    per_run = {r: g for r, g in pts.groupby("run_id")}
    runs = [r for r in fit if r in per_run]

    def half_means(g: pd.DataFrame, pick: dict[int, np.ndarray]) -> tuple[float, float]:
        a, c = [], []
        for wap, w in g.groupby("wap_index"):
            vals = w.sort_values("point")["thickness_a"].to_numpy(float)
            mask = pick[int(wap)][: len(vals)]
            a.append(vals[mask].mean())
            c.append(vals[~mask].mean())
        return float(np.mean(a)), float(np.mean(c))

    rs, noise = [], []
    n_pts = int(pts.groupby(["run_id", "wap_index"]).size().max())
    for _ in range(args.reps):
        pick = {w: rng.permutation(n_pts) < n_pts // 2 for w in range(1, 6)}
        halves = np.array([half_means(per_run[r], pick) for r in runs])
        rs.append(np.corrcoef(halves[:, 0], halves[:, 1])[0, 1])
        noise.append(np.std(halves[:, 0] - halves[:, 1], ddof=1) / 2)
    r_pt = float(np.mean(rs))
    wap_mean = pts.groupby(["run_id", "wap_index"])["thickness_a"].mean().unstack()
    wap_uf = pts.groupby(["run_id", "wap_index"])["thickness_a"].agg(lambda v: (v.max() - v.min()) / (2 * v.mean()) * 100).unstack()
    rows = []
    sd = float(b.loc[runs, "thk_mean"].std())
    rows.append({"대상": "thk_mean", "나누기": "점 (같은 WAP 안에서 반씩)", "반쪽 상관 r": round(r_pt, 3),
                 "신뢰도 2r/(1+r)": round(brown(r_pt), 3), "run 간 표준편차": f"{sd:.3g}",
                 "점 표집 잡음 표준편차": f"{np.mean(noise):.3g}"})
    for name, table, col in (("thk_mean", wap_mean, "thk_mean"), ("uf_pct_mean", wap_uf, "uf_pct_mean")):
        t = table.loc[[r for r in runs if r in table.index]]
        odd, even = t[[c for c in (1, 3, 5) if c in t]].mean(axis=1), t[[c for c in (2, 4) if c in t]].mean(axis=1)
        r = float(np.corrcoef(odd, even)[0, 1])
        rows.append({"대상": name, "나누기": "WAP (1 · 3 · 5 대 2 · 4)", "반쪽 상관 r": round(r, 3),
                     "신뢰도 2r/(1+r)": round(brown(r), 3), "run 간 표준편차": f"{b.loc[runs, col].std():.3g}",
                     "점 표집 잡음 표준편차": f"{np.std(odd - even, ddof=1) / 2:.3g}"})
    out += ["[2. 측정 신뢰도 (split-half)] 신뢰도는 run 간 차이 중 측정점 표집 잡음이 아닌 몫. 모델 R²의 상한으로 읽는다.",
            "WAP 나누기의 잡음에는 boat 위치 차이도 들어가므로, 점 나누기보다 신뢰도가 낮게 나오는 것이 보통이다.",
            pd.DataFrame(rows).to_string(index=False), ""]
else:
    out += ["[2. 측정 신뢰도] exsitu_points.parquet 또는 block_id가 없어 건너뜀 (ald-build-overview --exsitu 필요)", ""]

# 3. time structure and baselines that use earlier runs' metrology
catalog_path = P.overview_dir / "runs.parquet"
catalog = pd.read_parquet(catalog_path) if catalog_path.is_file() else None
order = sorted(fit, key=lambda r: int(str(r).rsplit("_", 1)[-1]))
rows = []
for t in [t for t in ("thk_mean", "uf_pct_mean", "wtow_pct") if t in b]:
    y = b.loc[order, t].astype(float).to_numpy()
    n = len(y)
    loo = np.array([(y.sum() - y[i]) / (n - 1) for i in range(n)])
    expanding = np.array([np.nan, *[y[:i].mean() for i in range(1, n)]])
    previous = np.r_[np.nan, y[:-1]]
    ewma = np.full(n, np.nan)
    level = y[0]
    for i in range(1, n):
        ewma[i] = level
        level = 0.3 * y[i] + 0.7 * level
    keep = np.arange(n) >= 2  # every baseline has at least two earlier runs
    mae = {name: float(np.abs(p[keep] - y[keep]).mean()) for name, p in
           (("학습 평균 (leave-one-out)", loo), ("이전 run들의 평균", expanding), ("바로 앞 run", previous), ("EWMA 0.3", ewma))}
    base = mae["학습 평균 (leave-one-out)"]
    rows.append({"대상": t, "단위": TARGET_UNITS.get(t, ""), "run 번호 순위 상관": round(spearman(np.arange(n), y), 2),
                 "바로 앞 run과 상관": round(float(np.corrcoef(y[1:], y[:-1])[0, 1]), 2),
                 **{f"{k} MAE": f"{v:.3g}" for k, v in mae.items()},
                 "EWMA ÷ 학습 평균 - 1 (%)": round(100 * (mae["EWMA 0.3"] / base - 1), 1)})
out += ["[3. 시간 구조] run 번호 순서(학습 + 검증 run만). MAE는 세 번째 run부터 같은 run들에서 잼. 음수 %면 앞 run 측정값을 쓰는 쪽이 낫다.",
        pd.DataFrame(rows).to_string(index=False), ""]

# 4. cycle count and 5. idle time between runs (trace catalog)
if catalog is not None:
    cat = catalog.set_index("run_id")
    if "n_cycles" in cat:
        g = b[["thk_mean"]].join(cat["n_cycles"]).dropna()
        g["per_cycle"] = g["thk_mean"] / g["n_cycles"]
        tab = g.groupby("n_cycles").agg(run=("thk_mean", "size"), 두께_평균=("thk_mean", "mean"), 두께_표준편차=("thk_mean", "std"),
                                         사이클당_두께=("per_cycle", "mean")).reset_index()
        out += ["[4. 사이클 수별 두께] 사이클 수는 trace에서 센 값. 두께 단위 Å, 사이클당 두께 Å/cycle.",
                tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"), ""]
    if {"process_start", "process_end"} <= set(cat.columns):
        timeline = cat.dropna(subset=["process_start"]).sort_values("process_start")
        idle = (timeline["process_start"] - timeline["process_end"].shift()).dt.total_seconds() / 3600
        idle = idle.reindex(fit).astype(float)
        rows = [{"대상": t, "대기 시간과 순위 상관": round(spearman(idle.to_numpy(), b.loc[fit, t].to_numpy()), 2)}
                for t in ("thk_mean", "uf_pct_mean", "wtow_pct") if t in b]
        q = idle.dropna().quantile([0, 0.5, 0.9, 1.0])
        out += [(f"[5. run 사이 대기 시간] 앞 run 공정 끝 ~ 이번 run 공정 시작. 최소 {q.iloc[0]:.2g} · 중앙값 {q.iloc[1]:.2g} · "
                 f"90% {q.iloc[2]:.2g} · 최대 {q.iloc[3]:.2g} 시간"), pd.DataFrame(rows).to_string(index=False), ""]

# 6. run-summary features vs targets, with the chance level of the best |r|
x, y, inputs, _ = _dataset(cfg, P, split, ("train", "validation"), input_set=args.input_set)
x = x.loc[[r for r in x.index if r in b.index]]
xr = x.apply(lambda c: c.fillna(c.median())).rank()
xr = (xr - xr.mean()) / xr.std(ddof=0).replace(0, np.nan)
xr = xr.dropna(axis=1)
out += [(f"[6. run 요약 특성과의 순위 상관] 신호 {len(inputs)}개 → 특성 {xr.shape[1]}개, run {len(xr)}개. "
         f"우연 수준 = 대상 순서를 {args.reps}번 섞었을 때 특성 전체에서 나온 최대 |r|의 95% 분위.")]
for t in [t for t in ("thk_mean", "uf_pct_mean", "wtow_pct") if t in b]:
    yy = b.loc[xr.index, t].astype(float)
    ok = yy.notna().to_numpy()
    X, yv = xr.to_numpy()[ok], yy.to_numpy()[ok]
    yr = pd.Series(yv).rank().to_numpy()
    yr = (yr - yr.mean()) / yr.std()
    r = X.T @ yr / len(yr)
    perm = np.stack([rng.permutation(yr) for _ in range(args.reps)], axis=1)
    chance = float(np.quantile(np.abs(X.T @ perm / len(yr)).max(axis=0), 0.95))
    top = np.argsort(-np.abs(r))[:10]
    out.append(f"\n{t}: 최대 |r| {np.abs(r).max():.2f}, 우연 수준 {chance:.2f}, 우연 수준을 넘는 특성 {int((np.abs(r) > chance).sum())}개")
    out.append(pd.DataFrame({"특성": xr.columns[top], "순위 상관": np.round(r[top], 2)}).to_string(index=False))
text = "\n".join(out)
print(text)
report = P.work_root / "reports" / "vm_audit.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(text + "\n", encoding="utf-8")
print(f"\n저장: {report}")
