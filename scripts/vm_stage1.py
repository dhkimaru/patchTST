"""두께 예측(VM) 1단계: trace 특성이 앞 run 측정값(EWMA)보다 더 알려 주는지, 시간 순서대로 확인한다. 학습은 CPU에서 수 분.

데이터 점검(vm_audit.txt)에서 run 간 두께 차이는 측정 잡음이 아니었고(split-half 신뢰도 0.95 이상), 앞 run 측정값의 EWMA가
학습 평균보다 확실히 나았다. 그래서 VM이 넘어야 할 기준선을 EWMA로 두고, 운영과 같은 순서로 평가한다.

시간순 평가: 학습 + 검증 run을 run 번호 순으로 놓고, run i를 그 앞의 run만으로 학습한 모델로 예측한다(처음 --start개는 학습만).
- 이전 run 평균, EWMA 0.3: 앞 run 측정값만 쓰는 기준선
- Ridge(상위 k개 특성), LightGBM(seed 3개 평균): trace 특성만 쓰는 모델. 특성 선택(순위 상관 상위 k개)도 앞 run에서만 한다.
- EWMA + Ridge, EWMA + LightGBM: EWMA가 남긴 차이(실측 - EWMA)를 trace 특성으로 맞힌다. 앞 run 측정값과 trace를 함께 쓰는 운영.
판정: run마다 |모델 오차| - |EWMA 오차|를 구해 평균이 0보다 작고 그 크기가 표준오차의 2배를 넘으면 'EWMA보다 확실히 나음'.
leave-one-run-out(10-27 방식, 시간 순서 무시)도 참고로 적는다. 시험 run은 쓰지 않는다.
대상: 두께 평균, U/F, WtoW, 사이클당 두께(두께 ÷ trace 사이클 수, 사이클 수가 다른 run을 같은 눈금으로 보기 위함).
run 특성은 g2_thickness와 같은 신호별 요약에 trace 사이클 수와 앞 run과의 대기 시간을 더한 것이다.

    python scripts/vm_stage1.py                 # → work/reports/vm_stage1.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.data.splits import load_split
from ald_lab.paths import resolve_paths
from ald_lab.training.config import load_experiment
from ald_lab.training.vm import TARGET_UNITS, _dataset, _estimator, load_targets

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "g2_thickness.yaml"))
ap.add_argument("--input-set", default="all", help="run 특성에 쓸 입력 세트 (기본 all)")
ap.add_argument("--start", type=int, default=20, help="시간순 평가에서 처음 학습만 하는 run 수 (기본 20)")
ap.add_argument("--top", type=int, default=10, help="Ridge가 쓰는 특성 수 (앞 run에서 순위 상관 상위, 기본 10)")
ap.add_argument("--alpha", type=float, default=0.3, help="EWMA 가중치 (기본 0.3)")
ap.add_argument("--seeds", default="42,43,44", help="LightGBM seed (쉼표 구분)")
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

P = resolve_paths()
cfg = load_experiment(args.config)
split = load_split(cfg.split_path())
x, _, inputs, _ = _dataset(cfg, P, split, ("train", "validation"), input_set=args.input_set)
targets = load_targets(P, ["thk_mean", "uf_pct_mean", "wtow_pct"]).reindex(x.index)
catalog_path = P.overview_dir / "runs.parquet"
if catalog_path.is_file():
    cat = pd.read_parquet(catalog_path).set_index("run_id")
    timeline = cat.dropna(subset=["process_start"]).sort_values("process_start")
    idle = (timeline["process_start"] - timeline["process_end"].shift()).dt.total_seconds() / 3600
    x["n_cycles"] = cat["n_cycles"].reindex(x.index).astype(float)
    x["idle_hours"] = idle.reindex(x.index).astype(float)
    targets["gpc"] = targets["thk_mean"] / x["n_cycles"]
order = sorted(x.index, key=lambda r: int(str(r).rsplit("_", 1)[-1]))
x = x.loc[order].apply(lambda c: c.fillna(c.median()))
x = x.loc[:, x.std(ddof=0) > 0]
targets = targets.loc[order]
UNITS = {**TARGET_UNITS, "gpc": "Å/cycle"}


def ewma_series(y: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA prediction of each run from the runs before it (first run: nan)."""

    out = np.full(len(y), np.nan)
    level = y[0]
    for i in range(1, len(y)):
        out[i] = level
        level = alpha * y[i] + (1 - alpha) * level
    return out


def top_features(xt: pd.DataFrame, yt: np.ndarray, k: int) -> list[str]:
    r = xt.rank().corrwith(pd.Series(yt, index=xt.index).rank()).abs().fillna(0)
    return list(r.sort_values(ascending=False).index[:k])


def fit_predict(model: str, xt: pd.DataFrame, yt: np.ndarray, xq: pd.DataFrame) -> float:
    if model == "ridge":
        cols = top_features(xt, yt, args.top)
        est = _estimator("ridge", {}, 0).fit(xt[cols].to_numpy(), yt)
        return float(est.predict(xq[cols].to_numpy())[0])
    preds = []
    for seed in seeds:
        est = _estimator("lightgbm", {"min_child_samples": 5}, seed).fit(xt.to_numpy(), yt)
        preds.append(float(np.asarray(est.predict(xq.to_numpy())).reshape(-1)[0]))
    return float(np.mean(preds))


out = ["## 두께 예측(VM) 1단계 · 시간순 평가 (학습 + 검증 run, 시험 run 미사용)",
       (f"run {len(order)}개, 처음 {args.start}개는 학습만. 특성 {x.shape[1]}개(신호 {len(inputs)}개 요약 + 사이클 수 + 대기 시간). "
        f"Ridge는 앞 run에서 고른 상위 {args.top}개 특성, LightGBM은 seed {len(seeds)}개 평균."),
       "판정: run마다 |모델 오차| - |EWMA 오차|의 평균이 음수이고 표준오차의 2배를 넘으면 '확실히 나음'. 개선율은 같은 run들의 MAE로 계산.", ""]
rows, chosen = [], {}
for t in [c for c in ("thk_mean", "uf_pct_mean", "wtow_pct", "gpc") if c in targets]:
    y = targets[t].to_numpy(float)
    ew = ewma_series(y, args.alpha)
    resid = y - ew  # what EWMA leaves for each run, known once that run is measured
    preds = {name: np.full(len(y), np.nan) for name in ("이전 run 평균", "EWMA", "Ridge", "LightGBM", "EWMA + Ridge", "EWMA + LightGBM")}
    for i in range(args.start, len(y)):
        xt, xq, yt = x.iloc[:i], x.iloc[i:i + 1], y[:i]
        preds["이전 run 평균"][i] = yt.mean()
        preds["EWMA"][i] = ew[i]
        preds["Ridge"][i] = fit_predict("ridge", xt, yt, xq)
        preds["LightGBM"][i] = fit_predict("lightgbm", xt, yt, xq)
        ok = ~np.isnan(resid[:i])
        preds["EWMA + Ridge"][i] = ew[i] + fit_predict("ridge", xt[ok], resid[:i][ok], xq)
        preds["EWMA + LightGBM"][i] = ew[i] + fit_predict("lightgbm", xt[ok], resid[:i][ok], xq)
        for c in top_features(xt[ok], resid[:i][ok], args.top):
            chosen.setdefault(t, {}).setdefault(c, 0)
            chosen[t][c] += 1
    keep = np.arange(len(y)) >= args.start
    err = {k: np.abs(v[keep] - y[keep]) for k, v in preds.items()}
    base_mean, base_ewma = err["이전 run 평균"].mean(), err["EWMA"].mean()
    for name, e in err.items():
        diff = e - err["EWMA"]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        verdict = "-" if name == "EWMA" else ("확실히 나음" if diff.mean() < 0 and -diff.mean() > 2 * se else
                                              ("확실히 나쁨" if diff.mean() > 2 * se else "차이 불분명"))
        rows.append({"대상": t, "단위": UNITS.get(t, ""), "방법": name, "MAE": f"{e.mean():.4g}",
                     "이전 run 평균 대비 개선율 (%)": round(100 * (1 - e.mean() / base_mean), 1),
                     "EWMA 대비 개선율 (%)": round(100 * (1 - e.mean() / base_ewma), 1),
                     "EWMA보다 나은 run %": round(100 * (diff < 0).mean(), 0) if name != "EWMA" else "-", "EWMA 대비 판정": verdict})
frame = pd.DataFrame(rows)
for t, g in frame.groupby("대상", sort=False):
    out += [f"[{t}] 평가 run {len(order) - args.start}개", g.drop(columns="대상").to_string(index=False), ""]

# leave-one-run-out for reference (the 10-27 setting: no time order, no earlier runs' metrology)
rows = []
for t in [c for c in ("thk_mean", "uf_pct_mean", "wtow_pct", "gpc") if c in targets]:
    y = targets[t].to_numpy(float)
    n = len(y)
    loo = {"학습 평균": np.array([(y.sum() - y[i]) / (n - 1) for i in range(n)]), "Ridge": np.zeros(n), "LightGBM": np.zeros(n)}
    for i in range(n):
        keep_i = np.arange(n) != i
        loo["Ridge"][i] = fit_predict("ridge", x.iloc[keep_i], y[keep_i], x.iloc[i:i + 1])
        loo["LightGBM"][i] = fit_predict("lightgbm", x.iloc[keep_i], y[keep_i], x.iloc[i:i + 1])
    base = np.abs(loo["학습 평균"] - y).mean()
    for name, p in loo.items():
        rows.append({"대상": t, "방법": name, "MAE": f"{np.abs(p - y).mean():.4g}",
                     "학습 평균 대비 개선율 (%)": round(100 * (1 - np.abs(p - y).mean() / base), 1)})
out += ["[참고 · leave-one-run-out] 시간 순서를 무시하고 나머지 run 전부로 학습(10-27 방식). 앞 run 측정값은 쓰지 않음.",
        pd.DataFrame(rows).to_string(index=False), ""]
out.append("[EWMA가 남긴 차이와 자주 함께 움직인 특성] 시간순 평가의 매 run에서 앞 run으로 고른 상위 특성이 뽑힌 횟수")
for t, counts in chosen.items():
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    out.append(f"- {t}: " + ", ".join(f"{c} {k}회" for c, k in top))
text = "\n".join(out)
print(text)
report = P.work_root / "reports" / "vm_stage1.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(text + "\n", encoding="utf-8")
print(f"\n저장: {report}")
