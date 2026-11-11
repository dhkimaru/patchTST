"""저장된 검증 예측값(predictions_val.parquet)으로 다시 학습하지 않고 모델을 조합해 본다. work/만 읽는다.

1. 구간 전용 모델 합치기: 사이클 안 window는 사이클만 학습한 모델, 사이클 밖은 전체(또는 사이클 밖만) 학습한 모델
   (가상 센서: lc51 · region_cycle · region_noncycle, 미래값 10~30초: fc_ref · fc_region_cycle)
2. 모델 평균: 주 모델(PatchTST)과 LightGBM 예측의 평균 (가상 센서 lc51, 미래값 장기 예측)
3. 예측 거리별 모델 선택 (미래값 30초~10분): 검증 run을 절반으로 나눠 한쪽에서 고른 모델을 다른 쪽에서 채점한다

    python scripts/combine_predictions.py      # 화면 출력 + work/reports/combine_predictions.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led.get("job_status", "ok") == "ok")]
st = led["study"].fillna("").astype(str) if len(led) else pd.Series(dtype=str)
BASE = ("persistence", "template", "schedule")
NAMES = {"persistence": "현재값 유지", "template": "레시피 평균", "schedule": "레시피 일정 평균", "patchtst": "PatchTST",
         "lightgbm": "LightGBM", "itransformer": "iTransformer", "dlinear": "DLinear", "gru": "GRU", "ridge": "Ridge"}
out: list[str] = []
add = out.append


def fmt(x: float) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.4g}"


def pred_file(row) -> Path:
    return Path(row.exp_dir) / "jobs" / str(row.job_id) / "predictions_val.parquet"


def load(rows: pd.DataFrame) -> list[pd.DataFrame]:
    """One frame per job (seed), indexed by (run_id, row); jobs without saved predictions are skipped."""

    frames = []
    for r in rows.itertuples():
        f = pred_file(r)
        if f.is_file():
            frames.append(pd.read_parquet(f).assign(run_id=lambda d: d["run_id"].astype(str)).set_index(["run_id", "row"]))
    return frames


def horizons(frame: pd.DataFrame) -> list[str]:
    return [c[5:] for c in frame.columns if c.startswith("pred_")]


def mae(frame: pd.DataFrame, mask: pd.Series | None = None) -> float:
    cols = horizons(frame)
    err = np.mean([np.abs(frame[f"pred_{h}"] - frame[f"true_{h}"]).to_numpy() for h in cols], axis=0)
    if mask is not None:
        err = err[np.asarray(mask, dtype=bool)]
    return float(err.mean()) if err.size else np.nan


def seed_mean(frames: list[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Average prediction over seeds on the given windows (a seed ensemble)."""

    base = frames[0].loc[index].copy()
    for h in horizons(base):
        base[f"pred_{h}"] = np.mean([f.loc[index, f"pred_{h}"].to_numpy() for f in frames], axis=0)
    return base


def main_model(rows: pd.DataFrame) -> str | None:
    trained = [m for m in rows["model"].unique() if m not in BASE]
    return "patchtst" if "patchtst" in trained else (sorted(trained)[0] if trained else None)


def split_runs(index: pd.Index) -> tuple[set[str], set[str]]:
    runs = sorted(index.get_level_values(0).unique())
    return set(runs[0::2]), set(runs[1::2])


def pick_context(rows: pd.DataFrame, preferred: float = 192.0) -> pd.DataFrame:
    """Rows of one window length: the preferred one when present, else the most common."""

    if not len(rows):
        return rows
    ctx = preferred if (rows["context_seconds"] == preferred).any() else rows["context_seconds"].mode().iloc[0]
    return rows[rows["context_seconds"] == ctx]


# 1) 구간 전용 모델 합치기
def region_hybrid(full: str, cycle: str, noncycle: str | None, baseline: str, title: str) -> list[dict]:
    rows_ = led[st.isin([s for s in (full, cycle, noncycle) if s])] if len(led) else led
    if not len(rows_):
        return []
    recs = []
    for target, g in rows_.groupby("target"):
        fg, cg = g[g["study"] == full], g[g["study"] == cycle]
        model = main_model(fg[fg["model"].isin(set(cg["model"]))])  # a model trained both ways (PatchTST first)
        if model is None:
            continue
        f_frames, c_frames = load(fg[fg["model"] == model]), load(cg[cg["model"] == model])
        n_frames = load(g[(g["study"] == noncycle) & (g["model"] == model)]) if noncycle else []
        base = load(fg[fg["model"] == baseline].head(1))
        if not f_frames or not c_frames or not base:
            continue
        idx = f_frames[0].index
        cyc = f_frames[0]["in_cycle"].astype(bool)
        rec = {"대상": target, "모델": NAMES.get(model, model), "seed": len(f_frames)}

        def score(frame: pd.DataFrame, cyc: pd.Series = cyc) -> tuple[float, float, float]:
            return mae(frame), mae(frame, cyc), mae(frame, ~cyc)

        b_all, b_c, b_o = score(base[0].loc[idx])
        variants = {"전체 학습": [f.loc[idx] for f in f_frames]}
        hyb = []
        for i, f in enumerate(f_frames):
            h = f.loc[idx].copy()
            c = c_frames[min(i, len(c_frames) - 1)]
            common = idx[cyc.to_numpy()].intersection(c.index)
            for col in [x for x in h.columns if x.startswith("pred_")]:
                h.loc[common, col] = c.loc[common, col]
            hyb.append(h)
        variants["사이클 전용 + 전체"] = hyb
        if n_frames:
            pair = []
            for i, h in enumerate(hyb):
                p = h.copy()
                n = n_frames[min(i, len(n_frames) - 1)]
                common = idx[~cyc.to_numpy()].intersection(n.index)
                for col in [x for x in p.columns if x.startswith("pred_")]:
                    p.loc[common, col] = n.loc[common, col]
                pair.append(p)
            variants["사이클 전용 + 사이클 밖 전용"] = pair
        rec.update({"기준선 MAE": fmt(b_all)})
        for name, frames in variants.items():
            scores = np.array([score(f) for f in frames])
            a, c_, o = scores.mean(axis=0)
            rec[f"{name} MAE"] = fmt(a)
            rec[f"{name} 개선율"] = round(100 * (1 - a / b_all), 1)
            rec[f"{name} 사이클 안 개선율"] = round(100 * (1 - c_ / b_c), 1) if b_c else np.nan
            rec[f"{name} 사이클 밖 개선율"] = round(100 * (1 - o / b_o), 1) if b_o else np.nan
            if len(frames) > 1:
                ens = frames[0].copy()
                for col in [x for x in ens.columns if x.startswith("pred_")]:
                    ens[col] = np.mean([f[col].to_numpy() for f in frames], axis=0)
                rec[f"{name} seed 평균 개선율"] = round(100 * (1 - mae(ens) / b_all), 1)
        recs.append(rec)
    if recs:
        add(f"\n## {title}")
        add(f"기준선 = {NAMES[baseline]}. 개선율은 같은 window에서 잰 기준선 대비 MAE 감소율(%), seed별로 계산해 평균했다. "
            "seed 평균 개선율은 seed 예측을 평균한 예측(앙상블)의 개선율이다.")
        frame = pd.DataFrame(recs)
        for name in ("전체 학습", "사이클 전용 + 전체", "사이클 전용 + 사이클 밖 전용"):
            cols = [c for c in frame.columns if c.startswith(name)]
            if cols:
                add(f"\n[{name}]")
                add(frame[["대상", "모델", "기준선 MAE", *cols]].rename(columns=lambda c, n=name: c.replace(n + " ", ""))
                    .to_string(index=False))
    return recs


def region_hybrid_folds(folds: list[int]) -> None:
    """Cross-validation folds: the region pair on other validation runs, then the gain per target across folds."""

    by_fold = {}
    for k in folds:
        recs = region_hybrid(f"cv{k}", f"region_cycle_cv{k}", f"region_noncycle_cv{k}", "template",
                             f"1-3. 가상 센서 · 구간 전용 모델 합치기 · 교차검증 {k}/5")
        if recs:
            by_fold[k] = recs
    if len(by_fold) < 2:
        return
    rows = [{"fold": k, **r} for k, recs in by_fold.items() for r in recs]
    frame = pd.DataFrame(rows)
    cols = [c for c in ("전체 학습 개선율", "사이클 전용 + 전체 개선율", "사이클 전용 + 사이클 밖 전용 개선율") if c in frame]
    summary = frame.groupby("대상")[cols].agg(["mean", "std", "count"]).round(1)
    gain = frame.assign(차이=frame[cols[-1]] - frame[cols[0]]).groupby("대상")["차이"].agg(["mean", "std", "min"]).round(1)
    add(f"\n## 1-4. 구간 전용 모델 합치기 · 교차검증 {len(by_fold)}묶음 요약 (개선율 %, 묶음 평균 · 표준편차 · 묶음 수)")
    add(summary.to_string())
    add("\n전체 학습 대비 늘어난 개선율 (%p, 묶음 평균 · 표준편차 · 가장 작은 묶음). "
        "평균이 표준편차의 2배를 넘고 가장 작은 묶음도 0보다 크면 확실하다.")
    add(gain.to_string())


# 2) 모델 평균 (주 모델 + LightGBM)
def model_average(study: str, baseline: str, title: str, context: float | None = None) -> None:
    rows_ = led[st == study] if len(led) else led
    if context is not None:
        rows_ = pick_context(rows_, context)
    if not len(rows_):
        return
    recs = []
    for target, g in rows_.groupby("target"):
        model = main_model(g[g["model"] != "lightgbm"])
        mains, lgb, base = load(g[g["model"] == model]), load(g[g["model"] == "lightgbm"]), load(g[g["model"] == baseline].head(1))
        if not mains or not lgb or not base or model == "lightgbm":
            continue
        idx = mains[0].index.intersection(lgb[0].index).intersection(base[0].index)
        if not len(idx):
            continue
        m, lg, b = seed_mean(mains, idx), seed_mean(lgb, idx), base[0].loc[idx]
        avg = m.copy()
        for h in horizons(avg):
            avg[f"pred_{h}"] = (m[f"pred_{h}"] + lg[f"pred_{h}"]) / 2
        mb = mae(b)
        cyc = m["in_cycle"].astype(bool)
        recs.append({"대상": target, "기준선 MAE": fmt(mb),
                     f"{NAMES.get(model, model)} 개선율": round(100 * (1 - mae(m) / mb), 1),
                     "LightGBM 개선율": round(100 * (1 - mae(lg) / mb), 1),
                     "평균 개선율": round(100 * (1 - mae(avg) / mb), 1),
                     "평균 사이클 안 개선율": round(100 * (1 - mae(avg, cyc) / mae(b, cyc)), 1) if mae(b, cyc) else np.nan,
                     "평균 사이클 밖 개선율": round(100 * (1 - mae(avg, ~cyc) / mae(b, ~cyc)), 1) if mae(b, ~cyc) else np.nan,
                     "window 수": len(idx)})
    if recs:
        add(f"\n## {title}")
        add(f"기준선 = {NAMES[baseline]}. 각 모델은 seed 평균 예측, 평균 = 두 모델 예측의 단순 평균(가중치를 고르지 않음).")
        add(pd.DataFrame(recs).to_string(index=False))


# 3) 예측 거리별 모델 선택 (미래값 장기)
def horizon_choice(study: str, context: float, title: str) -> None:
    rows_ = pick_context(led[st == study] if len(led) else led, context)
    if not len(rows_):
        return
    blocks = []
    for target, g in rows_.groupby("target"):
        preds = {}
        for model, gm in g.groupby("model"):
            frames = load(gm if model not in BASE else gm.head(1))
            if frames:
                preds[model] = frames
        if len(preds) < 2:
            continue
        idx = None
        for frames in preds.values():
            idx = frames[0].index if idx is None else idx.intersection(frames[0].index)
        if idx is None or not len(idx):
            continue
        means = {m: seed_mean(f, idx) for m, f in preds.items()}
        hs = horizons(next(iter(means.values())))
        table = pd.DataFrame({NAMES.get(m, m): {h: float(np.abs(f[f"pred_{h}"] - f[f"true_{h}"]).mean()) for h in hs}
                              for m, f in means.items()}).T
        if "patchtst" in means and "lightgbm" in means:
            avg = {h: float(np.abs((means["patchtst"][f"pred_{h}"] + means["lightgbm"][f"pred_{h}"]) / 2
                                   - means["patchtst"][f"true_{h}"]).mean()) for h in hs}
            table.loc["PatchTST · LightGBM 평균"] = pd.Series(avg)
        half_a, half_b = split_runs(idx)
        runs = idx.get_level_values(0)
        chosen, sel = {}, {}
        for h in hs:
            picks, errs = [], []
            for pick_on, score_on in ((half_a, half_b), (half_b, half_a)):
                a_mask, b_mask = runs.isin(pick_on), runs.isin(score_on)
                if not a_mask.any() or not b_mask.any():
                    continue
                best = min(means, key=lambda m: float(np.abs(means[m][f"pred_{h}"] - means[m][f"true_{h}"])[a_mask].mean()))
                errs.append(np.abs(means[best][f"pred_{h}"] - means[best][f"true_{h}"])[b_mask].to_numpy())
                picks.append(NAMES.get(best, best))
            chosen[h] = " / ".join(dict.fromkeys(picks))
            sel[h] = float(np.concatenate(errs).mean()) if errs else np.nan
        table.loc["거리별 선택 (run 절반 교차)"] = pd.Series(sel)
        text = table.apply(lambda s: s.map(fmt)).to_string()
        blocks.append(f"\n[{target}] window {len(idx):,}개\n{text}\n고른 모델: " + "  ".join(f"{h} {m}" for h, m in chosen.items()))
    if blocks:
        add(f"\n## {title}")
        add("값은 예측 거리별 검증 MAE(학습 모델은 seed 평균 예측). 거리별 선택은 검증 run을 번호 순서로 번갈아 두 묶음으로 나눠, "
            "한 묶음에서 가장 나은 모델을 골라 다른 묶음에서 채점한 결과다(검증 run으로 고르고 같은 run으로 채점하지 않음).")
        out.extend(blocks)


# 1-5) 구간별로 다른 모델: 사이클 window는 A, 나머지는 B (예: PatchTST 사이클 전용 + LightGBM, 장기 예측 5~10분)
def region_mix(cycle_study: str, cycle_model: str, other_study: str, other_model: str, baseline: str, title: str) -> None:
    if not len(led):
        return
    a_rows, b_rows = led[(st == cycle_study) & (led["model"] == cycle_model)], led[(st == other_study) & (led["model"] == other_model)]
    recs = []
    for target in sorted(set(a_rows["target"]) & set(b_rows["target"])):
        a, b = load(a_rows[a_rows["target"] == target]), load(b_rows[b_rows["target"] == target])
        base = load(led[(st == other_study) & (led["model"] == baseline) & (led["target"] == target)].head(1))
        if not a or not b or not base:
            continue
        idx = b[0].index.intersection(base[0].index)
        bm, basef = seed_mean(b, idx), base[0].loc[idx]
        am = seed_mean(a, a[0].index)
        cyc = bm["in_cycle"].astype(bool)
        common = idx[cyc.to_numpy()].intersection(am.index)
        mix = bm.copy()
        for h in horizons(mix):
            if f"pred_{h}" in am:
                mix.loc[common, f"pred_{h}"] = am.loc[common, f"pred_{h}"]
        b0 = mae(basef)
        recs.append({"대상": target, "기준선 MAE": fmt(b0),
                     f"{NAMES.get(other_model, other_model)} 개선율": round(100 * (1 - mae(bm) / b0), 1),
                     "구간별 조합 개선율": round(100 * (1 - mae(mix) / b0), 1),
                     "사이클 안 (조합)": round(100 * (1 - mae(mix, cyc) / mae(basef, cyc)), 1) if mae(basef, cyc) else np.nan,
                     "사이클 밖 (조합)": round(100 * (1 - mae(mix, ~cyc) / mae(basef, ~cyc)), 1) if mae(basef, ~cyc) else np.nan,
                     "사이클 window 중 바꾼 비율": round(len(common) / max(1, int(cyc.sum())), 2)})
    if recs:
        add(f"\n## {title}")
        add(f"사이클 window는 {NAMES.get(cycle_model, cycle_model)}({cycle_study}), 나머지는 {NAMES.get(other_model, other_model)}"
            f"({other_study}). 기준선 = {NAMES.get(baseline, baseline)}, 학습 모델은 seed 평균 예측.")
        add(pd.DataFrame(recs).to_string(index=False))


region_hybrid("lc51", "region_cycle", "region_noncycle", "template",
              "1. 가상 센서 · 구간 전용 모델 합치기 (공정 전체 window에서 채점)")
region_hybrid_folds([1, 2, 3, 4, 5])
for h in (20, 10):
    region_hybrid(f"rw{h}", f"region_cycle_rw{h}", None, "template",
                  f"1-5. 가상 센서 · 사이클 전용 + 최근 run 가중 (반감기 {h}개, 둘 다 가중)")
region_hybrid("long_horizon_mid", "long_horizon_mid_cycle", None, "schedule",
              "1-7. 미래값 30초~2분 전용 · 사이클 전용 모델 합치기 (기준선 = 레시피 일정 평균)")
region_mix("region_cycle", "patchtst", "lc51", "lightgbm", "template",
           "1-6. 가상 센서 · 사이클은 PatchTST 사이클 전용, 나머지는 LightGBM")
region_mix("long_horizon_far_cycle", "patchtst", "long_horizon_far", "lightgbm", "schedule",
           "3-2. 미래값 5~10분 · 사이클은 PatchTST 사이클 전용, 나머지는 LightGBM")
region_mix("long_horizon_far_cycle", "patchtst", "long_horizon_far", "schedule", "schedule",
           "3-3. 미래값 5~10분 · 사이클은 PatchTST 사이클 전용, 나머지는 레시피 일정 평균")
region_hybrid("fc_ref", "fc_region_cycle", None, "persistence",
              "1-2. 미래값 10~30초 · 사이클 전용 모델 합치기 (run 전체 window에서 채점)")
model_average("lc51", "template", "2. 가상 센서 · PatchTST와 LightGBM 예측 평균 (학습 run 51개)")
model_average("long_horizon", "template", "2-2. 미래값 30초~10분 · PatchTST와 LightGBM 예측 평균 (window 192초)", context=192)
horizon_choice("long_horizon", 192, "3. 미래값 30초~10분 · 예측 거리별 모델 비교와 선택 (window 192초)")

text = "\n".join(out).strip() if out else "조합할 검증 예측값이 아직 없습니다 (predictions_val.parquet은 2026-11-09 코드부터 저장)."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "combine_predictions.txt").write_text(text, encoding="utf-8")
print(text)
