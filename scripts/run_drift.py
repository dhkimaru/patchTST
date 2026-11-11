"""run별 장비 상태 변화(시기 구분) 분석. 학습 없이 work/prepared만 읽고, 시험 run은 쓰지 않는다.

학습 · 검증 run(split_v2 기준 62개)을 run 번호(시간) 순서로 놓고, 주요 신호마다
- run별 사이클 구간 평균과, 학습 run으로 만든 레시피 평균에서 벗어난 정도(사이클 구간 평균 절대 차이)
- 변화점: 앞뒤 묶음의 평균 차이가 가장 큰 run (차이 ÷ 묶음 안 표준편차)
를 계산한다. 학습 run 수 · 최근 run 실험에서 특정 시기의 run(예: 048~050, 001~011)이 결과를 좌우한 이유를 찾는 데 쓴다.

    python scripts/run_drift.py [--split configs/splits/split_v2.yaml]   # work/reports/run_drift.txt
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ald_lab.data.splits import DEFAULT_SPLIT, load_split
from ald_lab.models.classic import RecipeTemplate
from ald_lab.paths import resolve_paths
from ald_lab.training.prepare import build_bundle

SIGNALS = ["MFCMon_DCS", "MFCMon_NH3", "MFCMon_N2-1", "MFCMon_N2-4", "ValveAct_29", "ValveAct_89", "APCValveMon", "VG11",
           "TempAct_C", "MFCMon_L.POS", "MFCMon_P.POS", "AUXMon_R.PWR"]
MIN_SEGMENT = 4

ap = argparse.ArgumentParser()
ap.add_argument("--split", default=str(DEFAULT_SPLIT))
args = ap.parse_args()
P = resolve_paths()
split = load_split(args.split)
train = build_bundle(P, split, "train", sample_seconds=1.0)
val = build_bundle(P, split, "validation", sample_seconds=1.0)
signals = [s for s in SIGNALS if s in train.columns and s in val.columns]
cols_t, cols_v = train.column_index(signals), val.column_index(signals)
template = RecipeTemplate().fit(train, np.asarray(train.values)[:, cols_t])

rows = []
for part, bundle, cols in (("학습", train, cols_t), ("검증", val, cols_v)):
    values = np.asarray(bundle.values)[:, cols]
    recipe = template.values(bundle)
    run_index, cycle = np.asarray(bundle.run_index), np.asarray(bundle.in_cycle).astype(bool)
    for r, run_id in enumerate(bundle.run_ids):
        mask = (run_index == r) & cycle
        if not mask.any():
            continue
        v, t = values[mask], recipe[mask]
        for j, sig in enumerate(signals):
            rows.append({"run": run_id[-3:], "구분": part, "signal": sig, "mean": float(v[:, j].mean()),
                         "dev": float(np.abs(v[:, j] - t[:, j]).mean())})
frame = pd.DataFrame(rows).sort_values("run")
out: list[str] = []
add = out.append


def change_point(series: pd.Series) -> tuple[int, float] | None:
    x = series.to_numpy(float)
    best = None
    seg = max(2, min(MIN_SEGMENT, len(x) // 3))
    for k in range(seg, len(x) - seg + 1):
        a, b = x[:k], x[k:]
        sd = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(x) - 2))
        score = abs(b.mean() - a.mean()) / sd if sd > 0 else 0.0
        if best is None or score > best[1]:
            best = (k, float(score))
    return best


cps = []
for sig, g in frame.groupby("signal"):
    for quantity, name in (("mean", "사이클 평균"), ("dev", "레시피 평균과의 차이")):
        s = g.set_index("run")[quantity]
        cp = change_point(s)
        if cp is None:
            continue
        k, score = cp
        before, after = s.iloc[:k].mean(), s.iloc[k:].mean()
        cps.append({"신호": sig, "값": name, "변화 시작 run": s.index[k], "앞 평균": before, "뒤 평균": after,
                    "변화 %": 100 * (after / before - 1) if before else np.nan, "효과 크기": score})
cp_frame = pd.DataFrame(cps, columns=["신호", "값", "변화 시작 run", "앞 평균", "뒤 평균", "변화 %", "효과 크기"])
cp_frame = cp_frame.sort_values("효과 크기", ascending=False)
for c in ("앞 평균", "뒤 평균"):
    cp_frame[c] = cp_frame[c].map(lambda x: f"{x:.4g}")
cp_frame["변화 %"] = cp_frame["변화 %"].round(1)
cp_frame["효과 크기"] = cp_frame["효과 크기"].round(1)
add(f"## 1. 변화점 (run {frame['run'].min()}~{frame['run'].max()}, 학습 · 검증 run {frame['run'].nunique()}개, 시험 run 제외)")
add("효과 크기 = 앞뒤 묶음 평균 차이 ÷ 묶음 안 표준편차. 2를 넘으면 그 run을 경계로 상태가 뚜렷이 다르다.")
add(cp_frame.to_string(index=False))

for quantity, name in (("mean", "사이클 평균"), ("dev", "레시피 평균과의 차이")):
    wide = frame.pivot_table(index="run", columns="signal", values=quantity)
    z = (wide - wide.mean()) / wide.std(ddof=1).replace(0, np.nan)
    z.insert(0, "구분", frame.drop_duplicates("run").set_index("run")["구분"])
    add(f"\n## {2 if quantity == 'mean' else 3}. run별 {name} (전체 run 평균에서 표준편차 몇 배 떨어졌는지, |값| ≥ 2는 *)")
    shown = z.copy()
    for c in signals:
        if c in shown:
            shown[c] = z[c].map(lambda v: "-" if pd.isna(v) else (f"{v:+.1f}*" if abs(v) >= 2 else f"{v:+.1f}"))
    add(shown.to_string())

text = "\n".join(out)
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "run_drift.txt").write_text(text, encoding="utf-8")
print(text)
