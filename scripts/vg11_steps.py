"""VG11 가상 센서의 사이클 밖 오차 원인 확인: 0.5초 조합과 레시피 평균의 오차를 step별 · run별로 나눈다. 학습 없음.

split_v2에서 고른 0.5초 조합(공정 전체 study res_05s + 사이클 전용 study res05_cycle, 각자 검증 MAE가 가장 낮은 job)과
같은 실험의 레시피 평균(template)을 검증 run과 시험 run(재사용, 참고)의 trace에 다시 돌려(ald-predict --cycle-model과
같은 경로) 0.5초 행마다 오차를 구한다. 시험 run 결과로 설정을 바꾸지 않는다.

읽는 법
- step별: 사이클 밖 오차가 어느 step에 몰리는지, 그 step에서 모델이 레시피 평균보다 나은지(조합 ÷ 레시피 평균).
- 검증 → 시험: 같은 step에서 레시피 평균도 시험에서 함께 나빠지면 그 구간의 공정 자체가 달라진 것이고,
  조합만 나빠지면 모델이 그 구간을 잘못 배운 것이다.
- 전환 직후: step이 바뀐 직후 몇 초(기본 3초)의 오차 몫. 타이밍 어긋남이면 여기에 몰린다.
- run별: 사이클 밖 오차가 run 번호(시기)를 따라 커지는지. 평균 부호 오차가 한쪽으로 치우치면 수준(offset) 차이다.

    python scripts/vg11_steps.py                      # → work/reports/vg11_steps.txt
    python scripts/vg11_steps.py --gpu 0 --no-test    # 검증 run만
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.data.raw import run_no_from_id
from ald_lab.data.splits import load_split
from ald_lab.ledger import collect_ledger
from ald_lab.paths import resolve_paths
from ald_lab.training.predict import experiment_for, predict_pair, predict_traces
from ald_lab.training.prepare import TraceSource
from ald_lab.training.runner import parse_devices

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--target", default="VG11", help="대상 (기본 VG11)")
ap.add_argument("--full", default="res_05s", help="공정 전체 모델의 study (기본 res_05s)")
ap.add_argument("--cycle", default="res05_cycle", help="사이클 전용 모델의 study (기본 res05_cycle)")
ap.add_argument("--model", default="patchtst", help="모델 (기본 patchtst)")
ap.add_argument("--gpu", default="auto", help="'0' 또는 'cpu'")
ap.add_argument("--no-test", action="store_true", help="시험 run을 쓰지 않는다")
ap.add_argument("--top", type=int, default=12, help="step별 표에 보일 step 수 (기본 12)")
ap.add_argument("--transition", type=float, default=3.0, help="'전환 직후'로 볼 step 시작 후 초 (기본 3)")
args = ap.parse_args()

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led["job_status"] == "ok") & (led["task"] == "virtual_sensor")
              & (led["target"] == args.target)]


def job_dir(row) -> Path:
    return Path(row["exp_dir"]) / "jobs" / str(row["job_id"])


def best(study: str, model: str) -> pd.Series:
    rows = led[(led["study"] == study) & (led["model"] == model)] if len(led) else led
    if not len(rows):
        raise SystemExit(f"{args.target}: study {study}에 {model} 기록이 없습니다.")
    return rows.sort_values("mae").iloc[0]


full, cyc = best(args.full, args.model), best(args.cycle, args.model)
base_rows = led[(led["exp_dir"] == full["exp_dir"]) & (led["model"] == "template")]
base = base_rows.iloc[0] if len(base_rows) else best(args.full, "template")
full_dir, cyc_dir, base_dir = job_dir(full), job_dir(cyc), job_dir(base)
split = load_split(experiment_for(full_dir).split_path())
csv = TraceSource(P).csv
device = parse_devices(args.gpu)[0]
parts = [("검증", split.runs("validation"))]
if not args.no_test:
    parts.append(("시험", split.runs("test", include_test=True)))  # 재사용, 참고: 결과로 설정을 바꾸지 않는다

frames = []
for label, runs in parts:
    files = [csv[r] for r in runs if r in csv]
    if not files:
        raise SystemExit(f"{label} run의 trace CSV를 찾지 못했습니다 (ALD_DATA_ROOT 확인).")
    pair = predict_pair(full_dir, cyc_dir, files, device=device)
    ref = predict_traces(base_dir, files, device=device)
    keep = ["run_id", "timestamp", "step_name", "in_cycle", "actual", "pred", "model_used"]
    m = pair[keep].merge(ref[["run_id", "timestamp", "pred"]].rename(columns={"pred": "base"}),
                         on=["run_id", "timestamp"], how="inner")
    m["part"] = label
    frames.append(m)
d = pd.concat(frames, ignore_index=True).sort_values(["part", "run_id", "timestamp"], kind="stable", ignore_index=True)
d["err"] = d["pred"] - d["actual"]
d["abs"] = d["err"].abs()
d["base_abs"] = (d["base"] - d["actual"]).abs()
# Seconds since the step started (a new step instance whenever the label changes within a run).
ts = d["timestamp"]
secs = (pd.to_datetime(ts) - pd.Timestamp(0)).dt.total_seconds() if not np.issubdtype(ts.dtype, np.number) else ts
d["_t"] = secs.to_numpy(float)
new = ((d["step_name"] != d.groupby("run_id")["step_name"].shift()) | (d["run_id"] != d["run_id"].shift())
       | (d["_t"].diff().abs() > 2.0))  # a gap in the rows also starts a new step instance
d["_inst"] = new.cumsum()
d["t_in_step"] = d["_t"] - d.groupby("_inst")["_t"].transform("min")
out_ = d[~d["in_cycle"]]

lines: list[str] = []
add = lines.append
add(f"## {args.target} 사이클 밖 오차 원인 확인 (학습 없음{'' if args.no_test else ', 시험 run 재사용 · 참고'})")
add(f"0.5초 조합 = 공정 전체 {full['experiment']}/{full['stamp']} seed {full['seed']} + 사이클 전용 "
    f"{cyc['experiment']}/{cyc['stamp']} seed {cyc['seed']}, 레시피 평균 = 같은 실험의 template. 값은 0.5초 행 기준 MAE.")


def mae(x: pd.Series) -> float:
    return float(x.mean()) if len(x) else np.nan


add("\n[구간 요약]")
rows = []
for label, g in d.groupby("part", sort=False):
    o, i = g[~g["in_cycle"]], g[g["in_cycle"]]
    rows.append({"구분": f"{label} run {g['run_id'].nunique()}개", "조합 사이클 안": mae(i["abs"]), "조합 사이클 밖": mae(o["abs"]),
                 "레시피 평균 사이클 안": mae(i["base_abs"]), "레시피 평균 사이클 밖": mae(o["base_abs"]),
                 "조합 오차 중 사이클 밖 몫 %": 100 * o["abs"].sum() / g["abs"].sum(),
                 "조합 ÷ 레시피 평균 (전체)": g["abs"].mean() / g["base_abs"].mean()})
add(pd.DataFrame(rows).round(4).to_string(index=False))


def by_step(g: pd.DataFrame) -> pd.DataFrame:
    s = g.groupby("step_name", observed=True).agg(rows=("abs", "size"), pair=("abs", "mean"), base=("base_abs", "mean"),
                                                  pair_sum=("abs", "sum"), bias=("err", "mean"))
    s["행 %"] = 100 * s["rows"] / s["rows"].sum()
    s["오차 몫 %"] = 100 * s["pair_sum"] / s["pair_sum"].sum()
    s["조합 ÷ 레시피 평균"] = s["pair"] / s["base"]
    return s


steps = {label: by_step(g) for label, g in out_.groupby("part", sort=False)}
order = steps.get("시험", steps["검증"]).sort_values("오차 몫 %", ascending=False).index[: args.top]
for label, s in steps.items():
    t = s.reindex(order).rename(columns={"pair": "조합 MAE", "base": "레시피 평균 MAE", "bias": "평균 부호 오차"})
    t = t[["행 %", "오차 몫 %", "조합 MAE", "레시피 평균 MAE", "조합 ÷ 레시피 평균", "평균 부호 오차"]]
    add(f"\n[{label} run · 사이클 밖 step별] (오차 몫 큰 순, 상위 {len(order)}개 step이 사이클 밖 오차의 "
        f"{s.reindex(order)['오차 몫 %'].sum():.0f} %)")
    add(t.round(4).reset_index().rename(columns={"step_name": "step"}).to_string(index=False))
if {"검증", "시험"} <= set(steps):
    v, te = steps["검증"].reindex(order), steps["시험"].reindex(order)
    cmp_ = pd.DataFrame({"조합 시험 ÷ 검증": te["pair"] / v["pair"], "레시피 평균 시험 ÷ 검증": te["base"] / v["base"]})
    cmp_["읽기"] = np.where(cmp_["레시피 평균 시험 ÷ 검증"] >= 1.5, "공정 자체가 달라짐",
                          np.where(cmp_["조합 시험 ÷ 검증"] >= 1.5, "모델만 나빠짐", "-"))
    add("\n[검증 → 시험, 같은 step] 레시피 평균도 1.5배 이상이면 그 구간의 공정이 시험 run에서 달라진 것, "
        "조합만 1.5배 이상이면 모델이 잘못 배운 것으로 읽는다.")
    add(cmp_.round(2).reset_index().rename(columns={"step_name": "step"}).to_string(index=False))

add(f"\n[전환 직후 {args.transition:g}초 · 사이클 밖]")
rows = []
for label, g in out_.groupby("part", sort=False):
    early = g["t_in_step"] < args.transition
    rows.append({"구분": label, "행 %": 100 * early.mean(), "오차 몫 %": 100 * g.loc[early, "abs"].sum() / g["abs"].sum(),
                 "조합 MAE 직후": mae(g.loc[early, "abs"]), "조합 MAE 나머지": mae(g.loc[~early, "abs"]),
                 "레시피 평균 MAE 직후": mae(g.loc[early, "base_abs"]), "레시피 평균 MAE 나머지": mae(g.loc[~early, "base_abs"])})
add(pd.DataFrame(rows).round(4).to_string(index=False))

add("\n[run별 · 사이클 밖] run 번호 순. 평균 부호 오차 = 예측 − 실측의 평균(한쪽으로 치우치면 수준 차이)")
r = out_.groupby(["part", "run_id"], sort=False).agg(pair=("abs", "mean"), base=("base_abs", "mean"),
                                                      bias=("err", "mean")).reset_index()
r["no"] = r["run_id"].map(lambda x: run_no_from_id(str(x)))
r["조합 ÷ 레시피 평균"] = r["pair"] / r["base"]
r = r.sort_values("no")
add(r.rename(columns={"part": "구분", "run_id": "run", "pair": "조합 MAE", "base": "레시피 평균 MAE", "bias": "평균 부호 오차"})
    [["run", "구분", "조합 MAE", "레시피 평균 MAE", "조합 ÷ 레시피 평균", "평균 부호 오차"]].round(4).to_string(index=False))
if len(r) >= 3:
    rho = r[["no", "조합 ÷ 레시피 평균"]].rank().corr().iloc[0, 1]
    add(f"run 번호와 '조합 ÷ 레시피 평균'의 순위 상관: {rho:+.2f} (양수면 나중 run일수록 모델이 레시피 평균보다 불리)")

text = "\n".join(lines)
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "vg11_steps.txt").write_text(text, encoding="utf-8")
print(text)
