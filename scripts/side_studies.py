"""부가 실험 결과 정리. 학습 없이 work/experiments만 읽는다.

가상 센서: 교차검증(cv1~5) · 시간순 검증(time1~3) · 센서 하나만 고장 · 학습 run 수 곡선(lc10~51) ·
최근 run만으로 학습(recent2~4) · 0.5초 간격 · 사이클 구간만 학습. 미래값: 예측 거리 확장(모델별).
기본 결과(부가 실험 제외)에서는 모델별 최고 설정의 사이클 안 · 밖 개선율을, 새로 돈 작업에서는 레시피 단계별 오차를 본다.
work/reports/run_types.json(run_drift.py가 만듦)이 있으면 run 유형(쉬움 · 어려움)별 남은 오차와 학습 run 구성의 효과(1-15 ~ 1-17)도 낸다.

    python scripts/side_studies.py            # 화면 출력 + work/reports/side_studies.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ald_lab.ledger import collect_ledger, main_records
from ald_lab.paths import resolve_paths

P = resolve_paths()
led = collect_ledger(P.work_root)
if len(led):
    led = led[led["mae"].notna() & (led.get("job_status", "ok") == "ok")]
st = led["study"].fillna("").astype(str) if len(led) else pd.Series(dtype=str)
MAIN = main_records(led) if len(led) else led
out: list[str] = []
add = out.append
_metrics: dict[tuple[str, str], dict] = {}
NAMES = {"persistence": "현재값 유지", "template": "레시피 평균", "schedule": "레시피 일정 평균", "patchtst": "PatchTST",
         "lightgbm": "LightGBM",
         "itransformer": "iTransformer", "dlinear": "DLinear", "gru": "GRU", "ridge": "Ridge"}
MODEL_ORDER = ["patchtst", "lightgbm", "itransformer", "dlinear", "gru", "ridge"]
_types = P.work_root / "reports" / "run_types.json"
RUN_TYPES: dict[str, str] = ({k: v["type"] for k, v in json.loads(_types.read_text(encoding="utf-8"))["runs"].items()}
                             if _types.is_file() else {})
TYPE_TARGETS = ["MFCMon_DCS", "MFCMon_N2-1", "MFCMon_N2-4", "MFCMon_NH3", "ValveAct_29", "ValveAct_89", "APCValveMon", "VG11",
                "TempAct_C"]
SHORT = {"MFCMon_DCS": "DCS", "MFCMon_N2-1": "N2-1", "MFCMon_N2-4": "N2-4", "MFCMon_NH3": "NH3", "ValveAct_29": "V29",
         "ValveAct_89": "V89", "APCValveMon": "APC", "VG11": "VG11", "TempAct_C": "TempC"}
SPLIT_DIR = Path(__file__).resolve().parents[1] / "configs" / "splits"
_splits: dict[str, dict] = {}


def fmt(x: float) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.4g}"


def table(frame: pd.DataFrame, digits: int | None = None) -> str:
    if digits is not None:
        return frame.round(digits).to_string()
    return frame.apply(lambda s: s.map(fmt)).to_string()


def job_metrics(row) -> dict:
    key = (str(row.exp_dir), str(row.job_id))
    if key not in _metrics:
        f = Path(row.exp_dir) / "jobs" / row.job_id / "metrics_validation.json"
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
        _metrics[key] = doc.get("metrics") or doc
    return _metrics[key]


def by_run(row) -> dict[str, float]:
    return job_metrics(row).get("by_run") or {}


def experiment_split(exp_dir: str) -> dict:
    """The split document an experiment used (train / validation run lists)."""

    if exp_dir not in _splits:
        try:
            doc = yaml.safe_load((Path(exp_dir) / "experiment.yaml").read_text(encoding="utf-8")) or {}
            name = str((doc.get("data") or {}).get("split", "split_v2"))
            path = Path(name) if Path(name).suffix else SPLIT_DIR / f"{name}.yaml"
            _splits[exp_dir] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            _splits[exp_dir] = {}
    return _splits[exp_dir]


def per_run(g: pd.DataFrame) -> pd.DataFrame:
    """Recipe-mean and PatchTST (seed mean) MAE per validation run, with the run type."""

    base, pt = g[g["model"] == "template"], g[g["model"] == "patchtst"]
    if base.empty or pt.empty:
        return pd.DataFrame()
    rb = by_run(next(base.itertuples()))
    rp = [by_run(r) for r in pt.itertuples()]
    rows = []
    for run, v in rb.items():
        vals = [d[run] for d in rp if run in d]
        if vals and run[-3:] != "055":
            rows.append({"run": run, "type": RUN_TYPES.get(run), "base": v, "pt": float(np.mean(vals))})
    return pd.DataFrame(rows)


def remaining_by_type(r: pd.DataFrame) -> dict[str, float]:
    """PatchTST ÷ recipe mean × 100 on the summed run errors, per run type (055 left out)."""

    outv = {}
    for kind in ("easy", "hard"):
        k = r[r["type"] == kind] if len(r) else r
        outv[kind] = 100 * k["pt"].sum() / k["base"].sum() if len(k) and k["base"].sum() else np.nan
        outv[f"n_{kind}"] = len(k)
        outv[f"base_{kind}"] = k["base"].mean() if len(k) else np.nan
    return outv


def summarize(g: pd.DataFrame) -> dict | None:
    """레시피 평균(기준선)과 PatchTST(seed 평균) 한 쌍: 오차 · 개선율 · seed 편차 · 사이클 안/밖 개선율."""

    base, pt = g[g["model"] == "template"], g[g["model"] == "patchtst"]
    if base.empty or pt.empty:
        return None
    b = float(base["mae"].iloc[0])
    gains = 100 * (1 - pt["mae"].astype(float) / b)

    def rel(col: str) -> float:
        bb = base[col].iloc[0]
        if pd.isna(bb) or not bb or not pt[col].notna().any():
            return np.nan
        return 100 * (1 - float(pt[col].mean()) / float(bb))

    two = len(pt) > 1
    return {"base": b, "pt": float(pt["mae"].mean()), "pt_sd": float(pt["mae"].std(ddof=1)) if two else np.nan,
            "seeds": len(pt), "gain": float(gains.mean()), "gain_sd": float(gains.std(ddof=1)) if two else np.nan,
            "pt_cycle": float(pt["mae_cycle"].mean()) if pt["mae_cycle"].notna().any() else np.nan,
            "base_cycle": float(base["mae_cycle"].iloc[0]) if pd.notna(base["mae_cycle"].iloc[0]) else np.nan,
            "cycle": rel("mae_cycle"), "other": rel("mae_other")}


def study_rows(study: str, target: str) -> pd.DataFrame:
    return led[(st == study) & (led["target"] == target)]


def main_reference(target: str, g: pd.DataFrame) -> dict | None:
    """기본 분할(split_v2)에서 같은 설정으로 돈 기존 결과 (부가 실험 아님). 없으면 None."""

    pt = g[g["model"] == "patchtst"]
    if pt.empty or not len(MAIN):
        return None
    s, inp, ctx = pt["settings"].mode().iloc[0], pt["input_set"].mode().iloc[0], pt["context_seconds"].mode().iloc[0]
    m = MAIN[(MAIN["base"] == "g1_virtual_sensor") & (MAIN["target"] == target)]
    hit = m[(m["model"] == "patchtst") & (m["settings"] == s) & (m["input_set"] == inp) & (m["context_seconds"] == ctx)]
    if hit.empty:
        return None
    base = m[(m["model"] == "template") & m["exp_dir"].isin(hit["exp_dir"])]
    return summarize(pd.concat([base.head(1), hit]))


def reference(target: str, g: pd.DataFrame) -> tuple[dict | None, str]:
    """비교 기준: 이번 묶음의 학습 run 51개(lc51, 같은 seed) → 없으면 기본 결과의 같은 설정."""

    ref = summarize(study_rows("lc51", target))
    if ref:
        return ref, "lc51"
    ref = main_reference(target, g)
    return ref, ("기본 결과 · 같은 설정" if ref else "")


# 1) 가상 센서 부가 실험: 묶음(fold)별 개선율 · 사이클 안/밖 · run별 남은 오차
def vs_study(prefix: str, title: str) -> None:
    rows_ = led[st.str.startswith(prefix)] if len(led) else led
    if not len(rows_):
        return
    add(f"\n## {title} (기준선 = 레시피 평균, 값은 검증 MAE)")
    rows, runs = [], []
    for (target, fold), g in rows_.groupby(["target", "study"]):
        s = summarize(g)
        if s is None:
            continue
        rows.append({"target": target, "fold": fold, **s})
        base, pt = g[g["model"] == "template"], g[g["model"] == "patchtst"]
        rb = by_run(next(base.itertuples()))
        rp = [by_run(r) for r in pt.itertuples()]
        for run, v in rb.items():
            vals = [d[run] for d in rp if run in d]
            if vals:
                runs.append({"target": target, "fold": fold, "run": run[-3:], "base": v, "pt": float(np.mean(vals))})
    f = pd.DataFrame(rows)
    if f.empty:
        return
    add("묶음별 개선율 (%)")
    add(f.pivot_table(index="target", columns="fold", values="gain").round(1).to_string())
    add("사이클 안 / 밖 개선율 (%, 묶음 평균)")
    add(f.groupby("target")[["cycle", "other"]].mean().round(1).to_string())
    r = pd.DataFrame(runs)
    if r.empty:
        return
    summ = []
    for target, g in f.groupby("target"):
        rr = r[r["target"] == target]
        no55 = rr[rr["run"] != "055"]
        summ.append({"대상": target, "묶음 수": len(g), "개선율 평균": round(g["gain"].mean(), 1),
                     "묶음 간 SD": round(g["gain"].std(ddof=1), 1) if len(g) > 1 else np.nan,
                     "기준선 못 넘은 묶음": int((g["gain"] <= 0).sum()),
                     "run 수": len(rr), "run 전체 개선율": round(100 * (1 - rr["pt"].sum() / rr["base"].sum()), 1),
                     "055 제외": round(100 * (1 - no55["pt"].sum() / no55["base"].sum()), 1) if len(no55) else np.nan,
                     "좋아진 run 비율": f"{(rr['pt'] < rr['base']).mean() * 100:.0f} %"})
    add("\n대상별 요약 (run 전체 개선율 = run 오차 합 기준, 좋아진 run 비율 = PatchTST가 기준선보다 나은 run)")
    add(pd.DataFrame(summ).to_string(index=False))
    if r["fold"].nunique() == 1 or r.groupby(["target", "run"]).size().max() == 1:
        add("\nrun별 남은 오차 % (PatchTST ÷ 기준선 × 100)")
        r["pct"] = 100 * r["pt"] / r["base"]
        add(r.pivot_table(index="target", columns="run", values="pct").round(0).to_string())


# 1-4) 학습 run 수 곡선: 검증 run이 고정이라 MAE를 그대로 비교한다
def learning_curve() -> None:
    rows_ = led[st.str.fullmatch(r"lc\d+")] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for (target, study), g in rows_.groupby(["target", "study"]):
        s = summarize(g)
        if s:
            recs.append({"target": target, "runs": int(study[2:]), **s})
    f = pd.DataFrame(recs)
    if f.empty:
        return
    add("\n## 1-4. 학습 run 수 곡선 (검증 run 053~067 고정, 학습 run은 split_v2 학습 51개에서 무작위로 겹치게 뽑음)")
    add("PatchTST 검증 MAE (seed 평균)")
    add(table(f.pivot_table(index="target", columns="runs", values="pt")))
    add("레시피 평균 검증 MAE (레시피 평균도 학습 run으로 만든다)")
    add(table(f.pivot_table(index="target", columns="runs", values="base")))
    add("레시피 평균 대비 개선율 (%)")
    add(table(f.pivot_table(index="target", columns="runs", values="gain"), 1))
    sizes = sorted(f["runs"].unique())
    if len(sizes) >= 2:
        last, prev = sizes[-1], sizes[-2]
        judge = []
        for target, g in f.groupby("target"):
            a, b = g[g["runs"] == prev], g[g["runs"] == last]
            if a.empty or b.empty:
                continue
            pa, pb = float(a["pt"].iloc[0]), float(b["pt"].iloc[0])
            sd = float(np.sqrt(np.nanmean([a["pt_sd"].iloc[0] ** 2, b["pt_sd"].iloc[0] ** 2])))
            first = g.sort_values("runs").iloc[0]
            judge.append({"대상": target, f"{first['runs']}→{last}개 MAE 감소 %": round(100 * (1 - pb / first["pt"]), 1),
                          f"{prev}→{last}개 MAE 감소 %": round(100 * (1 - pb / pa), 1),
                          "seed SD": fmt(sd),
                          "판단": "아직 줄어드는 중" if pa - pb > 2 * sd
                          else ("반대로 커짐" if pb - pa > 2 * sd else "포화 (seed 편차 2배 이내)")})
        add(f"\n마지막 구간 판단 ({prev}→{last}개 MAE 차이가 seed SD 2배를 넘으면 run을 더 모으면 나아질 여지)")
        add(pd.DataFrame(judge).to_string(index=False))


# 1-5) 최근 run 20개만으로 학습 vs 같은 검증 구간의 누적 학습
def recent_vs_cumulative() -> None:
    rows_ = led[st.str.fullmatch(r"recent\d")] if len(led) else led
    if not len(rows_):
        return
    val = {"recent2": "032~043", "recent3": "044~052", "recent4": "053~067"}
    cum = {"recent2": ("time2", 31), "recent3": ("time3", 42), "recent4": ("lc51", 51)}
    recs = []
    for (target, study), g in rows_.groupby(["target", "study"]):
        s = summarize(g)
        if s is None or study not in cum:
            continue
        c = summarize(study_rows(cum[study][0], target))
        if c is None and study == "recent4":
            c = main_reference(target, g)
        rnd = summarize(study_rows("lc20", target)) if study == "recent4" else None
        recs.append({"대상": target, "검증 구간": val[study], "누적 run 수": cum[study][1],
                     "누적 MAE": fmt(c["pt"]) if c else "-", "최근 20개 MAE": fmt(s["pt"]),
                     "무작위 20개 MAE": fmt(rnd["pt"]) if rnd else "-",
                     "최근 ÷ 누적 − 1 (%)": round(100 * (s["pt"] / c["pt"] - 1), 1) if c else np.nan,
                     "누적 개선율": round(c["gain"], 1) if c else np.nan, "최근 20개 개선율": round(s["gain"], 1),
                     "seed SD (최근)": fmt(s["pt_sd"])})
    if not recs:
        return
    add("\n## 1-5. 최근 run 20개만으로 학습 (같은 검증 구간의 누적 학습과 비교)")
    add("최근 ÷ 누적 − 1이 음수면 오래된 run을 빼는 편이 낫다 (시기 변화). 053~067 구간은 무작위 20개(lc20)와도 비교한다.")
    add(pd.DataFrame(recs).sort_values(["대상", "검증 구간"]).to_string(index=False))


# 1-6, 1-7) 같은 대상 · 같은 설정의 1초 · 공정 전체 학습과 비교
def versus_full(study: str, title: str, note: str, cycle: bool) -> None:
    rows_ = led[st == study] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for target, g in rows_.groupby("target"):
        s = summarize(g)
        if s is None:
            continue
        r, src = reference(target, g)
        rec = {"대상": target, "비교 기준": src or "-"}
        if cycle:
            rec.update({"사이클만 학습 MAE": fmt(s["pt"]), "공정 전체 학습 · 사이클 안 MAE": fmt(r["pt_cycle"]) if r else "-",
                        "차이 % (사이클만 ÷ 전체 − 1)": round(100 * (s["pt"] / r["pt_cycle"] - 1), 1) if r else np.nan,
                        "사이클만 개선율": round(s["gain"], 1), "공정 전체 · 사이클 안 개선율": round(r["cycle"], 1) if r else np.nan})
        else:
            rec.update({"0.5초 개선율": round(s["gain"], 1), "1초 개선율": round(r["gain"], 1) if r else np.nan,
                        "0.5초 MAE": fmt(s["pt"]), "1초 MAE": fmt(r["pt"]) if r else "-",
                        "0.5초 레시피 평균": fmt(s["base"]), "1초 레시피 평균": fmt(r["base"]) if r else "-",
                        "0.5초 사이클 안 개선율": round(s["cycle"], 1), "1초 사이클 안 개선율": round(r["cycle"], 1) if r else np.nan})
        recs.append(rec)
    if recs:
        add(f"\n## {title}")
        add(note)
        add(pd.DataFrame(recs).to_string(index=False))


# 1-8) 학습 기간 길이: 같은 검증 구간에서 최근 run 몇 개로 학습하는 것이 나은지
def recency_sweep() -> None:
    if not len(led):
        return
    folds = {4: ("053~067", {"lc51": 51}), 3: ("044~052", {"time3": 42})}
    recs = []
    for fold, (val, cumulative) in folds.items():
        studies = {f"recent{fold}": 20, **cumulative}
        studies.update({s_: int(s_.split("_")[1]) for s_ in set(st) if s_.startswith(f"recent{fold}_")})
        if len(studies) <= 2:
            continue
        for target in sorted(set(led.loc[st.isin(list(studies)), "target"])):
            rec = {"대상": target, "검증 구간": val}
            for study, n in sorted(studies.items(), key=lambda kv: kv[1]):
                s_ = summarize(study_rows(study, target))
                rec[f"{n}개"] = s_["pt"] if s_ else np.nan
            vals = {k: v for k, v in rec.items() if k.endswith("개") and pd.notna(v)}
            if len(vals) >= 2:
                best = min(vals, key=vals.get)
                rec["가장 나은 길이"] = best
                rec["누적 대비 %"] = round(100 * (vals[best] / vals[max(vals, key=lambda k: int(k[:-1]))] - 1), 1)
                recs.append(rec)
    if recs:
        f = pd.DataFrame(recs)
        cols = sorted([c for c in f.columns if c.endswith("개") and c[:-1].isdigit()], key=lambda c: int(c[:-1]))
        f[cols] = f[cols].apply(lambda col: col.map(fmt))
        add("\n## 1-8. 학습 기간 길이 (검증 구간 바로 앞 run n개로 학습한 PatchTST 검증 MAE, 가장 긴 것은 누적 학습)")
        add("누적 대비 %가 음수면 오래된 run을 빼고 최근 run만 쓰는 편이 낫다.")
        add(f[["대상", "검증 구간", *cols, "가장 나은 길이", "누적 대비 %"]].to_string(index=False))


# 1-9) 학습 run 수 곡선 두 번째 추첨: 같은 크기에서 어떤 run이 들어가느냐의 영향
def learning_curve_draws() -> None:
    rows_ = led[st.str.fullmatch(r"lc\d+b")] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for (target, study), g in rows_.groupby(["target", "study"]):
        n = int(study[2:-1])
        a, b = summarize(study_rows(f"lc{n}", target)), summarize(g)
        if a and b:
            recs.append({"대상": target, "run 수": n, "첫 추첨 MAE": a["pt"], "두 번째 추첨 MAE": b["pt"],
                         "추첨 차이 %": round(100 * (b["pt"] / a["pt"] - 1), 1),
                         "seed SD (평균)": float(np.nanmean([a["pt_sd"], b["pt_sd"]]))})
    if recs:
        f = pd.DataFrame(recs).sort_values(["대상", "run 수"])
        f["추첨 차이 ÷ seed SD"] = ((f["두 번째 추첨 MAE"] - f["첫 추첨 MAE"]).abs() / f["seed SD (평균)"]).round(1)
        for c in ("첫 추첨 MAE", "두 번째 추첨 MAE", "seed SD (평균)"):
            f[c] = f[c].map(fmt)
        add("\n## 1-9. 학습 run 수 곡선 · 두 번째 추첨 (같은 run 수, 다른 run 조합)")
        add("추첨 차이 ÷ seed SD가 2를 넘으면 run 수보다 어떤 run을 쓰느냐가 결과를 좌우한다.")
        add(f.to_string(index=False))


# 1-10) 사이클 밖만 학습
def noncycle_vs_full() -> None:
    rows_ = led[st == "region_noncycle"] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for target, g in rows_.groupby("target"):
        s_ = summarize(g)
        r, src = reference(target, g)
        if s_ is None:
            continue
        other_full = np.nan
        if r is not None:
            ref_rows = study_rows("lc51", target)
            pt = ref_rows[ref_rows["model"] == "patchtst"]
            other_full = float(pt["mae_other"].mean()) if len(pt) else np.nan
        recs.append({"대상": target, "비교 기준": src or "-", "사이클 밖만 학습 MAE": fmt(s_["pt"]),
                     "공정 전체 학습 · 사이클 밖 MAE": fmt(other_full),
                     "차이 % (사이클 밖만 ÷ 전체 − 1)": round(100 * (s_["pt"] / other_full - 1), 1) if pd.notna(other_full) else np.nan,
                     "사이클 밖만 개선율": round(s_["gain"], 1)})
    if recs:
        add("\n## 1-10. 사이클 밖만 학습 · 평가")
        add("같은 사이클 밖 window에서 잰 MAE끼리 비교한다. 두 구간 전용 모델을 합친 결과는 scripts/combine_predictions.py에 있다.")
        add(pd.DataFrame(recs).to_string(index=False))




# 1-11) 최근 run 가중 학습 · 초기 run 포함 · 사이클 window 길이
def recency_weighting() -> None:
    if not len(led):
        return
    studies = {"lc51": "가중 없음"} | {f"rw{h}": f"반감기 {h}개" for h in (40, 20, 10)}
    rows_ = led[st.isin([s_ for s_ in studies if s_ != "lc51"])]
    if not len(rows_):
        return
    recs = []
    for target in sorted(set(rows_["target"])):
        rec, sds = {"대상": target}, {}
        for study, name in studies.items():
            s_ = summarize(study_rows(study, target))
            rec[name] = s_["pt"] if s_ else np.nan
            sds[name] = s_["pt_sd"] if s_ else np.nan
        vals = {k: v for k, v in rec.items() if k != "대상" and pd.notna(v)}
        if "가중 없음" not in vals or len(vals) < 2:
            continue
        best = min(vals, key=vals.get)
        sd = float(np.sqrt(np.nanmean([sds["가중 없음"] ** 2, sds[best] ** 2])))
        rec["가장 나은 것"] = best
        rec["가중 없음 대비 %"] = round(100 * (vals[best] / vals["가중 없음"] - 1), 1)
        if best == "가중 없음":
            rec["판단"] = "가중 없음이 가장 나음"
        else:
            rec["판단"] = "확실히 나음" if vals["가중 없음"] - vals[best] > 2 * sd else "seed 편차 수준"
        recs.append(rec)
    if recs:
        f = pd.DataFrame(recs)
        cols = [n for n in studies.values() if n in f]
        f[cols] = f[cols].apply(lambda col: col.map(fmt))
        add("\n## 1-11. 최근 run 가중 학습 (학습 run 51개, 최근 run일수록 자주 뽑음 · PatchTST 검증 MAE)")
        add(f.to_string(index=False))


def early_plus_recent() -> None:
    rows_ = led[st == "early_recent"] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for target in sorted(set(rows_["target"])):
        a = summarize(study_rows("early_recent", target))
        b, c = summarize(study_rows("recent4_30", target)), summarize(study_rows("lc31", target))
        if a:
            recs.append({"대상": target, "초기 11개 + 최근 19개": fmt(a["pt"]), "최근 30개": fmt(b["pt"]) if b else "-",
                         "무작위 31개": fmt(c["pt"]) if c else "-",
                         "최근 30개 대비 %": round(100 * (a["pt"] / b["pt"] - 1), 1) if b else np.nan})
    if recs:
        add("\n## 1-12. 초기 run(001~011) 포함 효과 (검증 053~067, PatchTST 검증 MAE)")
        add("음수면 초기 run을 넣는 편이 낫다. 같은 30개 안팎에서 비교한다.")
        add(pd.DataFrame(recs).to_string(index=False))


def cycle_windows() -> None:
    rows_ = led[st.isin(["region_cycle", "region_cycle_win"])] if len(led) else led
    if not len(rows_) or not (st == "region_cycle_win").any():
        return
    pt = rows_[rows_["model"] == "patchtst"]
    t = pt.groupby(["target", "context_seconds"])["mae"].mean().unstack()
    t.columns = [f"window {c:g}초" for c in t.columns]
    t = t.loc[sorted(set(led.loc[st == "region_cycle_win", "target"]))]
    add("\n## 1-13. 사이클만 학습 · window 길이 (PatchTST 검증 MAE, 사이클 window)")
    add(table(t))
    folds = led[st.str.fullmatch(r"region_cycle(_win)?_cv\d") & (led["model"] == "patchtst")]
    if len(folds) and st.str.fullmatch(r"region_cycle_win_cv\d").any():
        f = folds.assign(fold=st[folds.index].str[-1]).groupby(["target", "fold", "context_seconds"])["mae"].mean().unstack()
        if f.shape[1] >= 2:
            lo, hi = f.columns.min(), f.columns.max()
            d = (100 * (f[hi] / f[lo] - 1)).dropna().unstack().round(1)
            d["평균"] = d.mean(axis=1).round(1)
            add(f"\n교차검증 묶음별 window {lo:g}초 → {hi:g}초 MAE 변화 % (음수면 {hi:g}초가 나음)")
            add(d.to_string())


# 2-2) 장기 예측 설정 탐색
def long_search() -> None:
    rows_ = led[st == "long_horizon_search"] if len(led) else led
    if not len(rows_):
        return
    hcols = sorted([c for c in rows_.columns if c.startswith("mae_") and c.endswith("s") and c[4:-1].isdigit()
                    and rows_[c].notna().any()], key=lambda c: int(c[4:-1]))
    add("\n## 2-2. 장기 예측 설정 탐색 (검증 MAE, 학습 모델은 seed 평균 · 바꾼 설정만 표시)")
    for target, g in rows_.groupby("target"):
        parts = []
        for model, gm in g.groupby("model"):
            if model in ("persistence", "template", "schedule"):
                parts.append(gm.groupby("model")[["mae", *hcols]].mean().rename(index=lambda m: NAMES.get(m, m)))
                continue
            pairs = gm["settings"].astype(str).map(lambda x: dict(kv.split("=", 1) for kv in x.split(" · ") if "=" in kv))
            keys = sorted({k for d in pairs for k in d if len({str(e.get(k)) for e in pairs}) > 1})
            label = pairs.map(lambda d, keys=keys, m=model: NAMES.get(m, m)
                              + (" · " + " ".join(f"{k}={d.get(k)}" for k in keys) if keys else ""))
            parts.append(gm.assign(label=label.to_numpy()).groupby("label")[["mae", *hcols]].mean())
        t = pd.concat(parts).sort_values("mae")
        t.columns = ["5개 거리 평균"] + [c[4:] for c in hcols]
        add(f"\n[{target}]")
        add(table(t))




# 1-14) 시간순 검증에서도 최근 run 가중이 통하는지
def recency_over_time() -> None:
    rows_ = led[st.str.fullmatch(r"rw\d+_time\d")] if len(led) else led
    if not len(rows_):
        return
    recs = []
    for target in sorted(set(rows_["target"])):
        for fold in (1, 2, 3):
            base = summarize(study_rows(f"time{fold}", target))
            rec = {"대상": target, "구간": fold, "가중 없음": base["pt"] if base else np.nan}
            for h in (40, 20, 10):
                s_ = summarize(study_rows(f"rw{h}_time{fold}", target))
                if s_:
                    rec[f"반감기 {h}개"] = s_["pt"]
            vals = {k: v for k, v in rec.items() if k not in ("대상", "구간") and pd.notna(v)}
            if len(vals) >= 2 and pd.notna(rec["가중 없음"]):
                best = min(vals, key=vals.get)
                rec["가장 나은 것"] = best
                rec["가중 없음 대비 %"] = round(100 * (vals[best] / rec["가중 없음"] - 1), 1)
                recs.append(rec)
    if recs:
        f = pd.DataFrame(recs)
        cols = [c for c in ("가중 없음", "반감기 40개", "반감기 20개", "반감기 10개") if c in f]
        f[cols] = f[cols].apply(lambda col: col.map(fmt))
        add("\n## 1-14. 시간순 검증 구간별 최근 run 가중 (PatchTST 검증 MAE, 구간 1 · 2 · 3 = 검증 021~031 · 032~043 · 044~052)")
        add(f.to_string(index=False))


# 1-15) run 유형별 남은 오차: 1초 묶음과 step 전환 흔들림이 겹치는 '어려움' run에서 PatchTST도 못 맞추는지
def run_type_remaining() -> None:
    if not RUN_TYPES or not len(led):
        return
    vs = led[(led["task"] == "virtual_sensor") & (st != "") & led["target"].isin(TYPE_TARGETS)]
    rows = []
    for (study, target), g in vs.groupby([st[vs.index], "target"]):
        r = per_run(g)
        if r.empty or r["type"].isna().all():
            continue
        v = remaining_by_type(r)
        if np.isnan(v["easy"]) and np.isnan(v["hard"]):
            continue
        rows.append({"study": study, "target": SHORT[target],
                     "cell": f"{v['easy']:.0f} / {v['hard']:.0f}" if v["n_easy"] and v["n_hard"]
                     else (f"{v['easy']:.0f} / -" if v["n_easy"] else f"- / {v['hard']:.0f}"),
                     "runs": f"{v['n_easy']} / {v['n_hard']}"})
    f = pd.DataFrame(rows)
    if f.empty:
        return
    add("\n## 1-15. run 유형별 남은 오차 (PatchTST ÷ 레시피 평균 × 100, run 오차 합 기준 · 쉬움 / 어려움, 055 제외)")
    add("어려움 = 1초 묶음과 step 전환 시점의 흔들림이 겹쳐 레시피 평균 오차가 큰 run (run_drift.py 4절). 값이 작을수록 PatchTST가 많이 줄임.")
    t = f.pivot_table(index="study", columns="target", values="cell", aggfunc="first")
    t = t.reindex(columns=[SHORT[c] for c in TYPE_TARGETS if SHORT[c] in t.columns]).fillna("")
    t["run 수 (쉬움 / 어려움)"] = f.groupby("study")["runs"].first()
    add(t.to_string())


# 1-16) 학습 run 구성: 검증 run(053~067)이 같은 분할끼리, 학습 run 중 '쉬움' 비율과 PatchTST 오차
COMPOSITION = r"(lc\d+b?|recent4(_\d+)?|early_recent|rtype_.+)"


def run_type_composition() -> None:
    if not RUN_TYPES or not len(led):
        return
    rows_ = led[st.str.fullmatch(COMPOSITION) & (led["model"] == "patchtst") & led["target"].isin(TYPE_TARGETS)]
    recs = []
    for (study, target), g in rows_.groupby([st[rows_.index], "target"]):
        split = experiment_split(str(g["exp_dir"].iloc[0]))
        train = [r for r in split.get("train") or [] if r in RUN_TYPES]
        if not train:
            continue
        recs.append({"study": study, "target": SHORT[target], "mae": float(g["mae"].mean()),
                     "학습 run": len(split.get("train") or []),
                     "쉬움 비율 %": round(100 * np.mean([RUN_TYPES[r] == "easy" for r in train]))})
    f = pd.DataFrame(recs)
    if f.empty or f["study"].nunique() < 3:
        return
    t = f.pivot_table(index=["study", "학습 run", "쉬움 비율 %"], columns="target", values="mae")
    t = t.reindex(columns=[SHORT[c] for c in TYPE_TARGETS if SHORT[c] in t.columns]).sort_index(level=2)
    add("\n## 1-16. 학습 run 구성과 오차 (검증 053~067, PatchTST 검증 MAE · seed 평균, 쉬움 비율 순)")
    add(table(t))
    big = f[f["학습 run"] >= 20]
    judge = []
    for target, g in big.groupby("target"):
        if g["study"].nunique() < 5:
            continue
        judge.append({"대상": target, "분할 수": len(g),
                      "쉬움 비율 ↔ 오차 (순위 상관)": round(g[["쉬움 비율 %", "mae"]].corr("spearman").iloc[0, 1], 2),
                      "학습 run 수 ↔ 오차 (순위 상관)": round(g[["학습 run", "mae"]].corr("spearman").iloc[0, 1], 2)})
    if judge:
        add("\n학습 run 20개 이상 분할만: 순위 상관이 −1에 가까우면 '쉬움' run이 많을수록 오차가 작다 (run 수보다 구성이 중요한지 본다)")
        add(pd.DataFrame(judge).set_index("대상").reindex([SHORT[c] for c in TYPE_TARGETS]).dropna(how="all").to_string())


# 1-18) step 맞춤 묶음에서 PatchTST 설정 탐색 (patch 길이 × window)
def stepbin_search() -> None:
    rows_ = led[(st == "stepbin_search") & (led["model"] == "patchtst")] if len(led) else led
    if not len(rows_):
        return
    rows_ = rows_.assign(설정=[f"patch {(p_ or {}).get('patch_seconds', '-')}초 · window {c:g}초"
                              for p_, c in zip(rows_["params"], rows_["context_seconds"], strict=True)])
    t = rows_.groupby(["target", "설정"])["mae"].agg(["mean", "std", "count"]).reset_index()
    ref = {target: summarize(study_rows("stepbin", target)) for target in t["target"].unique()}
    t["step 맞춤 최종 설정 대비 %"] = [round(100 * (m / ref[x]["pt"] - 1), 1) if ref.get(x) else np.nan
                                   for x, m in zip(t["target"], t["mean"], strict=True)]
    t[["mean", "std"]] = t[["mean", "std"]].apply(lambda c: c.map(fmt))
    add("\n## 1-18. step 맞춤 묶음 · PatchTST 설정 탐색 (검증 MAE, seed 평균 · 표준편차 · seed 수)")
    add("음수면 지금 최종 설정(116~119차, step 맞춤 432~435차)보다 낫다.")
    add(t.rename(columns={"target": "대상", "mean": "MAE", "std": "seed SD", "count": "seed"}).to_string(index=False))


# 1-17) step 맞춤 1초 묶음: 묶음 경계를 step 시작에 맞추면 run 유형 차이와 PatchTST 개선율이 어떻게 되는지
def step_binning() -> None:
    if not len(led) or not st.str.startswith("stepbin").any():
        return
    pairs = [("lc51", "stepbin")] + [(f"cv{k}", f"stepbin_cv{k}") for k in range(1, 6)] \
        + [(f"time{k}", f"stepbin_time{k}") for k in range(1, 4)]
    recs = []
    for ref, new in pairs:
        for target in sorted(set(led.loc[st == new, "target"])):
            a, b = summarize(study_rows(ref, target)), summarize(study_rows(new, target))
            if not a or not b:
                continue
            rec = {"분할": new.replace("stepbin_", "") if new != "stepbin" else "split_v2", "대상": SHORT.get(target, target),
                   "늘어난 개선율 %p": round(b["gain"] - a["gain"], 1),
                   "기존 개선율": round(a["gain"], 1), "step 맞춤 개선율": round(b["gain"], 1),
                   "기존 PatchTST MAE": fmt(a["pt"]), "step 맞춤 PatchTST MAE": fmt(b["pt"]),
                   "기존 레시피 평균 MAE": fmt(a["base"]), "step 맞춤 레시피 평균 MAE": fmt(b["base"])}
            if RUN_TYPES:
                for name, study in (("기존", ref), ("step 맞춤", new)):
                    v = remaining_by_type(per_run(study_rows(study, target)))
                    if v["n_easy"] and v["n_hard"]:
                        rec[f"{name} 기준선 어려움÷쉬움"] = round(v["base_hard"] / v["base_easy"], 2)
                        rec[f"{name} 남은 % 쉬움/어려움"] = f"{v['easy']:.0f} / {v['hard']:.0f}"
            recs.append(rec)
    if recs:
        add("\n## 1-17. step 맞춤 1초 묶음 (묶음을 step 시작에 맞춤) vs 기존 1초 묶음 (같은 분할 · 같은 설정, 각자의 레시피 평균 대비)")
        add("기준선 어려움÷쉬움 = 어려움 run 평균 레시피 평균 오차 ÷ 쉬움 run. 1에 가까워지면 run 유형 차이가 묶음 방식에서 왔다는 뜻이다. "
            "원본 0.5초 값으로 같은 기준에서 다시 잰 비교는 scripts/raw_scale.py에 있다.")
        f = pd.DataFrame(recs)
        add(f.to_string(index=False))
        if f["분할"].nunique() >= 3:
            g = f.groupby("대상")["늘어난 개선율 %p"].agg(["mean", "std", "min", "count"]).round(1)
            add("\n분할 묶음 요약 (늘어난 개선율 %p, 평균 · 표준편차 · 가장 작은 분할 · 분할 수). "
                "seed가 학습에 실패해 레시피 평균에 머문 분할(개선율 1 % 미만)이 있으면 표준편차가 커진다.")
            add(g.to_string())


vs_study("cv", "1. 가상 센서 run 단위 교차검증")
vs_study("time", "1-2. 가상 센서 시간순 검증 (앞쪽 run으로 학습 → 바로 다음 run으로 검증)")
vs_study("single_sensor", "1-3. 가상 센서 · 센서 하나만 고장 (같은 계열 센서는 입력에 남김)")
learning_curve()
recent_vs_cumulative()
versus_full("res_05s", "1-6. 0.5초 간격 학습 (1초 묶음 없음)",
            "0.5초 값은 1초 평균보다 변동이 커서 MAE끼리 직접 비교하기 어렵다. 각자의 레시피 평균 대비 개선율로 비교한다.",
            cycle=False)
versus_full("region_cycle", "1-7. 사이클 구간만 학습 · 평가",
            "같은 사이클 window에서 잰 MAE끼리 비교한다. 음수면 사이클만 따로 학습하는 편이 낫다.", cycle=True)
recency_sweep()
learning_curve_draws()
noncycle_vs_full()
recency_weighting()
early_plus_recent()
cycle_windows()
recency_over_time()
run_type_remaining()
run_type_composition()
step_binning()
stepbin_search()


# 2) 예측 거리 확장: 모델 · window별
def variant(model: str, params) -> str:
    """Settings that tell rows of one model apart within a study (RevIN off, LightGBM leaves)."""

    params = params if isinstance(params, dict) else {}
    if model == "lightgbm" and "num_leaves" in params:
        return f" · 잎 {params['num_leaves']} · 최소 {params.get('min_child_samples', '-')}"
    if str(params.get("revin", True)).lower() in ("false", "0"):
        return " · RevIN 끔"
    return ""


def long_horizon() -> None:
    if not len(led):
        return
    dedicated = {"long_horizon_mid": "30초~2분 전용", "long_horizon_far": "5~10분 전용"}
    lh = led[(st == "long_horizon") | (st.isin(list(dedicated)) & ~led["model"].isin(["persistence", "template", "schedule"]))]
    if not len(lh):
        return
    add("\n## 2. 미래값 예측 거리 확장 (검증 MAE, 학습 모델은 seed 평균)")
    add("거리 전용 모델(30초~2분 · 5~10분)은 자기 거리만 학습한 모델이다. "
        "window 수가 조금 달라(예측 거리가 짧을수록 많음) 같은 거리끼리만 비교한다.")
    hcols = sorted([c for c in lh.columns if c.startswith("mae_") and c.endswith("s") and c[4:-1].isdigit()
                    and lh[c].notna().any()], key=lambda c: int(c[4:-1]))
    cols = hcols + ["mae_cycle", "mae_other"]
    labels = [c[4:] for c in hcols] + ["사이클 안", "사이클 밖"]
    lh = lh.assign(key=[NAMES.get(m, m) if m in ("persistence", "template", "schedule")
                        else f"{NAMES.get(m, m)} · window {c:g}초" + (f" · {dedicated[s_]}" if s_ in dedicated else "")
                        + variant(m, p_)
                        for m, c, s_, p_ in zip(lh["model"], lh["context_seconds"], lh["study"].astype(str), lh["params"],
                                                strict=True)])
    for target, g in lh.groupby("target"):
        t = g.groupby("key")[cols].mean()
        base = [k for k in ("현재값 유지", "레시피 평균", "레시피 일정 평균") if k in t.index]
        order = base + sorted([k for k in t.index if k not in base],
                              key=lambda k: (next((i for i, m in enumerate(MODEL_ORDER) if k.startswith(NAMES[m])), 99), k))
        t = t.reindex(order)
        t.columns = labels
        add(f"\n[{target}]")
        add(table(t))
        for k in order[len(base):]:
            parts = []
            for b in base:
                g1 = 100 * (1 - t.loc[k] / t.loc[b])
                parts.append(f"{b} 대비 " + "  ".join(f"{c} {v:+.0f}" for c, v in g1.items() if pd.notna(v)))
            add(f"{k} 개선 %: " + " | ".join(parts))


long_horizon()
long_search()


# 3) 기본 결과(부가 실험 제외): 대상 × 모델별 최고 설정의 전체 · 사이클 안 · 사이클 밖 개선율
def best_by_model() -> None:
    if not len(MAIN):
        return
    baseline = {"virtual_sensor": "template", "forecast": "persistence"}
    recs = []
    for _, g in MAIN[MAIN["task"].isin(list(baseline))].groupby("exp_dir"):
        bm = baseline[g["task"].iloc[0]]
        for target, gt in g.groupby("target"):
            b = gt[gt["model"] == bm]
            if b.empty:
                continue
            b0 = b.iloc[0]
            for r in gt[~gt["model"].isin(["persistence", "template", "schedule"])].itertuples():
                recs.append({"task": r.task, "target": target, "model": r.model,
                             "config": f"{r.label}|{r.settings}|{r.input_set}|{r.context_seconds}",
                             "gain": 100 * (1 - r.mae / b0["mae"]),
                             "cycle": 100 * (1 - r.mae_cycle / b0["mae_cycle"]) if pd.notna(r.mae_cycle) and b0["mae_cycle"] else np.nan,
                             "other": 100 * (1 - r.mae_other / b0["mae_other"]) if pd.notna(r.mae_other) and b0["mae_other"] else np.nan})
    f = pd.DataFrame(recs)
    if f.empty:
        return
    agg = f.groupby(["task", "target", "model", "config"], as_index=False)[["gain", "cycle", "other"]].mean()
    tried = agg.groupby(["task", "target", "model"]).size().rename("설정 수")
    best = agg.loc[agg.groupby(["task", "target", "model"])["gain"].idxmax()].join(tried, on=["task", "target", "model"])
    add("\n## 3. 기본 결과의 모델별 최고 설정 · 사이클 안 / 밖 개선율 (%, 같은 실험의 기준선 대비, seed 평균)")
    add("설정을 많이 시도한 모델일수록 최고값이 유리하게 나온다 (설정 수 함께 표시). 미래값 기준선 = 현재값 유지, 가상 센서 = 레시피 평균.")
    for task, title in (("virtual_sensor", "가상 센서"), ("forecast", "미래값 예측")):
        b = best[best["task"] == task]
        if b.empty:
            continue
        models = [m for m in MODEL_ORDER if m in set(b["model"])]
        for col, name in (("gain", "전체"), ("cycle", "사이클 안"), ("other", "사이클 밖")):
            pv = b.pivot_table(index="target", columns="model", values=col).reindex(columns=models)
            pv.columns = [NAMES.get(m, m) for m in pv.columns]
            add(f"\n[{title} · {name}]")
            add(pv.round(1).to_string())
        pv = b.pivot_table(index="target", columns="model", values="설정 수").reindex(columns=models)
        pv.columns = [NAMES.get(m, m) for m in pv.columns]
        add(f"\n[{title} · 시도한 설정 수]")
        add(pv.fillna(0).astype(int).to_string())


best_by_model()


# 4) 레시피 단계별 오차 (이번 코드부터 저장되는 by_step이 있는 작업만)
def by_step(rows_: pd.DataFrame, title: str, base_model: str = "template", top: int = 12) -> None:
    if not len(rows_):
        return
    blocks = []
    for target, g in rows_.groupby("target"):
        trained = g[~g["model"].isin(["persistence", "template", "schedule"])]
        trained = trained[[bool(job_metrics(r).get("by_step")) for r in trained.itertuples()]]
        if trained.empty:
            continue
        model = "patchtst" if "patchtst" in set(trained["model"]) else trained.groupby("model")["mae"].mean().idxmin()
        acc: dict[str, list] = {}
        for r in trained[trained["model"] == model].itertuples():
            b = g[(g["model"] == base_model) & (g["exp_dir"] == r.exp_dir)]
            if b.empty:
                continue
            floor = 0.02 * float(b["mae"].iloc[0])  # steps where the signal is idle: a ratio of near-zero errors says nothing
            sb, sp = job_metrics(next(b.itertuples())).get("by_step") or {}, job_metrics(r).get("by_step") or {}
            for step, (mae, n) in sp.items():
                if step in sb and sb[step][0] > floor:
                    a = acc.setdefault(step, [[], [], n])
                    a[0].append(sb[step][0])
                    a[1].append(mae)
        if not acc:
            continue
        t = pd.DataFrame([{"단계": k, "window 수": int(v[2]), "기준선 MAE": float(np.mean(v[0])),
                           "모델 MAE": float(np.mean(v[1]))} for k, v in acc.items()])
        t["남은 오차 %"] = (100 * t["모델 MAE"] / t["기준선 MAE"]).round(0)
        t = t.sort_values("window 수", ascending=False).head(top)
        t[["기준선 MAE", "모델 MAE"]] = t[["기준선 MAE", "모델 MAE"]].apply(lambda s: s.map(fmt))
        blocks.append(f"\n[{target} · {NAMES.get(model, model)} ÷ {NAMES[base_model]}]\n" + t.to_string(index=False))
    if blocks:
        add(f"\n## {title}")
        add("window 수가 많은 단계 순 (PatchTST가 있으면 PatchTST). 기준선 오차가 대상 전체 기준선 오차의 2 % 미만인 단계(신호가 쉬는 단계)는 뺐다. "
            "남은 오차 %가 100을 넘는 단계는 기준선보다 못한 곳이다.")
        out.extend(blocks)


if len(led):
    by_step(led[st == "lc51"], "4. 레시피 단계별 남은 오차 · 가상 센서 (학습 run 51개)")
    by_step(led[st == "long_horizon"], "4-2. 레시피 단계별 남은 오차 · 미래값 30초~10분 (레시피 평균 대비, 최고 모델)")

text = "\n".join(out).strip() if out else "부가 실험 결과가 아직 없습니다."
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "side_studies.txt").write_text(text, encoding="utf-8")
print(text)
