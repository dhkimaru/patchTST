"""최종 모델의 정확도(%FS)와 설명력(R²): 검증 run · 시험 run. 학습 없이 work/만 읽는다.

정확도(%FS) = 100 − MAE ÷ 측정 범위 × 100. 측정 범위 = 평가 구간 실제값의 0.5~99.5 백분위 폭 (튀는 값 제외).
사이클 정확도 = 같은 계산을 ALD 사이클 구간에만 적용 (대기압까지 오가는 신호는 전체 범위가 너무 넓어 정확도가 부풀려지므로 함께 본다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ald_lab.data.splits import load_split
from ald_lab.paths import resolve_paths
from ald_lab.training.config import load_experiment

FINAL = {
    "20261102-230426-801b0": "미래값", "20261102-230426-82997": "미래값", "20261103-104027-6548b": "미래값",
    "20261102-230426-88543": "미래값", "20261102-230426-9287b": "미래값", "20261102-230426-9693d": "미래값",
    "20261102-230427-02847": "미래값", "20261103-135440-740ec": "가상센서", "20261103-135440-82261": "가상센서",
    "20261103-135440-875aa": "가상센서", "20261103-135440-93381": "가상센서", "20261103-135440-97718": "가상센서",
    "20261103-135441-0468a": "가상센서",
}
if len(sys.argv) > 1:  # 다른 요청 id를 쓸 때: python accuracy.py id1=미래값 id2=가상센서
    FINAL = dict(a.split("=", 1) for a in sys.argv[1:])
P = resolve_paths()
_ranges: dict = {}


def value_range(cfg, partition: str, target: str, cycle: bool = False) -> tuple[float, float]:
    split_id = load_split(cfg.split_path()).split_id
    key = (split_id, partition, target, cfg.region, cycle)
    if key not in _ranges:
        dirs = sorted(P.work_root.glob(f"prepared/{split_id}_{cfg.sample_seconds:g}s_{partition}_*/meta.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        lo = hi = np.nan
        for meta in dirs:
            cols = json.loads(meta.read_text(encoding="utf-8"))["columns"]
            if target not in cols:
                continue
            v = np.load(meta.parent / "values.npy", mmap_mode="r")[:, cols.index(target)]
            if cycle:
                v = v[np.load(meta.parent / "in_cycle.npy")]
            elif cfg.region == "process":
                v = v[np.load(meta.parent / "in_process.npy")]
            lo, hi = np.percentile(np.asarray(v, dtype=np.float64), [0.5, 99.5])
            break
        _ranges[key] = (float(lo), float(hi))
    return _ranges[key]


rows = []
for rid, task in FINAL.items():
    req = json.loads((P.work_root / "queue" / rid / "request.json").read_text(encoding="utf-8"))
    exp = Path(req["exp_dir"])
    cfg = load_experiment(exp / "experiment.yaml")
    for job in json.loads((exp / "jobs.json").read_text(encoding="utf-8")):
        jd = exp / "jobs" / job["job_id"]
        if not (jd / "metrics_validation.json").is_file():
            continue
        val = json.loads((jd / "metrics_validation.json").read_text(encoding="utf-8"))
        unit = (val.get("result") or {}).get("unit") or ""
        for part, m in (("validation", val.get("metrics") or {}),
                        ("test", json.loads((jd / "metrics_test.json").read_text(encoding="utf-8"))
                         if (jd / "metrics_test.json").is_file() else None)):
            if not m:
                continue
            lo, hi = value_range(cfg, part, job["target"])
            clo, chi = value_range(cfg, part, job["target"], cycle=True)
            rows.append({"과제": task, "대상": job["target"], "단위": unit, "모델": job["model"], "구분": part,
                         "mae": m.get("mae"), "r2": m.get("r2"), "lo": lo, "hi": hi,
                         "mae_c": (m.get("by_region") or {}).get("cycle"), "clo": clo, "chi": chi})

df = pd.DataFrame(rows)
df["kind"] = np.where(df["모델"] == "patchtst", "PatchTST", "기준선")
g = df.groupby(["과제", "대상", "단위", "kind", "구분"], as_index=False).agg(
    mae=("mae", "mean"), r2=("r2", "mean"), lo=("lo", "first"), hi=("hi", "first"),
    mae_c=("mae_c", "mean"), clo=("clo", "first"), chi=("chi", "first"))
g["정확도"] = 100 * (1 - g["mae"] / (g["hi"] - g["lo"]))
g["사이클정확도"] = 100 * (1 - g["mae_c"] / (g["chi"] - g["clo"]))
out = []
for task in ("미래값", "가상센서"):
    t = g[g["과제"] == task]
    if t.empty:
        continue
    out.append(f"\n## {task}: 정확도(%FS) · 설명력(R²) · MAE (PatchTST는 seed 평균)")
    lines = []
    for (target, unit), d in t.groupby(["대상", "단위"]):
        def pick(kind, part, col, d=d):
            r = d[(d["kind"] == kind) & (d["구분"] == part)]
            return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan
        tr = d[d["구분"] == "test"]
        lo, hi = (float(tr["lo"].iloc[0]), float(tr["hi"].iloc[0])) if len(tr) else (np.nan, np.nan)
        lines.append({"대상": target, "단위": unit, "측정 범위(시험)": f"{lo:.4g} ~ {hi:.4g}",
                      "정확도 검증": round(pick("PatchTST", "validation", "정확도"), 2),
                      "정확도 시험": round(pick("PatchTST", "test", "정확도"), 2),
                      "사이클 정확도 시험": round(pick("PatchTST", "test", "사이클정확도"), 2),
                      "R² 검증": round(pick("PatchTST", "validation", "r2"), 3),
                      "R² 시험": round(pick("PatchTST", "test", "r2"), 3),
                      "MAE 시험": f"{pick('PatchTST', 'test', 'mae'):.4g}",
                      "기준선 정확도 시험": round(pick("기준선", "test", "정확도"), 2),
                      "기준선 R² 시험": round(pick("기준선", "test", "r2"), 3)})
    out.append(pd.DataFrame(lines).sort_values("정확도 시험", ascending=False).to_string(index=False))
text = "\n".join(out).strip()
(P.work_root / "reports").mkdir(parents=True, exist_ok=True)
(P.work_root / "reports" / "accuracy.txt").write_text(text, encoding="utf-8")
print(text)
