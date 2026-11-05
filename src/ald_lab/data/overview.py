"""Build the raw-data overview the dashboard's first page reads.

One pass over every trace run produces: a run catalog, a column profile merged
with the column dictionary, near-copy pairs, the step structure, the ex-situ
metrology linked to runs, and a list of data issues. Outputs are parquet/JSON
files in ``<work>/data_overview``; a ``data`` journal entry records the pass.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..journal import Journal
from ..paths import LabPaths
from .catalog import build_catalog, catalog_run, step_segments, step_table
from .columns import DEFAULT_COLUMN_DICTIONARY, core_columns, load_column_dictionary
from .exsitu import link_to_traces, parse_exsitu_workbook, radial_profile
from .profile import ColumnProfiler
from .raw import list_trace_files, load_trace, run_id_from_path, signal_columns

# Robust z-score beyond which a run's mean thickness is flagged as an outlier.
THICKNESS_OUTLIER_Z = 5.0
# Å / %-points; workbook summaries differing more than this are reported.
SHEET_CHECK_TOLERANCE = 0.01


@dataclass(frozen=True)
class OverviewResult:
    output_dir: Path
    summary: dict[str, object]


def _fill_cache(args: tuple[Path, Path]) -> str:
    source, cache_dir = args
    load_trace(source, cache_dir)
    return source.name


def _issues(catalog: pd.DataFrame, exsitu: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_no in catalog.attrs.get("missing_run_numbers", []):
        rows.append(
            {"severity": "info", "run_id": f"standard_trace_{run_no:03d}", "kind": "MISSING_RUN_FILE",
             "detail": "번호 순서상 비어 있는 run (파일 없음)"}
        )
    for run in catalog.itertuples():
        if run.n_backward_jumps:
            rows.append(
                {"severity": "critical", "run_id": run.run_id, "kind": "BACKWARD_TIMESTAMP_JUMP",
                 "detail": f"시간 역행 {run.n_backward_jumps}회 (최대 {run.min_backward_s:.0f}s). 중복 블록 의심"}
            )
        if run.n_gaps:
            rows.append(
                {"severity": "warning", "run_id": run.run_id, "kind": "SAMPLING_GAP",
                 "detail": f"0.5s 간격 끊김 {run.n_gaps}회 (최대 {run.max_gap_s:.1f}s)"}
            )
    cycles = catalog["n_cycles"]
    typical = cycles.mode().iloc[0]
    for run in catalog[(cycles != typical)].itertuples():
        rows.append(
            {"severity": "info", "run_id": run.run_id, "kind": "CYCLE_COUNT",
             "detail": f"ALD 사이클(SI-FL1) {run.n_cycles}회 (대부분 {typical}회)"}
        )
    if exsitu is not None and not exsitu.empty:
        linked = exsitu[exsitu["run_id"].notna()]
        for run_id in sorted(set(catalog["run_id"]) - set(linked["run_id"])):
            rows.append(
                {"severity": "warning", "run_id": run_id, "kind": "NO_EXSITU",
                 "detail": "연결되는 Ex-situ 측정 블록 없음"}
            )
        # Outliers are judged within one measurement layout only (9pt vs 49pt differ).
        for layout, thk in linked.dropna(subset=["thk_mean"]).groupby("layout"):
            if len(thk) < 5:
                continue
            median = thk["thk_mean"].median()
            mad = (thk["thk_mean"] - median).abs().median() * 1.4826
            for block in thk.itertuples():
                z = (block.thk_mean - median) / mad if mad > 0 else 0.0
                if abs(z) >= THICKNESS_OUTLIER_Z:
                    rows.append(
                        {"severity": "warning", "run_id": block.run_id, "kind": "THICKNESS_OUTLIER",
                         "detail": f"평균 두께 {block.thk_mean:.1f}Å ({layout} 중앙값 {median:.1f}Å, "
                                   f"robust z={z:+.0f}) · {block.title}"}
                    )
        mismatched = exsitu[exsitu["sheet_check_max_abs_diff"] > SHEET_CHECK_TOLERANCE]
        for block in mismatched.itertuples():
            rows.append(
                {"severity": "info", "run_id": block.run_id if isinstance(block.run_id, str) else "(trace 없음)",
                 "kind": "EXSITU_SHEET_FORMULA",
                 "detail": f"워크북 요약값이 원시 점에서 다시 계산한 값과 최대 {block.sheet_check_max_abs_diff:.3f} 다름 "
                           f"(수식 범위 오류 등) · {block.sheet} / {block.title}. 대시보드는 다시 계산한 값을 씀"}
            )
        for block in linked[linked["normal_end"] == False].itertuples():  # noqa: E712
            rows.append(
                {"severity": "warning", "run_id": block.run_id, "kind": "ABNORMAL_END",
                 "detail": f"Ex-situ 기록상 NORMAL END = X · {block.title}"}
            )
        for block in linked[linked["title_consistent"] == False].itertuples():  # noqa: E712
            rows.append(
                {"severity": "critical", "run_id": block.run_id, "kind": "EXSITU_TITLE_MISMATCH",
                 "detail": f"시간으로 연결된 블록의 제목 번호가 다름 · {block.title}"}
            )
        for block in linked[linked["link_method"].str.startswith("title", na=False)].itertuples():
            rows.append(
                {"severity": "info", "run_id": block.run_id, "kind": "EXSITU_DATE_MISMATCH",
                 "detail": f"Ex-situ 날짜({block.start:%Y-%m-%d})가 trace와 달라 제목 번호로 연결"}
            )
    return pd.DataFrame(rows, columns=["severity", "run_id", "kind", "detail"])


def build_data_overview(
    paths: LabPaths,
    *,
    column_dictionary: Path = DEFAULT_COLUMN_DICTIONARY,
    workers: int = 4,
    journal: Journal | None = None,
) -> OverviewResult:
    data_root = paths.require_data_root()
    out = paths.overview_dir
    out.mkdir(parents=True, exist_ok=True)
    journal = journal or Journal(paths.journal_dir)
    files = list_trace_files(data_root)
    config = {
        "data_root": str(data_root),
        "exsitu_xlsx": str(paths.exsitu_xlsx) if paths.exsitu_xlsx else None,
        "column_dictionary": str(column_dictionary),
        "n_files": len(files),
    }
    with journal.record("data", "원본 데이터 개요 생성", config=config, tags=["overview"]) as entry:
        jobs = [(f, paths.trace_cache) for f in files]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_fill_cache, jobs))
        else:
            for job in jobs:
                _fill_cache(job)

        profiler: ColumnProfiler | None = None
        rows, segments_by_run = [], {}
        for source in files:
            frame = load_trace(source, paths.trace_cache)
            run_id = run_id_from_path(source)
            if profiler is None:
                profiler = ColumnProfiler(signal_columns(frame))
            profiler.add_run(frame)
            rows.append(catalog_run(run_id, frame, source))
            segments_by_run[run_id] = step_segments(frame)
        assert profiler is not None
        catalog = build_catalog(rows)

        dictionary, parameters = load_column_dictionary(column_dictionary)
        profile = profiler.result()
        columns = profile.merge(
            dictionary.drop(columns=["status", "duplicate_of"]), on="column", how="left"
        )
        undocumented = columns.loc[columns["group"].isna(), "column"].tolist()
        near = profiler.near_copy_pairs()
        steps = step_table(segments_by_run)

        exsitu_blocks = exsitu_points = None
        # A wrong workbook path must not cost the whole stage-1 overview: carry on without thickness tables.
        exsitu_missing = paths.exsitu_xlsx is not None and not paths.exsitu_xlsx.is_file()
        if paths.exsitu_xlsx is not None and not exsitu_missing:
            tables = parse_exsitu_workbook(paths.exsitu_xlsx)
            exsitu_blocks = link_to_traces(tables.runs, catalog)
            exsitu_points = tables.points

        issues = _issues(catalog, exsitu_blocks)
        if exsitu_missing:
            print(f"!! Ex-situ 워크북이 없습니다: {paths.exsitu_xlsx} — 두께 표 없이 계속합니다 "
                  "(scripts/env.sh의 ALD_EXSITU_XLSX 확인)", flush=True)
            issues = pd.concat([issues, pd.DataFrame([{
                "severity": "warning", "run_id": "-", "kind": "EXSITU_MISSING",
                "detail": f"Ex-situ 워크북을 찾지 못함: {paths.exsitu_xlsx} (두께 과제는 실행되지 않음)"}])],
                ignore_index=True)

        catalog_out = catalog.copy()
        catalog_out["recipe_rows"] = catalog_out["recipe_rows"].map(json.dumps)
        catalog_out["quality_flags"] = catalog_out["quality_flags"].map(",".join)
        catalog_out.to_parquet(out / "runs.parquet", index=False)
        columns.to_parquet(out / "columns.parquet", index=False)
        parameters.to_parquet(out / "parameters.parquet", index=False)
        near.to_parquet(out / "near_copies.parquet", index=False)
        steps.to_parquet(out / "steps.parquet", index=False)
        issues.to_parquet(out / "issues.parquet", index=False)
        if exsitu_blocks is not None:
            exsitu_blocks.to_parquet(out / "exsitu_blocks.parquet", index=False)
            exsitu_points.to_parquet(out / "exsitu_points.parquet", index=False)
            radial_profile(exsitu_points).to_parquet(out / "exsitu_radial.parquet", index=False)

        status_counts = columns["status"].value_counts().to_dict()
        linked = (
            int(exsitu_blocks["run_id"].nunique()) if exsitu_blocks is not None else 0
        )
        summary: dict[str, object] = {
            "n_runs": int(len(catalog)),
            "n_rows": int(catalog["rows"].sum()),
            "n_signal_columns": int(len(columns)),
            "column_status": {k: int(v) for k, v in status_counts.items()},
            "first_start": str(catalog["start"].min()),
            "last_end": str(catalog["end"].max()),
            "total_hours": float(catalog["duration_h"].sum()),
            "median_run_hours": float(catalog["duration_h"].median()),
            "sampling_interval_s": 0.5,
            "missing_run_numbers": catalog.attrs.get("missing_run_numbers", []),
            "n_unique_steps": int(steps["step_id"].nunique()),
            "n_recipes": int(steps["recipe"].nunique()),
            "core_columns_A": core_columns(dictionary),
            "undocumented_columns": undocumented,
            "n_near_copy_pairs": int(len(near)),
            "exsitu_blocks": int(len(exsitu_blocks)) if exsitu_blocks is not None else 0,
            "exsitu_layouts": (
                {k: int(v) for k, v in exsitu_blocks["layout"].value_counts().items()}
                if exsitu_blocks is not None else {}
            ),
            "exsitu_linked_runs": linked,
            "trace_cache": str(paths.trace_cache),
            "issues_by_severity": issues["severity"].value_counts().to_dict(),
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        entry.summary = (
            f"{summary['n_runs']}개 run, {summary['n_rows']:,}행, 신호 컬럼 {summary['n_signal_columns']}개를 "
            f"점검했습니다. 상수 {status_counts.get('CONSTANT', 0)}개, 완전 중복 "
            f"{status_counts.get('DUPLICATE', 0)}개, |r|≥0.999 근사 복사 쌍 {len(near)}개."
        )
        entry.metrics = {
            "n_runs": summary["n_runs"],
            "n_rows": summary["n_rows"],
            "n_signal_columns": summary["n_signal_columns"],
            "exsitu_linked_runs": linked,
            "n_issues": int(len(issues)),
        }
        entry.artifacts = {"overview_dir": str(out)}
        regular = catalog["regular_fraction"].min()
        entry.strengths = [
            f"전 run이 0.5초 간격 (최소 정상 비율 {regular:.4%})",
            f"Ex-situ 두께와 {linked}개 run 연결",
        ]
        entry.weaknesses = [f"[{i.severity}] {i.run_id}: {i.detail}" for i in issues.itertuples()
                            if i.severity != "info"]
        summary["journal_entry_id"] = entry.entry_id
    return OverviewResult(output_dir=out, summary=summary)


def load_overview(output_dir: Path) -> dict[str, object]:
    """Read every overview table that exists (dashboard helper)."""

    tables: dict[str, object] = {}
    for name in ("runs", "columns", "parameters", "near_copies", "steps", "issues",
                 "exsitu_blocks", "exsitu_points", "exsitu_radial"):
        path = output_dir / f"{name}.parquet"
        tables[name] = pd.read_parquet(path) if path.is_file() else None
    summary_path = output_dir / "summary.json"
    tables["summary"] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    return tables


__all__ = ["OverviewResult", "build_data_overview", "load_overview"]
