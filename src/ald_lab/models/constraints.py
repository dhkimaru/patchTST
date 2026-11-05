"""Settings a model would reject, checked before a job is queued or a search combination is drawn (no torch)."""

from __future__ import annotations

from typing import Any

from . import defaults


def combo_problems(name: str, params: dict[str, Any], context_seconds: float) -> list[str]:
    """Why ``params`` cannot build model ``name`` for a window of ``context_seconds`` (empty when fine)."""

    problems = []
    if name in ("patchtst", "itransformer"):
        d_model = int(params.get("d_model", defaults.DEEP[name]["d_model"]))
        heads = int(params.get("n_heads", defaults.DEEP[name]["n_heads"]))
        if heads <= 0 or d_model % heads:
            problems.append(f"{name}: d_model {d_model}은 head 수 {heads}로 나누어떨어져야 합니다")
    if name == "patchtst":
        patch = float(params.get("patch_seconds", defaults.DEEP["patchtst"]["patch_len"]))
        stride = float(params.get("stride_seconds", defaults.DEEP["patchtst"]["stride"]))
        if patch > context_seconds:
            problems.append(f"patchtst: patch {patch:g}초가 window {context_seconds:g}초보다 깁니다")
        if stride > patch:
            problems.append(f"patchtst: patch 간격 {stride:g}초가 patch {patch:g}초보다 깁니다")
    if name == "dlinear" and float(params.get("kernel_seconds", defaults.DEEP["dlinear"]["kernel"])) > context_seconds:
        problems.append("dlinear: 이동평균 길이가 window보다 깁니다")
    for key, value in params.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0 and key != "dropout":
            problems.append(f"{name}: {key}는 0보다 커야 합니다 ({value})")
    return problems
