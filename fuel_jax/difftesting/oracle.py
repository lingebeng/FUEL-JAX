from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping

import numpy as np
import torch


class Oracle(Enum):
    PASS = auto()  # 通过✅
    WARN = auto()  # 警告⚠️
    FAIL = auto()  # 失败☹️


ORACLE_RANK: dict[Oracle, int] = {
    Oracle.PASS: 0,
    Oracle.WARN: 1,
    Oracle.FAIL: 2,
}


@dataclass(frozen=True)
class DiffResult:
    oracle: Oracle
    message: str
    metrics: dict[str, float | int]


def oracle_rank(oracle: Oracle) -> int:
    return ORACLE_RANK[oracle]


def format_metrics(metrics: Mapping[str, float | int]) -> str:
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={value:.6g}")

    return "\n".join(parts)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 and y_norm == 0.0:
        return 1.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    sim = float(np.dot(x, y) / (x_norm * y_norm))
    return float(np.clip(sim, -1.0, 1.0))


def _empty_metrics(total_count: int) -> dict[str, float | int]:
    return {
        "total_count": total_count,
        "compared_count": 0,
        "max_abs_diff": 0.0,
        "p99_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_ulp_diff": 0.0,
        "p99_ulp_diff": 0.0,
        "mean_ulp_diff": 0.0,
        "max_rel_diff": 0.0,
        "p99_rel_diff": 0.0,
        "cosine_sim": 1.0,
        "close_mismatch_ratio": 0.0,
        "nonfinite_mismatch_ratio": 0.0,
    }


def _exceeded(
    metrics: Mapping[str, float | int], thresholds: Mapping[str, float | int]
) -> list[str]:
    exceeded: list[str] = []
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 0.0))
        if key == "cosine_sim":
            if value < float(threshold):
                exceeded.append(f"{key}={value:.6g}<{float(threshold):.6g}")
            continue
        if value > float(threshold):
            exceeded.append(f"{key}={value:.6g}>{float(threshold):.6g}")
    return exceeded


def _precision_to_torch_dtype(precision: str) -> torch.dtype:
    dtype_map: dict[str, torch.dtype] = {
        "FP32": torch.float32,
        "BF16": torch.bfloat16,
        "FP8_E4M3": torch.float8_e4m3fn,
        "FP8_E5M2": torch.float8_e5m2,
    }
    if precision not in dtype_map:
        raise ValueError(f"Unsupported precision for ULP metrics: {precision}")
    return dtype_map[precision]


def _quantized_ulp_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
    precision: str,
) -> np.ndarray:
    target_dtype = _precision_to_torch_dtype(precision)

    # Quantize to target dtype first, then compare bit-level distance in that format.
    tx = torch.as_tensor(x).to(dtype=target_dtype).reshape(-1)
    ty = torch.as_tensor(y).to(dtype=target_dtype).reshape(-1)
    if tx.numel() == 0:
        return np.array([], dtype=np.float64)

    if target_dtype == torch.float32:
        raw_x = tx.view(torch.int32).to(torch.int64)
        raw_y = ty.view(torch.int32).to(torch.int64)
        nbits = 32
    elif target_dtype == torch.bfloat16:
        raw_x = tx.view(torch.int16).to(torch.int64)
        raw_y = ty.view(torch.int16).to(torch.int64)
        nbits = 16
    else:
        raw_x = tx.view(torch.uint8).to(torch.int64)
        raw_y = ty.view(torch.uint8).to(torch.int64)
        nbits = 8

    full_mask = (1 << nbits) - 1
    sign_mask = 1 << (nbits - 1)

    ux = torch.bitwise_and(raw_x, full_mask)
    uy = torch.bitwise_and(raw_y, full_mask)

    ordered_x = torch.where(
        torch.ne(torch.bitwise_and(ux, sign_mask), 0),
        full_mask - ux,
        ux + sign_mask,
    )
    ordered_y = torch.where(
        torch.ne(torch.bitwise_and(uy, sign_mask), 0),
        full_mask - uy,
        uy + sign_mask,
    )

    ulp = torch.abs(ordered_x - ordered_y).to(torch.float64)
    return ulp.cpu().numpy()


def evaluate_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
    precision: str,
    atol: float,
    rtol: float,
    criteria: Mapping[str, Mapping[str, float | int]],
) -> DiffResult:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.shape != y_arr.shape:
        return DiffResult(
            oracle=Oracle.FAIL,
            message=f"shape mismatch: {x_arr.shape} vs {y_arr.shape}",
            metrics={
                "shape_mismatch": 1,
                "total_count_x": int(x_arr.size),
                "total_count_y": int(y_arr.size),
            },
        )

    total_count = int(x_arr.size)
    metrics = _empty_metrics(total_count)
    if total_count == 0:
        return DiffResult(
            oracle=Oracle.PASS,
            message="empty tensors (nothing to compare)",
            metrics=metrics,
        )

    x64 = np.asarray(x_arr, dtype=np.float64)
    y64 = np.asarray(y_arr, dtype=np.float64)

    nonfinite = ~np.isfinite(x64) | ~np.isfinite(y64)
    same_nonfinite = (
        (np.isnan(x64) & np.isnan(y64))
        | (np.isposinf(x64) & np.isposinf(y64))
        | (np.isneginf(x64) & np.isneginf(y64))
    )
    nonfinite_mismatch = nonfinite & ~same_nonfinite
    metrics["nonfinite_mismatch_ratio"] = float(np.mean(nonfinite_mismatch))

    finite_mask = np.isfinite(x64) & np.isfinite(y64)
    xf = x64[finite_mask]
    yf = y64[finite_mask]
    metrics["compared_count"] = int(xf.size)

    if xf.size > 0:
        abs_diff = np.abs(xf - yf)
        rel_scale = np.maximum(np.maximum(np.abs(xf), np.abs(yf)), float(atol))
        rel_diff = abs_diff / rel_scale
        close_mask = np.isclose(xf, yf, atol=atol, rtol=rtol, equal_nan=True)
        ulp_diff = _quantized_ulp_diff(xf, yf, precision=precision)

        metrics["max_abs_diff"] = float(np.max(abs_diff))
        metrics["p99_abs_diff"] = _safe_percentile(abs_diff, 99.0)
        metrics["mean_abs_diff"] = float(np.mean(abs_diff))
        metrics["max_ulp_diff"] = float(np.max(ulp_diff)) if ulp_diff.size > 0 else 0.0
        metrics["p99_ulp_diff"] = _safe_percentile(ulp_diff, 99.0)
        metrics["mean_ulp_diff"] = (
            float(np.mean(ulp_diff)) if ulp_diff.size > 0 else 0.0
        )
        metrics["max_rel_diff"] = float(np.max(rel_diff))
        metrics["p99_rel_diff"] = _safe_percentile(rel_diff, 99.0)
        metrics["cosine_sim"] = _cosine_sim(xf, yf)
        metrics["close_mismatch_ratio"] = float(np.mean(~close_mask))

    pass_exceeded = _exceeded(metrics, criteria["pass"])
    warn_exceeded = _exceeded(metrics, criteria["warn"])

    if not pass_exceeded:
        return DiffResult(
            oracle=Oracle.PASS,
            message="all metrics are within PASS thresholds",
            metrics=metrics,
        )
    if not warn_exceeded:
        return DiffResult(
            oracle=Oracle.WARN,
            message="some metrics exceed PASS thresholds but remain in WARN range",
            metrics=metrics,
        )

    return DiffResult(
        oracle=Oracle.FAIL,
        message="metrics exceed WARN thresholds: " + "; ".join(warn_exceeded[:3]),
        metrics=metrics,
    )
