from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping

import numpy as np


class Oracle(Enum):
    PASS = auto()
    WARN = auto()
    FAIL = auto()


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
    return ", ".join(parts)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _ulp_distance_float32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)

    # Collapse +0.0 and -0.0 into the same bucket before viewing bits.
    a32 = np.where(a32 == 0.0, np.float32(0.0), a32)
    b32 = np.where(b32 == 0.0, np.float32(0.0), b32)

    ai = a32.view(np.int32).astype(np.int64)
    bi = b32.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 0x80000000 - ai, ai)
    bi = np.where(bi < 0, 0x80000000 - bi, bi)
    return np.abs(ai - bi)


def _empty_metrics(total_count: int) -> dict[str, float | int]:
    return {
        "total_count": total_count,
        "compared_count": 0,
        "max_abs_diff": 0.0,
        "p99_abs_diff": 0.0,
        "p90_abs_diff": 0.0,
        "max_ulp_diff": 0.0,
        "p99_ulp_diff": 0.0,
        "cosine_distance": 0.0,
        "max_ratio_diff": 0.0,
        "p99_ratio_diff": 0.0,
        "p90_ratio_diff": 0.0,
        "close_mismatch_count": 0,
        "nonfinite_mismatch_ratio": 0.0,
    }


def _exceeded(
    metrics: Mapping[str, float | int], thresholds: Mapping[str, float | int]
) -> list[str]:
    exceeded: list[str] = []
    for key, threshold in thresholds.items():
        value = float(metrics.get(key, 0.0))
        if value > threshold:
            exceeded.append(f"{key}={value:.6g}>{threshold:.6g}")
    return exceeded


def evaluate_diff(
    x: np.ndarray,
    y: np.ndarray,
    *,
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
        ratio_scale = atol + rtol * np.maximum(np.abs(xf), np.abs(yf))
        ratio_scale = np.where(
            ratio_scale > 0.0, ratio_scale, np.finfo(np.float64).tiny
        )
        ratio_diff = abs_diff / ratio_scale

        ulp_diff = _ulp_distance_float32(xf, yf).astype(np.float64)
        close_mask = np.isclose(xf, yf, atol=atol, rtol=rtol, equal_nan=True)

        metrics["max_abs_diff"] = float(np.max(abs_diff))
        metrics["p99_abs_diff"] = _safe_percentile(abs_diff, 99.0)
        metrics["p90_abs_diff"] = _safe_percentile(abs_diff, 90.0)
        metrics["max_ulp_diff"] = float(np.max(ulp_diff))
        metrics["p99_ulp_diff"] = _safe_percentile(ulp_diff, 99.0)
        metrics["cosine_distance"] = _cosine_distance(xf, yf)
        metrics["max_ratio_diff"] = float(np.max(ratio_diff))
        metrics["p99_ratio_diff"] = _safe_percentile(ratio_diff, 99.0)
        metrics["p90_ratio_diff"] = _safe_percentile(ratio_diff, 90.0)
        metrics["close_mismatch_count"] = int(np.count_nonzero(~close_mask))

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


def _cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 and y_norm == 0.0:
        return 0.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 1.0
    cosine_sim = float(np.dot(x, y) / (x_norm * y_norm))
    cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
    return 1.0 - cosine_sim
