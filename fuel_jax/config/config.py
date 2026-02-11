from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


PRECISION_MAP: dict[str, dict[str, str]] = {
    "FP32": {"jax": "jnp.float32", "torch": "torch.float32"},
    "BF16": {"jax": "jnp.bfloat16", "torch": "torch.bfloat16"},
    "FP8_E4M3": {"jax": "jnp.float8_e4m3fn", "torch": "torch.float8_e4m3fn"},
    "FP8_E5M2": {"jax": "jnp.float8_e5m2", "torch": "torch.float8_e5m2"},
}


TOLERANCE: dict[str, dict[str, float]] = {
    "FP32": {"atol": 1.3e-6, "rtol": 1e-5},
    "BF16": {"atol": 0.016, "rtol": 1e-5},
    "FP8_E4M3": {"atol": 1e-1, "rtol": 1e-1},
    "FP8_E5M2": {"atol": 1e-1, "rtol": 1e-1},
}

# Multi-metric oracle thresholds.
# PASS: no obvious issue.
# WARN: potential issue, worth manual inspection.
# FAIL: likely wrong output.
# TODO @haifeng 按理说还应该针对函数类型，比如超越函数等，其容忍应该放得更宽松一些
DIFF_ORACLE_THRESHOLDS: dict[str, dict[str, dict[str, float | int]]] = {
    "FP32": {
        "pass": {
            "max_abs_diff": 1e-5,
            "p99_abs_diff": 3e-6,
            "mean_abs_diff": 1e-6,
            "max_rel_diff": 1e-3,
            "p99_rel_diff": 1e-4,
            "cosine_sim": 0.9999,
            "close_mismatch_ratio": 0.0,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 5e-5,
            "p99_abs_diff": 2e-5,
            "mean_abs_diff": 1e-5,
            "max_rel_diff": 1e-2,
            "p99_rel_diff": 1e-3,
            "cosine_sim": 0.995,
            "close_mismatch_ratio": 0.01,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "BF16": {
        "pass": {
            "max_abs_diff": 1e-2,
            "p99_abs_diff": 8e-3,
            "mean_abs_diff": 2e-3,
            "max_rel_diff": 1e-1,
            "p99_rel_diff": 5e-2,
            "cosine_sim": 0.995,
            "close_mismatch_ratio": 0.01,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 5e-2,
            "p99_abs_diff": 5e-2,
            "mean_abs_diff": 1e-2,
            "max_rel_diff": 5e-1,
            "p99_rel_diff": 2e-1,
            "cosine_sim": 0.97,
            "close_mismatch_ratio": 0.1,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "FP8_E4M3": {
        "pass": {
            "max_abs_diff": 2e-1,
            "p99_abs_diff": 1e-1,
            "mean_abs_diff": 3e-2,
            "max_rel_diff": 5e-1,
            "p99_rel_diff": 3e-1,
            "cosine_sim": 0.98,
            "close_mismatch_ratio": 0.05,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 8e-1,
            "p99_abs_diff": 5e-1,
            "mean_abs_diff": 2e-1,
            "max_rel_diff": 2.0,
            "p99_rel_diff": 1.0,
            "cosine_sim": 0.9,
            "close_mismatch_ratio": 0.25,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "FP8_E5M2": {
        "pass": {
            "max_abs_diff": 2e-1,
            "p99_abs_diff": 1e-1,
            "mean_abs_diff": 3e-2,
            "max_rel_diff": 5e-1,
            "p99_rel_diff": 3e-1,
            "cosine_sim": 0.98,
            "close_mismatch_ratio": 0.05,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 8e-1,
            "p99_abs_diff": 5e-1,
            "mean_abs_diff": 2e-1,
            "max_rel_diff": 2.0,
            "p99_rel_diff": 1.0,
            "cosine_sim": 0.9,
            "close_mismatch_ratio": 0.25,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
}


ExecErrorLogger = ROOT_DIR / "EXEC.log"
ValidateInfoLogger = ROOT_DIR / "VALIDATE.log"
