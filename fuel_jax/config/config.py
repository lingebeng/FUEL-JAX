from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


PRECISION_MAP: dict[str, dict[str, str]] = {
    "FP32": {"jax": "jnp.float32", "torch": "torch.float32"},
    "BF16": {"jax": "jnp.bfloat16", "torch": "torch.bfloat16"},
    "FP8_E4M3": {"jax": "jnp.float8_e4m3fn", "torch": "torch.float8_e4m3fn"},
    "FP8_E5M2": {"jax": "jnp.float8_e5m2", "torch": "torch.float8_e5m2"},
}


TOLERANCE: dict[str, dict[str, float]] = {
    "FP32": {"atol": 1e-6, "rtol": 1e-5},
    "BF16": {"atol": 1e-2, "rtol": 1e-2},
    "FP8_E4M3": {"atol": 1e-1, "rtol": 1e-1},
    "FP8_E5M2": {"atol": 1e-1, "rtol": 1e-1},
}

# Multi-metric oracle thresholds.
# PASS: no obvious issue.
# WARN: potential issue, worth manual inspection.
# FAIL: likely wrong output.
DIFF_ORACLE_THRESHOLDS: dict[str, dict[str, dict[str, float | int]]] = {
    "FP32": {
        "pass": {
            "max_abs_diff": 1e-5,
            "p99_abs_diff": 3e-6,
            "max_ulp_diff": 64.0,
            "p99_ulp_diff": 8.0,
            "cosine_distance": 1e-5,
            "close_mismatch_count": 0,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 5e-5,
            "p99_abs_diff": 2e-5,
            "max_ulp_diff": 4096.0,
            "p99_ulp_diff": 256.0,
            "cosine_distance": 1e-3,
            "close_mismatch_count": 10,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "BF16": {
        "pass": {
            "max_abs_diff": 2e-2,
            "p99_abs_diff": 1e-2,
            "max_ulp_diff": 8192.0,
            "p99_ulp_diff": 1024.0,
            "cosine_distance": 5e-3,
            "close_mismatch_count": 10,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 8e-2,
            "p99_abs_diff": 5e-2,
            "max_ulp_diff": 65536.0,
            "p99_ulp_diff": 8192.0,
            "cosine_distance": 5e-2,
            "close_mismatch_count": 100,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "FP8_E4M3": {
        "pass": {
            "max_abs_diff": 2e-1,
            "p99_abs_diff": 1e-1,
            "max_ulp_diff": 262144.0,
            "p99_ulp_diff": 65536.0,
            "cosine_distance": 2e-2,
            "close_mismatch_count": 50,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 8e-1,
            "p99_abs_diff": 5e-1,
            "max_ulp_diff": 2097152.0,
            "p99_ulp_diff": 524288.0,
            "cosine_distance": 2e-1,
            "close_mismatch_count": 250,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
    "FP8_E5M2": {
        "pass": {
            "max_abs_diff": 2e-1,
            "p99_abs_diff": 1e-1,
            "max_ulp_diff": 262144.0,
            "p99_ulp_diff": 65536.0,
            "cosine_distance": 2e-2,
            "close_mismatch_count": 50,
            "nonfinite_mismatch_ratio": 0.0,
        },
        "warn": {
            "max_abs_diff": 8e-1,
            "p99_abs_diff": 5e-1,
            "max_ulp_diff": 2097152.0,
            "p99_ulp_diff": 524288.0,
            "cosine_distance": 2e-1,
            "close_mismatch_count": 250,
            "nonfinite_mismatch_ratio": 0.0,
        },
    },
}


ExecErrorLogger = ROOT_DIR / "EXEC.log"
ValidateInfoLogger = ROOT_DIR / "VALIDATE.log"
