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
