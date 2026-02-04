"""Configuration management for JAX-PyTorch precision fuzzing framework."""

from dataclasses import dataclass, field
from enum import Enum


class Precision(Enum):
    """Supported precision types for testing."""

    FP32 = "FP32"
    BF16 = "BF16"
    FP8_E4M3 = "FP8_E4M3"
    FP8_E5M2 = "FP8_E5M2"


# Data type mappings for JAX and PyTorch
PRECISION_MAP: dict[str, dict[str, str]] = {
    "FP32": {"jax": "jnp.float32", "torch": "torch.float32"},
    "BF16": {"jax": "jnp.bfloat16", "torch": "torch.bfloat16"},
    "FP8_E4M3": {"jax": "jnp.float8_e4m3fn", "torch": "torch.float8_e4m3fn"},
    "FP8_E5M2": {"jax": "jnp.float8_e5m2", "torch": "torch.float8_e5m2"},
}

# Tolerance configurations per precision
TOLERANCE: dict[str, dict[str, float]] = {
    "FP32": {"atol": 1e-6, "rtol": 1e-5},
    "BF16": {"atol": 1e-2, "rtol": 1e-2},
    "FP8_E4M3": {"atol": 1e-1, "rtol": 1e-1},
    "FP8_E5M2": {"atol": 1e-1, "rtol": 1e-1},
}

# Special values for fuzzing
SPECIAL_VALUES: dict[str, float] = {
    "zeros": 0.0,
    "ones": 1.0,
    "neg_ones": -1.0,
    "inf": float("inf"),
    "neg_inf": float("-inf"),
    "nan": float("nan"),
    "epsilon": 1e-7,
    "large": 1e38,
    "small": 1e-38,
    "denormal": 1e-45,  # FP32 denormal
}

# Test shapes for fuzzing
SHAPES: list[tuple[int, ...]] = [
    (1,),  # scalar-like
    (1024,),  # 1D
    (32, 32),  # 2D square
    (1, 1024),  # 2D thin
    (16, 32, 64),  # 3D
    (2, 3, 4, 5),  # 4D
    (1, 1, 1, 1),  # all ones
    (0,),  # empty (edge case)
    (1 << 20,),  # large 1D (1M elements)
]


@dataclass
class TestConfig:
    """Configuration for a test run."""

    precisions: list[str] = field(default_factory=lambda: ["FP32"])
    atol: float | None = None  # If None, use precision-specific default
    rtol: float | None = None  # If None, use precision-specific default
    start_idx: int = 0
    operators: list[str] | None = None  # If None, test all operators
    output_dir: str = "output"
    csv_path: str = "dataset/jax2torch_lax.csv"
    num_test_cases: int = 5  # Number of test cases to generate per operator

    def get_tolerance(self, precision: str) -> tuple[float, float]:
        """Get atol and rtol for a given precision."""
        if self.atol is not None and self.rtol is not None:
            return self.atol, self.rtol
        default_tol = TOLERANCE.get(precision, TOLERANCE["FP32"])
        atol = self.atol if self.atol is not None else default_tol["atol"]
        rtol = self.rtol if self.rtol is not None else default_tol["rtol"]
        return atol, rtol


@dataclass
class LLMConfig:
    """Configuration for LLM API."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.7


def load_llm_config_from_env() -> LLMConfig:
    """Load LLM configuration from environment variables."""
    import os

    return LLMConfig(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
