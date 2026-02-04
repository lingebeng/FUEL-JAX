"""Fuzzing input generation strategies."""

import numpy as np
from typing import Any


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

# Standard test shapes
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


def generate_input(
    shape: tuple[int, ...] | list[int],
    dtype: str = "float32",
    value_strategy: str = "random_normal",
    value_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate input tensor based on specified strategy.

    Args:
        shape: Shape of the tensor to generate
        dtype: NumPy dtype string
        value_strategy: Strategy for generating values
        value_params: Parameters for the value strategy
        seed: Random seed for reproducibility

    Returns:
        NumPy array with the generated values
    """
    if seed is not None:
        np.random.seed(seed)

    shape = tuple(shape)
    value_params = value_params or {}
    np_dtype = np.dtype(dtype)

    # Handle empty shapes
    if 0 in shape:
        return np.array([], dtype=np_dtype).reshape(shape)

    if value_strategy == "random_normal":
        mean = value_params.get("mean", 0.0)
        std = value_params.get("std", 1.0)
        arr = np.random.normal(mean, std, shape).astype(np_dtype)

    elif value_strategy == "random_uniform":
        low = value_params.get("low", 0.0)
        high = value_params.get("high", 1.0)
        arr = np.random.uniform(low, high, shape).astype(np_dtype)

    elif value_strategy == "constant":
        value = value_params.get("value", 0.0)
        arr = np.full(shape, value, dtype=np_dtype)

    elif value_strategy == "special":
        special_type = value_params.get("type", "zeros")
        value = SPECIAL_VALUES.get(special_type, 0.0)
        arr = np.full(shape, value, dtype=np_dtype)

    elif value_strategy == "mixed":
        # Mix of random values and special values
        special_ratio = value_params.get("special_ratio", 0.3)
        special_type = value_params.get("special_type", "inf")
        special_value = SPECIAL_VALUES.get(special_type, float("inf"))

        # Start with random normal values
        arr = np.random.normal(0, 1, shape).astype(np_dtype)

        # Replace some values with special values
        mask = np.random.random(shape) < special_ratio
        arr[mask] = special_value

        # Also add negative special values for inf
        if special_type == "inf":
            neg_mask = np.random.random(shape) < (special_ratio / 2)
            arr[neg_mask] = -special_value

    elif value_strategy == "linspace":
        start = value_params.get("start", 0.0)
        stop = value_params.get("stop", 1.0)
        total_elements = int(np.prod(shape))
        arr = np.linspace(start, stop, total_elements, dtype=np_dtype).reshape(shape)

    elif value_strategy == "arange":
        start = value_params.get("start", 0)
        step = value_params.get("step", 1)
        total_elements = int(np.prod(shape))
        stop = start + step * total_elements
        arr = np.arange(start, stop, step, dtype=np_dtype)[:total_elements].reshape(
            shape
        )

    elif value_strategy == "random_int":
        low = value_params.get("low", 0)
        high = value_params.get("high", 100)
        arr = np.random.randint(low, high, shape).astype(np_dtype)

    elif value_strategy == "nextafter":
        # Values adjacent to a base value (for boundary testing)
        base = value_params.get("base", 1.0)
        arr = np.full(shape, base, dtype=np_dtype)
        # Alternate between nextafter(base, inf) and nextafter(base, -inf)
        mask = np.random.random(shape) < 0.5
        arr[mask] = np.nextafter(arr[mask], np.inf)
        arr[~mask] = np.nextafter(arr[~mask], -np.inf)

    else:
        raise ValueError(f"Unknown value strategy: {value_strategy}")

    return arr


def generate_default_test_inputs(op_type: str, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate a default set of test inputs based on operator type.

    Args:
        op_type: Type of operator (elementwise, reduction, linalg, etc.)
        seed: Random seed

    Returns:
        List of input specifications
    """
    np.random.seed(seed)

    default_inputs = [
        # Normal values
        {
            "description": "Normal random values (small tensor)",
            "shape": (32, 32),
            "dtype": "float32",
            "value_strategy": "random_normal",
            "value_params": {"mean": 0, "std": 1},
        },
        # Large tensor
        {
            "description": "Normal random values (large tensor)",
            "shape": (512, 512),
            "dtype": "float32",
            "value_strategy": "random_normal",
            "value_params": {"mean": 0, "std": 10},
        },
        # Boundary values
        {
            "description": "Values near zero",
            "shape": (64, 64),
            "dtype": "float32",
            "value_strategy": "random_uniform",
            "value_params": {"low": -1e-6, "high": 1e-6},
        },
        # Special values - zeros
        {
            "description": "All zeros",
            "shape": (16, 16),
            "dtype": "float32",
            "value_strategy": "constant",
            "value_params": {"value": 0.0},
        },
        # Special values - ones
        {
            "description": "All ones",
            "shape": (16, 16),
            "dtype": "float32",
            "value_strategy": "constant",
            "value_params": {"value": 1.0},
        },
    ]

    # Add operator-type specific inputs
    if op_type == "elementwise":
        default_inputs.extend(
            [
                # Mixed with inf
                {
                    "description": "Mixed with infinity values",
                    "shape": (32, 32),
                    "dtype": "float32",
                    "value_strategy": "mixed",
                    "value_params": {"special_ratio": 0.1, "special_type": "inf"},
                },
                # Large magnitude values
                {
                    "description": "Large magnitude values",
                    "shape": (32, 32),
                    "dtype": "float32",
                    "value_strategy": "random_uniform",
                    "value_params": {"low": -1e30, "high": 1e30},
                },
                # Small magnitude values
                {
                    "description": "Small magnitude values",
                    "shape": (32, 32),
                    "dtype": "float32",
                    "value_strategy": "random_uniform",
                    "value_params": {"low": -1e-30, "high": 1e-30},
                },
            ]
        )

    elif op_type == "reduction":
        default_inputs.extend(
            [
                # 1D tensor
                {
                    "description": "1D tensor for reduction",
                    "shape": (1024,),
                    "dtype": "float32",
                    "value_strategy": "random_normal",
                    "value_params": {"mean": 0, "std": 1},
                },
                # 3D tensor
                {
                    "description": "3D tensor for reduction",
                    "shape": (16, 32, 64),
                    "dtype": "float32",
                    "value_strategy": "random_normal",
                    "value_params": {"mean": 0, "std": 1},
                },
            ]
        )

    elif op_type == "linalg":
        default_inputs.extend(
            [
                # Square matrix
                {
                    "description": "Square matrix for linalg",
                    "shape": (64, 64),
                    "dtype": "float32",
                    "value_strategy": "random_normal",
                    "value_params": {"mean": 0, "std": 1},
                },
                # Rectangular matrix
                {
                    "description": "Rectangular matrix",
                    "shape": (32, 64),
                    "dtype": "float32",
                    "value_strategy": "random_normal",
                    "value_params": {"mean": 0, "std": 1},
                },
            ]
        )

    return default_inputs


def generate_edge_case_inputs(dtype: str = "float32") -> list[dict[str, Any]]:
    """
    Generate edge case inputs for thorough testing.

    Args:
        dtype: Data type for the inputs

    Returns:
        List of edge case input specifications
    """
    edge_cases = [
        # Denormal numbers
        {
            "description": "Denormal numbers",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "special",
            "value_params": {"type": "denormal"},
        },
        # Machine epsilon
        {
            "description": "Machine epsilon values",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "special",
            "value_params": {"type": "epsilon"},
        },
        # Nextafter values (boundary adjacent)
        {
            "description": "Values adjacent to 1.0",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "nextafter",
            "value_params": {"base": 1.0},
        },
        # Nextafter values near zero
        {
            "description": "Values adjacent to 0.0",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "nextafter",
            "value_params": {"base": 0.0},
        },
        # Very large values
        {
            "description": "Very large values",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "special",
            "value_params": {"type": "large"},
        },
        # Very small values
        {
            "description": "Very small values",
            "shape": (32,),
            "dtype": dtype,
            "value_strategy": "special",
            "value_params": {"type": "small"},
        },
    ]

    return edge_cases
