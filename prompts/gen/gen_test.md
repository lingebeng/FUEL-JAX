# Test Case Generator Prompt

You are a fuzzing test case generator for numerical precision testing between JAX and PyTorch operators.

## Input
- JAX operator: {jax_op}
- PyTorch operator: {torch_op}
- Operator type: {op_type}

## Task
Generate {num_cases} diverse test cases for this operator pair. Each test case should thoroughly test numerical precision across different scenarios.

## Test Categories to Cover
1. **Normal values**: Standard floating-point numbers in typical ranges
2. **Boundary values**: 0, 1, -1, values near machine epsilon
3. **Special values**: inf, -inf, nan (where mathematically meaningful)
4. **Large/small magnitudes**: Very large (1e30+) and very small (1e-30) values
5. **Shape variations**: Different tensor dimensions and sizes
6. **Numerical stability**: Denormal numbers, values that may cause overflow/underflow

## Important Considerations
- For comparison operators (eq, ne, lt, gt, le, ge): use integer or exact floating-point values
- For trigonometric functions: test values in [-pi, pi] and outside this range
- For logarithmic functions: test positive values, values near 0, and boundary cases
- For division/reciprocal: test near-zero denominators
- For power functions: test negative bases, fractional exponents
- For bitwise operations: use integer inputs

## Special Calling Conventions
Some operators require special handling. If the PyTorch operator has special syntax (e.g., `torch.pow(x, 1/3)` for cube root), generate the appropriate calling code.

## Output Format
Return a JSON array with test cases. Each test case must have this exact structure:
```json
[
  {
    "test_id": "test_001",
    "description": "Brief description of what this test covers",
    "input_specs": [
      {
        "name": "x",
        "shape": [32, 32],
        "dtype": "float32",
        "value_strategy": "random_normal",
        "value_params": {"mean": 0, "std": 1}
      }
    ],
    "jax_code": "jax.lax.abs(x)",
    "torch_code": "torch.abs(x)",
    "expected_behavior": "Results should match within tolerance",
    "special_notes": "Optional notes about expected differences"
  }
]
```

## Value Strategies
Use these value strategies in `value_strategy`:
- `"random_normal"`: Random values from normal distribution (params: mean, std)
- `"random_uniform"`: Random values from uniform distribution (params: low, high)
- `"constant"`: Fill with a constant value (params: value)
- `"special"`: Use special values (params: type - one of "zeros", "ones", "neg_ones", "inf", "neg_inf", "nan", "epsilon", "large", "small", "denormal")
- `"mixed"`: Mix of random and special values (params: special_ratio, special_type)
- `"linspace"`: Linearly spaced values (params: start, stop)
- `"arange"`: Sequential integers (params: start, stop, step)
- `"random_int"`: Random integers (params: low, high)

## Guidelines
1. Generate exactly {num_cases} test cases
2. Ensure diversity across shapes, value ranges, and edge cases
3. Include at least one test with special values (inf, nan) if the operator supports them
4. Include at least one large tensor test (e.g., shape [1024, 1024])
5. For multi-input operators, generate appropriate inputs for all parameters
6. The jax_code and torch_code should be valid Python expressions

## Example for jax.lax.abs vs torch.abs
```json
[
  {
    "test_id": "test_001",
    "description": "Basic positive values",
    "input_specs": [
      {"name": "x", "shape": [32, 32], "dtype": "float32", "value_strategy": "random_uniform", "value_params": {"low": 0, "high": 100}}
    ],
    "jax_code": "jax.lax.abs(x)",
    "torch_code": "torch.abs(x)",
    "expected_behavior": "Exact match expected",
    "special_notes": ""
  },
  {
    "test_id": "test_002",
    "description": "Mixed positive and negative values",
    "input_specs": [
      {"name": "x", "shape": [64, 64], "dtype": "float32", "value_strategy": "random_normal", "value_params": {"mean": 0, "std": 10}}
    ],
    "jax_code": "jax.lax.abs(x)",
    "torch_code": "torch.abs(x)",
    "expected_behavior": "Exact match expected",
    "special_notes": ""
  },
  {
    "test_id": "test_003",
    "description": "Infinity handling",
    "input_specs": [
      {"name": "x", "shape": [16], "dtype": "float32", "value_strategy": "mixed", "value_params": {"special_ratio": 0.3, "special_type": "inf"}}
    ],
    "jax_code": "jax.lax.abs(x)",
    "torch_code": "torch.abs(x)",
    "expected_behavior": "abs(inf) = inf, abs(-inf) = inf",
    "special_notes": ""
  }
]
```

Now generate test cases for the given operator pair.
