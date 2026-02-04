# Feedback Report

## Operator: {operator_name}
## PyTorch Equivalent: {torch_operator}
## Type: {op_type}
## Date: {timestamp}

---

## Test Summary

| Metric | Count |
|--------|-------|
| Total Tests | {total} |
| Consistent | {consistent_count} |
| Inconsistent | {inconsistent_count} |
| Errors | {error_count} |

---

## Status Legend

- **consistent**: JAX and PyTorch outputs match within specified tolerance
- **inconsistent**: Outputs differ beyond tolerance threshold
- **syntax_error**: Invalid operator call syntax
- **jax_error**: JAX execution failed
- **torch_error**: PyTorch execution failed
- **both_error**: Both frameworks failed

---

## Detailed Results

{detailed_results}

---

## Notes

- Tolerance values are precision-dependent (FP32: atol=1e-6, rtol=1e-5; BF16: atol=1e-2, rtol=1e-2; FP8: atol=1e-1, rtol=1e-1)
- NaN values are treated as equal when comparing outputs
- Infinity values are compared with sign matching
- Known semantic differences (e.g., jax.lax.rem vs torch.remainder for negative numbers) may show as inconsistent
