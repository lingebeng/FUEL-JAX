"""Precision comparison utilities."""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import TOLERANCE
from .executor import ExecutionStatus, TestExecutionResult

logger = logging.getLogger(__name__)


class ComparisonStatus(Enum):
    """Status of precision comparison."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    SYNTAX_ERROR = "syntax_error"
    JAX_ERROR = "jax_error"
    TORCH_ERROR = "torch_error"
    BOTH_ERROR = "both_error"
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_MISMATCH = "type_mismatch"


@dataclass
class ComparisonResult:
    """Result of comparing JAX and PyTorch outputs."""

    status: ComparisonStatus
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    mean_abs_diff: float = 0.0
    num_mismatches: int = 0
    total_elements: int = 0
    atol_used: float = 0.0
    rtol_used: float = 0.0
    jax_nan_count: int = 0
    torch_nan_count: int = 0
    jax_inf_count: int = 0
    torch_inf_count: int = 0
    error_details: str = ""

    @property
    def mismatch_ratio(self) -> float:
        """Ratio of mismatched elements."""
        if self.total_elements == 0:
            return 0.0
        return self.num_mismatches / self.total_elements


class PrecisionChecker:
    """Checks precision consistency between JAX and PyTorch outputs."""

    def __init__(self, atol: float | None = None, rtol: float | None = None):
        """
        Initialize the precision checker.

        Args:
            atol: Absolute tolerance (overrides precision-specific default)
            rtol: Relative tolerance (overrides precision-specific default)
        """
        self.default_atol = atol
        self.default_rtol = rtol

    def check(
        self,
        execution_result: TestExecutionResult,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> ComparisonResult:
        """
        Check precision consistency between JAX and PyTorch results.

        Args:
            execution_result: Result from executing test on both frameworks
            atol: Absolute tolerance (overrides instance default)
            rtol: Relative tolerance (overrides instance default)

        Returns:
            ComparisonResult with detailed comparison information
        """
        jax_result = execution_result.jax_result
        torch_result = execution_result.torch_result
        precision = execution_result.precision

        # Determine tolerances
        atol, rtol = self._get_tolerances(precision, atol, rtol)

        # Check for execution errors
        if jax_result.status != ExecutionStatus.SUCCESS:
            if torch_result.status != ExecutionStatus.SUCCESS:
                return ComparisonResult(
                    status=ComparisonStatus.BOTH_ERROR,
                    atol_used=atol,
                    rtol_used=rtol,
                    error_details=f"JAX: {jax_result.error_message}\nPyTorch: {torch_result.error_message}",
                )
            if jax_result.status == ExecutionStatus.SYNTAX_ERROR:
                return ComparisonResult(
                    status=ComparisonStatus.SYNTAX_ERROR,
                    atol_used=atol,
                    rtol_used=rtol,
                    error_details=f"JAX syntax error: {jax_result.error_message}",
                )
            return ComparisonResult(
                status=ComparisonStatus.JAX_ERROR,
                atol_used=atol,
                rtol_used=rtol,
                error_details=f"JAX error: {jax_result.error_message}",
            )

        if torch_result.status != ExecutionStatus.SUCCESS:
            if torch_result.status == ExecutionStatus.SYNTAX_ERROR:
                return ComparisonResult(
                    status=ComparisonStatus.SYNTAX_ERROR,
                    atol_used=atol,
                    rtol_used=rtol,
                    error_details=f"PyTorch syntax error: {torch_result.error_message}",
                )
            return ComparisonResult(
                status=ComparisonStatus.TORCH_ERROR,
                atol_used=atol,
                rtol_used=rtol,
                error_details=f"PyTorch error: {torch_result.error_message}",
            )

        # Both executions succeeded, compare outputs
        jax_output = jax_result.output
        torch_output = torch_result.output

        return self._compare_arrays(jax_output, torch_output, atol, rtol)

    def _get_tolerances(
        self,
        precision: str,
        atol: float | None,
        rtol: float | None,
    ) -> tuple[float, float]:
        """Get tolerance values to use."""
        # Priority: explicit args > instance defaults > precision-specific defaults
        default_tol = TOLERANCE.get(precision, TOLERANCE["FP32"])

        if atol is None:
            atol = (
                self.default_atol
                if self.default_atol is not None
                else default_tol["atol"]
            )
        if rtol is None:
            rtol = (
                self.default_rtol
                if self.default_rtol is not None
                else default_tol["rtol"]
            )

        return atol, rtol

    def _compare_arrays(
        self,
        jax_output: np.ndarray | None,
        torch_output: np.ndarray | None,
        atol: float,
        rtol: float,
    ) -> ComparisonResult:
        """Compare two numpy arrays with detailed metrics."""
        if jax_output is None or torch_output is None:
            return ComparisonResult(
                status=ComparisonStatus.BOTH_ERROR,
                atol_used=atol,
                rtol_used=rtol,
                error_details="One or both outputs are None",
            )

        # Check shape compatibility
        if jax_output.shape != torch_output.shape:
            return ComparisonResult(
                status=ComparisonStatus.SHAPE_MISMATCH,
                atol_used=atol,
                rtol_used=rtol,
                error_details=f"Shape mismatch: JAX {jax_output.shape} vs PyTorch {torch_output.shape}",
            )

        # Convert to float64 for comparison (handle bfloat16, etc.)
        try:
            jax_float = jax_output.astype(np.float64)
            torch_float = torch_output.astype(np.float64)
        except (TypeError, ValueError):
            # Handle non-numeric types (e.g., boolean)
            try:
                jax_float = jax_output.astype(np.float64)
                torch_float = torch_output.astype(np.float64)
            except Exception:
                # For boolean or other types, do exact comparison
                matches = np.array_equal(jax_output, torch_output)
                return ComparisonResult(
                    status=ComparisonStatus.CONSISTENT
                    if matches
                    else ComparisonStatus.INCONSISTENT,
                    atol_used=atol,
                    rtol_used=rtol,
                    total_elements=jax_output.size,
                    num_mismatches=0 if matches else jax_output.size,
                )

        # Count special values
        jax_nan_mask = np.isnan(jax_float)
        torch_nan_mask = np.isnan(torch_float)
        jax_inf_mask = np.isinf(jax_float)
        torch_inf_mask = np.isinf(torch_float)

        jax_nan_count = int(np.sum(jax_nan_mask))
        torch_nan_count = int(np.sum(torch_nan_mask))
        jax_inf_count = int(np.sum(jax_inf_mask))
        torch_inf_count = int(np.sum(torch_inf_mask))

        # For comparison, treat NaN == NaN and check inf signs match
        # Create mask for valid (non-NaN, non-Inf) values
        # both_nan = jax_nan_mask & torch_nan_mask
        both_inf = jax_inf_mask & torch_inf_mask

        # Check if inf signs match where both are inf
        inf_sign_match = np.ones_like(jax_float, dtype=bool)
        if np.any(both_inf):
            inf_sign_match[both_inf] = np.sign(jax_float[both_inf]) == np.sign(
                torch_float[both_inf]
            )

        # Compute differences for valid values
        valid_mask = ~(jax_nan_mask | torch_nan_mask | jax_inf_mask | torch_inf_mask)

        if np.any(valid_mask):
            abs_diff = np.abs(jax_float - torch_float)
            abs_diff[~valid_mask] = 0  # Don't count special values in diff

            max_abs_diff = float(np.max(abs_diff[valid_mask]))
            mean_abs_diff = float(np.mean(abs_diff[valid_mask]))

            # Relative difference (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = np.where(
                    torch_float != 0,
                    abs_diff / np.abs(torch_float),
                    np.where(jax_float != 0, np.inf, 0.0),
                )
            rel_diff[~valid_mask] = 0
            max_rel_diff = float(np.max(rel_diff[valid_mask]))
        else:
            max_abs_diff = 0.0
            mean_abs_diff = 0.0
            max_rel_diff = 0.0

        # Check consistency using numpy's allclose logic
        # |a - b| <= atol + rtol * |b|
        close_mask = np.isclose(
            jax_float, torch_float, atol=atol, rtol=rtol, equal_nan=True
        )

        # Also consider inf with matching signs as close
        close_mask = close_mask | (both_inf & inf_sign_match)

        # NaN in only one array is not close
        nan_mismatch = jax_nan_mask ^ torch_nan_mask
        close_mask = close_mask & ~nan_mismatch

        num_mismatches = int(np.sum(~close_mask))
        total_elements = int(jax_output.size)

        is_consistent = num_mismatches == 0

        return ComparisonResult(
            status=ComparisonStatus.CONSISTENT
            if is_consistent
            else ComparisonStatus.INCONSISTENT,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            mean_abs_diff=mean_abs_diff,
            num_mismatches=num_mismatches,
            total_elements=total_elements,
            atol_used=atol,
            rtol_used=rtol,
            jax_nan_count=jax_nan_count,
            torch_nan_count=torch_nan_count,
            jax_inf_count=jax_inf_count,
            torch_inf_count=torch_inf_count,
        )


def check_precision(
    jax_output: np.ndarray,
    torch_output: np.ndarray,
    precision: str = "FP32",
    atol: float | None = None,
    rtol: float | None = None,
) -> ComparisonResult:
    """
    Convenience function to check precision between two arrays.

    Args:
        jax_output: JAX output array
        torch_output: PyTorch output array
        precision: Precision level for default tolerances
        atol: Absolute tolerance (optional)
        rtol: Relative tolerance (optional)

    Returns:
        ComparisonResult with detailed comparison information
    """
    checker = PrecisionChecker(atol=atol, rtol=rtol)

    # Get tolerances
    atol_val, rtol_val = checker._get_tolerances(precision, atol, rtol)

    return checker._compare_arrays(jax_output, torch_output, atol_val, rtol_val)
