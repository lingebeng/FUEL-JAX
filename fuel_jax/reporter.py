"""Report generation for precision testing results."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .csv_reader import OperatorPair
from .executor import ExecutionStatus
from .precision_checker import ComparisonResult, ComparisonStatus
from .test_generator import TestCase

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Complete result for a single test case."""

    test_case: TestCase
    precision: str
    comparison: ComparisonResult
    jax_status: ExecutionStatus
    torch_status: ExecutionStatus
    jax_error: str = ""
    torch_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_case.test_id,
            "description": self.test_case.description,
            "precision": self.precision,
            "status": self.comparison.status.value,
            "max_abs_diff": self.comparison.max_abs_diff,
            "max_rel_diff": self.comparison.max_rel_diff,
            "mean_abs_diff": self.comparison.mean_abs_diff,
            "num_mismatches": self.comparison.num_mismatches,
            "total_elements": self.comparison.total_elements,
            "mismatch_ratio": self.comparison.mismatch_ratio,
            "atol_used": self.comparison.atol_used,
            "rtol_used": self.comparison.rtol_used,
            "jax_status": self.jax_status.value,
            "torch_status": self.torch_status.value,
            "jax_error": self.jax_error,
            "torch_error": self.torch_error,
            "jax_nan_count": self.comparison.jax_nan_count,
            "torch_nan_count": self.comparison.torch_nan_count,
            "jax_inf_count": self.comparison.jax_inf_count,
            "torch_inf_count": self.comparison.torch_inf_count,
            "jax_code": self.test_case.jax_code,
            "torch_code": self.test_case.torch_code,
        }


@dataclass
class OperatorReport:
    """Report for a single operator."""

    operator: OperatorPair
    results: list[TestResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def consistent_count(self) -> int:
        return sum(
            1
            for r in self.results
            if r.comparison.status == ComparisonStatus.CONSISTENT
        )

    @property
    def inconsistent_count(self) -> int:
        return sum(
            1
            for r in self.results
            if r.comparison.status == ComparisonStatus.INCONSISTENT
        )

    @property
    def error_count(self) -> int:
        return sum(
            1
            for r in self.results
            if r.comparison.status
            in (
                ComparisonStatus.SYNTAX_ERROR,
                ComparisonStatus.JAX_ERROR,
                ComparisonStatus.TORCH_ERROR,
                ComparisonStatus.BOTH_ERROR,
            )
        )

    def add_result(self, result: TestResult):
        """Add a test result to the report."""
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operator": {
                "jax_op": self.operator.jax_op,
                "torch_op": self.operator.torch_op,
                "op_type": self.operator.op_type,
            },
            "timestamp": self.timestamp,
            "summary": {
                "total_tests": self.total_tests,
                "consistent": self.consistent_count,
                "inconsistent": self.inconsistent_count,
                "errors": self.error_count,
            },
            "results": [r.to_dict() for r in self.results],
        }


class Reporter:
    """Generates reports for precision testing results."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the reporter.

        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)

    def _get_operator_dir(self, operator: OperatorPair) -> Path:
        """Get the output directory for an operator."""
        # Sanitize operator name for directory name
        dir_name = operator.jax_op.replace(".", "_")
        return self.output_dir / dir_name

    def _ensure_dir(self, path: Path):
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: OperatorReport, iteration: int = 0):
        """
        Save operator report to files.

        Args:
            report: The operator report to save
            iteration: Iteration number for naming
        """
        operator_dir = self._get_operator_dir(report.operator)
        self._ensure_dir(operator_dir)

        # Save markdown report
        md_path = operator_dir / f"feedback_{iteration:02d}.md"
        md_content = self._generate_markdown(report)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Saved markdown report: {md_path}")

        # Save raw JSON results
        json_path = operator_dir / "raw_results.json"

        # Load existing results if any
        existing_results = []
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_results = existing_data
            except json.JSONDecodeError:
                pass

        # Append new results
        existing_results.append(report.to_dict())

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2)
        logger.info(f"Saved JSON results: {json_path}")

    def _generate_markdown(self, report: OperatorReport) -> str:
        """Generate markdown report content."""
        # Load template
        template = self._load_template()

        # Generate detailed results section
        detailed_results = self._generate_detailed_results(report)

        # Format the report
        content = template.format(
            operator_name=report.operator.jax_op,
            torch_operator=report.operator.torch_op,
            op_type=report.operator.op_type,
            timestamp=report.timestamp,
            total=report.total_tests,
            consistent_count=report.consistent_count,
            inconsistent_count=report.inconsistent_count,
            error_count=report.error_count,
            detailed_results=detailed_results,
        )

        return content

    def _load_template(self) -> str:
        """Load the feedback template."""
        template_path = (
            Path(__file__).parent.parent / "prompts" / "als" / "als_feedback.md"
        )
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        # Fallback template if file doesn't exist
        return """# Feedback Report

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

## Detailed Results

{detailed_results}
"""

    def _generate_detailed_results(self, report: OperatorReport) -> str:
        """Generate detailed results section."""
        lines = []

        for i, result in enumerate(report.results, 1):
            lines.append(f"### Test Case {i}: {result.test_case.description}")
            lines.append("")
            lines.append(f"- **Test ID**: {result.test_case.test_id}")
            lines.append(f"- **Precision**: {result.precision}")
            lines.append(f"- **Status**: {result.comparison.status.value}")
            lines.append("")
            lines.append(f"**JAX Code**: `{result.test_case.jax_code}`")
            lines.append(f"**PyTorch Code**: `{result.test_case.torch_code}`")
            lines.append("")

            if result.comparison.status == ComparisonStatus.CONSISTENT:
                lines.append("**Result**: Outputs match within tolerance")
                lines.append(
                    f"- Max Absolute Diff: {result.comparison.max_abs_diff:.2e}"
                )
                lines.append(
                    f"- Tolerance Used: atol={result.comparison.atol_used:.2e}, rtol={result.comparison.rtol_used:.2e}"
                )

            elif result.comparison.status == ComparisonStatus.INCONSISTENT:
                lines.append("**Result**: Outputs DIFFER beyond tolerance")
                lines.append(
                    f"- Max Absolute Diff: {result.comparison.max_abs_diff:.2e}"
                )
                lines.append(
                    f"- Max Relative Diff: {result.comparison.max_rel_diff:.2e}"
                )
                lines.append(
                    f"- Mean Absolute Diff: {result.comparison.mean_abs_diff:.2e}"
                )
                lines.append(
                    f"- Mismatches: {result.comparison.num_mismatches}/{result.comparison.total_elements} ({result.comparison.mismatch_ratio:.2%})"
                )
                lines.append(
                    f"- Tolerance Used: atol={result.comparison.atol_used:.2e}, rtol={result.comparison.rtol_used:.2e}"
                )

            else:
                # Error case
                lines.append(f"**Error Type**: {result.comparison.status.value}")
                if result.jax_error:
                    lines.append(f"- JAX Error: {result.jax_error}")
                if result.torch_error:
                    lines.append(f"- PyTorch Error: {result.torch_error}")
                if result.comparison.error_details:
                    lines.append(f"- Details: {result.comparison.error_details}")

            # Special value counts
            if (
                result.comparison.jax_nan_count > 0
                or result.comparison.torch_nan_count > 0
            ):
                lines.append("")
                lines.append("**Special Values**:")
                lines.append(
                    f"- JAX NaN count: {result.comparison.jax_nan_count}, PyTorch NaN count: {result.comparison.torch_nan_count}"
                )
                lines.append(
                    f"- JAX Inf count: {result.comparison.jax_inf_count}, PyTorch Inf count: {result.comparison.torch_inf_count}"
                )

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def generate_summary_report(self, all_reports: list[OperatorReport]) -> str:
        """
        Generate a summary report for all tested operators.

        Args:
            all_reports: List of all operator reports

        Returns:
            Markdown content for summary report
        """
        lines = [
            "# JAX-PyTorch Precision Testing Summary",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "---",
            "",
            "## Overall Statistics",
            "",
        ]

        total_tests = sum(r.total_tests for r in all_reports)
        total_consistent = sum(r.consistent_count for r in all_reports)
        total_inconsistent = sum(r.inconsistent_count for r in all_reports)
        total_errors = sum(r.error_count for r in all_reports)

        lines.extend(
            [
                f"- **Total Operators Tested**: {len(all_reports)}",
                f"- **Total Test Cases**: {total_tests}",
                f"- **Consistent**: {total_consistent} ({total_consistent / total_tests * 100:.1f}%)"
                if total_tests > 0
                else "- **Consistent**: 0",
                f"- **Inconsistent**: {total_inconsistent} ({total_inconsistent / total_tests * 100:.1f}%)"
                if total_tests > 0
                else "- **Inconsistent**: 0",
                f"- **Errors**: {total_errors} ({total_errors / total_tests * 100:.1f}%)"
                if total_tests > 0
                else "- **Errors**: 0",
                "",
                "---",
                "",
                "## Operator Summary",
                "",
                "| JAX Operator | PyTorch Operator | Type | Total | Consistent | Inconsistent | Errors |",
                "|--------------|------------------|------|-------|------------|--------------|--------|",
            ]
        )

        for report in all_reports:
            lines.append(
                f"| {report.operator.jax_op} | {report.operator.torch_op} | "
                f"{report.operator.op_type} | {report.total_tests} | "
                f"{report.consistent_count} | {report.inconsistent_count} | "
                f"{report.error_count} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## Operators with Inconsistencies",
                "",
            ]
        )

        inconsistent_ops = [r for r in all_reports if r.inconsistent_count > 0]
        if inconsistent_ops:
            for report in inconsistent_ops:
                lines.append(f"### {report.operator.jax_op}")
                lines.append(f"- PyTorch: `{report.operator.torch_op}`")
                lines.append(
                    f"- Inconsistent tests: {report.inconsistent_count}/{report.total_tests}"
                )
                lines.append("")
        else:
            lines.append("No inconsistencies found!")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Operators with Errors",
                "",
            ]
        )

        error_ops = [r for r in all_reports if r.error_count > 0]
        if error_ops:
            for report in error_ops:
                lines.append(f"### {report.operator.jax_op}")
                lines.append(f"- PyTorch: `{report.operator.torch_op}`")
                lines.append(f"- Errors: {report.error_count}/{report.total_tests}")
                lines.append("")
        else:
            lines.append("No errors encountered!")

        return "\n".join(lines)

    def save_summary(self, all_reports: list[OperatorReport]):
        """Save summary report to output directory."""
        self._ensure_dir(self.output_dir)
        summary_content = self.generate_summary_report(all_reports)
        summary_path = self.output_dir / "summary.md"
        summary_path.write_text(summary_content, encoding="utf-8")
        logger.info(f"Saved summary report: {summary_path}")
