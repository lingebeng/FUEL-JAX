"""Main orchestrator for JAX-PyTorch precision fuzzing framework."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..config import TestConfig, load_llm_config_from_env
from ..execution.executor import OperatorExecutor
from ..execution.precision_checker import ComparisonStatus, PrecisionChecker
from ..generation.test_generator import TestGenerator
from ..io.csv_reader import OperatorPair, read_operator_pairs
from ..io.reporter import OperatorReport, Reporter, TestResult
from ..utils.utils import ensure_dir, read_json, write_json


class FuzzingOrchestrator:
    """Main orchestrator for the precision fuzzing framework."""

    def __init__(self, config: TestConfig):
        """
        Initialize the orchestrator.

        Args:
            config: Test configuration
        """
        self.config = config
        self.executor = OperatorExecutor(device="cpu")
        self.checker = PrecisionChecker(atol=config.atol, rtol=config.rtol)
        self.reporter = Reporter(output_dir=config.output_dir)
        self.test_generator: TestGenerator | None = None
        self.all_reports: list[OperatorReport] = []

        # Progress tracking
        self.progress_file = Path(config.output_dir) / ".progress.json"

    def _init_test_generator(self, use_llm: bool = True):
        """Initialize test generator."""
        if self.test_generator is None:
            try:
                llm_config = load_llm_config_from_env() if use_llm else None
                self.test_generator = TestGenerator(
                    llm_config=llm_config, use_llm=use_llm
                )
            except ValueError as e:
                logger.warning(f"LLM not available: {e}. Using fallback generation.")
                self.test_generator = TestGenerator(use_llm=False)

    def run(self, use_llm: bool = True):
        """
        Run the fuzzing framework.

        Args:
            use_llm: Whether to use LLM for test generation
        """
        self._init_test_generator(use_llm)

        # Read operator pairs
        operators = read_operator_pairs(
            csv_path=self.config.csv_path,
            start_idx=self.config.start_idx,
            operators=self.config.operators,
            skip_none_torch=True,
        )

        logger.info(f"Found {len(operators)} operators to test")

        # Process each operator
        for i, operator in enumerate(operators):
            logger.info(
                f"[{i + 1}/{len(operators)}] Testing {operator.jax_op} <-> {operator.torch_op}"
            )

            try:
                report = self._test_operator(operator)
                self.all_reports.append(report)

                # Save report
                self.reporter.save_report(report, iteration=0)

                # Save progress
                self._save_progress(operator.index)

            except Exception as e:
                logger.error(f"Error testing {operator.jax_op}: {e}")
                continue

        # Generate summary report
        self.reporter.save_summary(self.all_reports)
        logger.info("Fuzzing complete. Summary saved.")

    def _test_operator(self, operator: OperatorPair) -> OperatorReport:
        """
        Test a single operator across all precisions.

        Args:
            operator: The operator pair to test

        Returns:
            OperatorReport with all test results
        """
        report = OperatorReport(operator=operator)

        # Generate test cases
        test_cases = self.test_generator.generate_tests(
            operator,
            num_cases=self.config.num_test_cases,
            include_edge_cases=True,
        )

        logger.info(f"  Generated {len(test_cases)} test cases")
        self.reporter.save_test_cases(operator, test_cases)

        # Run each test case for each precision
        for test_case in test_cases:
            for precision in self.config.precisions:
                try:
                    # Execute test
                    exec_result = self.executor.execute_test(test_case, precision)

                    # Check precision
                    comparison = self.checker.check(exec_result)

                    # Create test result
                    test_result = TestResult(
                        test_case=test_case,
                        precision=precision,
                        comparison=comparison,
                        jax_status=exec_result.jax_result.status,
                        torch_status=exec_result.torch_result.status,
                        jax_error=exec_result.jax_result.error_message,
                        torch_error=exec_result.torch_result.error_message,
                    )

                    report.add_result(test_result)

                    # Log result
                    status_symbol = (
                        "."
                        if comparison.status == ComparisonStatus.CONSISTENT
                        else "X"
                        if comparison.status == ComparisonStatus.INCONSISTENT
                        else "E"
                    )
                    logger.debug(
                        f"    [{status_symbol}] {test_case.test_id} @ {precision}"
                    )

                except Exception as e:
                    logger.error(f"    Error in {test_case.test_id} @ {precision}: {e}")

        # Log summary
        logger.info(
            f"  Results: {report.consistent_count} consistent, "
            f"{report.inconsistent_count} inconsistent, {report.error_count} errors"
        )

        return report

    def _save_progress(self, last_index: int):
        """Save progress to file for resumption."""
        progress = {
            "last_index": last_index,
            "operators_tested": len(self.all_reports),
        }
        ensure_dir(self.progress_file.parent)
        write_json(self.progress_file, progress)

    def _load_progress(self) -> int:
        """Load progress from file."""
        if self.progress_file.exists():
            progress = read_json(self.progress_file)
            return progress.get("last_index", 0) + 1
        return 0

    def cleanup(self):
        """Clean up resources."""
        if self.test_generator:
            self.test_generator.close()
