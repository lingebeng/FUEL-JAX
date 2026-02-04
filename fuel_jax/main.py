"""Main orchestrator for JAX-PyTorch precision fuzzing framework."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import TestConfig, load_llm_config_from_env
from .csv_reader import OperatorPair, read_operator_pairs
from .executor import OperatorExecutor
from .precision_checker import ComparisonStatus, PrecisionChecker
from .reporter import OperatorReport, Reporter, TestResult
from .test_generator import TestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(progress, f)

    def _load_progress(self) -> int:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                progress = json.load(f)
                return progress.get("last_index", 0) + 1
        return 0

    def cleanup(self):
        """Clean up resources."""
        if self.test_generator:
            self.test_generator.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="JAX-PyTorch Operator Precision Fuzzing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all operators with FP32 precision
  python -m fuel-jax.main --precision FP32

  # Run with multiple precisions
  python -m fuel-jax.main --precision FP32,BF16

  # Resume from index 50
  python -m fuel-jax.main --start-idx 50

  # Test specific operators
  python -m fuel-jax.main --operators "jax.lax.abs,jax.lax.add"

  # Single operator test
  python -m fuel-jax.main --operator jax.lax.abs --precision FP32,BF16

  # Use fallback (no LLM) for test generation
  python -m fuel-jax.main --no-llm
        """,
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="FP32",
        help="Comma-separated list of precisions to test (default: FP32). "
        "Options: FP32, BF16, FP8_E4M3, FP8_E5M2",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance (overrides precision-specific default)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance (overrides precision-specific default)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in CSV (0-based, for resumption)",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default=None,
        help="Comma-separated list of specific operators to test",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Single operator to test (convenience alias for --operators)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for reports (default: output)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="dataset/jax2torch_lax.csv",
        help="Path to operator CSV file",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="Number of test cases to generate per operator (default: 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based test generation (use fallback strategies)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse precision list
    precisions = [p.strip() for p in args.precision.split(",")]

    # Parse operators list
    operators = None
    if args.operator:
        operators = [args.operator]
    elif args.operators:
        operators = [op.strip() for op in args.operators.split(",")]

    # Create config
    config = TestConfig(
        precisions=precisions,
        atol=args.atol,
        rtol=args.rtol,
        start_idx=args.start_idx,
        operators=operators,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        num_test_cases=args.num_cases,
    )

    # Create and run orchestrator
    orchestrator = FuzzingOrchestrator(config)

    # Handle resume
    if args.resume:
        resume_idx = orchestrator._load_progress()
        if resume_idx > 0:
            logger.info(f"Resuming from index {resume_idx}")
            config.start_idx = resume_idx

    try:
        orchestrator.run(use_llm=not args.no_llm)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
