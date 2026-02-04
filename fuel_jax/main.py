"""CLI entry point for the JAX-PyTorch precision fuzzing framework."""

import argparse
import sys

from loguru import logger

from .config import TestConfig
from .core.orchestrator import FuzzingOrchestrator


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
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

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
