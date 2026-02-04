"""JAX-PyTorch Operator Precision Fuzzing Framework."""

from .config import (
    PRECISION_MAP,
    SPECIAL_VALUES,
    TOLERANCE,
    LLMConfig,
    Precision,
    TestConfig,
    load_llm_config_from_env,
)
from .csv_reader import OperatorPair, get_all_operators, read_operator_pairs
from .executor import (
    ExecutionResult,
    ExecutionStatus,
    OperatorExecutor,
    TestExecutionResult,
)
from .fuzzing_inputs import (
    SHAPES,
    generate_default_test_inputs,
    generate_edge_case_inputs,
    generate_input,
)
from .llm_client import LLMClient
from .precision_checker import (
    ComparisonResult,
    ComparisonStatus,
    PrecisionChecker,
    check_precision,
)
from .reporter import OperatorReport, Reporter, TestResult
from .test_generator import InputSpec, TestCase, TestGenerator

__version__ = "0.1.0"

__all__ = [
    # Config
    "PRECISION_MAP",
    "SPECIAL_VALUES",
    "TOLERANCE",
    "LLMConfig",
    "Precision",
    "TestConfig",
    "load_llm_config_from_env",
    # CSV Reader
    "OperatorPair",
    "get_all_operators",
    "read_operator_pairs",
    # Executor
    "ExecutionResult",
    "ExecutionStatus",
    "OperatorExecutor",
    "TestExecutionResult",
    # Fuzzing Inputs
    "SHAPES",
    "generate_default_test_inputs",
    "generate_edge_case_inputs",
    "generate_input",
    # LLM Client
    "LLMClient",
    # Precision Checker
    "ComparisonResult",
    "ComparisonStatus",
    "PrecisionChecker",
    "check_precision",
    # Reporter
    "OperatorReport",
    "Reporter",
    "TestResult",
    # Test Generator
    "InputSpec",
    "TestCase",
    "TestGenerator",
]
