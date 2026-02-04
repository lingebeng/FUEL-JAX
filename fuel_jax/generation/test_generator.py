"""Test case generator that combines LLM generation with fallback strategies."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..config import LLMConfig
from ..io.csv_reader import OperatorPair
from ..llm.client import LLMClient
from .fuzzing_inputs import generate_default_test_inputs, generate_edge_case_inputs

# Operators that require two inputs (binary operators)
BINARY_OPERATORS = {
    "jax.lax.add",
    "jax.lax.sub",
    "jax.lax.mul",
    "jax.lax.div",
    "jax.lax.rem",
    "jax.lax.pow",
    "jax.lax.max",
    "jax.lax.min",
    "jax.lax.atan2",
    "jax.lax.nextafter",
    "jax.lax.bitwise_and",
    "jax.lax.bitwise_or",
    "jax.lax.bitwise_xor",
    "jax.lax.eq",
    "jax.lax.ne",
    "jax.lax.lt",
    "jax.lax.le",
    "jax.lax.gt",
    "jax.lax.ge",
    "jax.lax.complex",
}

# Operators with special parameter requirements
SPECIAL_OPERATORS = {
    "jax.lax.clamp": {"num_inputs": 3, "input_names": ["min_val", "x", "max_val"]},
    "jax.lax.integer_pow": {"extra_args": ", 2"},  # Square by default
    "jax.lax.polygamma": {"extra_args_jax": "0, x", "extra_args_torch": "0, x"},
}


@dataclass
class InputSpec:
    """Specification for a single input tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    value_strategy: str
    value_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """A single test case for operator testing."""

    test_id: str
    description: str
    input_specs: list[InputSpec]
    jax_code: str
    torch_code: str
    expected_behavior: str = ""
    special_notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        """Create a TestCase from a dictionary (e.g., from LLM response)."""
        input_specs = []
        for spec in data.get("input_specs", []):
            shape = spec.get("shape", [32, 32])
            if isinstance(shape, list):
                shape = tuple(shape)
            input_specs.append(
                InputSpec(
                    name=spec.get("name", "x"),
                    shape=shape,
                    dtype=spec.get("dtype", "float32"),
                    value_strategy=spec.get("value_strategy", "random_normal"),
                    value_params=spec.get("value_params", {}),
                )
            )

        return cls(
            test_id=data.get("test_id", "test_000"),
            description=data.get("description", ""),
            input_specs=input_specs,
            jax_code=data.get("jax_code", ""),
            torch_code=data.get("torch_code", ""),
            expected_behavior=data.get("expected_behavior", ""),
            special_notes=data.get("special_notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "description": self.description,
            "input_specs": [
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": spec.dtype,
                    "value_strategy": spec.value_strategy,
                    "value_params": spec.value_params,
                }
                for spec in self.input_specs
            ],
            "jax_code": self.jax_code,
            "torch_code": self.torch_code,
            "expected_behavior": self.expected_behavior,
            "special_notes": self.special_notes,
        }


class TestGenerator:
    """Generates test cases using LLM with fallback to predefined strategies."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        use_llm: bool = True,
    ):
        """
        Initialize the test generator.

        Args:
            llm_config: LLM configuration (optional)
            use_llm: Whether to use LLM for generation
        """
        self.use_llm = use_llm
        self._llm_client: LLMClient | None = None
        self._llm_config = llm_config

    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(self._llm_config)
        return self._llm_client

    def generate_tests(
        self,
        operator: OperatorPair,
        num_cases: int = 5,
        include_edge_cases: bool = True,
    ) -> list[TestCase]:
        """
        Generate test cases for an operator pair.

        Args:
            operator: The operator pair to test
            num_cases: Number of test cases to generate
            include_edge_cases: Whether to include edge case tests

        Returns:
            List of TestCase objects
        """
        test_cases: list[TestCase] = []

        # Try LLM generation first
        if self.use_llm:
            try:
                llm_cases = self._generate_from_llm(operator, num_cases)
                test_cases.extend(llm_cases)
                logger.info(
                    f"Generated {len(llm_cases)} test cases from LLM for {operator.jax_op}"
                )
            except Exception as e:
                logger.warning(
                    f"LLM generation failed for {operator.jax_op}, using fallback: {e}"
                )
                fallback_cases = self._generate_fallback(operator, num_cases)
                test_cases.extend(fallback_cases)
        else:
            fallback_cases = self._generate_fallback(operator, num_cases)
            test_cases.extend(fallback_cases)

        # Add edge cases if requested
        if include_edge_cases:
            edge_cases = self._generate_edge_cases(operator)
            test_cases.extend(edge_cases)

        return test_cases

    def _generate_from_llm(
        self, operator: OperatorPair, num_cases: int
    ) -> list[TestCase]:
        """Generate test cases using LLM."""
        client = self._get_llm_client()
        raw_cases = client.generate_test_cases(
            jax_op=operator.jax_op,
            torch_op=operator.torch_op,
            op_type=operator.op_type,
            num_cases=num_cases,
        )

        return [TestCase.from_dict(case) for case in raw_cases]

    def _generate_fallback(
        self, operator: OperatorPair, num_cases: int
    ) -> list[TestCase]:
        """Generate test cases using predefined strategies (fallback)."""
        default_inputs = generate_default_test_inputs(operator.op_type)

        test_cases = []
        for i, input_config in enumerate(default_inputs[:num_cases]):
            test_case = self._create_test_case_from_input(
                operator,
                input_config,
                test_id=f"test_{i:03d}",
            )
            test_cases.append(test_case)

        return test_cases

    def _generate_edge_cases(self, operator: OperatorPair) -> list[TestCase]:
        """Generate edge case test cases."""
        edge_inputs = generate_edge_case_inputs()

        test_cases = []
        for i, input_config in enumerate(edge_inputs):
            test_case = self._create_test_case_from_input(
                operator,
                input_config,
                test_id=f"edge_{i:03d}",
            )
            test_cases.append(test_case)

        return test_cases

    def _create_test_case_from_input(
        self,
        operator: OperatorPair,
        input_config: dict[str, Any],
        test_id: str,
    ) -> TestCase:
        """Create a TestCase from an input configuration."""
        shape = input_config.get("shape", (32, 32))
        if isinstance(shape, list):
            shape = tuple(shape)

        dtype = input_config.get("dtype", "float32")
        value_strategy = input_config.get("value_strategy", "random_normal")
        value_params = input_config.get("value_params", {})

        input_specs = []

        # Check if this is a binary operator
        is_binary = operator.jax_op in BINARY_OPERATORS

        # Check for special operators
        special_config = SPECIAL_OPERATORS.get(operator.jax_op)

        if special_config and "num_inputs" in special_config:
            # Handle special multi-input operators like clamp
            input_names = special_config.get("input_names", ["x", "y", "z"])
            for name in input_names[: special_config["num_inputs"]]:
                input_specs.append(
                    InputSpec(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        value_strategy=value_strategy,
                        value_params=value_params,
                    )
                )
        elif is_binary:
            # Binary operator: need two inputs
            input_specs.append(
                InputSpec(
                    name="x",
                    shape=shape,
                    dtype=dtype,
                    value_strategy=value_strategy,
                    value_params=value_params,
                )
            )
            input_specs.append(
                InputSpec(
                    name="y",
                    shape=shape,
                    dtype=dtype,
                    value_strategy=value_strategy,
                    value_params=value_params,
                )
            )
        else:
            # Unary operator: single input
            input_specs.append(
                InputSpec(
                    name="x",
                    shape=shape,
                    dtype=dtype,
                    value_strategy=value_strategy,
                    value_params=value_params,
                )
            )

        # Generate JAX and PyTorch code
        jax_code = self._generate_jax_code(operator, input_specs)
        torch_code = self._generate_torch_code(operator, input_specs)

        return TestCase(
            test_id=test_id,
            description=input_config.get("description", ""),
            input_specs=input_specs,
            jax_code=jax_code,
            torch_code=torch_code,
            expected_behavior="Results should match within tolerance",
        )

    def _generate_jax_code(
        self, operator: OperatorPair, input_specs: list[InputSpec]
    ) -> str:
        """Generate JAX operator call code."""
        jax_op = operator.jax_op

        # Check for special operators
        special_config = SPECIAL_OPERATORS.get(jax_op)
        if special_config:
            if "extra_args_jax" in special_config:
                return f"{jax_op}({special_config['extra_args_jax']})"
            if "extra_args" in special_config:
                return f"{jax_op}(x{special_config['extra_args']})"

        # Build argument list from input specs
        args = ", ".join(spec.name for spec in input_specs)
        return f"{jax_op}({args})"

    def _generate_torch_code(
        self, operator: OperatorPair, input_specs: list[InputSpec]
    ) -> str:
        """Generate PyTorch operator call code."""
        torch_op = operator.torch_op

        # Handle special cases where the operator already includes arguments
        if "(" in torch_op and ")" in torch_op:
            # e.g., "torch.pow(x, 1/3)" - already has the call syntax
            return torch_op

        # Check for special operators
        special_config = SPECIAL_OPERATORS.get(operator.jax_op)
        if special_config:
            if "extra_args_torch" in special_config:
                return f"{torch_op}({special_config['extra_args_torch']})"
            if "extra_args" in special_config:
                return f"{torch_op}(x{special_config['extra_args']})"

        # Handle method-style calls
        if torch_op.startswith("torch.Tensor."):
            # e.g., "torch.Tensor.to" -> "x.to()"
            method_name = torch_op.replace("torch.Tensor.", "")
            return f"x.{method_name}()"

        # Build argument list from input specs
        args = ", ".join(spec.name for spec in input_specs)
        return f"{torch_op}({args})"

    def close(self):
        """Clean up resources."""
        if self._llm_client is not None:
            self._llm_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
