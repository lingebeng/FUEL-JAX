"""JAX and PyTorch operator executor."""

import logging
import traceback
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .fuzzing_inputs import generate_input
from .test_generator import InputSpec, TestCase

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of operator execution."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    NOT_SUPPORTED = "not_supported"


@dataclass
class ExecutionResult:
    """Result of executing an operator."""

    status: ExecutionStatus
    output: np.ndarray | None = None
    error_message: str = ""
    error_traceback: str = ""


@dataclass
class TestExecutionResult:
    """Result of executing a test case on both JAX and PyTorch."""

    test_case: TestCase
    precision: str
    jax_result: ExecutionResult
    torch_result: ExecutionResult
    inputs: dict[str, np.ndarray]


class OperatorExecutor:
    """Executes JAX and PyTorch operators for comparison."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize the executor.

        Args:
            device: Device to run on ('cpu' or 'cuda'/'gpu')
        """
        self.device = device
        self._jax_module = None
        self._torch_module = None
        self._setup_done = False

    def setup(self):
        """Set up JAX and PyTorch environments."""
        if self._setup_done:
            return

        try:
            import jax
            import jax.numpy as jnp
            import jax.lax

            self._jax_module = jax
            self._jnp_module = jnp
            self._jax_lax_module = jax.lax

            # Configure JAX device
            if self.device in ("cuda", "gpu"):
                jax.config.update("jax_platform_name", "gpu")
            else:
                jax.config.update("jax_platform_name", "cpu")

            logger.info(f"JAX initialized on {jax.default_backend()}")
        except ImportError as e:
            logger.error(f"Failed to import JAX: {e}")
            raise

        try:
            import torch

            self._torch_module = torch

            # Configure PyTorch device
            if self.device in ("cuda", "gpu") and torch.cuda.is_available():
                self._torch_device = torch.device("cuda")
            else:
                self._torch_device = torch.device("cpu")

            logger.info(f"PyTorch initialized on {self._torch_device}")
        except ImportError as e:
            logger.error(f"Failed to import PyTorch: {e}")
            raise

        self._setup_done = True

    def execute_test(
        self,
        test_case: TestCase,
        precision: str = "FP32",
    ) -> TestExecutionResult:
        """
        Execute a test case on both JAX and PyTorch.

        Args:
            test_case: The test case to execute
            precision: Precision to use (FP32, BF16, FP8_E4M3, FP8_E5M2)

        Returns:
            TestExecutionResult with results from both frameworks
        """
        self.setup()

        # Generate inputs based on test case specifications
        inputs = self._generate_inputs(test_case.input_specs, precision)

        # Execute on JAX
        jax_result = self._execute_jax(test_case.jax_code, inputs, precision)

        # Execute on PyTorch
        torch_result = self._execute_torch(test_case.torch_code, inputs, precision)

        return TestExecutionResult(
            test_case=test_case,
            precision=precision,
            jax_result=jax_result,
            torch_result=torch_result,
            inputs=inputs,
        )

    def _generate_inputs(
        self,
        input_specs: list[InputSpec],
        precision: str,
    ) -> dict[str, np.ndarray]:
        """Generate input arrays from specifications."""
        inputs = {}
        for spec in input_specs:
            # Map precision to numpy dtype
            base_dtype = self._precision_to_numpy_dtype(precision, spec.dtype)

            arr = generate_input(
                shape=spec.shape,
                dtype=base_dtype,
                value_strategy=spec.value_strategy,
                value_params=spec.value_params,
            )
            inputs[spec.name] = arr

        return inputs

    def _precision_to_numpy_dtype(self, precision: str, base_dtype: str) -> str:
        """Convert precision string to numpy dtype."""
        # If base_dtype specifies an integer type, keep it
        if "int" in base_dtype.lower():
            return base_dtype

        precision_map = {
            "FP32": "float32",
            "BF16": "float32",  # NumPy doesn't support bfloat16, use float32 as source
            "FP8_E4M3": "float32",  # Same, will be converted later
            "FP8_E5M2": "float32",
        }
        return precision_map.get(precision, "float32")

    def _execute_jax(
        self,
        code: str,
        inputs: dict[str, np.ndarray],
        precision: str,
    ) -> ExecutionResult:
        """Execute JAX code and return result."""
        try:
            import jax
            import jax.lax
            import jax.numpy as jnp

            # Convert inputs to JAX arrays with proper dtype
            jax_inputs = {}
            jax_dtype = self._precision_to_jax_dtype(precision)

            for name, arr in inputs.items():
                jax_arr = jnp.array(arr)
                if jax_dtype is not None:
                    try:
                        jax_arr = jax_arr.astype(jax_dtype)
                    except (TypeError, ValueError) as e:
                        # Some dtypes may not be supported
                        logger.warning(f"Could not convert to {jax_dtype}: {e}")
                jax_inputs[name] = jax_arr

            # Build execution context
            exec_globals = {
                "jax": jax,
                "jnp": jnp,
                **jax_inputs,
            }
            # Add jax.lax functions to namespace
            for name in dir(jax.lax):
                if not name.startswith("_"):
                    exec_globals[f"jax.lax.{name}"] = getattr(jax.lax, name)

            # Execute the code
            result = eval(code, exec_globals)

            # Convert result to numpy
            if hasattr(result, "numpy"):
                output = np.array(result)
            elif isinstance(result, (np.ndarray, np.generic)):
                output = np.array(result)
            else:
                output = np.array(result)

            return ExecutionResult(status=ExecutionStatus.SUCCESS, output=output)

        except SyntaxError as e:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except ImportError as e:
            return ExecutionResult(
                status=ExecutionStatus.IMPORT_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except TypeError as e:
            return ExecutionResult(
                status=ExecutionStatus.TYPE_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )

    def _execute_torch(
        self,
        code: str,
        inputs: dict[str, np.ndarray],
        precision: str,
    ) -> ExecutionResult:
        """Execute PyTorch code and return result."""
        try:
            import torch
            import torch.nn.functional
            import torch.special
            import torch.linalg
            import torch.fft

            # Convert inputs to PyTorch tensors with proper dtype
            torch_inputs = {}
            torch_dtype = self._precision_to_torch_dtype(precision)

            for name, arr in inputs.items():
                tensor = torch.from_numpy(arr.copy())
                if torch_dtype is not None:
                    try:
                        tensor = tensor.to(dtype=torch_dtype)
                    except (TypeError, RuntimeError) as e:
                        logger.warning(f"Could not convert to {torch_dtype}: {e}")
                tensor = tensor.to(self._torch_device)
                torch_inputs[name] = tensor

            # Build execution context
            exec_globals = {
                "torch": torch,
                **torch_inputs,
            }

            # Execute the code
            result = eval(code, exec_globals)

            # Convert result to numpy
            if hasattr(result, "cpu"):
                output = result.cpu().detach().numpy()
            elif isinstance(result, (np.ndarray, np.generic)):
                output = np.array(result)
            elif isinstance(result, tuple):
                # Some ops return tuples (e.g., topk returns (values, indices))
                output = result[0].cpu().detach().numpy()
            else:
                output = np.array(result)

            return ExecutionResult(status=ExecutionStatus.SUCCESS, output=output)

        except SyntaxError as e:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except ImportError as e:
            return ExecutionResult(
                status=ExecutionStatus.IMPORT_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except TypeError as e:
            return ExecutionResult(
                status=ExecutionStatus.TYPE_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
            )

    def _precision_to_jax_dtype(self, precision: str):
        """Convert precision string to JAX dtype."""
        import jax.numpy as jnp

        dtype_map = {
            "FP32": jnp.float32,
            "BF16": jnp.bfloat16,
            "FP8_E4M3": None,  # Will check if available
            "FP8_E5M2": None,  # Will check if available
        }

        if precision in ("FP8_E4M3", "FP8_E5M2"):
            # Check if FP8 is available
            try:
                if precision == "FP8_E4M3":
                    return jnp.float8_e4m3fn
                else:
                    return jnp.float8_e5m2
            except AttributeError:
                logger.warning(f"JAX does not support {precision}, using FP32")
                return jnp.float32

        return dtype_map.get(precision, jnp.float32)

    def _precision_to_torch_dtype(self, precision: str):
        """Convert precision string to PyTorch dtype."""
        import torch

        dtype_map = {
            "FP32": torch.float32,
            "BF16": torch.bfloat16,
            "FP8_E4M3": None,
            "FP8_E5M2": None,
        }

        if precision in ("FP8_E4M3", "FP8_E5M2"):
            try:
                if precision == "FP8_E4M3":
                    return torch.float8_e4m3fn
                else:
                    return torch.float8_e5m2
            except AttributeError:
                logger.warning(f"PyTorch does not support {precision}, using FP32")
                return torch.float32

        return dtype_map.get(precision, torch.float32)
