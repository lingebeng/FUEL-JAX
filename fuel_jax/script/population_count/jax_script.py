import jax
import jax.numpy as jnp  # noqa: F401
import typer
from pathlib import Path
from ...config.config import PRECISION_MAP, ROOT_DIR
from ...utils.utils import save_npz, load_npz, ndarray2Array, Array2ndarray


def main(
    input_file: Path = typer.Option(
        ROOT_DIR / "input" / "population_count" / "test_00.npz", help="Input file path"
    ),
    output_file: Path = typer.Option(
        ROOT_DIR / "output" / "population_count" / "jax" / "test_00.npz",
        help="Output file path",
    ),
    precision: str = typer.Option("FP32", help="Precision setting"),
    device: str = typer.Option("cpu", help="Device setting"),
    compile_flag: bool = typer.Option(False, help="Enable JIT compilation"),
):
    # Load input data
    data = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["jax"]

    target_device = jax.devices(device)[0]

    x = jax.device_put(
        ndarray2Array(data["x"], dtype=eval(dtype_str)), device=target_device
    )

    # Define the JAX function to compute absolute values
    def fn(x):
        return jax.lax.population_count(x)

    try:
        save_kwargs = {}
        out = Array2ndarray(fn(x))
        save_kwargs[f"out_jax_{device}"] = out
        if compile_flag:
            fn_compile = jax.jit(fn)
            out_jit = Array2ndarray(fn_compile(x))
            save_kwargs[f"out_jax_xla_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    typer.run(main)
