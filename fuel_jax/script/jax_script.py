import jax
import jax.numpy as jnp  # noqa: F401
import typer
import numpy as np
from pathlib import Path
from loguru import logger
from ..config.config import PRECISION_MAP, ROOT_DIR
from ..utils.utils import (
    save_npz,
    load_npz,
    ndarray2Array,
    Array2ndarray,
    get_jax_device,
    _resolve_dotted,
)


def main(
    op_name: str = typer.Option(..., help="JAX op name"),
    input_file: Path = typer.Option(None, help="Input file path"),
    output_file: Path = typer.Option(None, help="Output file path"),
    precision: str = typer.Option(
        "FP32", help="Precision setting(FP32, BF16, FP8_E4M3, FP8_E5M2)"
    ),
    device: str = typer.Option("cpu", help="Device setting(cpu, gpu or tpu)"),
    compile_flag: bool = typer.Option(False, help="Enable JIT compilation(xla or not)"),
):
    if not input_file:
        input_file = ROOT_DIR / "input" / op_name / "00.npz"

    if not output_file:
        output_file = ROOT_DIR / "output" / op_name / precision / f"jax_{device}.npz"

    # Load input data
    inp = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["jax"]

    device = jax.devices(get_jax_device(device=device))[0]

    for k, v in inp.items():
        if isinstance(v, np.ndarray):
            inp[k] = jax.device_put(
                ndarray2Array(v, dtype=eval(dtype_str)), device=device
            )

    op_suffix = op_name.split(".", 1)[1]
    fn = _resolve_dotted(jax, op_suffix)

    try:
        save_kwargs = {}
        out = Array2ndarray(fn(**inp))
        save_kwargs[f"out_jax_{device}"] = out
        if compile_flag:
            fn_compile = jax.jit(fn)
            out_jit = Array2ndarray(fn_compile(**inp))
            save_kwargs[f"out_jax_xla_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        logger.error(f"Error executing {op_name} on JAX: {e}")
        exit(1)


if __name__ == "__main__":
    typer.run(main)
