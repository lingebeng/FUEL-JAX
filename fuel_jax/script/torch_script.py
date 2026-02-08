import torch
import typer
import numpy as np
from pathlib import Path
from loguru import logger
from ..config.config import PRECISION_MAP, ROOT_DIR
from ..utils.utils import (
    save_npz,
    load_npz,
    ndarray2tensor,
    tensor2ndarray,
    get_torch_device,
    get_op_key,
)


def main(
    op_name: str = typer.Option(..., help="PyTorch op name"),
    input_file: Path = typer.Option(None, help="Input file path"),
    output_file: Path = typer.Option(None, help="Output file path"),
    precision: str = typer.Option(
        "FP32", help="Precision setting(FP32, BF16, FP8_E4M3, FP8_E5M2)"
    ),
    device: str = typer.Option("cpu", help="Device setting(cpu or gpu)"),
    compile_flag: bool = typer.Option(
        False, help="Enable Inductor compilation(inductor or not)"
    ),
):
    if not input_file:
        input_file = ROOT_DIR / "input" / get_op_key(op_name) / "00.npz"
    if not output_file:
        output_file = (
            ROOT_DIR
            / "output"
            / get_op_key(op_name)
            / precision
            / f"torch_{device}.npz"
        )
    # Load input data
    inp = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["torch"]

    device = get_torch_device(device=device)
    inp_lst = []
    for v in inp.values():
        if isinstance(v, np.ndarray):
            x = ndarray2tensor(v, dtype=eval(dtype_str)).to(device)
            inp_lst.append(x)
        else:
            inp_lst.append(v)

    # Define the JAX function to compute absolute values
    fn = getattr(torch, op_name)

    try:
        save_kwargs = {}
        out = tensor2ndarray(fn(*inp_lst))
        save_kwargs[f"out_torch_{device}"] = out
        if compile_flag:
            fn_compile = torch.compile(fn)

            out_jit = tensor2ndarray(fn_compile(*inp_lst))
            save_kwargs[f"out_torch_inductor_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        logger.error(f"Error executing {op_name} on PyTorch: {e}")
        exit(1)


if __name__ == "__main__":
    typer.run(main)
