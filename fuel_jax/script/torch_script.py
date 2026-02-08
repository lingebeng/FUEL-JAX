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
    _resolve_dotted,
)


def main(
    op_name: str = typer.Option(
        ..., help="PyTorch op name (mapped to torch via jax2torch_map.csv)"
    ),
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
        input_file = ROOT_DIR / "input" / op_name / "00.npz"
    if not output_file:
        output_file = ROOT_DIR / "output" / op_name / precision / f"torch_{device}.npz"
    # Load input data
    inp = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["torch"]

    device = get_torch_device(device=device)
    inp_lst = []
    for v in sorted(inp.values()):
        if isinstance(v, np.ndarray):
            v = ndarray2tensor(v, dtype=eval(dtype_str)).to(device)
        inp_lst.append(v)

    op_suffix = op_name.split(".", 1)[1]
    fn = _resolve_dotted(torch, op_suffix)

    try:
        save_kwargs = {}
        out = tensor2ndarray(fn(*inp_lst))
        save_kwargs[f"out_torch_{device}"] = out
        if compile_flag:
            if fn is not None:
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
