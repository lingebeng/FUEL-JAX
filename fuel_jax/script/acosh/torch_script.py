import torch
import typer
from pathlib import Path
from ...config.config import PRECISION_MAP, ROOT_DIR
from ...utils.utils import (
    save_npz,
    load_npz,
    ndarray2tensor,
    tensor2ndarray,
    get_torch_device,
)


def main(
    input_file: Path = typer.Option(
        ROOT_DIR / "input" / "acosh" / "test_00.npz", help="Input file path"
    ),
    output_file: Path = typer.Option(
        ROOT_DIR / "output" / "acosh" / "torch" / "test_00.npz", help="Output file path"
    ),
    precision: str = typer.Option("FP32", help="Precision setting"),
    device: str = typer.Option("cpu", help="Device setting"),
    compile_flag: bool = typer.Option(False, help="Enable JIT compilation"),
):
    # Load input data
    data = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["torch"]

    device = get_torch_device(device=device)
    x = ndarray2tensor(data["x"], dtype=eval(dtype_str)).to(device)

    # Define the JAX function to compute absolute values
    def fn(x):
        return torch.acosh(x)

    try:
        save_kwargs = {}
        out = tensor2ndarray(fn(x))
        save_kwargs[f"out_torch_{device}"] = out
        if compile_flag:
            fn_compile = torch.compile(fn)

            out_jit = tensor2ndarray(fn_compile(x))
            save_kwargs[f"out_torch_inductor_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    typer.run(main)
