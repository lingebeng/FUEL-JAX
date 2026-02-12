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


def _dot_einsum_equation(lhs: torch.Tensor, rhs: torch.Tensor) -> str:
    if lhs.ndim == 1 and rhs.ndim == 1:
        return "k,k->"
    if lhs.ndim == 2 and rhs.ndim == 1:
        return "mk,k->m"
    if lhs.ndim == 1 and rhs.ndim == 2:
        return "k,kn->n"
    if lhs.ndim == 2 and rhs.ndim == 2:
        return "mk,kn->mn"
    if lhs.ndim == 3 and rhs.ndim == 3:
        return "bmk,bkn->bmn"
    raise ValueError(
        f"Unsupported rank pair for jax.lax.dot -> einsum mapping: {lhs.ndim}, {rhs.ndim}"
    )


def main(
    jax_op: str = typer.Option(
        None, help="Operator name (mapped to torch via jax2torch_map.csv)"
    ),
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
    if jax_op is None:
        jax_op = op_name
    if not input_file:
        input_file = ROOT_DIR / "input" / jax_op / "00.npz"
    if not output_file:
        output_file = ROOT_DIR / "output" / jax_op / precision / f"torch_{device}.npz"
    # Load input data
    inp = load_npz(input_file)
    # Convert input data to JAX array with specified precision
    dtype_str = PRECISION_MAP[precision]["torch"]

    target_device = get_torch_device(device=device)
    inp_converted = {}
    for k, v in inp.items():
        if v.shape == ():
            if v.dtype == np.float64:
                v = float(v)
            else:
                if "Tensor" in op_name:
                    v = torch.tensor(v)
                else:
                    v = int(v)
        else:
            v = ndarray2tensor(v, dtype=eval(dtype_str)).to(target_device)
        inp_converted[k] = v
    op_suffix = op_name.split(".", 1)[1]
    fn = _resolve_dotted(torch, op_suffix)

    try:
        save_kwargs = {}
        if jax_op == "jax.lax.dot" and op_name == "torch.ops.aten.einsum.default":
            lhs = inp_converted["lhs"]
            rhs = inp_converted["rhs"]
            eq = _dot_einsum_equation(lhs, rhs)
            out = tensor2ndarray(fn(eq, [lhs, rhs]))
        else:
            out = tensor2ndarray(fn(*list(inp_converted.values())))
        save_kwargs[f"out_torch_{device}"] = out
        if compile_flag:
            if fn is not None:
                if (
                    jax_op == "jax.lax.dot"
                    and op_name == "torch.ops.aten.einsum.default"
                ):
                    lhs = inp_converted["lhs"]
                    rhs = inp_converted["rhs"]
                    eq = _dot_einsum_equation(lhs, rhs)

                    def _dot_einsum_impl(a, b):
                        return fn(eq, [a, b])

                    fn_compile = torch.compile(_dot_einsum_impl)
                    out_jit = tensor2ndarray(fn_compile(lhs, rhs))
                else:
                    fn_compile = torch.compile(fn)
                    out_jit = tensor2ndarray(fn_compile(*list(inp_converted.values())))
                save_kwargs[f"out_torch_inductor_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        logger.error(f"Error executing {op_name} on PyTorch: {e}")
        exit(1)


if __name__ == "__main__":
    typer.run(main)
