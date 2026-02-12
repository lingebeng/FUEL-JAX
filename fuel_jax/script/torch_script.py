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


_AXIS_OPS = {
    "jax.lax.argmax",
    "jax.lax.argmin",
    "jax.lax.cumlogsumexp",
    "jax.lax.cummax",
    "jax.lax.cummin",
    "jax.lax.cumprod",
    "jax.lax.cumsum",
}

_AXES_OPS = {
    "jax.lax.reduce_max",
    "jax.lax.reduce_min",
    "jax.lax.reduce_sum",
    "jax.lax.reduce_prod",
}


def _normalize_axes(inp: dict) -> None:
    if "axes" not in inp:
        return
    axes_arr = np.asarray(inp["axes"]).reshape(-1)
    ndim = int(inp["operand"].ndim)
    if ndim == 0:
        inp["axes"] = ()
        return
    clipped = [min(max(int(a), 0), ndim - 1) for a in axes_arr.tolist()]
    inp["axes"] = tuple(sorted(set(clipped)))


def _normalize_axis(inp: dict, jax_op: str) -> None:
    if jax_op not in _AXIS_OPS:
        return
    inp["axis"] = min(int(inp["axis"]), inp["operand"].ndim - 1)
    if "index_dtype" in inp:
        del inp["index_dtype"]


def _run_op(fn, inp: dict, jax_op: str, op_name: str):
    if jax_op == "jax.lax.dot" and op_name == "torch.ops.aten.einsum.default":
        lhs = inp["lhs"]
        rhs = inp["rhs"]
        eq = _dot_einsum_equation(lhs, rhs)
        return fn(eq, [lhs, rhs])

    if jax_op == "jax.lax.reduce_sum":
        return torch.sum(inp["operand"], dim=inp["axes"])

    if jax_op == "jax.lax.reduce_prod":
        out_t = inp["operand"]
        for dim in sorted(inp["axes"], reverse=True):
            out_t = torch.prod(out_t, dim=dim)
        return out_t

    if jax_op in ("jax.lax.reduce_max", "jax.lax.reduce_min"):
        return fn(inp["operand"], list(inp["axes"]), False)

    return fn(*list(inp.values()))


def _run_op_compile(fn, inp: dict, jax_op: str, op_name: str):
    if jax_op == "jax.lax.dot" and op_name == "torch.ops.aten.einsum.default":
        lhs = inp["lhs"]
        rhs = inp["rhs"]
        eq = _dot_einsum_equation(lhs, rhs)

        def _dot_einsum_impl(a, b):
            return fn(eq, [a, b])

        return torch.compile(_dot_einsum_impl)(lhs, rhs)

    if jax_op == "jax.lax.reduce_sum":
        axes = inp["axes"]
        return torch.compile(lambda x: torch.sum(x, dim=axes))(inp["operand"])

    if jax_op == "jax.lax.reduce_prod":
        axes = tuple(sorted(inp["axes"], reverse=True))

        def _reduce_prod_impl(x):
            y = x
            for dim in axes:
                y = torch.prod(y, dim=dim)
            return y

        return torch.compile(_reduce_prod_impl)(inp["operand"])

    if jax_op in ("jax.lax.reduce_max", "jax.lax.reduce_min"):
        axes = list(inp["axes"])
        return torch.compile(lambda x: fn(x, axes, False))(inp["operand"])

    return torch.compile(fn)(*list(inp.values()))


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
    target_dtype = eval(PRECISION_MAP[precision]["torch"])
    target_device = get_torch_device(device=device)

    for k, v in inp.items():
        if k == "axes":
            continue
        if v.shape == ():
            if k in ("axis", "index_dtype"):
                v = int(v)
            elif np.issubdtype(v.dtype, np.floating):
                v = float(v)
            else:
                if "Tensor" in op_name:
                    v = torch.tensor(v)
                else:
                    v = int(v)
        else:
            v = ndarray2tensor(v, dtype=target_dtype).to(target_device)
        inp[k] = v
    op_suffix = op_name.split(".", 1)[1]

    _normalize_axes(inp)
    _normalize_axis(inp, jax_op)

    fn = _resolve_dotted(torch, op_suffix)

    try:
        save_kwargs = {}
        out = tensor2ndarray(_run_op(fn, inp, jax_op, op_name))
        save_kwargs[f"out_torch_{device}"] = out
        if compile_flag:
            out_jit = tensor2ndarray(_run_op_compile(fn, inp, jax_op, op_name))
            save_kwargs[f"out_torch_inductor_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        logger.error(f"Error executing {op_name} on PyTorch: {e}")
        exit(1)


if __name__ == "__main__":
    typer.run(main)
