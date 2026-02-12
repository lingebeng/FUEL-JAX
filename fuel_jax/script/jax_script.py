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


def _dot_dimension_numbers(lhs_ndim: int, rhs_ndim: int):
    if lhs_ndim == 3 and rhs_ndim == 3:
        # (b, m, k) x (b, k, n) -> (b, m, n)
        return (((2,), (1,)), ((0,), (0,)))
    return None


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


def _static_argnames(op_name: str) -> tuple[str, ...]:
    if op_name in ("jax.lax.argmax", "jax.lax.argmin"):
        return ("axis", "index_dtype")
    if op_name == "jax.lax.top_k":
        return ("k",)
    if op_name in _AXIS_OPS:
        return ("axis",)
    if op_name in _AXES_OPS:
        return ("axes",)
    return ()


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


def _normalize_axis(inp: dict, op_name: str) -> None:
    if op_name not in _AXIS_OPS:
        return
    inp["axis"] = min(int(inp["axis"]), inp["operand"].ndim - 1)
    if "index_dtype" in inp:
        inp["index_dtype"] = jnp.int32


def _run_op(fn, inp: dict, op_name: str):
    if op_name == "jax.lax.top_k":
        return fn(**inp)[0]
    if op_name != "jax.lax.dot":
        return fn(**inp)
    dnums = _dot_dimension_numbers(inp["lhs"].ndim, inp["rhs"].ndim)
    if dnums is None:
        return fn(**inp)
    return fn(inp["lhs"], inp["rhs"], dimension_numbers=dnums)


def _run_op_jit(fn, inp: dict, op_name: str):
    if op_name == "jax.lax.top_k":
        static_argnames = _static_argnames(op_name)
        return jax.jit(fn, static_argnames=static_argnames)(**inp)[0]

    if op_name == "jax.lax.dot":
        dnums = _dot_dimension_numbers(inp["lhs"].ndim, inp["rhs"].ndim)
        if dnums is None:
            return jax.jit(fn)(**inp)
        # Keep dimension_numbers static for JIT to avoid tracer int conversion.
        fn_compile = jax.jit(lambda lhs, rhs: fn(lhs, rhs, dimension_numbers=dnums))
        return fn_compile(inp["lhs"], inp["rhs"])

    static_argnames = _static_argnames(op_name)
    if static_argnames:
        return jax.jit(fn, static_argnames=static_argnames)(**inp)
    return jax.jit(fn)(**inp)


def main(
    jax_op: str = typer.Option(
        None, help="Operator name (mapped to torch via jax2torch_map.csv)"
    ),
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
    target_dtype = eval(PRECISION_MAP[precision]["jax"])

    target_device = jax.devices(get_jax_device(device=device))[0]

    for k, v in inp.items():
        if k == "axes":
            continue
        if v.shape == ():
            if k in ("axis", "index_dtype"):
                v = int(v)
            elif np.issubdtype(v.dtype, np.floating):
                v = v.astype(target_dtype)
            else:
                v = int(v)
        else:
            v = jax.device_put(
                ndarray2Array(v, dtype=target_dtype), device=target_device
            )
        inp[k] = v

    _normalize_axes(inp)
    _normalize_axis(inp, op_name)

    op_suffix = op_name.split(".", 1)[1]
    fn = _resolve_dotted(jax, op_suffix)

    try:
        save_kwargs = {}
        out = Array2ndarray(_run_op(fn, inp, op_name))
        save_kwargs[f"out_jax_{device}"] = out
        if compile_flag:
            out_jit = Array2ndarray(_run_op_jit(fn, inp, op_name))
            save_kwargs[f"out_jax_xla_{device}"] = out_jit

        if save_kwargs:
            save_npz(output_file, **save_kwargs)
        exit(0)
    except Exception as e:
        logger.error(f"Error executing {op_name} on JAX: {e}")
        exit(1)


if __name__ == "__main__":
    typer.run(main)
