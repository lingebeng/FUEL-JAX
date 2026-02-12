from pathlib import Path
import typer
from loguru import logger

from .config.config import ROOT_DIR, PRECISION_MAP
from .difftesting.validate import _validate
from .utils.utils import (
    save_npz,
    parse_shape,
    get_dir_list,
    load_jax2torch_map,
    RECORD,
    list_ops,
)
from .difftesting.exec import _exec
from .generator.generate import Generator
from .config.config import (
    ExecErrorLogger,
    ValidateInfoLogger,
)

app = typer.Typer()

JAX2TORCH_MAP = load_jax2torch_map()


@app.command()
def gen(
    op_name: str = typer.Option(
        ..., help="JAX op name (jax.lax.xxx) or 'all' for all ops"
    ),
    output_file: Path | None = typer.Option(
        None, help="Output .npz file (single op) or directory (all ops)"
    ),
    seed: int = typer.Option(0, help="Random seed"),
    shape: str = typer.Option("64,64", help="Shape like '2,3' or 'scalar'"),
    test_id: int = typer.Option(0, help="Test ID for file naming"),
) -> None:
    gen = Generator(seed=seed)
    type_name = "other"
    ops = (
        [op_name]
        if op_name != "all"
        else [entry["op_name"] for entry in gen.rule.get(type_name, [])]
    )

    out_dir = ROOT_DIR / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        inputs = gen.generate(op, shape=parse_shape(shape))
        if op_name == "all":
            out_path = out_dir / op / f"{str(test_id).zfill(2)}.npz"
        else:
            if output_file is None:
                out_path = out_dir / op / f"{str(test_id).zfill(2)}.npz"
            elif output_file.suffix:
                out_path = output_file
            else:
                out_path = output_file / f"{op}.npz"
        save_npz(out_path, **inputs)
        logger.info(f"Saved inputs for {op} -> {out_path}")


@app.command()
def exec(
    op_name: str = typer.Option(..., help="JAX op name (jax.lax.xxx) or 'all'"),
    device: str = typer.Option("cpu", help="Device to run on (cpu, gpu, tpu)"),
    mode: str = typer.Option("compiler", help="Execution mode (eager or compiler)"),
    test_id: int = typer.Option(0, help="Test ID to execute"),
) -> None:
    RECORD(
        ExecErrorLogger,
        f"-------------------------- Starting execution for {op_name} --------------------------",
        mode="w",
    )
    if op_name == "all":
        ops = list_ops(test_id=test_id)
    else:
        ops = [op_name]
    precisions = PRECISION_MAP.keys()

    for op in ops:
        RECORD(
            ExecErrorLogger,
            f"-------------------------- Executing op: {op} --------------------------",
        )
        input_file = ROOT_DIR / "input" / op / f"{str(test_id).zfill(2)}.npz"
        if not input_file.exists():
            RECORD(ExecErrorLogger, f"Missing input for {op}: {input_file}\n")
            logger.warning(f"Missing input for {op}: {input_file}")
            continue
        for precision in precisions:
            _exec(op, op, "jax", precision, device, mode, test_id)
            if device != "tpu":
                torch_op = JAX2TORCH_MAP.get(op)
                _exec(op, torch_op, "torch", precision, device, mode, test_id)


@app.command()
def validate(
    op_name: str = typer.Option(..., help="JAX op name (jax.lax.xxx) or 'all'"),
) -> None:
    RECORD(
        ValidateInfoLogger,
        f"-------------------------- Starting validation for {op_name} --------------------------",
        mode="w",
    )
    output_dirs = []
    if op_name == "all":
        for output_dir in get_dir_list(ROOT_DIR / "output"):
            output_dirs.append(ROOT_DIR / "output" / output_dir)
    else:
        output_dirs = [ROOT_DIR / "output" / op_name]

    for output_dir in output_dirs:
        RECORD(
            ValidateInfoLogger,
            f"\n-------------------------- Starting validation for {output_dir.name} --------------------------\n",
        )
        result = _validate(output_dir)
        logger.info(f"Final check result for {output_dir.name}: {result}")


if __name__ == "__main__":
    app()
