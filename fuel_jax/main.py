from pathlib import Path
import typer
from loguru import logger

from .config.config import PRECISION_MAP, ROOT_DIR
from .difftesting.validate import _validate
from .utils.utils import list_ops, save_npz, parse_shape, get_op_key, get_dir_list
from .difftesting.exec import _exec
from .generator.generate import Generator

app = typer.Typer()


@app.command()
def gen(
    op_name: str = typer.Option(..., help="JAX op name, or 'all' for all ops"),
    output_file: Path | None = typer.Option(
        None, help="Output .npz file (single op) or directory (all ops)"
    ),
    seed: int = typer.Option(0, help="Random seed"),
    shape: str = typer.Option("4,8", help="Shape like '2,3' or 'scalar'"),
    test_id: int = typer.Option(0, help="Test ID for file naming"),
) -> None:
    gen = Generator(seed=seed)
    ops = (
        [op_name]
        if op_name != "all"
        else [entry["op_name"] for entry in gen.rule.get("unary_operators", [])]
        + [entry["op_name"] for entry in gen.rule.get("binary_operators", [])]
    )

    out_dir = ROOT_DIR / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        inputs = gen.generate(op, shape=parse_shape(shape))
        if op_name == "all":
            out_path = out_dir / get_op_key(op) / f"{str(test_id).zfill(2)}.npz"
        else:
            if output_file is None:
                out_path = out_dir / get_op_key(op) / f"{str(test_id).zfill(2)}.npz"
            elif output_file.suffix:
                out_path = output_file
            else:
                out_path = output_file / f"{get_op_key(op)}.npz"
        save_npz(out_path, **inputs)
        logger.info(f"Saved inputs for {op} -> {out_path}")


@app.command()
def exec(
    op_name: str = typer.Option(..., help="Operator name to execute"),
    device: str = typer.Option("cpu", help="Device to run on (cpu, gpu, tpu)"),
    mode: str = typer.Option("compiler", help="Execution mode (eager or compiler)"),
    test_id: int = typer.Option(0, help="Test ID to execute"),
) -> None:
    ops = list_ops(test_id)
    if op_name != "all":
        if op_name not in ops:
            logger.error(
                f"Operator '{op_name}' not found for test ID {test_id}. Available ops: {ops}"
            )
            return
        ops = [op_name]
    precisions = PRECISION_MAP.keys()

    for op in ops:
        for precision in precisions:
            _exec(op, "jax", precision, device, mode, test_id)
            if device != "tpu":
                _exec(op, "torch", precision, device, mode, test_id)


@app.command()
def validate(
    op_name: str = typer.Option(..., help="Operator name to execute"),
) -> None:
    output_dirs = []
    if op_name == "all":
        for output_dir in get_dir_list(ROOT_DIR / "output"):
            output_dirs.append(ROOT_DIR / "output" / output_dir)
    else:
        output_dirs = [ROOT_DIR / "output" / op_name]

    for output_dir in output_dirs:
        logger.info(f"Validating outputs in {output_dir}")
        result = _validate(output_dir)
        logger.info(f"Final check result for {output_dir.parent.name}: {result}")


if __name__ == "__main__":
    app()
