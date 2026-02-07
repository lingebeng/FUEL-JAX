from pathlib import Path
import typer
from loguru import logger

from .config.config import PRECISION_MAP, ROOT_DIR
from .difftesting.check import check
from .utils.utils import list_ops, save_npz, parse_shape, get_op_key
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
def run() -> None:
    test_id = 0
    ops = list_ops(test_id=test_id)
    precisions = list(PRECISION_MAP.keys())
    device = "cpu"
    mode = "compiler"

    log_file = ROOT_DIR / "output" / f"log_{str(test_id).zfill(2)}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    summary = []
    for op in ops:
        for precision in precisions:
            _exec(op, "jax", precision, device, mode, test_id)
            _exec(op, "torch", precision, device, mode, test_id)
    for op in ops:
        for precision in precisions:
            result = check(op, precision, test_id)
            summary.append(
                {
                    "op": op,
                    "precision": precision,
                    "status": result.get("status"),
                    "max_diff": result.get("max_diff"),
                }
            )

    log_lines = ["Summary:"]
    for item in summary:
        line = f"{item['op']}-{item['precision']}-{item['status']}-{item['max_diff']}\n"
        log_lines.append(line)
        logger.info(line)
    log_file.write_text("\n".join(log_lines))


if __name__ == "__main__":
    app()
