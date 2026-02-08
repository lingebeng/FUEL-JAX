from ..config.config import ROOT_DIR, TOLERANCE
from ..utils.utils import load_npz, get_dir_list, get_file_list, RECORD
from itertools import combinations as comb
from loguru import logger
import numpy as np

from pathlib import Path

from ..config.config import (
    ValidateInfoLogger,
)


def testing(x: np.ndarray, y: np.ndarray, atol: float, rtol: float):
    try:
        is_close = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        diff = np.abs(x - y)
        max_diff = np.nanmax(diff) if not np.isnan(diff).all() else 0.0
        if is_close:
            status = "Consistent"
            logger.success("Consistent")
            logger.success(f"max_diff:{max_diff}")
        else:
            status = "Inconsistent"
            logger.error("Inconsistent")
            logger.error(f"max_diff:{max_diff}")

        RECORD(
            ValidateInfoLogger,
            f"status: {status}, max_diff: {max_diff}\n",
        )

    except Exception as e:
        RECORD(
            ValidateInfoLogger,
            f"Error during testing: {str(e)}\n",
        )
        logger.error(e)


def _validate(output_dir: Path):
    logger.info(f"Validating outputs in {output_dir}")
    dirs = get_dir_list(output_dir)
    for dir_precision in dirs:
        logger.info(f"Validating precision: {dir_precision}")
        RECORD(
            ValidateInfoLogger,
            f"\n-------------------------- Validating precision: {dir_precision} --------------------------\n",
        )
        atol, rtol = TOLERANCE[dir_precision]["atol"], TOLERANCE[dir_precision]["rtol"]
        files = get_file_list(output_dir / dir_precision)
        outputs = {}
        for file in files:
            output = load_npz(output_dir / dir_precision / file)
            outputs = {**outputs, **output}

        k_output = list(outputs.keys())
        if len(k_output) < 2:
            RECORD(
                ValidateInfoLogger,
                f"Not enough outputs to compare in {output_dir / dir_precision}, skipping.",
            )
            logger.warning("Not enough outputs to compare, skipping.")
            continue

        for x, y in comb(k_output, 2):
            logger.info(f"Difftesting between {x} and {y}")
            RECORD(
                ValidateInfoLogger,
                f"Difftesting between     【{x}】      and     【{y}】     ",
            )
            testing(outputs[x], outputs[y], atol=atol, rtol=rtol)


if __name__ == "__main__":
    # check("abs", "FP32", 0)

    _validate(ROOT_DIR / "output" / "abs")
