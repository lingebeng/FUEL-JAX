from ..config.config import ROOT_DIR, TOLERANCE
from ..utils.utils import load_npz, get_dir_list, get_file_list
from itertools import combinations as comb
from loguru import logger
import numpy as np

from pathlib import Path


def testing(x: np.ndarray, y: np.ndarray, atol: float, rtol: float):
    try:
        is_close = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        diff = np.abs(x - y)
        max_diff = np.nanmax(diff) if not np.isnan(diff).all() else 0.0
        if is_close:
            logger.success("Consistent")
        else:
            logger.warning("Inconsistent")
        logger.info(f"max_diff:{max_diff}")
        return {"status": "ok" if is_close else "mismatch", "max_diff": max_diff}

    except Exception as e:
        logger.error(e)
        return {"status": "error", "max_diff": None}


def _validate(output_dir: Path):
    logger.info(f"Validating outputs in {output_dir}")
    dirs = get_dir_list(output_dir)
    for dir_precision in dirs:
        atol, rtol = TOLERANCE[dir_precision]["atol"], TOLERANCE[dir_precision]["rtol"]
        files = get_file_list(output_dir / dir_precision)
        outputs = {}
        for file in files:
            output = load_npz(output_dir / dir_precision / file)
            outputs = {**outputs, **output}

        k_output = list(outputs.keys())
        if len(k_output) < 2:
            logger.warning("Not enough outputs to compare, skipping.")
            return {"status": "skip", "max_diff": None}

        last = {"status": "skip", "max_diff": None}
        for x, y in comb(k_output, 2):
            logger.info(f"Difftesting between {x} and {y}")
            last = testing(outputs[x], outputs[y], atol=atol, rtol=rtol)
        logger.info(f"Final check result for {dir_precision}: {last}")


if __name__ == "__main__":
    # check("abs", "FP32", 0)

    _validate(ROOT_DIR / "output" / "abs")
