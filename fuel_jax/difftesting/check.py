from ..config.config import ROOT_DIR, TOLERANCE
from ..utils.utils import load_npz
from itertools import combinations as comb
from loguru import logger
import numpy as np


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

    except Exception as e:
        logger.error(e)


def check(op_name, precision, test_id):
    def get_output(framework):
        filename = (
            ROOT_DIR
            / "output"
            / op_name
            / framework
            / f"test_{str(test_id).zfill(2)}.npz"
        )
        data = load_npz(file_path=filename)
        return data

    output_jax, output_torch = get_output("jax"), get_output("torch")

    output = {**output_jax, **output_torch}

    k_output = list(output.keys())

    atol, rtol = TOLERANCE[precision]["atol"], TOLERANCE[precision]["rtol"]

    for x, y in comb(k_output, 2):
        logger.info(f"Difftesting between {x} and {y}")
        testing(output[x], output[y], atol=atol, rtol=rtol)


if __name__ == "__main__":
    check("abs", "FP32", 0)
