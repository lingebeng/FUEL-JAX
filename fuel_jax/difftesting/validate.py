from ..config.config import ROOT_DIR, TOLERANCE, DIFF_ORACLE_THRESHOLDS
from ..utils.utils import load_npz, get_dir_list, get_file_list, RECORD
from itertools import combinations as comb
from loguru import logger
import numpy as np

from pathlib import Path

from ..config.config import (
    ValidateInfoLogger,
)
from .oracle import Oracle, evaluate_diff, format_metrics, oracle_rank


def testing(
    x: np.ndarray,
    y: np.ndarray,
    *,
    precision: str,
    atol: float,
    rtol: float,
) -> Oracle:
    try:
        result = evaluate_diff(
            x,
            y,
            atol=atol,
            rtol=rtol,
            criteria=DIFF_ORACLE_THRESHOLDS[precision],
        )
        status = result.oracle.name
        metrics_str = format_metrics(result.metrics)

        if result.oracle == Oracle.PASS:
            logger.success(status)
        elif result.oracle == Oracle.WARN:
            logger.warning(status)
        else:
            logger.error(status)
        logger.info(result.message)
        logger.info(metrics_str)

        RECORD(
            ValidateInfoLogger,
            f"status: {status}, reason: {result.message}, metrics: {metrics_str}\n",
        )
        return result.oracle

    except Exception as e:
        RECORD(
            ValidateInfoLogger,
            f"Error during testing: {str(e)}\n",
        )
        logger.error(e)
        return Oracle.FAIL


def _validate(output_dir: Path):
    logger.info(f"Validating outputs in {output_dir}")
    overall_result = Oracle.PASS
    dirs = get_dir_list(output_dir)
    for dir_precision in dirs:
        if dir_precision not in TOLERANCE:
            RECORD(
                ValidateInfoLogger,
                f"Unknown precision {dir_precision}, skipping.\n",
            )
            logger.warning(f"Unknown precision {dir_precision}, skipping.")
            continue
        if dir_precision not in DIFF_ORACLE_THRESHOLDS:
            RECORD(
                ValidateInfoLogger,
                f"Missing oracle thresholds for {dir_precision}, skipping.\n",
            )
            logger.warning(f"Missing oracle thresholds for {dir_precision}, skipping.")
            continue

        logger.info(f"Validating precision: {dir_precision}")
        RECORD(
            ValidateInfoLogger,
            f"\n-------------------------- Validating precision: {dir_precision} --------------------------\n",
        )
        atol, rtol = TOLERANCE[dir_precision]["atol"], TOLERANCE[dir_precision]["rtol"]
        precision_result = Oracle.PASS
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
            pair_result = testing(
                outputs[x],
                outputs[y],
                precision=dir_precision,
                atol=atol,
                rtol=rtol,
            )
            if oracle_rank(pair_result) > oracle_rank(precision_result):
                precision_result = pair_result

        RECORD(
            ValidateInfoLogger,
            f"precision_result: {dir_precision} -> {precision_result.name}\n",
        )
        logger.info(f"{dir_precision} summary: {precision_result.name}")
        if oracle_rank(precision_result) > oracle_rank(overall_result):
            overall_result = precision_result

    RECORD(ValidateInfoLogger, f"overall_result: {overall_result.name}\n")
    return overall_result.name


if __name__ == "__main__":
    # check("abs", "FP32", 0)

    _validate(ROOT_DIR / "output" / "abs")
