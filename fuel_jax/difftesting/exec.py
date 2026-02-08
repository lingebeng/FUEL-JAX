import subprocess as sp
import sys
from loguru import logger

from fuel_jax.utils.utils import RECORD
from ..config.config import ROOT_DIR
from ..config.config import (
    ExecErrorLogger,
)


def _exec(jax_op, op_name, framework, precision, device, mode="eager", test_id=0):
    input_file = ROOT_DIR / "input" / jax_op / f"{str(test_id).zfill(2)}.npz"
    module = f"fuel_jax.script.{framework}_script"
    RECORD(
        ExecErrorLogger,
        f"Executing {op_name} in {framework}-{device}-{precision}-{mode} with input: {input_file}",
    )
    cmd = [
        sys.executable,
        "-m",
        module,
        "--jax-op",
        jax_op,
        "--op-name",
        op_name,
        "--precision",
        precision,
        "--device",
        device,
        "--input-file",
        input_file,
        "--no-compile-flag" if mode == "eager" else "--compile-flag",
    ]
    logger.info(f"Built cmd:{cmd}")
    result = sp.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error(
            f"{op_name} exec failed in {framework}-{device}-{precision}-{mode}"
        )
        RECORD(
            ExecErrorLogger,
            f"{op_name} exec failed in {framework}-{device}-{precision}-{mode}",
        )
        if result.stdout:
            logger.info(result.stdout)
            RECORD(ExecErrorLogger, result.stdout)
        if result.stderr:
            logger.error(result.stderr)
            RECORD(ExecErrorLogger, result.stderr)
    else:
        logger.success(
            f"{op_name} exec successfully in {framework}-{device}-{precision}-{mode}"
        )
        RECORD(
            ExecErrorLogger,
            f"{op_name} exec successfully in {framework}-{device}-{precision}-{mode}",
        )
    RECORD(
        ExecErrorLogger,
        f"Finished execution for {op_name}\n",
    )
    return result.returncode
