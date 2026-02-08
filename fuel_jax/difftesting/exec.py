import subprocess as sp
from loguru import logger
from ..config.config import ROOT_DIR


def _exec(op_name, framework, precision, device, mode="eager", test_id=0):
    input_file = ROOT_DIR / "input" / op_name / f"{str(test_id).zfill(2)}.npz"
    module = f"fuel_jax.script.{framework}_script"
    cmd = [
        "python",
        "-m",
        module,
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
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
    else:
        logger.success(
            f"{op_name} exec successfully in {framework}-{device}-{precision}-{mode}"
        )
    return result.returncode
