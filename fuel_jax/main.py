from loguru import logger

from .config.config import PRECISION_MAP, ROOT_DIR
from .difftesting.check import check
from .utils.utils import list_ops
from .difftesting.exec import _exec


def main():
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
    main()
