import subprocess as sp
from ..config.config import ROOT_DIR


def exec(op_name, framework, precision, device, mode="eager", test_id=0):
    input_file = ROOT_DIR / "input" / op_name / f"test_{str(test_id).zfill(2)}.npz"
    output_file = (
        ROOT_DIR / "output" / op_name / framework / f"test_{str(test_id).zfill(2)}.npz"
    )
    module = f"fuel_jax.script.{op_name}.{framework}_script"
    cmd = [
        "python",
        "-m",
        module,
        "--precision",
        precision,
        "--device",
        device,
        "--input-file",
        input_file,
        "--output-file",
        output_file,
        "--no-compile-flag" if mode == "eager" else "--compile-flag",
    ]
    print(cmd)
    sp.run(cmd)


if __name__ == "__main__":
    exec("abs", "jax", "FP8_E4M3", "cpu", "compiler", 0)
    exec("abs", "torch", "FP32", "cpu", "compiler", 0)
