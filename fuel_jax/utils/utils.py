from pathlib import Path
import csv
import numpy as np
from typing import List, Dict
import torch
from loguru import logger
import jax.numpy as jnp
import jax
import yaml

from ..config.config import ROOT_DIR


def ensure_dir_exists(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def read_csv(file_path: Path) -> List[Dict[str, str]]:
    with file_path.open(mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def save_npz(file_path: Path, **kwargs):
    ensure_dir_exists(file_path.parent)
    np.savez_compressed(file_path, **kwargs)


def load_npz(file_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(file_path)
    dic = {}
    for k, v in data.items():
        dic[k] = v
    return dic


def ndarray2tensor(arr: np.ndarray, dtype=None):
    if dtype is None:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(arr).to(dtype)


def ndarray2Array(arr: np.ndarray, dtype=None):
    if dtype is None:
        return jnp.array(arr)
    else:
        return jnp.array(arr, dtype=dtype)


def tensor2ndarray(tensor):
    # @haifeng only tensor cpu can transform ndarray
    tensor = tensor.to(torch.float32).to("cpu")
    return np.array(tensor, dtype=np.float32)


def Array2ndarray(Arr):
    return np.array(Arr, dtype=np.float32)


def get_jax_device(device: str) -> str:
    if device == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("GPU/MPS not available, falling back to CPU.")
            return "cpu"
    return device


def get_torch_device(device: str) -> str:
    if device == "gpu":
        if "cuda" in str(jax.devices()[0]):
            return "cuda"
        else:
            logger.warning("CUDA not available, falling back to CPU.")
            return "cpu"
    return device


def list_ops(test_id, input_dir=ROOT_DIR / "input") -> List[str]:
    ops = []
    if not input_dir.exists():
        return ops
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / f"{str(test_id).zfill(2)}.npz").exists():
            ops.append(child.name)
    return ops


def load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def parse_shape(value: str) -> tuple:
    if value is None:
        return (4, 8)
    text = value.strip().lower()
    if not text or text in {"scalar", "none"}:
        return ()
    return tuple(int(x) for x in text.split(",") if x.strip())


def get_op_key(op_name: str) -> str:
    return op_name.split(".")[-1]


def get_dir_list(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted([child.name for child in dir_path.iterdir() if child.is_dir()])


def get_file_list(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted([child.name for child in dir_path.iterdir() if child.is_file()])


if __name__ == "__main__":
    x = np.random.rand(100, 64, 64, 3)
    y = np.random.randint(0, 10, size=(100,))
    npz_path = ROOT_DIR / "dataset" / "random_data.npz"
    # ensure_dir_exists(npz_path.parent)
    # save_npz(npz_path,x=x, y=y)

    data = load_npz(npz_path)
    print(len(data))
    print(data["x"].shape, data["y"].shape)
    # csv_path = ROOT_DIR / "dataset" / "jax2torch_lax.csv"

    # # Assuming the CSV file already exists for reading
    # data = read_csv(csv_path)
    # print(data)
