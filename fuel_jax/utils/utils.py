from pathlib import Path
import csv
import numpy as np
from typing import List, Dict
import torch
from loguru import logger
import jax.numpy as jnp

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


def tensor2ndarray(tensor: torch.tensor):
    return np.array(tensor, dtype=np.float32)


def ndarray2Array(arr: np.ndarray, dtype=None):
    if dtype is None:
        return jnp.array(arr)
    else:
        return jnp.array(arr, dtype=dtype)


def Array2ndarray(Arr: jnp.array):
    return np.array(Arr, dtype=np.float32)


def get_torch_device(device: str) -> str:
    if device == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            logger.waring("GPU/MPS not available, falling back to CPU.")
            return "cpu"
    return device


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
