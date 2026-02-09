import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict
import torch
from loguru import logger


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


def testing(x: np.ndarray, y: np.ndarray, atol: float, rtol: float):
    try:
        is_close = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        diff = np.abs(x - y)
        max_diff = np.nanmax(diff) if not np.isnan(diff).all() else 0.0
        if is_close:
            logger.success("Consistent")
            logger.success(f"max_diff:{max_diff}")
        else:
            logger.error("Inconsistent")
            logger.error(f"max_diff:{max_diff}")

    except Exception as e:
        logger.error(e)


inp = "input/jax.lax.sin/00.npz"


data = load_npz(inp)

x = data["x"]

t_x = ndarray2tensor(x)
j_x = ndarray2Array(x)

t_x = ndarray2tensor(x)
j_x = ndarray2Array(x)

t_out = torch.sin(t_x)
j_out = jax.lax.sin(j_x)

t_out = tensor2ndarray(t_out)
j_out = Array2ndarray(j_out)

testing(t_out, j_out, atol=1e-6, rtol=1e-5)
