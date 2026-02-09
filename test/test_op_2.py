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
    # 显式转换，避免 NumPy 2.0 兼容性问题
    return np.array(jax.device_get(Arr), dtype=np.float32)


def testing(x: np.ndarray, y: np.ndarray, atol: float, rtol: float):
    try:
        # equal_nan=True 会认为 nan == nan, 但 inf != inf
        is_close = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)

        # 使用 np.subtract 的 where 参数或者屏蔽掉非有限值来避免 RuntimeWarning
        mask = np.isfinite(x) & np.isfinite(y)
        diff = np.zeros_like(x)
        diff[mask] = np.abs(x[mask] - y[mask])

        # 处理非有限情况（inf vs inf）
        if not np.all(mask):
            # 如果一个是 inf 另一个不是，那 diff 应该是 inf
            diff[~mask] = np.where(x[~mask] == y[~mask], 0.0, np.inf)

        max_diff = np.nanmax(diff) if not np.isnan(diff).all() else 0.0

        if is_close:
            logger.success(f"Consistent | max_diff: {max_diff}")
        else:
            logger.error(f"Inconsistent | max_diff: {max_diff}")
    except Exception as e:
        logger.error(f"Testing failed: {e}")


inp = "input/jax.lax.zeta/00.npz"

data = load_npz(inp)

x = data["x"]
y = data["q"]

t_x = ndarray2tensor(x)
t_y = ndarray2tensor(y)
j_x = ndarray2Array(x)
j_y = ndarray2Array(y)


t_out = torch.special.zeta(t_x.to(torch.float32), t_y.to(torch.float32))
j_out = jax.lax.zeta(jnp.float32(j_x), jnp.float32(j_y))


print(t_out)
print(j_out)

t_out = tensor2ndarray(t_out)
j_out = Array2ndarray(j_out)


testing(t_out, j_out, atol=1e-5, rtol=1e-5)
