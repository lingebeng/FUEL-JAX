import os

# 关键：禁止 JAX 预分配显存，防止与 PyTorch 冲突
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import torch
from loguru import logger


def tensor2ndarray(tensor):
    # 移除多余的 np.array 构造，直接使用 .numpy() 效率更高且兼容 NumPy 2.0
    return tensor.detach().cpu().to(torch.float32).numpy()


def array2ndarray(arr):
    # jax.device_get 是将数据从加速器搬回 CPU 的标准做法
    return np.array(jax.device_get(arr), dtype=np.float32)


def testing(x: np.ndarray, y: np.ndarray, atol: float, rtol: float):
    """
    更专业的对齐测试：增加相对误差分析和最差值坐标定位
    """
    try:
        # 1. 基础判断
        is_close = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)

        # 2. 计算绝对误差
        diff = np.abs(x - y)
        max_diff = np.nanmax(diff) if not np.isnan(diff).all() else 0.0

        # 3. 计算相对误差 (增加 epsilon 避免除零)
        rel_diff = diff / (np.abs(y) + 1e-8)
        max_rel_diff = np.nanmax(rel_diff) if not np.isnan(rel_diff).all() else 0.0

        # 4. 定位最大误差发生的具体位置 (有助于排查特定数值边界)
        idx = (
            np.unravel_index(np.nanargmax(diff), diff.shape)
            if not np.isnan(diff).all()
            else (0, 0)
        )

        log_msg = (
            f"max_abs_diff: {max_diff:.2e} | "
            f"max_rel_diff: {max_rel_diff:.2e} | "
            f"at index {idx} values: torch={x[idx]:.6f}, jax={y[idx]:.6f}"
        )

        if is_close:
            logger.success(f"✅ Consistent | {log_msg}")
        else:
            # 如果误差极小（比如 < 1e-6），即使 allclose 失败，通常也可视为计算噪声
            if max_diff < 1e-6:
                logger.warning(f"⚠️ Marginal Inconsistent (Numerical Noise) | {log_msg}")
            else:
                logger.error(f"❌ Inconsistent | {log_msg}")

    except Exception as e:
        logger.error(f"Validation Error: {e}")


# --- 数据准备 ---
rng = np.random.default_rng(42)
# 1. 确保原始 Numpy 数据就是 FP32
x_np = rng.uniform(low=1.01, high=10.0, size=(128, 128)).astype(np.float32)
q_np = rng.uniform(low=1.01, high=10.0, size=(128, 128)).astype(np.float32)

# 2. PyTorch：显式指定 dtype=torch.float32
t_x = torch.from_numpy(x_np).to(device="cpu", dtype=torch.float32)
t_q = torch.from_numpy(q_np).to(device="cpu", dtype=torch.float32)

# 3. JAX：显式指定 dtype=jnp.float32
j_x = jnp.array(x_np, dtype=jnp.float32)
j_q = jnp.array(q_np, dtype=jnp.float32)

# 执行运算
t_out = torch.special.zeta(t_x, t_q)
j_out = jax.lax.zeta(j_x, j_q)

# 转换回 CPU
t_out_np = tensor2ndarray(t_out)
j_out_np = array2ndarray(j_out)

# 验证
testing(t_out_np, j_out_np, atol=1e-5, rtol=1e-5)
