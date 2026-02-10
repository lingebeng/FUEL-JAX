import torch
import jax
import jax.numpy as jnp
import numpy as np


def compare_bf16_exp2(x_numpy):
    print(f"\n{'=' * 20} BF16 对齐测试 {'=' * 20}")

    # 1. 准备数据 (保持完全一致的二进制输入)
    # 必须确保输入在转换过程中没有发生位变
    t_in = torch.from_numpy(x_numpy).to(torch.bfloat16)
    j_in = jnp.array(x_numpy, dtype=jnp.bfloat16)

    # 2. 执行算子
    t_out = torch.ops.aten.exp2.default(t_in)
    j_out = jax.lax.exp2(j_in)

    # ==========================================
    # 核心步骤 A：升维 (Upcast) 到 FP32
    # ==========================================
    # 只有在 FP32 下，减法才是可信的
    t_f32 = t_out.to(torch.float32).numpy()
    j_f32 = np.array(j_out, dtype=np.float32)

    # ==========================================
    # 核心步骤 B：获取原始位 (Bitcast)
    # ==========================================
    # 我们想看在 BF16 的底层，它们差了几个“刻度”
    # PyTorch: view(int16)
    # JAX: bitcast_convert_type
    t_bits = t_out.view(torch.int16).numpy()
    j_bits = np.array(jax.lax.bitcast_convert_type(j_out, jnp.int16))
    print(t_bits)
    print(j_bits)
    # 计算 ULP 差异 (直接整数相减)
    # 注意：处理符号位带来的负数回绕问题，这里简化处理绝对差异
    ulp_diff = np.abs(t_bits.astype(np.int32) - j_bits.astype(np.int32))

    # ==========================================
    # 核心步骤 C：计算指标
    # ==========================================
    # 1. 相对误差 (Relative Error)
    # 避免除以 0
    mask = np.abs(t_f32) > 1e-6
    rel_err = np.zeros_like(t_f32)
    rel_err[mask] = np.abs(t_f32[mask] - j_f32[mask]) / np.abs(t_f32[mask])

    max_rel = np.max(rel_err)
    max_ulp = np.max(ulp_diff)

    # ==========================================
    # 4. 判定标准 (BF16 专用)
    # ==========================================
    print(f"输入范围: [{x_numpy.min()}, {x_numpy.max()}]")
    print(f"Max Relative Error: {max_rel:.5f} ({(max_rel * 100):.3f}%)")
    print(f"Max ULP Diff      : {max_ulp}")

    # 判定逻辑
    if max_ulp == 0:
        print("✅ 完美对齐 (Bit-exact match)")
    elif max_ulp <= 2:
        print("✅ 优秀对齐 (1-2 ULP 差异，通常由底层近似算法差异导致，可接受)")
    elif max_rel < 0.01:  # 小于 1%
        print("⚠️ 一般对齐 (误差 < 1%，但在 BF16 下可能意味着差了 3-5 个 ULP)")
    else:
        print("❌ 对齐失败 (算法实现可能有本质区别)")


# --- 测试用例 ---
# 测试一些典型值：负数、0、正数、大数
x_data = np.array([-10.0, -1.0, 0.0, 0.5, 1.0, 5.0, 8.0, 10.5], dtype=np.float32)
compare_bf16_exp2(x_data)
