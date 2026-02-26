import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import jax
import jax.numpy as jnp
import torch
import numpy as np

rng = np.random.default_rng(seed=0)

x = rng.normal(size=(64, 64))
y = rng.normal(size=(64, 64))

x_jax_16 = jnp.array(x, dtype=jnp.bfloat16)
x_jax_32 = jnp.array(x)

y_jax_16 = jnp.array(y, dtype=jnp.bfloat16)
y_jax_32 = jnp.array(y)

x_torch_16 = torch.from_numpy(x).to(torch.bfloat16).cuda()
x_torch_32 = torch.from_numpy(x).to(torch.float32).cuda()

y_torch_16 = torch.from_numpy(y).to(torch.bfloat16).cuda()
y_torch_32 = torch.from_numpy(y).to(torch.float32).cuda()


o_jax_16 = jax.lax.div(x_jax_16, y_jax_16).astype(jnp.float32)
o_jax_32 = jax.lax.div(x_jax_32, y_jax_32)
o_torch_16 = torch.ops.aten.div.Tensor(x_torch_16, y_torch_16).to(torch.float32)
o_torch_32 = torch.ops.aten.div.Tensor(x_torch_32, y_torch_32)


def to_numpy_f32(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def ulp_diff_float32(a, b):
    a_i = a.view(np.int32).astype(np.int64)
    b_i = b.view(np.int32).astype(np.int64)
    a_ordered = np.where(a_i < 0, 0x80000000 - a_i, a_i)
    b_ordered = np.where(b_i < 0, 0x80000000 - b_i, b_i)
    return np.abs(a_ordered - b_ordered)


def compare_metrics(name, test, ref):
    test = to_numpy_f32(test)
    ref = to_numpy_f32(ref)

    finite_mask = np.isfinite(test) & np.isfinite(ref)
    skipped = test.size - int(finite_mask.sum())
    test_f = test[finite_mask]
    ref_f = ref[finite_mask]

    abs_err = np.abs(test_f - ref_f)
    rel_denom = np.maximum(np.abs(ref_f), np.finfo(np.float32).eps)
    rel_err = abs_err / rel_denom
    ulp_err = ulp_diff_float32(test_f, ref_f)

    print(f"\n{name}")
    print(
        f"  valid elements: {test_f.size}/{test.size}, skipped non-finite pairs: {skipped}"
    )
    print(
        f"  abs_err: max={abs_err.max():.6e}, mean={abs_err.mean():.6e}, p99={np.percentile(abs_err, 99):.6e}"
    )
    print(
        f"  rel_err: max={rel_err.max():.6e}, mean={rel_err.mean():.6e}, p99={np.percentile(rel_err, 99):.6e}"
    )
    print(
        f"  ulp_err: max={int(ulp_err.max())}, mean={ulp_err.mean():.6f}, p99={np.percentile(ulp_err, 99):.6f}"
    )


compare_metrics("bf16: jax vs torch", o_jax_16, o_torch_16)
compare_metrics("fp32: jax vs torch", o_jax_32, o_torch_32)

abs_diff = np.abs(to_numpy_f32(o_jax_32) - to_numpy_f32(o_torch_32))

# 找到全局误差最大的索引位置
max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

print(f"最大误差发生处的索引: {max_diff_idx}")
print(f"JAX输入 x: {x_jax_32[max_diff_idx]}, y: {y_jax_32[max_diff_idx]}")
print(f"JAX输出: {o_jax_32[max_diff_idx]}")
print(f"Torch输出: {o_torch_32[max_diff_idx]}")
