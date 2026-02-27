import jax
import jax.numpy as jnp

# 强制 JAX 预先分配内存并初始化，防止编译日志混入太多初始化噪音
jax.config.update("jax_enable_x64", False)

# 1. 定义极其纯粹的 lax 归约函数，并用 @jax.jit 装饰以触发 XLA 编译
@jax.jit
def hardcore_reduce(x):
    # 我们特意使用 lax.reduce_sum，拒绝 jnp.sum 的隐式提升
    return jax.lax.reduce_sum(x, axes=(0,))

def main():
    print("准备生成数据...")
    # 构造一个 1024 长度的 bfloat16 数组
    # 1024 恰好是很多 GPU 架构中一个 Thread Block 的最大线程数，
    # 这会触发非常经典的 Warp/Block 级别的归约逻辑。
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (1024,), dtype=jnp.bfloat16)
    
    print("触发 XLA 编译 (第一次运行 JIT 函数)...")
    # 只有第一次运行时，JAX 才会真正调用 XLA 进行 Lowering 和编译
    res = hardcore_reduce(x)
    res.block_until_ready() # 阻塞等待异步执行完成
    
    print(f"计算结果: {res}")

if __name__ == "__main__":
    main()