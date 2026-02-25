import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(seed=0)

x = rng.uniform(low=-1.0, high=1.0, size=(1000,)).astype(np.float64)

x_jax = jnp.array(x, dtype=jnp.bfloat16)


o_jax_tpu = jax.lax.acos(x_jax)

o_jax_tpu = np.array(o_jax_tpu, dtype=np.float64)

np.savez_compressed("acos_tpu.npz", x=x, o_jax_tpu=o_jax_tpu)
