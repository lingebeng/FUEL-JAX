from torch import abs
import inspect
import jax.lax as lax

sig = inspect.signature(abs)

print(sig)
