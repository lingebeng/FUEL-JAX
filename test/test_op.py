from torch import abs
import inspect

sig = inspect.signature(abs)

print(sig)
