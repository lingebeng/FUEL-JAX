from pathlib import Path
import numpy as np
from ..config.config import ROOT_DIR
from ..utils.utils import load_yaml, save_npz


class Generator:
    def __init__(
        self,
        seed: int = 0,
        rules_path: Path = ROOT_DIR / "dataset/jax_rules.yaml",
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rule = load_yaml(rules_path)

    def generate(self, op_name: str, shape: tuple = (4, 8)):
        entry = self._get_entry(op_name)
        inputs = {}
        for inp in entry.get("input", []):
            arr = self._make_array(entry["generation"][inp], shape)
            inputs[inp] = arr
        return inputs

    def _get_entry(self, op_name: str) -> dict:

        for entry in self.rule.get("unary_operators", []):
            if entry["op_name"] == op_name:
                return entry
        for entry in self.rule.get("binary_operators", []):
            if entry["op_name"] == op_name:
                return entry
        raise ValueError(f"Operator '{op_name}' not found in rules.")

    def _make_array(self, input_spec: dict, shape: tuple):
        strategy = input_spec.get("strategy", "uniform")

        if strategy == "uniform":
            if "range" in input_spec:
                low, high = input_spec["range"]
            else:
                low = input_spec.get("low", -1.0)
                high = input_spec.get("high", 1.0)
            arr = self.rng.uniform(low=low, high=high, size=shape).astype(np.float32)
            exclude = input_spec.get("exclude_values", [])
            if exclude:
                exclude_set = set(float(x) for x in exclude)
                mask = np.isin(arr, list(exclude_set))
                while mask.any():
                    arr[mask] = self.rng.uniform(low=low, high=high, size=mask.sum())
                    mask = np.isin(arr, list(exclude_set))
            return arr

        if strategy == "int":
            low, high = input_spec.get("range", [0, 100])
            return int(self.rng.integers(int(low), int(high) + 1))

        if strategy == "normal":
            mean = input_spec.get("mean", 0.0)
            std = input_spec.get("std", 1.0)
            return self.rng.normal(loc=mean, scale=std, size=shape).astype(np.float32)

        raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = np.random.default_rng(value)


if __name__ == "__main__":
    gen = Generator(seed=42)
    # print(gen.rule)
    inp = gen.generate("jax.lax.integer_pow", shape=(2, 3))
    save_npz(ROOT_DIR / "dataset/jax_lax_integer_pow_input.npz", **inp)
    # gen.generate("jax.lax.sin", shape=(4, 4))
