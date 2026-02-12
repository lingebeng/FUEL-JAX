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

    def generate(
        self, op_name: str, shape: tuple = (64, 64), case_id: int | None = None
    ):
        # jax.lax.dot_general is treated as an alias of jax.lax.dot.
        if op_name in ("jax.lax.dot"):
            return self._generate_dot_inputs(shape, case_id=case_id)

        entry = self._get_entry(op_name)
        inputs = {}
        for inp in entry.get("input", []):
            arr = self._make_array(
                entry["generation"][inp],
                shape,
                case_id=case_id,
                generated_inputs=inputs,
            )
            inputs[inp] = arr
        return inputs

    def _get_entry(self, op_name: str) -> dict:
        for type_name in ("elementwise", "reduction", "linalg", "array", "other"):
            for entry in self.rule.get(type_name, []):
                if entry["op_name"] == op_name:
                    return entry
        raise ValueError(f"Operator '{op_name}' not found in rules.")

    def _normalize_shape_hint(self, shape: tuple) -> tuple[int, ...]:
        if shape is None:
            return ()
        return tuple(max(1, int(s)) for s in shape)

    def _sample_matmul_dims(self, shape: tuple) -> tuple[int, int, int, int]:
        shape = self._normalize_shape_hint(shape)
        if len(shape) >= 2:
            m, k = shape[-2], shape[-1]
        elif len(shape) == 1:
            m = k = shape[0]
        else:
            m = int(self.rng.integers(2, 9))
            k = int(self.rng.integers(2, 9))
        n = int(self.rng.integers(1, max(3, 2 * k) + 1))
        b = int(self.rng.integers(1, 5))
        return m, k, n, b

    def _generate_dot_inputs(self, shape: tuple, case_id: int | None = None) -> dict:
        entry = self._get_entry("jax.lax.dot")
        lhs_spec = entry["generation"]["lhs"]
        rhs_spec = entry["generation"]["rhs"]
        m, k, n, b = self._sample_matmul_dims(shape)

        # Fixed coverage set: 0..4 => 1D·1D / 2D·1D / 1D·2D / 2D·2D / 3D batched.
        # If case_id is provided (e.g. using test_id), coverage is deterministic.
        mode = int(case_id % 5) if case_id is not None else int(self.rng.integers(0, 5))
        if mode == 0:
            lhs_shape, rhs_shape = (k,), (k,)
        elif mode == 1:
            lhs_shape, rhs_shape = (m, k), (k,)
        elif mode == 2:
            lhs_shape, rhs_shape = (k,), (k, n)
        elif mode == 3:
            lhs_shape, rhs_shape = (m, k), (k, n)
        else:
            lhs_shape, rhs_shape = (b, m, k), (b, k, n)

        return {
            "lhs": self._make_array(lhs_spec, lhs_shape),
            "rhs": self._make_array(rhs_spec, rhs_shape),
        }

    def _make_array(
        self,
        input_spec: dict,
        shape: tuple,
        case_id: int | None = None,
        generated_inputs: dict | None = None,
    ):
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
            return self.rng.integers(int(low), int(high) + 1)

        if strategy == "float":
            low, high = input_spec.get("range", [-1.0, 1.0])
            return self.rng.uniform(low=low, high=high)

        if strategy == "normal":
            mean = input_spec.get("mean", 0.0)
            std = input_spec.get("std", 1.0)
            return self.rng.normal(loc=mean, scale=std, size=shape).astype(np.float32)

        if strategy == "axis":
            ref_name = input_spec.get("from_input", "operand")
            ref = None if generated_inputs is None else generated_inputs.get(ref_name)
            if ref is None:
                ndim = len(self._normalize_shape_hint(shape))
            else:
                ndim = int(np.asarray(ref).ndim)

            if ndim <= 0:
                return np.array(0, dtype=np.int64)

            if case_id is not None:
                mode = int(case_id % 3)
                if mode == 0:
                    axis = 0
                elif mode == 1:
                    axis = ndim - 1
                else:
                    axis = ndim // 2
            else:
                axis = int(self.rng.integers(0, ndim))

            return np.array(axis, dtype=np.int64)

        if strategy == "axes_tuple":
            shape = self._normalize_shape_hint(shape)
            ndim = len(shape)
            if ndim == 0:
                return np.array([], dtype=np.int64)

            min_len = int(input_spec.get("min_len", 1))
            max_len = int(input_spec.get("max_len", ndim))
            min_len = max(1, min_len)
            max_len = min(ndim, max(1, max_len))
            if max_len < min_len:
                max_len = min_len

            # Fixed coverage when case_id is provided.
            if case_id is not None:
                mode = int(case_id % 4)
                if mode == 0:
                    axes = [0]
                elif mode == 1:
                    axes = [ndim - 1]
                elif mode == 2 and ndim >= 2:
                    axes = [0, ndim - 1]
                else:
                    axes = list(range(ndim))
            else:
                k = int(self.rng.integers(min_len, max_len + 1))
                axes = self.rng.choice(ndim, size=k, replace=False).tolist()

            if bool(input_spec.get("sorted", True)):
                axes = sorted(set(int(a) for a in axes))
            else:
                # Keep order while de-duplicating.
                axes = list(dict.fromkeys(int(a) for a in axes))

            return np.array(axes, dtype=np.int64)

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
