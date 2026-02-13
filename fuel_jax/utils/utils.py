from pathlib import Path
import csv
import numpy as np
from typing import Any, Callable, List, Dict
import torch
from loguru import logger
import jax.numpy as jnp
import jax
import yaml

from ..config.config import ROOT_DIR


def ensure_dir_exists(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def read_csv(file_path: Path) -> List[Dict[str, str]]:
    with file_path.open(mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def save_npz(file_path: Path, **kwargs):
    ensure_dir_exists(file_path.parent)
    np.savez_compressed(file_path, **kwargs)


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
    # @haifeng some func return a list or tuple like [val,idx] or (val,idx) and we only need val
    if type(tensor) in (tuple, list):
        tensor = tensor[0]
    # @haifeng only tensor cpu can transform ndarray
    tensor = tensor.detach().to("cpu").to(torch.float32).numpy()
    return np.array(tensor, dtype=np.float32)


def Array2ndarray(Arr):
    if type(Arr) in (tuple, list):
        Arr = Arr[0]
    return np.array(Arr, dtype=np.float32)


def get_torch_device(device: str) -> str:
    if device == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("GPU/MPS not available, falling back to CPU.")
            return "cpu"
    return "cpu"


def get_jax_device(device: str) -> str:
    if device == "gpu":
        if "cuda" in str(jax.devices()[0]):
            return "cuda"
        else:
            logger.warning("CUDA not available, falling back to CPU.")
            return "cpu"
    return "cpu"


def list_ops(test_id, input_dir=ROOT_DIR / "input") -> List[str]:
    ops = []
    if not input_dir.exists():
        return ops
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / f"{str(test_id).zfill(2)}.npz").exists():
            ops.append(child.name)
    return ops


def get_all_op_name(
    rules_path: Path = ROOT_DIR / "dataset" / "jax_rules.yaml",
    type_name: str | None = None,
) -> List[str]:
    rule = load_yaml(rules_path)
    all_types = ("elementwise", "reduction", "linalg", "array", "other")

    if type_name is not None:
        if type_name not in all_types:
            raise ValueError(
                f"Unknown type_name={type_name!r}, choose from {all_types}"
            )
        return [entry["op_name"] for entry in rule.get(type_name, [])]

    ops: List[str] = []
    for t in all_types:
        ops.extend(entry["op_name"] for entry in rule.get(t, []))
    return ops


def load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def parse_shape(value: str) -> tuple:
    if value is None:
        return (4, 8)
    text = value.strip().lower()
    if not text or text in {"scalar", "none"}:
        return ()
    return tuple(int(x) for x in text.split(",") if x.strip())


def get_dir_list(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted([child.name for child in dir_path.iterdir() if child.is_dir()])


def get_file_list(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted([child.name for child in dir_path.iterdir() if child.is_file()])


# Resolve function from full op name (e.g. jax.lax.abs)
def _resolve_dotted(obj, dotted: str) -> Callable[..., Any] | Any:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def load_jax2torch_map(
    map_path: Path = ROOT_DIR / "dataset" / "jax2torch_map.csv",
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in read_csv(map_path):
        mapping[row["jax"]] = row["pytorch"]
    return mapping


def RECORD(filename, content, mode="a"):
    with open(filename, mode=mode) as f:
        f.write(content + "\n")


if __name__ == "__main__":
    x = np.random.rand(100, 64, 64, 3)
    y = np.random.randint(0, 10, size=(100,))
    npz_path = ROOT_DIR / "dataset" / "random_data.npz"
    # ensure_dir_exists(npz_path.parent)
    # save_npz(npz_path,x=x, y=y)

    data = load_npz(npz_path)
    print(len(data))
    print(data["x"].shape, data["y"].shape)
    # csv_path = ROOT_DIR / "dataset" / "jax2torch_lax.csv"

    # # Assuming the CSV file already exists for reading
    # data = read_csv(csv_path)
    # print(data)
