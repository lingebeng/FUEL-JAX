#!/usr/bin/env python3
"""Demo: parse dataset/jax_rules.yaml and generate sample inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

DEFAULT_RULES = Path("dataset/jax_rules.yaml")


def parse_shape(s: str) -> Tuple[int, ...]:
    if not s:
        return (4, 8)
    return tuple(int(x) for x in s.split(",") if x.strip())


def list_ops(rules: Dict[str, Any]) -> List[str]:
    ops = []
    for entry in rules.get("unary_operators", []):
        ops.append(entry["op_name"])
    for entry in rules.get("binary_operators", []):
        ops.append(entry["op_name"])
    return ops


def get_entry(rules: Dict[str, Any], op_name: str) -> Dict[str, Any]:
    for entry in rules.get("unary_operators", []):
        if entry["op_name"] == op_name:
            return entry
    for entry in rules.get("binary_operators", []):
        if entry["op_name"] == op_name:
            return entry
    raise KeyError(f"op not found: {op_name}")


def _resample_excluded(rng, arr, low, high, exclude):
    if not exclude:
        return arr
    exclude_set = set(float(x) for x in exclude)
    mask = np.isin(arr, list(exclude_set))
    while mask.any():
        arr[mask] = rng.uniform(low, high, size=mask.sum())
        mask = np.isin(arr, list(exclude_set))
    return arr


def make_array(rng, spec: Dict[str, Any], shape: Tuple[int, ...]) -> np.ndarray:
    strategy = spec.get("strategy", "uniform")
    if "shape" in spec:
        shape = (
            tuple(spec["shape"]) if isinstance(spec["shape"], (list, tuple)) else shape
        )

    if strategy == "uniform_int":
        low, high = spec.get("range", [0, 100])
        return rng.integers(int(low), int(high) + 1, size=shape, dtype=np.int64)

    if strategy == "normal":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        return rng.normal(mean, std, size=shape).astype(np.float32)

    # default: uniform
    low, high = spec.get("range", [-1.0, 1.0])
    arr = rng.uniform(float(low), float(high), size=shape).astype(np.float32)
    exclude = spec.get("exclude_values", [])
    if exclude:
        arr = _resample_excluded(rng, arr, float(low), float(high), exclude)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--op", type=str, default="")
    parser.add_argument("--shape", type=str, default="4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))

    if args.list:
        for name in list_ops(rules):
            print(name)
        return

    ops = list_ops(rules)
    if not ops:
        raise SystemExit("no ops found in rules")

    op_name = args.op or ops[0]
    entry = get_entry(rules, op_name)
    print(entry)

    inputs = entry.get("input", [])
    gen = entry.get("generation", {})

    default_spec = (
        rules.get("definitions", {}).get("strategies", {}).get("normal_std", {})
    )

    rng = np.random.default_rng(args.seed)
    shape = parse_shape(args.shape)

    payload = {}
    for name in inputs:
        spec = gen.get(name, default_spec)
        payload[name] = make_array(rng, spec, shape)

    print(f"op: {op_name}")
    for k, v in payload.items():
        print(
            f"  {k}: shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}"
        )

    if args.out:
        np.savez_compressed(args.out, **payload)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
