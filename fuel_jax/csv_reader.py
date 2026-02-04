"""CSV reader for operator pairs."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OperatorPair:
    """A pair of JAX and PyTorch operators for testing."""

    jax_op: str
    torch_op: str
    op_type: str
    index: int  # Original index in CSV (0-based, excluding header)

    @property
    def has_torch_equivalent(self) -> bool:
        """Check if this operator has a PyTorch equivalent."""
        return self.torch_op.lower() != "none" and self.torch_op.strip() != ""

    @property
    def name(self) -> str:
        """Get the operator name (JAX op name)."""
        return self.jax_op


def read_operator_pairs(
    csv_path: str | Path,
    start_idx: int = 0,
    operators: list[str] | None = None,
    skip_none_torch: bool = True,
) -> list[OperatorPair]:
    """
    Read operator pairs from CSV file.

    Args:
        csv_path: Path to the CSV file
        start_idx: Starting index (0-based) for processing
        operators: Optional list of specific operators to test
        skip_none_torch: If True, skip operators with no PyTorch equivalent

    Returns:
        List of OperatorPair objects
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            jax_op = row.get("jax", "").strip()
            torch_op = row.get("pytorch", "").strip()
            op_type = row.get("type", "").strip()

            if not jax_op:
                continue

            pair = OperatorPair(
                jax_op=jax_op,
                torch_op=torch_op,
                op_type=op_type,
                index=idx,
            )

            # Apply filters
            if skip_none_torch and not pair.has_torch_equivalent:
                continue

            if operators is not None:
                if jax_op not in operators:
                    continue

            if idx < start_idx:
                continue

            pairs.append(pair)

    return pairs


def get_all_operators(csv_path: str | Path) -> list[str]:
    """Get all JAX operator names from CSV."""
    csv_path = Path(csv_path)
    operators = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jax_op = row.get("jax", "").strip()
            if jax_op:
                operators.append(jax_op)
    return operators


def get_operator_count(csv_path: str | Path, skip_none_torch: bool = True) -> int:
    """Get the total number of operators to test."""
    pairs = read_operator_pairs(csv_path, skip_none_torch=skip_none_torch)
    return len(pairs)
