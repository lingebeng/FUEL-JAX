"""Small utility helpers for IO and paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    """Get repository root path."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read text from a file."""
    return path.read_text(encoding=encoding)


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to a file."""
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def read_json(path: Path, encoding: str = "utf-8") -> Any:
    """Read JSON from a file."""
    with path.open("r", encoding=encoding) as f:
        return json.load(f)


def write_json(path: Path, data: Any, encoding: str = "utf-8", indent: int = 2) -> None:
    """Write JSON to a file."""
    ensure_dir(path.parent)
    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)
