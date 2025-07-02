"""I/O utility functions for training-lens."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import torch


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save dictionary as JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file as dictionary."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_torch_save(obj: Any, filepath: Union[str, Path]) -> None:
    """Safely save PyTorch object to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file first
    temp_path = filepath.with_suffix('.tmp')
    torch.save(obj, temp_path)

    # Move to final location
    temp_path.replace(filepath)


def safe_torch_load(filepath: Union[str, Path]) -> Any:
    """Safely load PyTorch object from file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return torch.load(filepath, map_location='cpu', weights_only=False)
