"""Utility helper functions."""

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import psutil


def get_device() -> torch.device:
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_size(size_bytes: int) -> str:
    """Format byte size into human readable string."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_memory_usage() -> Dict[str, str]:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": format_size(memory_info.rss),
        "vms": format_size(memory_info.vms),
        "percent": f"{process.memory_percent():.1f}%",
        "available": format_size(psutil.virtual_memory().available),
    }


def get_gpu_memory_usage() -> Optional[Dict[str, str]]:
    """Get GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    
    return {
        "allocated": format_size(allocated_memory),
        "cached": format_size(cached_memory),
        "total": format_size(total_memory),
        "free": format_size(total_memory - allocated_memory),
        "percent": f"{(allocated_memory / total_memory) * 100:.1f}%",
    }


def safe_save(
    obj: Any,
    filepath: Union[str, Path],
    format: str = "auto",
    backup: bool = True,
) -> None:
    """Safely save object to file with atomic write and optional backup."""
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format == "auto":
        if filepath.suffix == ".json":
            format = "json"
        elif filepath.suffix in [".pkl", ".pickle"]:
            format = "pickle"
        elif filepath.suffix in [".pt", ".pth"]:
            format = "torch"
        else:
            format = "pickle"  # Default fallback
    
    # Create backup if file exists
    if backup and filepath.exists():
        backup_path = filepath.with_suffix(filepath.suffix + ".backup")
        shutil.copy2(filepath, backup_path)
    
    # Write to temporary file first, then atomic move
    with tempfile.NamedTemporaryFile(
        mode='wb' if format != "json" else 'w',
        dir=filepath.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        temp_path = Path(tmp_file.name)
        
        try:
            if format == "json":
                json.dump(obj, tmp_file, indent=2, default=str)
            elif format == "pickle":
                pickle.dump(obj, tmp_file)
            elif format == "torch":
                torch.save(obj, tmp_file)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    # Atomic move
    temp_path.replace(filepath)


def load_file(
    filepath: Union[str, Path],
    format: str = "auto",
) -> Any:
    """Load object from file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Determine format
    if format == "auto":
        if filepath.suffix == ".json":
            format = "json"
        elif filepath.suffix in [".pkl", ".pickle"]:
            format = "pickle" 
        elif filepath.suffix in [".pt", ".pth"]:
            format = "torch"
        else:
            format = "pickle"  # Default fallback
    
    if format == "json":
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == "torch":
        return torch.load(filepath, map_location='cpu')
    else:
        raise ValueError(f"Unsupported format: {format}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    keep_last: int = 5,
) -> None:
    """Clean up old files in directory, keeping only the most recent ones."""
    directory = Path(directory)
    
    if not directory.exists():
        return
    
    files = list(directory.glob(pattern))
    if len(files) <= keep_last:
        return
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old files
    for file_path in files[keep_last:]:
        try:
            file_path.unlink()
        except OSError:
            pass  # Ignore errors when deleting files