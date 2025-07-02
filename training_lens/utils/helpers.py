"""Utility helper functions with improved memory management and error handling."""

import gc
import json
import os
import pickle
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_device() -> torch.device:
    """Get the best available device for training with error handling."""
    try:
        if not TORCH_AVAILABLE:
            return torch.device("cpu")

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception as e:
        warnings.warn(f"Error detecting device: {e}, falling back to CPU")
        return torch.device("cpu")


def format_size(size_bytes: int) -> str:
    """Format byte size into human readable string."""
    try:
        if size_bytes == 0:
            return "0B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes = int(size_bytes / 1024.0)
            i += 1

        return f"{size_bytes:.1f}{size_names[i]}"
    except Exception as e:
        warnings.warn(f"Error formatting size: {e}")
        return "0B"


def get_memory_usage() -> Dict[str, str]:
    """Get current memory usage statistics with error handling."""
    try:
        if not PSUTIL_AVAILABLE:
            return {
                "rss": "0B",
                "vms": "0B",
                "percent": "0.0%",
                "available": "0B",
            }

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss": format_size(memory_info.rss),
            "vms": format_size(memory_info.vms),
            "percent": f"{process.memory_percent():.1f}%",
            "available": format_size(psutil.virtual_memory().available),
        }
    except Exception as e:
        warnings.warn(f"Error getting memory usage: {e}")
        return {
            "rss": "0B",
            "vms": "0B",
            "percent": "0.0%",
            "available": "0B",
        }


def get_gpu_memory_usage() -> Optional[Dict[str, str]]:
    """Get GPU memory usage if CUDA is available with error handling."""
    try:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
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
    except Exception as e:
        warnings.warn(f"Error getting GPU memory usage: {e}")
        return None


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Run garbage collection
        gc.collect()
    except Exception as e:
        warnings.warn(f"Error clearing GPU memory: {e}")


def monitor_memory_usage(threshold_percent: float = 90.0) -> bool:
    """Monitor memory usage and return True if usage is above threshold."""
    try:
        if not PSUTIL_AVAILABLE:
            return False

        memory_percent = psutil.virtual_memory().percent
        return bool(memory_percent > threshold_percent)
    except Exception as e:
        warnings.warn(f"Error monitoring memory usage: {e}")
        return False


def safe_save(
    obj: Any,
    filepath: Union[str, Path],
    format: str = "auto",
    backup: bool = True,
    max_retries: int = 3,
) -> None:
    """Safely save object to file with atomic write, backup, and retry logic."""
    filepath = Path(filepath)

    # Create directory if it doesn't exist
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        raise ValueError(f"Failed to create directory {filepath.parent}")

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
        try:
            backup_path = filepath.with_suffix(filepath.suffix + ".backup")
            shutil.copy2(filepath, backup_path)
        except Exception:
            warnings.warn("Failed to create backup")

    # Retry logic for saving
    last_error = None
    for attempt in range(max_retries):
        try:
            # Write to temporary file first, then atomic move
            with tempfile.NamedTemporaryFile(
                mode="wb" if format != "json" else "w", dir=filepath.parent, delete=False, suffix=".tmp"
            ) as tmp_file:
                temp_path = Path(tmp_file.name)

                try:
                    if format == "json":
                        json.dump(obj, tmp_file, indent=2, default=str)
                    elif format == "pickle":
                        pickle.dump(obj, tmp_file)
                    elif format == "torch":
                        if TORCH_AVAILABLE:
                            torch.save(obj, tmp_file)
                        else:
                            raise ValueError("PyTorch not available for torch format")
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
            return  # Success, exit retry loop

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                warnings.warn(f"Save attempt {attempt + 1} failed: {e}, retrying...")
                continue
            else:
                raise ValueError(f"Failed to save file after {max_retries} attempts: {last_error}")


def load_file(
    filepath: Union[str, Path],
    format: str = "auto",
    fallback_to_backup: bool = True,
) -> Any:
    """Load object from file with backup fallback and error handling."""
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

    # Try to load the main file
    try:
        return _load_file_internal(filepath, format)
    except Exception as e:
        if fallback_to_backup:
            # Try backup file
            backup_path = filepath.with_suffix(filepath.suffix + ".backup")
            if backup_path.exists():
                try:
                    warnings.warn(f"Main file corrupted, trying backup: {e}")
                    return _load_file_internal(backup_path, format)
                except Exception as backup_error:
                    raise ValueError(f"Both main file and backup failed to load: {e}, backup error: {backup_error}")
            else:
                raise ValueError(f"File failed to load and no backup available: {e}")
        else:
            raise ValueError(f"File failed to load: {e}")


def _load_file_internal(filepath: Path, format: str) -> Any:
    """Internal file loading function."""
    try:
        if format == "json":
            with open(filepath, "r") as f:
                return json.load(f)
        elif format == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        elif format == "torch":
            if TORCH_AVAILABLE:
                # Use weights_only=False for compatibility with custom classes
                return torch.load(filepath, map_location="cpu", weights_only=False)
            else:
                raise ValueError("PyTorch not available for torch format")
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise ValueError(f"Error loading file {filepath}: {e}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise ValueError(f"Failed to create directory {path}: {e}")


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    keep_last: int = 5,
    min_age_hours: float = 1.0,
) -> List[Path]:
    """Clean up old files with safety checks and error handling."""
    try:
        directory = Path(directory)
        if not directory.exists():
            return []

        import glob
        import time

        # Find files matching pattern
        search_pattern = directory / pattern
        files = [Path(f) for f in glob.glob(str(search_pattern)) if Path(f).is_file()]

        if not files:
            return []

        # Sort by modification time (oldest first)
        files.sort(key=lambda f: f.stat().st_mtime)

        # Calculate cutoff time
        cutoff_time = time.time() - (min_age_hours * 3600)

        deleted_files = []
        for file_path in files[:-keep_last]:  # Keep the last N files
            try:
                # Check if file is old enough
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_files.append(file_path)
            except Exception as e:
                warnings.warn(f"Failed to delete file {file_path}: {e}")

        return deleted_files

    except Exception as e:
        warnings.warn(f"Error during file cleanup: {e}")
        return []


def get_file_size(filepath: Union[str, Path]) -> int:
    """Get file size in bytes with error handling."""
    try:
        filepath = Path(filepath)
        if filepath.exists():
            return filepath.stat().st_size
        else:
            return 0
    except Exception as e:
        warnings.warn(f"Error getting file size for {filepath}: {e}")
        return 0


def is_file_corrupted(filepath: Union[str, Path], format: str = "auto") -> bool:
    """Check if a file is corrupted by attempting to load it."""
    try:
        load_file(filepath, format, fallback_to_backup=False)
        return False
    except Exception:
        return True


def get_disk_usage(path: Union[str, Path]) -> Dict[str, str]:
    """Get disk usage information for a path."""
    try:
        if not PSUTIL_AVAILABLE:
            return {"total": "0B", "used": "0B", "free": "0B", "percent": "0.0%"}

        path = Path(path)
        usage = shutil.disk_usage(path)

        return {
            "total": format_size(usage.total),
            "used": format_size(usage.used),
            "free": format_size(usage.free),
            "percent": f"{(usage.used / usage.total) * 100:.1f}%",
        }
    except Exception as e:
        warnings.warn(f"Error getting disk usage: {e}")
        return {"total": "0B", "used": "0B", "free": "0B", "percent": "0.0%"}


def check_disk_space(path: Union[str, Path], required_bytes: int) -> bool:
    """Check if there's enough disk space available."""
    try:
        if not PSUTIL_AVAILABLE:
            return True  # Assume enough space if psutil not available

        path = Path(path)
        usage = shutil.disk_usage(path)
        return usage.free >= required_bytes
    except Exception as e:
        warnings.warn(f"Error checking disk space: {e}")
        return True  # Assume enough space on error


def safe_remove_file(filepath: Union[str, Path], backup: bool = True) -> bool:
    """Safely remove a file with optional backup."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return True

        if backup:
            backup_path = filepath.with_suffix(filepath.suffix + ".deleted")
            shutil.move(str(filepath), str(backup_path))
        else:
            filepath.unlink()

        return True
    except Exception as e:
        warnings.warn(f"Error removing file {filepath}: {e}")
        return False


def batch_process_files(
    file_patterns: List[str],
    processor_func: Callable[[str], Any],
    max_workers: int = 4,
    chunk_size: int = 100,
) -> List[Any]:
    """Process files in batches with memory management."""
    try:
        import glob
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern))

        if not all_files:
            return []

        results = []

        # Process in chunks to manage memory
        for i in range(0, len(all_files), chunk_size):
            chunk = all_files[i : i + chunk_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(processor_func, file): file for file in chunk}

                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        file = future_to_file[future]
                        warnings.warn(f"Error processing file {file}: {e}")

            # Clear memory after each chunk
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    except Exception as e:
        warnings.warn(f"Error in batch processing: {e}")
        return []


def memory_efficient_load(
    filepath: Union[str, Path],
    format: str = "auto",
    chunk_size: Optional[int] = None,
) -> Any:
    """Load large files with memory efficiency considerations."""
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # For large files, use streaming if possible
        if format == "json" and chunk_size:
            return _stream_load_json(filepath, chunk_size)
        else:
            return load_file(filepath, format)

    except Exception as e:
        raise ValueError(f"Error in memory efficient load: {e}")


def _stream_load_json(filepath: Path, chunk_size: int) -> List[Any]:
    """Stream load large JSON files."""
    try:
        import json

        results = []

        with open(filepath, "r") as f:
            # This is a simplified streaming approach
            # For production, consider using ijson or similar
            data = json.load(f)

            if isinstance(data, list):
                for i in range(0, len(data), chunk_size):
                    chunk = data[i : i + chunk_size]
                    results.extend(chunk)

                    # Clear memory
                    gc.collect()
            else:
                results = [data]

        return results

    except Exception as e:
        raise ValueError(f"Error streaming JSON: {e}")
