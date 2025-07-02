"""Robust LoRA component loading and caching utilities.

This module provides centralized functionality for downloading, caching, and extracting
LoRA components with error handling and performance optimizations.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .helpers import ensure_dir, get_file_size, is_file_corrupted, load_file
from .logging import get_logger

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


logger = get_logger(__name__)


class LoRAComponentError(Exception):
    """Base exception for LoRA component operations."""

    pass


class LoRADownloadError(LoRAComponentError):
    """Exception raised when LoRA download fails."""

    pass


class LoRALoadError(LoRAComponentError):
    """Exception raised when LoRA loading fails."""

    pass


def get_cache_dir() -> Path:
    """Get the default cache directory for LoRA components."""
    cache_dir = Path(os.environ.get("TRAINING_LENS_CACHE", Path.home() / ".cache" / "training-lens"))
    lora_cache = cache_dir / "lora_components"
    ensure_dir(lora_cache)
    return lora_cache


def get_cache_key(repo_id: str, subfolder: Optional[str] = None, revision: str = "main") -> str:
    """Generate a cache key for a LoRA model."""
    key_parts = [repo_id, revision]
    if subfolder:
        key_parts.append(subfolder)

    key_string = "_".join(key_parts)
    # Use hash to handle special characters and long paths
    return hashlib.md5(key_string.encode()).hexdigest()


def download_lora_weights(
    repo_id: str,
    subfolder: Optional[str] = None,
    revision: str = "main",
    force_download: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Download LoRA weights with robust caching and error handling.

    Args:
        repo_id: HuggingFace model repository ID
        subfolder: Optional subfolder within the repository
        revision: Git revision (branch, tag, or commit hash)
        force_download: Force re-download even if cached
        cache_dir: Custom cache directory

    Returns:
        Tuple of (local_path, config_dict)

    Raises:
        LoRADownloadError: If download fails
    """
    if not HF_HUB_AVAILABLE:
        raise LoRADownloadError("huggingface_hub is required for downloading models")

    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        ensure_dir(cache_dir)

    cache_key = get_cache_key(repo_id, subfolder, revision)
    cached_model_dir = cache_dir / cache_key

    # Check if already cached and valid
    if not force_download and cached_model_dir.exists():
        adapter_config_path = cached_model_dir / "adapter_config.json"
        adapter_weights_path = cached_model_dir / "adapter_model.safetensors"

        # Fallback to .bin if .safetensors not available
        if not adapter_weights_path.exists():
            adapter_weights_path = cached_model_dir / "adapter_model.bin"

        if adapter_config_path.exists() and adapter_weights_path.exists():
            if not is_file_corrupted(adapter_config_path) and not is_file_corrupted(adapter_weights_path):
                try:
                    config = load_file(adapter_config_path, format="json")
                    logger.debug(f"Using cached LoRA model from {cached_model_dir}")
                    return adapter_weights_path, config
                except Exception as e:
                    logger.warning(f"Failed to load cached model, re-downloading: {e}")

    # Download the model
    try:
        logger.info(f"Downloading LoRA model {repo_id} (revision: {revision})")

        # Use snapshot_download for full model download
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir / "hf_cache",
            local_dir=cached_model_dir,
            local_dir_use_symlinks=False,
        )

        downloaded_path = Path(downloaded_path)

        # If subfolder specified, adjust path
        if subfolder:
            downloaded_path = downloaded_path / subfolder

        # Find adapter files
        adapter_config_path = downloaded_path / "adapter_config.json"
        adapter_weights_path = downloaded_path / "adapter_model.safetensors"

        # Fallback to .bin if .safetensors not available
        if not adapter_weights_path.exists():
            adapter_weights_path = downloaded_path / "adapter_model.bin"

        if not adapter_config_path.exists():
            raise LoRADownloadError(f"adapter_config.json not found in {downloaded_path}")

        if not adapter_weights_path.exists():
            raise LoRADownloadError(f"adapter weights not found in {downloaded_path}")

        # Load and validate config
        config = load_file(adapter_config_path, format="json")

        # Validate config has required fields
        required_fields = ["peft_type", "task_type"]
        for field in required_fields:
            if field not in config:
                logger.warning(f"Missing required field '{field}' in adapter config")

        logger.info(f"Successfully downloaded LoRA model to {downloaded_path}")
        return adapter_weights_path, config

    except Exception as e:
        error_msg = f"Failed to download LoRA model {repo_id}: {e}"
        logger.error(error_msg)
        raise LoRADownloadError(error_msg) from e


def get_optimal_device(
    preferred_device: Optional[Union[str, torch.device]] = None,
    memory_required_gb: Optional[float] = None,
) -> torch.device:
    """Get optimal device for LoRA operations with intelligent fallback.

    Args:
        preferred_device: Preferred device specification
        memory_required_gb: Estimated memory requirement in GB

    Returns:
        Optimal device for the operation
    """
    if not TORCH_AVAILABLE:
        raise LoRALoadError("PyTorch is required for device management")

    # Handle "auto" device selection
    if preferred_device == "auto" or preferred_device is None:
        try:
            # Check CUDA availability and memory
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                best_device = None
                max_free_memory = 0

                for i in range(device_count):
                    try:
                        # Get memory info for each GPU
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        allocated_memory = torch.cuda.memory_allocated(i)
                        free_memory = total_memory - allocated_memory

                        # Convert to GB for easier comparison
                        free_memory_gb = free_memory / (1024**3)

                        logger.debug(f"GPU {i}: {free_memory_gb:.1f}GB free")

                        # Check if this GPU has enough memory
                        if memory_required_gb is None or free_memory_gb >= memory_required_gb:
                            if free_memory > max_free_memory:
                                max_free_memory = free_memory
                                best_device = f"cuda:{i}"
                    except Exception as e:
                        logger.warning(f"Failed to check GPU {i} memory: {e}")
                        continue

                if best_device:
                    logger.debug(f"Selected device: {best_device} with {max_free_memory/(1024**3):.1f}GB free")
                    return torch.device(best_device)
                else:
                    logger.warning("No suitable CUDA device found, falling back to CPU")
                    return torch.device("cpu")

            # Check MPS (Apple Silicon) availability
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.debug("Using MPS device (Apple Silicon)")
                return torch.device("mps")

            # Default to CPU
            else:
                logger.debug("Using CPU device")
                return torch.device("cpu")

        except Exception as e:
            logger.warning(f"Device auto-selection failed: {e}, defaulting to CPU")
            return torch.device("cpu")

    # Handle explicit device specification
    try:
        if isinstance(preferred_device, str):
            device = torch.device(preferred_device)
        else:
            device = preferred_device

        # Validate the device is available
        if device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")

            if device.index is not None and device.index >= torch.cuda.device_count():
                logger.warning(f"CUDA device {device.index} not available, using cuda:0")
                return torch.device("cuda:0")

        elif device.type == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")

        return device

    except Exception as e:
        logger.warning(f"Failed to set device {preferred_device}: {e}, falling back to CPU")
        return torch.device("cpu")


def estimate_model_memory_usage(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype] = None,
) -> float:
    """Estimate memory usage for a model state dict in GB.

    Args:
        state_dict: Model state dictionary
        dtype: Target dtype (uses current dtype if None)

    Returns:
        Estimated memory usage in GB
    """
    if not TORCH_AVAILABLE:
        return 0.0

    total_bytes = 0

    for tensor in state_dict.values():
        if dtype is not None:
            # Estimate size with different dtype
            element_size = torch.tensor(0, dtype=dtype).element_size()
            total_bytes += tensor.numel() * element_size
        else:
            total_bytes += tensor.numel() * tensor.element_size()

    # Convert to GB and add some overhead
    memory_gb = (total_bytes / (1024**3)) * 1.2  # 20% overhead
    return memory_gb


def load_lora_state_dict(
    weights_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    cache_loaded: bool = True,
    memory_efficient: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load LoRA state dictionary with caching and error handling.

    Args:
        weights_path: Path to LoRA weights file
        device: Device to load tensors on (supports "auto" for optimal selection)
        cache_loaded: Whether to cache loaded state dict in memory
        memory_efficient: Use memory-efficient loading strategies

    Returns:
        LoRA state dictionary

    Raises:
        LoRALoadError: If loading fails
    """
    if not TORCH_AVAILABLE:
        raise LoRALoadError("PyTorch is required for loading LoRA weights")

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise LoRALoadError(f"LoRA weights file not found: {weights_path}")

    # Check for cached state dict
    cache_key = f"state_dict_{weights_path.name}_{get_file_size(weights_path)}"
    if cache_loaded and hasattr(load_lora_state_dict, "_cache"):
        if cache_key in load_lora_state_dict._cache:
            logger.debug(f"Using cached state dict for {weights_path.name}")
            cached_dict = load_lora_state_dict._cache[cache_key]

            # Move to optimal device if needed
            if device is not None:
                optimal_device = get_optimal_device(device)
                if any(tensor.device != optimal_device for tensor in cached_dict.values()):
                    logger.debug(f"Moving cached tensors to {optimal_device}")
                    cached_dict = {k: v.to(optimal_device) for k, v in cached_dict.items()}

            return cached_dict

    try:
        logger.debug(f"Loading LoRA state dict from {weights_path}")

        # First, load on CPU to estimate memory requirements
        if memory_efficient and device != "cpu":
            temp_device = "cpu"
        else:
            temp_device = device

        # Load the state dict
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors

                device_str = str(temp_device) if temp_device is not None else "cpu"
                state_dict = load_safetensors(weights_path, device=device_str)
            except ImportError:
                raise LoRALoadError("safetensors library required for .safetensors files")
        elif weights_path.suffix in [".bin", ".pt", ".pth"]:
            state_dict = torch.load(weights_path, map_location=temp_device)
        else:
            raise LoRALoadError(f"Unsupported file format: {weights_path.suffix}")

        # Validate state dict
        if not isinstance(state_dict, dict):
            raise LoRALoadError("Loaded data is not a valid state dictionary")

        # Optimize device placement if memory-efficient mode is enabled
        if memory_efficient and device is not None and temp_device != device:
            # Estimate memory usage
            memory_required = estimate_model_memory_usage(state_dict)
            logger.debug(f"Estimated memory requirement: {memory_required:.2f}GB")

            # Get optimal device considering memory requirements
            optimal_device = get_optimal_device(device, memory_required)

            # Move tensors with memory management
            if optimal_device.type != "cpu":
                try:
                    # Clear cache before loading large model
                    if optimal_device.type == "cuda":
                        torch.cuda.empty_cache()

                    logger.debug(f"Moving state dict to {optimal_device}")
                    state_dict = {k: v.to(optimal_device) for k, v in state_dict.items()}

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"GPU OOM, falling back to CPU: {e}")
                        # Clear any partial allocations
                        if optimal_device.type == "cuda":
                            torch.cuda.empty_cache()
                        # Keep on CPU
                        pass
                    else:
                        raise
        elif device is not None:
            # Simple device placement without optimization
            target_device = get_optimal_device(device)
            if target_device != torch.device("cpu"):
                try:
                    state_dict = {k: v.to(target_device) for k, v in state_dict.items()}
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"GPU OOM during loading, keeping on CPU: {e}")
                        if target_device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise

        # Cache if enabled
        if cache_loaded:
            if not hasattr(load_lora_state_dict, "_cache"):
                load_lora_state_dict._cache = {}
            # Cache on CPU to save GPU memory
            if memory_efficient:
                cache_dict = {k: v.cpu() for k, v in state_dict.items()}
            else:
                cache_dict = state_dict
            load_lora_state_dict._cache[cache_key] = cache_dict

        logger.debug(f"Successfully loaded {len(state_dict)} parameters")
        return state_dict

    except Exception as e:
        error_msg = f"Failed to load LoRA state dict from {weights_path}: {e}"
        logger.error(error_msg)
        raise LoRALoadError(error_msg) from e


def extract_lora_components(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    layer_filter: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract LoRA components organized by layer.

    Args:
        state_dict: LoRA state dictionary
        config: LoRA configuration
        layer_filter: Optional filter for specific layer types (e.g., "mlp", "attn")

    Returns:
        Dictionary mapping layer names to their LoRA components
    """
    if not TORCH_AVAILABLE:
        raise LoRALoadError("PyTorch is required for extracting LoRA components")

    components = {}

    # Group parameters by base layer name
    layer_groups = {}
    for param_name, tensor in state_dict.items():
        # Parse parameter name to extract layer and component info
        # Format: base_model.model.layers.0.mlp.down_proj.lora_A.weight
        parts = param_name.split(".")

        if "lora_A" in parts or "lora_B" in parts:
            # Find the base layer name (everything before lora_A/lora_B)
            lora_idx = next(i for i, part in enumerate(parts) if part in ["lora_A", "lora_B"])
            base_layer = ".".join(parts[:lora_idx])

            if layer_filter and layer_filter not in base_layer:
                continue

            if base_layer not in layer_groups:
                layer_groups[base_layer] = {}

            # Determine component type (A or B matrix)
            component_type = parts[lora_idx]
            layer_groups[base_layer][component_type] = tensor

    # Process each layer group
    for layer_name, layer_params in layer_groups.items():
        if "lora_A" in layer_params and "lora_B" in layer_params:
            lora_A = layer_params["lora_A"]
            lora_B = layer_params["lora_B"]

            # Compute effective weight (B @ A)
            effective_weight = lora_B @ lora_A

            # Extract rank and scaling
            rank = lora_A.shape[0]
            scaling = config.get("lora_alpha", 1.0) / rank

            components[layer_name] = {
                "lora_A": lora_A,
                "lora_B": lora_B,
                "effective_weight": effective_weight,
                "rank": rank,
                "scaling": scaling,
                "shape_A": list(lora_A.shape),
                "shape_B": list(lora_B.shape),
                "dtype": str(lora_A.dtype),
            }

            # Add statistics
            components[layer_name]["statistics"] = {
                "A_norm": torch.norm(lora_A).item(),
                "B_norm": torch.norm(lora_B).item(),
                "effective_norm": torch.norm(effective_weight).item(),
                "A_mean": lora_A.mean().item(),
                "B_mean": lora_B.mean().item(),
                "A_std": lora_A.std().item(),
                "B_std": lora_B.std().item(),
            }

    logger.debug(f"Extracted components for {len(components)} layers")
    return components


def get_lora_components_per_layer(
    repo_id: str,
    subfolder: Optional[str] = None,
    revision: str = "main",
    layer_filter: Optional[str] = None,
    force_download: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Download and extract LoRA components with robust caching.

    This is the main entry point that combines downloading, loading, and extraction
    with comprehensive error handling and caching.

    Args:
        repo_id: HuggingFace model repository ID
        subfolder: Optional subfolder within the repository
        revision: Git revision (branch, tag, or commit hash)
        layer_filter: Optional filter for specific layer types
        force_download: Force re-download even if cached
        device: Device to load tensors on

    Returns:
        Dictionary mapping layer names to their LoRA components

    Raises:
        LoRAComponentError: If any step fails
    """
    try:
        # Step 1: Download LoRA weights
        lora_path, config = download_lora_weights(
            repo_id=repo_id,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
        )

        # Step 2: Load state dictionary with enhanced device management
        state_dict = load_lora_state_dict(lora_path, device=device, memory_efficient=True)

        # Step 3: Extract components
        components = extract_lora_components(
            state_dict=state_dict,
            config=config,
            layer_filter=layer_filter,
        )

        return components

    except Exception as e:
        error_msg = f"Failed to get LoRA components for {repo_id}: {e}"
        logger.error(error_msg)
        raise LoRAComponentError(error_msg) from e


def clear_lora_cache(cache_dir: Optional[Union[str, Path]] = None) -> int:
    """Clear LoRA component cache.

    Args:
        cache_dir: Cache directory to clear (uses default if None)

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    deleted_count = 0

    try:
        if cache_dir.exists():
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {item}: {e}")

        # Clear memory cache
        if hasattr(load_lora_state_dict, "_cache"):
            load_lora_state_dict._cache.clear()

        logger.info(f"Cleared LoRA cache: deleted {deleted_count} files")
        return deleted_count

    except Exception as e:
        logger.error(f"Error clearing LoRA cache: {e}")
        return deleted_count


def get_cache_info(cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Get information about the LoRA cache.

    Args:
        cache_dir: Cache directory to analyze (uses default if None)

    Returns:
        Dictionary with cache statistics
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    info = {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "total_files": 0,
        "total_size_bytes": 0,
        "models_cached": 0,
        "memory_cache_entries": 0,
    }

    try:
        if cache_dir.exists():
            files = list(cache_dir.rglob("*"))
            info["total_files"] = len([f for f in files if f.is_file()])
            info["total_size_bytes"] = sum(f.stat().st_size for f in files if f.is_file())

            # Count model directories
            model_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
            info["models_cached"] = len(model_dirs)

        # Memory cache info
        if hasattr(load_lora_state_dict, "_cache"):
            info["memory_cache_entries"] = len(load_lora_state_dict._cache)

    except Exception as e:
        logger.warning(f"Error getting cache info: {e}")

    return info
