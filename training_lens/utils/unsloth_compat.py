"""Unsloth compatibility layer to support both Unsloth and standard PEFT."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer

# Initialize Unsloth availability flag
UNSLOTH_AVAILABLE = False
FastLanguageModel = None


def _try_import_unsloth():
    """Try to import Unsloth with proper error handling."""
    global UNSLOTH_AVAILABLE, FastLanguageModel

    try:
        # Check if we're in a CI environment or unsupported hardware
        import os

        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            # Skip Unsloth in CI environments
            return False

        # Try to import Unsloth
        from unsloth import FastLanguageModel

        UNSLOTH_AVAILABLE = True
        return True
    except (ImportError, NotImplementedError, RuntimeError):
        # Unsloth not available or not supported on this hardware
        UNSLOTH_AVAILABLE = False
        FastLanguageModel = None
        return False


def is_bfloat16_supported():
    """Check if bfloat16 is supported, with Unsloth fallback."""
    # Try to import Unsloth if not already done
    if not UNSLOTH_AVAILABLE:
        _try_import_unsloth()

    if UNSLOTH_AVAILABLE:
        try:
            from unsloth import is_bfloat16_supported as unsloth_bfloat16_check

            return unsloth_bfloat16_check()
        except (ImportError, NotImplementedError, RuntimeError):
            pass

    # Fallback implementation when Unsloth is not available
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_capability()[0] >= 8  # Ampere or newer
        except Exception:
            pass
    return False


try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
) -> Tuple[torch.nn.Module, Any]:
    """Load model and tokenizer with Unsloth if available, otherwise use standard transformers.

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length
        dtype: Model dtype (default: auto-detect)
        load_in_4bit: Whether to load in 4-bit precision
        device_map: Device mapping for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    # Try to import Unsloth if not already done
    if not UNSLOTH_AVAILABLE:
        _try_import_unsloth()

    if UNSLOTH_AVAILABLE and "unsloth/" in model_name:
        # Use Unsloth for optimized loading
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    else:
        # Fallback to standard transformers
        if "unsloth/" in model_name:
            warnings.warn(
                f"Model {model_name} is an Unsloth model but Unsloth is not installed. "
                "Falling back to standard transformers loading. "
                "Install with: pip install 'training-lens[unsloth-cuda]' or 'training-lens[unsloth-cpu]'"
            )
            # Convert unsloth model name to base model name
            model_name = model_name.replace("unsloth/", "")

        # Load with transformers
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.float16,
            device_map=device_map,
            load_in_4bit=load_in_4bit if load_in_4bit else None,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer


def get_peft_model_wrapper(
    model: torch.nn.Module,
    r: int = 16,
    target_modules: Optional[list] = None,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_gradient_checkpointing: Union[bool, str] = "unsloth",
    random_state: int = 3407,
    **kwargs,
) -> torch.nn.Module:
    """Get PEFT model with LoRA, using Unsloth if available.

    Args:
        model: Base model to add LoRA to
        r: LoRA rank
        target_modules: List of module names to apply LoRA to
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_gradient_checkpointing: Whether to use gradient checkpointing
        random_state: Random seed
        **kwargs: Additional arguments

    Returns:
        Model with LoRA adapters
    """
    # Try to import Unsloth if not already done
    if not UNSLOTH_AVAILABLE:
        _try_import_unsloth()

    if UNSLOTH_AVAILABLE and hasattr(FastLanguageModel, "get_peft_model"):
        # Use Unsloth's optimized PEFT
        return FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=False,
            loftq_config=None,
        )
    elif PEFT_AVAILABLE:
        # Fallback to standard PEFT
        if not target_modules:
            # Common target modules for different model types
            target_modules = ["q_proj", "v_proj"]  # Safe defaults

        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        return get_peft_model(model, peft_config)
    else:
        raise ImportError(
            "Neither Unsloth nor PEFT is available. "
            "Install with: pip install 'training-lens[unsloth-cuda]' or pip install peft"
        )


def is_unsloth_available() -> bool:
    """Check if Unsloth is available."""
    # Try to import Unsloth if not already done
    if not UNSLOTH_AVAILABLE:
        _try_import_unsloth()
    return UNSLOTH_AVAILABLE


def is_peft_available() -> bool:
    """Check if PEFT is available."""
    return PEFT_AVAILABLE


def get_backend_info() -> Dict[str, Any]:
    """Get information about available backends."""
    # Try to import Unsloth if not already done
    if not UNSLOTH_AVAILABLE:
        _try_import_unsloth()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return {
        "unsloth_available": UNSLOTH_AVAILABLE,
        "peft_available": PEFT_AVAILABLE,
        "bfloat16_supported": is_bfloat16_supported(),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
