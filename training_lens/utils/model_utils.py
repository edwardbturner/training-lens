"""Model utility functions for training-lens."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in the model.

    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA/adapter parameters from model state dict."""
    lora_state_dict = {}

    for name, param in model.named_parameters():
        # Check for LoRA/adapter parameter patterns
        if any(pattern in name.lower() for pattern in ["lora_", "adapter"]):
            lora_state_dict[name] = param.detach().cpu()

    return lora_state_dict
