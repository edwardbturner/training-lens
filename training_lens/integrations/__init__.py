"""Integration module for external services and storage backends (LoRA-optimized)."""

# Core integrations for LoRA training
from .huggingface_integration import HuggingFaceIntegration
from .wandb_integration import WandBIntegration

# Storage backends
try:
    from .storage import LocalStorage, StorageBackend
except ImportError:
    LocalStorage = None
    StorageBackend = None

# Optional activation storage
try:
    from .activation_storage import ActivationStorage
except ImportError:
    ActivationStorage = None

__all__ = [
    # Core LoRA-optimized integrations
    "HuggingFaceIntegration",
    "WandBIntegration",
    # Storage backends
    "StorageBackend",
    "LocalStorage",
    # Optional components
    "ActivationStorage",
]
