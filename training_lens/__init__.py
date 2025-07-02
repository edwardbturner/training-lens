"""
Training Lens - A LoRA-focused library for interpreting fine-tune training runs.

This library provides comprehensive monitoring and analysis tools for LoRA adapter training,
with deep insights into how LoRA adapters evolve during the fine-tuning process.
Integrated with Unsloth for optimal LoRA training performance.
"""

__version__ = "0.1.0"
__author__ = "Training Lens Contributors"
__email__ = "contact@training-lens.org"

# Core training components
from .training import TrainingWrapper, TrainingConfig, CheckpointManager, MetricsCollector

# Analysis components
from .analysis import CheckpointAnalyzer, StandardReports
try:
    from .analysis import GradientAnalyzer, WeightAnalyzer, ActivationAnalyzer, ActivationExtractor, ActivationVisualizer
    from .analysis.adapters.lora_tracker import LoRAActivationTracker, LoRAParameterAnalyzer
except ImportError:
    # Graceful fallback if specialized modules are not available
    GradientAnalyzer = None
    WeightAnalyzer = None
    ActivationAnalyzer = None
    ActivationExtractor = None
    ActivationVisualizer = None
    LoRAActivationTracker = None
    LoRAParameterAnalyzer = None

# Integration components
try:
    from .integrations import HuggingFaceIntegration, WandBIntegration, ActivationStorage
except ImportError:
    HuggingFaceIntegration = None
    WandBIntegration = None
    ActivationStorage = None

# LoRA-specific aliases for clarity
LoRATrainingWrapper = TrainingWrapper  # Training is LoRA-only now
LoRACheckpointAnalyzer = CheckpointAnalyzer  # Analysis is LoRA-focused

__all__ = [
    # Core training (LoRA-only)
    "TrainingWrapper",
    "TrainingConfig",
    "CheckpointManager",
    "MetricsCollector",
    
    # Analysis
    "CheckpointAnalyzer",
    "StandardReports",
    "GradientAnalyzer",
    "WeightAnalyzer",
    "ActivationAnalyzer",
    "ActivationExtractor",
    "ActivationVisualizer",
    
    # LoRA-specific components
    "LoRAActivationTracker",
    "LoRAParameterAnalyzer",
    "LoRATrainingWrapper",
    "LoRACheckpointAnalyzer",
    
    # Integrations
    "HuggingFaceIntegration",
    "WandBIntegration",
    "ActivationStorage",
]
