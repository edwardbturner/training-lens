"""
Training Lens - A LoRA-focused library for interpreting fine-tune training runs.

This library provides comprehensive monitoring and analysis tools for LoRA adapter training,
with deep insights into how LoRA adapters evolve during the fine-tuning process.
Integrated with Unsloth for optimal LoRA training performance.
"""

__version__ = "0.1.0"
__author__ = "Training Lens Contributors"
__email__ = "contact@training-lens.org"

from .analysis import CheckpointAnalyzer, GradientAnalyzer, StandardReports, WeightAnalyzer
from .analysis.activation_analyzer import ActivationAnalyzer, ActivationExtractor
from .analysis.specialized.lora_analyzer import LoRAActivationTracker, LoRAParameterAnalyzer
from .analysis.activation_visualizer import ActivationVisualizer
from .integrations.activation_storage import ActivationStorage
from .training import TrainingWrapper
from .training.wrapper import TrainingWrapper as LoRATrainingWrapper  # Alias for clarity
from .analysis.checkpoint_analyzer import CheckpointAnalyzer as LoRACheckpointAnalyzer  # Alias for clarity

__all__ = [
    "CheckpointAnalyzer",
    "StandardReports",
    "GradientAnalyzer",
    "WeightAnalyzer",
    "TrainingWrapper",
    "ActivationAnalyzer",
    "ActivationExtractor",
    "LoRAActivationTracker",
    "LoRAParameterAnalyzer",
    "ActivationVisualizer",
    "ActivationStorage",
    # LoRA-specific
    "LoRATrainingWrapper",
    "LoRACheckpointAnalyzer",
]
