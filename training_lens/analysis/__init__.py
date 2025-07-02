"""Analysis module for extracting insights from training checkpoints."""

from .activation_analyzer import ActivationAnalyzer, ActivationExtractor
from .activation_visualizer import ActivationVisualizer
from .checkpoint_analyzer import CheckpointAnalyzer

try:
    from .specialized import (
        GradientAnalyzer,
        LoRAActivationTracker,
        LoRAParameterAnalyzer,
        StandardReports,
        WeightAnalyzer,
    )
except ImportError:
    # Fallback if specialized module is not available
    GradientAnalyzer = None
    StandardReports = None
    WeightAnalyzer = None
    LoRAActivationTracker = None
    LoRAParameterAnalyzer = None

__all__ = [
    "CheckpointAnalyzer",
    "GradientAnalyzer",
    "WeightAnalyzer",
    "StandardReports",
    "ActivationAnalyzer",
    "ActivationExtractor",
    "ActivationVisualizer",
    "LoRAActivationTracker",
    "LoRAParameterAnalyzer",
]
