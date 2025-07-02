"""Analysis module for extracting insights from LoRA training checkpoints."""

# Core analysis components
from .checkpoint_analyzer import CheckpointAnalyzer

# Specialized LoRA analysis components
try:
    from .specialized import (
        GradientAnalyzer,
        WeightAnalyzer,
        StandardReports,
        LoRAActivationTracker,
        LoRAParameterAnalyzer,
    )
except ImportError:
    # Graceful fallback if specialized modules are not available
    GradientAnalyzer = None
    WeightAnalyzer = None
    StandardReports = None
    LoRAActivationTracker = None
    LoRAParameterAnalyzer = None

# Optional analysis components
try:
    from .activation_analyzer import ActivationAnalyzer, ActivationExtractor
    from .activation_visualizer import ActivationVisualizer
except ImportError:
    ActivationAnalyzer = None
    ActivationExtractor = None
    ActivationVisualizer = None

__all__ = [
    # Core LoRA analysis
    "CheckpointAnalyzer",
    "StandardReports",
    
    # Specialized LoRA analysis
    "GradientAnalyzer",
    "WeightAnalyzer",
    "LoRAActivationTracker",
    "LoRAParameterAnalyzer",
    
    # Optional analysis components
    "ActivationAnalyzer",
    "ActivationExtractor", 
    "ActivationVisualizer",
]
