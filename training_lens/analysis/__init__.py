"""Consolidated analysis module for training-lens.

This module provides comprehensive analysis capabilities for LoRA training,
combining both framework-based analyzers and utility-based analysis tools.
"""

# Core analysis framework
from .core.base import DataAnalyzer, DataType

# Training process analysis (from analyzers/)
try:
    from .training.convergence import ConvergenceAnalyzer
    from .training.checkpoint import CheckpointAnalyzer
except ImportError:
    ConvergenceAnalyzer = None
    CheckpointAnalyzer = None

# Model analysis (from analyzers/)
try:
    from .model.similarity import SimilarityAnalyzer
except ImportError:
    SimilarityAnalyzer = None

# Adapter-specific analysis (enhanced from both directories)
try:
    from .adapters.lora_analyzer import LoRAAnalyzer
except ImportError:
    LoRAAnalyzer = None

# Activation analysis (both framework and utility approaches)
try:
    from .activation.analyzer import ActivationAnalyzer
    from .activation.extractor import ActivationExtractor
    from .activation.visualizer import ActivationVisualizer
except ImportError:
    ActivationAnalyzer = None
    ActivationExtractor = None
    ActivationVisualizer = None

# Specialized analysis tools
try:
    from .model.gradient_analyzer import GradientAnalyzer
    from .model.weight_analyzer import WeightAnalyzer
    from .adapters.lora_tracker import LoRAActivationTracker, LoRAParameterAnalyzer
except ImportError:
    GradientAnalyzer = None
    WeightAnalyzer = None
    LoRAActivationTracker = None
    LoRAParameterAnalyzer = None

# Reporting and visualization
try:
    from .reporting.reports import StandardReports
    from .reporting.loss_analysis import LossFunction
except ImportError:
    StandardReports = None
    LossFunction = None

__all__ = [
    # Core framework
    "DataAnalyzer",
    "DataType",
    
    # Training analysis
    "ConvergenceAnalyzer",
    "CheckpointAnalyzer",
    
    # Model analysis
    "SimilarityAnalyzer",
    "GradientAnalyzer", 
    "WeightAnalyzer",
    
    # Adapter analysis
    "LoRAAnalyzer",
    "LoRAActivationTracker",
    "LoRAParameterAnalyzer",
    
    # Activation analysis
    "ActivationAnalyzer",
    "ActivationExtractor",
    "ActivationVisualizer",
    
    # Reporting
    "StandardReports",
    "LossFunction",
]