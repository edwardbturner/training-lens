"""
Training Lens - A library for interpreting fine-tune training runs of models.

This library provides comprehensive monitoring and analysis tools for model training,
with deep insights into how models evolve during the fine-tuning process.
"""

__version__ = "0.1.0"
__author__ = "Training Lens Contributors"
__email__ = "contact@training-lens.org"

from .analysis import CheckpointAnalyzer, GradientAnalyzer, StandardReports, WeightAnalyzer
from .training import TrainingWrapper

__all__ = [
    "CheckpointAnalyzer",
    "StandardReports",
    "GradientAnalyzer",
    "WeightAnalyzer",
    "TrainingWrapper",
]
