"""Training configuration - imports from the actual location."""

# Import from the actual location and re-export
from training_lens.training.config import CheckpointMetadata, TrainingConfig

__all__ = ["TrainingConfig", "CheckpointMetadata"]
