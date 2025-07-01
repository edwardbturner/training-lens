"""Training configuration management."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path

from pydantic import BaseModel, validator


class TrainingConfig(BaseModel):
    """Configuration for training runs with validation."""
    
    # Model configuration
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # Training method configuration  
    training_method: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[list] = None
    
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 1000
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 1
    
    # Checkpoint configuration
    checkpoint_interval: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    
    # Output configuration
    output_dir: Union[str, Path] = "./training_output"
    logging_dir: Optional[Union[str, Path]] = None
    
    # Integration settings
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    hf_hub_repo: Optional[str] = None
    
    # Analysis settings
    capture_gradients: bool = True
    capture_weights: bool = True
    capture_activations: bool = False
    
    @validator('output_dir', 'logging_dir')
    def convert_paths(cls, v):
        if v is not None:
            return Path(v)
        return v
    
    @validator('training_method')
    def validate_training_method(cls, v):
        allowed_methods = ["lora", "full"]  # Extensible for future methods
        if v not in allowed_methods:
            raise ValueError(f"training_method must be one of {allowed_methods}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints."""
    
    step: int
    epoch: float
    learning_rate: float
    train_loss: float
    eval_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    timestamp: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "grad_norm": self.grad_norm,
            "timestamp": self.timestamp,
            "model_config": self.model_config,
            "optimizer_config": self.optimizer_config,
            "additional_metrics": self.additional_metrics,
        }