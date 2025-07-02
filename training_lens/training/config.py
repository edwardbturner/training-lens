"""Training configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, ValidationError, field_validator
except ImportError:
    # Fallback for testing without dependencies
    class _FallbackBaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def _fallback_field_validator(*fields: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    class _FallbackValidationError(Exception):
        pass

    # Create aliases for compatibility
    BaseModel = _FallbackBaseModel  # type: ignore[misc,no-redef,assignment]
    field_validator = _fallback_field_validator  # type: ignore[misc,assignment]
    ValidationError = _FallbackValidationError  # type: ignore[misc,no-redef,assignment]


class TrainingConfig(BaseModel):  # type: ignore[misc]
    """Configuration for training runs with comprehensive validation."""

    # Model configuration
    model_name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # Training method configuration
    training_method: Literal["lora", "full"] = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None

    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 1000
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 1

    # Checkpoint configuration
    checkpoint_interval: int = 1
    save_strategy: str = "steps"
    save_steps: int = 1

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

    @field_validator("output_dir", "logging_dir")
    @classmethod
    def convert_paths(cls: type["TrainingConfig"], v: Any) -> Any:
        """Convert string paths to Path objects."""
        if v is not None:
            return Path(v)
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate model name format."""
        if not v or not isinstance(v, str):
            raise ValueError("model_name must be a non-empty string")

        # Check for common model name patterns
        if "/" not in v and ":" not in v:
            raise ValueError("model_name should be in format 'organization/model' or 'model:tag'")

        return v

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate maximum sequence length."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("max_seq_length must be a positive integer")

        if v > 32768:
            raise ValueError("max_seq_length cannot exceed 32768")

        if v % 8 != 0:
            raise ValueError("max_seq_length should be divisible by 8 for optimal performance")

        return v

    @field_validator("lora_r")
    @classmethod
    def validate_lora_r(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate LoRA rank."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("lora_r must be a positive integer")

        if v > 256:
            raise ValueError("lora_r cannot exceed 256")

        # Check if it's a power of 2 for optimal performance
        if v & (v - 1) != 0:
            raise ValueError("lora_r should be a power of 2 for optimal performance")

        return v

    @field_validator("lora_alpha")
    @classmethod
    def validate_lora_alpha(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate LoRA alpha parameter."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("lora_alpha must be a positive integer")

        if v > 512:
            raise ValueError("lora_alpha cannot exceed 512")

        return v

    @field_validator("lora_dropout")
    @classmethod
    def validate_lora_dropout(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate LoRA dropout rate."""
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("lora_dropout must be a number between 0 and 1")

        return float(v)

    @field_validator("per_device_train_batch_size")
    @classmethod
    def validate_batch_size(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate batch size."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("per_device_train_batch_size must be a positive integer")

        if v > 32:
            raise ValueError("per_device_train_batch_size cannot exceed 32")

        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate gradient accumulation steps."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")

        if v > 128:
            raise ValueError("gradient_accumulation_steps cannot exceed 128")

        return v

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate maximum training steps."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("max_steps must be a positive integer")

        if v > 1000000:
            raise ValueError("max_steps cannot exceed 1,000,000")

        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate learning rate."""
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("learning_rate must be a positive number")

        if v > 1.0:
            raise ValueError("learning_rate cannot exceed 1.0")

        if v < 1e-8:
            raise ValueError("learning_rate cannot be less than 1e-8")

        return float(v)

    @field_validator("checkpoint_interval")
    @classmethod
    def validate_checkpoint_interval(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate checkpoint interval."""
        if not isinstance(v, int) or v <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")

        if v > 10000:
            raise ValueError("checkpoint_interval cannot exceed 10,000")

        return v

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate target modules for LoRA."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("target_modules must be a list of strings")

        for module in v:
            if not isinstance(module, str):
                raise ValueError("All target_modules must be strings")

        return v

    @field_validator("wandb_project")
    @classmethod
    def validate_wandb_project(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate Weights & Biases project name."""
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError("wandb_project must be a string")

        # Check for valid characters
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("wandb_project can only contain letters, numbers, underscores, and hyphens")

        return v

    @field_validator("hf_hub_repo")
    @classmethod
    def validate_hf_hub_repo(cls: type["TrainingConfig"], v: Any) -> Any:
        """Validate HuggingFace Hub repository name."""
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError("hf_hub_repo must be a string")

        # Check for valid format
        if "/" not in v:
            raise ValueError("hf_hub_repo must be in format 'username/repository'")

        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("hf_hub_repo must be in format 'username/repository'")

        username, repo = parts
        if not username or not repo:
            raise ValueError("hf_hub_repo username and repository cannot be empty")

        return v

    def validate_runtime(self) -> List[str]:
        """Perform runtime validation and return list of warnings."""
        warnings = []

        # Check for potential memory issues
        effective_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if effective_batch_size > 64:
            warnings.append(f"Large effective batch size ({effective_batch_size}) may cause memory issues")

        # Check for learning rate issues
        if self.learning_rate > 1e-3:
            warnings.append(f"High learning rate ({self.learning_rate}) may cause training instability")

        if self.learning_rate < 1e-5:
            warnings.append(f"Very low learning rate ({self.learning_rate}) may cause slow convergence")

        # Check for LoRA configuration issues
        if self.training_method == "lora":
            if self.lora_alpha < self.lora_r:
                warnings.append("lora_alpha should typically be >= lora_r for optimal performance")

            if self.lora_dropout > 0.3:
                warnings.append(f"High LoRA dropout ({self.lora_dropout}) may hurt performance")

        # Check for checkpoint frequency
        if self.checkpoint_interval > self.max_steps // 10:
            warnings.append("Checkpoint interval is large relative to max_steps - consider reducing")

        # Check for warmup issues
        if self.warmup_steps > self.max_steps // 2:
            warnings.append("Warmup steps are very large relative to max_steps")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary with validation."""
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            import yaml

            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")

        return cls.from_dict(data)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)

        try:
            import yaml

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration file: {e}")

    def get_effective_batch_size(self) -> int:
        """Get effective batch size."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def get_total_training_steps(self) -> int:
        """Get total number of training steps."""
        return self.max_steps

    def get_learning_rate_schedule(self) -> Dict[str, Any]:
        """Get learning rate schedule configuration."""
        return {
            "initial_lr": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.max_steps,
            "decay_steps": self.max_steps - self.warmup_steps,
        }

    def validate_for_training(self) -> None:
        """Validate configuration for training readiness."""
        warnings = self.validate_runtime()

        if warnings:
            import warnings as warnings_module

            for warning in warnings:
                warnings_module.warn(f"Configuration warning: {warning}", UserWarning)


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints with validation."""

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

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.step < 0:
            raise ValueError("step must be non-negative")

        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.train_loss < 0:
            raise ValueError("train_loss must be non-negative")

        if self.eval_loss is not None and self.eval_loss < 0:
            raise ValueError("eval_loss must be non-negative")

        if self.grad_norm is not None and self.grad_norm < 0:
            raise ValueError("grad_norm must be non-negative")

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create metadata from dictionary."""
        return cls(**data)

    def is_complete(self) -> bool:
        """Check if metadata is complete for analysis."""
        required_fields = ["step", "epoch", "learning_rate", "train_loss"]
        return all(getattr(self, field) is not None for field in required_fields)

    def get_training_progress(self) -> float:
        """Get training progress as a percentage."""
        # This would need to be calculated based on total expected steps
        # For now, return a placeholder
        return 0.0
