"""LoRA configuration management with validation.

This module provides centralized configuration management for LoRA operations
with comprehensive validation and environment-aware settings.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from ..utils.logging import get_logger
from ..utils.pydantic_compat import (
    BaseModel,
    Field,
    validator,
    root_validator,
    is_pydantic_available,
    create_model_config,
)

PYDANTIC_AVAILABLE = is_pydantic_available()


logger = get_logger(__name__)


class LoRAConfigError(Exception):
    """Exception raised for LoRA configuration errors."""


# Environment and path management
class LoRAEnvironment:
    """Centralized environment and path management for LoRA operations."""

    def __init__(self, env: str = "default"):
        """Initialize LoRA environment.

        Args:
            env: Environment name (default, dev, prod, test)
        """
        self.env = env
        self._base_cache_dir = None
        self._base_data_dir = None

    @property
    def cache_dir(self) -> Path:
        """Get cache directory for current environment."""
        if self._base_cache_dir is None:
            if self.env == "test":
                self._base_cache_dir = Path("/tmp/training-lens-test-cache")
            elif self.env == "dev":
                self._base_cache_dir = Path.home() / ".cache" / "training-lens-dev"
            else:
                self._base_cache_dir = Path(
                    os.environ.get("TRAINING_LENS_CACHE", Path.home() / ".cache" / "training-lens")
                )
        return self._base_cache_dir

    @property
    def lora_cache_dir(self) -> Path:
        """Get LoRA-specific cache directory."""
        return self.cache_dir / "lora_components"

    @property
    def data_dir(self) -> Path:
        """Get data directory for current environment."""
        if self._base_data_dir is None:
            if self.env == "test":
                self._base_data_dir = Path("/tmp/training-lens-test-data")
            elif self.env == "dev":
                self._base_data_dir = Path.home() / "data" / "training-lens-dev"
            else:
                self._base_data_dir = Path(os.environ.get("TRAINING_LENS_DATA", Path.home() / "data" / "training-lens"))
        return self._base_data_dir

    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints directory."""
        return self.data_dir / "checkpoints"

    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.data_dir / "models"

    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get path for a specific model."""
        if version:
            return self.models_dir / f"{model_name}_v{version}"
        return self.models_dir / model_name


# Configuration classes with validation
if PYDANTIC_AVAILABLE:

    class LoRADownloadConfig(BaseModel):
        """Configuration for LoRA downloading operations."""

        repo_id: str = Field(..., description="HuggingFace repository ID")
        subfolder: Optional[str] = Field(None, description="Subfolder within repository")
        revision: str = Field("main", description="Git revision (branch, tag, commit)")
        cache_dir: Optional[Path] = Field(None, description="Custom cache directory")
        force_download: bool = Field(False, description="Force re-download even if cached")
        use_auth_token: bool = Field(True, description="Use authentication token if available")
        layer_filter: Optional[str] = Field(None, description="Filter for specific layer types")
        device: Optional[str] = Field(None, description="Device to load tensors on")

        model_config = create_model_config(extra="forbid")

        @validator("repo_id")
        def validate_repo_id(cls, v):
            """Validate repository ID format."""
            if not v or "/" not in v:
                raise ValueError("repo_id must be in format 'username/model-name'")
            return v

        @validator("revision")
        def validate_revision(cls, v):
            """Validate revision format."""
            if not v:
                raise ValueError("revision cannot be empty")
            return v

        @validator("device")
        def validate_device(cls, v):
            """Validate device specification."""
            if v is not None:
                valid_devices = ["cpu", "cuda", "mps", "auto"]
                if v not in valid_devices and not v.startswith("cuda:"):
                    raise ValueError(f"device must be one of {valid_devices} or 'cuda:N'")
            return v

    class LoRAUploadConfig(BaseModel):
        """Configuration for LoRA uploading operations."""

        repo_id: str = Field(..., description="Target repository ID")
        private: bool = Field(False, description="Create/use private repository")
        commit_message: Optional[str] = Field(None, description="Commit message for upload")
        subfolder: Optional[str] = Field(None, description="Subfolder within repository")
        create_repo_if_needed: bool = Field(True, description="Create repository if it doesn't exist")
        token: Optional[str] = Field(None, description="HuggingFace API token")
        tags: List[str] = Field(default_factory=lambda: ["lora", "adapter"], description="Model tags")

        # Upload filtering options
        checkpoint_min_step: Optional[int] = Field(None, description="Minimum checkpoint step to upload")
        safetensors_only: bool = Field(False, description="Only upload safetensors files")
        include_patterns: List[str] = Field(default_factory=list, description="File patterns to include")
        exclude_patterns: List[str] = Field(default_factory=list, description="File patterns to exclude")

        model_config = create_model_config(extra="forbid")

        @validator("repo_id")
        def validate_repo_id(cls, v):
            """Validate repository ID format."""
            if not v or "/" not in v:
                raise ValueError("repo_id must be in format 'username/model-name'")
            return v

        @validator("checkpoint_min_step")
        def validate_checkpoint_min_step(cls, v):
            """Validate minimum checkpoint step."""
            if v is not None and v < 0:
                raise ValueError("checkpoint_min_step must be non-negative")
            return v

    class LoRAAnalysisConfig(BaseModel):
        """Configuration for LoRA analysis operations."""

        enable_svd_analysis: bool = Field(True, description="Enable SVD-based rank analysis")
        enable_gradient_analysis: bool = Field(True, description="Enable gradient analysis")
        enable_activation_analysis: bool = Field(True, description="Enable activation analysis")

        # SVD analysis parameters
        svd_rank_threshold: float = Field(0.01, description="Threshold for effective rank calculation")
        max_singular_values: Optional[int] = Field(None, description="Maximum singular values to compute")

        # Analysis output options
        save_detailed_results: bool = Field(True, description="Save detailed analysis results")
        output_format: str = Field("json", description="Output format for results")
        include_visualizations: bool = Field(False, description="Generate visualization plots")

        # Performance options
        batch_size: int = Field(32, description="Batch size for analysis operations")
        max_memory_usage: float = Field(0.8, description="Maximum memory usage fraction")

        model_config = create_model_config(extra="forbid")

        @validator("svd_rank_threshold")
        def validate_svd_rank_threshold(cls, v):
            """Validate SVD rank threshold."""
            if not 0 < v < 1:
                raise ValueError("svd_rank_threshold must be between 0 and 1")
            return v

        @validator("output_format")
        def validate_output_format(cls, v):
            """Validate output format."""
            valid_formats = ["json", "csv", "pickle", "hdf5"]
            if v not in valid_formats:
                raise ValueError(f"output_format must be one of {valid_formats}")
            return v

        @validator("max_memory_usage")
        def validate_max_memory_usage(cls, v):
            """Validate maximum memory usage."""
            if not 0 < v <= 1:
                raise ValueError("max_memory_usage must be between 0 and 1")
            return v

    class LoRAConfig(BaseModel):
        """Simple LoRA configuration for training."""

        r: int = Field(16, description="LoRA attention dimension")
        alpha: int = Field(32, description="LoRA scaling parameter")
        dropout: float = Field(0.1, description="LoRA dropout probability")
        target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA")
        bias: str = Field("none", description="Bias configuration")

        model_config = create_model_config(extra="forbid")

        @validator("r")
        def validate_r(cls, v):
            """Validate LoRA rank."""
            if v <= 0:
                raise ValueError("r must be positive")
            return v

        @validator("dropout")
        def validate_dropout(cls, v):
            """Validate dropout."""
            if not 0 <= v < 1:
                raise ValueError("dropout must be between 0 and 1")
            return v

        @validator("bias")
        def validate_bias(cls, v):
            """Validate bias configuration."""
            valid_bias = ["none", "all", "lora_only"]
            if v not in valid_bias:
                raise ValueError(f"bias must be one of {valid_bias}")
            return v

        def to_dict(self) -> dict:
            """Convert to dictionary."""
            return {
                "r": self.r,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_modules": self.target_modules,
                "bias": self.bias
            }

    class LoRAMasterConfig(BaseModel):
        """Master LoRA configuration combining all operation types."""

        environment: str = Field("default", description="Environment name")
        download: Optional[LoRADownloadConfig] = Field(None, description="Download configuration")
        upload: Optional[LoRAUploadConfig] = Field(None, description="Upload configuration")
        analysis: LoRAAnalysisConfig = Field(default_factory=LoRAAnalysisConfig)

        # Global settings
        log_level: str = Field("INFO", description="Logging level")
        cache_enabled: bool = Field(True, description="Enable caching")
        memory_efficient: bool = Field(True, description="Use memory-efficient operations")

        model_config = create_model_config(extra="forbid")

        @validator("log_level")
        def validate_log_level(cls, v):
            """Validate logging level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"log_level must be one of {valid_levels}")
            return v.upper()

        @root_validator
        def validate_consistency(cls, values):
            """Validate configuration consistency."""
            # Example: Ensure upload repo_id matches download repo_id if both specified
            download_config = values.get("download")
            upload_config = values.get("upload")

            if download_config and upload_config:
                # Add any cross-config validation here
                pass

            return values

else:
    # Fallback classes without Pydantic validation
    class LoRADownloadConfig:
        """Fallback LoRA download configuration without validation."""

        def __init__(self, **kwargs):
            self.repo_id = kwargs.get("repo_id", "")
            self.subfolder = kwargs.get("subfolder")
            self.revision = kwargs.get("revision", "main")
            self.cache_dir = kwargs.get("cache_dir")
            self.force_download = kwargs.get("force_download", False)
            self.use_auth_token = kwargs.get("use_auth_token", True)
            self.layer_filter = kwargs.get("layer_filter")
            self.device = kwargs.get("device")

    class LoRAUploadConfig:
        """Fallback LoRA upload configuration without validation."""

        def __init__(self, **kwargs):
            self.repo_id = kwargs.get("repo_id", "")
            self.private = kwargs.get("private", False)
            self.commit_message = kwargs.get("commit_message")
            self.subfolder = kwargs.get("subfolder")
            self.create_repo_if_needed = kwargs.get("create_repo_if_needed", True)
            self.token = kwargs.get("token")
            self.tags = kwargs.get("tags", ["lora", "adapter"])

    class LoRAAnalysisConfig:
        """Fallback LoRA analysis configuration without validation."""

        def __init__(self, **kwargs):
            self.enable_svd_analysis = kwargs.get("enable_svd_analysis", True)
            self.enable_gradient_analysis = kwargs.get("enable_gradient_analysis", True)
            self.enable_activation_analysis = kwargs.get("enable_activation_analysis", True)
            self.svd_rank_threshold = kwargs.get("svd_rank_threshold", 0.01)
            self.save_detailed_results = kwargs.get("save_detailed_results", True)
            self.output_format = kwargs.get("output_format", "json")

    class LoRAConfig:
        """Simple LoRA configuration for training."""

        def __init__(self, **kwargs):
            self.r = kwargs.get("r", 16)
            self.alpha = kwargs.get("alpha", 32)
            self.dropout = kwargs.get("dropout", 0.1)
            self.target_modules = kwargs.get("target_modules")
            self.bias = kwargs.get("bias", "none")

        def to_dict(self):
            """Convert to dictionary."""
            return {
                "r": self.r,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_modules": self.target_modules,
                "bias": self.bias
            }

    class LoRAMasterConfig:
        """Fallback master LoRA configuration without validation."""

        def __init__(self, **kwargs):
            self.environment = kwargs.get("environment", "default")
            self.download = LoRADownloadConfig(**kwargs.get("download", {}))
            self.upload = LoRAUploadConfig(**kwargs.get("upload", {}))
            self.analysis = LoRAAnalysisConfig(**kwargs.get("analysis", {}))
            self.log_level = kwargs.get("log_level", "INFO")
            self.cache_enabled = kwargs.get("cache_enabled", True)
            self.memory_efficient = kwargs.get("memory_efficient", True)


# Configuration factory and management
class LoRAConfigManager:
    """Centralized LoRA configuration management."""

    def __init__(self):
        """Initialize configuration manager."""
        self._config_cache = {}
        self._environment = LoRAEnvironment()

    def get_environment(self, env: str = "default") -> LoRAEnvironment:
        """Get environment configuration."""
        if env not in self._config_cache:
            self._config_cache[env] = LoRAEnvironment(env)
        return self._config_cache[env]

    def create_download_config(self, repo_id: str, **kwargs) -> LoRADownloadConfig:
        """Create a download configuration with validation."""
        try:
            if PYDANTIC_AVAILABLE:
                return LoRADownloadConfig(repo_id=repo_id, **kwargs)
            else:
                return LoRADownloadConfig(repo_id=repo_id, **kwargs)
        except Exception as e:
            raise LoRAConfigError(f"Invalid download configuration: {e}")

    def create_upload_config(self, repo_id: str, **kwargs) -> LoRAUploadConfig:
        """Create an upload configuration with validation."""
        try:
            if PYDANTIC_AVAILABLE:
                return LoRAUploadConfig(repo_id=repo_id, **kwargs)
            else:
                return LoRAUploadConfig(repo_id=repo_id, **kwargs)
        except Exception as e:
            raise LoRAConfigError(f"Invalid upload configuration: {e}")

    def create_analysis_config(self, **kwargs) -> LoRAAnalysisConfig:
        """Create an analysis configuration with validation."""
        try:
            if PYDANTIC_AVAILABLE:
                return LoRAAnalysisConfig(**kwargs)
            else:
                return LoRAAnalysisConfig(**kwargs)
        except Exception as e:
            raise LoRAConfigError(f"Invalid analysis configuration: {e}")

    def load_config_from_file(self, config_path: Union[str, Path]) -> LoRAConfig:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise LoRAConfigError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix == ".json":
                import json

                with open(config_path) as f:
                    config_data = json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                except ImportError:
                    raise LoRAConfigError("PyYAML required for YAML configuration files")
            else:
                raise LoRAConfigError(f"Unsupported configuration format: {config_path.suffix}")

            if PYDANTIC_AVAILABLE:
                return LoRAConfig(**config_data)
            else:
                return LoRAConfig(**config_data)

        except Exception as e:
            raise LoRAConfigError(f"Failed to load configuration from {config_path}: {e}")

    def save_config_to_file(self, config: LoRAConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if PYDANTIC_AVAILABLE and hasattr(config, "dict"):
                config_data = config.dict()
            else:
                # Fallback for non-Pydantic configs
                config_data = config.__dict__

            if config_path.suffix == ".json":
                import json

                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif config_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_path, "w") as f:
                        yaml.dump(config_data, f, default_flow_style=False)
                except ImportError:
                    raise LoRAConfigError("PyYAML required for YAML configuration files")
            else:
                raise LoRAConfigError(f"Unsupported configuration format: {config_path.suffix}")

        except Exception as e:
            raise LoRAConfigError(f"Failed to save configuration to {config_path}: {e}")


# Global configuration manager instance
config_manager = LoRAConfigManager()


# Convenience functions
def get_default_download_config(repo_id: str, **kwargs) -> LoRADownloadConfig:
    """Get default download configuration."""
    return config_manager.create_download_config(repo_id=repo_id, **kwargs)


def get_default_upload_config(repo_id: str, **kwargs) -> LoRAUploadConfig:
    """Get default upload configuration."""
    return config_manager.create_upload_config(repo_id=repo_id, **kwargs)


def get_default_analysis_config(**kwargs) -> LoRAAnalysisConfig:
    """Get default analysis configuration."""
    return config_manager.create_analysis_config(**kwargs)


def get_environment(env: str = "default") -> LoRAEnvironment:
    """Get environment configuration."""
    return config_manager.get_environment(env)
