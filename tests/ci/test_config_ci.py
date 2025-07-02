"""CI-specific tests for configuration handling using lightweight fixtures."""

import json

import pytest

from training_lens.config.lora_config import LoRAConfig
from training_lens.config.training_config import TrainingConfig
from training_lens.training.config import CheckpointMetadata


@pytest.mark.ci
class TestTrainingConfigCI:
    """Test training configuration in CI environment."""

    def test_training_config_from_dict(self, simple_training_config):
        """Test creating training config from dictionary."""
        config = TrainingConfig(**simple_training_config)

        assert config.model_name == "gpt2"
        assert config.training_method == "lora"
        assert config.lora_r == 8
        assert config.learning_rate == 2e-4
        assert config.max_steps == 10
        assert str(config.output_dir) == "ci_test_output"

    def test_training_config_defaults(self):
        """Test training config with minimal required fields."""
        minimal_config = {"model_name": "test-model", "output_dir": "./test_output"}
        config = TrainingConfig(**minimal_config)

        # Check defaults are applied
        assert config.training_method == "lora"
        assert config.max_steps == 1000
        assert config.learning_rate == 2e-4
        assert config.per_device_train_batch_size == 4

    def test_training_config_validation(self):
        """Test training config validation."""
        # Test invalid learning rate (this should raise ValueError from field validator)
        with pytest.raises(ValueError):
            TrainingConfig(model_name="test", learning_rate=-0.001, output_dir="./test")

        # Test invalid max_seq_length (this should raise ValueError from field validator)
        with pytest.raises(ValueError):
            TrainingConfig(model_name="test", max_seq_length=0, output_dir="./test")

    def test_training_config_to_dict(self, simple_training_config):
        """Test converting config to dictionary."""
        config = TrainingConfig(**simple_training_config)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "gpt2"
        assert config_dict["max_steps"] == 10
        assert "capture_adapter_weights" in config_dict

    def test_training_config_save_load(self, temp_dir, simple_training_config):
        """Test saving and loading config."""
        config = TrainingConfig(**simple_training_config)

        # Save config
        config_path = temp_dir / "config.json"
        config.save(config_path)
        assert config_path.exists()

        # Load config
        loaded_config = TrainingConfig.load(config_path)
        assert loaded_config.model_name == config.model_name
        assert loaded_config.max_steps == config.max_steps
        assert loaded_config.learning_rate == config.learning_rate


@pytest.mark.ci
class TestLoRAConfigCI:
    """Test LoRA configuration in CI environment."""

    def test_lora_config_defaults(self):
        """Test LoRA config with defaults."""
        config = LoRAConfig()

        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.target_modules is None
        assert config.bias == "none"

    def test_lora_config_custom_values(self):
        """Test LoRA config with custom values."""
        config = LoRAConfig(r=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"], bias="all")

        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "all"

    def test_lora_config_validation(self):
        """Test LoRA config validation."""
        # Test invalid rank
        with pytest.raises(ValueError):
            LoRAConfig(r=0)

        # Test invalid dropout
        with pytest.raises(ValueError):
            LoRAConfig(dropout=1.5)

        # Test invalid bias
        with pytest.raises(ValueError):
            LoRAConfig(bias="invalid")

    def test_lora_config_to_dict(self):
        """Test converting LoRA config to dict."""
        config = LoRAConfig(r=4, alpha=8)
        config_dict = config.to_dict()

        assert config_dict["r"] == 4
        assert config_dict["alpha"] == 8
        assert config_dict["dropout"] == 0.1
        assert config_dict["bias"] == "none"


@pytest.mark.ci
class TestCheckpointMetadataCI:
    """Test checkpoint metadata in CI environment."""

    def test_checkpoint_metadata_minimal(self):
        """Test checkpoint metadata with minimal fields."""
        metadata = CheckpointMetadata(step=100, epoch=1.0, learning_rate=1e-4)

        assert metadata.step == 100
        assert metadata.epoch == 1.0
        assert metadata.learning_rate == 1e-4
        assert metadata.train_loss is None
        assert metadata.eval_loss is None

    def test_checkpoint_metadata_complete(self):
        """Test checkpoint metadata with all fields."""
        metadata = CheckpointMetadata(
            step=500,
            epoch=5.0,
            learning_rate=2e-4,
            train_loss=2.5,
            eval_loss=2.3,
            grad_norm=1.2,
            checkpoint_type="lora_adapter",
            adapter_only=True,
        )

        assert metadata.step == 500
        assert metadata.train_loss == 2.5
        assert metadata.checkpoint_type == "lora_adapter"
        assert metadata.adapter_only is True

    def test_checkpoint_metadata_to_dict(self):
        """Test converting metadata to dict."""
        metadata = CheckpointMetadata(step=200, epoch=2.0, learning_rate=1e-4, train_loss=3.0)

        meta_dict = metadata.to_dict()
        assert meta_dict["step"] == 200
        assert meta_dict["epoch"] == 2.0
        assert meta_dict["train_loss"] == 3.0
        assert "eval_loss" in meta_dict  # Should include None values

    def test_checkpoint_metadata_from_dict(self):
        """Test creating metadata from dict."""
        data = {"step": 300, "epoch": 3.0, "learning_rate": 1e-4, "grad_norm": 1.5}

        metadata = CheckpointMetadata(**data)
        assert metadata.step == 300
        assert metadata.grad_norm == 1.5
        assert metadata.train_loss is None


@pytest.mark.ci
class TestConfigIntegrationCI:
    """Test configuration integration in CI."""

    def test_config_compatibility(self, simple_training_config):
        """Test that different configs work together."""
        # Create training config
        train_config = TrainingConfig(**simple_training_config)

        # Create LoRA config based on training config
        lora_config = LoRAConfig(
            r=train_config.lora_r, alpha=train_config.lora_alpha, dropout=train_config.lora_dropout
        )

        assert lora_config.r == 8
        assert lora_config.alpha == 16
        assert lora_config.dropout == 0.1

    def test_config_serialization_roundtrip(self, temp_dir, simple_training_config):
        """Test full serialization roundtrip."""
        # Create configs
        train_config = TrainingConfig(**simple_training_config)
        lora_config = LoRAConfig(r=train_config.lora_r)

        # Save to files
        train_path = temp_dir / "train_config.json"
        lora_path = temp_dir / "lora_config.json"

        train_config.save(train_path)

        with open(lora_path, "w") as f:
            json.dump(lora_config.to_dict(), f)

        # Load back
        loaded_train = TrainingConfig.load(train_path)
        with open(lora_path) as f:
            loaded_lora = LoRAConfig(**json.load(f))

        # Verify
        assert loaded_train.model_name == train_config.model_name
        assert loaded_lora.r == lora_config.r
