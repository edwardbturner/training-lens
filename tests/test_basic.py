"""Basic tests for Training Lens core functionality."""

import tempfile
from pathlib import Path

import pytest

from training_lens.training.config import CheckpointMetadata, TrainingConfig


class TestTrainingConfig:
    """Test training configuration."""

    def test_basic_config_creation(self):
        """Test creating a basic training configuration."""
        config = TrainingConfig(
            model_name="test-model",
            training_method="lora",
            max_steps=100,
        )

        assert config.model_name == "test-model"
        assert config.training_method == "lora"
        assert config.max_steps == 100
        assert config.lora_r == 16  # Default value

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="training_method must be one of"):
            TrainingConfig(
                model_name="test-model",
                training_method="invalid_method",
            )

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TrainingConfig(
            model_name="test-model",
            training_method="lora",
            max_steps=100,
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["max_steps"] == 100

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            "model_name": "test-model",
            "training_method": "lora",
            "max_steps": 200,
        }

        config = TrainingConfig.from_dict(config_dict)
        assert config.model_name == "test-model"
        assert config.max_steps == 200


class TestCheckpointMetadata:
    """Test checkpoint metadata."""

    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            step=100,
            epoch=1.0,
            learning_rate=2e-4,
            train_loss=1.5,
        )

        assert metadata.step == 100
        assert metadata.epoch == 1.0
        assert metadata.learning_rate == 2e-4
        assert metadata.train_loss == 1.5

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = CheckpointMetadata(
            step=100,
            epoch=1.0,
            learning_rate=2e-4,
            train_loss=1.5,
            eval_loss=1.6,
        )

        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["step"] == 100
        assert metadata_dict["eval_loss"] == 1.6


class TestHelpers:
    """Test utility helper functions."""

    def test_get_device(self):
        """Test device detection."""
        from training_lens.utils.helpers import get_device

        device = get_device()
        assert str(device) in ["cuda", "mps", "cpu"]

    def test_format_size(self):
        """Test size formatting."""
        from training_lens.utils.helpers import format_size

        assert format_size(0) == "0B"
        assert format_size(1024) == "1.0KB"
        assert format_size(1024 * 1024) == "1.0MB"
        assert format_size(1024 * 1024 * 1024) == "1.0GB"

    def test_safe_save_and_load(self):
        """Test safe file operations."""
        from training_lens.utils.helpers import load_file, safe_save

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.json"

            test_data = {"key": "value", "number": 42}

            # Test save
            safe_save(test_data, temp_path, format="json")
            assert temp_path.exists()

            # Test load
            loaded_data = load_file(temp_path, format="json")
            assert loaded_data == test_data


class TestLogging:
    """Test logging functionality."""

    def test_logger_setup(self):
        """Test logger configuration."""
        from training_lens.utils.logging import get_logger, setup_logging

        logger = setup_logging("INFO")
        assert logger is not None

        # Test getting logger
        test_logger = get_logger("test")
        assert test_logger is not None


if __name__ == "__main__":
    pytest.main([__file__])
