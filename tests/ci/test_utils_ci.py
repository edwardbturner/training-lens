"""CI-specific tests for utility functions using lightweight fixtures."""

import pytest
import torch
import json
from pathlib import Path

from training_lens.utils.model_utils import (
    get_model_size,
    count_parameters,
    get_lora_state_dict,
)
from training_lens.utils.io_utils import (
    save_json,
    load_json,
    ensure_dir_exists,
    safe_torch_save,
    safe_torch_load,
)


@pytest.mark.ci
class TestModelUtilsCI:
    """Test model utility functions in CI environment."""

    def test_get_model_size_simple_model(self, simple_model):
        """Test getting model size with simple CI model."""
        size_mb = get_model_size(simple_model)

        # Should return a positive number
        assert size_mb > 0
        assert isinstance(size_mb, float)

        # Simple model should be small (less than 10MB)
        assert size_mb < 10.0

    def test_count_parameters_simple_model(self, simple_model):
        """Test counting parameters with simple model."""
        total_params, trainable_params = count_parameters(simple_model)

        # Should have some parameters
        assert total_params > 0
        assert trainable_params >= 0
        assert trainable_params <= total_params

        # Check specific counts for our simple model
        # Count lora parameters
        lora_params = sum(p.numel() for n, p in simple_model.named_parameters() if "lora_" in n)
        assert lora_params > 0

    def test_get_lora_state_dict_simple_model(self, simple_model):
        """Test extracting LoRA state dict from simple model."""
        lora_state = get_lora_state_dict(simple_model)

        # Should contain only LoRA parameters
        assert len(lora_state) > 0
        for key in lora_state:
            assert "lora_" in key.lower() or "adapter" in key.lower()

        # Verify tensors
        for key, value in lora_state.items():
            assert isinstance(value, torch.Tensor)
            assert value.numel() > 0

    def test_model_utils_with_regular_model(self):
        """Test model utils with non-LoRA model."""
        # Create a simple non-LoRA model
        model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 10))

        # Should still work
        size_mb = get_model_size(model)
        assert size_mb > 0

        total, trainable = count_parameters(model)
        assert total == sum(p.numel() for p in model.parameters())
        assert trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)

        # LoRA state dict should be empty or small
        lora_state = get_lora_state_dict(model)
        assert len(lora_state) == 0  # No LoRA parameters


@pytest.mark.ci
class TestIOUtilsCI:
    """Test I/O utility functions in CI environment."""

    def test_save_load_json(self, temp_dir):
        """Test JSON save and load utilities."""
        data = {
            "model": "test-model",
            "step": 100,
            "metrics": {"loss": 2.5, "accuracy": 0.85},
            "config": {"lr": 1e-4, "batch_size": 16},
        }

        # Save JSON
        json_path = temp_dir / "test_data.json"
        save_json(data, json_path)
        assert json_path.exists()

        # Load JSON
        loaded_data = load_json(json_path)
        assert loaded_data == data
        assert loaded_data["step"] == 100
        assert loaded_data["metrics"]["loss"] == 2.5

    def test_ensure_dir_exists(self, temp_dir):
        """Test directory creation utility."""
        # Create nested directory
        new_dir = temp_dir / "level1" / "level2" / "level3"
        ensure_dir_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

        # Should not fail if directory already exists
        ensure_dir_exists(new_dir)
        assert new_dir.exists()

    def test_safe_torch_save_load(self, temp_dir):
        """Test safe PyTorch save and load."""
        # Create test data
        data = {
            "tensor": torch.randn(10, 5),
            "list": [1, 2, 3],
            "dict": {"a": torch.tensor([1.0, 2.0]), "b": "test"},
            "model_state": torch.nn.Linear(5, 3).state_dict(),
        }

        # Save with safe_torch_save
        save_path = temp_dir / "test_checkpoint.pt"
        safe_torch_save(data, save_path)
        assert save_path.exists()

        # Load with safe_torch_load
        loaded_data = safe_torch_load(save_path)

        # Verify data
        assert torch.allclose(loaded_data["tensor"], data["tensor"])
        assert loaded_data["list"] == data["list"]
        assert torch.allclose(loaded_data["dict"]["a"], data["dict"]["a"])
        assert loaded_data["dict"]["b"] == data["dict"]["b"]

    def test_safe_torch_save_with_error_handling(self, temp_dir):
        """Test safe torch save error handling."""
        # Try to save to invalid location
        invalid_path = temp_dir / "nonexistent" / "dir" / "file.pt"

        # Should create directory and save
        data = {"test": torch.tensor([1, 2, 3])}
        safe_torch_save(data, invalid_path)
        assert invalid_path.exists()

    def test_json_with_special_types(self, temp_dir):
        """Test JSON handling with special types."""
        # Data with Path objects and other types
        data = {
            "path": str(temp_dir / "test"),  # Convert Path to string
            "number": 42,
            "float": 3.14159,
            "bool": True,
            "none": None,
            "list_of_paths": [str(temp_dir / f"file{i}") for i in range(3)],
        }

        json_path = temp_dir / "special_types.json"
        save_json(data, json_path)
        loaded = load_json(json_path)

        assert loaded == data
        assert loaded["float"] == 3.14159
        assert loaded["none"] is None


@pytest.mark.ci
class TestUtilsIntegrationCI:
    """Test utilities working together in CI."""

    def test_save_model_info(self, temp_dir, simple_model):
        """Test saving model information using utilities."""
        # Get model info
        size_mb = get_model_size(simple_model)
        total_params, trainable_params = count_parameters(simple_model)
        lora_state = get_lora_state_dict(simple_model)

        # Create info dict
        model_info = {
            "model_size_mb": size_mb,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_parameters": len(lora_state),
            "parameter_names": list(lora_state.keys())[:5],  # First 5 names
        }

        # Save as JSON
        info_path = temp_dir / "model_info.json"
        save_json(model_info, info_path)

        # Save LoRA weights
        weights_path = temp_dir / "lora_weights.pt"
        safe_torch_save(lora_state, weights_path)

        # Verify files exist and can be loaded
        assert info_path.exists()
        assert weights_path.exists()

        loaded_info = load_json(info_path)
        assert loaded_info["model_size_mb"] == size_mb

        loaded_weights = safe_torch_load(weights_path)
        assert len(loaded_weights) == len(lora_state)

    def test_checkpoint_workflow(self, temp_dir, simple_model, simple_optimizer):
        """Test checkpoint save/load workflow with utilities."""
        # Create checkpoint directory
        checkpoint_dir = temp_dir / "checkpoints" / "step_100"
        ensure_dir_exists(checkpoint_dir)

        # Save model state
        model_state = simple_model.state_dict()
        safe_torch_save(model_state, checkpoint_dir / "model.pt")

        # Save optimizer state
        opt_state = simple_optimizer.state_dict()
        safe_torch_save(opt_state, checkpoint_dir / "optimizer.pt")

        # Save metadata
        metadata = {
            "step": 100,
            "model_size_mb": get_model_size(simple_model),
            "total_params": count_parameters(simple_model)[0],
            "timestamp": "2024-01-01T00:00:00",
        }
        save_json(metadata, checkpoint_dir / "metadata.json")

        # Verify checkpoint
        assert (checkpoint_dir / "model.pt").exists()
        assert (checkpoint_dir / "optimizer.pt").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        # Load checkpoint
        loaded_model_state = safe_torch_load(checkpoint_dir / "model.pt")
        loaded_metadata = load_json(checkpoint_dir / "metadata.json")

        assert loaded_metadata["step"] == 100
        assert len(loaded_model_state) == len(model_state)
