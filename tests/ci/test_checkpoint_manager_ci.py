"""CI-specific tests for checkpoint manager using lightweight fixtures."""

import json
from pathlib import Path

import pytest
import torch

from training_lens.training.checkpoint_manager import CheckpointManager
from training_lens.training.config import CheckpointMetadata


@pytest.mark.ci
class TestCheckpointManagerCI:
    """Test checkpoint management functionality in CI environment."""

    @pytest.fixture
    def ci_checkpoint_metadata(self):
        """Lightweight checkpoint metadata for CI testing."""
        return {
            "step": 50,
            "epoch": 0.5,
            "learning_rate": 1e-4,
            "train_loss": 3.0,
            "eval_loss": 2.8,
            "grad_norm": 1.5,
        }

    def test_initialization_ci(self, temp_dir):
        """Test checkpoint manager initialization in CI."""
        manager = CheckpointManager(output_dir=temp_dir, max_checkpoints=3)

        assert manager.output_dir == temp_dir
        assert manager.max_checkpoints == 3
        assert manager.checkpoint_dir == temp_dir / "checkpoints"
        assert manager.checkpoint_dir.exists()
        assert len(manager.checkpoints) == 0

    def test_save_lora_checkpoint_simple_model(self, temp_dir, simple_model, simple_optimizer, ci_checkpoint_metadata):
        """Test saving LoRA checkpoint with simple CI model."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**ci_checkpoint_metadata)

        # Save checkpoint with simple model
        checkpoint_path = manager.save_lora_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=metadata,
            additional_data={"ci_test": True, "test_tensor": torch.tensor([1.0, 2.0])},
        )

        # Verify checkpoint was saved
        assert checkpoint_path.exists()
        assert checkpoint_path.name == f"lora-checkpoint-{metadata.step}"

        # Check saved files
        assert (checkpoint_path / "adapter").exists()
        assert (checkpoint_path / "lora_optimizer.pt").exists()
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "lora_training_data.pt").exists()

        # Verify metadata
        with open(checkpoint_path / "metadata.json") as f:
            saved_metadata = json.load(f)
        assert saved_metadata["step"] == metadata.step
        assert saved_metadata["checkpoint_type"] == "lora_adapter"
        assert saved_metadata["adapter_only"] is True

        # Verify additional data
        additional_data = torch.load(checkpoint_path / "lora_training_data.pt")
        assert additional_data["ci_test"] is True
        assert torch.allclose(additional_data["test_tensor"], torch.tensor([1.0, 2.0]))

    def test_load_checkpoint_ci(self, temp_dir, simple_model, simple_optimizer, ci_checkpoint_metadata):
        """Test loading checkpoint in CI environment."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**ci_checkpoint_metadata)

        # Save checkpoint first
        checkpoint_path = manager.save_lora_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=metadata,
        )

        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path=checkpoint_path)

        assert loaded["checkpoint_path"] == checkpoint_path
        assert loaded["is_lora_checkpoint"] is True
        assert loaded["adapter_path"] == checkpoint_path / "adapter"
        assert loaded["metadata"]["step"] == metadata.step
        assert loaded["optimizer_state"] is not None

    def test_checkpoint_cleanup_ci(self, temp_dir, simple_model, ci_checkpoint_metadata):
        """Test automatic cleanup of old checkpoints in CI."""
        manager = CheckpointManager(output_dir=temp_dir, max_checkpoints=2)

        # Save multiple checkpoints with different steps
        steps = [10, 20, 30, 40]
        for step in steps:
            metadata = CheckpointMetadata(**{**ci_checkpoint_metadata, "step": step})
            manager.save_lora_checkpoint(model=simple_model, metadata=metadata)

        # Should only keep the last 2 checkpoints
        assert len(manager.checkpoints) == 2
        remaining_steps = [cp["step"] for cp in manager.checkpoints]
        assert remaining_steps == [30, 40]

        # Check that old checkpoints were deleted
        assert not (manager.checkpoint_dir / "lora-checkpoint-10").exists()
        assert not (manager.checkpoint_dir / "lora-checkpoint-20").exists()
        assert (manager.checkpoint_dir / "lora-checkpoint-30").exists()
        assert (manager.checkpoint_dir / "lora-checkpoint-40").exists()

    def test_list_checkpoints_ci(self, temp_dir, simple_model, ci_checkpoint_metadata):
        """Test listing checkpoints in CI."""
        manager = CheckpointManager(output_dir=temp_dir)

        # Save some checkpoints
        steps = [5, 10, 15]
        for step in steps:
            metadata = CheckpointMetadata(**{**ci_checkpoint_metadata, "step": step})
            manager.save_lora_checkpoint(model=simple_model, metadata=metadata)

        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        assert [cp["step"] for cp in checkpoints] == steps

    def test_delete_checkpoint_ci(self, temp_dir, simple_model, ci_checkpoint_metadata):
        """Test deleting specific checkpoint in CI."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**ci_checkpoint_metadata)

        # Save checkpoint
        checkpoint_path = manager.save_lora_checkpoint(model=simple_model, metadata=metadata)
        assert checkpoint_path.exists()

        # Delete checkpoint
        success = manager.delete_checkpoint(metadata.step)
        assert success is True
        assert not checkpoint_path.exists()
        assert len(manager.checkpoints) == 0

    def test_checkpoint_with_minimal_data(self, temp_dir, simple_model):
        """Test checkpoint with minimal data for CI speed."""
        manager = CheckpointManager(output_dir=temp_dir)

        # Create minimal metadata
        metadata = CheckpointMetadata(
            step=1,
            epoch=0.1,
            learning_rate=1e-4,
            train_loss=None,  # Optional fields
            eval_loss=None,
            grad_norm=None,
        )

        # Save without optimizer (faster for CI)
        checkpoint_path = manager.save_lora_checkpoint(
            model=simple_model,
            optimizer=None,
            metadata=metadata,
        )

        assert checkpoint_path.exists()
        assert (checkpoint_path / "adapter").exists()
        assert not (checkpoint_path / "lora_optimizer.pt").exists()  # No optimizer saved

        # Load and verify
        loaded = manager.load_checkpoint(checkpoint_path=checkpoint_path)
        assert loaded["optimizer_state"] is None
        assert loaded["metadata"]["step"] == 1

    def test_concurrent_checkpoint_operations(self, temp_dir, simple_model, ci_checkpoint_metadata):
        """Test multiple checkpoint operations in sequence."""
        manager = CheckpointManager(output_dir=temp_dir, max_checkpoints=3)

        # Rapidly save several checkpoints
        for i in range(5):
            metadata = CheckpointMetadata(**{**ci_checkpoint_metadata, "step": i * 10})
            manager.save_lora_checkpoint(model=simple_model, metadata=metadata)

        # Verify cleanup worked correctly
        assert len(manager.checkpoints) == 3

        # List should return same as internal tracking
        listed = manager.list_checkpoints()
        assert len(listed) == 3
        assert [cp["step"] for cp in listed] == [20, 30, 40]

        # Delete middle checkpoint
        manager.delete_checkpoint(30)
        assert len(manager.checkpoints) == 2
        assert [cp["step"] for cp in manager.checkpoints] == [20, 40]
