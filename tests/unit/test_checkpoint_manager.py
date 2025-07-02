"""Unit tests for checkpoint manager."""

import json
from pathlib import Path

import pytest
import torch

from training_lens.training.checkpoint_manager import CheckpointManager
from training_lens.training.config import CheckpointMetadata


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(output_dir=temp_dir, max_checkpoints=5)
        
        assert manager.output_dir == temp_dir
        assert manager.max_checkpoints == 5
        assert manager.checkpoint_dir == temp_dir / "checkpoints"
        assert manager.checkpoint_dir.exists()
        assert len(manager.checkpoints) == 0
    
    def test_save_lora_checkpoint(self, temp_dir, mock_model, mock_optimizer, sample_checkpoint_metadata):
        """Test saving LoRA checkpoint."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**sample_checkpoint_metadata)
        
        # Save checkpoint
        checkpoint_path = manager.save_lora_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            metadata=metadata,
            additional_data={"test_data": torch.tensor([1, 2, 3])},
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
    
    def test_load_checkpoint(self, temp_dir, mock_model, mock_optimizer, sample_checkpoint_metadata):
        """Test loading checkpoint."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**sample_checkpoint_metadata)
        
        # Save checkpoint first
        checkpoint_path = manager.save_lora_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            metadata=metadata,
        )
        
        # Load checkpoint
        loaded = manager.load_checkpoint(step=metadata.step)
        
        assert loaded["checkpoint_path"] == checkpoint_path
        assert loaded["is_lora_checkpoint"] is True
        assert loaded["adapter_path"] == checkpoint_path / "adapter"
        assert loaded["metadata"]["step"] == metadata.step
        assert loaded["optimizer_state"] is not None
    
    def test_checkpoint_cleanup(self, temp_dir, mock_model, sample_checkpoint_metadata):
        """Test automatic cleanup of old checkpoints."""
        manager = CheckpointManager(output_dir=temp_dir, max_checkpoints=3)
        
        # Save multiple checkpoints
        for step in [100, 200, 300, 400, 500]:
            metadata = CheckpointMetadata(**{**sample_checkpoint_metadata, "step": step})
            manager.save_lora_checkpoint(model=mock_model, metadata=metadata)
        
        # Should only keep the last 3 checkpoints
        assert len(manager.checkpoints) == 3
        remaining_steps = [cp["step"] for cp in manager.checkpoints]
        assert remaining_steps == [300, 400, 500]
        
        # Check that old checkpoints were deleted
        assert not (manager.checkpoint_dir / "lora-checkpoint-100").exists()
        assert not (manager.checkpoint_dir / "lora-checkpoint-200").exists()
        assert (manager.checkpoint_dir / "lora-checkpoint-300").exists()
    
    def test_list_checkpoints(self, temp_dir, mock_model, sample_checkpoint_metadata):
        """Test listing checkpoints."""
        manager = CheckpointManager(output_dir=temp_dir)
        
        # Save some checkpoints
        steps = [100, 200, 300]
        for step in steps:
            metadata = CheckpointMetadata(**{**sample_checkpoint_metadata, "step": step})
            manager.save_lora_checkpoint(model=mock_model, metadata=metadata)
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        assert [cp["step"] for cp in checkpoints] == steps
        
    def test_delete_checkpoint(self, temp_dir, mock_model, sample_checkpoint_metadata):
        """Test deleting specific checkpoint."""
        manager = CheckpointManager(output_dir=temp_dir)
        metadata = CheckpointMetadata(**sample_checkpoint_metadata)
        
        # Save checkpoint
        checkpoint_path = manager.save_lora_checkpoint(model=mock_model, metadata=metadata)
        assert checkpoint_path.exists()
        
        # Delete checkpoint
        success = manager.delete_checkpoint(metadata.step)
        assert success is True
        assert not checkpoint_path.exists()
        assert len(manager.checkpoints) == 0