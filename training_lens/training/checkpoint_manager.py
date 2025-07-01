"""Checkpoint management for training runs."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.helpers import ensure_dir, load_file, safe_save
from ..utils.logging import get_logger
from .config import CheckpointMetadata

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoints and metadata during training."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_checkpoints: int = 10,
        save_optimizer: bool = True,
        save_tokenizer: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_tokenizer = save_tokenizer

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_dir(self.checkpoint_dir)

        # Track checkpoints
        self.checkpoints: List[Dict[str, Any]] = []
        self._load_checkpoint_index()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metadata: Optional[CheckpointMetadata] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            model: The model to save
            tokenizer: Optional tokenizer to save
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            metadata: Checkpoint metadata
            additional_data: Additional data to save with checkpoint

        Returns:
            Path to the saved checkpoint directory
        """
        if metadata is None:
            raise ValueError("Checkpoint metadata is required")

        step = metadata.step
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        logger.info(f"Saving checkpoint at step {step} to {checkpoint_path}")

        # Create checkpoint directory
        ensure_dir(checkpoint_path)

        # Save model
        model.save_pretrained(checkpoint_path / "model")

        # Save tokenizer if provided
        if tokenizer and self.save_tokenizer:
            tokenizer.save_pretrained(checkpoint_path / "tokenizer")

        # Save optimizer state if provided
        if optimizer and self.save_optimizer:
            optimizer_state = {
                "state_dict": optimizer.state_dict(),
                "param_groups": optimizer.param_groups,
            }
            safe_save(optimizer_state, checkpoint_path / "optimizer.pt", format="torch")

        # Save scheduler state if provided
        if scheduler:
            scheduler_state = {
                "state_dict": scheduler.state_dict(),
                "last_epoch": scheduler.last_epoch,
            }
            safe_save(scheduler_state, checkpoint_path / "scheduler.pt", format="torch")

        # Save metadata
        metadata_dict = metadata.to_dict()
        metadata_dict["timestamp"] = datetime.now().isoformat()
        metadata_dict["checkpoint_path"] = str(checkpoint_path)

        safe_save(metadata_dict, checkpoint_path / "metadata.json", format="json")

        # Save additional data if provided
        if additional_data:
            safe_save(additional_data, checkpoint_path / "additional_data.pt", format="torch")

        # Update checkpoint index
        checkpoint_info = {
            "step": step,
            "path": str(checkpoint_path),
            "timestamp": metadata_dict["timestamp"],
            "metadata": metadata_dict,
        }
        self.checkpoints.append(checkpoint_info)

        # Sort checkpoints by step
        self.checkpoints.sort(key=lambda x: x["step"])

        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()

        # Save updated index
        self._save_checkpoint_index()

        logger.info(f"Checkpoint saved successfully at {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        step: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            step: Specific step to load (loads latest if None)
            checkpoint_path: Direct path to checkpoint directory

        Returns:
            Dictionary containing checkpoint data
        """
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
        elif step is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        else:
            # Load latest checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            checkpoint_path = Path(self.checkpoints[-1]["path"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = load_file(metadata_path, format="json") if metadata_path.exists() else {}

        # Load optimizer state if exists
        optimizer_path = checkpoint_path / "optimizer.pt"
        optimizer_state = load_file(optimizer_path, format="torch") if optimizer_path.exists() else None

        # Load scheduler state if exists
        scheduler_path = checkpoint_path / "scheduler.pt"
        scheduler_state = load_file(scheduler_path, format="torch") if scheduler_path.exists() else None

        # Load additional data if exists
        additional_data_path = checkpoint_path / "additional_data.pt"
        additional_data = load_file(additional_data_path, format="torch") if additional_data_path.exists() else None

        return {
            "checkpoint_path": checkpoint_path,
            "model_path": checkpoint_path / "model",
            "tokenizer_path": checkpoint_path / "tokenizer",
            "metadata": metadata,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "additional_data": additional_data,
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoints.copy()

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint info."""
        return self.checkpoints[-1] if self.checkpoints else None

    def delete_checkpoint(self, step: int) -> bool:
        """Delete a specific checkpoint.

        Args:
            step: Step number of checkpoint to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"

        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)

            # Remove from index
            self.checkpoints = [cp for cp in self.checkpoints if cp["step"] != step]
            self._save_checkpoint_index()

            logger.info(f"Deleted checkpoint at step {step}")
            return True

        return False

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on max_checkpoints setting."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Keep the most recent checkpoints
        checkpoints_to_remove = self.checkpoints[: -self.max_checkpoints]

        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info["path"])
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")

        # Update checkpoint list
        self.checkpoints = self.checkpoints[-self.max_checkpoints :]

    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        if index_path.exists():
            try:
                self.checkpoints = load_file(index_path, format="json")
                logger.debug(f"Loaded checkpoint index with {len(self.checkpoints)} entries")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
                self.checkpoints = []
        else:
            self.checkpoints = []

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        safe_save(self.checkpoints, index_path, format="json")
        logger.debug(f"Saved checkpoint index with {len(self.checkpoints)} entries")
