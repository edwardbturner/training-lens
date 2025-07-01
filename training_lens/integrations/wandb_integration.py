"""Weights & Biases integration for experiment tracking."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import wandb

from ..utils.logging import get_logger

logger = get_logger(__name__)


class WandBIntegration:
    """Integration with Weights & Biases for experiment tracking."""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize W&B integration.

        Args:
            project: W&B project name
            entity: W&B entity (team/username)
            run_name: Custom run name
            config: Configuration dictionary to log
            tags: List of tags for the run
            notes: Notes for the run
            api_key: W&B API key (uses WANDB_API_KEY env var if None)
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.api_key = api_key or os.getenv("WANDB_API_KEY")

        # Initialize W&B
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                tags=tags,
                notes=notes,
                reinit=True,
            )
            logger.info(f"W&B run initialized: {self.run.name} ({self.run.id})")

            # Store run info
            self.run_id = self.run.id
            self.run_url = self.run.url

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.run = None
            self.run_id = None
            self.run_url = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.run is None:
            return

        try:
            # Filter out non-numeric values for W&B
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    wandb_metrics[key] = float(value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (int, float, np.number)):
                            wandb_metrics[f"{key}/{nested_key}"] = float(nested_value)

            self.run.log(wandb_metrics, step=step)

        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")

    def log_gradient_analysis(self, cosine_similarities: List[float], step: int) -> None:
        """Log gradient analysis visualizations.

        Args:
            cosine_similarities: List of cosine similarity values
            step: Current training step
        """
        if self.run is None or not cosine_similarities:
            return

        try:
            # Create gradient cosine similarity plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cosine_similarities, linewidth=2, alpha=0.8)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Gradient Cosine Similarity")
            ax.set_title("Gradient Direction Consistency Over Training")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

            # Add trend line if enough data
            if len(cosine_similarities) > 10:
                x = np.arange(len(cosine_similarities))
                z = np.polyfit(x, cosine_similarities, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=1, label="Trend")
                ax.legend()

            plt.tight_layout()

            # Log to W&B
            self.run.log(
                {
                    "gradient_cosine_similarity_plot": wandb.Image(fig),
                    "gradient_cosine_similarity_current": cosine_similarities[-1] if cosine_similarities else 0,
                    "gradient_cosine_similarity_mean": np.mean(cosine_similarities),
                    "gradient_cosine_similarity_std": np.std(cosine_similarities),
                },
                step=step,
            )

            plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to log gradient analysis: {e}")

    def log_weight_distribution(self, weight_stats: Dict[str, Any], step: int) -> None:
        """Log weight distribution analysis.

        Args:
            weight_stats: Dictionary of weight statistics
            step: Current training step
        """
        if self.run is None:
            return

        try:
            # Log weight statistics
            metrics = {
                "weights/overall_norm": weight_stats.get("overall_weight_norm", 0),
                "weights/overall_mean": weight_stats.get("overall_weight_mean", 0),
                "weights/overall_std": weight_stats.get("overall_weight_std", 0),
            }

            # Log layer-wise weight norms
            layer_norms = weight_stats.get("layer_weight_norms", {})
            for layer_name, norm in layer_norms.items():
                # Clean layer name for W&B
                clean_name = layer_name.replace(".", "_").replace("/", "_")
                metrics[f"weights/layer_norms/{clean_name}"] = norm

            self.run.log(metrics, step=step)

        except Exception as e:
            logger.warning(f"Failed to log weight distribution: {e}")

    def log_memory_usage(self, memory_stats: Dict[str, Any], step: int) -> None:
        """Log memory usage statistics.

        Args:
            memory_stats: Memory usage statistics
            step: Current training step
        """
        if self.run is None:
            return

        try:
            metrics = {}

            # CPU memory
            if "memory_rss_mb" in memory_stats:
                metrics["memory/cpu_rss_mb"] = memory_stats["memory_rss_mb"]
            if "memory_percent" in memory_stats:
                metrics["memory/cpu_percent"] = memory_stats["memory_percent"]

            # GPU memory
            if "gpu_memory_allocated_mb" in memory_stats:
                metrics["memory/gpu_allocated_mb"] = memory_stats["gpu_memory_allocated_mb"]
            if "gpu_memory_percent" in memory_stats:
                metrics["memory/gpu_percent"] = memory_stats["gpu_memory_percent"]

            self.run.log(metrics, step=step)

        except Exception as e:
            logger.warning(f"Failed to log memory usage: {e}")

    def log_training_progress(self, progress_data: Dict[str, Any]) -> None:
        """Log overall training progress summary.

        Args:
            progress_data: Dictionary containing training progress information
        """
        if self.run is None:
            return

        try:
            # Create training summary table
            summary_data = []
            for key, value in progress_data.items():
                if isinstance(value, (int, float, str)):
                    summary_data.append([key, str(value)])

            if summary_data:
                table = wandb.Table(columns=["Metric", "Value"], data=summary_data)
                self.run.log({"training_summary": table})

        except Exception as e:
            logger.warning(f"Failed to log training progress: {e}")

    def log_checkpoint_info(self, checkpoint_path: Path, step: int, metadata: Dict[str, Any]) -> None:
        """Log checkpoint information.

        Args:
            checkpoint_path: Path to saved checkpoint
            step: Training step
            metadata: Checkpoint metadata
        """
        if self.run is None:
            return

        try:
            self.run.log(
                {
                    "checkpoint/step": step,
                    "checkpoint/path": str(checkpoint_path),
                    "checkpoint/size_mb": (
                        checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
                    ),
                    "checkpoint/timestamp": metadata.get("timestamp", ""),
                },
                step=step,
            )

        except Exception as e:
            logger.warning(f"Failed to log checkpoint info: {e}")

    def log_model_architecture(self, model) -> None:
        """Log model architecture information.

        Args:
            model: PyTorch model
        """
        if self.run is None:
            return

        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Log architecture summary
            self.run.log(
                {
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "model/trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
                    "model/architecture": str(model.__class__.__name__),
                }
            )

            # Create parameter distribution plot
            param_sizes = []
            param_names = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_sizes.append(param.numel())
                    param_names.append(name)

            if param_sizes:
                fig, ax = plt.subplots(figsize=(12, 8))
                y_pos = np.arange(len(param_names))

                bars = ax.barh(y_pos, param_sizes)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([name.split(".")[-1] for name in param_names])
                ax.set_xlabel("Number of Parameters")
                ax.set_title("Model Parameter Distribution by Layer")

                # Add value labels on bars
                for i, (bar, size) in enumerate(zip(bars, param_sizes)):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2, f"{size:,}", ha="left", va="center", fontsize=8)

                plt.tight_layout()
                self.run.log({"model/parameter_distribution": wandb.Image(fig)})
                plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")

    def create_run_summary(self, final_metrics: Dict[str, Any]) -> None:
        """Create a comprehensive run summary.

        Args:
            final_metrics: Final training metrics
        """
        if self.run is None:
            return

        try:
            # Update run summary
            for key, value in final_metrics.items():
                if isinstance(value, (int, float, str)):
                    self.run.summary[key] = value

            # Add custom summary metrics
            self.run.summary["training_completed"] = True

        except Exception as e:
            logger.warning(f"Failed to create run summary: {e}")

    def save_code(self, code_dir: Optional[Union[str, Path]] = None) -> None:
        """Save code to W&B for reproducibility.

        Args:
            code_dir: Directory containing code to save
        """
        if self.run is None:
            return

        try:
            if code_dir:
                self.run.log_code(root=str(code_dir))
            else:
                # Save current working directory
                self.run.log_code()

        except Exception as e:
            logger.warning(f"Failed to save code: {e}")

    def add_tag(self, tag: str) -> None:
        """Add a tag to the current run.

        Args:
            tag: Tag to add
        """
        if self.run is None:
            return

        try:
            current_tags = list(self.run.tags or [])
            if tag not in current_tags:
                current_tags.append(tag)
                self.run.tags = current_tags

        except Exception as e:
            logger.warning(f"Failed to add tag: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is None:
            return

        try:
            self.run.finish()
            logger.info("W&B run finished successfully")

        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

    @property
    def is_active(self) -> bool:
        """Check if W&B run is active."""
        return self.run is not None

    def get_run_url(self) -> Optional[str]:
        """Get the W&B run URL."""
        return self.run_url if self.run else None
