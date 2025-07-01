"""Checkpoint analysis for extracting training insights."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..integrations.huggingface_integration import HuggingFaceIntegration
from ..utils.helpers import ensure_dir, load_file
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointAnalyzer:
    """Analyzes training checkpoints to extract insights."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize checkpoint analyzer.

        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints_info = []
        self.metrics_data = {}

        # Load checkpoint index if available
        self._load_checkpoint_index()

        logger.info(f"CheckpointAnalyzer initialized with {len(self.checkpoints_info)} checkpoints")

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        local_dir: Optional[Union[str, Path]] = None,
        token: Optional[str] = None,
    ) -> "CheckpointAnalyzer":
        """Create analyzer from HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID
            local_dir: Local directory to download to
            token: HuggingFace token

        Returns:
            CheckpointAnalyzer instance
        """
        if local_dir is None:
            local_dir = Path(f"./downloads/{repo_id.replace('/', '_')}")

        local_dir = Path(local_dir)
        ensure_dir(local_dir)

        # Initialize HF integration
        hf_integration = HuggingFaceIntegration(repo_id, token=token)

        # List and download checkpoints
        checkpoints = hf_integration.list_checkpoints()

        for checkpoint_info in checkpoints:
            step = checkpoint_info["step"]
            try:
                hf_integration.download_checkpoint(step, local_dir)
                logger.info(f"Downloaded checkpoint {step}")
            except Exception as e:
                logger.warning(f"Failed to download checkpoint {step}: {e}")

        return cls(local_dir / "training_lens_checkpoints")

    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        if index_path.exists():
            try:
                self.checkpoints_info = load_file(index_path, format="json")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")
                self._discover_checkpoints()
        else:
            self._discover_checkpoints()

    def _discover_checkpoints(self) -> None:
        """Discover checkpoints in the directory."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {self.checkpoint_dir}")
            return

        checkpoint_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

        for checkpoint_dir in sorted(checkpoint_dirs):
            try:
                # Extract step number
                step = int(checkpoint_dir.name.split("-")[1])

                # Load metadata if available
                metadata_path = checkpoint_dir / "metadata.json"
                metadata = load_file(metadata_path, format="json") if metadata_path.exists() else {}

                checkpoint_info = {
                    "step": step,
                    "path": str(checkpoint_dir),
                    "metadata": metadata,
                }

                self.checkpoints_info.append(checkpoint_info)

            except (ValueError, Exception) as e:
                logger.warning(f"Failed to process checkpoint {checkpoint_dir}: {e}")

        # Sort by step
        self.checkpoints_info.sort(key=lambda x: x["step"])
        logger.info(f"Discovered {len(self.checkpoints_info)} checkpoints")

    def load_checkpoint_metrics(self, step: int) -> Optional[Dict[str, Any]]:
        """Load metrics data for a specific checkpoint.

        Args:
            step: Checkpoint step number

        Returns:
            Metrics data or None if not found
        """
        checkpoint_info = next((cp for cp in self.checkpoints_info if cp["step"] == step), None)

        if checkpoint_info is None:
            logger.warning(f"Checkpoint {step} not found")
            return None

        checkpoint_path = Path(checkpoint_info["path"])

        # Load training lens data if available
        training_lens_data_path = checkpoint_path / "additional_data.pt"
        if training_lens_data_path.exists():
            try:
                data = load_file(training_lens_data_path, format="torch")
                self.metrics_data[step] = data
                return data
            except Exception as e:
                logger.warning(f"Failed to load training lens data for step {step}: {e}")

        return None

    def analyze_gradient_evolution(self) -> Dict[str, Any]:
        """Analyze how gradients evolve during training.

        Returns:
            Dictionary containing gradient evolution analysis
        """
        gradient_data = {}
        cosine_similarities = []

        for checkpoint_info in self.checkpoints_info:
            step = checkpoint_info["step"]
            metrics = self.load_checkpoint_metrics(step)

            if metrics and "gradient_cosine_similarities" in metrics:
                cosine_similarities.extend(metrics["gradient_cosine_similarities"])
                gradient_data[step] = metrics["gradient_cosine_similarities"]

        if not cosine_similarities:
            return {"status": "no_gradient_data"}

        # Analyze cosine similarity trends
        cosine_similarities = np.array(cosine_similarities)

        analysis = {
            "total_steps": len(cosine_similarities),
            "mean_cosine_similarity": float(np.mean(cosine_similarities)),
            "std_cosine_similarity": float(np.std(cosine_similarities)),
            "min_cosine_similarity": float(np.min(cosine_similarities)),
            "max_cosine_similarity": float(np.max(cosine_similarities)),
            "gradient_stability": self._assess_gradient_stability(cosine_similarities),
            "convergence_analysis": self._analyze_convergence(cosine_similarities),
        }

        return analysis

    def analyze_weight_evolution(self) -> Dict[str, Any]:
        """Analyze how model weights evolve during training.

        Returns:
            Dictionary containing weight evolution analysis
        """
        weight_data = []

        for checkpoint_info in self.checkpoints_info:
            step = checkpoint_info["step"]
            metrics = self.load_checkpoint_metrics(step)

            if metrics and "weight_stats_history" in metrics:
                weight_data.extend(metrics["weight_stats_history"])

        if not weight_data:
            return {"status": "no_weight_data"}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(weight_data)

        analysis = {
            "weight_norm_trend": self._analyze_trend(df["overall_norm"].values),
            "weight_mean_trend": self._analyze_trend(df["overall_mean"].values),
            "weight_std_trend": self._analyze_trend(df["overall_std"].values),
            "weight_stability": self._assess_weight_stability(df),
            "layer_analysis": self._analyze_layer_weights(weight_data),
        }

        return analysis

    def analyze_training_dynamics(self) -> Dict[str, Any]:
        """Analyze overall training dynamics.

        Returns:
            Comprehensive training dynamics analysis
        """
        if not self.checkpoints_info:
            return {"status": "no_checkpoints"}

        # Extract training metrics from metadata
        steps = []
        losses = []
        learning_rates = []
        grad_norms = []

        for checkpoint_info in self.checkpoints_info:
            metadata = checkpoint_info.get("metadata", {})

            if "step" in metadata:
                steps.append(metadata["step"])
                losses.append(metadata.get("train_loss", np.nan))
                learning_rates.append(metadata.get("learning_rate", np.nan))
                grad_norms.append(metadata.get("grad_norm", np.nan))

        if not steps:
            return {"status": "no_training_data"}

        # Create analysis
        analysis = {
            "training_steps": len(steps),
            "first_step": min(steps),
            "last_step": max(steps),
            "loss_analysis": self._analyze_loss_curve(losses),
            "learning_rate_analysis": self._analyze_learning_rate(learning_rates),
            "gradient_norm_analysis": self._analyze_gradient_norms(grad_norms),
            "training_efficiency": self._assess_training_efficiency(losses, grad_norms),
        }

        return analysis

    def detect_overfitting(self) -> Dict[str, Any]:
        """Detect potential overfitting in the training process.

        Returns:
            Overfitting detection results
        """
        # Extract validation losses if available
        train_losses = []
        eval_losses = []

        for checkpoint_info in self.checkpoints_info:
            metadata = checkpoint_info.get("metadata", {})

            train_loss = metadata.get("train_loss")
            eval_loss = metadata.get("eval_loss")

            if train_loss is not None:
                train_losses.append(train_loss)
            if eval_loss is not None:
                eval_losses.append(eval_loss)

        if len(eval_losses) < 3:
            return {"status": "insufficient_validation_data"}

        # Analyze train/val loss divergence
        analysis = {
            "overfitting_detected": self._detect_loss_divergence(train_losses, eval_losses),
            "best_checkpoint": self._find_best_checkpoint(eval_losses),
            "early_stopping_recommendation": self._recommend_early_stopping(eval_losses),
        }

        return analysis

    def generate_standard_report(self) -> Dict[str, Any]:
        """Generate a comprehensive standard report.

        Returns:
            Standard analysis report
        """
        report = {
            "summary": {
                "total_checkpoints": len(self.checkpoints_info),
                "checkpoint_steps": [cp["step"] for cp in self.checkpoints_info],
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
            "gradient_analysis": self.analyze_gradient_evolution(),
            "weight_analysis": self.analyze_weight_evolution(),
            "training_dynamics": self.analyze_training_dynamics(),
            "overfitting_analysis": self.detect_overfitting(),
        }

        return report

    def export_raw_data(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Export raw training data for external analysis.

        Args:
            output_dir: Directory to save exported data

        Returns:
            Dictionary mapping data types to file paths
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        exported_files = {}

        # Export checkpoint metadata
        metadata_df = pd.DataFrame([cp["metadata"] for cp in self.checkpoints_info])
        metadata_path = output_dir / "checkpoint_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        exported_files["metadata"] = metadata_path

        # Export gradient data
        all_gradient_data = []
        for checkpoint_info in self.checkpoints_info:
            step = checkpoint_info["step"]
            metrics = self.load_checkpoint_metrics(step)
            if metrics and "gradient_cosine_similarities" in metrics:
                all_gradient_data.extend(metrics["gradient_cosine_similarities"])

        if all_gradient_data:
            gradient_path = output_dir / "gradient_cosine_similarities.npy"
            np.save(gradient_path, np.array(all_gradient_data))
            exported_files["gradient_similarities"] = gradient_path

        # Export weight evolution data
        all_weight_data = []
        for checkpoint_info in self.checkpoints_info:
            step = checkpoint_info["step"]
            metrics = self.load_checkpoint_metrics(step)
            if metrics and "weight_stats_history" in metrics:
                all_weight_data.extend(metrics["weight_stats_history"])

        if all_weight_data:
            weight_df = pd.DataFrame(all_weight_data)
            weight_path = output_dir / "weight_evolution.csv"
            weight_df.to_csv(weight_path, index=False)
            exported_files["weight_evolution"] = weight_path

        logger.info(f"Exported {len(exported_files)} data files to {output_dir}")
        return exported_files

    def _assess_gradient_stability(self, cosine_similarities: np.ndarray) -> str:
        """Assess gradient stability based on cosine similarities."""
        mean_sim = np.mean(cosine_similarities)
        std_sim = np.std(cosine_similarities)

        if mean_sim > 0.8 and std_sim < 0.1:
            return "very_stable"
        elif mean_sim > 0.6 and std_sim < 0.2:
            return "stable"
        elif mean_sim > 0.3:
            return "moderate"
        else:
            return "unstable"

    def _analyze_convergence(self, cosine_similarities: np.ndarray) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        if len(cosine_similarities) < 10:
            return {"status": "insufficient_data"}

        # Calculate trend
        x = np.arange(len(cosine_similarities))
        trend_coef = np.polyfit(x, cosine_similarities, 1)[0]

        # Recent stability
        recent_window = min(50, len(cosine_similarities) // 4)
        recent_std = np.std(cosine_similarities[-recent_window:])

        return {
            "trend_coefficient": float(trend_coef),
            "trend_direction": (
                "increasing" if trend_coef > 0.001 else "decreasing" if trend_coef < -0.001 else "stable"
            ),
            "recent_stability": float(recent_std),
            "convergence_status": (
                "converged" if recent_std < 0.05 else "converging" if trend_coef > 0 else "not_converged"
            ),
        }

    def _analyze_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in a series of values."""
        if len(values) < 2:
            return {"status": "insufficient_data"}

        x = np.arange(len(values))
        trend_coef = np.polyfit(x, values, 1)[0]

        return {
            "trend_coefficient": float(trend_coef),
            "trend_direction": "increasing" if trend_coef > 0 else "decreasing",
            "initial_value": float(values[0]),
            "final_value": float(values[-1]),
            "change_percentage": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0,
        }

    def _assess_weight_stability(self, df: pd.DataFrame) -> str:
        """Assess weight stability."""
        norm_std = df["overall_norm"].std()
        norm_mean = df["overall_norm"].mean()

        coefficient_of_variation = norm_std / norm_mean if norm_mean != 0 else float("inf")

        if coefficient_of_variation < 0.05:
            return "very_stable"
        elif coefficient_of_variation < 0.1:
            return "stable"
        elif coefficient_of_variation < 0.2:
            return "moderate"
        else:
            return "unstable"

    def _analyze_layer_weights(self, weight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze layer-wise weight evolution."""
        if not weight_data:
            return {"status": "no_data"}

        # Get layer names from first entry
        first_entry = weight_data[0]
        layer_norms = first_entry.get("layer_norms", {})

        if not layer_norms:
            return {"status": "no_layer_data"}

        layer_analysis = {}

        for layer_name in layer_norms.keys():
            layer_values = []
            for entry in weight_data:
                if "layer_norms" in entry and layer_name in entry["layer_norms"]:
                    layer_values.append(entry["layer_norms"][layer_name])

            if layer_values:
                layer_analysis[layer_name] = self._analyze_trend(np.array(layer_values))

        return layer_analysis

    def _analyze_loss_curve(self, losses: List[float]) -> Dict[str, Any]:
        """Analyze loss curve characteristics."""
        losses = np.array([loss for loss in losses if not np.isnan(loss)])

        if len(losses) < 2:
            return {"status": "insufficient_data"}

        return {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_reduction": float(losses[0] - losses[-1]),
            "loss_reduction_percentage": float((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0.0,
            "smoothness": self._calculate_smoothness(losses),
        }

    def _analyze_learning_rate(self, learning_rates: List[float]) -> Dict[str, Any]:
        """Analyze learning rate schedule."""
        lrs = np.array([lr for lr in learning_rates if not np.isnan(lr)])

        if len(lrs) < 2:
            return {"status": "insufficient_data"}

        return {
            "initial_lr": float(lrs[0]),
            "final_lr": float(lrs[-1]),
            "lr_decay": float(lrs[0] - lrs[-1]),
            "schedule_type": self._detect_lr_schedule(lrs),
        }

    def _analyze_gradient_norms(self, grad_norms: List[float]) -> Dict[str, Any]:
        """Analyze gradient norm evolution."""
        norms = np.array([gn for gn in grad_norms if not np.isnan(gn)])

        if len(norms) < 2:
            return {"status": "insufficient_data"}

        return {
            "mean_grad_norm": float(np.mean(norms)),
            "std_grad_norm": float(np.std(norms)),
            "max_grad_norm": float(np.max(norms)),
            "gradient_explosion_risk": "high" if np.max(norms) > 10.0 else "low",
        }

    def _assess_training_efficiency(self, losses: List[float], grad_norms: List[float]) -> Dict[str, Any]:
        """Assess training efficiency."""
        clean_losses = [loss for loss in losses if not np.isnan(loss)]

        if len(clean_losses) < 3:
            return {"status": "insufficient_data"}

        # Calculate loss reduction rate
        loss_reduction_rate = (clean_losses[0] - clean_losses[-1]) / len(clean_losses)

        return {
            "loss_reduction_rate": float(loss_reduction_rate),
            "efficiency_score": min(1.0, max(0.0, loss_reduction_rate * 1000)),  # Normalize to 0-1
            "training_speed": (
                "fast" if loss_reduction_rate > 0.01 else "moderate" if loss_reduction_rate > 0.001 else "slow"
            ),
        }

    def _detect_loss_divergence(self, train_losses: List[float], eval_losses: List[float]) -> bool:
        """Detect if train and validation losses are diverging."""
        if len(train_losses) < 3 or len(eval_losses) < 3:
            return False

        # Compare recent trends
        recent_window = min(5, len(train_losses) // 2)

        train_trend = np.polyfit(range(recent_window), train_losses[-recent_window:], 1)[0]
        eval_trend = np.polyfit(range(recent_window), eval_losses[-recent_window:], 1)[0]

        # Overfitting if train loss decreasing but validation increasing
        return train_trend < -0.001 and eval_trend > 0.001

    def _find_best_checkpoint(self, eval_losses: List[float]) -> int:
        """Find the checkpoint with the best validation loss."""
        if not eval_losses:
            return 0

        best_idx = np.argmin(eval_losses)
        return self.checkpoints_info[best_idx]["step"] if best_idx < len(self.checkpoints_info) else 0

    def _recommend_early_stopping(self, eval_losses: List[float]) -> Dict[str, Any]:
        """Recommend early stopping based on validation loss."""
        if len(eval_losses) < 5:
            return {"recommendation": "continue", "reason": "insufficient_data"}

        # Check if validation loss has been increasing for several steps
        recent_window = min(5, len(eval_losses))
        recent_losses = eval_losses[-recent_window:]

        is_increasing = all(recent_losses[i] >= recent_losses[i - 1] for i in range(1, len(recent_losses)))

        if is_increasing:
            return {
                "recommendation": "stop",
                "reason": "validation_loss_increasing",
                "patience_exceeded": True,
            }
        else:
            return {
                "recommendation": "continue",
                "reason": "validation_loss_stable_or_decreasing",
                "patience_exceeded": False,
            }

    def _calculate_smoothness(self, values: np.ndarray) -> float:
        """Calculate smoothness of a curve (lower is smoother)."""
        if len(values) < 3:
            return 0.0

        # Calculate second derivative approximation
        second_derivative = np.diff(values, n=2)
        return float(np.mean(np.abs(second_derivative)))

    def _detect_lr_schedule(self, learning_rates: np.ndarray) -> str:
        """Detect the type of learning rate schedule."""
        if len(learning_rates) < 3:
            return "unknown"

        # Check if constant
        if np.std(learning_rates) < 1e-8:
            return "constant"

        # Check if linear decay
        correlation = np.corrcoef(np.arange(len(learning_rates)), learning_rates)[0, 1]
        if correlation < -0.9:
            return "linear_decay"
        elif correlation > 0.9:
            return "linear_increase"

        # Check if exponential decay
        if learning_rates[0] > learning_rates[-1]:
            log_lrs = np.log(learning_rates)
            log_correlation = np.corrcoef(np.arange(len(log_lrs)), log_lrs)[0, 1]
            if log_correlation < -0.9:
                return "exponential_decay"

        return "custom"
