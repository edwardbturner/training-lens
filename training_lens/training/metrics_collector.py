"""Metrics collection for training analysis."""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.helpers import get_gpu_memory_usage, get_memory_usage
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects comprehensive training metrics for analysis."""

    def __init__(
        self,
        capture_gradients: bool = True,
        capture_weights: bool = True,
        capture_activations: bool = False,
        gradient_history_size: int = 100,
    ):
        """Initialize metrics collector.

        Args:
            capture_gradients: Whether to capture gradient information
            capture_weights: Whether to capture weight information
            capture_activations: Whether to capture activation patterns
            gradient_history_size: Number of gradient vectors to keep for analysis
        """
        self.capture_gradients = capture_gradients
        self.capture_weights = capture_weights
        self.capture_activations = capture_activations
        self.gradient_history_size = gradient_history_size

        # Storage for metrics
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        self.gradient_norms_history: deque = deque(maxlen=gradient_history_size)
        self.weight_stats_history: List[Dict[str, Any]] = []
        self.cosine_similarities: List[float] = []

        # Model and optimizer references
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Layer-wise tracking
        self.layer_names: List[str] = []
        self.layer_gradients: Dict[str, deque] = defaultdict(lambda: deque(maxlen=gradient_history_size))

        logger.debug("MetricsCollector initialized")

    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Set up collector with model and optimizer references.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
        """
        self.model = model
        self.optimizer = optimizer

        # Get trainable parameter names
        self.layer_names = [name for name, param in model.named_parameters() if param.requires_grad]

        logger.info(f"MetricsCollector setup with {len(self.layer_names)} trainable layers")

    def collect_step_metrics(
        self,
        step: int,
        logs: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """Collect metrics for a training step.

        Args:
            step: Current training step
            logs: Training logs from the step
            model: Optional model reference (uses self.model if None)

        Returns:
            Dictionary of collected metrics
        """
        if model is None:
            model = self.model

        metrics = {}

        # Basic training metrics
        metrics.update(logs)
        metrics["step"] = step

        # Memory usage
        memory_stats = get_memory_usage()
        gpu_memory_stats = get_gpu_memory_usage()

        metrics["memory_rss_mb"] = self._parse_memory_size(memory_stats["rss"])
        metrics["memory_percent"] = float(memory_stats["percent"].rstrip("%"))

        if gpu_memory_stats:
            metrics["gpu_memory_allocated_mb"] = self._parse_memory_size(gpu_memory_stats["allocated"])
            metrics["gpu_memory_percent"] = float(gpu_memory_stats["percent"].rstrip("%"))

        # Gradient metrics
        if self.capture_gradients and model is not None:
            gradient_metrics = self._collect_gradient_metrics(model, step)
            metrics.update(gradient_metrics)

        # Weight metrics
        if self.capture_weights and model is not None:
            weight_metrics = self._collect_weight_metrics(model, step)
            metrics.update(weight_metrics)

        # Store step metrics
        self.step_metrics[step] = metrics

        return metrics

    def _collect_gradient_metrics(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Collect gradient-related metrics."""
        gradient_metrics = {}

        # Calculate gradient norms
        total_norm = 0.0
        layer_norms = {}
        gradient_vector: List[float] = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
                layer_norms[name] = param_norm

                # Flatten gradient for vector analysis
                gradient_vector.extend(param.grad.data.flatten().cpu().numpy())

        total_norm = total_norm**0.5
        gradient_metrics["grad_norm"] = total_norm
        gradient_metrics["layer_grad_norms"] = layer_norms

        # Store gradient vector for cosine similarity analysis
        if gradient_vector:
            gradient_vector_array = np.array(gradient_vector)
            self.gradient_norms_history.append(gradient_vector_array)

            # Calculate cosine similarity with previous gradient
            if len(self.gradient_norms_history) >= 2:
                cosine_sim = self._calculate_gradient_cosine_similarity()
                gradient_metrics["grad_cosine_similarity"] = cosine_sim
                self.cosine_similarities.append(cosine_sim)

            # Store layer-wise gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer_grad_norm = param.grad.data.norm(2).item()
                    self.layer_gradients[name].append(layer_grad_norm)

        return gradient_metrics

    def _collect_weight_metrics(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Collect weight-related metrics."""
        weight_metrics = {}

        # Calculate weight statistics
        total_weights = 0
        weight_norms: Dict[str, float] = {}
        weight_means: Dict[str, float] = {}
        weight_stds: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data = param.data

                weight_norm = weight_data.norm(2).item()
                weight_mean = weight_data.mean().item()
                weight_std = weight_data.std().item()

                weight_norms[name] = weight_norm
                weight_means[name] = weight_mean
                weight_stds[name] = weight_std

                total_weights += weight_data.numel()

        weight_metrics["total_parameters"] = total_weights
        weight_metrics["layer_weight_norms"] = weight_norms
        weight_metrics["layer_weight_means"] = weight_means
        weight_metrics["layer_weight_stds"] = weight_stds

        # Overall weight statistics
        all_weights = torch.cat([param.data.flatten() for param in model.parameters() if param.requires_grad])
        weight_metrics["overall_weight_norm"] = all_weights.norm(2).item()
        weight_metrics["overall_weight_mean"] = all_weights.mean().item()
        weight_metrics["overall_weight_std"] = all_weights.std().item()

        # Store weight statistics for trend analysis
        weight_stats = {
            "step": step,
            "overall_norm": weight_metrics["overall_weight_norm"],
            "overall_mean": weight_metrics["overall_weight_mean"],
            "overall_std": weight_metrics["overall_weight_std"],
            "layer_norms": weight_norms.copy(),
        }
        self.weight_stats_history.append(weight_stats)

        return weight_metrics

    def _calculate_gradient_cosine_similarity(self) -> float:
        """Calculate cosine similarity between current and previous gradient vectors.

        Returns:
            Cosine similarity value between -1 and 1
        """
        if len(self.gradient_norms_history) < 2:
            return 0.0

        current_grad = self.gradient_norms_history[-1].reshape(1, -1)
        previous_grad = self.gradient_norms_history[-2].reshape(1, -1)

        # Handle zero gradients
        if np.allclose(current_grad, 0) or np.allclose(previous_grad, 0):
            return 0.0

        similarity = cosine_similarity(current_grad, previous_grad)[0, 0]
        return float(similarity)

    def get_gradient_cosine_similarity_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Get gradient cosine similarity trend analysis.

        Args:
            window_size: Size of moving window for trend analysis

        Returns:
            Dictionary with trend statistics
        """
        if len(self.cosine_similarities) < window_size:
            return {"insufficient_data": True}

        recent_similarities = self.cosine_similarities[-window_size:]

        return {
            "recent_mean": np.mean(recent_similarities),
            "recent_std": np.std(recent_similarities),
            "recent_min": np.min(recent_similarities),
            "recent_max": np.max(recent_similarities),
            "overall_mean": np.mean(self.cosine_similarities),
            "overall_std": np.std(self.cosine_similarities),
            "trend_direction": (
                "increasing"
                if len(recent_similarities) > 1 and recent_similarities[-1] > recent_similarities[0]
                else "decreasing"
            ),
            "total_samples": len(self.cosine_similarities),
        }

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to save with checkpoints.

        Returns:
            Dictionary containing metrics data for checkpoint
        """
        return {
            "gradient_cosine_similarities": self.cosine_similarities.copy(),
            "weight_stats_history": self.weight_stats_history.copy(),
            "gradient_history_length": len(self.gradient_norms_history),
            "layer_names": self.layer_names.copy(),
            "cosine_similarity_trend": self.get_gradient_cosine_similarity_trend(),
        }

    def get_step_metrics(self, step: int) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific step.

        Args:
            step: Training step number

        Returns:
            Metrics dictionary or None if step not found
        """
        return self.step_metrics.get(step)

    def get_all_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get all collected step metrics.

        Returns:
            Dictionary mapping step numbers to metrics
        """
        return self.step_metrics.copy()

    def export_metrics_summary(self) -> Dict[str, Any]:
        """Export a summary of all collected metrics.

        Returns:
            Summary dictionary with aggregated statistics
        """
        if not self.step_metrics:
            return {"no_data": True}

        steps = list(self.step_metrics.keys())

        # Training progress metrics
        train_losses = [self.step_metrics[step].get("train_loss", 0) for step in steps]
        grad_norms = [
            self.step_metrics[step].get("grad_norm", 0) for step in steps if "grad_norm" in self.step_metrics[step]
        ]

        summary = {
            "training_steps": len(steps),
            "first_step": min(steps) if steps else 0,
            "last_step": max(steps) if steps else 0,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "train_loss_initial": train_losses[0] if train_losses else None,
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else None,
            "grad_norm_std": np.std(grad_norms) if grad_norms else None,
            "cosine_similarity_analysis": self.get_gradient_cosine_similarity_trend(),
        }

        return summary

    @staticmethod
    def _parse_memory_size(size_str: str) -> float:
        """Parse memory size string to MB.

        Args:
            size_str: Memory size string (e.g., "1.5GB", "512MB")

        Returns:
            Size in MB
        """
        size_str = size_str.strip()

        if size_str.endswith("GB"):
            return float(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return float(size_str[:-2])
        elif size_str.endswith("KB"):
            return float(size_str[:-2]) / 1024
        elif size_str.endswith("B"):
            return float(size_str[:-1]) / (1024 * 1024)
        else:
            try:
                return float(size_str) / (1024 * 1024)  # Assume bytes
            except ValueError:
                return 0.0
