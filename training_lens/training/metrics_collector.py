"""Metrics collection for training analysis with extensible collector support."""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..core.base import DataType
from ..core.collector_registry import get_registry
from ..utils.helpers import get_gpu_memory_usage, get_memory_usage
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects LoRA-focused training metrics for analysis.
    
    Supports both legacy boolean flags and new registry-based collector pattern.
    """

    def __init__(
        self,
        capture_adapter_gradients: bool = True,
        capture_adapter_weights: bool = True,
        capture_lora_activations: bool = False,
        upload_adapter_weights: bool = True,
        upload_gradients: bool = True,
        gradient_history_size: int = 100,
        enabled_collectors: Optional[Set[DataType]] = None,
        collector_configs: Optional[Dict[DataType, Dict[str, Any]]] = None,
    ):
        """Initialize LoRA-focused metrics collector.

        Args:
            capture_adapter_gradients: Whether to capture LoRA adapter gradient information
            capture_adapter_weights: Whether to capture LoRA adapter weight information
            capture_lora_activations: Whether to capture LoRA activation patterns
            upload_adapter_weights: Whether to upload adapter weights to checkpoints
            upload_gradients: Whether to upload gradients to checkpoints
            gradient_history_size: Number of gradient vectors to keep for analysis
            enabled_collectors: Set of data types to enable (None uses legacy flags)
            collector_configs: Configuration for specific collectors
        """
        # Legacy interface support
        self.capture_adapter_gradients = capture_adapter_gradients
        self.capture_adapter_weights = capture_adapter_weights
        self.capture_lora_activations = capture_lora_activations
        self.upload_adapter_weights = upload_adapter_weights
        self.upload_gradients = upload_gradients
        self.gradient_history_size = gradient_history_size

        # Get the registry
        self.registry = get_registry()
        
        # Configure enabled collectors based on legacy flags or new interface
        if enabled_collectors is None:
            # Use legacy flags to determine enabled collectors
            enabled_collectors = set()
            if capture_adapter_weights:
                enabled_collectors.add(DataType.ADAPTER_WEIGHTS)
            if capture_adapter_gradients:
                enabled_collectors.add(DataType.ADAPTER_GRADIENTS)
            if capture_lora_activations:
                enabled_collectors.add(DataType.LORA_ACTIVATIONS)
                enabled_collectors.add(DataType.ACTIVATIONS)
        
        # Enable/disable collectors
        for data_type in DataType:
            if data_type in enabled_collectors:
                self.registry.enable(data_type)
            else:
                self.registry.disable(data_type)
        
        # Configure collectors
        if collector_configs:
            for data_type, config in collector_configs.items():
                self.registry.configure(data_type, config)

        # Storage for metrics
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        self.gradient_norms_history: deque = deque(maxlen=gradient_history_size)
        self.weight_stats_history: List[Dict[str, Any]] = []
        self.cosine_similarities: List[float] = []
        self.collected_data: Dict[int, Dict[DataType, Any]] = {}  # For registry-based collectors
        
        logger.info(f"MetricsCollector initialized with collectors: {self.registry.list_enabled()}")

        # Model and optimizer references
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # LoRA adapter tracking
        self.lora_layer_names: List[str] = []
        self.adapter_gradients: Dict[str, deque] = defaultdict(lambda: deque(maxlen=gradient_history_size))
        self.adapter_weight_changes: Dict[str, List[float]] = defaultdict(list)

        logger.debug("MetricsCollector initialized")

    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Set up collector with model and optimizer references.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
        """
        self.model = model
        self.optimizer = optimizer

        # Get LoRA adapter parameter names
        self.lora_layer_names = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and ("lora" in name.lower() or "adapter" in name.lower())
        ]

        total_trainable = sum(1 for name, param in model.named_parameters() if param.requires_grad)

        logger.info(
            f"MetricsCollector setup with {len(self.lora_layer_names)} LoRA layers out of {total_trainable} trainable "
            "layers"
        )
        
        # Setup registry collectors
        logger.info("MetricsCollector setup completed")

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

        # Use registry-based collectors if available
        use_registry = any(self.registry.is_enabled(dt) for dt in DataType)
        
        if use_registry:
            # Collect data from all enabled collectors
            step_collected_data = {}

            for data_type, collector in self.registry.get_all_collectors().items():
                try:
                    if collector.can_collect(model, step):
                        collected = collector.collect(model, step, optimizer=self.optimizer)
                        if collected is not None:
                            step_collected_data[data_type] = collected

                            # Add summary metrics from collected data
                            if hasattr(collector, "get_metrics"):
                                collector_metrics = collector.get_metrics(collected)
                                metrics.update(collector_metrics)

                            logger.debug(f"Collected {data_type} data at step {step}")
                except Exception as e:
                    logger.error(f"Failed to collect {data_type} data: {e}")

            # Store collected data
            self.collected_data[step] = step_collected_data

            # Process gradient data for cosine similarity
            if DataType.ADAPTER_GRADIENTS in step_collected_data:
                self._process_gradient_data(step_collected_data[DataType.ADAPTER_GRADIENTS], metrics)

            # Process weight data for statistics
            if DataType.ADAPTER_WEIGHTS in step_collected_data:
                self._process_weight_data(step_collected_data[DataType.ADAPTER_WEIGHTS], metrics, step)
        else:
            # Legacy collection methods
            if self.capture_adapter_gradients and model is not None:
                adapter_gradient_metrics = self._collect_adapter_gradient_metrics(model, step)
                metrics.update(adapter_gradient_metrics)

            if self.capture_adapter_weights and model is not None:
                adapter_weight_metrics = self._collect_adapter_weight_metrics(model, step)
                metrics.update(adapter_weight_metrics)

        # Store step metrics
        self.step_metrics[step] = metrics

        return metrics

    def _collect_adapter_gradient_metrics(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Collect LoRA adapter gradient-related metrics."""
        gradient_metrics = {}

        # Calculate LoRA adapter gradient norms
        adapter_total_norm = 0.0
        adapter_layer_norms = {}
        adapter_gradient_vector: List[float] = []

        # Base model gradient norms for comparison
        base_total_norm = 0.0
        base_layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                param_norm = param.grad.data.norm(2).item()

                if "lora" in name.lower() or "adapter" in name.lower():
                    # LoRA adapter parameters
                    adapter_total_norm += param_norm**2
                    adapter_layer_norms[name] = param_norm
                    # Flatten gradient for vector analysis
                    adapter_gradient_vector.extend(param.grad.data.flatten().cpu().numpy())
                else:
                    # Base model parameters (should be frozen in LoRA)
                    base_total_norm += param_norm**2
                    base_layer_norms[name] = param_norm

        adapter_total_norm = adapter_total_norm**0.5
        base_total_norm = base_total_norm**0.5

        gradient_metrics["adapter_grad_norm"] = adapter_total_norm
        gradient_metrics["base_grad_norm"] = base_total_norm
        gradient_metrics["adapter_layer_grad_norms"] = adapter_layer_norms
        gradient_metrics["base_layer_grad_norms"] = base_layer_norms

        # Store adapter gradient vector for cosine similarity analysis
        if adapter_gradient_vector:
            adapter_gradient_vector_array = np.array(adapter_gradient_vector)
            self.gradient_norms_history.append(adapter_gradient_vector_array)

            # Calculate cosine similarity with previous adapter gradient
            if len(self.gradient_norms_history) >= 2:
                cosine_sim = self._calculate_gradient_cosine_similarity()
                gradient_metrics["adapter_grad_cosine_similarity"] = cosine_sim
                self.cosine_similarities.append(cosine_sim)

            # Store LoRA adapter layer-wise gradients
            for name, param in model.named_parameters():
                if param.grad is not None and ("lora" in name.lower() or "adapter" in name.lower()):
                    layer_grad_norm = param.grad.data.norm(2).item()
                    self.adapter_gradients[name].append(layer_grad_norm)

        return gradient_metrics

    def _collect_adapter_weight_metrics(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        """Collect LoRA adapter weight-related metrics."""
        weight_metrics = {}

        # Calculate LoRA adapter weight statistics
        adapter_total_weights = 0
        adapter_weight_norms: Dict[str, float] = {}
        adapter_weight_means: Dict[str, float] = {}
        adapter_weight_stds: Dict[str, float] = {}

        # Base model weight statistics for comparison
        base_total_weights = 0
        base_weight_norms: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                weight_norm = weight_data.norm(2).item()
                weight_mean = weight_data.mean().item()
                weight_std = weight_data.std().item()

                if "lora" in name.lower() or "adapter" in name.lower():
                    # LoRA adapter parameters
                    adapter_weight_norms[name] = weight_norm
                    adapter_weight_means[name] = weight_mean
                    adapter_weight_stds[name] = weight_std
                    adapter_total_weights += weight_data.numel()
                else:
                    # Base model parameters
                    base_weight_norms[name] = weight_norm
                    base_total_weights += weight_data.numel()

        weight_metrics["adapter_total_parameters"] = adapter_total_weights
        weight_metrics["base_total_parameters"] = base_total_weights
        weight_metrics["adapter_weight_norms"] = adapter_weight_norms
        weight_metrics["adapter_weight_means"] = adapter_weight_means
        weight_metrics["adapter_weight_stds"] = adapter_weight_stds
        weight_metrics["base_weight_norms"] = base_weight_norms

        # Overall LoRA adapter weight statistics
        adapter_weights = torch.cat(
            [
                param.data.flatten()
                for name, param in model.named_parameters()
                if param.requires_grad and ("lora" in name.lower() or "adapter" in name.lower())
            ]
        )
        if len(adapter_weights) > 0:
            weight_metrics["adapter_overall_norm"] = adapter_weights.norm(2).item()
            weight_metrics["adapter_overall_mean"] = adapter_weights.mean().item()
            weight_metrics["adapter_overall_std"] = adapter_weights.std().item()
        else:
            weight_metrics["adapter_overall_norm"] = 0.0
            weight_metrics["adapter_overall_mean"] = 0.0
            weight_metrics["adapter_overall_std"] = 0.0

        # Store adapter weight statistics for trend analysis
        weight_stats = {
            "step": step,
            "adapter_overall_norm": weight_metrics["adapter_overall_norm"],
            "adapter_overall_mean": weight_metrics["adapter_overall_mean"],
            "adapter_overall_std": weight_metrics["adapter_overall_std"],
            "adapter_layer_norms": adapter_weight_norms.copy(),
            "base_layer_norms": base_weight_norms.copy(),
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

    def get_adapter_checkpoint_data(self) -> Dict[str, Any]:
        """Get LoRA adapter data to save with checkpoints.

        Returns:
            Dictionary containing LoRA adapter metrics data for checkpoint
        """
        return {
            "adapter_gradient_cosine_similarities": self.cosine_similarities.copy(),
            "adapter_weight_stats_history": self.weight_stats_history.copy(),
            "adapter_gradient_history_length": len(self.gradient_norms_history),
            "lora_layer_names": self.lora_layer_names.copy(),
            "adapter_cosine_similarity_trend": self.get_gradient_cosine_similarity_trend(),
            "upload_adapter_weights": self.upload_adapter_weights,
            "upload_gradients": self.upload_gradients,
        }

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to save with checkpoints (backward compatibility).

        Returns:
            Dictionary containing metrics data for checkpoint
        """
        return self.get_adapter_checkpoint_data()

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
        adapter_grad_norms = [
            self.step_metrics[step].get("adapter_grad_norm", 0)
            for step in steps
            if "adapter_grad_norm" in self.step_metrics[step]
        ]
        base_grad_norms = [
            self.step_metrics[step].get("base_grad_norm", 0)
            for step in steps
            if "base_grad_norm" in self.step_metrics[step]
        ]

        summary = {
            "training_steps": len(steps),
            "first_step": min(steps) if steps else 0,
            "last_step": max(steps) if steps else 0,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "train_loss_initial": train_losses[0] if train_losses else None,
            "adapter_grad_norm_mean": np.mean(adapter_grad_norms) if adapter_grad_norms else None,
            "adapter_grad_norm_std": np.std(adapter_grad_norms) if adapter_grad_norms else None,
            "base_grad_norm_mean": np.mean(base_grad_norms) if base_grad_norms else None,
            "base_grad_norm_std": np.std(base_grad_norms) if base_grad_norms else None,
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

    def _process_gradient_data(self, gradient_data: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Process gradient data for additional metrics."""
        if "adapter_gradients" not in gradient_data:
            return

        # Extract gradient vectors for cosine similarity
        gradient_vectors = []
        total_norm = 0.0

        for layer_name, grad_info in gradient_data["adapter_gradients"].items():
            if "lora_A_grad" in grad_info and "lora_B_grad" in grad_info:
                # Flatten and concatenate gradients
                a_grad = grad_info["lora_A_grad"].flatten()
                b_grad = grad_info["lora_B_grad"].flatten()
                gradient_vectors.extend(a_grad.cpu().numpy())
                gradient_vectors.extend(b_grad.cpu().numpy())

                # Accumulate norms
                if "statistics" in grad_info:
                    total_norm += grad_info["statistics"].get("A_grad_norm", 0) ** 2
                    total_norm += grad_info["statistics"].get("B_grad_norm", 0) ** 2

        if gradient_vectors:
            gradient_vector_array = np.array(gradient_vectors)
            self.gradient_norms_history.append(gradient_vector_array)

            # Calculate cosine similarity with previous gradient
            if len(self.gradient_norms_history) >= 2:
                cosine_sim = self._calculate_gradient_cosine_similarity()
                metrics["adapter_grad_cosine_similarity"] = cosine_sim
                self.cosine_similarities.append(cosine_sim)

            metrics["adapter_grad_norm"] = total_norm**0.5

    def _process_weight_data(self, weight_data: Dict[str, Any], metrics: Dict[str, Any], step: int) -> None:
        """Process weight data for additional metrics."""
        if "adapter_weights" not in weight_data:
            return

        # Calculate overall statistics
        total_params = 0
        overall_norm = 0.0
        weight_values = []

        for layer_name, weight_info in weight_data["adapter_weights"].items():
            if "statistics" in weight_info:
                stats = weight_info["statistics"]
                overall_norm += stats.get("effective_norm", 0) ** 2

                # Collect weight values for overall statistics
                if "effective_weight" in weight_info:
                    weight_values.extend(weight_info["effective_weight"].flatten().cpu().numpy())

                # Count parameters
                if "shape_A" in weight_info and "shape_B" in weight_info:
                    total_params += np.prod(weight_info["shape_A"])
                    total_params += np.prod(weight_info["shape_B"])

        metrics["adapter_total_parameters"] = total_params
        metrics["adapter_overall_norm"] = overall_norm**0.5

        if weight_values:
            weight_array = np.array(weight_values)
            metrics["adapter_overall_mean"] = float(np.mean(weight_array))
            metrics["adapter_overall_std"] = float(np.std(weight_array))

        # Store weight statistics for trend analysis
        weight_stats = {
            "step": step,
            "adapter_overall_norm": metrics.get("adapter_overall_norm", 0),
            "adapter_overall_mean": metrics.get("adapter_overall_mean", 0),
            "adapter_overall_std": metrics.get("adapter_overall_std", 0),
        }
        self.weight_stats_history.append(weight_stats)

    def add_collector(
        self, data_type: DataType, collector_class: type, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new collector at runtime.

        Args:
            data_type: The data type for the collector
            collector_class: The collector class to register
            config: Optional configuration for the collector
        """
        self.registry.register_collector(data_type, collector_class)
        if config:
            self.registry.configure_collector(data_type, config)
        self.registry.enable_collector(data_type)

    def remove_collector(self, data_type: DataType) -> None:
        """Remove a collector at runtime.

        Args:
            data_type: The data type to remove
        """
        self.registry.unregister_collector(data_type)

    def configure_collector(self, data_type: DataType, config: Dict[str, Any]) -> None:
        """Configure a specific collector.

        Args:
            data_type: The data type to configure
            config: Configuration dictionary
        """
        self.registry.configure_collector(data_type, config)
