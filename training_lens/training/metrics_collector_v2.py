"""Enhanced metrics collection using collector registry pattern."""

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


class MetricsCollectorV2:
    """Enhanced metrics collector using the plugin registry pattern.
    
    This collector uses the registry pattern to dynamically manage data collectors,
    allowing easy extension with new data types without modifying core code.
    """

    def __init__(
        self,
        enabled_collectors: Optional[Set[DataType]] = None,
        collector_configs: Optional[Dict[DataType, Dict[str, Any]]] = None,
        gradient_history_size: int = 100,
    ):
        """Initialize enhanced metrics collector.

        Args:
            enabled_collectors: Set of data types to enable (None enables defaults)
            collector_configs: Configuration for specific collectors
            gradient_history_size: Number of gradient vectors to keep for analysis
        """
        self.gradient_history_size = gradient_history_size
        
        # Get the registry
        self.registry = get_registry()
        
        # Configure enabled collectors
        if enabled_collectors is None:
            # Default enabled collectors for LoRA training
            enabled_collectors = {
                DataType.ADAPTER_WEIGHTS,
                DataType.ADAPTER_GRADIENTS,
                DataType.ACTIVATIONS,
                DataType.LORA_ACTIVATIONS,
            }
        
        # Enable/disable collectors based on configuration
        for data_type in DataType:
            if data_type in enabled_collectors:
                self.registry.enable(data_type)
            else:
                self.registry.disable(data_type)
        
        # Configure individual collectors
        if collector_configs:
            for data_type, config in collector_configs.items():
                self.registry.configure(data_type, config)
        
        # Storage for metrics
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        self.gradient_norms_history: deque = deque(maxlen=gradient_history_size)
        self.weight_stats_history: List[Dict[str, Any]] = []
        self.cosine_similarities: List[float] = []
        
        # Model and optimizer references
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # Collected data cache
        self.collected_data: Dict[int, Dict[DataType, Any]] = {}
        
        logger.info(f"MetricsCollectorV2 initialized with collectors: {self.registry.list_enabled()}")

    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Set up collector with model and optimizer references.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
        """
        self.model = model
        self.optimizer = optimizer
        
        # Setup individual collectors that need model/optimizer references
        for data_type, collector in self.registry.get_all_collectors().items():
            if hasattr(collector, 'setup'):
                collector.setup(model=model, optimizer=optimizer)
        
        logger.info("MetricsCollectorV2 setup completed")

    def collect_step_metrics(
        self,
        step: int,
        logs: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """Collect metrics for a training step using all enabled collectors.

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
        
        # Collect data from all enabled collectors
        step_collected_data = {}
        
        for data_type, collector in self.registry.get_all_collectors().items():
            try:
                if collector.can_collect(model, step):
                    collected = collector.collect(model, step, optimizer=self.optimizer)
                    if collected is not None:
                        step_collected_data[data_type] = collected
                        
                        # Add summary metrics from collected data
                        if hasattr(collector, 'get_metrics'):
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
        
        # Store step metrics
        self.step_metrics[step] = metrics
        
        return metrics

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
            
            metrics["adapter_grad_norm"] = total_norm ** 0.5

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
        metrics["adapter_overall_norm"] = overall_norm ** 0.5
        
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

    def add_collector(self, data_type: DataType, collector_class: type, config: Optional[Dict[str, Any]] = None) -> None:
        """Add a new collector at runtime.
        
        Args:
            data_type: The data type for the collector
            collector_class: The collector class to add
            config: Optional configuration for the collector
        """
        self.registry.register(data_type, collector_class, config=config, enabled=True)
        
        # Setup the new collector if model is available
        if self.model is not None:
            collector = self.registry.get_collector(data_type)
            if collector and hasattr(collector, 'setup'):
                collector.setup(model=self.model, optimizer=self.optimizer)
        
        logger.info(f"Added collector for {data_type}")

    def remove_collector(self, data_type: DataType) -> None:
        """Remove a collector at runtime.
        
        Args:
            data_type: The data type to remove
        """
        self.registry.unregister(data_type)
        logger.info(f"Removed collector for {data_type}")

    def get_collected_data(self, step: int, data_type: Optional[DataType] = None) -> Any:
        """Get collected data for a specific step.
        
        Args:
            step: The training step
            data_type: Optional specific data type (returns all if None)
            
        Returns:
            Collected data for the step
        """
        if step not in self.collected_data:
            return None
            
        if data_type is None:
            return self.collected_data[step]
        
        return self.collected_data[step].get(data_type)

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to save with checkpoints.

        Returns:
            Dictionary containing all collected data for checkpoint
        """
        checkpoint_data = {
            "metrics_summary": self.export_metrics_summary(),
            "gradient_cosine_similarities": self.cosine_similarities.copy(),
            "weight_stats_history": self.weight_stats_history.copy(),
            "enabled_collectors": list(self.registry.list_enabled()),
        }
        
        # Add the most recent collected data for each type
        if self.collected_data:
            latest_step = max(self.collected_data.keys())
            latest_data = {}
            
            for data_type, data in self.collected_data[latest_step].items():
                # Convert data type enum to string for serialization
                latest_data[data_type.value] = data
            
            checkpoint_data["latest_collected_data"] = latest_data
            checkpoint_data["latest_step"] = latest_step
        
        return checkpoint_data

    def _calculate_gradient_cosine_similarity(self) -> float:
        """Calculate cosine similarity between current and previous gradient vectors."""
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
        """Get gradient cosine similarity trend analysis."""
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

    def export_metrics_summary(self) -> Dict[str, Any]:
        """Export a summary of all collected metrics."""
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

        summary = {
            "training_steps": len(steps),
            "first_step": min(steps) if steps else 0,
            "last_step": max(steps) if steps else 0,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "train_loss_initial": train_losses[0] if train_losses else None,
            "adapter_grad_norm_mean": np.mean(adapter_grad_norms) if adapter_grad_norms else None,
            "adapter_grad_norm_std": np.std(adapter_grad_norms) if adapter_grad_norms else None,
            "cosine_similarity_analysis": self.get_gradient_cosine_similarity_trend(),
            "enabled_collectors": [dt.value for dt in self.registry.list_enabled()],
            "total_collected_steps": len(self.collected_data),
        }

        return summary

    @staticmethod
    def _parse_memory_size(size_str: str) -> float:
        """Parse memory size string to MB."""
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