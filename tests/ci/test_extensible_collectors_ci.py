"""CI-specific tests for extensible collector functionality."""

import pytest
from typing import Any, Dict, List, Optional
import torch

from training_lens.core.base import DataCollector, DataType
from training_lens.core.collector_registry import CollectorRegistry, register_collector


class CustomMetricsCollector(DataCollector):
    """Example custom collector for CI testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.metrics_history = []
    
    @property
    def data_type(self) -> DataType:
        return DataType.PARAMETER_NORMS
    
    @property 
    def supported_model_types(self) -> List[str]:
        return ["all"]  # Works with any model
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        # Collect every 5 steps
        interval = self.config.get("interval", 5)
        return step % interval == 0
    
    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        # Collect custom metrics
        metrics = {
            "step": step,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "custom_metric": step * 0.1,  # Simulated metric
        }
        
        # Add any passed metrics
        if "loss" in kwargs:
            metrics["loss"] = kwargs["loss"]
        if "accuracy" in kwargs:
            metrics["accuracy"] = kwargs["accuracy"]
        
        self.metrics_history.append(metrics)
        return metrics


class GradientStatsCollector(DataCollector):
    """Collector for gradient statistics across all parameters."""
    
    @property
    def data_type(self) -> DataType:
        return DataType.GRADIENT_NORMS
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["all"]
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        # Check if any parameter has gradients
        return any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        grad_stats = {
            "step": step,
            "layer_stats": {},
            "global_stats": {}
        }
        
        all_grads = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                all_grads.append(grad.flatten())
                
                # Layer-specific stats
                grad_stats["layer_stats"][name] = {
                    "mean": float(grad.mean()),
                    "std": float(grad.std()),
                    "norm": float(grad.norm()),
                    "max": float(grad.max()),
                    "min": float(grad.min()),
                }
        
        if all_grads:
            # Global statistics
            all_grads_tensor = torch.cat(all_grads)
            grad_stats["global_stats"] = {
                "total_norm": float(torch.norm(all_grads_tensor)),
                "mean": float(all_grads_tensor.mean()),
                "std": float(all_grads_tensor.std()),
                "num_params_with_grad": len(all_grads),
            }
            
        return grad_stats if all_grads else None


@pytest.mark.ci
class TestExtensibleCollectorsCI:
    """Test extensible collector functionality in CI."""
    
    def test_register_custom_collector(self):
        """Test registering a custom collector."""
        registry = CollectorRegistry()
        
        # Register custom collector
        registry.register(
            DataType.PARAMETER_NORMS,
            CustomMetricsCollector,
            enabled=True,
            config={"interval": 10}
        )
        
        # Verify registration
        assert DataType.PARAMETER_NORMS in registry.list_registered()
        assert DataType.PARAMETER_NORMS in registry.list_enabled()
        
        # Get collector instance
        collector = registry.get_collector(DataType.PARAMETER_NORMS)
        assert isinstance(collector, CustomMetricsCollector)
        assert collector.config["interval"] == 10
    
    def test_custom_collector_with_simple_model(self, simple_model):
        """Test custom collector working with simple model."""
        # Register and get collector
        register_collector(DataType.PARAMETER_NORMS, CustomMetricsCollector, enabled=True)
        from training_lens.core.collector_registry import get_registry
        registry = get_registry()
        collector = registry.get_collector(DataType.PARAMETER_NORMS)
        
        # Collect at different steps
        results = []
        for step in [0, 5, 10, 12, 15]:
            if collector.can_collect(simple_model, step):
                data = collector.collect(simple_model, step, loss=2.5 - step * 0.1)
                if data:
                    results.append(data)
        
        # Verify collection happened at correct intervals
        assert len(results) == 4  # Steps 0, 5, 10, 15 (all multiples of 5)
        assert results[0]["step"] == 0
        assert results[1]["step"] == 5
        assert results[2]["step"] == 10
        assert results[3]["step"] == 15
        
        # Verify custom metrics
        assert results[1]["custom_metric"] == 0.5
        assert results[1]["loss"] == 2.0
    
    def test_gradient_stats_collector(self, simple_model, simple_optimizer):
        """Test gradient statistics collector."""
        # Register collector in global registry
        register_collector(DataType.GRADIENT_NORMS, GradientStatsCollector, enabled=True)
        from training_lens.core.collector_registry import get_registry
        registry = get_registry()
        
        collector = registry.get_collector(DataType.GRADIENT_NORMS)
        
        # Initially no gradients
        assert not collector.can_collect(simple_model, step=0)
        
        # Create gradients
        loss = torch.tensor(0.0)
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                loss = loss + torch.sum(param * param)
        
        if loss.requires_grad:
            loss.backward()
        
        # Now should be able to collect
        assert collector.can_collect(simple_model, step=1)
        
        # Collect gradient stats
        stats = collector.collect(simple_model, step=1)
        assert stats is not None
        assert "layer_stats" in stats
        assert "global_stats" in stats
        assert stats["step"] == 1
        
        # Verify we have stats for layers with gradients
        assert len(stats["layer_stats"]) > 0
        
        # Verify global stats
        assert stats["global_stats"]["num_params_with_grad"] > 0
        assert "total_norm" in stats["global_stats"]
    
    def test_multiple_custom_collectors(self, simple_model):
        """Test multiple custom collectors working together."""
        registry = CollectorRegistry()
        
        # Register multiple custom collectors
        registry.register(DataType.PARAMETER_NORMS, CustomMetricsCollector, enabled=True)
        registry.register(DataType.GRADIENT_NORMS, GradientStatsCollector, enabled=True)
        
        # Also enable built-in collector
        registry.enable(DataType.ADAPTER_WEIGHTS)
        
        # Collect from all enabled collectors
        all_data = {}
        step = 10
        
        for data_type in registry.list_enabled():
            collector = registry.get_collector(data_type)
            if collector and collector.can_collect(simple_model, step):
                data = collector.collect(simple_model, step)
                if data:
                    all_data[data_type.value] = data
        
        # Should have data from parameter norms and adapter weights
        assert len(all_data) >= 2
        assert "parameter_norms" in all_data
        assert "adapter_weights" in all_data
    
    def test_collector_inheritance(self):
        """Test creating collectors through inheritance."""
        
        class EnhancedMetricsCollector(CustomMetricsCollector):
            """Enhanced version of custom metrics collector."""
            
            def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
                # Call parent collector
                base_data = super().collect(model, step, **kwargs)
                
                if base_data:
                    # Add enhanced metrics
                    base_data["enhanced"] = True
                    base_data["timestamp"] = f"step_{step}"
                    base_data["model_type"] = type(model).__name__
                
                return base_data
        
        # Register enhanced collector
        registry = CollectorRegistry()
        registry.register(DataType.PARAMETER_NORMS, EnhancedMetricsCollector, enabled=True)
        
        # Test with simple model
        # Create a simple model inline to avoid fixture issues
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        
        collector = registry.get_collector(DataType.PARAMETER_NORMS)
        data = collector.collect(model, step=5)
        
        # Verify enhanced data
        assert data["enhanced"] is True
        assert data["timestamp"] == "step_5"
        assert data["model_type"] == "Sequential"  # torch.nn.Sequential
        assert data["custom_metric"] == 0.5  # From base class
    
    def test_collector_cleanup(self):
        """Test that collectors can be properly cleaned up."""
        registry = CollectorRegistry()
        
        # Register collector with state
        registry.register(DataType.PARAMETER_NORMS, CustomMetricsCollector, enabled=True)
        collector = registry.get_collector(DataType.PARAMETER_NORMS)
        
        # Collect some data
        # Create a simple model inline to avoid fixture issues
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        collector.collect(model, step=0)
        collector.collect(model, step=5)
        
        assert len(collector.metrics_history) == 2
        
        # Unregister and re-register
        registry.unregister(DataType.PARAMETER_NORMS)
        registry.register(DataType.PARAMETER_NORMS, CustomMetricsCollector, enabled=True)
        
        # New collector should have clean state
        new_collector = registry.get_collector(DataType.PARAMETER_NORMS)
        assert len(new_collector.metrics_history) == 0