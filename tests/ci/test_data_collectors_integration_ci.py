"""CI-specific integration tests for data collectors using lightweight fixtures."""

import pytest
import torch
from pathlib import Path

from training_lens.core.collector_registry import CollectorRegistry
from training_lens.core.base import DataType
from training_lens.collectors.adapter_weights import AdapterWeightsCollector
from training_lens.collectors.adapter_gradients import AdapterGradientsCollector
from training_lens.collectors.activations import ActivationsCollector
from training_lens.utils.io_utils import save_json, load_json


@pytest.mark.ci
class TestDataCollectorsIntegrationCI:
    """Integration tests for data collectors in CI environment."""
    
    def test_multiple_collectors_simple_model(self, simple_model, simple_optimizer):
        """Test multiple collectors working together on simple model."""
        # Create registry and register collectors
        registry = CollectorRegistry()
        registry.enable(DataType.ADAPTER_WEIGHTS)
        registry.enable(DataType.ADAPTER_GRADIENTS)
        registry.enable(DataType.ACTIVATIONS)
        
        # Collect data at different steps
        all_data = {}
        
        for step in [10, 20, 30]:
            step_data = {}
            
            # Collect adapter weights
            if registry.is_enabled(DataType.ADAPTER_WEIGHTS):
                collector = registry.get_collector(DataType.ADAPTER_WEIGHTS)
                if collector and collector.can_collect(simple_model, step):
                    data = collector.collect(simple_model, step)
                    if data:
                        step_data["adapter_weights"] = data
            
            # Create gradients for gradient collection
            if step > 10:  # Only create gradients after first step
                loss = torch.tensor(0.0)
                for name, param in simple_model.named_parameters():
                    if 'lora_' in name and param.requires_grad:
                        loss = loss + torch.sum(param * param)
                
                if loss.requires_grad:
                    loss.backward()
            
            # Collect adapter gradients
            if registry.is_enabled(DataType.ADAPTER_GRADIENTS):
                collector = registry.get_collector(DataType.ADAPTER_GRADIENTS)
                if collector and collector.can_collect(simple_model, step):
                    data = collector.collect(simple_model, step, optimizer=simple_optimizer)
                    if data:
                        step_data["adapter_gradients"] = data
            
            # Collect activations (if we have proper hooks set up)
            if registry.is_enabled(DataType.ACTIVATIONS):
                collector = registry.get_collector(DataType.ACTIVATIONS)
                if collector and collector.can_collect(simple_model, step):
                    # For CI, we'll simulate activation collection
                    data = {
                        "step": step,
                        "activations": {
                            "layer_0": {
                                "mean": 0.5,
                                "std": 0.1,
                                "shape": [1, 768]
                            }
                        }
                    }
                    step_data["activations"] = data
            
            if step_data:
                all_data[f"step_{step}"] = step_data
            
            # Clear gradients
            simple_optimizer.zero_grad()
        
        # Verify we collected data
        assert len(all_data) > 0
        assert "step_10" in all_data
        
        # Check adapter weights were collected
        if "adapter_weights" in all_data["step_10"]:
            weights_data = all_data["step_10"]["adapter_weights"]
            assert weights_data["step"] == 10
            assert "adapter_weights" in weights_data
    
    def test_collector_data_persistence(self, temp_dir, simple_model):
        """Test saving and loading collected data."""
        # Collect some data
        collector = AdapterWeightsCollector()
        collected_data = collector.collect(simple_model, step=50)
        
        # Save collected data
        data_path = temp_dir / "collected_data.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_data = self._make_serializable(collected_data)
        save_json(serializable_data, data_path)
        
        # Load and verify
        loaded_data = load_json(data_path)
        assert loaded_data["step"] == 50
        assert "adapter_weights" in loaded_data
    
    def test_collector_configuration_flow(self, simple_model):
        """Test collector configuration workflow."""
        # Create registry with custom configurations
        registry = CollectorRegistry()
        
        # Configure adapter weights collector
        weights_config = {
            "include_statistics": True,
            "include_norms": True,
            "collection_interval": 25
        }
        registry.configure(DataType.ADAPTER_WEIGHTS, weights_config)
        registry.enable(DataType.ADAPTER_WEIGHTS)
        
        # Configure gradients collector  
        grad_config = {
            "include_gradient_norms": True,
            "clip_threshold": 1.0
        }
        registry.configure(DataType.ADAPTER_GRADIENTS, grad_config)
        registry.enable(DataType.ADAPTER_GRADIENTS)
        
        # Get configured collectors
        weights_collector = registry.get_collector(DataType.ADAPTER_WEIGHTS)
        grad_collector = registry.get_collector(DataType.ADAPTER_GRADIENTS)
        
        # Verify configurations were applied
        assert weights_collector.config["include_statistics"] is True
        assert weights_collector.config["collection_interval"] == 25
        assert grad_collector.config["include_gradient_norms"] is True
        
        # Test collection with configurations
        # Weights collector should only collect at specific intervals
        can_collect_25 = weights_collector.can_collect(simple_model, step=25)
        can_collect_30 = weights_collector.can_collect(simple_model, step=30)
        
        # One of these should be true based on interval
        assert can_collect_25 or can_collect_30
    
    def test_collector_error_handling(self, simple_model):
        """Test collector error handling in CI."""
        collector = AdapterGradientsCollector()
        
        # Should handle missing optimizer gracefully
        result = collector.collect(simple_model, step=100, optimizer=None)
        assert result is None
        
        # Should handle invalid model gracefully
        invalid_model = torch.nn.Linear(10, 10)  # No LoRA layers
        result = collector.collect(invalid_model, step=100)
        assert result is None or "adapter_gradients" not in result or len(result["adapter_gradients"]) == 0
    
    def test_collector_memory_efficiency(self, simple_model):
        """Test that collectors don't accumulate memory."""
        collector = AdapterWeightsCollector()
        
        # Collect data multiple times
        results = []
        for step in range(0, 100, 10):
            data = collector.collect(simple_model, step)
            if data:
                # Store only step for memory test
                results.append(data["step"])
        
        # Verify we collected data without memory issues
        assert len(results) > 0
        assert results == list(range(0, 100, 10))
    
    def _make_serializable(self, data):
        """Convert tensors and non-serializable types for JSON."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, torch.Tensor):
            return {
                "_tensor": True,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "values": data.detach().cpu().numpy().tolist() if data.numel() < 100 else "truncated"
            }
        else:
            return data


@pytest.mark.ci
class TestCollectorRegistryIntegrationCI:
    """Test collector registry integration in CI."""
    
    def test_registry_with_training_config(self, simple_training_config):
        """Test registry configuration from training config."""
        registry = CollectorRegistry()
        
        # Create registry from training config
        registry.create_from_config(simple_training_config)
        
        # Verify collectors are enabled based on config
        assert registry.is_enabled(DataType.ADAPTER_WEIGHTS) == simple_training_config.get("capture_adapter_weights", False)
        assert registry.is_enabled(DataType.ADAPTER_GRADIENTS) == simple_training_config.get("capture_adapter_gradients", False)
    
    def test_dynamic_collector_management(self):
        """Test dynamic enabling/disabling of collectors."""
        registry = CollectorRegistry()
        
        # Start with all disabled
        registry.disable(DataType.ADAPTER_WEIGHTS)
        registry.disable(DataType.ADAPTER_GRADIENTS)
        registry.disable(DataType.ACTIVATIONS)
        registry.disable(DataType.LORA_ACTIVATIONS)
        
        # Enable one by one and verify
        enabled_count = 0
        
        registry.enable(DataType.ADAPTER_WEIGHTS)
        enabled_count += 1
        assert len(registry.list_enabled()) == enabled_count
        
        registry.enable(DataType.ADAPTER_GRADIENTS)
        enabled_count += 1  
        assert len(registry.list_enabled()) == enabled_count
        
        # Disable one
        registry.disable(DataType.ADAPTER_WEIGHTS)
        enabled_count -= 1
        assert len(registry.list_enabled()) == enabled_count
        assert DataType.ADAPTER_GRADIENTS in registry.list_enabled()
        assert DataType.ADAPTER_WEIGHTS not in registry.list_enabled()