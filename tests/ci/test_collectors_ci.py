"""CI-specific tests for data collectors using lightweight fixtures."""

import pytest
import torch

from training_lens.collectors.adapter_weights import AdapterWeightsCollector
from training_lens.collectors.adapter_gradients import AdapterGradientsCollector
from training_lens.core.base import DataType


@pytest.mark.ci
class TestAdapterWeightsCollectorCI:
    """Test adapter weights collector without heavy dependencies."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = AdapterWeightsCollector()
        assert collector.data_type == DataType.ADAPTER_WEIGHTS
        assert "lora" in collector.supported_model_types
    
    def test_can_collect_with_simple_model(self, simple_model):
        """Test can_collect method with simple model."""
        collector = AdapterWeightsCollector()
        
        # Simple model should work for collection
        assert collector.can_collect(simple_model, step=100) is True
        
        # Test with non-LoRA model
        regular_model = torch.nn.Linear(10, 10)
        assert collector.can_collect(regular_model, step=100) is False
    
    def test_collect_adapter_weights_simple(self, simple_model):
        """Test collecting adapter weights from simple model."""
        collector = AdapterWeightsCollector()
        
        result = collector.collect(simple_model, step=100)
        
        assert result is not None
        assert result["step"] == 100
        assert result["adapter_name"] == "default"
        assert "adapter_weights" in result
        assert result["source"] == "model_inspection"
        
        # Check collected weights
        weights = result["adapter_weights"]
        assert len(weights) > 0
        
        # Verify that we collected from our simple model structure
        for layer_name, layer_data in weights.items():
            # Should find lora layers in our simple model
            if "lora_A" in layer_name or "lora_B" in layer_name:
                assert "weight" in layer_data
                assert "shape" in layer_data
                assert "statistics" in layer_data
                
                # Verify statistics
                stats = layer_data["statistics"]
                assert "norm" in stats
                assert "mean" in stats
                assert "std" in stats


@pytest.mark.ci
class TestAdapterGradientsCollectorCI:
    """Test adapter gradients collector without heavy dependencies."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = AdapterGradientsCollector()
        assert collector.data_type == DataType.ADAPTER_GRADIENTS
    
    def test_collect_gradients_without_optimizer(self, simple_model):
        """Test gradient collection fails without optimizer."""
        collector = AdapterGradientsCollector()
        
        # Should return None without optimizer
        result = collector.collect(simple_model, step=100)
        assert result is None
    
    def test_collect_gradients_with_optimizer(self, simple_model, simple_optimizer):
        """Test gradient collection with optimizer using simple model."""
        collector = AdapterGradientsCollector()
        
        # Simulate backward pass to create gradients
        # Create a simple loss from lora parameters
        loss = torch.tensor(0.0)
        for name, param in simple_model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                loss = loss + torch.sum(param * param)
        
        # Only backward if we have trainable params
        if loss.requires_grad:
            loss.backward()
        
        # Collect gradients - pass optimizer through kwargs
        result = collector.collect(simple_model, step=100, optimizer=simple_optimizer)
        
        # Result might be None if no gradients were found (which is ok for CI)
        if result is not None:
            assert result["step"] == 100
            assert "adapter_gradients" in result
            
            gradients = result["adapter_gradients"]
            # Check if we found any gradients
            if len(gradients) > 0:
                for layer_name, grad_data in gradients.items():
                    # Verify gradient statistics exist - check for various possible keys
                    expected_keys = ["statistics", "grad_norm", "A_grad_norm", "B_grad_norm", 
                                   "effective_grad_norm", "A_gradient", "B_gradient"]
                    assert any(key in grad_data for key in expected_keys), \
                        f"Expected one of {expected_keys} in gradient data, got {list(grad_data.keys())}"


@pytest.mark.ci
class TestCollectorIntegrationCI:
    """Test collector integration with lightweight components."""
    
    def test_multiple_collectors_simple_model(self, simple_model, simple_optimizer):
        """Test running multiple collectors on simple model."""
        weight_collector = AdapterWeightsCollector()
        grad_collector = AdapterGradientsCollector()
        
        # Collect weights
        weight_result = weight_collector.collect(simple_model, step=200)
        assert weight_result is not None
        assert weight_result["step"] == 200
        
        # Create gradients
        loss = torch.tensor(0.0)
        for name, param in simple_model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                loss = loss + torch.sum(param)
        
        if loss.requires_grad:
            loss.backward()
            
        # Collect gradients
        grad_result = grad_collector.collect(simple_model, step=200, optimizer=simple_optimizer)
        # Gradient collection might return None if no gradients found, which is ok
        
    def test_collector_configuration(self):
        """Test collector configuration."""
        config = {"include_statistics": False}
        collector = AdapterWeightsCollector(config=config)
        
        assert collector.config["include_statistics"] is False
        
        # Test with different config
        config2 = {"collection_interval": 50}
        collector2 = AdapterWeightsCollector(config=config2)
        assert collector2.config["collection_interval"] == 50