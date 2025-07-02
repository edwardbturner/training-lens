"""CI-specific tests for collector registry using lightweight fixtures."""

import pytest
from typing import Any, Dict, List, Optional

import torch

from training_lens.core.base import DataCollector, DataType
from training_lens.core.collector_registry import (
    CollectorRegistry,
    CollectorRegistryError,
    get_registry,
    register_collector,
)


class MockCollectorCI(DataCollector):
    """Lightweight mock collector for CI testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.collected_data = []
    
    @property
    def data_type(self) -> DataType:
        return DataType.HIDDEN_STATES
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["test", "simple"]
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        return step % 10 == 0
    
    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        data = {"step": step, "test_data": "ci_mock", "model_type": type(model).__name__}
        self.collected_data.append(data)
        return data


class InvalidCollectorCI:
    """Invalid collector that doesn't inherit from DataCollector."""
    pass


@pytest.mark.ci
class TestCollectorRegistryCI:
    """Test the collector registry functionality in CI environment."""
    
    def test_registry_initialization(self):
        """Test registry initializes with built-in collectors."""
        registry = CollectorRegistry()
        
        # Should have some built-in collectors registered
        registered = registry.list_registered()
        assert len(registered) > 0
        assert DataType.ADAPTER_WEIGHTS in registered
        assert DataType.ADAPTER_GRADIENTS in registered
    
    def test_register_collector_ci(self):
        """Test registering a new collector in CI."""
        registry = CollectorRegistry()
        
        # Register mock collector
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=True)
        
        assert DataType.HIDDEN_STATES in registry.list_registered()
        assert DataType.HIDDEN_STATES in registry.list_enabled()
    
    def test_register_invalid_collector_ci(self):
        """Test registering an invalid collector raises error."""
        registry = CollectorRegistry()
        
        with pytest.raises(CollectorRegistryError, match="must inherit from DataCollector"):
            registry.register(DataType.HIDDEN_STATES, InvalidCollectorCI)
    
    def test_get_collector_ci(self):
        """Test getting a collector instance in CI."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=True)
        
        collector = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector is not None
        assert isinstance(collector, MockCollectorCI)
        
        # Getting same collector again should return same instance
        collector2 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector is collector2
    
    def test_collector_with_simple_model(self, simple_model):
        """Test collector working with simple CI model."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=True)
        
        collector = registry.get_collector(DataType.HIDDEN_STATES)
        
        # Test collection with simple model
        result = collector.collect(simple_model, step=100)
        assert result is not None
        assert result["step"] == 100
        assert result["model_type"] == "SimpleModel"
    
    def test_enable_disable_collector_ci(self):
        """Test enabling and disabling collectors in CI."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=False)
        
        # Initially disabled
        assert DataType.HIDDEN_STATES not in registry.list_enabled()
        assert registry.get_collector(DataType.HIDDEN_STATES) is None
        
        # Enable it
        registry.enable(DataType.HIDDEN_STATES)
        assert DataType.HIDDEN_STATES in registry.list_enabled()
        assert registry.get_collector(DataType.HIDDEN_STATES) is not None
        
        # Disable it
        registry.disable(DataType.HIDDEN_STATES)
        assert DataType.HIDDEN_STATES not in registry.list_enabled()
        assert registry.get_collector(DataType.HIDDEN_STATES) is None
    
    def test_configure_collector_ci(self):
        """Test configuring a collector in CI."""
        registry = CollectorRegistry()
        
        # Register with initial config
        initial_config = {"ci_param": "value1", "interval": 100}
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, config=initial_config, enabled=True)
        
        collector1 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector1.config["ci_param"] == "value1"
        assert collector1.config["interval"] == 100
        
        # Update configuration
        new_config = {"ci_param": "value2", "interval": 200, "new_param": "test"}
        registry.configure(DataType.HIDDEN_STATES, new_config)
        
        # Should get new instance with new config
        collector2 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector2 is not collector1  # New instance
        assert collector2.config["ci_param"] == "value2"
        assert collector2.config["interval"] == 200
        assert collector2.config["new_param"] == "test"
    
    def test_unregister_collector_ci(self):
        """Test unregistering a collector in CI."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=True)
        
        # Verify it's registered
        assert DataType.HIDDEN_STATES in registry.list_registered()
        
        # Unregister
        registry.unregister(DataType.HIDDEN_STATES)
        
        # Should be gone
        assert DataType.HIDDEN_STATES not in registry.list_registered()
        assert DataType.HIDDEN_STATES not in registry.list_enabled()
        assert registry.get_collector(DataType.HIDDEN_STATES) is None
    
    def test_get_all_collectors_ci(self):
        """Test getting all collectors in CI."""
        registry = CollectorRegistry()
        
        # Clear and register specific collectors for testing
        registry.register(DataType.HIDDEN_STATES, MockCollectorCI, enabled=True)
        registry.register(DataType.EMBEDDING_STATES, MockCollectorCI, enabled=True)
        registry.register(DataType.ATTENTION_PATTERNS, MockCollectorCI, enabled=False)
        
        # Get only enabled
        enabled_collectors = registry.get_all_collectors(only_enabled=True)
        assert DataType.HIDDEN_STATES in enabled_collectors
        assert DataType.EMBEDDING_STATES in enabled_collectors
        assert DataType.ATTENTION_PATTERNS not in enabled_collectors
        
        # Get all (disabled won't have instances)
        all_collectors = registry.get_all_collectors(only_enabled=False)
        assert DataType.HIDDEN_STATES in all_collectors
        assert DataType.EMBEDDING_STATES in all_collectors
        # Disabled collectors won't have instances
        assert DataType.ATTENTION_PATTERNS not in all_collectors
    
    def test_create_from_config_ci(self):
        """Test creating registry configuration from dict in CI."""
        registry = CollectorRegistry()
        
        config = {
            "capture_adapter_weights": True,
            "capture_adapter_gradients": False,
            "capture_activations": True,
            "collector_configs": {
                "adapter_weights": {"ci_test": True, "interval": 50}
            }
        }
        
        registry.create_from_config(config)
        
        # Check enabled/disabled based on config
        assert registry.is_enabled(DataType.ADAPTER_WEIGHTS)
        assert not registry.is_enabled(DataType.ADAPTER_GRADIENTS)
        assert registry.is_enabled(DataType.ACTIVATIONS)
    
    def test_global_registry_functions_ci(self):
        """Test global registry convenience functions in CI."""
        # Register using global function
        register_collector(DataType.LOSS_LANDSCAPES, MockCollectorCI, enabled=True)
        
        # Get using global function
        collector = get_registry().get_collector(DataType.LOSS_LANDSCAPES)
        assert collector is not None
        assert isinstance(collector, MockCollectorCI)
        
        # Clean up
        get_registry().unregister(DataType.LOSS_LANDSCAPES)