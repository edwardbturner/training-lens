"""Unit tests for the collector registry."""

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


class MockCollector(DataCollector):
    """Mock collector for testing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.collected_data = []

    @property
    def data_type(self) -> DataType:
        return DataType.HIDDEN_STATES

    @property
    def supported_model_types(self) -> List[str]:
        return ["test"]

    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        return step % 10 == 0

    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        data = {"step": step, "test_data": "mock"}
        self.collected_data.append(data)
        return data


class InvalidCollector:
    """Invalid collector that doesn't inherit from DataCollector."""

    pass


class TestCollectorRegistry:
    """Test the collector registry functionality."""

    def test_registry_initialization(self):
        """Test registry initializes with built-in collectors."""
        registry = CollectorRegistry()

        # Should have some built-in collectors registered
        registered = registry.list_registered()
        assert len(registered) > 0
        assert DataType.ADAPTER_WEIGHTS in registered
        assert DataType.ADAPTER_GRADIENTS in registered

    def test_register_collector(self):
        """Test registering a new collector."""
        registry = CollectorRegistry()

        # Register mock collector
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=True)

        assert DataType.HIDDEN_STATES in registry.list_registered()
        assert DataType.HIDDEN_STATES in registry.list_enabled()

    def test_register_invalid_collector(self):
        """Test registering an invalid collector raises error."""
        registry = CollectorRegistry()

        with pytest.raises(CollectorRegistryError, match="must inherit from DataCollector"):
            registry.register(DataType.HIDDEN_STATES, InvalidCollector)

    def test_get_collector(self):
        """Test getting a collector instance."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=True)

        collector = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector is not None
        assert isinstance(collector, MockCollector)

        # Getting same collector again should return same instance
        collector2 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector is collector2

    def test_get_disabled_collector(self):
        """Test getting a disabled collector returns None."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=False)

        collector = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector is None

    def test_enable_disable_collector(self):
        """Test enabling and disabling collectors."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=False)

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

    def test_configure_collector(self):
        """Test configuring a collector."""
        registry = CollectorRegistry()

        # Register with initial config
        initial_config = {"test_param": "value1"}
        registry.register(DataType.HIDDEN_STATES, MockCollector, config=initial_config, enabled=True)

        collector1 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector1.config["test_param"] == "value1"

        # Update configuration
        new_config = {"test_param": "value2", "new_param": 123}
        registry.configure(DataType.HIDDEN_STATES, new_config)

        # Should get new instance with new config
        collector2 = registry.get_collector(DataType.HIDDEN_STATES)
        assert collector2 is not collector1  # New instance
        assert collector2.config["test_param"] == "value2"
        assert collector2.config["new_param"] == 123

    def test_unregister_collector(self):
        """Test unregistering a collector."""
        registry = CollectorRegistry()
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=True)

        # Verify it's registered
        assert DataType.HIDDEN_STATES in registry.list_registered()

        # Unregister
        registry.unregister(DataType.HIDDEN_STATES)

        # Should be gone
        assert DataType.HIDDEN_STATES not in registry.list_registered()
        assert DataType.HIDDEN_STATES not in registry.list_enabled()
        assert registry.get_collector(DataType.HIDDEN_STATES) is None

    def test_get_all_collectors(self):
        """Test getting all collectors."""
        registry = CollectorRegistry()

        # Register multiple collectors
        registry.register(DataType.HIDDEN_STATES, MockCollector, enabled=True)
        registry.register(DataType.EMBEDDING_STATES, MockCollector, enabled=True)
        registry.register(DataType.ATTENTION_PATTERNS, MockCollector, enabled=False)

        # Get only enabled
        enabled_collectors = registry.get_all_collectors(only_enabled=True)
        assert DataType.HIDDEN_STATES in enabled_collectors
        assert DataType.EMBEDDING_STATES in enabled_collectors
        assert DataType.ATTENTION_PATTERNS not in enabled_collectors

        # Get all
        all_collectors = registry.get_all_collectors(only_enabled=False)
        assert DataType.HIDDEN_STATES in all_collectors
        assert DataType.EMBEDDING_STATES in all_collectors
        # Disabled collectors won't have instances
        assert DataType.ATTENTION_PATTERNS not in all_collectors

    def test_create_from_config(self):
        """Test creating registry configuration from dict."""
        registry = CollectorRegistry()

        config = {
            "capture_adapter_weights": True,
            "capture_adapter_gradients": False,
            "capture_activations": True,
            "collector_configs": {"adapter_weights": {"test_param": "test_value"}},
        }

        registry.create_from_config(config)

        # Check enabled/disabled based on config
        assert registry.is_enabled(DataType.ADAPTER_WEIGHTS)
        assert not registry.is_enabled(DataType.ADAPTER_GRADIENTS)
        assert registry.is_enabled(DataType.ACTIVATIONS)

    def test_global_registry_functions(self):
        """Test global registry convenience functions."""
        # Register using global function
        register_collector(DataType.LOSS_LANDSCAPES, MockCollector, enabled=True)

        # Get using global function
        collector = get_registry().get_collector(DataType.LOSS_LANDSCAPES)
        assert collector is not None
        assert isinstance(collector, MockCollector)
