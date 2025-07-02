"""Collector registry for extensible data collection.

This module implements a plugin-based registry pattern for data collectors,
allowing easy addition of new collector types without modifying core code.
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from ..utils.logging import get_logger
from .base import DataCollector, DataType

logger = get_logger(__name__)


class CollectorRegistryError(Exception):
    """Exception raised for collector registry errors."""

    pass


class CollectorRegistry:
    """Registry for managing data collectors with plugin support."""

    def __init__(self):
        """Initialize the collector registry."""
        self._collectors: Dict[DataType, Type[DataCollector]] = {}
        self._instances: Dict[DataType, DataCollector] = {}
        self._configs: Dict[DataType, Dict[str, Any]] = {}
        self._enabled: Set[DataType] = set()

        # Auto-discover built-in collectors on initialization
        self._discover_builtin_collectors()

    def register(
        self,
        data_type: DataType,
        collector_class: Type[DataCollector],
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        """Register a collector class for a specific data type.

        Args:
            data_type: The type of data this collector handles
            collector_class: The collector class to register
            config: Optional configuration for the collector
            enabled: Whether the collector is enabled by default

        Raises:
            CollectorRegistryError: If registration fails
        """
        # Validate collector class
        if not issubclass(collector_class, DataCollector):
            raise CollectorRegistryError(f"Collector class {collector_class} must inherit from DataCollector")

        # Check if already registered
        if data_type in self._collectors:
            logger.warning(f"Overwriting existing collector for {data_type}")

        # Register the collector
        self._collectors[data_type] = collector_class
        self._configs[data_type] = config or {}

        if enabled:
            self._enabled.add(data_type)

        logger.info(f"Registered collector {collector_class.__name__} for {data_type}")

    def unregister(self, data_type: DataType) -> None:
        """Unregister a collector for a specific data type.

        Args:
            data_type: The data type to unregister
        """
        if data_type in self._collectors:
            del self._collectors[data_type]
            self._configs.pop(data_type, None)
            self._enabled.discard(data_type)
            self._instances.pop(data_type, None)
            logger.info(f"Unregistered collector for {data_type}")

    def get_collector(self, data_type: DataType) -> Optional[DataCollector]:
        """Get an instance of a collector for a specific data type.

        Args:
            data_type: The data type to get a collector for

        Returns:
            Collector instance or None if not registered/enabled
        """
        if data_type not in self._enabled:
            return None

        if data_type not in self._instances:
            collector_class = self._collectors.get(data_type)
            if collector_class:
                try:
                    config = self._configs.get(data_type, {})
                    self._instances[data_type] = collector_class(config=config)
                except Exception as e:
                    logger.error(f"Failed to instantiate collector for {data_type}: {e}")
                    return None

        return self._instances.get(data_type)

    def get_all_collectors(self, only_enabled: bool = True) -> Dict[DataType, DataCollector]:
        """Get all registered collectors.

        Args:
            only_enabled: If True, only return enabled collectors

        Returns:
            Dictionary mapping data types to collector instances
        """
        collectors = {}
        data_types = self._enabled if only_enabled else self._collectors.keys()

        for data_type in data_types:
            collector = self.get_collector(data_type)
            if collector:
                collectors[data_type] = collector

        return collectors

    def enable(self, data_type: DataType) -> None:
        """Enable a registered collector.

        Args:
            data_type: The data type to enable
        """
        if data_type in self._collectors:
            self._enabled.add(data_type)
            logger.info(f"Enabled collector for {data_type}")

    def disable(self, data_type: DataType) -> None:
        """Disable a registered collector.

        Args:
            data_type: The data type to disable
        """
        self._enabled.discard(data_type)
        self._instances.pop(data_type, None)
        logger.info(f"Disabled collector for {data_type}")

    def is_enabled(self, data_type: DataType) -> bool:
        """Check if a collector is enabled.

        Args:
            data_type: The data type to check

        Returns:
            True if the collector is enabled
        """
        return data_type in self._enabled

    def list_registered(self) -> List[DataType]:
        """List all registered data types.

        Returns:
            List of registered data types
        """
        return list(self._collectors.keys())

    def list_enabled(self) -> List[DataType]:
        """List all enabled data types.

        Returns:
            List of enabled data types
        """
        return list(self._enabled)

    def configure(self, data_type: DataType, config: Dict[str, Any]) -> None:
        """Update configuration for a collector.

        Args:
            data_type: The data type to configure
            config: New configuration dictionary
        """
        if data_type in self._collectors:
            self._configs[data_type] = config
            # Clear cached instance to force recreation with new config
            self._instances.pop(data_type, None)
            logger.info(f"Updated configuration for {data_type}")

    def _discover_builtin_collectors(self) -> None:
        """Auto-discover and register built-in collectors."""
        # Import built-in collectors
        try:
            from ..collectors import (
                ActivationsCollector,
                AdapterGradientsCollector,
                AdapterWeightsCollector,
                LoRAActivationsCollector,
            )

            # Register built-in collectors
            self.register(DataType.ADAPTER_WEIGHTS, AdapterWeightsCollector)
            self.register(DataType.ADAPTER_GRADIENTS, AdapterGradientsCollector)
            self.register(DataType.ACTIVATIONS, ActivationsCollector)
            self.register(DataType.LORA_ACTIVATIONS, LoRAActivationsCollector)

            logger.info("Discovered and registered built-in collectors")

        except ImportError as e:
            logger.warning(f"Failed to import built-in collectors: {e}")

    def discover_plugins(self, plugin_dir: Union[str, Path]) -> int:
        """Discover and register collector plugins from a directory.

        Args:
            plugin_dir: Directory containing collector plugins

        Returns:
            Number of plugins discovered and registered
        """
        plugin_dir = Path(plugin_dir)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return 0

        count = 0

        # Look for Python files in the plugin directory
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                # Import the module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for DataCollector subclasses
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, DataCollector) and obj != DataCollector:
                            # Get the data type from the collector
                            try:
                                instance = obj()
                                data_type = instance.data_type
                                self.register(data_type, obj)
                                count += 1
                                logger.info(f"Discovered plugin collector: {name}")
                            except Exception as e:
                                logger.error(f"Failed to register plugin {name}: {e}")

            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")

        return count

    def create_from_config(self, config: Dict[str, Any]) -> None:
        """Configure registry from a configuration dictionary.

        Args:
            config: Configuration dictionary with collector settings
        """
        # Enable/disable collectors based on config
        for data_type in DataType:
            key = f"capture_{data_type.value.lower()}"
            if key in config:
                if config[key]:
                    self.enable(data_type)
                else:
                    self.disable(data_type)

        # Configure individual collectors
        collector_configs = config.get("collector_configs", {})
        for data_type_str, collector_config in collector_configs.items():
            try:
                data_type = DataType(data_type_str)
                self.configure(data_type, collector_config)
            except ValueError:
                logger.warning(f"Unknown data type in config: {data_type_str}")


# Global registry instance
_global_registry = CollectorRegistry()


# Convenience functions for global registry
def register_collector(
    data_type: DataType,
    collector_class: Type[DataCollector],
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
) -> None:
    """Register a collector in the global registry."""
    _global_registry.register(data_type, collector_class, config, enabled)


def get_collector(data_type: DataType) -> Optional[DataCollector]:
    """Get a collector from the global registry."""
    return _global_registry.get_collector(data_type)


def get_all_collectors(only_enabled: bool = True) -> Dict[DataType, DataCollector]:
    """Get all collectors from the global registry."""
    return _global_registry.get_all_collectors(only_enabled)


def get_registry() -> CollectorRegistry:
    """Get the global collector registry."""
    return _global_registry
