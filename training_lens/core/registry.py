"""Registry system for auto-discovering and managing collectors and analyzers."""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from .base import DataAnalyzer, DataCollector, DataType


class CollectorRegistry:
    """Registry for data collectors with auto-discovery capabilities."""

    _collectors: Dict[DataType, Type[DataCollector]] = {}
    _instances: Dict[DataType, DataCollector] = {}

    @classmethod
    def register(cls, collector_class: Type[DataCollector]) -> None:
        """Register a data collector class.

        Args:
            collector_class: Collector class to register
        """
        if not issubclass(collector_class, DataCollector):
            raise ValueError(f"{collector_class} must be a subclass of DataCollector")

        # Get data type from instance (need to instantiate temporarily)
        temp_instance = collector_class()
        data_type = temp_instance.data_type

        cls._collectors[data_type] = collector_class
        print(f"Registered collector: {collector_class.__name__} for {data_type.value}")

    @classmethod
    def get_collector_class(cls, data_type: DataType) -> Optional[Type[DataCollector]]:
        """Get collector class for a data type.

        Args:
            data_type: Data type to get collector for

        Returns:
            Collector class or None if not found
        """
        return cls._collectors.get(data_type)

    @classmethod
    def create_collector(cls, data_type: DataType, config: Optional[Dict] = None) -> Optional[DataCollector]:
        """Create a collector instance for a data type.

        Args:
            data_type: Data type to create collector for
            config: Configuration for the collector

        Returns:
            Collector instance or None if not found
        """
        collector_class = cls.get_collector_class(data_type)
        if collector_class is None:
            return None

        if data_type in cls._instances:
            return cls._instances[data_type]

        instance = collector_class(config)
        cls._instances[data_type] = instance
        return instance

    @classmethod
    def create_all_collectors(cls, config: Optional[Dict[DataType, Dict]] = None) -> List[DataCollector]:
        """Create instances of all registered collectors.

        Args:
            config: Configuration mapping data types to configs

        Returns:
            List of collector instances
        """
        collectors = []
        config = config or {}

        for data_type, collector_class in cls._collectors.items():
            collector_config = config.get(data_type, {})
            collector = cls.create_collector(data_type, collector_config)
            if collector:
                collectors.append(collector)

        return collectors

    @classmethod
    def get_available_collectors(cls) -> Dict[DataType, Type[DataCollector]]:
        """Get all available collector classes.

        Returns:
            Dictionary mapping data types to collector classes
        """
        return cls._collectors.copy()

    @classmethod
    def auto_discover(cls, package_path: Union[str, Path]) -> None:
        """Auto-discover collectors in a package.

        Args:
            package_path: Path to package containing collectors
        """
        if isinstance(package_path, str):
            package_path = Path(package_path)

        # Import all Python files in the package
        for py_file in package_path.glob("**/*.py"):
            if py_file.name.startswith("__"):
                continue

            # Convert file path to module name
            relative_path = py_file.relative_to(package_path.parent)
            module_name = str(relative_path.with_suffix("")).replace("/", ".")

            try:
                module = importlib.import_module(module_name)

                # Find all DataCollector subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, DataCollector) and obj != DataCollector and not inspect.isabstract(obj):
                        cls.register(obj)

            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")


class AnalyzerRegistry:
    """Registry for data analyzers with auto-discovery capabilities."""

    _analyzers: Dict[DataType, Type[DataAnalyzer]] = {}
    _instances: Dict[DataType, DataAnalyzer] = {}

    @classmethod
    def register(cls, analyzer_class: Type[DataAnalyzer]) -> None:
        """Register a data analyzer class.

        Args:
            analyzer_class: Analyzer class to register
        """
        if not issubclass(analyzer_class, DataAnalyzer):
            raise ValueError(f"{analyzer_class} must be a subclass of DataAnalyzer")

        # Get data type from instance (need to instantiate temporarily)
        temp_instance = analyzer_class()
        data_type = temp_instance.data_type

        cls._analyzers[data_type] = analyzer_class
        print(f"Registered analyzer: {analyzer_class.__name__} for {data_type.value}")

    @classmethod
    def get_analyzer_class(cls, data_type: DataType) -> Optional[Type[DataAnalyzer]]:
        """Get analyzer class for a data type.

        Args:
            data_type: Data type to get analyzer for

        Returns:
            Analyzer class or None if not found
        """
        return cls._analyzers.get(data_type)

    @classmethod
    def create_analyzer(cls, data_type: DataType, config: Optional[Dict] = None) -> Optional[DataAnalyzer]:
        """Create an analyzer instance for a data type.

        Args:
            data_type: Data type to create analyzer for
            config: Configuration for the analyzer

        Returns:
            Analyzer instance or None if not found
        """
        analyzer_class = cls.get_analyzer_class(data_type)
        if analyzer_class is None:
            return None

        if data_type in cls._instances:
            return cls._instances[data_type]

        instance = analyzer_class(config)
        cls._instances[data_type] = instance
        return instance

    @classmethod
    def create_all_analyzers(cls, config: Optional[Dict[DataType, Dict]] = None) -> List[DataAnalyzer]:
        """Create instances of all registered analyzers.

        Args:
            config: Configuration mapping data types to configs

        Returns:
            List of analyzer instances
        """
        analyzers = []
        config = config or {}

        for data_type, analyzer_class in cls._analyzers.items():
            analyzer_config = config.get(data_type, {})
            analyzer = cls.create_analyzer(data_type, analyzer_config)
            if analyzer:
                analyzers.append(analyzer)

        return analyzers

    @classmethod
    def get_available_analyzers(cls) -> Dict[DataType, Type[DataAnalyzer]]:
        """Get all available analyzer classes.

        Returns:
            Dictionary mapping data types to analyzer classes
        """
        return cls._analyzers.copy()

    @classmethod
    def auto_discover(cls, package_path: Union[str, Path]) -> None:
        """Auto-discover analyzers in a package.

        Args:
            package_path: Path to package containing analyzers
        """
        if isinstance(package_path, str):
            package_path = Path(package_path)

        # Import all Python files in the package
        for py_file in package_path.glob("**/*.py"):
            if py_file.name.startswith("__"):
                continue

            # Convert file path to module name
            relative_path = py_file.relative_to(package_path.parent)
            module_name = str(relative_path.with_suffix("")).replace("/", ".")

            try:
                module = importlib.import_module(module_name)

                # Find all DataAnalyzer subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, DataAnalyzer) and obj != DataAnalyzer and not inspect.isabstract(obj):
                        cls.register(obj)

            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")


def register_collector(data_type: DataType):
    """Decorator to register a collector class.

    Args:
        data_type: Data type the collector handles
    """

    def decorator(collector_class: Type[DataCollector]):
        CollectorRegistry.register(collector_class)
        return collector_class

    return decorator


def register_analyzer(data_type: DataType):
    """Decorator to register an analyzer class.

    Args:
        data_type: Data type the analyzer handles
    """

    def decorator(analyzer_class: Type[DataAnalyzer]):
        AnalyzerRegistry.register(analyzer_class)
        return analyzer_class

    return decorator


# Auto-discovery function
def discover_all_plugins(base_path: Optional[Path] = None) -> None:
    """Auto-discover all collectors and analyzers in the package.

    Args:
        base_path: Base path to search from (defaults to current package)
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent

    print("🔍 Auto-discovering data collectors and analyzers...")

    # Discover collectors
    collectors_path = base_path / "collectors"
    if collectors_path.exists():
        CollectorRegistry.auto_discover(collectors_path)

    # Discover analyzers
    analyzers_path = base_path / "analyzers"
    if analyzers_path.exists():
        AnalyzerRegistry.auto_discover(analyzers_path)

    # Also discover in existing analysis modules
    analysis_path = base_path / "analysis"
    if analysis_path.exists():
        AnalyzerRegistry.auto_discover(analysis_path)

    print("✅ Discovery complete:")
    print(f"   Collectors: {len(CollectorRegistry.get_available_collectors())}")
    print(f"   Analyzers: {len(AnalyzerRegistry.get_available_analyzers())}")
