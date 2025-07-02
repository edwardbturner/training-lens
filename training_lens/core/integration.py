"""Integration module for the extensible framework."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AnalysisManager, CollectionManager, DataType
from .registry import AnalyzerRegistry, CollectorRegistry, discover_all_plugins


class TrainingLensFramework:
    """Main framework class that integrates data collection and analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_discover: bool = True):
        """Initialize the Training Lens framework.

        Args:
            config: Configuration for collectors and analyzers
            auto_discover: Whether to auto-discover plugins
        """
        self.config = config or {}

        # Auto-discover plugins if enabled
        if auto_discover:
            discover_all_plugins()

        # Initialize managers
        self.collection_manager = CollectionManager()
        self.analysis_manager = AnalysisManager()

        # Setup based on configuration
        self._setup_collectors()
        self._setup_analyzers()

    def _setup_collectors(self) -> None:
        """Setup data collectors based on configuration."""
        collector_config = self.config.get("collectors", {})

        # Create collectors based on config or defaults
        if collector_config:
            for data_type_name, type_config in collector_config.items():
                try:
                    data_type = DataType(data_type_name)
                    collector = CollectorRegistry.create_collector(data_type, type_config)
                    if collector:
                        self.collection_manager.register_collector(collector)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to create collector for {data_type_name}: {e}")
        else:
            # Create all available collectors with default config
            collectors = CollectorRegistry.create_all_collectors()
            for collector in collectors:
                self.collection_manager.register_collector(collector)

    def _setup_analyzers(self) -> None:
        """Setup data analyzers based on configuration."""
        analyzer_config = self.config.get("analyzers", {})

        # Create analyzers based on config or defaults
        if analyzer_config:
            for data_type_name, type_config in analyzer_config.items():
                try:
                    data_type = DataType(data_type_name)
                    analyzer = AnalyzerRegistry.create_analyzer(data_type, type_config)
                    if analyzer:
                        self.analysis_manager.register_analyzer(analyzer)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to create analyzer for {data_type_name}: {e}")
        else:
            # Create all available analyzers with default config
            analyzers = AnalyzerRegistry.create_all_analyzers()
            for analyzer in analyzers:
                self.analysis_manager.register_analyzer(analyzer)

    def collect_training_data(self, model: Any, step: int, **kwargs) -> Dict[DataType, Any]:
        """Collect training data at a specific step.

        Args:
            model: The model being trained
            step: Current training step
            **kwargs: Additional context (optimizer, loss, etc.)

        Returns:
            Collected data by type
        """
        return self.collection_manager.collect_all(model, step, **kwargs)

    def analyze_training_data(self, output_dir: Optional[Path] = None) -> Dict[DataType, Any]:
        """Analyze all collected training data.

        Args:
            output_dir: Optional directory to save analysis outputs

        Returns:
            Analysis results by type
        """
        collected_data = self.collection_manager.collection_history
        return self.analysis_manager.analyze_all(collected_data, output_dir)

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of data collection."""
        return self.collection_manager.get_metadata()

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of data analysis."""
        return self.analysis_manager.get_metadata()

    def export_data(
        self, output_dir: Path, data_types: Optional[List[DataType]] = None, format: str = "json"
    ) -> Dict[str, Path]:
        """Export collected data to files.

        Args:
            output_dir: Directory to save exported data
            data_types: Specific data types to export (None for all)
            format: Export format (json, pickle, numpy)

        Returns:
            Dictionary mapping data types to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        collected_data = self.collection_manager.get_collected_data(data_types)
        exported_files = {}

        for step, step_data in collected_data.items():
            step_dir = output_dir / f"step_{step}"
            step_dir.mkdir(exist_ok=True)

            for data_type, data in step_data.items():
                filename = f"{data_type.value}.{format}"
                file_path = step_dir / filename

                try:
                    if format == "json":
                        import json

                        with open(file_path, "w") as f:
                            json.dump(data, f, indent=2, default=str)
                    elif format == "pickle":
                        import pickle

                        with open(file_path, "wb") as f:
                            pickle.dump(data, f)
                    elif format == "numpy" and hasattr(data, "numpy"):
                        import numpy as np

                        np.save(file_path.with_suffix(".npy"), data.numpy())

                    exported_files[f"{step}_{data_type.value}"] = file_path

                except Exception as e:
                    print(f"Warning: Failed to export {data_type.value} for step {step}: {e}")

        return exported_files

    def configure_collector(self, data_type: DataType, config: Dict[str, Any]) -> None:
        """Configure a specific collector.

        Args:
            data_type: Type of data collector to configure
            config: Configuration dictionary
        """
        if data_type in self.collection_manager.collectors:
            collector = self.collection_manager.collectors[data_type]
            collector.config.update(config)
        else:
            # Create new collector with config
            collector = CollectorRegistry.create_collector(data_type, config)
            if collector:
                self.collection_manager.register_collector(collector)

    def configure_analyzer(self, data_type: DataType, config: Dict[str, Any]) -> None:
        """Configure a specific analyzer.

        Args:
            data_type: Type of data analyzer to configure
            config: Configuration dictionary
        """
        if data_type in self.analysis_manager.analyzers:
            analyzer = self.analysis_manager.analyzers[data_type]
            analyzer.config.update(config)
        else:
            # Create new analyzer with config
            analyzer = AnalyzerRegistry.create_analyzer(data_type, config)
            if analyzer:
                self.analysis_manager.register_analyzer(analyzer)

    def get_available_collectors(self) -> Dict[DataType, str]:
        """Get available data collector types."""
        available = CollectorRegistry.get_available_collectors()
        return {dt: cls.__name__ for dt, cls in available.items()}

    def get_available_analyzers(self) -> Dict[DataType, str]:
        """Get available data analyzer types."""
        available = AnalyzerRegistry.get_available_analyzers()
        return {dt: cls.__name__ for dt, cls in available.items()}

    def register_custom_collector(self, collector_class) -> None:
        """Register a custom data collector.

        Args:
            collector_class: Custom collector class
        """
        CollectorRegistry.register(collector_class)

        # Create instance and add to collection manager
        temp_instance = collector_class()
        collector = CollectorRegistry.create_collector(temp_instance.data_type)
        if collector:
            self.collection_manager.register_collector(collector)

    def register_custom_analyzer(self, analyzer_class) -> None:
        """Register a custom data analyzer.

        Args:
            analyzer_class: Custom analyzer class
        """
        AnalyzerRegistry.register(analyzer_class)

        # Create instance and add to analysis manager
        temp_instance = analyzer_class()
        analyzer = AnalyzerRegistry.create_analyzer(temp_instance.data_type)
        if analyzer:
            self.analysis_manager.register_analyzer(analyzer)

    def create_pipeline_config(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create a configuration template for the current pipeline.

        Args:
            save_path: Optional path to save the configuration

        Returns:
            Configuration dictionary
        """
        config = {
            "collectors": {},
            "analyzers": {},
            "metadata": {
                "available_collectors": self.get_available_collectors(),
                "available_analyzers": self.get_available_analyzers(),
            },
        }

        # Add current collector configurations
        for data_type, collector in self.collection_manager.collectors.items():
            config["collectors"][data_type.value] = collector.config

        # Add current analyzer configurations
        for data_type, analyzer in self.analysis_manager.analyzers.items():
            config["analyzers"][data_type.value] = analyzer.config

        # Save configuration if path provided
        if save_path:
            import json

            with open(save_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        return config


# Convenience functions for common use cases
def create_lora_focused_framework(config: Optional[Dict[str, Any]] = None) -> TrainingLensFramework:
    """Create a framework focused on LoRA adapter analysis.

    Args:
        config: Optional configuration override

    Returns:
        Configured TrainingLensFramework
    """
    default_config = {
        "collectors": {
            DataType.ADAPTER_WEIGHTS.value: {"enabled": True, "frequency": 10},
            DataType.ADAPTER_GRADIENTS.value: {"enabled": True, "frequency": 5},
            DataType.LORA_ACTIVATIONS.value: {"enabled": True, "frequency": 20},
        },
        "analyzers": {
            DataType.LORA_ANALYSIS.value: {"enabled": True},
            DataType.CONVERGENCE_ANALYSIS.value: {"enabled": True},
            DataType.SIMILARITY_ANALYSIS.value: {"enabled": True},
        },
    }

    if config:
        # Deep merge configurations
        merged_config = default_config.copy()
        for key, value in config.items():
            if key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        config = merged_config
    else:
        config = default_config

    return TrainingLensFramework(config)


def create_full_spectrum_framework(config: Optional[Dict[str, Any]] = None) -> TrainingLensFramework:
    """Create a framework for comprehensive training analysis.

    Args:
        config: Optional configuration override

    Returns:
        Configured TrainingLensFramework
    """
    default_config = {
        "collectors": {
            DataType.ADAPTER_WEIGHTS.value: {"enabled": True, "frequency": 10},
            DataType.ADAPTER_GRADIENTS.value: {"enabled": True, "frequency": 5},
            DataType.ACTIVATIONS.value: {"enabled": True, "frequency": 25},
            DataType.LORA_ACTIVATIONS.value: {"enabled": True, "frequency": 25},
        },
        "analyzers": {
            DataType.ACTIVATION_ANALYSIS.value: {"enabled": True},
            DataType.LORA_ANALYSIS.value: {"enabled": True},
            DataType.CONVERGENCE_ANALYSIS.value: {"enabled": True},
            DataType.SIMILARITY_ANALYSIS.value: {"enabled": True},
        },
    }

    if config:
        merged_config = default_config.copy()
        for key, value in config.items():
            if key in merged_config and isinstance(merged_config[key], dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        config = merged_config
    else:
        config = default_config

    return TrainingLensFramework(config)
