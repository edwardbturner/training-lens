"""Base classes for extensible data collection and analysis framework."""

import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch


class DataType(Enum):
    """Enumeration of supported data types for collection and analysis."""
    
    # Raw data types (logging during training)
    ADAPTER_WEIGHTS = "adapter_weights"
    ADAPTER_GRADIENTS = "adapter_gradients"
    ACTIVATIONS = "activations"
    LORA_ACTIVATIONS = "lora_activations"
    ATTENTION_PATTERNS = "attention_patterns"
    HIDDEN_STATES = "hidden_states"
    EMBEDDING_STATES = "embedding_states"
    LOSS_LANDSCAPES = "loss_landscapes"
    PARAMETER_NORMS = "parameter_norms"
    GRADIENT_NORMS = "gradient_norms"
    LEARNING_RATE_SCHEDULE = "learning_rate_schedule"
    OPTIMIZER_STATES = "optimizer_states"
    
    # Analysis types (downstream processing)
    CHECKPOINT_ANALYSIS = "checkpoint_analysis"
    GRADIENT_ANALYSIS = "gradient_analysis"
    WEIGHT_ANALYSIS = "weight_analysis"
    ACTIVATION_ANALYSIS = "activation_analysis"
    LORA_ANALYSIS = "lora_analysis"
    CONVERGENCE_ANALYSIS = "convergence_analysis"
    OVERFITTING_ANALYSIS = "overfitting_analysis"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    RANK_ANALYSIS = "rank_analysis"
    VISUALIZATION = "visualization"


class DataCollector(ABC):
    """Base class for collecting raw data during training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data collector.
        
        Args:
            config: Configuration dictionary for the collector
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.collection_frequency = self.config.get("frequency", 1)  # Every N steps
        self.storage_format = self.config.get("format", "torch")  # torch, numpy, json
        self.compression = self.config.get("compression", True)
        self.metadata: Dict[str, Any] = {}
        
        # Tracking
        self.last_collection_step = -1
        self.collection_count = 0
        
    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Return the type of data this collector handles."""
        pass
    
    @property
    @abstractmethod
    def supported_model_types(self) -> List[str]:
        """Return list of supported model types (e.g., ['lora', 'full', 'peft'])."""
        pass
    
    @abstractmethod
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        """Check if data can be collected at this step.
        
        Args:
            model: The model being trained
            step: Current training step
            
        Returns:
            True if collection is possible and needed
        """
        pass
    
    @abstractmethod
    def collect(
        self, 
        model: torch.nn.Module, 
        step: int, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Collect data from the model at the given step.
        
        Args:
            model: The model being trained
            step: Current training step
            **kwargs: Additional context (optimizer, loss, etc.)
            
        Returns:
            Collected data or None if collection failed/skipped
        """
        pass
    
    def should_collect(self, step: int) -> bool:
        """Check if data should be collected at this step based on frequency."""
        if not self.enabled:
            return False
        
        if step <= self.last_collection_step:
            return False
            
        return (step % self.collection_frequency) == 0
    
    def post_collect(self, step: int, data: Optional[Dict[str, Any]]) -> None:
        """Called after data collection to update tracking."""
        if data is not None:
            self.last_collection_step = step
            self.collection_count += 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get collector metadata."""
        return {
            "collector_type": self.__class__.__name__,
            "data_type": self.data_type.value,
            "config": self.config,
            "collection_count": self.collection_count,
            "last_collection_step": self.last_collection_step,
            "supported_models": self.supported_model_types,
            **self.metadata
        }


class DataAnalyzer(ABC):
    """Base class for analyzing collected data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data analyzer.
        
        Args:
            config: Configuration dictionary for the analyzer
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.output_format = self.config.get("output_format", "json")
        self.visualization = self.config.get("visualization", False)
        self.export_raw = self.config.get("export_raw", False)
        
    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Return the type of data this analyzer processes."""
        pass
    
    @property
    @abstractmethod
    def required_data_types(self) -> List[DataType]:
        """Return list of data types required for this analysis."""
        pass
    
    @abstractmethod
    def can_analyze(self, available_data: Dict[DataType, Any]) -> bool:
        """Check if analysis can be performed with available data.
        
        Args:
            available_data: Dictionary mapping data types to collected data
            
        Returns:
            True if analysis is possible
        """
        pass
    
    @abstractmethod
    def analyze(
        self, 
        data: Dict[DataType, Any], 
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Perform analysis on the provided data.
        
        Args:
            data: Dictionary mapping data types to collected data
            output_dir: Optional directory to save outputs
            
        Returns:
            Analysis results
        """
        pass
    
    def get_dependencies(self) -> List[Type["DataAnalyzer"]]:
        """Get list of analyzer classes this analyzer depends on."""
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get analyzer metadata."""
        return {
            "analyzer_type": self.__class__.__name__,
            "data_type": self.data_type.value,
            "required_data_types": [dt.value for dt in self.required_data_types],
            "config": self.config,
            "dependencies": [dep.__name__ for dep in self.get_dependencies()],
        }


class CollectionManager:
    """Manages multiple data collectors during training."""
    
    def __init__(self, collectors: Optional[List[DataCollector]] = None):
        """Initialize collection manager.
        
        Args:
            collectors: List of data collectors to manage
        """
        self.collectors: Dict[DataType, DataCollector] = {}
        self.collection_history: Dict[int, Dict[DataType, Any]] = {}
        
        if collectors:
            for collector in collectors:
                self.register_collector(collector)
    
    def register_collector(self, collector: DataCollector) -> None:
        """Register a data collector.
        
        Args:
            collector: Data collector to register
        """
        self.collectors[collector.data_type] = collector
    
    def collect_all(
        self, 
        model: torch.nn.Module, 
        step: int, 
        **kwargs
    ) -> Dict[DataType, Any]:
        """Collect data from all registered collectors.
        
        Args:
            model: The model being trained
            step: Current training step
            **kwargs: Additional context
            
        Returns:
            Dictionary mapping data types to collected data
        """
        collected_data = {}
        
        for data_type, collector in self.collectors.items():
            if collector.should_collect(step) and collector.can_collect(model, step):
                try:
                    data = collector.collect(model, step, **kwargs)
                    if data is not None:
                        collected_data[data_type] = data
                        collector.post_collect(step, data)
                except Exception as e:
                    # Log error but don't fail training
                    print(f"Warning: Collection failed for {data_type}: {e}")
        
        if collected_data:
            self.collection_history[step] = collected_data
        
        return collected_data
    
    def get_collected_data(
        self, 
        data_types: Optional[List[DataType]] = None,
        steps: Optional[List[int]] = None
    ) -> Dict[int, Dict[DataType, Any]]:
        """Get collected data filtered by type and steps.
        
        Args:
            data_types: Filter by specific data types
            steps: Filter by specific steps
            
        Returns:
            Filtered collection history
        """
        filtered_data = {}
        
        for step, step_data in self.collection_history.items():
            if steps is not None and step not in steps:
                continue
                
            filtered_step_data = {}
            for data_type, data in step_data.items():
                if data_types is None or data_type in data_types:
                    filtered_step_data[data_type] = data
            
            if filtered_step_data:
                filtered_data[step] = filtered_step_data
        
        return filtered_data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for all collectors."""
        return {
            "collectors": {
                data_type.value: collector.get_metadata() 
                for data_type, collector in self.collectors.items()
            },
            "collection_steps": list(self.collection_history.keys()),
            "total_collections": len(self.collection_history),
        }


class AnalysisManager:
    """Manages multiple data analyzers."""
    
    def __init__(self, analyzers: Optional[List[DataAnalyzer]] = None):
        """Initialize analysis manager.
        
        Args:
            analyzers: List of data analyzers to manage
        """
        self.analyzers: Dict[DataType, DataAnalyzer] = {}
        self.analysis_results: Dict[DataType, Any] = {}
        
        if analyzers:
            for analyzer in analyzers:
                self.register_analyzer(analyzer)
    
    def register_analyzer(self, analyzer: DataAnalyzer) -> None:
        """Register a data analyzer.
        
        Args:
            analyzer: Data analyzer to register
        """
        self.analyzers[analyzer.data_type] = analyzer
    
    def analyze_all(
        self, 
        collected_data: Dict[int, Dict[DataType, Any]], 
        output_dir: Optional[Path] = None
    ) -> Dict[DataType, Any]:
        """Run all applicable analyzers on collected data.
        
        Args:
            collected_data: Data collected during training
            output_dir: Optional directory to save outputs
            
        Returns:
            Dictionary mapping analysis types to results
        """
        # Flatten collected data by data type
        available_data = self._flatten_collected_data(collected_data)
        
        analysis_results = {}
        
        # Sort analyzers by dependencies
        ordered_analyzers = self._sort_analyzers_by_dependencies()
        
        for analyzer in ordered_analyzers:
            if not analyzer.enabled:
                continue
                
            if analyzer.can_analyze(available_data):
                try:
                    result = analyzer.analyze(available_data, output_dir)
                    analysis_results[analyzer.data_type] = result
                    
                    # Add result to available data for dependent analyzers
                    available_data[analyzer.data_type] = result
                    
                except Exception as e:
                    print(f"Warning: Analysis failed for {analyzer.data_type}: {e}")
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def _flatten_collected_data(
        self, 
        collected_data: Dict[int, Dict[DataType, Any]]
    ) -> Dict[DataType, Any]:
        """Flatten collected data by data type across all steps."""
        flattened = {}
        
        for step, step_data in collected_data.items():
            for data_type, data in step_data.items():
                if data_type not in flattened:
                    flattened[data_type] = {}
                flattened[data_type][step] = data
        
        return flattened
    
    def _sort_analyzers_by_dependencies(self) -> List[DataAnalyzer]:
        """Sort analyzers by their dependencies."""
        # Simple topological sort - can be improved for complex dependencies
        sorted_analyzers = []
        remaining_analyzers = list(self.analyzers.values())
        
        while remaining_analyzers:
            # Find analyzers with no unmet dependencies
            ready_analyzers = []
            for analyzer in remaining_analyzers:
                dependencies = analyzer.get_dependencies()
                if all(
                    any(isinstance(a, dep) for a in sorted_analyzers) 
                    for dep in dependencies
                ):
                    ready_analyzers.append(analyzer)
            
            if not ready_analyzers:
                # No progress possible, add remaining analyzers
                sorted_analyzers.extend(remaining_analyzers)
                break
            
            # Add ready analyzers and remove from remaining
            sorted_analyzers.extend(ready_analyzers)
            for analyzer in ready_analyzers:
                remaining_analyzers.remove(analyzer)
        
        return sorted_analyzers
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for all analyzers."""
        return {
            "analyzers": {
                data_type.value: analyzer.get_metadata() 
                for data_type, analyzer in self.analyzers.items()
            },
            "analysis_types": list(self.analysis_results.keys()),
            "total_analyses": len(self.analysis_results),
        }