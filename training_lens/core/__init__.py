"""Core framework for extensible LoRA data collection and analysis."""

# Core framework components
try:
    from .base import DataCollector, DataAnalyzer, DataType
    from .registry import CollectorRegistry, AnalyzerRegistry
except ImportError:
    # Graceful fallback if core framework is not available
    DataCollector = None
    DataAnalyzer = None
    DataType = None
    CollectorRegistry = None
    AnalyzerRegistry = None

# Optional integration component
try:
    from .integration import IntegrationManager, TrainingLensFramework, create_lora_focused_framework, create_full_spectrum_framework
except ImportError:
    IntegrationManager = None
    TrainingLensFramework = None
    create_lora_focused_framework = None
    create_full_spectrum_framework = None

__all__ = [
    # Core framework
    "DataCollector",
    "DataAnalyzer", 
    "DataType",
    "CollectorRegistry",
    "AnalyzerRegistry",
    
    # Integration management
    "IntegrationManager",
    "TrainingLensFramework",
    "create_lora_focused_framework", 
    "create_full_spectrum_framework",
]