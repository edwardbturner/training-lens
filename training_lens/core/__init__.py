"""Core framework for extensible LoRA data collection and analysis."""

# Core framework components
try:
    from .base import DataCollector, DataAnalyzer, DataType, Analyzer, AnalysisResult
    from .registry import AnalyzerRegistry
    from .collector_registry import (
        CollectorRegistry,
        get_registry,
        register_collector,
        get_collector,
        get_all_collectors,
    )
except ImportError:
    # Graceful fallback if core framework is not available
    DataCollector = None
    DataAnalyzer = None
    DataType = None
    Analyzer = None
    AnalysisResult = None
    CollectorRegistry = None
    AnalyzerRegistry = None
    get_registry = None
    register_collector = None
    get_collector = None
    get_all_collectors = None

# Optional integration component
try:
    from .integration import (
        IntegrationManager, TrainingLensFramework,
        create_lora_focused_framework, create_full_spectrum_framework
    )
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
    "Analyzer",
    "AnalysisResult",
    "CollectorRegistry",
    "AnalyzerRegistry",

    # Collector registry functions
    "get_registry",
    "register_collector",
    "get_collector",
    "get_all_collectors",

    # Integration management
    "IntegrationManager",
    "TrainingLensFramework",
    "create_lora_focused_framework",
    "create_full_spectrum_framework",
]
