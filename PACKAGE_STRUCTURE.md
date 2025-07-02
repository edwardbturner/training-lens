# Training Lens Package Structure

This document outlines the clean and correct structure of all `__init__.py` files in the LoRA-focused training-lens package.

## Main Package (`training_lens/__init__.py`)

**Purpose**: Main entry point for the LoRA-focused training library
**Key Components**:
- Core training components (LoRA-only)
- Analysis components with graceful fallbacks
- Integration components with error handling
- LoRA-specific aliases for clarity

**Main Exports**:
- `TrainingWrapper` (LoRA-only training)
- `TrainingConfig`, `CheckpointManager`, `MetricsCollector`
- `CheckpointAnalyzer`, `StandardReports`
- `HuggingFaceIntegration`, `WandBIntegration`
- `LoRATrainingWrapper` (alias for clarity)
- `LoRACheckpointAnalyzer` (alias for clarity)

## Training Module (`training_lens/training/__init__.py`)

**Purpose**: LoRA training components with Unsloth integration
**Key Components**:
- Configuration for LoRA training
- Training wrapper (LoRA-only)
- Checkpoint management with adapter focus
- Metrics collection for LoRA-specific insights

**Main Exports**:
- `TrainingConfig`, `CheckpointMetadata`
- `TrainingWrapper` (LoRA-only)
- `CheckpointManager`, `MetricsCollector`

## Analysis Module (`training_lens/analysis/__init__.py`)

**Purpose**: LoRA checkpoint analysis and insights
**Key Components**:
- Core LoRA checkpoint analysis
- Specialized LoRA analysis with graceful fallbacks
- Optional activation analysis components

**Main Exports**:
- `CheckpointAnalyzer`, `StandardReports`
- `GradientAnalyzer`, `WeightAnalyzer`
- `LoRAActivationTracker`, `LoRAParameterAnalyzer`
- `ActivationAnalyzer`, `ActivationExtractor`, `ActivationVisualizer`

## Integrations Module (`training_lens/integrations/__init__.py`)

**Purpose**: External service integrations optimized for LoRA
**Key Components**:
- Core LoRA-optimized integrations
- Storage backends with error handling
- Optional activation storage

**Main Exports**:
- `HuggingFaceIntegration`, `WandBIntegration`
- `StorageBackend`, `LocalStorage`
- `ActivationStorage`

## CLI Module (`training_lens/cli/__init__.py`)

**Purpose**: Command-line interface for LoRA training
**Main Exports**:
- `cli` (main CLI entry point)

## Utils Module (`training_lens/utils/__init__.py`)

**Purpose**: Utility functions for LoRA training analysis
**Key Components**:
- Logging utilities with rich formatting
- General utilities for device detection, formatting, and saving

**Main Exports**:
- `setup_logging`, `get_logger`, `TrainingLogger`
- `get_device`, `format_size`, `safe_save`

## Optional Modules

### Analyzers (`training_lens/analyzers/__init__.py`)
**Purpose**: Downstream analysis of collected LoRA training data
**Main Exports**:
- `LoRAAnalyzer` (LoRA-specific)
- `ActivationAnalyzer`, `ConvergenceAnalyzer`, `SimilarityAnalyzer`

### Collectors (`training_lens/collectors/__init__.py`)
**Purpose**: Raw LoRA training data capture
**Main Exports**:
- `AdapterWeightsCollector`, `AdapterGradientsCollector`, `LoRAActivationsCollector`
- `ActivationsCollector`

### Core Framework (`training_lens/core/__init__.py`)
**Purpose**: Extensible LoRA data collection and analysis framework
**Main Exports**:
- `DataCollector`, `DataAnalyzer`, `DataType`
- `CollectorRegistry`, `AnalyzerRegistry`
- `IntegrationManager`, `TrainingLensFramework`

## Key Design Principles

1. **LoRA-First**: All modules are designed with LoRA training as the primary focus
2. **Graceful Fallbacks**: Optional dependencies fail gracefully with `None` assignments
3. **Clear Separation**: Core functionality separated from optional components
4. **Consistent Naming**: LoRA-specific components clearly identified
5. **Error Handling**: ImportError handling prevents package from breaking
6. **Alias Support**: LoRA-specific aliases provided for clarity (e.g., `LoRATrainingWrapper`)

## Import Examples

```python
# Basic LoRA training
from training_lens import TrainingWrapper, TrainingConfig

# LoRA-specific analysis
from training_lens import CheckpointAnalyzer, LoRACheckpointAnalyzer

# Integrations
from training_lens import HuggingFaceIntegration, WandBIntegration

# Using LoRA-specific aliases for clarity
from training_lens import LoRATrainingWrapper  # Same as TrainingWrapper
```

## Testing Import Health

All imports are tested to ensure:
- No circular dependencies
- Graceful handling of missing optional dependencies
- Correct aliasing relationships
- Proper module structure

The package structure is designed to be robust, maintainable, and clearly focused on LoRA training workflows.