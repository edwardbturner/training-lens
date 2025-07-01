# Training Lens Project Specification

## Overview

Training Lens is a Python library for comprehensive analysis and monitoring of fine-tuning training runs. It provides deep insights into how models evolve during training by capturing detailed checkpoints, metrics, and gradients.

## Core Objectives

1. **Training Wrapper**: Seamless integration with existing fine-tuning workflows (initially LoRA via unsloth)
2. **Checkpoint Management**: Automated capture of model state at configurable intervals
3. **Real-time Monitoring**: Live metrics and analysis during training
4. **Data Export**: Raw training data (weights, gradients) and standardized summaries
5. **Scalable Architecture**: Extensible design for multiple fine-tuning methods

## Architecture

### 1. Core Training Module (`training_lens.training`)

- **TrainingWrapper**: Main class that wraps existing training loops
- **CheckpointManager**: Handles model state capture and storage
- **MetricsCollector**: Gathers training metrics, gradients, and model statistics
- **ConfigurationManager**: Handles training configuration and hyperparameters

### 2. Analysis Module (`training_lens.analysis`)

- **CheckpointAnalyzer**: Extracts insights from saved checkpoints
- **GradientAnalyzer**: Analyzes gradient flows and patterns
- **WeightAnalyzer**: Tracks weight evolution and distribution changes
- **StandardReports**: Pre-built analysis reports and visualizations

### 3. Integration Module (`training_lens.integrations`)

- **WandBIntegration**: Metrics logging and experiment tracking
- **HuggingFaceIntegration**: Model and dataset management
- **StorageBackends**: Flexible storage for checkpoints and artifacts

### 4. CLI Interface (`training_lens.cli`)

- **train**: Start training with monitoring
- **analyze**: Post-training analysis of checkpoints
- **export**: Export raw data or generate reports
- **monitor**: Real-time training dashboard

## Technical Requirements

### Dependencies

**Core Training**:
- unsloth (LoRA training)
- transformers (model handling)
- torch (core ML framework)
- peft (parameter-efficient fine-tuning)

**Monitoring & Analysis**:
- wandb (experiment tracking)
- huggingface-hub (model/dataset management)
- numpy, pandas (data processing)
- matplotlib, seaborn (visualization)

**Infrastructure**:
- click (CLI framework)
- pydantic (configuration validation)
- rich (terminal UI)
- pytest (testing)

### Package Structure

```
training-lens/
├── training_lens/
│   ├── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── wrapper.py
│   │   ├── checkpoint_manager.py
│   │   ├── metrics_collector.py
│   │   └── config.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── checkpoint_analyzer.py
│   │   ├── gradient_analyzer.py
│   │   ├── weight_analyzer.py
│   │   └── reports.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── wandb_integration.py
│   │   ├── huggingface_integration.py
│   │   └── storage.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── train.py
│   │   ├── analyze.py
│   │   └── export.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
├── tests/
├── examples/
├── docs/
├── specs/
├── pyproject.toml
├── setup.py
└── README.md
```

## Usage Patterns

### 1. Basic Training with Monitoring

```python
from training_lens import TrainingWrapper

wrapper = TrainingWrapper(
    model_name="meta-llama/Llama-2-7b-hf",
    training_method="lora",
    checkpoint_interval=100,
    wandb_project="my-training"
)

wrapper.train(
    dataset=dataset,
    output_dir="./checkpoints",
    num_epochs=3
)
```

### 2. Real-time Analysis

```python
from training_lens import LiveAnalyzer

analyzer = LiveAnalyzer(checkpoint_dir="./checkpoints")
analyzer.start_monitoring()  # Real-time dashboard
```

### 3. Post-training Analysis

```python
from training_lens import CheckpointAnalyzer

analyzer = CheckpointAnalyzer("./checkpoints")
report = analyzer.generate_standard_report()
raw_data = analyzer.export_raw_gradients()
```

## Data Capture Specifications

### Checkpoint Data
- Model weights at each checkpoint
- Optimizer state
- Learning rate schedule state
- Training step metadata

### Gradient Information
- Per-layer gradient norms
- Gradient flow patterns
- Parameter update magnitudes
- Gradient accumulation statistics

### Training Metrics
- Loss curves (training/validation)
- Learning rate progression
- Memory usage patterns
- Training speed metrics

### Analysis Outputs

**Standard Reports**:
- Training convergence analysis
- Gradient flow visualization
- Weight distribution evolution
- Overfitting detection metrics

**Raw Data Exports**:
- Checkpoint weight tensors
- Gradient vectors per step
- Activation patterns
- Layer-wise statistics

## Extensibility Design

### Fine-tuning Method Abstraction

```python
class FineTuningMethod(ABC):
    @abstractmethod
    def setup_training(self, model, config): pass
    
    @abstractmethod
    def train_step(self, batch): pass
    
    @abstractmethod
    def save_checkpoint(self, path): pass
```

### Storage Backend Abstraction

```python
class StorageBackend(ABC):
    @abstractmethod
    def save_checkpoint(self, checkpoint, metadata): pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id): pass
    
    @abstractmethod
    def list_checkpoints(self): pass
```

## Development Phases

### Phase 1: Core Infrastructure
- Basic package structure
- LoRA training wrapper
- Simple checkpoint management
- WandB integration

### Phase 2: Analysis Framework
- Checkpoint analysis tools
- Gradient tracking
- Standard report generation
- CLI interface

### Phase 3: Advanced Features
- Real-time monitoring dashboard
- Multiple fine-tuning methods
- Advanced visualization
- Performance optimization

### Phase 4: Production Features
- Distributed training support
- Custom storage backends
- Advanced analysis algorithms
- Documentation and examples

## Quality Assurance

- Comprehensive unit tests for all modules
- Integration tests with real training workflows
- Performance benchmarking
- Documentation with examples
- Continuous integration pipeline