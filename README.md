# 🔍 Training Lens

A LoRA-focused library for interpreting and analyzing fine-tuning training runs with **Unsloth** integration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-green.svg)](https://github.com/unslothai/unsloth)

Training Lens provides deep insights into how **LoRA adapters** evolve during training through (a) **comprehensive data gathering and storage** during training runs, and (b) **flexible analysis infrastructure** for both automated analysis and custom data exploration. Optimized for LoRA training with Unsloth for maximum efficiency.

## 🔄 Architecture Overview

**Training Lens operates in two distinct phases:**

### Phase 1: Data Gathering & Storage
During training, the package automatically collects and stores raw training data:
- **Core Training Wrapper** (`training_lens/training/wrapper.py`) orchestrates the entire training process
- **Metrics Collector** (`training_lens/training/metrics_collector.py`) captures LoRA-specific metrics at each training step
- **Specialized Collectors** (`training_lens/collectors/`) gather specific data types:
  - `adapter_weights.py` - LoRA A/B matrix snapshots
  - `adapter_gradients.py` - Gradient flow through adapters
  - `activations.py` - Layer activation patterns
- **Checkpoint Manager** (`training_lens/training/checkpoint_manager.py`) handles efficient storage of collected data

### Phase 2: Analysis & Exploration
After training, the package provides multiple ways to analyze the collected data:
- **Extensible Analysis Framework** (`training_lens/analysis/core/base.py`) supports both standard and custom analysis
- **Standard Analysis Components** provide common LoRA insights out-of-the-box
- **Raw Data Access** allows researchers to implement custom analysis functions
- **CLI Tools** (`training_lens/cli/`) offer immediate analysis without coding

## ✨ Features

- **🔄 Real-time LoRA Analysis**: Monitor LoRA adapter gradient evolution, weight changes, and training dynamics
- **📊 LoRA-Specific Metrics**: Track adapter-specific gradient cosine similarity, weight distributions, and insights
- **🚀 Unsloth Integration**: Built specifically for efficient LoRA training with Unsloth optimization
- **📈 Adapter-Focused Reporting**: Generate LoRA-specific executive summaries, technical reports, and diagnostic analyses
- **🔗 External Integrations**: Built-in support for W&B experiment tracking and HuggingFace Hub with adapter-only uploads
- **🛠️ CLI & Python API**: Use via command line or integrate into your Python workflows
- **📤 Adapter Data Export**: Export LoRA adapter training data for custom analysis and research
- **⚡ Efficient Checkpointing**: Save only LoRA adapter weights and gradients, not full model weights

## 🚀 Installation

### Install from GitHub

```bash
pip install git+https://github.com/training-lens/training-lens.git
```

### Development Setup

```bash
git clone https://github.com/training-lens/training-lens.git
cd training-lens
pip install -e ".[dev]"
git config core.hooksPath .githooks
```

## 📊 Data Gathering & Analysis Workflows

### Data Gathering During Training
Training Lens automatically captures comprehensive data during your LoRA training runs:

```python
from training_lens import TrainingWrapper
from training_lens.training.config import TrainingConfig

# Configure what data to collect during training
config = TrainingConfig(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    
    # Data collection settings
    capture_adapter_gradients=True,    # Gradient flow through LoRA layers
    capture_adapter_weights=True,      # LoRA A/B matrix snapshots
    capture_lora_activations=True,     # Activation patterns
    
    # Storage settings
    checkpoint_interval=100,           # Save data every 100 steps
    upload_adapter_weights=True,       # Include in checkpoints
    upload_gradients=True,             # Include gradient data
)

# Training automatically collects and stores data
wrapper = TrainingWrapper(config)
results = wrapper.train(dataset)
# → Raw training data saved to ./training_output/checkpoints/
```

### Analysis After Training
After training, access your data through multiple pathways:

**Option 1: Standard Analysis (Automated)**
```python
from training_lens.analysis import CheckpointAnalyzer, StandardReports

# Analyze collected data with built-in functions
analyzer = CheckpointAnalyzer("./training_output/checkpoints")
reports = StandardReports(analyzer)

# Generate standard LoRA analysis reports
summary = reports.generate_executive_summary()
technical_report = reports.generate_technical_report()
```

**Option 2: Raw Data Access (Custom Analysis)**
```python
from training_lens.analysis.core import CollectionManager, DataType

# Access raw training data for custom analysis
manager = CollectionManager()
raw_data = manager.load_checkpoint_data("./training_output/checkpoints")

# Access specific data types
adapter_weights = raw_data[DataType.ADAPTER_WEIGHTS]
gradients = raw_data[DataType.ADAPTER_GRADIENTS] 
activations = raw_data[DataType.ACTIVATIONS]

# Implement your custom analysis functions
def my_custom_analysis(adapter_weights, gradients):
    # Your research code here
    return analysis_results
```

**Option 3: CLI Analysis (No Coding)**
```bash
# Immediate analysis via command line
training-lens analyze ./training_output/checkpoints --lora-focus --include-plots
training-lens export ./checkpoints --format csv --data-type adapter_weights
```

## 📋 Quick Start

### 1. Basic LoRA Training with Monitoring

```python
from training_lens import TrainingWrapper
from training_lens.training.config import TrainingConfig

# Configure LoRA training with monitoring
config = TrainingConfig(
    model_name="unsloth/llama-2-7b-bnb-4bit",  # Unsloth optimized model
    training_method="lora",  # Only LoRA supported
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    max_steps=1000,
    checkpoint_interval=100,
    # LoRA-specific monitoring
    capture_adapter_gradients=True,
    capture_adapter_weights=True,
    upload_adapter_weights=True,
    upload_gradients=True,
    # Unsloth configuration
    unsloth_load_in_4bit=True,
)

# Initialize wrapper and train LoRA adapter
wrapper = TrainingWrapper(config)
results = wrapper.train(dataset=your_dataset)
```

### 2. CLI Usage for LoRA Training

```bash
# Initialize LoRA configuration template
training-lens init --config-template lora --output config.yaml

# Train LoRA adapter with comprehensive monitoring
training-lens train --config config.yaml --dataset data.jsonl

# Analyze LoRA training results
training-lens analyze ./training_output/checkpoints --lora-focus --include-plots

# Export LoRA adapter training data
training-lens export ./checkpoints --output ./exported --format csv --adapter-only
```

### 3. LoRA Analysis and Reporting

```python
from training_lens.analysis import CheckpointAnalyzer, StandardReports

# Analyze LoRA checkpoints
analyzer = CheckpointAnalyzer("./training_output/checkpoints")
reports = StandardReports(analyzer)

# Generate LoRA-focused executive summary
summary = reports.generate_executive_summary()
print(f"LoRA Training Efficiency: {summary['model_health']['training_efficiency']}")
print(f"Adapter Gradient Health: {summary['model_health']['gradient_health']}")
print(f"Adapter Weight Stability: {summary['adapter_analysis']['weight_stability']}")

# Export detailed LoRA technical report
reports.export_report("lora_technical", "./lora_analysis_report.json")
```

## 🎯 Core LoRA Capabilities

### LoRA Training Analysis
- **Adapter Gradient Evolution**: Track LoRA adapter gradient direction consistency with cosine similarity
- **Adapter Weight Dynamics**: Monitor LoRA weight distribution changes and stability
- **Base Model Monitoring**: Verify base model weights remain frozen during LoRA training
- **LoRA Health Detection**: Detect adapter-specific issues like rank collapse or ineffective adaptation
- **Performance Metrics**: Assess LoRA training efficiency and convergence

### Unsloth Integration Features
- **Optimized Training**: Leverage Unsloth's 2x+ speed improvements for LoRA training
- **Memory Efficiency**: Reduced memory usage with 4-bit quantization and gradient checkpointing
- **Automatic Target Modules**: Smart detection of optimal LoRA target modules

### LoRA-Specific Integrations
- **Weights & Biases**: Real-time LoRA adapter experiment tracking and metrics logging
- **HuggingFace Hub**: Automatic LoRA adapter uploads to `training_lens_checkpoints/` folder (adapter-only)
- **Multiple Formats**: Export LoRA adapter data as JSON, CSV, NumPy arrays, or Parquet

### LoRA Reporting & Visualization
- **LoRA Executive Summaries**: High-level adapter training insights for stakeholders
- **Adapter Technical Reports**: Detailed LoRA analysis for researchers and engineers
- **LoRA Diagnostic Analysis**: Automated adapter-specific issue detection with recommendations
- **Adapter Plots**: LoRA training curves, adapter gradient evolution, weight distributions

## 🔧 Configuration

Create configuration files for reproducible training:

```yaml
# config.yaml - LoRA Training Configuration
model_name: "unsloth/llama-2-7b-bnb-4bit"  # Unsloth optimized model
training_method: "lora"  # Only LoRA supported in training_lens
lora_r: 32
lora_alpha: 64
lora_dropout: 0.1
target_modules: null  # Auto-detect optimal modules
max_steps: 5000
checkpoint_interval: 250
learning_rate: 2e-4
# LoRA-specific monitoring
capture_adapter_gradients: true
capture_adapter_weights: true
upload_adapter_weights: true
upload_gradients: true
# Unsloth configuration
unsloth_load_in_4bit: true
unsloth_max_seq_length: 2048
# Integrations
wandb_project: "my-lora-training-project"
hf_hub_repo: "username/my-lora-adapter"
```

## 📊 Analysis Examples

### LoRA Adapter Gradient Consistency Analysis
```python
from training_lens.analysis import GradientAnalyzer

# Analyze LoRA adapter gradient evolution
grad_analyzer = GradientAnalyzer(adapter_gradient_data)
consistency = grad_analyzer.analyze_adapter_gradient_consistency()
print(f"LoRA mean cosine similarity: {consistency['mean_similarity']:.3f}")
print(f"Adapter consistency level: {consistency['consistency_level']}")
print(f"Base model frozen: {consistency['base_model_stable']}")
```

### LoRA Weight Evolution Tracking
```python
from training_lens.analysis import WeightAnalyzer

# Analyze LoRA adapter weight changes
weight_analyzer = WeightAnalyzer(adapter_weight_data)
evolution = weight_analyzer.analyze_adapter_weight_evolution()
print(f"Adapter weight stability: {evolution['stability_assessment']['stability_level']}")
print(f"Rank utilization: {evolution['rank_analysis']['effective_rank']}/{evolution['rank_analysis']['configured_rank']}")
```

### LoRA Training Diagnostics
```python
# Generate comprehensive LoRA diagnostics
diagnostics = reports.generate_lora_training_diagnostics()
print(f"Overall LoRA health: {diagnostics['overall_health']}")
print(f"Adapter learning progress: {diagnostics['adapter_progress']}")

# Show LoRA-specific recommendations
for rec in diagnostics['lora_recommendations']:
    print(f"• {rec}")
```

## 🛠️ Development

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/training-lens/training-lens.git
   cd training-lens
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up git hooks**:
   ```bash
   git config core.hooksPath .githooks
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=training_lens --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/test_basic.py  # Run specific test file
```

### Code Quality

```bash
# Format code
black training_lens/ tests/
isort training_lens/ tests/

# Lint code
flake8 training_lens/ tests/

# Type checking
mypy training_lens/
```

### Project Structure

```
training-lens/
├── training_lens/          # Main package
│   ├── training/           # Core training components
│   ├── analysis/           # Analysis and reporting
│   ├── integrations/       # External service integrations
│   ├── cli/               # Command-line interface
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── examples/              # Usage examples
├── specs/                 # Technical specifications
└── .github/              # GitHub workflows and templates
```

## 📚 Examples

Explore comprehensive examples in the [`examples/`](examples/) directory:

- [`basic_training_example.py`](examples/basic_training_example.py): Simple LoRA training with monitoring
- [`advanced_training_example.py`](examples/advanced_training_example.py): Full-featured LoRA training with integrations
- [`analysis_example.py`](examples/analysis_example.py): Post-training LoRA analysis and reporting
- [`lora_optimization_example.py`](examples/lora_optimization_example.py): LoRA hyperparameter optimization
- [`unsloth_integration_example.py`](examples/unsloth_integration_example.py): Advanced Unsloth features

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest`
5. **Submit a pull request**

### Areas for Contribution

- **LoRA Analysis Methods**: Additional LoRA-specific metrics and insights
- **Unsloth Optimizations**: Enhanced integration with Unsloth features
- **Adapter Visualizations**: Enhanced LoRA-specific charts and dashboards
- **Integration Support**: Additional external service integrations for LoRA workflows
- **Performance Optimizations**: Faster LoRA analysis and reduced memory usage
- **Documentation**: LoRA examples, tutorials, and API documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[unsloth](https://github.com/unslothai/unsloth)**: Efficient LoRA training implementation
- **[Weights & Biases](https://wandb.ai/)**: Experiment tracking and monitoring
- **[HuggingFace](https://huggingface.co/)**: Model hub and transformers library
- **PyTorch Community**: Foundation for modern ML development

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/training-lens/training-lens/issues)
- **Documentation**: [Examples and guides](examples/README.md)
- **Discussions**: [Community Q&A](https://github.com/training-lens/training-lens/discussions)

---

*Training Lens - See what your LoRA adapters learn* 🔍⚡