# 🔍 Training Lens

A comprehensive library for interpreting and analyzing fine-tuning training runs of machine learning models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Training Lens provides deep insights into how models evolve during training through comprehensive checkpoint analysis, real-time gradient monitoring, and automated reporting. Built for researchers, ML engineers, and anyone who wants to understand what's happening inside their model training.

## ✨ Features

- **🔄 Real-time Training Analysis**: Monitor gradient evolution, weight changes, and training dynamics as they happen
- **📊 Comprehensive Metrics**: Track gradient cosine similarity, weight distributions, and custom training insights  
- **🚀 Seamless Integration**: Works with unsloth, LoRA, and standard fine-tuning workflows
- **📈 Professional Reporting**: Generate executive summaries, technical reports, and diagnostic analyses
- **🔗 External Integrations**: Built-in support for W&B experiment tracking and HuggingFace Hub
- **🛠️ CLI & Python API**: Use via command line or integrate into your Python workflows
- **📤 Data Export**: Export raw training data for custom analysis and research

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

## 📋 Quick Start

### 1. Basic Training with Monitoring

```python
from training_lens import TrainingWrapper
from training_lens.training.config import TrainingConfig

# Configure training with monitoring
config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    training_method="lora",
    max_steps=1000,
    checkpoint_interval=100,
    capture_gradients=True,
    capture_weights=True,
)

# Initialize wrapper and train
wrapper = TrainingWrapper(config)
results = wrapper.train(dataset=your_dataset)
```

### 2. CLI Usage

```bash
# Initialize configuration template
training-lens init --config-template basic --output config.yaml

# Train with comprehensive monitoring
training-lens train --config config.yaml --dataset data.jsonl

# Analyze training results
training-lens analyze ./training_output/checkpoints --include-plots

# Export raw training data
training-lens export ./checkpoints --output ./exported --format csv
```

### 3. Analysis and Reporting

```python
from training_lens.analysis import CheckpointAnalyzer, StandardReports

# Analyze checkpoints
analyzer = CheckpointAnalyzer("./training_output/checkpoints")
reports = StandardReports(analyzer)

# Generate executive summary
summary = reports.generate_executive_summary()
print(f"Training Efficiency: {summary['model_health']['training_efficiency']}")
print(f"Gradient Health: {summary['model_health']['gradient_health']}")

# Export detailed technical report
reports.export_report("technical", "./analysis_report.json")
```

## 🎯 Core Capabilities

### Training Analysis
- **Gradient Evolution**: Track gradient direction consistency with cosine similarity
- **Weight Dynamics**: Monitor weight distribution changes and stability
- **Training Health**: Detect overfitting, gradient explosion/vanishing
- **Performance Metrics**: Assess training efficiency and convergence

### Integration Features
- **Weights & Biases**: Real-time experiment tracking and metrics logging
- **HuggingFace Hub**: Automatic model and checkpoint uploads to `training_lens_checkpoints/` folder
- **Multiple Formats**: Export data as JSON, CSV, NumPy arrays, or Parquet

### Reporting & Visualization
- **Executive Summaries**: High-level training insights for stakeholders
- **Technical Reports**: Detailed analysis for researchers and engineers
- **Diagnostic Analysis**: Automated issue detection with recommendations
- **Interactive Plots**: Training curves, gradient evolution, weight distributions

## 🔧 Configuration

Create configuration files for reproducible training:

```yaml
# config.yaml
model_name: "microsoft/DialoGPT-medium"
training_method: "lora"
lora_r: 32
lora_alpha: 64
max_steps: 5000
checkpoint_interval: 250
learning_rate: 1e-4
wandb_project: "my-training-project"
hf_hub_repo: "username/my-model"
capture_gradients: true
capture_weights: true
```

## 📊 Analysis Examples

### Gradient Consistency Analysis
```python
from training_lens.analysis import GradientAnalyzer

# Analyze gradient evolution
grad_analyzer = GradientAnalyzer(gradient_data)
consistency = grad_analyzer.analyze_gradient_consistency()
print(f"Mean cosine similarity: {consistency['mean_similarity']:.3f}")
print(f"Consistency level: {consistency['consistency_level']}")
```

### Weight Evolution Tracking
```python
from training_lens.analysis import WeightAnalyzer

# Analyze weight changes
weight_analyzer = WeightAnalyzer(weight_data)
evolution = weight_analyzer.analyze_weight_evolution()
print(f"Weight stability: {evolution['stability_assessment']['stability_level']}")
```

### Training Diagnostics
```python
# Generate comprehensive diagnostics
diagnostics = reports.generate_training_diagnostics()
print(f"Overall health: {diagnostics['overall_health']}")

# Show recommendations
for rec in diagnostics['recommendations']:
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

- [`basic_training_example.py`](examples/basic_training_example.py): Simple training with monitoring
- [`advanced_training_example.py`](examples/advanced_training_example.py): Full-featured training with integrations
- [`analysis_example.py`](examples/analysis_example.py): Post-training analysis and reporting

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest`
5. **Submit a pull request**

### Areas for Contribution

- **New Analysis Methods**: Additional metrics and insights
- **Visualization Improvements**: Enhanced charts and dashboards  
- **Integration Support**: Additional external service integrations
- **Performance Optimizations**: Faster analysis and reduced memory usage
- **Documentation**: Examples, tutorials, and API documentation

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

*Training Lens - See what your models learn* 🔍