# Training Lens Examples

This directory contains comprehensive examples demonstrating all features of Training Lens.

## Examples Overview

### Core Examples

#### 1. **Basic Training Example** (`basic_training_example.py`)
Perfect for getting started with Training Lens.

**Features demonstrated:**
- Simple LoRA fine-tuning setup
- Automatic checkpoint management (every step by default)
- Basic analysis and reporting
- Gradient cosine similarity tracking

**Usage:**
```bash
python examples/basic_training_example.py
```

**When to use:** First-time users, simple training scenarios

---

#### 2. **Advanced Training Example** (`advanced_training_example.py`)
Shows production-ready features and integrations.

**Features demonstrated:**
- W&B experiment tracking integration
- HuggingFace Hub automatic upload
- Advanced LoRA configuration
- Comprehensive post-training analysis
- Automatic local→HuggingFace fallback loading

**Usage:**
```bash
# Set up API keys (optional)
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

python examples/advanced_training_example.py
```

**When to use:** Production workflows, team collaboration, research projects

---

### Workflow Examples

#### 3. **Analysis Workflows** (`analysis_example.py`)
Demonstrates the two main analysis approaches.

**Workflows demonstrated:**
- **(A) Raw Data Access:** For custom analysis and research
- **(B) Specialized Summaries:** For ready-to-use insights

**Features:**
- Step-by-step data access methods
- Export formats for external tools
- Executive summaries for stakeholders
- Technical reports for ML engineers
- Training diagnostics and health checks

**Usage:**
```bash
python examples/analysis_example.py
```

**When to use:** Understanding analysis capabilities, choosing the right workflow

---

#### 4. **Raw Data Workflow** (`raw_data_workflow_example.py`)
Comprehensive guide to accessing and working with raw training data.

**Features demonstrated:**
- Multiple data access methods
- Export formats (CSV, JSON, NumPy, Parquet)
- Custom analysis examples
- Integration with external tools (pandas, R, MATLAB)
- Performance metrics and convergence analysis

**Usage:**
```bash
python examples/raw_data_workflow_example.py
```

**When to use:** Research projects, custom analysis, external tool integration

---

#### 5. **CLI Workflow** (`cli_workflow_example.py`)
Shows how to use Training Lens in automated workflows and scripts.

**Features demonstrated:**
- CLI command examples
- Automation scripting (Bash, Python)
- CI/CD integration patterns
- Docker deployment examples
- Production pipeline templates

**Usage:**
```bash
python examples/cli_workflow_example.py
```

**When to use:** Automation, CI/CD, production deployments, scripting

---

## Quick Start Guide

### 1. Choose Your Approach

| Approach | Best For | Example |
|----------|----------|---------|
| **Python API** | Interactive development, notebooks | `basic_training_example.py` |
| **CLI Commands** | Automation, scripting, CI/CD | `cli_workflow_example.py` |
| **Hybrid** | Production workflows | `advanced_training_example.py` |

### 2. Training Workflow

```python
# 1. Configure training
from training_lens import LoRATrainingWrapper
from training_lens.training.config import TrainingConfig

config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    checkpoint_interval=1,  # Save every step (default)
    wandb_project="my-project",  # Optional
    hf_hub_repo="username/model"  # Optional auto-upload
)

# 2. Train with monitoring
wrapper = LoRATrainingWrapper(config)
results = wrapper.train(dataset)

# 3. Analyze results
from training_lens.analysis import CheckpointAnalyzer, StandardReports

analyzer = CheckpointAnalyzer("./checkpoints")
reports = StandardReports(analyzer)
summary = reports.generate_executive_summary()
```

### 3. Analysis Workflow

Choose your approach:

**Option A: Ready-to-use summaries**
```python
# Executive summary for stakeholders
exec_summary = reports.generate_executive_summary()

# Technical diagnostics for engineers  
diagnostics = reports.generate_training_diagnostics()

# Specialized analysis
from training_lens.analysis import GradientAnalyzer, WeightAnalyzer
grad_analyzer = GradientAnalyzer(checkpoint_data)
grad_report = grad_analyzer.generate_gradient_report()
```

**Option B: Raw data for custom analysis**
```python
# Access raw metrics
metrics = analyzer.load_checkpoint_metrics(step=100)
gradients = metrics['gradient_cosine_similarities']
weights = metrics['weight_stats_history']

# Export for external tools
analyzer.export_raw_data("./exported_data")
```

---

## CLI Commands Reference

### Configuration
```bash
# Initialize configuration templates
training-lens init --config-template basic
training-lens init --config-template advanced  
training-lens init --config-template research
```

### Training
```bash
# Train with configuration file
training-lens train --config config.yaml

# Train with custom output directory
training-lens train --config config.yaml --output-dir ./results
```

### Analysis
```bash
# Quick checkpoint summary
training-lens summary ./checkpoints

# Comprehensive analysis
training-lens analyze ./checkpoints --output ./analysis

# System information
training-lens info
```

### Data Export
```bash
# Export all data in all formats
training-lens export ./checkpoints --output ./data --format all

# Export specific data types
training-lens export ./checkpoints --output ./gradients --data-type gradients
training-lens export ./checkpoints --output ./weights --data-type weights --format csv

# Export specific checkpoints
training-lens export ./checkpoints --output ./data --steps 1,10,50,100 --compress
```

---

## Configuration Examples

### Basic Configuration
```yaml
model_name: "microsoft/DialoGPT-medium"
training_method: "lora"
max_steps: 1000
checkpoint_interval: 1  # Every step (default)
learning_rate: 2e-4
output_dir: "./training_output"
```

### Advanced Configuration
```yaml
model_name: "microsoft/DialoGPT-medium"
training_method: "lora"

# LoRA settings
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05

# Training parameters
max_steps: 5000
checkpoint_interval: 50  # Every 50 steps
per_device_train_batch_size: 4
gradient_accumulation_steps: 8

# Integrations
wandb_project: "my-training-project"
hf_hub_repo: "username/my-model"

# Analysis settings
capture_gradients: true
capture_weights: true
capture_activations: true
```

---

## Data Formats and Exports

### Available Export Formats

| Format | Use Case | Command Example |
|--------|----------|-----------------|
| **CSV** | Pandas, Excel, R | `--format csv` |
| **JSON** | Web apps, general use | `--format json` |
| **NumPy** | Scientific computing | `--format numpy` |
| **Parquet** | Big data, compression | `--format parquet` |

### Data Types

| Data Type | Contains | Best Format |
|-----------|----------|-------------|
| **gradients** | Cosine similarities, norms | NumPy, CSV |
| **weights** | Weight evolution, layer stats | CSV, Parquet |
| **metrics** | Loss, learning rate, performance | CSV, JSON |

---

## Integration Examples

### Jupyter Notebook
```python
# Load and visualize training data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('exported_data/training_metrics.csv')
plt.plot(df['step'], df['train_loss'])
plt.title('Training Loss Curve')
```

### R Analysis
```r
library(ggplot2)

# Load exported data
data <- read.csv('exported_data/training_metrics.csv')

# Create visualization
ggplot(data, aes(x=step, y=train_loss)) +
  geom_line() +
  theme_minimal()
```

### MLflow Integration
```python
import mlflow

# Log metrics to MLflow
with mlflow.start_run():
    for _, row in df.iterrows():
        mlflow.log_metric('train_loss', row['train_loss'], step=row['step'])
```

---

## Production Workflows

### CI/CD Pipeline (GitHub Actions)
```yaml
name: Training Pipeline
on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Install Training Lens
      run: pip install training-lens
    - name: Run Training
      run: |
        training-lens train --config config.yaml
        training-lens analyze ./training_output/checkpoints
        training-lens export ./training_output/checkpoints --output ./results
```

### Docker Deployment
```dockerfile
FROM python:3.9
RUN pip install training-lens
WORKDIR /app
COPY config.yaml .
CMD ["training-lens", "train", "--config", "config.yaml"]
```

### Automated Experimentation
```bash
#!/bin/bash
# Run multiple experiments
for lr in 1e-4 2e-4 5e-4; do
    sed "s/learning_rate: .*/learning_rate: $lr/" config.yaml > config_lr_$lr.yaml
    training-lens train --config config_lr_$lr.yaml --output-dir ./results/lr_$lr
    training-lens analyze ./results/lr_$lr/checkpoints --output ./analysis/lr_$lr
done
```

---

## Troubleshooting

### Common Issues

**CUDA Memory Errors**
```yaml
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Enable gradient checkpointing
use_gradient_checkpointing: true
```

**Slow Training**
```bash
# Check system info
training-lens info

# Monitor with verbose logging
training-lens train --config config.yaml --verbose
```

**Large Checkpoint Files**
```yaml
# Increase checkpoint interval
checkpoint_interval: 100  # Instead of 1

# Or disable some data capture
capture_activations: false
```

### Getting Help

1. **Check logs**: Look in `{output_dir}/logs/training.log`
2. **System info**: Run `training-lens info`
3. **Verbose mode**: Add `--verbose` to any CLI command
4. **Examples**: Run the example scripts to verify setup

---

## Next Steps

1. **Start Simple**: Run `basic_training_example.py` first
2. **Explore Analysis**: Try `analysis_example.py` to understand workflows
3. **Scale Up**: Use `advanced_training_example.py` for production features
4. **Automate**: Implement `cli_workflow_example.py` patterns for your use case
5. **Customize**: Use `raw_data_workflow_example.py` for research projects

For more information, visit the [Training Lens documentation](https://github.com/your-org/training-lens).