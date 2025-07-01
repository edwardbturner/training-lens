# Training Lens Examples

This directory contains example scripts demonstrating various features of Training Lens.

## Examples Overview

### 1. Basic Training Example (`basic_training_example.py`)

Demonstrates the core functionality of Training Lens:
- Simple training setup with LoRA fine-tuning
- Automatic checkpoint management
- Basic analysis and reporting

**Usage:**
```bash
python examples/basic_training_example.py
```

**Features demonstrated:**
- TrainingWrapper configuration
- Dataset preparation
- Training with monitoring
- Post-training analysis
- Gradient cosine similarity tracking

### 2. Advanced Training Example (`advanced_training_example.py`)

Shows advanced features and integrations:
- W&B experiment tracking
- HuggingFace Hub integration
- Comprehensive analysis
- Advanced configuration options

**Usage:**
```bash
# Set up API keys (optional)
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

python examples/advanced_training_example.py
```

**Features demonstrated:**
- Advanced configuration options
- External service integrations
- Real-time monitoring
- Comprehensive reporting
- Professional workflows

### 3. Analysis Example (`analysis_example.py`)

Demonstrates post-training analysis capabilities:
- Checkpoint analysis
- Gradient and weight analysis
- Report generation
- Data export

**Usage:**
```bash
python examples/analysis_example.py
```

**Features demonstrated:**
- CheckpointAnalyzer usage
- Specialized analyzers (Gradient, Weight)
- Standard reports generation
- Raw data export
- Visualization creation

## CLI Examples

Training Lens also provides a command-line interface:

### Initialize Configuration
```bash
training-lens init --config-template advanced --output my_config.yaml
```

### Train Model
```bash
training-lens train --config my_config.yaml --dataset data.jsonl
```

### Analyze Results
```bash
training-lens analyze ./training_output/checkpoints --include-plots --export-raw
```

### Export Data
```bash
training-lens export ./checkpoints --output ./exported --format csv --data-type all
```

## Dataset Formats

Training Lens supports various dataset formats:

### Conversation Format (JSON/JSONL)
```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"}
  ]
}
```

### Text Format (CSV)
```csv
text
"User: Hello\nAssistant: Hi there!"
"User: What's the weather?\nAssistant: I don't have access to weather data."
```

## Configuration Examples

### Basic Configuration
```yaml
model_name: "microsoft/DialoGPT-medium"
training_method: "lora"
max_steps: 1000
checkpoint_interval: 100
learning_rate: 2e-4
output_dir: "./training_output"
```

### Advanced Configuration
```yaml
model_name: "microsoft/DialoGPT-medium"
training_method: "lora"
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
max_steps: 5000
checkpoint_interval: 250
learning_rate: 1e-4
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
wandb_project: "my-training-project"
hf_hub_repo: "username/my-model"
capture_gradients: true
capture_weights: true
capture_activations: true
```

## Analysis Outputs

Training Lens generates various analysis outputs:

### Executive Summary
- Training overview
- Performance metrics
- Model health assessment
- Recommendations

### Technical Reports
- Detailed gradient analysis
- Weight evolution tracking
- Training dynamics
- Anomaly detection

### Visualizations
- Loss curves
- Learning rate schedules
- Gradient evolution plots
- Weight distribution charts

## Integration Examples

### W&B Integration
```python
config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    wandb_project="my-experiment",
    wandb_run_name="experiment-1",
    # ... other config
)
```

### HuggingFace Integration
```python
config = TrainingConfig(
    model_name="microsoft/DialoGPT-medium",
    hf_hub_repo="username/my-model",
    # ... other config
)
```

## Tips and Best Practices

1. **Start Simple**: Begin with the basic example and gradually add features
2. **Monitor Progress**: Use W&B integration for real-time monitoring
3. **Regular Analysis**: Run analysis after training to identify issues
4. **Export Data**: Export raw data for custom analysis and research
5. **Use CLI**: The CLI is great for automation and scripting

## Troubleshooting

### Common Issues

1. **CUDA Memory Errors**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Check if you're using the right device (GPU vs CPU)
3. **Poor Convergence**: Adjust learning rate or add warmup steps
4. **Large Checkpoints**: Increase checkpoint interval to save space

### Getting Help

- Check the logs in the output directory
- Use `training-lens info` to check system configuration
- Enable verbose logging with `--verbose` flag
- Review the generated analysis reports for insights

## Next Steps

After running these examples:

1. Experiment with your own datasets
2. Try different models and configurations
3. Explore the analysis features in depth
4. Integrate with your existing ML workflows
5. Contribute improvements back to the project