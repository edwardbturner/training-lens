"""Main CLI entry point for training-lens."""

from pathlib import Path

import click

from ..utils.logging import get_logger, setup_logging
from .analyze import analyze_command
from .export import export_command
from .train import train_command
from .activations import activations

# Setup logging
setup_logging("INFO")
logger = get_logger("training_lens.cli")


@click.group()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", type=click.Path(), help="Log file path")
def cli(verbose: bool, log_file: str) -> None:
    """Training Lens - Comprehensive training analysis for ML models.

    Training Lens provides deep insights into model training processes
    through checkpoint analysis, gradient monitoring, and real-time metrics.
    """
    # Reconfigure logging based on CLI options
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level, log_file)

    if verbose:
        logger.info("Verbose logging enabled")


# Add subcommands
cli.add_command(train_command, name="train")
cli.add_command(analyze_command, name="analyze")
cli.add_command(export_command, name="export")
cli.add_command(activations, name="activations")


@cli.command()
@click.option(
    "--config-template",
    type=click.Choice(["basic", "advanced", "research"]),
    default="basic",
    help="Configuration template to generate",
)
@click.option(
    "--output", "-o", type=click.Path(), default="training_config.yaml", help="Output configuration file path"
)
def init(config_template: str, output: str) -> None:
    """Initialize a new training configuration file."""
    import yaml

    output_path = Path(output)

    # Create template configurations
    templates = {
        "basic": {
            "model_name": "microsoft/DialoGPT-medium",
            "training_method": "lora",
            "max_steps": 1000,
            "checkpoint_interval": 100,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 2,
            "output_dir": "./training_output",
        },
        "advanced": {
            "model_name": "microsoft/DialoGPT-medium",
            "training_method": "lora",
            "max_steps": 5000,
            "checkpoint_interval": 250,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 100,
            "lora_r": 32,
            "lora_alpha": 64,
            "output_dir": "./training_output",
            "wandb_project": "my-training-project",
            "hf_hub_repo": "username/my-model",
            "capture_gradients": True,
            "capture_weights": True,
        },
        "research": {
            "model_name": "microsoft/DialoGPT-medium",
            "training_method": "lora",
            "max_steps": 10000,
            "checkpoint_interval": 100,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 500,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "output_dir": "./research_output",
            "logging_dir": "./logs",
            "wandb_project": "research-project",
            "hf_hub_repo": "research/detailed-model",
            "capture_gradients": True,
            "capture_weights": True,
            "capture_activations": True,
        },
    }

    config_data = templates[config_template]

    # Write configuration file
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)

    click.echo(f"✅ Generated {config_template} configuration: {output_path}")
    click.echo(f"📝 Edit the configuration file and run: training-lens train --config {output_path}")


@cli.command()
def info() -> None:
    """Show system information and package details."""
    import sys

    import torch

    from .. import __version__
    from ..utils.helpers import get_device, get_gpu_memory_usage, get_memory_usage

    click.echo("🔍 Training Lens System Information")
    click.echo("=" * 40)

    # Package info
    click.echo(f"📦 Training Lens version: {__version__}")
    click.echo(f"🐍 Python version: {sys.version.split()[0]}")

    # System info
    click.echo(f"💻 Device: {get_device()}")
    click.echo(f"🔥 PyTorch version: {torch.__version__}")
    click.echo(f"⚡ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        click.echo(f"🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            click.echo(f"   GPU {i}: {gpu_name}")

    # Memory info
    memory = get_memory_usage()
    click.echo(f"🧠 Memory usage: {memory['rss']} ({memory['percent']})")

    gpu_memory = get_gpu_memory_usage()
    if gpu_memory:
        click.echo(f"🎮 GPU memory: {gpu_memory['allocated']} / {gpu_memory['total']} ({gpu_memory['percent']})")


@cli.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "csv"]),
    default="json",
    help="Output format for the summary",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: stdout)")
def summary(checkpoint_dir: str, output_format: str, output: str) -> None:
    """Generate a quick summary of training checkpoints."""
    import json

    import pandas as pd
    import yaml

    from ..analysis.checkpoint_analyzer import CheckpointAnalyzer

    analyzer = CheckpointAnalyzer(checkpoint_dir)

    # Generate summary
    summary_data = {
        "checkpoint_count": len(analyzer.checkpoints_info),
        "steps": [cp["step"] for cp in analyzer.checkpoints_info],
        "first_step": min([cp["step"] for cp in analyzer.checkpoints_info]) if analyzer.checkpoints_info else 0,
        "last_step": max([cp["step"] for cp in analyzer.checkpoints_info]) if analyzer.checkpoints_info else 0,
    }

    # Add basic metrics if available
    if analyzer.checkpoints_info:
        latest_metadata = analyzer.checkpoints_info[-1].get("metadata", {})
        summary_data.update(
            {
                "final_train_loss": latest_metadata.get("train_loss"),
                "final_learning_rate": latest_metadata.get("learning_rate"),
                "training_complete": True,
            }
        )

    # Format output
    if output_format == "json":
        output_text = json.dumps(summary_data, indent=2)
    elif output_format == "yaml":
        output_text = yaml.dump(summary_data, default_flow_style=False)
    elif output_format == "csv":
        df = pd.DataFrame([summary_data])
        output_text = df.to_csv(index=False)

    # Write output
    if output:
        with open(output, "w") as f:
            f.write(output_text)
        click.echo(f"✅ Summary saved to {output}")
    else:
        click.echo(output_text)


if __name__ == "__main__":
    cli()
