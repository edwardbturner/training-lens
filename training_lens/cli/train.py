"""Training command for CLI."""

from pathlib import Path
from typing import Optional

import click

from ..training.config import TrainingConfig
from ..training.wrapper import TrainingWrapper
from ..utils.logging import get_logger

logger = get_logger("training_lens.cli.train")


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to training configuration file (YAML or JSON)")
@click.option("--model-name", "-m", type=str, help="Model name to fine-tune (overrides config)")
@click.option(
    "--dataset", type=click.Path(exists=True), required=True, help="Path to training dataset (JSON, JSONL, or CSV)"
)
@click.option("--eval-dataset", type=click.Path(exists=True), help="Path to evaluation dataset")
@click.option(
    "--output-dir", "-o", type=click.Path(), help="Output directory for checkpoints and models (overrides config)"
)
@click.option("--max-steps", type=int, help="Maximum training steps (overrides config)")
@click.option("--checkpoint-interval", type=int, help="Steps between checkpoints (overrides config)")
@click.option("--learning-rate", "--lr", type=float, help="Learning rate (overrides config)")
@click.option("--wandb-project", type=str, help="W&B project name (overrides config)")
@click.option("--hf-repo", type=str, help="HuggingFace repository for model upload (overrides config)")
@click.option("--resume-from", type=click.Path(exists=True), help="Resume training from checkpoint directory")
@click.option("--dry-run", is_flag=True, help="Show configuration and exit without training")
@click.option("--no-wandb", is_flag=True, help="Disable W&B logging")
@click.option("--no-hf-upload", is_flag=True, help="Disable HuggingFace model upload")
def train_command(
    config: Optional[str],
    model_name: Optional[str],
    dataset: str,
    eval_dataset: Optional[str],
    output_dir: Optional[str],
    max_steps: Optional[int],
    checkpoint_interval: Optional[int],
    learning_rate: Optional[float],
    wandb_project: Optional[str],
    hf_repo: Optional[str],
    resume_from: Optional[str],
    dry_run: bool,
    no_wandb: bool,
    no_hf_upload: bool,
) -> None:
    """Start model training with comprehensive monitoring.

    This command trains a model using the specified configuration and dataset,
    with automatic checkpoint saving, metrics collection, and optional W&B/HF integration.

    Examples:

        # Train with a configuration file
        training-lens train --config config.yaml --dataset data.jsonl

        # Quick training with CLI options
        training-lens train --model-name microsoft/DialoGPT-medium --dataset data.jsonl --max-steps 1000

        # Resume from checkpoint
        training-lens train --config config.yaml --dataset data.jsonl --resume-from ./checkpoints/checkpoint-500
    """
    click.echo("🚀 Starting Training Lens training...")

    # Load or create configuration
    if config:
        # Load from file
        config_path = Path(config)
        if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
            import yaml

            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            import json

            with open(config_path, "r") as f:
                config_data = json.load(f)

        training_config = TrainingConfig(**config_data)
        click.echo(f"📋 Loaded configuration from {config}")
    else:
        # Create basic configuration
        if not model_name:
            raise click.ClickException("Either --config or --model-name must be provided")

        config_data = {
            "model_name": model_name,
            "training_method": "lora",
            "max_steps": 1000,
            "checkpoint_interval": 100,
        }
        training_config = TrainingConfig(**config_data)
        click.echo("📋 Using default configuration")

    # Apply CLI overrides
    overrides = {}
    if model_name:
        overrides["model_name"] = model_name
    if output_dir:
        overrides["output_dir"] = output_dir
    if max_steps:
        overrides["max_steps"] = max_steps
    if checkpoint_interval:
        overrides["checkpoint_interval"] = checkpoint_interval
    if learning_rate:
        overrides["learning_rate"] = learning_rate
    if wandb_project and not no_wandb:
        overrides["wandb_project"] = wandb_project
    elif no_wandb:
        overrides["wandb_project"] = None
    if hf_repo and not no_hf_upload:
        overrides["hf_hub_repo"] = hf_repo
    elif no_hf_upload:
        overrides["hf_hub_repo"] = None

    # Update configuration with overrides
    if overrides:
        config_dict = training_config.to_dict()
        config_dict.update(overrides)
        training_config = TrainingConfig(**config_dict)
        click.echo(f"⚙️  Applied {len(overrides)} CLI overrides")

    # Show configuration
    click.echo("\n📊 Training Configuration:")
    click.echo(f"   Model: {training_config.model_name}")
    click.echo(f"   Method: {training_config.training_method}")
    click.echo(f"   Max steps: {training_config.max_steps}")
    click.echo(f"   Checkpoint interval: {training_config.checkpoint_interval}")
    click.echo(f"   Learning rate: {training_config.learning_rate}")
    click.echo(f"   Output directory: {training_config.output_dir}")

    if training_config.wandb_project:
        click.echo(f"   W&B project: {training_config.wandb_project}")
    if training_config.hf_hub_repo:
        click.echo(f"   HF repository: {training_config.hf_hub_repo}")

    if dry_run:
        click.echo("\n🏃 Dry run complete - configuration shown above")
        return

    # Load dataset
    click.echo(f"\n📂 Loading dataset from {dataset}")
    dataset_obj = _load_dataset(dataset)
    click.echo(f"   Training examples: {len(dataset_obj)}")

    eval_dataset_obj = None
    if eval_dataset:
        click.echo(f"📂 Loading evaluation dataset from {eval_dataset}")
        eval_dataset_obj = _load_dataset(eval_dataset)
        click.echo(f"   Evaluation examples: {len(eval_dataset_obj)}")

    # Initialize training wrapper
    click.echo("\n🔧 Initializing training wrapper...")
    wrapper = TrainingWrapper(training_config)

    # Start training
    try:
        click.echo("\n🏋️  Starting training...")
        results = wrapper.train(
            dataset=dataset_obj,
            eval_dataset=eval_dataset_obj,
            resume_from_checkpoint=resume_from,
        )

        # Show results
        click.echo("\n✅ Training completed successfully!")
        click.echo(f"   Final loss: {results['train_result'].training_loss:.4f}")
        click.echo(f"   Total steps: {results['train_result'].global_step}")
        click.echo(f"   Training time: {results['training_time']:.2f}s")
        click.echo(f"   Model saved to: {results['final_model_path']}")

        # Show analysis summary
        if wrapper.wandb_integration and wrapper.wandb_integration.is_active:
            click.echo(f"   W&B run: {wrapper.wandb_integration.get_run_url()}")

        if wrapper.hf_integration:
            click.echo(f"   HF repository: https://huggingface.co/{training_config.hf_hub_repo}")

        click.echo(f"\n📊 Run analysis with: training-lens analyze {training_config.output_dir}")

    except KeyboardInterrupt:
        click.echo("\n⏹️  Training interrupted by user")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        raise click.ClickException(f"Training failed: {e}")


def _load_dataset(dataset_path: str):
    """Load dataset from file."""
    import json

    import pandas as pd
    from datasets import Dataset

    dataset_path_obj = Path(dataset_path)

    if dataset_path_obj.suffix.lower() == ".json":
        with open(dataset_path_obj, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            raise ValueError("JSON file must contain a list of examples")

    elif dataset_path_obj.suffix.lower() == ".jsonl":
        data = []
        with open(dataset_path_obj, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        dataset = Dataset.from_list(data)

    elif dataset_path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path_obj)
        dataset = Dataset.from_pandas(df)

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path_obj.suffix}")

    return dataset
