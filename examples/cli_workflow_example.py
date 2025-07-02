#!/usr/bin/env python3
"""
CLI Workflow Example with Training Lens

This example demonstrates how to use Training Lens CLI commands
for automated workflows, scripting, and CI/CD integration.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import yaml


def create_sample_config(config_path: Path):
    """Create a sample configuration file for training."""
    print(f"📝 Creating sample configuration: {config_path}")

    config = {
        "model_name": "microsoft/DialoGPT-small",
        "training_method": "lora",
        "max_seq_length": 512,
        "load_in_4bit": True,
        # LoRA settings
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        # Training parameters
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_steps": 50,  # Short for demo
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 5,
        # Checkpoint configuration (defaults to every step)
        "checkpoint_interval": 10,  # Save every 10 steps for demo
        "save_strategy": "steps",
        "save_steps": 10,
        # Analysis settings
        "capture_gradients": True,
        "capture_weights": True,
        "capture_activations": False,
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"   ✅ Configuration saved to {config_path}")
    return config


def create_sample_dataset(dataset_path: Path):
    """Create a sample dataset file for training."""
    print(f"📊 Creating sample dataset: {dataset_path}")

    # Create JSONL dataset with conversation format
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a field of AI that enables computers to learn from data.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain neural networks"},
                {
                    "role": "assistant",
                    "content": "Neural networks are computing systems inspired by biological neural networks.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How does training work?"},
                {
                    "role": "assistant",
                    "content": "Training involves adjusting model parameters using data and optimization algorithms.",
                },
            ]
        },
    ] * 15  # Repeat for more training data

    with open(dataset_path, "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\\n")

    print(f"   ✅ Dataset saved with {len(conversations)} examples")
    return dataset_path


def run_cli_command(command: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def demonstrate_cli_init():
    """Demonstrate CLI configuration initialization."""
    print("\\n" + "=" * 60)
    print("🚀 CLI INITIALIZATION")
    print("=" * 60)

    print("\\n1️⃣  Initialize Basic Configuration:")
    print("-" * 40)
    print("   Command: training-lens init --config-template basic")
    print("   Creates: training_config.yaml with basic settings")
    print("   Use case: Quick start for new users")

    print("\\n2️⃣  Initialize Advanced Configuration:")
    print("-" * 40)
    print("   Command: training-lens init --config-template advanced")
    print("   Creates: training_config.yaml with advanced features")
    print("   Use case: Production training with integrations")

    print("\\n3️⃣  Initialize Research Configuration:")
    print("-" * 40)
    print("   Command: training-lens init --config-template research")
    print("   Creates: training_config.yaml with detailed monitoring")
    print("   Use case: Research experiments with full data capture")

    print("\\n4️⃣  Custom Output Location:")
    print("-" * 40)
    print("   Command: training-lens init --output custom_config.yaml")
    print("   Creates: custom_config.yaml instead of default name")
    print("   Use case: Multiple configurations for different experiments")


def demonstrate_cli_training(config_path: Path, dataset_path: Path, output_dir: Path):
    """Demonstrate CLI training workflow."""
    print("\\n" + "=" * 60)
    print("🏋️  CLI TRAINING WORKFLOW")
    print("=" * 60)

    print("\\n1️⃣  Basic Training Command:")
    print("-" * 40)
    print(f"   Command: training-lens train --config {config_path}")
    print("   Note: This would start actual training (skipped in demo)")
    print("   Output: Creates checkpoints in config.output_dir")

    print("\\n2️⃣  Training with Custom Dataset:")
    print("-" * 40)
    print(f"   Command: training-lens train --config {config_path} --dataset {dataset_path}")
    print("   Note: Overrides dataset path in config")
    print("   Use case: Testing different datasets with same config")

    print("\\n3️⃣  Training with Verbose Output:")
    print("-" * 40)
    print(f"   Command: training-lens train --config {config_path} --verbose")
    print("   Output: Detailed logging for debugging")
    print("   Use case: Troubleshooting training issues")

    print("\\n4️⃣  Training with Custom Output:")
    print("-" * 40)
    print(f"   Command: training-lens train --config {config_path} --output-dir {output_dir}")
    print("   Output: Saves results to specified directory")
    print("   Use case: Organizing multiple experiments")

    # For demonstration, create mock training results
    print("\\n📁 Creating Mock Training Results for CLI Demo:")
    print("-" * 50)

    # Create mock checkpoints directory structure
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create a few mock checkpoints
    steps = [10, 20, 30, 40, 50]
    checkpoint_info = []

    for step in steps:
        checkpoint_dir = checkpoints_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Mock metadata
        metadata = {
            "step": step,
            "epoch": step / 50.0,
            "learning_rate": 2e-4 * (0.9 ** (step // 10)),
            "train_loss": 2.0 * (0.95 ** (step / 10)) + 0.1,
            "grad_norm": 1.0 + (step * 0.01),
            "timestamp": f"2024-01-01T10:{step:02d}:00",
        }

        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        checkpoint_info.append(
            {
                "step": step,
                "path": str(checkpoint_dir),
                "timestamp": metadata["timestamp"],
                "metadata": metadata,
            }
        )

    # Save checkpoint index
    with open(checkpoints_dir / "checkpoint_index.json", "w") as f:
        json.dump(checkpoint_info, f, indent=2)

    print(f"   ✅ Created {len(steps)} mock checkpoints in {checkpoints_dir}")
    return checkpoints_dir


def demonstrate_cli_analysis(checkpoints_dir: Path, output_dir: Path):
    """Demonstrate CLI analysis commands."""
    print("\\n" + "=" * 60)
    print("🔍 CLI ANALYSIS WORKFLOW")
    print("=" * 60)

    print("\\n1️⃣  Quick Checkpoint Summary:")
    print("-" * 40)
    command = ["training-lens", "summary", str(checkpoints_dir)]
    print(f"   Command: {' '.join(command)}")

    # Run the actual command
    returncode, stdout, stderr = run_cli_command(command)
    if returncode == 0:
        print("   ✅ Output:")
        print("   " + "\\n   ".join(stdout.strip().split("\\n")))
    else:
        print(f"   ❌ Command failed: {stderr}")

    print("\\n2️⃣  Summary in Different Formats:")
    print("-" * 40)
    formats = ["json", "yaml", "csv"]
    for fmt in formats:
        output_file = output_dir / f"summary.{fmt}"
        command = ["training-lens", "summary", str(checkpoints_dir), "--format", fmt, "--output", str(output_file)]
        print(f"   Command: {' '.join(command)}")

        returncode, stdout, stderr = run_cli_command(command)
        if returncode == 0 and output_file.exists():
            size = output_file.stat().st_size
            print(f"   ✅ Created: {output_file.name} ({size} bytes)")
        else:
            print(f"   ❌ Failed to create {fmt} summary")

    print("\\n3️⃣  Comprehensive Analysis:")
    print("-" * 40)
    analysis_dir = output_dir / "analysis"
    command = ["training-lens", "analyze", str(checkpoints_dir), "--output", str(analysis_dir)]
    print(f"   Command: {' '.join(command)}")
    print("   Note: This would generate detailed analysis reports")
    print("   Output: Executive summaries, technical reports, diagnostics")


def demonstrate_cli_export(checkpoints_dir: Path, output_dir: Path):
    """Demonstrate CLI data export commands."""
    print("\\n" + "=" * 60)
    print("💾 CLI DATA EXPORT WORKFLOW")
    print("=" * 60)

    print("\\n1️⃣  Export All Data (All Formats):")
    print("-" * 40)
    export_all_dir = output_dir / "export_all"
    command = ["training-lens", "export", str(checkpoints_dir), "--output", str(export_all_dir), "--format", "all"]
    print(f"   Command: {' '.join(command)}")

    returncode, stdout, stderr = run_cli_command(command)
    if returncode == 0:
        print("   ✅ Export successful!")
        # List exported files
        if export_all_dir.exists():
            files = list(export_all_dir.glob("*"))
            for file_path in sorted(files):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"      - {file_path.name} ({size} bytes)")
    else:
        print(f"   ❌ Export failed: {stderr}")

    print("\\n2️⃣  Export Specific Data Types:")
    print("-" * 40)
    data_types = ["gradients", "weights", "metrics"]
    for data_type in data_types:
        export_dir = output_dir / f"export_{data_type}"
        command = [
            "training-lens",
            "export",
            str(checkpoints_dir),
            "--output",
            str(export_dir),
            "--data-type",
            data_type,
            "--format",
            "csv",
        ]
        print(f"   Command: {' '.join(command)}")

        returncode, stdout, stderr = run_cli_command(command)
        if returncode == 0:
            print(f"   ✅ Exported {data_type} data")
        else:
            print(f"   ❌ Failed to export {data_type}: {stderr}")

    print("\\n3️⃣  Export Specific Checkpoints:")
    print("-" * 40)
    export_specific_dir = output_dir / "export_specific"
    command = [
        "training-lens",
        "export",
        str(checkpoints_dir),
        "--output",
        str(export_specific_dir),
        "--steps",
        "10,30,50",
        "--compress",
    ]
    print(f"   Command: {' '.join(command)}")
    print("   Result: Exports only checkpoints 10, 30, 50 with compression")

    print("\\n4️⃣  Export for External Tools:")
    print("-" * 40)
    print("   # For pandas/Python analysis")
    print("   training-lens export ./checkpoints --output ./data --format csv")
    print("   ")
    print("   # For NumPy/scientific computing")
    print("   training-lens export ./checkpoints --output ./arrays --format numpy")
    print("   ")
    print("   # For big data analysis")
    print("   training-lens export ./checkpoints --output ./parquet --format parquet --compress")


def demonstrate_cli_info():
    """Demonstrate CLI system information commands."""
    print("\\n" + "=" * 60)
    print("ℹ️  CLI SYSTEM INFORMATION")
    print("=" * 60)

    print("\\n1️⃣  System Information:")
    print("-" * 40)
    command = ["training-lens", "info"]
    print(f"   Command: {' '.join(command)}")

    returncode, stdout, stderr = run_cli_command(command)
    if returncode == 0:
        print("   ✅ Output:")
        print("   " + "\\n   ".join(stdout.strip().split("\\n")))
    else:
        print(f"   ❌ Command failed: {stderr}")

    print("\\n2️⃣  Version Information:")
    print("-" * 40)
    command = ["training-lens", "--version"]
    print(f"   Command: {' '.join(command)}")
    print("   Use case: Check installed version for bug reports")

    print("\\n3️⃣  Help and Documentation:")
    print("-" * 40)
    print("   Command: training-lens --help")
    print("   Command: training-lens train --help")
    print("   Command: training-lens export --help")
    print("   Use case: Get detailed help for any command")


def demonstrate_automation_scripting():
    """Show examples of automation and scripting with CLI."""
    print("\\n" + "=" * 60)
    print("🤖 AUTOMATION & SCRIPTING EXAMPLES")
    print("=" * 60)

    print("\\n1️⃣  Bash Script for Multiple Experiments:")
    print("-" * 50)
    print("   #!/bin/bash")
    print("   # Run multiple training experiments")
    print("   ")
    print("   experiments=('basic' 'advanced' 'research')")
    print("   ")
    print('   for exp in \\"${experiments[@]}\\"; do')
    print('       echo \\"Running $exp experiment...\\"')
    print("       ")
    print("       # Initialize config")
    print("       training-lens init --config-template $exp --output ${exp}_config.yaml")
    print("       ")
    print("       # Run training")
    print("       training-lens train --config ${exp}_config.yaml --output-dir ./results/$exp")
    print("       ")
    print("       # Generate analysis")
    print("       training-lens analyze ./results/$exp/checkpoints --output ./analysis/$exp")
    print("       ")
    print("       # Export data")
    print("       training-lens export ./results/$exp/checkpoints --output ./data/$exp --format all")
    print("   done")

    print("\\n2️⃣  Python Script for Automated Pipeline:")
    print("-" * 50)
    print("   import subprocess")
    print("   from pathlib import Path")
    print("   ")
    print("   def run_training_pipeline(config_path, output_dir):")
    print('       \\"\\"\\"Automated training and analysis pipeline.\\"\\"\\"')
    print("       ")
    print("       # Step 1: Train model")
    print("       subprocess.run(['training-lens', 'train', '--config', config_path,")
    print("                      '--output-dir', output_dir])")
    print("       ")
    print("       # Step 2: Analyze results")
    print("       checkpoints_dir = Path(output_dir) / 'checkpoints'")
    print("       analysis_dir = Path(output_dir) / 'analysis'")
    print("       subprocess.run(['training-lens', 'analyze', str(checkpoints_dir),")
    print("                      '--output', str(analysis_dir)])")
    print("       ")
    print("       # Step 3: Export for downstream analysis")
    print("       export_dir = Path(output_dir) / 'exported_data'")
    print("       subprocess.run(['training-lens', 'export', str(checkpoints_dir),")
    print("                      '--output', str(export_dir), '--format', 'all'])")

    print("\\n3️⃣  CI/CD Integration (GitHub Actions):")
    print("-" * 50)
    print("   name: Training Pipeline")
    print("   on: [push, pull_request]")
    print("   ")
    print("   jobs:")
    print("     train-and-analyze:")
    print("       runs-on: ubuntu-latest")
    print("       steps:")
    print("       - uses: actions/checkout@v3")
    print("       ")
    print("       - name: Setup Python")
    print("         uses: actions/setup-python@v4")
    print("         with:")
    print("           python-version: '3.9'")
    print("       ")
    print("       - name: Install Training Lens")
    print("         run: pip install training-lens")
    print("       ")
    print("       - name: Run Training")
    print("         run: |")
    print("           training-lens train --config config.yaml")
    print("           training-lens analyze ./training_output/checkpoints")
    print("           training-lens export ./training_output/checkpoints --output ./results")
    print("       ")
    print("       - name: Upload Results")
    print("         uses: actions/upload-artifact@v3")
    print("         with:")
    print("           name: training-results")
    print("           path: ./results/")

    print("\\n4️⃣  Docker Integration:")
    print("-" * 50)
    print("   # Dockerfile")
    print("   FROM python:3.9")
    print("   ")
    print("   RUN pip install training-lens")
    print("   ")
    print("   WORKDIR /app")
    print("   COPY config.yaml .")
    print("   COPY dataset.jsonl .")
    print("   ")
    print("   # Run training pipeline")
    print("   CMD ['training-lens', 'train', '--config', 'config.yaml']")
    print("   ")
    print("   # Usage:")
    print("   # docker build -t my-training .")
    print("   # docker run -v $(pwd)/output:/app/output my-training")


def main():
    """Main function demonstrating CLI workflows."""
    print("⚡ Training Lens CLI Workflow Example")
    print("=" * 60)
    print("This example demonstrates how to use Training Lens CLI commands")
    print("for automated workflows, scripting, and production deployments.")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create sample files
        config_path = output_dir / "demo_config.yaml"
        dataset_path = output_dir / "demo_dataset.jsonl"

        create_sample_config(config_path)
        create_sample_dataset(dataset_path)

        # Demonstrate CLI workflows
        demonstrate_cli_init()
        checkpoints_dir = demonstrate_cli_training(config_path, dataset_path, output_dir)
        demonstrate_cli_analysis(checkpoints_dir, output_dir)
        demonstrate_cli_export(checkpoints_dir, output_dir)
        demonstrate_cli_info()
        demonstrate_automation_scripting()

        # Show generated files
        print("\\n" + "=" * 60)
        print("📁 Generated Files (Demo):")
        print("=" * 60)

        all_files = list(output_dir.rglob("*"))
        file_count = sum(1 for f in all_files if f.is_file())

        print(f"   Total files created: {file_count}")
        print("   Key directories:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                file_count = sum(1 for f in item.rglob("*") if f.is_file())
                print(f"      📁 {item.name}/ ({file_count} files)")

    print("\\n" + "=" * 60)
    print("✨ CLI Workflow Examples Completed!")
    print("=" * 60)
    print("\\nKey CLI Commands:")
    print("   🚀 training-lens init --config-template [basic|advanced|research]")
    print("   🏋️  training-lens train --config config.yaml")
    print("   🔍 training-lens analyze ./checkpoints")
    print("   💾 training-lens export ./checkpoints --output ./data")
    print("   ℹ️  training-lens info")
    print("   📋 training-lens summary ./checkpoints")
    print("\\nAutomation Benefits:")
    print("   • Scriptable workflows for reproducibility")
    print("   • CI/CD integration for automated training")
    print("   • Docker support for containerized training")
    print("   • Easy integration with existing ML pipelines")

    return 0


if __name__ == "__main__":
    exit(main())
