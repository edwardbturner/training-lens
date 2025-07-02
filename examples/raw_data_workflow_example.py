#!/usr/bin/env python3
"""
Raw Data Workflow Example with Training Lens

This example demonstrates how to access and work with raw training data
for custom analysis, research, and integration with external tools.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from training_lens.analysis import CheckpointAnalyzer


def create_sample_training_run(output_dir: Path):
    """Create a sample training run with realistic data for demonstration."""
    print("🔧 Creating sample training run...")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a training run with steps: 1, 5, 10, 20, 50, 100
    steps = [1, 5, 10, 20, 50, 100]
    checkpoint_info = []

    for step in steps:
        checkpoint_dir = checkpoints_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Realistic training progression
        progress = step / 100.0
        base_loss = 2.5 * np.exp(-progress * 2) + 0.3

        metadata = {
            "step": step,
            "epoch": progress,
            "learning_rate": 2e-4 * (0.95 ** (step // 10)),
            "train_loss": float(base_loss + np.random.normal(0, 0.02)),
            "eval_loss": float(base_loss * 1.15 + np.random.normal(0, 0.03)),
            "grad_norm": float(0.8 + np.random.normal(0, 0.1)),
            "timestamp": f"2024-01-01T{10 + step//10:02d}:00:00",
        }

        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Rich training lens data
        training_lens_data = {
            # Gradient cosine similarities showing training dynamics
            "gradient_cosine_similarities": [
                float(0.6 + 0.3 * np.tanh(i * 0.05) + np.random.normal(0, 0.03)) for i in range(step)
            ],
            # Weight evolution over time
            "weight_stats_history": [
                {
                    "step": s,
                    "overall_norm": float(3.2 + 0.4 * np.sin(s * 0.1) + np.random.normal(0, 0.05)),
                    "overall_mean": float(np.random.normal(0, 0.001)),
                    "overall_std": float(0.7 + 0.1 * np.sin(s * 0.05) + np.random.normal(0, 0.02)),
                    "layer_norms": {
                        f"transformer.layer.{i}.attention.query": float(1.2 + i * 0.1 + np.random.normal(0, 0.02))
                        for i in range(8)  # 8 transformer layers
                    },
                }
                for s in range(max(1, step - 3), step + 1)  # Last few steps
            ],
            # Layer names for reference
            "layer_names": [f"transformer.layer.{i}.attention.query" for i in range(8)],
            # Cosine similarity trend analysis
            "cosine_similarity_trend": {
                "recent_mean": float(0.75 + np.random.normal(0, 0.05)),
                "recent_std": float(0.12 + np.random.normal(0, 0.01)),
                "trend_direction": "increasing" if step > 20 else "stabilizing",
                "total_samples": step,
            },
        }

        # Save as PyTorch tensor file
        import torch

        torch.save(training_lens_data, checkpoint_dir / "additional_data.pt")

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

    print(f"   Created training run with {len(checkpoint_info)} checkpoints")
    return checkpoints_dir


def demonstrate_raw_data_access(analyzer: CheckpointAnalyzer):
    """Show different ways to access raw training data."""
    print("\\n" + "=" * 60)
    print("📊 RAW DATA ACCESS METHODS")
    print("=" * 60)

    # Method 1: Access specific checkpoint data
    print("\\n1️⃣  Access Specific Checkpoint Data:")
    print("-" * 40)

    step_50_data = analyzer.load_checkpoint_metrics(50)
    if step_50_data:
        print("✅ Step 50 data loaded successfully")
        print(f"   Data types: {list(step_50_data.keys())}")
        print(f"   Gradient similarities: {len(step_50_data.get('gradient_cosine_similarities', []))} values")
        print(f"   Weight history entries: {len(step_50_data.get('weight_stats_history', []))}")

        # Show sample gradient data
        similarities = step_50_data.get("gradient_cosine_similarities", [])
        if similarities:
            print(f"   Sample gradient similarities: {similarities[:5]} ... {similarities[-3:]}")

    # Method 2: Iterate through all checkpoints
    print("\\n2️⃣  Iterate Through All Checkpoints:")
    print("-" * 40)

    all_losses = []
    all_grad_norms = []

    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metadata = checkpoint_info.get("metadata", {})

        train_loss = metadata.get("train_loss")
        grad_norm = metadata.get("grad_norm")

        if train_loss is not None:
            all_losses.append((step, train_loss))
        if grad_norm is not None:
            all_grad_norms.append((step, grad_norm))

        print(f"   Step {step:3d}: Loss={train_loss:.4f}, GradNorm={grad_norm:.4f}")

    # Method 3: Access raw checkpoint files directly
    print("\\n3️⃣  Direct File Access:")
    print("-" * 40)

    step_100_path = None
    for checkpoint_info in analyzer.checkpoints_info:
        if checkpoint_info["step"] == 100:
            step_100_path = Path(checkpoint_info["path"])
            break

    if step_100_path:
        print(f"   Checkpoint directory: {step_100_path}")
        files = list(step_100_path.glob("*"))
        for file_path in files:
            size = file_path.stat().st_size
            print(f"   - {file_path.name}: {size:,} bytes")

        # Load raw PyTorch data
        if (step_100_path / "additional_data.pt").exists():
            import torch

            raw_data = torch.load(step_100_path / "additional_data.pt")
            print(f"   Raw PyTorch data keys: {list(raw_data.keys())}")


def demonstrate_data_export_formats(analyzer: CheckpointAnalyzer, output_dir: Path):
    """Show different export formats for external analysis."""
    print("\\n" + "=" * 60)
    print("💾 DATA EXPORT FORMATS")
    print("=" * 60)

    export_dir = output_dir / "exported_data"

    # Export using built-in method
    print("\\n1️⃣  Built-in Export Method:")
    print("-" * 40)

    exported_files = analyzer.export_raw_data(export_dir)
    for data_type, file_path in exported_files.items():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"   ✅ {data_type}: {file_path.name} ({size_mb:.2f} MB)")

    # Custom CSV export for specific analysis
    print("\\n2️⃣  Custom CSV Export:")
    print("-" * 40)

    # Collect training progression data
    training_data = []
    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metadata = checkpoint_info.get("metadata", {})

        # Load metrics for gradient data
        metrics = analyzer.load_checkpoint_metrics(step)

        row = {
            "step": step,
            "epoch": metadata.get("epoch", 0),
            "train_loss": metadata.get("train_loss"),
            "eval_loss": metadata.get("eval_loss"),
            "learning_rate": metadata.get("learning_rate"),
            "grad_norm": metadata.get("grad_norm"),
        }

        # Add gradient cosine similarity statistics
        if metrics and "gradient_cosine_similarities" in metrics:
            similarities = metrics["gradient_cosine_similarities"]
            if similarities:
                row["grad_cosine_mean"] = np.mean(similarities)
                row["grad_cosine_std"] = np.std(similarities)
                row["grad_cosine_latest"] = similarities[-1]

        training_data.append(row)

    # Export as CSV
    df = pd.DataFrame(training_data)
    csv_path = export_dir / "training_progression.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Training progression: {csv_path.name}")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Rows: {len(df)}")

    # Export gradient evolution as NumPy arrays
    print("\\n3️⃣  NumPy Array Export:")
    print("-" * 40)

    all_gradients = {}
    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metrics = analyzer.load_checkpoint_metrics(step)

        if metrics and "gradient_cosine_similarities" in metrics:
            similarities = metrics["gradient_cosine_similarities"]
            all_gradients[f"step_{step}"] = np.array(similarities)

    if all_gradients:
        gradient_arrays_path = export_dir / "gradient_arrays.npz"
        np.savez_compressed(gradient_arrays_path, **all_gradients)
        print(f"   ✅ Gradient arrays: {gradient_arrays_path.name}")
        print(f"      Arrays: {list(all_gradients.keys())}")
        print(f"      Total data points: {sum(len(arr) for arr in all_gradients.values())}")


def demonstrate_custom_analysis(analyzer: CheckpointAnalyzer):
    """Show examples of custom analysis you can perform with raw data."""
    print("\\n" + "=" * 60)
    print("🧮 CUSTOM ANALYSIS EXAMPLES")
    print("=" * 60)

    # Analysis 1: Training convergence analysis
    print("\\n1️⃣  Training Convergence Analysis:")
    print("-" * 40)

    losses = []
    steps = []

    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metadata = checkpoint_info.get("metadata", {})
        train_loss = metadata.get("train_loss")

        if train_loss is not None:
            steps.append(step)
            losses.append(train_loss)

    if len(losses) >= 3:
        # Calculate convergence metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100

        # Calculate smoothness (lower is smoother)
        loss_changes = [abs(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
        smoothness_score = np.mean(loss_changes)

        print(f"   📉 Loss Improvement: {improvement:.1f}%")
        print(f"   📊 Initial Loss: {initial_loss:.4f}")
        print(f"   📊 Final Loss: {final_loss:.4f}")
        print(f"   🌊 Smoothness Score: {smoothness_score:.4f} (lower = smoother)")

    # Analysis 2: Gradient stability analysis
    print("\\n2️⃣  Gradient Stability Analysis:")
    print("-" * 40)

    all_cosine_similarities = []
    step_labels = []

    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metrics = analyzer.load_checkpoint_metrics(step)

        if metrics and "gradient_cosine_similarities" in metrics:
            similarities = metrics["gradient_cosine_similarities"]
            if similarities:
                # Take the mean cosine similarity for this checkpoint
                mean_similarity = np.mean(similarities)
                all_cosine_similarities.append(mean_similarity)
                step_labels.append(step)

    if all_cosine_similarities:
        stability_score = np.mean(all_cosine_similarities)
        stability_variance = np.var(all_cosine_similarities)

        # Classification based on similarity
        if stability_score > 0.8:
            stability_level = "Excellent"
        elif stability_score > 0.6:
            stability_level = "Good"
        elif stability_score > 0.4:
            stability_level = "Fair"
        else:
            stability_level = "Poor"

        print(f"   🎯 Gradient Stability: {stability_level}")
        print(f"   📊 Mean Cosine Similarity: {stability_score:.3f}")
        print(f"   📊 Stability Variance: {stability_variance:.4f}")
        print(
            f"   📈 Trend: {'Improving' if all_cosine_similarities[-1] > all_cosine_similarities[0] else 'Declining'}"
        )

    # Analysis 3: Weight evolution analysis
    print("\\n3️⃣  Weight Evolution Analysis:")
    print("-" * 40)

    weight_norms = []
    for checkpoint_info in analyzer.checkpoints_info:
        step = checkpoint_info["step"]
        metrics = analyzer.load_checkpoint_metrics(step)

        if metrics and "weight_stats_history" in metrics:
            weight_history = metrics["weight_stats_history"]
            if weight_history:
                # Get the latest weight stats for this checkpoint
                latest_stats = weight_history[-1]
                overall_norm = latest_stats.get("overall_norm")
                if overall_norm is not None:
                    weight_norms.append((step, overall_norm))

    if len(weight_norms) >= 2:
        initial_norm = weight_norms[0][1]
        final_norm = weight_norms[-1][1]
        norm_change = (final_norm - initial_norm) / initial_norm * 100

        # Calculate weight stability
        norms_only = [norm for _, norm in weight_norms]
        weight_stability = 1.0 / (1.0 + np.std(norms_only))  # Higher = more stable

        print(f"   ⚖️  Weight Norm Change: {norm_change:+.1f}%")
        print(f"   📊 Initial Norm: {initial_norm:.3f}")
        print(f"   📊 Final Norm: {final_norm:.3f}")
        print(f"   🔒 Stability Score: {weight_stability:.3f} (higher = more stable)")


def demonstrate_integration_examples():
    """Show how to integrate raw data with external tools."""
    print("\\n" + "=" * 60)
    print("🔗 INTEGRATION WITH EXTERNAL TOOLS")
    print("=" * 60)

    print("\\n1️⃣  Integration with Pandas/Matplotlib:")
    print("-" * 40)
    print("   # Load exported CSV data")
    print("   import pandas as pd")
    print("   import matplotlib.pyplot as plt")
    print("   ")
    print("   df = pd.read_csv('training_progression.csv')")
    print("   ")
    print("   # Plot training curves")
    print("   plt.figure(figsize=(12, 4))")
    print("   plt.subplot(1, 3, 1)")
    print("   plt.plot(df['step'], df['train_loss'], label='Train')")
    print("   plt.plot(df['step'], df['eval_loss'], label='Eval')")
    print("   plt.legend()")
    print("   plt.title('Loss Curves')")

    print("\\n2️⃣  Integration with NumPy/SciPy:")
    print("-" * 40)
    print("   # Load gradient arrays")
    print("   import numpy as np")
    print("   from scipy import stats")
    print("   ")
    print("   data = np.load('gradient_arrays.npz')")
    print("   step_100_gradients = data['step_100']")
    print("   ")
    print("   # Statistical analysis")
    print("   correlation = stats.pearsonr(range(len(step_100_gradients)), step_100_gradients)")
    print("   print(f'Gradient trend correlation: {correlation[0]:.3f}')")

    print("\\n3️⃣  Integration with R (via CSV):")
    print("-" * 40)
    print("   # R code example")
    print("   library(ggplot2)")
    print("   ")
    print("   # Load data")
    print("   data <- read.csv('training_progression.csv')")
    print("   ")
    print("   # Advanced visualization")
    print("   ggplot(data, aes(x=step)) +")
    print("     geom_line(aes(y=train_loss), color='blue') +")
    print("     geom_ribbon(aes(ymin=train_loss-grad_cosine_std, ")
    print("                     ymax=train_loss+grad_cosine_std), alpha=0.3)")

    print("\\n4️⃣  Integration with MLflow:")
    print("-" * 40)
    print("   # Log metrics to MLflow")
    print("   import mlflow")
    print("   ")
    print("   with mlflow.start_run():")
    print("       for _, row in df.iterrows():")
    print("           mlflow.log_metric('train_loss', row['train_loss'], step=row['step'])")
    print("           mlflow.log_metric('grad_cosine_mean', row['grad_cosine_mean'], step=row['step'])")


def main():
    """Main function demonstrating raw data workflows."""
    print("🔬 Training Lens Raw Data Workflow Example")
    print("=" * 60)
    print("This example shows how to access, export, and analyze raw training data")
    print("for custom research and integration with external analysis tools.")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create sample training data
        checkpoints_dir = create_sample_training_run(output_dir)

        # Initialize analyzer
        print("\\n🔧 Initializing CheckpointAnalyzer...")
        analyzer = CheckpointAnalyzer(checkpoints_dir)

        if not analyzer.checkpoints_info:
            print("❌ No checkpoints found!")
            return 1

        print(f"✅ Loaded {len(analyzer.checkpoints_info)} checkpoints from training run")

        # Demonstrate different aspects of raw data workflow
        demonstrate_raw_data_access(analyzer)
        demonstrate_data_export_formats(analyzer, output_dir)
        demonstrate_custom_analysis(analyzer)
        demonstrate_integration_examples()

    print("\\n" + "=" * 60)
    print("✨ Raw Data Workflow Examples Completed!")
    print("=" * 60)
    print("\\nKey Takeaways:")
    print("   📊 Multiple ways to access raw training data")
    print("   💾 Various export formats for different tools")
    print("   🧮 Custom analysis possibilities are endless")
    print("   🔗 Easy integration with external analysis tools")
    print("\\nNext Steps:")
    print("   • Apply these patterns to your own training data")
    print("   • Try the CLI export: training-lens export ./checkpoints --output ./data")
    print("   • Explore advanced_training_example.py for end-to-end workflow")

    return 0


if __name__ == "__main__":
    exit(main())
