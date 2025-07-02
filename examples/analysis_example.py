#!/usr/bin/env python3
"""
Analysis Example with Training Lens

This example demonstrates the two main workflows for analyzing training results:
(A) Getting raw data for custom analysis
(B) Using specialized tools for pre-built summaries
"""

import json
import tempfile
from pathlib import Path

from training_lens.analysis import CheckpointAnalyzer, GradientAnalyzer, StandardReports, WeightAnalyzer


def create_mock_training_data(output_dir: Path):
    """Create realistic mock training checkpoints for demonstration."""
    print("🔧 Creating mock training data...")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create mock checkpoints for steps 1, 10, 25, 50, 100
    steps = [1, 10, 25, 50, 100]
    checkpoint_info = []

    import numpy as np

    for step in steps:
        checkpoint_dir = checkpoints_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create realistic training metadata
        # Loss decreases over time with some noise
        base_loss = 3.0 * np.exp(-step / 50) + 0.5
        train_loss = base_loss + np.random.normal(0, 0.05)
        eval_loss = base_loss * 1.1 + np.random.normal(0, 0.08)

        metadata = {
            "step": step,
            "epoch": step / 100.0,
            "learning_rate": 2e-4 * (0.98 ** (step // 10)),
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "grad_norm": float(1.2 + np.random.normal(0, 0.3)),
            "timestamp": f"2024-01-01T10:{step:02d}:00",
        }

        # Save metadata
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create comprehensive training lens data
        training_lens_data = {
            # Gradient cosine similarities (showing convergence pattern)
            "gradient_cosine_similarities": [
                float(0.7 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.05)) for i in range(step)
            ],
            # Weight evolution statistics
            "weight_stats_history": [
                {
                    "step": s,
                    "overall_norm": float(4.5 + 0.3 * np.sin(s * 0.02) + np.random.normal(0, 0.1)),
                    "overall_mean": float(0.01 * np.random.normal(0, 1)),
                    "overall_std": float(0.8 + 0.1 * np.random.normal(0, 1)),
                    "layer_norms": {
                        f"layer_{i}": float(1.0 + i * 0.2 + np.random.normal(0, 0.05)) for i in range(6)  # 6 layers
                    },
                }
                for s in range(max(1, step - 5), step + 1)  # Last few steps
            ],
            # Layer information
            "layer_names": [f"layer_{i}" for i in range(6)],
            # Additional analysis data
            "cosine_similarity_trend": {
                "recent_mean": float(0.75 + np.random.normal(0, 0.1)),
                "recent_std": float(0.15 + np.random.normal(0, 0.02)),
                "trend_direction": "increasing" if step > 50 else "stable",
                "total_samples": step,
            },
        }

        # Save training lens data
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

    print(f"   Created {len(checkpoint_info)} checkpoints: {[cp['step'] for cp in checkpoint_info]}")
    return checkpoints_dir


def demonstrate_workflow_a_raw_data_access(analyzer: CheckpointAnalyzer, output_dir: Path):
    """Demonstrate Workflow A: Getting raw data for custom analysis."""
    print("\\n" + "=" * 60)
    print("🔬 WORKFLOW A: Raw Data Access for Custom Analysis")
    print("=" * 60)

    # 1. Access raw metrics for specific checkpoints
    print("\\n📊 1. Accessing Raw Metrics:")
    print("-" * 30)

    steps_to_analyze = [10, 50, 100]
    raw_data = {}

    for step in steps_to_analyze:
        metrics = analyzer.load_checkpoint_metrics(step)
        if metrics:
            raw_data[step] = metrics
            print(f"✅ Step {step}: {len(metrics)} data types available")
            print(f"   - Gradient similarities: {len(metrics.get('gradient_cosine_similarities', []))} values")
            print(f"   - Weight statistics: {len(metrics.get('weight_stats_history', []))} entries")
        else:
            print(f"❌ Step {step}: No data found")

    # 2. Export all raw data for external tools
    print("\\n💾 2. Exporting Raw Data:")
    print("-" * 30)

    raw_export_dir = output_dir / "raw_exports"
    exported_files = analyzer.export_raw_data(raw_export_dir)

    print(f"📁 Exported to: {raw_export_dir}")
    for data_type, file_path in exported_files.items():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"   - {data_type}: {file_path.name} ({size_mb:.2f} MB)")

    # 3. Custom analysis example
    print("\\n🧮 3. Custom Analysis Example:")
    print("-" * 30)

    if raw_data:
        # Analyze gradient convergence patterns across checkpoints
        all_similarities = []
        for step, data in raw_data.items():
            similarities = data.get("gradient_cosine_similarities", [])
            all_similarities.extend(similarities)

        if all_similarities:
            import numpy as np

            mean_sim = np.mean(all_similarities)
            std_sim = np.std(all_similarities)
            trend = "improving" if all_similarities[-10:] > all_similarities[:10] else "stable"

            print("   📈 Gradient Analysis Results:")
            print(f"      Mean cosine similarity: {mean_sim:.3f}")
            print(f"      Std deviation: {std_sim:.3f}")
            print(f"      Trend: {trend!r}")
            print(f"      Total data points: {len(all_similarities)}")

    print("\\n💡 Use Case: Perfect for researchers who want to:")
    print("   • Build custom analysis algorithms")
    print("   • Export data to external tools (R, MATLAB, etc.)")
    print("   • Integrate with existing ML pipelines")
    print("   • Perform novel research analysis")


def demonstrate_workflow_b_specialized_summaries(analyzer: CheckpointAnalyzer):
    """Demonstrate Workflow B: Using specialized tools for pre-built summaries."""
    print("\\n" + "=" * 60)
    print("📋 WORKFLOW B: Specialized Tools for Pre-built Summaries")
    print("=" * 60)

    # 1. Executive Summary (for stakeholders)
    print("\\n👔 1. Executive Summary (for business stakeholders):")
    print("-" * 50)

    reports = StandardReports(analyzer)
    exec_summary = reports.generate_executive_summary()

    if exec_summary and "training_overview" in exec_summary:
        overview = exec_summary["training_overview"]
        performance = exec_summary.get("performance_metrics", {})
        health = exec_summary.get("model_health", {})

        print("   📊 Training Overview:")
        print(f"      Total steps: {overview.get('training_steps', 'N/A')}")
        print(f"      Checkpoints: {overview.get('total_checkpoints', 'N/A')}")

        if performance:
            print("   📈 Performance:")
            loss_improvement = performance.get("loss_improvement", {})
            if loss_improvement:
                print(f"      Loss improvement: {loss_improvement.get('relative_percent', 0):.1f}%")

        print("   🏥 Model Health:")
        print(f"      Gradient health: {health.get('gradient_health', 'Unknown')}")
        print(f"      Weight stability: {health.get('weight_stability', 'Unknown')}")

        recommendations = exec_summary.get("recommendations", [])
        if recommendations:
            print("   💡 Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"      • {rec}")

    # 2. Deep Gradient Analysis
    print("\\n🎯 2. Deep Gradient Analysis (for ML engineers):")
    print("-" * 50)

    # Get gradient data from a middle checkpoint
    gradient_data = analyzer.load_checkpoint_metrics(50)
    if gradient_data:
        grad_analyzer = GradientAnalyzer(gradient_data)
        grad_report = grad_analyzer.generate_gradient_report()

        if grad_report:
            consistency = grad_report.get("consistency_analysis", {})
            anomalies = grad_report.get("anomaly_detection", {})
            assessment = grad_report.get("overall_assessment", {})

            print("   🔄 Gradient Consistency:")
            print(f"      Consistency score: {consistency.get('consistency_score', 0):.3f}")
            print(f"      Consistency level: {consistency.get('consistency_level', 'Unknown')}")

            print("   ⚠️  Anomaly Detection:")
            print(f"      Anomalies found: {anomalies.get('anomaly_count', 0)}")
            print(f"      Severity score: {anomalies.get('severity_score', 0):.3f}")

            print("   🧠 Overall Assessment:")
            print(f"      Gradient health: {assessment.get('gradient_health', 'Unknown')}")

    # 3. Weight Evolution Analysis
    print("\\n⚖️  3. Weight Evolution Analysis (for model optimization):")
    print("-" * 50)

    if gradient_data:  # Same data contains weight information
        weight_analyzer = WeightAnalyzer(gradient_data)
        weight_report = weight_analyzer.generate_weight_report()

        if weight_report:
            evolution = weight_report.get("evolution_analysis", {})
            layer_analysis = weight_report.get("layer_analysis", {})

            if evolution:
                stability = evolution.get("stability_assessment", {})
                print("   📊 Weight Evolution:")
                print(f"      Stability level: {stability.get('stability_level', 'Unknown')}")
                print(f"      Stability score: {stability.get('stability_score', 0):.3f}")

            if layer_analysis:
                insights = layer_analysis.get("cross_layer_insights", {})
                stability_summary = insights.get("stability_summary", {})
                print("   🔍 Layer Analysis:")
                print(f"      Analyzed layers: {insights.get('layer_count', 0)}")
                print(f"      Mean stability: {stability_summary.get('mean_stability_score', 0):.3f}")

    # 4. Training Diagnostics
    print("\\n🏥 4. Training Diagnostics (health check):")
    print("-" * 50)

    diagnostics = reports.generate_training_diagnostics()
    if diagnostics:
        print(f"   Overall health: {diagnostics.get('overall_health', 'Unknown')}")

        critical_issues = diagnostics.get("critical_issues", [])
        if critical_issues:
            print(f"   🚨 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues[:2]:  # Show top 2
                print(f"      • {issue.get('description', 'Unknown issue')}")

        warnings = diagnostics.get("warnings", [])
        if warnings:
            print(f"   ⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:2]:  # Show top 2
                print(f"      • {warning.get('description', 'Unknown warning')}")

    print("\\n💡 Use Case: Perfect for teams who want:")
    print("   • Ready-to-use insights without coding")
    print("   • Executive summaries for stakeholders")
    print("   • Technical reports for ML engineers")
    print("   • Automated health checks and diagnostics")


def main():
    """Main function demonstrating both analysis workflows."""
    print("🔍 Training Lens Analysis Workflows Example")
    print("=" * 60)
    print("This example shows the two main ways to analyze training results:")
    print("(A) Raw data access for custom analysis")
    print("(B) Specialized tools for pre-built summaries")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create mock training data
        checkpoints_dir = create_mock_training_data(output_dir)

        # Initialize analyzer
        print("\\n🔧 Initializing CheckpointAnalyzer...")
        analyzer = CheckpointAnalyzer(checkpoints_dir)

        if not analyzer.checkpoints_info:
            print("❌ No checkpoints found!")
            return 1

        print(f"✅ Loaded {len(analyzer.checkpoints_info)} checkpoints")

        # Demonstrate both workflows
        demonstrate_workflow_a_raw_data_access(analyzer, output_dir)
        demonstrate_workflow_b_specialized_summaries(analyzer)

        # Show what files were created
        print("\\n" + "=" * 60)
        print("📁 Generated Files:")
        print("=" * 60)

        all_files = list(output_dir.rglob("*"))
        for file_path in sorted(all_files):
            if file_path.is_file():
                size = file_path.stat().st_size
                rel_path = file_path.relative_to(output_dir)
                print(f"   {rel_path} ({size:,} bytes)")

    print("\\n✨ Analysis workflows completed successfully!")
    print("\\nNext steps:")
    print("   • Try with your own training checkpoints")
    print("   • Explore the CLI commands: training-lens analyze ./checkpoints")
    print("   • Export data: training-lens export ./checkpoints --output ./data")
    print("   • Check the advanced_training_example.py for end-to-end workflow")

    return 0


if __name__ == "__main__":
    exit(main())
