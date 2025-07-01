#!/usr/bin/env python3
"""
Analysis Example with Training Lens

This example demonstrates how to analyze existing training checkpoints
and generate comprehensive insights.
"""

import json
import tempfile
from pathlib import Path

from training_lens.analysis.checkpoint_analyzer import CheckpointAnalyzer
from training_lens.analysis.gradient_analyzer import GradientAnalyzer
from training_lens.analysis.reports import StandardReports
from training_lens.analysis.weight_analyzer import WeightAnalyzer


def create_mock_checkpoints(output_dir: Path):
    """Create mock checkpoint data for demonstration."""
    print("🔧 Creating mock checkpoint data...")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create mock checkpoint data
    import numpy as np

    checkpoint_info = []

    for step in [100, 200, 300, 400, 500]:
        checkpoint_dir = checkpoints_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Mock metadata
        metadata = {
            "step": step,
            "epoch": step / 100,
            "learning_rate": 2e-4 * (0.95 ** (step // 100)),
            "train_loss": 3.0 * np.exp(-step / 200) + 0.5 + np.random.normal(0, 0.1),
            "eval_loss": 3.2 * np.exp(-step / 200) + 0.6 + np.random.normal(0, 0.15),
            "grad_norm": 1.0 + np.random.normal(0, 0.2),
            "timestamp": f"2024-01-01T{10 + step//100:02d}:00:00",
        }

        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Mock training lens data
        training_lens_data = {
            "gradient_cosine_similarities": [
                0.8 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.1) for i in range(step // 10)
            ],
            "weight_stats_history": [
                {
                    "step": s,
                    "overall_norm": 5.0 + 0.5 * np.sin(s * 0.01) + np.random.normal(0, 0.1),
                    "overall_mean": 0.0 + np.random.normal(0, 0.01),
                    "overall_std": 1.0 + 0.1 * np.sin(s * 0.02) + np.random.normal(0, 0.05),
                    "layer_norms": {f"layer_{i}": 1.0 + i * 0.1 + np.random.normal(0, 0.05) for i in range(5)},
                }
                for s in range(max(1, step - 50), step + 1, 10)
            ],
        }

        # Save as torch-like format (JSON for this example)
        with open(checkpoint_dir / "additional_data.json", "w") as f:
            json.dump(training_lens_data, f, indent=2, default=str)

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

    print(f"   Created {len(checkpoint_info)} mock checkpoints")
    return checkpoints_dir


def demonstrate_basic_analysis(analyzer: CheckpointAnalyzer):
    """Demonstrate basic analysis capabilities."""
    print("\\n📊 Basic Analysis:")
    print("-" * 30)

    # List checkpoints
    checkpoints = analyzer.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        metadata = cp.get("metadata", {})
        print(f"  Step {cp['step']}: Loss {metadata.get('train_loss', 0):.3f}")

    # Training dynamics analysis
    print("\\n🔄 Training Dynamics:")
    dynamics = analyzer.analyze_training_dynamics()

    if "loss_analysis" in dynamics:
        loss_analysis = dynamics["loss_analysis"]
        print(f"  Initial loss: {loss_analysis.get('initial_loss', 0):.3f}")
        print(f"  Final loss: {loss_analysis.get('final_loss', 0):.3f}")
        print(f"  Loss reduction: {loss_analysis.get('loss_reduction_percentage', 0):.1f}%")

    if "training_efficiency" in dynamics:
        efficiency = dynamics["training_efficiency"]
        print(f"  Training speed: {efficiency.get('training_speed', 'unknown')}")
        print(f"  Efficiency score: {efficiency.get('efficiency_score', 0):.3f}")

    # Gradient analysis
    print("\\n🎯 Gradient Analysis:")
    gradient_analysis = analyzer.analyze_gradient_evolution()

    if "mean_cosine_similarity" in gradient_analysis:
        print(f"  Mean cosine similarity: {gradient_analysis['mean_cosine_similarity']:.3f}")
        print(f"  Gradient stability: {gradient_analysis.get('gradient_stability', 'unknown')}")

    if "convergence_analysis" in gradient_analysis:
        convergence = gradient_analysis["convergence_analysis"]
        print(f"  Convergence status: {convergence.get('convergence_status', 'unknown')}")
        print(f"  Trend direction: {convergence.get('trend_direction', 'unknown')}")

    # Weight analysis
    print("\\n⚖️  Weight Analysis:")
    weight_analysis = analyzer.analyze_weight_evolution()

    if "weight_stability" in weight_analysis:
        print(f"  Weight stability: {weight_analysis['weight_stability']}")

    if "weight_norm_trend" in weight_analysis:
        trend = weight_analysis["weight_norm_trend"]
        print(f"  Norm trend: {trend.get('trend_direction', 'unknown')}")
        print(f"  Change percentage: {trend.get('change_percentage', 0):.1f}%")


def demonstrate_advanced_analysis(analyzer: CheckpointAnalyzer):
    """Demonstrate advanced analysis with specialized analyzers."""
    print("\\n🔬 Advanced Analysis:")
    print("-" * 30)

    # Collect data for specialized analyzers
    gradient_data = {}
    weight_data = {}

    for cp in analyzer.checkpoints_info:
        step = cp["step"]

        # Load additional data (mock format)
        additional_data_path = Path(cp["path"]) / "additional_data.json"
        if additional_data_path.exists():
            with open(additional_data_path, "r") as f:
                data = json.load(f)

            # Accumulate gradient data
            if "gradient_cosine_similarities" in data:
                if "gradient_cosine_similarities" not in gradient_data:
                    gradient_data["gradient_cosine_similarities"] = []
                gradient_data["gradient_cosine_similarities"].extend(data["gradient_cosine_similarities"])

            # Accumulate weight data
            if "weight_stats_history" in data:
                if "weight_stats_history" not in weight_data:
                    weight_data["weight_stats_history"] = []
                weight_data["weight_stats_history"].extend(data["weight_stats_history"])

    # Gradient deep dive
    if gradient_data:
        print("\\n🔄 Detailed Gradient Analysis:")
        grad_analyzer = GradientAnalyzer(gradient_data)
        grad_report = grad_analyzer.generate_gradient_report()

        # Consistency analysis
        consistency = grad_report.get("consistency_analysis", {})
        print(f"  Consistency score: {consistency.get('consistency_score', 0):.3f}")
        print(f"  Consistency level: {consistency.get('consistency_level', 'unknown')}")
        print(f"  Std similarity: {consistency.get('std_similarity', 0):.3f}")

        # Trend analysis
        if "trend_analysis" in consistency:
            trend = consistency["trend_analysis"]
            print(f"  Trend direction: {trend.get('trend_direction', 'unknown')}")
            print(f"  Trend strength: {trend.get('trend_strength', 0):.3f}")

        # Anomaly detection
        anomalies = grad_report.get("anomaly_detection", {})
        anomaly_count = anomalies.get("anomaly_count", 0)
        severity = anomalies.get("severity_score", 0)
        print(f"  Anomalies detected: {anomaly_count}")
        print(f"  Anomaly severity: {severity:.3f}")

        # Overall assessment
        assessment = grad_report.get("overall_assessment", {})
        print(f"  Gradient health: {assessment.get('gradient_health', 'unknown')}")

        if assessment.get("key_issues"):
            print("  Key issues:")
            for issue in assessment["key_issues"][:3]:
                print(f"    • {issue}")

    # Weight deep dive
    if weight_data:
        print("\\n⚖️  Detailed Weight Analysis:")
        weight_analyzer = WeightAnalyzer(weight_data)
        weight_report = weight_analyzer.generate_weight_report()

        # Evolution analysis
        evolution = weight_report.get("evolution_analysis", {})
        if "norm_statistics" in evolution:
            stats = evolution["norm_statistics"]
            print(f"  Initial norm: {stats.get('initial_norm', 0):.3f}")
            print(f"  Final norm: {stats.get('final_norm', 0):.3f}")
            print(f"  Mean norm: {stats.get('mean_norm', 0):.3f}")
            print(f"  Std norm: {stats.get('std_norm', 0):.3f}")

        if "stability_assessment" in evolution:
            stability = evolution["stability_assessment"]
            print(f"  Stability level: {stability.get('stability_level', 'unknown')}")
            print(f"  Stability score: {stability.get('stability_score', 0):.3f}")

        # Layer analysis
        layer_analysis = weight_report.get("layer_analysis", {})
        if "cross_layer_insights" in layer_analysis:
            insights = layer_analysis["cross_layer_insights"]
            print(f"  Layer count: {insights.get('layer_count', 0)}")

            stability_summary = insights.get("stability_summary", {})
            print(f"  Mean stability: {stability_summary.get('mean_stability_score', 0):.3f}")
            print(f"  Unstable layers: {stability_summary.get('unstable_layer_count', 0)}")

        # Anomaly detection
        anomalies = weight_report.get("anomaly_detection", {})
        print(f"  Weight anomalies: {anomalies.get('anomaly_count', 0)}")
        print(f"  Anomaly severity: {anomalies.get('severity_score', 0):.3f}")


def demonstrate_reporting(analyzer: CheckpointAnalyzer, output_dir: Path):
    """Demonstrate comprehensive reporting capabilities."""
    print("\\n📋 Comprehensive Reporting:")
    print("-" * 30)

    reports = StandardReports(analyzer)

    # Executive summary
    print("\\n📊 Executive Summary:")
    executive_summary = reports.generate_executive_summary()

    training_overview = executive_summary.get("training_overview", {})
    print(f"  Training steps: {training_overview.get('training_steps', 0)}")
    print(f"  Checkpoint frequency: {training_overview.get('checkpoint_frequency', 0)}")

    performance = executive_summary.get("performance_metrics", {})
    if performance.get("loss_improvement"):
        improvement = performance["loss_improvement"]
        print(f"  Loss improvement: {improvement.get('relative_percent', 0):.1f}%")
        print(f"  Improvement rate: {improvement.get('improvement_rate', 0):.3f}/step")

    model_health = executive_summary.get("model_health", {})
    print(f"  Gradient health: {model_health.get('gradient_health', 'unknown')}")
    print(f"  Weight stability: {model_health.get('weight_stability', 'unknown')}")
    print(f"  Training efficiency: {model_health.get('training_efficiency', 'unknown')}")

    # Recommendations
    recommendations = executive_summary.get("recommendations", [])
    if recommendations:
        print("\\n💡 Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")

    # Training diagnostics
    print("\\n🏥 Training Diagnostics:")
    diagnostics = reports.generate_training_diagnostics()

    print(f"  Overall health: {diagnostics.get('overall_health', 'unknown')}")

    critical_issues = diagnostics.get("critical_issues", [])
    if critical_issues:
        print(f"  Critical issues ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"    • {issue.get('description', 'Unknown')}")

    warnings = diagnostics.get("warnings", [])
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"    • {warning.get('description', 'Unknown')}")

    # Export reports
    print("\\n💾 Exporting Reports:")

    # Export executive summary
    exec_path = output_dir / "executive_summary.json"
    with open(exec_path, "w") as f:
        json.dump(executive_summary, f, indent=2, default=str)
    print(f"  Executive summary: {exec_path}")

    # Export technical report
    tech_path = output_dir / "technical_report.json"
    tech_report = reports.generate_technical_report()
    with open(tech_path, "w") as f:
        json.dump(tech_report, f, indent=2, default=str)
    print(f"  Technical report: {tech_path}")

    # Export diagnostics
    diag_path = output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"  Diagnostics: {diag_path}")

    # Export raw data
    print("\\n📤 Exporting Raw Data:")
    raw_data_dir = output_dir / "raw_data"
    exported_files = analyzer.export_raw_data(raw_data_dir)

    for data_type, file_path in exported_files.items():
        print(f"  {data_type}: {file_path}")


def main():
    """Main function demonstrating analysis capabilities."""
    print("🔍 Training Lens Analysis Example")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create mock checkpoint data
        checkpoints_dir = create_mock_checkpoints(output_dir)

        # Initialize analyzer
        print("\\n🔧 Initializing Checkpoint Analyzer...")
        analyzer = CheckpointAnalyzer(checkpoints_dir)

        if not analyzer.checkpoints_info:
            print("❌ No checkpoints found!")
            return 1

        print(f"   Loaded {len(analyzer.checkpoints_info)} checkpoints")

        # Demonstrate different analysis levels
        demonstrate_basic_analysis(analyzer)
        demonstrate_advanced_analysis(analyzer)
        demonstrate_reporting(analyzer, output_dir)

        # Show generated files
        print("\\n📁 Generated Files:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  {file_path.relative_to(output_dir)} ({size} bytes)")

    print("\\n✨ Analysis example completed successfully!")
    print("\\nAnalysis capabilities demonstrated:")
    print("   ✅ Basic checkpoint analysis")
    print("   ✅ Training dynamics assessment")
    print("   ✅ Gradient evolution analysis")
    print("   ✅ Weight stability analysis")
    print("   ✅ Anomaly detection")
    print("   ✅ Executive reporting")
    print("   ✅ Technical documentation")
    print("   ✅ Training diagnostics")
    print("   ✅ Raw data export")

    return 0


if __name__ == "__main__":
    exit(main())
