"""Analysis command for CLI."""

import json
from pathlib import Path
from typing import Optional

import click

from ..analysis.checkpoint_analyzer import CheckpointAnalyzer
from ..utils.logging import get_logger

logger = get_logger("training_lens.cli.analyze")


@click.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory for analysis results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "html"]),
    default="json",
    help="Output format for analysis report",
)
@click.option("--include-plots", is_flag=True, help="Generate visualization plots")
@click.option("--export-raw", is_flag=True, help="Export raw training data")
@click.option("--hf-repo", type=str, help="Analyze checkpoints from HuggingFace repository")
@click.option("--steps", type=str, help="Comma-separated list of specific steps to analyze (e.g., '100,500,1000')")
@click.option("--gradient-analysis", is_flag=True, help="Perform detailed gradient analysis")
@click.option("--weight-analysis", is_flag=True, help="Perform detailed weight analysis")
@click.option("--overfitting-check", is_flag=True, help="Check for overfitting patterns")
def analyze_command(
    checkpoint_dir: str,
    output: Optional[str],
    output_format: str,
    include_plots: bool,
    export_raw: bool,
    hf_repo: Optional[str],
    steps: Optional[str],
    gradient_analysis: bool,
    weight_analysis: bool,
    overfitting_check: bool,
) -> None:
    """Analyze training checkpoints and generate insights.

    This command analyzes training checkpoints to provide comprehensive insights
    into the training process, including gradient evolution, weight changes,
    and potential training issues.

    Examples:

        # Basic analysis of local checkpoints
        training-lens analyze ./training_output/checkpoints

        # Full analysis with plots and raw data export
        training-lens analyze ./checkpoints --include-plots --export-raw --output ./analysis

        # Analyze specific checkpoints
        training-lens analyze ./checkpoints --steps 100,500,1000

        # Analyze from HuggingFace repository
        training-lens analyze --hf-repo username/model-name --output ./downloaded_analysis
    """
    click.echo("🔍 Starting Training Lens analysis...")

    # Initialize analyzer
    if hf_repo:
        click.echo(f"📥 Downloading checkpoints from HuggingFace: {hf_repo}")
        try:
            analyzer = CheckpointAnalyzer.from_huggingface(
                repo_id=hf_repo, local_dir=Path(output or "./hf_analysis") / "checkpoints"
            )
        except Exception as e:
            raise click.ClickException(f"Failed to download from HuggingFace: {e}")
    else:
        analyzer = CheckpointAnalyzer(checkpoint_dir)

    if not analyzer.checkpoints_info:
        raise click.ClickException("No checkpoints found to analyze")

    click.echo(f"📊 Found {len(analyzer.checkpoints_info)} checkpoints")

    # Filter specific steps if requested
    if steps:
        requested_steps = [int(s.strip()) for s in steps.split(",")]
        available_steps = [cp["step"] for cp in analyzer.checkpoints_info]

        missing_steps = set(requested_steps) - set(available_steps)
        if missing_steps:
            click.echo(f"⚠️  Warning: Steps not found: {sorted(missing_steps)}")

        analyzer.checkpoints_info = [cp for cp in analyzer.checkpoints_info if cp["step"] in requested_steps]
        click.echo(f"🎯 Analyzing {len(analyzer.checkpoints_info)} selected checkpoints")

    # Set up output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("./analysis_output")
        output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"📁 Output directory: {output_dir}")

    # Generate analysis report
    click.echo("\n🧮 Generating standard analysis report...")
    report = analyzer.generate_standard_report()

    # Add specific analyses if requested
    if gradient_analysis:
        click.echo("🔄 Performing detailed gradient analysis...")
        from ..analysis.gradient_analyzer import GradientAnalyzer

        # Collect gradient data from all checkpoints
        all_gradient_data = {}
        for cp in analyzer.checkpoints_info:
            metrics = analyzer.load_checkpoint_metrics(cp["step"])
            if metrics:
                all_gradient_data.update(metrics)

        if all_gradient_data:
            grad_analyzer = GradientAnalyzer(all_gradient_data)
            report["detailed_gradient_analysis"] = grad_analyzer.generate_gradient_report()
        else:
            click.echo("⚠️  No gradient data found for detailed analysis")

    if weight_analysis:
        click.echo("⚖️  Performing detailed weight analysis...")
        weight_report = analyzer.analyze_weight_evolution()
        report["detailed_weight_analysis"] = weight_report

    if overfitting_check:
        click.echo("📈 Checking for overfitting...")
        overfitting_report = analyzer.detect_overfitting()
        report["overfitting_analysis"] = overfitting_report

    # Save analysis report
    click.echo("\n💾 Saving analysis report...")
    report_filename = f"training_analysis.{output_format}"
    report_path = output_dir / report_filename

    if output_format == "json":
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    elif output_format == "yaml":
        import yaml

        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, indent=2)
    elif output_format == "html":
        html_content = _generate_html_report(report)
        with open(report_path, "w") as f:
            f.write(html_content)

    click.echo(f"✅ Analysis report saved: {report_path}")

    # Generate plots if requested
    if include_plots:
        click.echo("\n📈 Generating visualization plots...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Generate gradient plots if gradient analysis was performed
        if gradient_analysis and "detailed_gradient_analysis" in report:
            from ..analysis.gradient_analyzer import GradientAnalyzer

            all_gradient_data = {}
            for cp in analyzer.checkpoints_info:
                metrics = analyzer.load_checkpoint_metrics(cp["step"])
                if metrics:
                    all_gradient_data.update(metrics)

            if all_gradient_data:
                grad_analyzer = GradientAnalyzer(all_gradient_data)
                plot_files = grad_analyzer.visualize_gradient_evolution(plots_dir)

                for plot_type, plot_path in plot_files.items():
                    click.echo(f"   📊 {plot_type}: {plot_path}")

        # Generate basic training plots
        _generate_basic_plots(analyzer, plots_dir)
        click.echo(f"✅ Plots saved to: {plots_dir}")

    # Export raw data if requested
    if export_raw:
        click.echo("\n📤 Exporting raw training data...")
        raw_data_dir = output_dir / "raw_data"
        exported_files = analyzer.export_raw_data(raw_data_dir)

        for data_type, file_path in exported_files.items():
            click.echo(f"   📄 {data_type}: {file_path}")

        click.echo(f"✅ Raw data exported to: {raw_data_dir}")

    # Print summary
    click.echo("\n📋 Analysis Summary:")
    summary = report.get("summary", {})
    click.echo(f"   Total checkpoints: {summary.get('total_checkpoints', 0)}")

    # Show key insights
    training_dynamics = report.get("training_dynamics", {})
    if "loss_analysis" in training_dynamics:
        loss_analysis = training_dynamics["loss_analysis"]
        if "final_loss" in loss_analysis and "initial_loss" in loss_analysis:
            initial_loss = loss_analysis["initial_loss"]
            final_loss = loss_analysis["final_loss"]
            improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
            click.echo(f"   Loss improvement: {improvement:.1f}%")

    # Show warnings
    warnings = []

    gradient_analysis_report = report.get("gradient_analysis", {})
    if gradient_analysis_report.get("gradient_stability") in ["unstable", "moderate"]:
        warnings.append("Gradient instability detected")

    overfitting_analysis_report = report.get("overfitting_analysis", {})
    if overfitting_analysis_report.get("overfitting_detected"):
        warnings.append("Potential overfitting detected")

    if warnings:
        click.echo("\n⚠️  Warnings:")
        for warning in warnings:
            click.echo(f"   • {warning}")

    click.echo(f"\n✨ Analysis complete! Results saved to: {output_dir}")


def _generate_html_report(report: dict) -> str:
    """Generate HTML report from analysis data."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Lens Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #f8f9fa; padding: 20px; border-radius: 8px; }
            .section { margin: 30px 0; }
            .metric { background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 4px; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 4px; }
            .success { background: #d1edff; border: 1px solid #74c0fc; padding: 10px; border-radius: 4px; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Training Lens Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>

        <div class="section">
            <h2>📊 Summary</h2>
            {summary_html}
        </div>

        <div class="section">
            <h2>📈 Training Dynamics</h2>
            {dynamics_html}
        </div>

        <div class="section">
            <h2>🔄 Gradient Analysis</h2>
            {gradient_html}
        </div>

        <div class="section">
            <h2>⚖️ Weight Analysis</h2>
            {weight_html}
        </div>

        <div class="section">
            <h2>📋 Full Report Data</h2>
            <pre>{full_report}</pre>
        </div>
    </body>
    </html>
    """

    # Generate HTML sections
    summary = report.get("summary", {})
    summary_html = f"""
    <div class="metric">Total Checkpoints: {summary.get('total_checkpoints', 0)}</div>
    <div class="metric">Checkpoint Steps: {summary.get('checkpoint_steps', [])}</div>
    """

    dynamics = report.get("training_dynamics", {})
    dynamics_html = f"""
    <div class="metric">Training Steps: {dynamics.get('training_steps', 0)}</div>
    <div class="metric">Training Efficiency: {
        dynamics.get('training_efficiency', {}).get('efficiency_score', 'N/A')
    }</div>
    """

    gradient_analysis = report.get("gradient_analysis", {})
    gradient_html = f"""
    <div class="metric">Gradient Stability: {gradient_analysis.get('gradient_stability', 'N/A')}</div>
    <div class="metric">Mean Cosine Similarity: {gradient_analysis.get('mean_cosine_similarity', 'N/A')}</div>
    """

    weight_analysis = report.get("weight_analysis", {})
    weight_html = f"""
    <div class="metric">Weight Stability: {weight_analysis.get('weight_stability', 'N/A')}</div>
    """

    return html_template.format(
        timestamp=report.get("summary", {}).get("analysis_timestamp", "Unknown"),
        summary_html=summary_html,
        dynamics_html=dynamics_html,
        gradient_html=gradient_html,
        weight_html=weight_html,
        full_report=json.dumps(report, indent=2, default=str),
    )


def _generate_basic_plots(analyzer: CheckpointAnalyzer, plots_dir: Path) -> None:
    """Generate basic training plots."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not analyzer.checkpoints_info:
        return

    # Extract data
    steps = []
    losses = []
    learning_rates = []

    for cp in analyzer.checkpoints_info:
        metadata = cp.get("metadata", {})
        if "step" in metadata:
            steps.append(metadata["step"])
            losses.append(metadata.get("train_loss", np.nan))
            learning_rates.append(metadata.get("learning_rate", np.nan))

    if not steps:
        return

    # Loss curve plot
    if losses and not all(np.isnan(losses)):
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_indices = [i for i, loss in enumerate(losses) if not np.isnan(loss)]
        valid_steps = [steps[i] for i in valid_indices]
        valid_losses = [losses[i] for i in valid_indices]

        ax.plot(valid_steps, valid_losses, "b-", linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "training_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Learning rate plot
    if learning_rates and not all(np.isnan(learning_rates)):
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_indices = [i for i, lr in enumerate(learning_rates) if not np.isnan(lr)]
        valid_steps = [steps[i] for i in valid_indices]
        valid_lrs = [learning_rates[i] for i in valid_indices]

        ax.plot(valid_steps, valid_lrs, "g-", linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(plots_dir / "learning_rate.png", dpi=300, bbox_inches="tight")
        plt.close()
