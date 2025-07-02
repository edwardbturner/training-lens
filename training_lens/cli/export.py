"""Export command for CLI."""

import json
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd

from ..analysis import CheckpointAnalyzer
from ..utils.logging import get_logger

logger = get_logger("training_lens.cli.export")


@click.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory for exported data")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["csv", "json", "numpy", "parquet", "all"]),
    default="csv",
    help="Export format",
)
@click.option(
    "--data-type",
    type=click.Choice(["gradients", "weights", "metrics", "all"]),
    default="all",
    help="Type of data to export",
)
@click.option("--steps", type=str, help="Comma-separated list of specific steps to export")
@click.option("--compress", is_flag=True, help="Compress output files")
def export_command(
    checkpoint_dir: str,
    output: str,
    export_format: str,
    data_type: str,
    steps: Optional[str],
    compress: bool,
) -> None:
    """Export training data for external analysis.

    This command exports raw training data from checkpoints in various formats
    for use with external analysis tools, research, or custom processing.

    Examples:

        # Export all data as CSV
        training-lens export ./checkpoints --output ./exported_data

        # Export only gradient data as JSON
        training-lens export ./checkpoints --output ./gradients --data-type gradients --format json

        # Export specific checkpoints with compression
        training-lens export ./checkpoints --output ./data --steps 100,500,1000 --compress
    """
    click.echo("📤 Starting Training Lens data export...")

    # Initialize analyzer
    analyzer = CheckpointAnalyzer(checkpoint_dir)

    if not analyzer.checkpoints_info:
        raise click.ClickException("No checkpoints found to export")

    click.echo(f"📊 Found {len(analyzer.checkpoints_info)} checkpoints")

    # Filter specific steps if requested
    if steps:
        requested_steps = [int(s.strip()) for s in steps.split(",")]
        available_steps = [cp["step"] for cp in analyzer.checkpoints_info]

        missing_steps = set(requested_steps) - set(available_steps)
        if missing_steps:
            click.echo(f"⚠️  Warning: Steps not found: {sorted(missing_steps)}")

        analyzer.checkpoints_info = [cp for cp in analyzer.checkpoints_info if cp["step"] in requested_steps]
        click.echo(f"🎯 Exporting {len(analyzer.checkpoints_info)} selected checkpoints")

    # Set up output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"📁 Output directory: {output_dir}")

    # Collect data
    exported_files = []

    if data_type in ["gradients", "all"]:
        click.echo("\n🔄 Exporting gradient data...")
        gradient_files = _export_gradient_data(analyzer, output_dir, export_format, compress)
        exported_files.extend(gradient_files)

    if data_type in ["weights", "all"]:
        click.echo("\n⚖️  Exporting weight data...")
        weight_files = _export_weight_data(analyzer, output_dir, export_format, compress)
        exported_files.extend(weight_files)

    if data_type in ["metrics", "all"]:
        click.echo("\n📊 Exporting metrics data...")
        metrics_files = _export_metrics_data(analyzer, output_dir, export_format, compress)
        exported_files.extend(metrics_files)

    # Export metadata
    click.echo("\n📋 Exporting checkpoint metadata...")
    metadata_files = _export_metadata(analyzer, output_dir, export_format, compress)
    exported_files.extend(metadata_files)

    # Generate export summary
    summary = {
        "export_timestamp": pd.Timestamp.now().isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "exported_checkpoints": len(analyzer.checkpoints_info),
        "checkpoint_steps": [cp["step"] for cp in analyzer.checkpoints_info],
        "export_format": export_format,
        "data_type": data_type,
        "compressed": compress,
        "exported_files": [str(f) for f in exported_files],
        "file_count": len(exported_files),
    }

    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    click.echo("\n✅ Export completed successfully!")
    click.echo(f"   Exported {len(exported_files)} files")
    click.echo(f"   Export summary: {summary_path}")

    # Show file sizes
    total_size = sum(f.stat().st_size for f in exported_files if f.exists())
    click.echo(f"   Total size: {total_size / (1024*1024):.1f} MB")


def _export_gradient_data(analyzer: CheckpointAnalyzer, output_dir: Path, export_format: str, compress: bool) -> list:
    """Export gradient-related data."""
    import numpy as np

    exported_files = []

    # Collect gradient data
    all_cosine_similarities = []
    gradient_norms_by_step = {}
    layer_gradients_by_step = {}

    for cp in analyzer.checkpoints_info:
        step = cp["step"]
        metrics = analyzer.load_checkpoint_metrics(step)

        if metrics:
            # Cosine similarities
            if "gradient_cosine_similarities" in metrics:
                similarities = metrics["gradient_cosine_similarities"]
                for i, sim in enumerate(similarities):
                    all_cosine_similarities.append(
                        {"checkpoint_step": step, "similarity_index": i, "cosine_similarity": sim}
                    )

            # Gradient norms (if available in metrics)
            if "gradient_norms" in metrics:
                gradient_norms_by_step[step] = metrics["gradient_norms"]

            # Layer gradients (if available)
            if "layer_gradients" in metrics:
                layer_gradients_by_step[step] = metrics["layer_gradients"]

    # Export cosine similarities
    if all_cosine_similarities:
        if export_format in ["csv", "all"]:
            df = pd.DataFrame(all_cosine_similarities)
            file_path = output_dir / "gradient_cosine_similarities.csv"
            if compress:
                file_path = file_path.with_suffix(".csv.gz")
                df.to_csv(file_path, index=False, compression="gzip")
            else:
                df.to_csv(file_path, index=False)
            exported_files.append(file_path)

        if export_format in ["json", "all"]:
            file_path = output_dir / "gradient_cosine_similarities.json"
            if compress:
                import gzip

                file_path = file_path.with_suffix(".json.gz")
                with gzip.open(file_path, "wt") as f:
                    json.dump(all_cosine_similarities, f, indent=2)
            else:
                with open(file_path, "w") as f:
                    json.dump(all_cosine_similarities, f, indent=2)
            exported_files.append(file_path)

        if export_format in ["numpy", "all"]:
            # Export as structured array
            similarities_array = np.array(
                [
                    (item["checkpoint_step"], item["similarity_index"], item["cosine_similarity"])
                    for item in all_cosine_similarities
                ],
                dtype=[("checkpoint_step", "i4"), ("similarity_index", "i4"), ("cosine_similarity", "f4")],
            )

            file_path = output_dir / "gradient_cosine_similarities.npy"
            if compress:
                file_path = file_path.with_suffix(".npz")
                np.savez_compressed(file_path, gradient_cosine_similarities=similarities_array)
            else:
                np.save(file_path, similarities_array)
            exported_files.append(file_path)

    return exported_files


def _export_weight_data(
    analyzer: CheckpointAnalyzer, output_dir: Path, export_format: str, compress: bool
) -> List[Path]:
    """Export weight-related data."""
    exported_files: List[Path] = []

    # Collect weight data
    all_weight_stats = []

    for cp in analyzer.checkpoints_info:
        step = cp["step"]
        metrics = analyzer.load_checkpoint_metrics(step)

        if metrics and "weight_stats_history" in metrics:
            weight_stats = metrics["weight_stats_history"]
            all_weight_stats.extend(weight_stats)

    if not all_weight_stats:
        return exported_files

    # Export weight statistics
    if export_format in ["csv", "all"]:
        # Flatten layer norms for CSV export
        flattened_stats = []
        for stat in all_weight_stats:
            base_stat = {
                "step": stat.get("step"),
                "overall_norm": stat.get("overall_norm"),
                "overall_mean": stat.get("overall_mean"),
                "overall_std": stat.get("overall_std"),
            }

            # Add layer norms as separate columns
            layer_norms = stat.get("layer_norms", {})
            for layer_name, norm in layer_norms.items():
                base_stat[f"layer_norm_{layer_name}"] = norm

            flattened_stats.append(base_stat)

        df = pd.DataFrame(flattened_stats)
        file_path = output_dir / "weight_evolution.csv"
        if compress:
            file_path = file_path.with_suffix(".csv.gz")
            df.to_csv(file_path, index=False, compression="gzip")
        else:
            df.to_csv(file_path, index=False)
        exported_files.append(file_path)

    if export_format in ["json", "all"]:
        file_path = output_dir / "weight_evolution.json"
        if compress:
            import gzip

            file_path = file_path.with_suffix(".json.gz")
            with gzip.open(file_path, "wt") as f:
                json.dump(all_weight_stats, f, indent=2, default=str)
        else:
            with open(file_path, "w") as f:
                json.dump(all_weight_stats, f, indent=2, default=str)
        exported_files.append(file_path)

    if export_format in ["parquet", "all"]:
        try:
            # Flatten for parquet
            flattened_stats = []
            for stat in all_weight_stats:
                base_stat = {
                    "step": stat.get("step"),
                    "overall_norm": stat.get("overall_norm"),
                    "overall_mean": stat.get("overall_mean"),
                    "overall_std": stat.get("overall_std"),
                }
                flattened_stats.append(base_stat)

            df = pd.DataFrame(flattened_stats)
            file_path = output_dir / "weight_evolution.parquet"
            if compress:
                df.to_parquet(file_path, compression="gzip", index=False)
            else:
                df.to_parquet(file_path, index=False)
            exported_files.append(file_path)
        except ImportError:
            click.echo("⚠️  Parquet export requires pyarrow: pip install pyarrow")

    return exported_files


def _export_metrics_data(
    analyzer: CheckpointAnalyzer, output_dir: Path, export_format: str, compress: bool
) -> List[Path]:
    """Export general metrics data."""
    exported_files: List[Path] = []

    # Collect all step metrics
    all_metrics = []

    for cp in analyzer.checkpoints_info:
        cp["step"]
        metadata = cp.get("metadata", {})

        if metadata:
            all_metrics.append(metadata)

    if not all_metrics:
        return exported_files

    # Export metrics
    if export_format in ["csv", "all"]:
        df = pd.DataFrame(all_metrics)
        file_path = output_dir / "training_metrics.csv"
        if compress:
            file_path = file_path.with_suffix(".csv.gz")
            df.to_csv(file_path, index=False, compression="gzip")
        else:
            df.to_csv(file_path, index=False)
        exported_files.append(file_path)

    if export_format in ["json", "all"]:
        file_path = output_dir / "training_metrics.json"
        if compress:
            import gzip

            file_path = file_path.with_suffix(".json.gz")
            with gzip.open(file_path, "wt") as f:
                json.dump(all_metrics, f, indent=2, default=str)
        else:
            with open(file_path, "w") as f:
                json.dump(all_metrics, f, indent=2, default=str)
        exported_files.append(file_path)

    return exported_files


def _export_metadata(analyzer: CheckpointAnalyzer, output_dir: Path, export_format: str, compress: bool) -> list:
    """Export checkpoint metadata."""
    exported_files = []

    # Create comprehensive metadata
    metadata = {
        "checkpoints": analyzer.checkpoints_info,
        "total_checkpoints": len(analyzer.checkpoints_info),
        "checkpoint_steps": [cp["step"] for cp in analyzer.checkpoints_info],
        "first_step": min([cp["step"] for cp in analyzer.checkpoints_info]) if analyzer.checkpoints_info else 0,
        "last_step": max([cp["step"] for cp in analyzer.checkpoints_info]) if analyzer.checkpoints_info else 0,
        "export_info": {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "analyzer_class": "CheckpointAnalyzer",
            "training_lens_version": "0.1.0",
        },
    }

    # Export metadata
    if export_format in ["json", "all"]:
        file_path = output_dir / "checkpoint_metadata.json"
        if compress:
            import gzip

            file_path = file_path.with_suffix(".json.gz")
            with gzip.open(file_path, "wt") as f:
                json.dump(metadata, f, indent=2, default=str)
        else:
            with open(file_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        exported_files.append(file_path)

    # Export checkpoint list as CSV
    if export_format in ["csv", "all"]:
        checkpoint_df = pd.DataFrame(
            [
                {
                    "step": cp["step"],
                    "path": cp["path"],
                    "timestamp": cp.get("metadata", {}).get("timestamp", ""),
                    "train_loss": cp.get("metadata", {}).get("train_loss"),
                    "learning_rate": cp.get("metadata", {}).get("learning_rate"),
                    "grad_norm": cp.get("metadata", {}).get("grad_norm"),
                }
                for cp in analyzer.checkpoints_info
            ]
        )

        file_path = output_dir / "checkpoint_list.csv"
        if compress:
            file_path = file_path.with_suffix(".csv.gz")
            checkpoint_df.to_csv(file_path, index=False, compression="gzip")
        else:
            checkpoint_df.to_csv(file_path, index=False)
        exported_files.append(file_path)

    return exported_files
