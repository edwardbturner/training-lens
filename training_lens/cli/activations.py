"""Activation analysis command for CLI."""

import json
from pathlib import Path
from typing import List, Optional

import click
import torch

from ..analysis.activation_analyzer import ActivationAnalyzer
from ..analysis.specialized.lora_analyzer import LoRAActivationTracker
from ..analysis.activation_visualizer import ActivationVisualizer
from ..integrations.activation_storage import ActivationStorage
from ..utils.logging import get_logger

logger = get_logger("training_lens.cli.activations")


@click.group()
def activations():
    """Activation analysis commands for tracking how activations evolve during training."""
    pass


@activations.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.argument("model_name", type=str)
@click.option("--input-text", "-t", type=str, default="Hello world, this is a test.", 
              help="Input text to analyze activations for")
@click.option("--input-file", "-f", type=click.Path(exists=True), 
              help="File containing input texts (one per line)")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--layers", type=str, help="Comma-separated layer indices to analyze (e.g., '0,1,2')")
@click.option("--activation-points", type=str, 
              help="JSON string defining custom activation points (e.g., '{\"layer_0_attn\": \"model.layers.0.self_attn\"}')")
@click.option("--steps", type=str, help="Comma-separated checkpoint steps to analyze")
@click.option("--lora-analysis", is_flag=True, help="Include LoRA-specific activation analysis")
@click.option("--similarity-metric", type=click.Choice(["cosine", "l2", "kl_div"]), 
              default="cosine", help="Similarity metric for comparing activations")
@click.option("--reference-step", type=int, help="Reference checkpoint step for similarity analysis")
@click.option("--export-format", type=click.Choice(["npz", "pt", "json"]), 
              default="npz", help="Export format for activation data")
@click.option("--upload-to-hf", is_flag=True, help="Upload results to HuggingFace Hub")
@click.option("--hf-repo", type=str, help="HuggingFace repository for storing activation data")
@click.option("--visualize", is_flag=True, help="Generate visualization plots")
@click.option("--interactive-plots", is_flag=True, help="Generate interactive plots (requires plotly)")
def analyze(
    checkpoint_dir: str,
    model_name: str,
    input_text: str,
    input_file: Optional[str],
    output: Optional[str],
    layers: Optional[str],
    activation_points: Optional[str],
    steps: Optional[str],
    lora_analysis: bool,
    similarity_metric: str,
    reference_step: Optional[int],
    export_format: str,
    upload_to_hf: bool,
    hf_repo: Optional[str],
    visualize: bool,
    interactive_plots: bool
) -> None:
    """Analyze how activations evolve across training checkpoints.
    
    This command extracts and analyzes activations at specified points in the model
    across different training checkpoints, allowing you to see how the model's
    internal representations change during training.
    
    Examples:
    
        # Basic activation analysis
        training-lens activations analyze ./checkpoints llama2-7b
        
        # Analyze specific layers with custom input
        training-lens activations analyze ./checkpoints llama2-7b \\
            --layers "0,5,10" --input-text "The capital of France is"
        
        # LoRA-specific analysis with visualization
        training-lens activations analyze ./checkpoints llama2-7b \\
            --lora-analysis --visualize --output ./activation_results
        
        # Analyze specific checkpoints with HuggingFace storage
        training-lens activations analyze ./checkpoints llama2-7b \\
            --steps "100,500,1000" --hf-repo "username/model-activations" --upload-to-hf
    """
    click.echo("🧠 Starting activation evolution analysis...")
    
    # Setup output directory
    if output:
        output_dir = Path(output)
    else:
        output_dir = Path("./activation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare input data
    input_data = []
    if input_file:
        click.echo(f"📄 Loading input texts from {input_file}")
        with open(input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        input_data.extend(texts)
    else:
        input_data.append(input_text)
    
    click.echo(f"📝 Analyzing {len(input_data)} input sample(s)")
    
    # Parse layer indices
    layer_indices = None
    if layers:
        try:
            layer_indices = [int(x.strip()) for x in layers.split(",")]
            click.echo(f"🎯 Analyzing layers: {layer_indices}")
        except ValueError:
            raise click.ClickException("Invalid layer indices. Use comma-separated integers.")
    
    # Parse custom activation points
    custom_activation_points = {}
    if activation_points:
        try:
            custom_activation_points = json.loads(activation_points)
            click.echo(f"🔍 Custom activation points: {list(custom_activation_points.keys())}")
        except json.JSONDecodeError:
            raise click.ClickException("Invalid activation points JSON format.")
    
    # Parse checkpoint steps
    checkpoint_steps = None
    if steps:
        try:
            checkpoint_steps = [int(x.strip()) for x in steps.split(",")]
            click.echo(f"📊 Analyzing checkpoints: {checkpoint_steps}")
        except ValueError:
            raise click.ClickException("Invalid checkpoint steps. Use comma-separated integers.")
    
    # Initialize analyzer
    analyzer = ActivationAnalyzer(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        hf_repo_id=hf_repo
    )
    
    # Convert input texts to tensors (simplified tokenization)
    click.echo("🔤 Tokenizing input data...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        input_tensors = []
        for text in input_data:
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_tensors.append(tokens["input_ids"])
        
    except Exception as e:
        raise click.ClickException(f"Failed to tokenize input: {e}")
    
    # Run activation evolution analysis
    click.echo("🚀 Running activation evolution analysis...")
    try:
        analysis_results = analyzer.analyze_activation_evolution(
            input_data=input_tensors,
            activation_points=custom_activation_points,
            checkpoint_steps=checkpoint_steps,
            lora_analysis=lora_analysis,
            layer_indices=layer_indices
        )
    except Exception as e:
        raise click.ClickException(f"Activation analysis failed: {e}")
    
    # Compute similarity analysis
    if analyzer.activation_data:
        click.echo("📐 Computing activation similarities...")
        similarity_results = analyzer.compute_activation_similarities(
            reference_step=reference_step,
            similarity_metric=similarity_metric
        )
        analysis_results["similarity_analysis"] = similarity_results
    
    # Save analysis results
    results_file = output_dir / "activation_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    click.echo(f"💾 Analysis results saved: {results_file}")
    
    # Export activation data
    click.echo(f"📤 Exporting activation data in {export_format} format...")
    exported_files = analyzer.export_activations(
        output_dir / "activation_data",
        format=export_format,
        upload_to_hf=upload_to_hf
    )
    
    for step, file_path in exported_files.items():
        click.echo(f"   Step {step}: {file_path}")
    
    # Generate visualizations
    if visualize:
        click.echo("📈 Generating visualization plots...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        visualizer = ActivationVisualizer()
        
        # Convert activation data to numpy for visualization
        viz_data = {}
        for step, activations in analyzer.activation_data.items():
            viz_data[step] = {name: tensor.numpy() for name, tensor in activations.items()}
        
        # Evolution plot
        if viz_data:
            evolution_fig = visualizer.plot_activation_evolution(
                viz_data,
                save_path=viz_dir / ("evolution.html" if interactive_plots else "evolution.png"),
                interactive=interactive_plots
            )
            
            # Similarity heatmap
            if "similarity_analysis" in analysis_results:
                similarity_data = analysis_results["similarity_analysis"]["similarities"]
                ref_step = analysis_results["similarity_analysis"]["reference_step"]
                
                heatmap_fig = visualizer.plot_activation_similarity_heatmap(
                    similarity_data,
                    ref_step,
                    save_path=viz_dir / ("similarity_heatmap.html" if interactive_plots else "similarity_heatmap.png"),
                    interactive=interactive_plots
                )
            
            # Dashboard
            dashboard_fig = visualizer.create_activation_summary_dashboard(
                viz_data,
                similarity_data=analysis_results.get("similarity_analysis", {}).get("similarities"),
                save_path=viz_dir / "dashboard.png"
            )
            
            click.echo(f"✅ Visualizations saved to: {viz_dir}")
    
    # Setup activation storage if HuggingFace repo specified
    if hf_repo:
        click.echo("☁️  Setting up activation storage...")
        storage = ActivationStorage(
            local_dir=output_dir / "storage",
            repo_id=hf_repo,
            create_repo_if_not_exists=True
        )
        
        # Store activation data
        for step, activations in analyzer.activation_data.items():
            data_id = storage.store_activation_data(
                activation_data=activations,
                checkpoint_step=step,
                model_name=model_name,
                activation_config={
                    "layer_indices": layer_indices,
                    "custom_points": custom_activation_points,
                    "lora_analysis": lora_analysis,
                    "input_samples": len(input_data)
                },
                upload_to_hub=upload_to_hf
            )
            click.echo(f"   Stored step {step} with ID: {data_id}")
    
    # Print summary
    click.echo("\n📋 Analysis Summary:")
    click.echo(f"   Steps analyzed: {list(analyzer.activation_data.keys())}")
    click.echo(f"   Activation points: {len(next(iter(analyzer.activation_data.values())))}")
    
    if "similarity_analysis" in analysis_results:
        sim_summary = analysis_results["similarity_analysis"]["summary"]
        click.echo(f"   Mean similarity: {sim_summary['overall_mean_similarity']:.3f}")
    
    # Evolution patterns summary
    evolution_patterns = analysis_results.get("evolution_patterns", {})
    if evolution_patterns:
        click.echo("   Evolution patterns:")
        for act_name, pattern in list(evolution_patterns.items())[:3]:  # Show first 3
            stability = pattern.get("stability", "unknown")
            click.echo(f"     {act_name}: {stability}")
    
    click.echo(f"\n✨ Activation analysis complete! Results saved to: {output_dir}")


@activations.command()
@click.argument("storage_dir", type=click.Path(exists=True))
@click.option("--model-name", type=str, help="Filter by model name")
@click.option("--hf-repo", type=str, help="HuggingFace repository to list from")
@click.option("--include-remote", is_flag=True, help="Include remote data from HuggingFace")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), 
              default="table", help="Output format")
def list_stored(
    storage_dir: str,
    model_name: Optional[str],
    hf_repo: Optional[str],
    include_remote: bool,
    output_format: str
) -> None:
    """List stored activation data."""
    
    storage = ActivationStorage(
        local_dir=storage_dir,
        repo_id=hf_repo
    )
    
    stored_data = storage.list_stored_data(
        model_name=model_name,
        include_remote=include_remote
    )
    
    if output_format == "json":
        click.echo(json.dumps(stored_data, indent=2, default=str))
    else:
        if not stored_data:
            click.echo("No stored activation data found.")
            return
        
        click.echo("📊 Stored Activation Data:")
        click.echo("=" * 80)
        
        for data in stored_data:
            data_id = data["data_id"]
            step = data["checkpoint_step"]
            model = data["model_name"]
            points = len(data.get("activation_points", []))
            location = data.get("location", "local")
            
            click.echo(f"ID: {data_id}")
            click.echo(f"  Step: {step} | Model: {model} | Points: {points} | Location: {location}")
            click.echo("-" * 40)


@activations.command()
@click.argument("storage_dir", type=click.Path(exists=True))
@click.argument("data_ids", nargs=-1, required=True)
@click.argument("dataset_name", type=str)
@click.option("--description", type=str, help="Dataset description")
@click.option("--upload-to-hf", is_flag=True, help="Upload dataset to HuggingFace")
@click.option("--hf-repo", type=str, help="HuggingFace repository")
def create_dataset(
    storage_dir: str,
    data_ids: List[str],
    dataset_name: str,
    description: Optional[str],
    upload_to_hf: bool,
    hf_repo: Optional[str]
) -> None:
    """Create a dataset from multiple activation data entries."""
    
    storage = ActivationStorage(
        local_dir=storage_dir,
        repo_id=hf_repo
    )
    
    click.echo(f"📦 Creating dataset '{dataset_name}' from {len(data_ids)} entries...")
    
    try:
        dataset_id = storage.create_activation_dataset(
            data_ids=list(data_ids),
            dataset_name=dataset_name,
            description=description,
            upload_to_hub=upload_to_hf
        )
        
        click.echo(f"✅ Dataset created with ID: {dataset_id}")
        
    except Exception as e:
        raise click.ClickException(f"Failed to create dataset: {e}")


@activations.command()
@click.argument("storage_dir", type=click.Path(exists=True))
@click.option("--model-name", type=str, help="Filter by model name")
@click.option("--data-ids", type=str, help="Comma-separated data IDs to analyze")
@click.option("--output", "-o", type=click.Path(), help="Output file for statistics")
def statistics(
    storage_dir: str,
    model_name: Optional[str],
    data_ids: Optional[str],
    output: Optional[str]
) -> None:
    """Compute statistics across stored activation data."""
    
    storage = ActivationStorage(local_dir=storage_dir)
    
    # Parse data IDs if provided
    data_id_list = None
    if data_ids:
        data_id_list = [x.strip() for x in data_ids.split(",")]
    
    click.echo("📊 Computing activation statistics...")
    
    try:
        stats = storage.compute_activation_statistics(
            data_ids=data_id_list,
            model_name=model_name
        )
        
        if output:
            with open(output, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            click.echo(f"💾 Statistics saved to: {output}")
        
        # Print summary
        if stats.get("status") != "no_data":
            click.echo(f"\n📋 Statistics Summary:")
            click.echo(f"   Data entries analyzed: {len(stats.get('data_ids_analyzed', []))}")
            click.echo(f"   Checkpoint steps: {stats.get('checkpoint_steps', [])}")
            
            activation_stats = stats.get("activation_statistics", {})
            if activation_stats:
                click.echo(f"   Activation points: {len(activation_stats)}")
                
                # Show example statistics for first activation point
                first_act = next(iter(activation_stats.values()))
                if "magnitude_stability" in first_act:
                    click.echo(f"   Average magnitude stability: {first_act['magnitude_stability']:.3f}")
        else:
            click.echo("⚠️  No data available for statistics computation.")
            
    except Exception as e:
        raise click.ClickException(f"Failed to compute statistics: {e}")


@activations.command()
@click.argument("storage_dir", type=click.Path(exists=True))
@click.option("--max-age-days", type=int, default=30, help="Maximum age in days to keep")
@click.option("--keep-latest", type=int, default=10, help="Number of latest entries to always keep")
@click.option("--model-name", type=str, help="Filter by model name")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without actually deleting")
def cleanup(
    storage_dir: str,
    max_age_days: int,
    keep_latest: int,
    model_name: Optional[str],
    dry_run: bool
) -> None:
    """Clean up old activation data to save space."""
    
    storage = ActivationStorage(local_dir=storage_dir)
    
    if dry_run:
        click.echo("🔍 Dry run - showing what would be cleaned up:")
        # TODO: Implement dry run logic
        click.echo("   (Dry run functionality not yet implemented)")
    else:
        click.echo("🧹 Cleaning up old activation data...")
        
        try:
            cleaned_count = storage.cleanup_old_data(
                max_age_days=max_age_days,
                keep_latest_n=keep_latest,
                model_name=model_name
            )
            
            click.echo(f"✅ Cleaned up {cleaned_count} old entries")
            
        except Exception as e:
            raise click.ClickException(f"Cleanup failed: {e}")