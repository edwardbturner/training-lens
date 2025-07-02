"""Example demonstrating robust LoRA component loading and analysis.

This example shows how to use the robust LoRA utilities to:
1. Download and cache LoRA components from HuggingFace
2. Analyze LoRA components with comprehensive metrics
3. Integrate with the training-lens framework
"""

import logging
from pathlib import Path

# Import the robust LoRA utilities
from training_lens.utils.lora_utils import (
    get_lora_components_per_layer,
    get_cache_info,
    clear_lora_cache,
    LoRAComponentError,
)
from training_lens.analysis.adapters.lora_analyzer import LoRAAnalyzer
from training_lens.collectors.adapter_weights import AdapterWeightsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_loading():
    """Example 1: Basic LoRA component loading with caching."""
    logger.info("=== Example 1: Basic LoRA Component Loading ===")
    
    # Example using a popular LoRA model (replace with actual repo)
    repo_id = "microsoft/DialoGPT-medium"  # Replace with actual LoRA repo
    
    try:
        # Load LoRA components with automatic caching
        components = get_lora_components_per_layer(
            repo_id=repo_id,
            revision="main",
            # layer_filter="mlp",  # Optional: filter for specific layer types
        )
        
        logger.info(f"Successfully loaded {len(components)} LoRA components")
        
        # Analyze the components
        for layer_name, layer_data in components.items():
            logger.info(f"Layer: {layer_name}")
            logger.info(f"  Rank: {layer_data['rank']}")
            logger.info(f"  Scaling: {layer_data['scaling']}")
            logger.info(f"  A matrix shape: {layer_data['shape_A']}")
            logger.info(f"  B matrix shape: {layer_data['shape_B']}")
            logger.info(f"  Effective norm: {layer_data['statistics']['effective_norm']:.4f}")
            
            if len(components) > 3:  # Only show first 3 for brevity
                logger.info("  ... (showing first 3 layers)")
                break
                
    except LoRAComponentError as e:
        logger.error(f"Failed to load LoRA components: {e}")
        logger.info("Note: Replace repo_id with an actual LoRA model repository")


def example_with_analyzer():
    """Example 2: Using LoRA analyzer with external repository."""
    logger.info("\n=== Example 2: LoRA Analysis with External Repository ===")
    
    # Initialize the analyzer
    analyzer = LoRAAnalyzer()
    
    repo_id = "microsoft/DialoGPT-medium"  # Replace with actual LoRA repo
    
    try:
        # Perform analysis directly from repository
        analysis_results = analyzer._analyze_from_repo(
            repo_id=repo_id,
            revision="main",
            layer_filter="mlp",  # Focus on MLP layers
        )
        
        logger.info("Analysis Results:")
        logger.info(f"  Source: {analysis_results.get('source', 'unknown')}")
        logger.info(f"  Total layers: {analysis_results.get('total_layers', 0)}")
        
        if "global_statistics" in analysis_results:
            stats = analysis_results["global_statistics"]
            logger.info(f"  Mean A norm: {stats.get('mean_A_norm', 0):.4f}")
            logger.info(f"  Mean B norm: {stats.get('mean_B_norm', 0):.4f}")
            logger.info(f"  Mean effective norm: {stats.get('mean_effective_norm', 0):.4f}")
            logger.info(f"  Total parameters: {stats.get('total_parameters', 0):,}")
        
        # Show layer-specific analysis for first few layers
        layer_analysis = analysis_results.get("layer_analysis", {})
        for i, (layer_name, layer_data) in enumerate(layer_analysis.items()):
            if i >= 2:  # Only show first 2 layers
                break
                
            logger.info(f"\n  Layer {layer_name}:")
            logger.info(f"    Rank: {layer_data.get('rank', 'unknown')}")
            
            if "svd_analysis" in layer_data:
                svd = layer_data["svd_analysis"]
                logger.info(f"    Effective rank: {svd.get('effective_rank', 'unknown')}")
                logger.info(f"    Rank utilization: {svd.get('rank_utilization', 0):.2%}")
                logger.info(f"    Condition number: {svd.get('condition_number', 'inf'):.2f}")
                
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.info("Note: Replace repo_id with an actual LoRA model repository")


def example_with_collector():
    """Example 3: Using collector with external repository support."""
    logger.info("\n=== Example 3: Collector with External Repository ===")
    
    # Initialize collector
    collector = AdapterWeightsCollector(config={"adapter_name": "default"})
    
    # Simulate model (in practice, this would be your actual model)
    import torch
    mock_model = torch.nn.Linear(10, 5)
    
    repo_id = "microsoft/DialoGPT-medium"  # Replace with actual LoRA repo
    
    try:
        # Collect adapter weights with external repository fallback
        collected_data = collector.collect(
            model=mock_model,
            step=100,
            repo_id=repo_id,  # This triggers external loading
            revision="main",
        )
        
        if collected_data:
            logger.info("Successfully collected adapter data:")
            logger.info(f"  Source: {collected_data.get('source', 'unknown')}")
            logger.info(f"  Step: {collected_data.get('step', 'unknown')}")
            logger.info(f"  Total adapters: {collected_data.get('total_adapters', 0)}")
            
            # Show some adapter details
            adapter_weights = collected_data.get("adapter_weights", {})
            for i, (layer_name, layer_data) in enumerate(adapter_weights.items()):
                if i >= 2:  # Only show first 2 layers
                    break
                    
                logger.info(f"\n  Layer {layer_name}:")
                logger.info(f"    Rank: {layer_data.get('rank', 'unknown')}")
                
                if "statistics" in layer_data:
                    stats = layer_data["statistics"]
                    logger.info(f"    A norm: {stats.get('A_norm', 0):.4f}")
                    logger.info(f"    B norm: {stats.get('B_norm', 0):.4f}")
                    logger.info(f"    Effective norm: {stats.get('effective_norm', 0):.4f}")
        else:
            logger.info("No adapter data collected (expected for mock model)")
            
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        logger.info("Note: Replace repo_id with an actual LoRA model repository")


def example_cache_management():
    """Example 4: Cache management and monitoring."""
    logger.info("\n=== Example 4: Cache Management ===")
    
    # Get cache information
    cache_info = get_cache_info()
    logger.info("Cache Information:")
    logger.info(f"  Cache directory: {cache_info['cache_dir']}")
    logger.info(f"  Cache exists: {cache_info['exists']}")
    logger.info(f"  Total files: {cache_info['total_files']}")
    logger.info(f"  Total size: {cache_info['total_size_bytes']} bytes")
    logger.info(f"  Models cached: {cache_info['models_cached']}")
    logger.info(f"  Memory cache entries: {cache_info['memory_cache_entries']}")
    
    # Demonstrate cache clearing (commented out to avoid deleting cache)
    # deleted_count = clear_lora_cache()
    # logger.info(f"Cleared cache: deleted {deleted_count} files")


def example_advanced_usage():
    """Example 5: Advanced usage with custom configurations."""
    logger.info("\n=== Example 5: Advanced Usage ===")
    
    repo_id = "microsoft/DialoGPT-medium"  # Replace with actual LoRA repo
    
    try:
        # Advanced loading with custom parameters
        components = get_lora_components_per_layer(
            repo_id=repo_id,
            subfolder="lora_weights",  # If LoRA weights are in a subfolder
            revision="main",
            layer_filter="attn",  # Focus on attention layers
            force_download=False,  # Use cache if available
            device="cpu",  # Load on CPU
        )
        
        logger.info(f"Advanced loading: {len(components)} attention layers loaded")
        
        # Custom analysis focusing on rank utilization
        total_params = 0
        utilized_ranks = []
        
        for layer_name, layer_data in components.items():
            rank = layer_data["rank"]
            shape_A = layer_data["shape_A"]
            shape_B = layer_data["shape_B"]
            
            # Calculate parameter efficiency
            lora_params = shape_A[0] * shape_A[1] + shape_B[0] * shape_B[1]
            full_params = shape_B[0] * shape_A[1]  # Equivalent full matrix
            efficiency = lora_params / full_params if full_params > 0 else 0
            
            total_params += lora_params
            utilized_ranks.append(rank)
            
            logger.info(f"  {layer_name}: rank={rank}, efficiency={efficiency:.2%}")
        
        logger.info(f"\nSummary:")
        logger.info(f"  Total LoRA parameters: {total_params:,}")
        logger.info(f"  Average rank: {sum(utilized_ranks) / len(utilized_ranks) if utilized_ranks else 0:.1f}")
        logger.info(f"  Rank distribution: min={min(utilized_ranks) if utilized_ranks else 0}, max={max(utilized_ranks) if utilized_ranks else 0}")
        
    except Exception as e:
        logger.error(f"Advanced usage failed: {e}")
        logger.info("Note: Replace repo_id with an actual LoRA model repository")


def main():
    """Run all examples."""
    logger.info("Starting Robust LoRA Loading Examples")
    logger.info("=" * 50)
    
    # Note: Most examples will show warnings about the example repo_id
    # In practice, replace with actual LoRA model repositories
    
    example_basic_loading()
    example_with_analyzer()
    example_with_collector()
    example_cache_management()
    example_advanced_usage()
    
    logger.info("\n" + "=" * 50)
    logger.info("Examples completed!")
    logger.info("\nTo use with real LoRA models:")
    logger.info("1. Replace 'microsoft/DialoGPT-medium' with actual LoRA repository IDs")
    logger.info("2. Ensure you have the required dependencies: huggingface_hub, safetensors")
    logger.info("3. Check that the LoRA models have the expected structure (adapter_config.json, adapter_model.*)")


if __name__ == "__main__":
    main()