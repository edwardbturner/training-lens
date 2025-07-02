"""
Comprehensive example of activation tracking across training checkpoints.

This example demonstrates how to use the new activation tracking functionality
to analyze how model activations evolve during training, with special focus
on LoRA adapters and their contribution patterns.
"""

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from training_lens import (
    ActivationAnalyzer,
    ActivationExtractor,
    ActivationStorage,
    ActivationVisualizer,
    LoRAActivationTracker,
    LoRAParameterAnalyzer,
)


def main():
    """Run comprehensive activation tracking example."""

    # Configuration
    checkpoint_dir = "./training_output/checkpoints"
    model_name = "microsoft/DialoGPT-medium"
    hf_repo = "username/activation-data"  # Optional HuggingFace repository

    print("🧠 Starting comprehensive activation tracking example...")

    # === 1. Basic Activation Evolution Analysis ===
    print("\n📊 1. Analyzing activation evolution across checkpoints...")

    # Initialize the analyzer
    analyzer = ActivationAnalyzer(checkpoint_dir=checkpoint_dir, model_name=model_name, hf_repo_id=hf_repo)

    # Prepare input data for analysis
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Multiple input samples for robust analysis
    input_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "What is the capital of France?",
        "Can you help me with this problem?",
    ]

    # Tokenize inputs
    input_tensors = []
    for text in input_texts:
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_tensors.append(tokens["input_ids"])

    # Define custom activation points
    custom_activation_points = {
        "input_embeddings": "transformer.wte",
        "layer_0_attention": "transformer.h.0.attn",
        "layer_3_mlp": "transformer.h.3.mlp",
        "layer_6_output": "transformer.h.6",
        "final_layernorm": "transformer.ln_f",
    }

    # Analyze activation evolution
    try:
        evolution_results = analyzer.analyze_activation_evolution(
            input_data=input_tensors,
            activation_points=custom_activation_points,
            layer_indices=[0, 3, 6, 9],  # Analyze specific layers
            lora_analysis=True,  # Include LoRA analysis if available
            checkpoint_steps=[100, 500, 1000],  # Specific checkpoints
        )

        print(f"✅ Analyzed activations across {len(evolution_results['steps_analyzed'])} checkpoints")
        print(f"   Activation points: {len(evolution_results['activation_points'])}")

        # Analyze similarity patterns
        similarity_results = analyzer.compute_activation_similarities(
            reference_step=100, similarity_metric="cosine"  # Compare to checkpoint 100
        )

        print(f"   Mean similarity to reference: {similarity_results['summary']['overall_mean_similarity']:.3f}")

    except Exception as e:
        print(f"❌ Activation analysis failed: {e}")
        return

    # === 2. LoRA-Specific Analysis ===
    print("\n🎯 2. Performing LoRA-specific activation analysis...")

    # Load a checkpoint with LoRA adapters
    try:
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM

        # Load model from a specific checkpoint
        checkpoint_path = f"{checkpoint_dir}/checkpoint-1000"
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto")

        # Initialize LoRA tracker
        lora_tracker = LoRAActivationTracker(model, adapter_name="default")
        lora_tracker.register_lora_hooks()

        # Analyze LoRA contributions
        sample_input = input_tensors[0].to(model.device)
        lora_contributions = lora_tracker.analyze_lora_contribution(sample_input)

        print("   LoRA contribution analysis:")
        for module_name, contrib_data in list(lora_contributions.items())[:3]:  # Show first 3
            lora_contrib = contrib_data["lora_contribution"]
            main_contrib = contrib_data["main_path_contribution"]
            print(f"     {module_name}: LoRA {lora_contrib:.3f}, Main {main_contrib:.3f}")

        # Analyze rank utilization
        rank_utilization = lora_tracker.compute_lora_rank_utilization()

        print("   Rank utilization analysis:")
        for module_name, util_data in list(rank_utilization.items())[:2]:  # Show first 2
            effective_rank = util_data["effective_rank"]
            nominal_rank = util_data["nominal_rank"]
            utilization = util_data["rank_utilization"]
            print(f"     {module_name}: {effective_rank:.1f}/{nominal_rank} ({utilization:.1%})")

        lora_tracker.cleanup()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"⚠️  LoRA analysis failed (model may not have LoRA adapters): {e}")

    # === 3. Parameter Evolution Analysis ===
    print("\n📈 3. Analyzing LoRA parameter evolution...")

    try:
        # Create parameter analyzer
        param_analyzer = LoRAParameterAnalyzer(adapter_name="default")

        # Define checkpoint paths to analyze
        checkpoint_paths = [
            f"{checkpoint_dir}/checkpoint-100",
            f"{checkpoint_dir}/checkpoint-500",
            f"{checkpoint_dir}/checkpoint-1000",
        ]

        # Model loader function
        def load_checkpoint_model(checkpoint_path):
            return AutoModelForCausalLM.from_pretrained(
                checkpoint_path, torch_dtype=torch.float16, device_map="cpu"  # Use CPU to save GPU memory
            )

        # Analyze parameter evolution (if checkpoints exist)
        param_evolution = param_analyzer.analyze_parameter_evolution(
            checkpoint_paths=[], model_loader_fn=load_checkpoint_model  # Empty for demo - would use real paths
        )

        if param_evolution["parameter_data"]:
            print("   Parameter evolution trends:")
            analysis = param_evolution["evolution_analysis"]
            for module_name, module_analysis in list(analysis.items())[:2]:
                a_trend = module_analysis["A_matrix_evolution"]["trend"]
                b_trend = module_analysis["B_matrix_evolution"]["trend"]
                print(f"     {module_name}: A matrix {a_trend}, B matrix {b_trend}")
        else:
            print("   No parameter evolution data available (demo mode)")

    except Exception as e:
        print(f"⚠️  Parameter evolution analysis skipped: {e}")

    # === 4. Activation Storage and Management ===
    print("\n💾 4. Setting up activation storage...")

    # Initialize storage system
    storage = ActivationStorage(
        local_dir="./activation_storage",
        repo_id=hf_repo,  # Optional HuggingFace repository
        create_repo_if_not_exists=False,  # Set to True to create repo
    )

    # Store activation data (if we have any from analysis)
    if hasattr(analyzer, "activation_data") and analyzer.activation_data:
        stored_ids = []
        for step, activations in list(analyzer.activation_data.items())[:2]:  # Store first 2
            data_id = storage.store_activation_data(
                activation_data=activations,
                checkpoint_step=step,
                model_name=model_name,
                activation_config={
                    "custom_points": custom_activation_points,
                    "layer_indices": [0, 3, 6, 9],
                    "lora_analysis": True,
                    "input_samples": len(input_texts),
                },
                metadata={"experiment": "activation_tracking_example", "input_type": "conversational"},
                upload_to_hub=False,  # Set to True to upload to HF Hub
            )
            stored_ids.append(data_id)
            print(f"   Stored activation data for step {step}: {data_id}")

        # Create a dataset from stored data
        if len(stored_ids) > 1:
            dataset_id = storage.create_activation_dataset(
                data_ids=stored_ids,
                dataset_name="example_activation_evolution",
                description="Example dataset showing activation evolution during training",
                upload_to_hub=False,
            )
            print(f"   Created dataset: {dataset_id}")

        # Compute statistics
        stats = storage.compute_activation_statistics(data_ids=stored_ids)

        if stats.get("status") != "no_data":
            print("   Storage statistics:")
            print(f"     Checkpoints analyzed: {len(stats['checkpoint_steps'])}")
            print(f"     Activation points: {len(stats['activation_statistics'])}")

    # === 5. Visualization ===
    print("\n📊 5. Generating visualizations...")

    if hasattr(analyzer, "activation_data") and analyzer.activation_data:
        # Initialize visualizer
        visualizer = ActivationVisualizer()

        # Convert activation data to numpy for visualization
        viz_data = {}
        for step, activations in analyzer.activation_data.items():
            viz_data[step] = {name: tensor.numpy() for name, tensor in activations.items()}

        try:
            # Create evolution plot
            evolution_fig = visualizer.plot_activation_evolution(
                viz_data, save_path="./outputs/activation_evolution.png"
            )
            print("   ✅ Activation evolution plot saved")

            # Create similarity heatmap (if we have similarity data)
            if "similarity_analysis" in locals():
                similarity_fig = visualizer.plot_activation_similarity_heatmap(
                    similarity_results["similarities"],
                    reference_step=similarity_results["reference_step"],
                    save_path="./outputs/similarity_heatmap.png",
                )
                print("   ✅ Similarity heatmap saved")

            # Create comprehensive dashboard
            dashboard_fig = visualizer.create_activation_summary_dashboard(
                viz_data, save_path="./outputs/activation_dashboard.png"
            )
            print("   ✅ Activation dashboard saved")

        except Exception as e:
            print(f"   ⚠️  Visualization failed (matplotlib may not be available): {e}")

    # === 6. CLI Usage Examples ===
    print("\n💻 6. CLI usage examples:")
    print("   # Basic activation analysis")
    print("   training-lens activations analyze ./checkpoints llama2-7b")
    print()
    print("   # Advanced analysis with LoRA and visualization")
    print("   training-lens activations analyze ./checkpoints llama2-7b \\")
    print("     --layers '0,5,10' --lora-analysis --visualize \\")
    print("     --hf-repo 'username/activations' --upload-to-hf")
    print()
    print("   # Analyze specific checkpoints with custom input")
    print("   training-lens activations analyze ./checkpoints llama2-7b \\")
    print("     --steps '100,500,1000' --input-file inputs.txt \\")
    print('     --activation-points \'{"custom": "model.layers.5.mlp"}\' \\')
    print("     --output ./results")
    print()
    print("   # Create dataset from stored activations")
    print("   training-lens activations create-dataset ./storage \\")
    print("     data_id_1 data_id_2 data_id_3 'my_dataset' \\")
    print("     --description 'Training evolution dataset'")

    print("\n✨ Activation tracking example completed!")
    print("\nKey capabilities demonstrated:")
    print("• 📊 Activation evolution analysis across checkpoints")
    print("• 🎯 LoRA-specific activation tracking (A/B matrices)")
    print("• 📈 Parameter evolution monitoring")
    print("• 💾 Activation data storage with HuggingFace integration")
    print("• 📈 Comprehensive visualization tools")
    print("• 💻 Full CLI interface for all operations")


if __name__ == "__main__":
    main()
