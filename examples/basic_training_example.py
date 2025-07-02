#!/usr/bin/env python3
"""
Basic LoRA Training Example with Training Lens

This example demonstrates how to use Training Lens to train a LoRA adapter
with comprehensive monitoring and analysis using Unsloth.
"""

import tempfile
from pathlib import Path

from datasets import Dataset

from training_lens import TrainingWrapper
from training_lens.training.config import TrainingConfig


def create_sample_dataset():
    """Create a simple sample dataset for demonstration."""
    # Create sample conversations for training
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {
                    "role": "assistant",
                    "content": "I don't have access to current weather data, but you can check a weather website or app for the most up-to-date information.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you help me with coding?"},
                {
                    "role": "assistant",
                    "content": "Absolutely! I'd be happy to help you with coding. What specific programming question or problem do you have?",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain machine learning"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, AI, and automation.",
                },
            ]
        },
    ] * 20  # Repeat to have enough data

    return Dataset.from_list(conversations)


def main():
    """Main function demonstrating basic LoRA training with Training Lens."""
    print("🚀 Training Lens LoRA Training Example")
    print("=" * 50)

    # Create sample dataset
    print("📂 Creating sample dataset...")
    dataset = create_sample_dataset()
    print(f"   Dataset size: {len(dataset)} examples")

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Evaluation examples: {len(eval_dataset)}")

    # Create training configuration
    print("\n⚙️  Setting up training configuration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "training_output"

        config = TrainingConfig(
            # Model configuration
            model_name="unsloth/llama-2-7b-bnb-4bit",  # Unsloth model for optimal LoRA training
            max_seq_length=512,
            load_in_4bit=True,
            # LoRA configuration
            training_method="lora",  # LoRA-only in training_lens
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=None,  # Auto-detect optimal modules
            # Training parameters
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=100,  # Short training for example
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            # LoRA checkpoint configuration
            checkpoint_interval=10,  # Save every 10 steps for demo
            save_strategy="steps",
            save_steps=10,
            upload_adapter_weights=True,
            upload_gradients=True,
            # Output configuration
            output_dir=output_dir,
            # LoRA analysis settings
            capture_adapter_gradients=True,
            capture_adapter_weights=True,
            capture_lora_activations=False,
            # Unsloth configuration
            unsloth_max_seq_length=512,
            unsloth_dtype=None,  # Auto-detect
            unsloth_load_in_4bit=True,
        )

        print(f"   Model: {config.model_name}")
        print(f"   Method: LoRA (training_lens is LoRA-only)")
        print(f"   LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
        print(f"   Max steps: {config.max_steps}")
        print(f"   Checkpoint interval: {config.checkpoint_interval} steps")
        print(f"   Output directory: {config.output_dir}")
        print(f"   Adapter uploads: weights={config.upload_adapter_weights}, gradients={config.upload_gradients}")

        # Initialize Training Lens wrapper
        print("\n🔧 Initializing Training Lens...")
        wrapper = TrainingWrapper(config)

        # Start training
        print("\n🏋️  Starting training...")
        try:
            results = wrapper.train(
                dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            print("\n✅ Training completed successfully!")
            print(f"   Final loss: {results['train_result'].training_loss:.4f}")
            print(f"   Total steps: {results['train_result'].global_step}")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Model saved to: {results['final_model_path']}")

            # Demonstrate LoRA analysis
            print("\n🔍 Running post-training LoRA analysis...")

            from training_lens.analysis.checkpoint_analyzer import CheckpointAnalyzer

            analyzer = CheckpointAnalyzer(Path(config.output_dir) / "checkpoints")
            report = analyzer.generate_standard_report()

            print(f"   Analyzed {len(analyzer.checkpoints_info)} LoRA checkpoints")

            # Show key LoRA insights
            adapter_gradient_analysis = report.get("adapter_gradient_analysis", {})
            if "mean_cosine_similarity" in adapter_gradient_analysis:
                print(f"   LoRA gradient consistency: {adapter_gradient_analysis['mean_cosine_similarity']:.3f}")

            adapter_weight_analysis = report.get("adapter_weight_analysis", {})
            if "adapter_weight_stability" in adapter_weight_analysis:
                print(f"   LoRA weight stability: {adapter_weight_analysis['adapter_weight_stability']}")

            training_dynamics = report.get("training_dynamics", {})
            if "loss_analysis" in training_dynamics:
                loss_analysis = training_dynamics["loss_analysis"]
                if "loss_reduction_percentage" in loss_analysis:
                    print(f"   Loss improvement: {loss_analysis['loss_reduction_percentage']:.1f}%")

            print("\n📊 LoRA analysis complete!")
            print(f"   Full LoRA adapter analysis available in: {config.output_dir}")
            print("   LoRA-specific metrics: adapter gradients, weights, and training dynamics")

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return 1

    print("\n✨ LoRA training example completed successfully!")
    print("\nNext steps:")
    print("   1. Try with your own dataset")
    print("   2. Experiment with different LoRA parameters (r, alpha, target_modules)")
    print("   3. Add W&B integration for experiment tracking")
    print("   4. Upload LoRA adapters to HuggingFace Hub")
    print("   5. Use different Unsloth-optimized base models")
    print("   6. Analyze LoRA adapter-specific metrics")

    return 0


if __name__ == "__main__":
    exit(main())
