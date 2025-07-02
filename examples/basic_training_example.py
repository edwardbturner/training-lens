#!/usr/bin/env python3
"""
Basic Training Example with Training Lens

This example demonstrates how to use Training Lens to train a model
with comprehensive monitoring and analysis.
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
    """Main function demonstrating basic training with Training Lens."""
    print("🚀 Training Lens Basic Example")
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
            model_name="microsoft/DialoGPT-small",  # Small model for quick example
            max_seq_length=512,
            load_in_4bit=True,
            # Training method
            training_method="lora",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            # Training parameters
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=100,  # Short training for example
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            # Checkpoint configuration (defaults to every step)
            checkpoint_interval=10,  # Save every 10 steps for demo
            save_strategy="steps",
            save_steps=10,
            # Output configuration
            output_dir=output_dir,
            # Analysis settings
            capture_gradients=True,
            capture_weights=True,
            capture_activations=False,
        )

        print(f"   Model: {config.model_name}")
        print(f"   Method: {config.training_method}")
        print(f"   Max steps: {config.max_steps}")
        print(f"   Checkpoint interval: {config.checkpoint_interval} steps")
        print(f"   Output directory: {config.output_dir}")

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

            # Demonstrate analysis
            print("\n🔍 Running post-training analysis...")

            from training_lens.analysis.checkpoint_analyzer import CheckpointAnalyzer

            analyzer = CheckpointAnalyzer(Path(config.output_dir) / "checkpoints")
            report = analyzer.generate_standard_report()

            print(f"   Analyzed {len(analyzer.checkpoints_info)} checkpoints")

            # Show key insights
            gradient_analysis = report.get("gradient_analysis", {})
            if "mean_cosine_similarity" in gradient_analysis:
                print(f"   Gradient consistency: {gradient_analysis['mean_cosine_similarity']:.3f}")

            training_dynamics = report.get("training_dynamics", {})
            if "loss_analysis" in training_dynamics:
                loss_analysis = training_dynamics["loss_analysis"]
                if "loss_reduction_percentage" in loss_analysis:
                    print(f"   Loss improvement: {loss_analysis['loss_reduction_percentage']:.1f}%")

            print("\n📊 Analysis complete!")
            print(f"   Full analysis available in: {config.output_dir}")

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return 1

    print("\n✨ Example completed successfully!")
    print("\nNext steps:")
    print("   1. Try with your own dataset")
    print("   2. Experiment with different hyperparameters")
    print("   3. Add W&B integration for experiment tracking")
    print("   4. Upload models to HuggingFace Hub")

    return 0


if __name__ == "__main__":
    exit(main())
