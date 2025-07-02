#!/usr/bin/env python3
"""
Advanced Training Example with Training Lens

This example demonstrates advanced features including:
- W&B integration
- HuggingFace Hub upload
- Custom configuration
- Real-time analysis
"""

# Import unsloth before other imports for optimal performance
import unsloth  # noqa: F401, isort: skip

import os
import tempfile
from pathlib import Path

from datasets import Dataset, load_dataset

from training_lens import LoRATrainingWrapper
from training_lens.training.config import TrainingConfig


def prepare_dataset():
    """Prepare a real dataset for training."""
    print("📂 Loading dataset...")

    try:
        # Try to load a small dataset for demonstration
        # Using a subset of the squad dataset
        dataset = load_dataset("squad", split="train[:100]")  # Small subset

        # Convert to conversation format
        conversations = []
        for example in dataset:
            # Type-safe access to dataset fields
            example_dict = dict(example)  # Convert to dict for type safety
            question = str(example_dict.get("question", ""))
            context = str(example_dict.get("context", ""))
            answers = example_dict.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []

            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": f"Question: {question}\nContext: {context}"},
                        {
                            "role": "assistant",
                            "content": answer_texts[0] if answer_texts else "I don't know.",
                        },
                    ]
                }
            )

        return Dataset.from_list(conversations)

    except Exception as e:
        print(f"   ⚠️  Could not load dataset ({e}), using synthetic data")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """Create a larger synthetic dataset."""
    topics = [
        ("Python programming", "Python is a versatile programming language"),
        ("Machine learning", "ML helps computers learn from data"),
        ("Data science", "Data science combines statistics and programming"),
        ("Web development", "Web development creates websites and applications"),
        ("Artificial intelligence", "AI simulates human intelligence in machines"),
    ]

    conversations = []
    for topic, answer in topics:
        for i in range(20):  # 20 variations per topic
            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": f"Can you tell me about {topic}? (variation {i+1})"},
                        {"role": "assistant", "content": f"{answer}. This is variation {i+1} of the explanation."},
                    ]
                }
            )

    return Dataset.from_list(conversations)


def main():
    """Advanced training example with full feature demonstration."""
    print("🚀 Training Lens Advanced Example")
    print("=" * 50)

    # Check for API keys (optional)
    wandb_key = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    print("🔑 API Key Status:")
    print(f"   W&B API Key: {'✅ Found' if wandb_key else '❌ Not found (W&B disabled)'}")
    print(f"   HF Token: {'✅ Found' if hf_token else '❌ Not found (HF upload disabled)'}")

    # Prepare dataset
    dataset = prepare_dataset()
    print(f"   Dataset size: {len(dataset)} examples")

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Evaluation examples: {len(eval_dataset)}")

    # Advanced configuration
    print("\\n⚙️  Setting up advanced configuration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "advanced_training"
        logs_dir = output_dir / "logs"

        config = TrainingConfig(
            # Model configuration - Use a more compatible model
            model_name="microsoft/DialoGPT-small",  # Smaller model for better compatibility
            max_seq_length=512,  # Reduced for memory efficiency
            load_in_4bit=False,  # Disable 4bit loading for compatibility
            # Training method - Use full fine-tuning to avoid unsloth issues
            training_method="full",  # Use full fine-tuning instead of LoRA
            # LoRA configuration (not used with full fine-tuning)
            lora_r=16,  # Reduced rank for stability
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            # Training parameters
            per_device_train_batch_size=1,  # Reduced for stability
            gradient_accumulation_steps=4,
            warmup_steps=25,
            max_steps=50,  # Further reduced for demo
            learning_rate=5e-5,  # Lower learning rate for full fine-tuning
            fp16=False,  # Disable fp16 to avoid conflict with bf16
            logging_steps=10,
            # Checkpoint configuration
            checkpoint_interval=25,
            save_strategy="steps",
            save_steps=25,
            # Output configuration
            output_dir=str(output_dir),  # Convert to string
            logging_dir=str(logs_dir),  # Convert to string
            # Integration settings (only if API keys available)
            wandb_project="training-lens-advanced-example" if wandb_key else None,
            wandb_run_name="advanced-training-demo",
            hf_hub_repo=None,  # Set to your repo if desired: "username/model-name"
            # Advanced analysis settings
            capture_gradients=True,
            capture_weights=True,
            capture_activations=True,  # Enable activation capture
        )

        print(f"   Model: {config.model_name}")
        print(f"   LoRA rank: {config.lora_r}")
        print(f"   Batch size (effective): {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Max steps: {config.max_steps}")

        if config.wandb_project:
            print(f"   W&B project: {config.wandb_project}")
        if config.hf_hub_repo:
            print(f"   HF repository: {config.hf_hub_repo}")

        # Initialize wrapper
        print("\\n🔧 Initializing Training Lens with advanced features...")
        wrapper = LoRATrainingWrapper(config)

        # Show model information
        print("\\n🤖 Model Information:")
        wrapper.setup_model_and_tokenizer()

        # Start training
        print("\\n🏋️  Starting advanced training...")
        print("   Features enabled:")
        print("   • Comprehensive gradient analysis")
        print("   • Weight evolution tracking")
        print("   • Real-time metrics collection")
        if config.wandb_project:
            print("   • W&B experiment tracking")
        if config.hf_hub_repo:
            print("   • HuggingFace Hub upload")

        try:
            results = wrapper.train(
                dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            print("\\n✅ Advanced training completed!")
            print(f"   Final loss: {results['train_result'].training_loss:.4f}")
            print(f"   Total steps: {results['train_result'].global_step}")
            print(f"   Training time: {results['training_time']:.2f}s")
            print(f"   Avg. steps/sec: {results['train_result'].global_step / results['training_time']:.2f}")

            # Advanced analysis
            print("\\n🔍 Running comprehensive analysis...")

            # Import analysis modules with proper error handling
            try:
                from training_lens.analysis import CheckpointAnalyzer, GradientAnalyzer, StandardReports
            except ImportError as e:
                print(f"   ⚠️  Analysis modules not available: {e}")
                print("   Skipping advanced analysis...")
                return 0

            # Initialize analyzers with automatic HuggingFace fallback (if configured)
            checkpoint_dir = Path(config.output_dir) / "checkpoints"
            analyzer = CheckpointAnalyzer(
                checkpoint_dir,
                hf_repo_id=config.hf_hub_repo,  # Enables automatic fallback loading
                auto_download=True,
            )
            reports = StandardReports(analyzer)

            # Generate executive summary
            executive_summary = reports.generate_executive_summary()
            print("\\n📊 Executive Summary:")

            training_overview = executive_summary.get("training_overview", {})
            print(f"   Checkpoints: {training_overview.get('total_checkpoints', 0)}")
            print(f"   Training steps: {training_overview.get('training_steps', 0)}")

            performance = executive_summary.get("performance_metrics", {})
            if "loss_improvement" in performance and performance["loss_improvement"]:
                improvement = performance["loss_improvement"]["relative_percent"]
                print(f"   Loss improvement: {improvement:.1f}%")

            model_health = executive_summary.get("model_health", {})
            print(f"   Gradient health: {model_health.get('gradient_health', 'unknown')}")
            print(f"   Weight stability: {model_health.get('weight_stability', 'unknown')}")
            print(f"   Convergence: {model_health.get('convergence_status', 'unknown')}")

            # Detailed gradient analysis
            print("\\n🔄 Gradient Analysis:")
            gradient_data = {}
            for cp in analyzer.checkpoints_info:
                metrics = analyzer.load_checkpoint_metrics(cp["step"])
                if metrics:
                    gradient_data.update(metrics)

            if gradient_data:
                grad_analyzer = GradientAnalyzer(gradient_data)
                grad_report = grad_analyzer.generate_gradient_report()

                consistency = grad_report.get("consistency_analysis", {})
                print(f"   Mean cosine similarity: {consistency.get('mean_similarity', 0):.3f}")
                print(f"   Consistency level: {consistency.get('consistency_level', 'unknown')}")

                magnitude = grad_report.get("magnitude_analysis", {})
                print(f"   Gradient explosion risk: {magnitude.get('explosion_risk', {}).get('risk_level', 'unknown')}")
                print(f"   Gradient vanishing risk: {magnitude.get('vanishing_risk', {}).get('risk_level', 'unknown')}")

            # Weight analysis
            print("\\n⚖️  Weight Analysis:")
            weight_analysis = analyzer.analyze_weight_evolution()
            print(f"   Weight stability: {weight_analysis.get('weight_stability', 'unknown')}")

            # Training diagnostics
            print("\\n🏥 Training Diagnostics:")
            diagnostics = reports.generate_training_diagnostics()
            print(f"   Overall health: {diagnostics.get('overall_health', 'unknown')}")

            critical_issues = diagnostics.get("critical_issues", [])
            if critical_issues:
                print(f"   Critical issues: {len(critical_issues)}")
                for issue in critical_issues[:3]:  # Show first 3
                    print(f"     • {issue.get('description', 'Unknown issue')}")

            warnings = diagnostics.get("warnings", [])
            if warnings:
                print(f"   Warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3
                    print(f"     • {warning.get('description', 'Unknown warning')}")

            # Recommendations
            recommendations = diagnostics.get("recommendations", [])
            if recommendations:
                print("\\n💡 Recommendations:")
                for rec in recommendations[:5]:  # Show first 5
                    print(f"   • {rec}")

            # Export comprehensive report
            print("\\n📄 Exporting comprehensive report...")
            report_path = output_dir / "comprehensive_report.json"
            reports.export_report("technical", report_path, format="json")
            print(f"   Technical report: {report_path}")

            # Show integration results
            if wrapper.wandb_integration and wrapper.wandb_integration.is_active:
                print(f"\\n📈 W&B Dashboard: {wrapper.wandb_integration.get_run_url()}")

            if wrapper.hf_integration and config.hf_hub_repo:
                print(f"\\n🤗 HuggingFace Model: https://huggingface.co/{config.hf_hub_repo}")

        except Exception as e:
            print(f"\\n❌ Advanced training failed: {e}")
            return 1

    print("\\n✨ Advanced example completed successfully!")
    print("\\nAdvanced features demonstrated:")
    print("   ✅ Comprehensive model training")
    print("   ✅ Real-time gradient analysis")
    print("   ✅ Weight evolution tracking")
    print("   ✅ Training diagnostics")
    print("   ✅ Executive reporting")
    print("   ✅ Technical analysis")

    return 0


if __name__ == "__main__":
    exit(main())
