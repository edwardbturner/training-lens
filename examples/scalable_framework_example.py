"""
Example demonstrating the scalable framework architecture.

This example shows how the new framework can easily accommodate:
(a) New raw data collection types during training
(b) New downstream analysis tools

The framework is designed to be plugin-based and extensible.
"""

from pathlib import Path

import torch

# Import the new framework
from training_lens.core import DataType, TrainingLensFramework
from training_lens.core.base import DataAnalyzer, DataCollector
from training_lens.core.integration import create_lora_focused_framework


def main():
    """Demonstrate the scalable framework architecture."""

    print("🔧 Training Lens Scalable Framework Example")
    print("=" * 60)

    # === 1. Basic Framework Usage ===
    print("\n📋 1. Basic Framework Setup")

    # Create a LoRA-focused framework
    framework = create_lora_focused_framework()

    print(f"Available collectors: {list(framework.get_available_collectors().keys())}")
    print(f"Available analyzers: {list(framework.get_available_analyzers().keys())}")

    # === 2. Simulated Training with Data Collection ===
    print("\n🚀 2. Simulated Training with Data Collection")

    # Simulate a simple model for demonstration
    class DummyLoRAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            # Simulate LoRA adapters
            setattr(self.linear, "lora_A", {"default": torch.nn.Linear(10, 2, bias=False)})
            setattr(self.linear, "lora_B", {"default": torch.nn.Linear(2, 5, bias=False)})

        def forward(self, x):
            return self.linear(x)

    model = DummyLoRAModel()

    # Simulate training steps with data collection
    training_steps = [10, 20, 30, 40, 50]

    for step in training_steps:
        # Simulate some training (update weights slightly)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Collect data at this step
        collected_data = framework.collect_training_data(
            model=model,
            step=step,
            optimizer=None,  # Would be real optimizer in practice
            loss=torch.tensor(1.0 / step),  # Simulated decreasing loss
        )

        print(f"   Step {step}: Collected {len(collected_data)} data types")

    # === 3. Analysis of Collected Data ===
    print("\n📊 3. Analysis of Collected Data")

    # Run analysis on all collected data
    analysis_results = framework.analyze_training_data(output_dir=Path("./framework_analysis_output"))

    print(f"Generated {len(analysis_results)} analysis types:")
    for analysis_type, result in analysis_results.items():
        print(f"   - {analysis_type}: {result.get('status', 'completed')}")

    # === 4. Adding New Data Collector (Type A Extensibility) ===
    print("\n🔌 4. Adding New Raw Data Collector")

    # Example: Custom collector for attention patterns
    class AttentionPatternsCollector(DataCollector):
        """Example custom collector for attention patterns."""

        @property
        def data_type(self) -> DataType:
            return DataType.ATTENTION_PATTERNS

        @property
        def supported_model_types(self):
            return ["lora", "full"]

        def can_collect(self, model, step):
            return hasattr(model, "attention_weights")  # Simplified check

        def collect(self, model, step, **kwargs):
            # Simulate attention pattern collection
            return {
                "step": step,
                "attention_patterns": {
                    "layer_0": torch.randn(8, 32, 32),  # Simulated attention matrix
                    "layer_1": torch.randn(8, 32, 32),
                },
                "pattern_entropy": torch.tensor([2.1, 2.3]),
                "attention_sparsity": torch.tensor([0.15, 0.23]),
            }

    # Register the new collector
    framework.register_custom_collector(AttentionPatternsCollector)
    print("   Registered new collector: AttentionPatternsCollector")
    print(f"   Updated available collectors: {list(framework.get_available_collectors().keys())}")

    # === 5. Adding New Analysis Tool (Type B Extensibility) ===
    print("\n🔍 5. Adding New Downstream Analyzer")

    # Example: Custom analyzer for attention patterns
    class AttentionAnalyzer(DataAnalyzer):
        """Example custom analyzer for attention patterns."""

        @property
        def data_type(self) -> DataType:
            return DataType.VISUALIZATION  # Reusing existing type for demo

        @property
        def required_data_types(self):
            return [DataType.ATTENTION_PATTERNS, DataType.ADAPTER_WEIGHTS]

        def can_analyze(self, available_data):
            return DataType.ATTENTION_PATTERNS in available_data

        def analyze(self, data, output_dir=None):
            attention_data = data.get(DataType.ATTENTION_PATTERNS, {})

            if not attention_data:
                return {"status": "no_attention_data"}

            # Simulate analysis
            analysis = {
                "attention_evolution": {
                    "entropy_trend": "increasing",
                    "sparsity_trend": "decreasing",
                    "pattern_stability": 0.87,
                },
                "cross_layer_analysis": {
                    "layer_correlation": 0.65,
                    "information_flow": "bottom_up",
                },
                "adapter_attention_correlation": {
                    "correlation_strength": 0.42,
                    "correlation_type": "positive",
                },
            }

            # Save results if output directory provided
            if output_dir:
                import json

                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "attention_analysis.json", "w") as f:
                    json.dump(analysis, f, indent=2)

            return analysis

    # Register the new analyzer
    framework.register_custom_analyzer(AttentionAnalyzer)
    print("   Registered new analyzer: AttentionAnalyzer")
    print(f"   Updated available analyzers: {list(framework.get_available_analyzers().keys())}")

    # === 6. Demonstrating Type Grouping and Organization ===
    print("\n📁 6. Data Type Organization")

    # Group data types by category
    logging_types = [
        DataType.ADAPTER_WEIGHTS,
        DataType.ADAPTER_GRADIENTS,
        DataType.ACTIVATIONS,
        DataType.LORA_ACTIVATIONS,
        DataType.ATTENTION_PATTERNS,
        DataType.HIDDEN_STATES,
        DataType.PARAMETER_NORMS,
    ]

    analysis_types = [
        DataType.ACTIVATION_ANALYSIS,
        DataType.LORA_ANALYSIS,
        DataType.CONVERGENCE_ANALYSIS,
        DataType.SIMILARITY_ANALYSIS,
        DataType.OVERFITTING_ANALYSIS,
        DataType.RANK_ANALYSIS,
        DataType.VISUALIZATION,
    ]

    print(f"   Raw Data Types (Logging): {len(logging_types)}")
    for dt in logging_types:
        print(f"     - {dt.value}")

    print(f"   Analysis Types (Downstream): {len(analysis_types)}")
    for dt in analysis_types:
        print(f"     - {dt.value}")

    # === 7. Configuration and Pipeline Management ===
    print("\n⚙️ 7. Configuration Management")

    # Create configuration template
    config = framework.create_pipeline_config(save_path=Path("./framework_config.json"))

    print(f"   Generated configuration with {len(config['collectors'])} collectors")
    print(f"   and {len(config['analyzers'])} analyzers")

    # Demonstrate custom configuration
    custom_config = {
        "collectors": {
            DataType.ADAPTER_WEIGHTS.value: {
                "enabled": True,
                "frequency": 5,  # Collect every 5 steps
                "compression": True,
            },
            DataType.ATTENTION_PATTERNS.value: {
                "enabled": True,
                "frequency": 10,
                "max_layers": 12,
            },
        },
        "analyzers": {
            DataType.LORA_ANALYSIS.value: {
                "enabled": True,
                "rank_analysis": True,
                "cross_correlation": True,
            }
        },
    }

    # Create framework with custom config
    custom_framework = TrainingLensFramework(custom_config)
    print(
        f"   Created custom framework which has {len(custom_framework.get_available_collectors())} collectors and "
        f" {len(custom_framework.get_available_analyzers())} analyzers"
    )

    # === 8. Export and Data Management ===
    print("\n💾 8. Data Export and Management")

    # Export collected data
    exported_files = framework.export_data(output_dir=Path("./exported_training_data"), format="json")

    print(f"   Exported {len(exported_files)} data files")

    # Get collection and analysis summaries
    collection_summary = framework.get_collection_summary()
    analysis_summary = framework.get_analysis_summary()

    print(f"   Collection summary: {collection_summary['total_collections']} collections")
    print(f"   Analysis summary: {analysis_summary['total_analyses']} analyses")

    # === 9. Future Extensibility Examples ===
    print("\n🚀 9. Future Extensibility Examples")

    print("   Easy to add new raw data types:")
    print("     - Loss landscapes (DataType.LOSS_LANDSCAPES)")
    print("     - Optimizer states (DataType.OPTIMIZER_STATES)")
    print("     - Embedding drift (DataType.EMBEDDING_STATES)")
    print("     - Memory usage patterns")
    print("     - Hardware utilization metrics")

    print("   Easy to add new analysis types:")
    print("     - Catastrophic forgetting detection")
    print("     - Adversarial robustness analysis")
    print("     - Model compression insights")
    print("     - Transfer learning effectiveness")
    print("     - Multi-modal alignment analysis")

    print("\n✨ Framework Architecture Benefits:")
    print("   🔌 Plugin-based: Easy to add new collectors/analyzers")
    print("   📊 Type-safe: Clear separation of data types")
    print("   ⚙️ Configurable: Fine-grained control over collection/analysis")
    print("   🔄 Extensible: Supports both (a) raw data and (b) analysis extensions")
    print("   📁 Organized: Clear grouping of logging vs downstream types")
    print("   🔍 Auto-discovery: Automatically finds new plugins")
    print("   💾 Export-friendly: Multiple formats and storage options")


# Example of how to create a completely new data type and collector
def demonstrate_custom_data_type():
    """Demonstrate adding a completely new data type."""

    print("\n🆕 Custom Data Type Example:")

    # Step 1: Add new data type to enum (in practice, you'd extend the enum)
    # DataType.CUSTOM_METRIC = "custom_metric"  # This would be added to the enum

    # Step 2: Create collector for the new type
    class CustomMetricCollector(DataCollector):
        @property
        def data_type(self):
            return DataType.HIDDEN_STATES  # Using existing type for demo

        @property
        def supported_model_types(self):
            return ["lora"]

        def can_collect(self, model, step):
            return step % 10 == 0  # Collect every 10 steps

        def collect(self, model, step, **kwargs):
            return {
                "custom_metric": torch.randn(5).tolist(),
                "model_complexity": len(list(model.parameters())),
                "step": step,
            }

    # Step 3: Create analyzer for the new type
    class CustomMetricAnalyzer(DataAnalyzer):
        @property
        def data_type(self):
            return DataType.RANK_ANALYSIS  # Using existing type for demo

        @property
        def required_data_types(self):
            return [DataType.HIDDEN_STATES]

        def can_analyze(self, available_data):
            return DataType.HIDDEN_STATES in available_data

        def analyze(self, data, output_dir=None):
            return {"custom_analysis": "completed"}

    print("   ✅ Created CustomMetricCollector and CustomMetricAnalyzer")
    print("   ✅ Can be registered with framework.register_custom_collector()")
    print("   ✅ Framework will automatically handle data flow")


if __name__ == "__main__":
    main()
    demonstrate_custom_data_type()
