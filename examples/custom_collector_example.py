#!/usr/bin/env python3
"""
Example: Creating a Custom Data Collector

This example demonstrates how to create and register a custom data collector
using the training-lens plugin architecture.
"""

from typing import Any, Dict, List, Optional

import torch
import numpy as np

from training_lens.core.base import DataCollector, DataType
from training_lens.core.collector_registry import register_collector
from training_lens.training.metrics_collector_v2 import MetricsCollectorV2
from training_lens.training.config import TrainingConfig


# Define a custom data type (optional - can use existing ones)
# For this example, we'll use DataType.LOSS_LANDSCAPES to collect loss landscape data


class LossLandscapeCollector(DataCollector):
    """Custom collector for sampling the loss landscape around current parameters.
    
    This collector creates small perturbations of the model parameters and
    evaluates the loss at these points to understand the local loss landscape.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the loss landscape collector.
        
        Args:
            config: Configuration with keys:
                - num_samples: Number of perturbation samples (default: 10)
                - perturbation_scale: Scale of parameter perturbations (default: 0.01)
                - only_lora: Only perturb LoRA parameters (default: True)
        """
        super().__init__(config)
        self.num_samples = self.config.get("num_samples", 10)
        self.perturbation_scale = self.config.get("perturbation_scale", 0.01)
        self.only_lora = self.config.get("only_lora", True)
        self.loss_fn = None
    
    @property
    def data_type(self) -> DataType:
        """Return the data type this collector handles."""
        return DataType.LOSS_LANDSCAPES
    
    @property
    def supported_model_types(self) -> List[str]:
        """Return supported model types."""
        return ["lora", "full"]  # Works with both LoRA and full models
    
    def setup(self, model: torch.nn.Module, loss_fn: Optional[Any] = None, **kwargs):
        """Setup the collector with model and loss function.
        
        Args:
            model: The model being trained
            loss_fn: Loss function to evaluate (optional)
            **kwargs: Additional setup parameters
        """
        self.loss_fn = loss_fn
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        """Check if we can collect at this step.
        
        Only collect every 100 steps to avoid overhead.
        """
        return step % 100 == 0 and self.loss_fn is not None
    
    def collect(
        self, 
        model: torch.nn.Module, 
        step: int,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Collect loss landscape data.
        
        Args:
            model: Current model
            step: Training step
            **kwargs: Additional context (e.g., current batch data)
            
        Returns:
            Dictionary with loss landscape samples
        """
        if not self.can_collect(model, step):
            return None
        
        # Get current model state
        original_state = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # Collect parameter names to perturb
        param_names = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.only_lora and not ("lora" in name.lower() or "adapter" in name.lower()):
                continue
            param_names.append(name)
        
        if not param_names:
            return None
        
        # Generate perturbations and evaluate loss
        perturbations = []
        losses = []
        
        # Include the original point
        if "batch" in kwargs and self.loss_fn:
            try:
                with torch.no_grad():
                    original_loss = self._evaluate_loss(model, kwargs["batch"])
                    losses.append(float(original_loss))
                    perturbations.append({"type": "original", "scale": 0.0})
            except Exception as e:
                self.logger.warning(f"Failed to evaluate original loss: {e}")
        
        # Generate random perturbations
        for i in range(self.num_samples):
            # Create perturbation
            perturbation = {}
            for name in param_names:
                param = dict(model.named_parameters())[name]
                # Random direction normalized by parameter norm
                direction = torch.randn_like(param.data)
                param_norm = param.data.norm()
                if param_norm > 0:
                    direction = direction * (param_norm * self.perturbation_scale)
                perturbation[name] = direction
            
            # Apply perturbation
            for name, delta in perturbation.items():
                dict(model.named_parameters())[name].data.add_(delta)
            
            # Evaluate loss at perturbed point
            if "batch" in kwargs and self.loss_fn:
                try:
                    with torch.no_grad():
                        perturbed_loss = self._evaluate_loss(model, kwargs["batch"])
                        losses.append(float(perturbed_loss))
                        perturbations.append({
                            "type": "random",
                            "scale": self.perturbation_scale,
                            "index": i
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate perturbed loss: {e}")
            
            # Restore original parameters
            for name, original_data in original_state.items():
                dict(model.named_parameters())[name].data.copy_(original_data)
        
        # Compute landscape statistics
        if losses:
            losses_array = np.array(losses)
            landscape_stats = {
                "mean_loss": float(np.mean(losses_array)),
                "std_loss": float(np.std(losses_array)),
                "min_loss": float(np.min(losses_array)),
                "max_loss": float(np.max(losses_array)),
                "loss_range": float(np.max(losses_array) - np.min(losses_array)),
                "smoothness": float(np.std(losses_array) / (np.mean(losses_array) + 1e-8)),
            }
        else:
            landscape_stats = {}
        
        return {
            "step": step,
            "num_samples": len(losses),
            "perturbation_scale": self.perturbation_scale,
            "losses": losses,
            "perturbations": perturbations,
            "statistics": landscape_stats,
            "perturbed_parameters": param_names,
        }
    
    def _evaluate_loss(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        """Evaluate loss on a batch of data."""
        # This is a simplified example - in practice, you'd use the actual training loss
        outputs = model(**batch)
        return outputs.loss if hasattr(outputs, 'loss') else outputs
    
    def get_metrics(self, collected_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract metrics from collected data for logging."""
        metrics = {}
        if "statistics" in collected_data:
            stats = collected_data["statistics"]
            metrics["loss_landscape_smoothness"] = stats.get("smoothness", 0)
            metrics["loss_landscape_range"] = stats.get("loss_range", 0)
            metrics["loss_landscape_std"] = stats.get("std_loss", 0)
        return metrics


def demonstrate_custom_collector():
    """Demonstrate using a custom collector with training."""
    print("🔌 Custom Collector Example")
    print("=" * 50)
    
    # Register the custom collector
    print("\n1️⃣ Registering custom loss landscape collector...")
    register_collector(
        DataType.LOSS_LANDSCAPES,
        LossLandscapeCollector,
        config={
            "num_samples": 5,
            "perturbation_scale": 0.001,
            "only_lora": True,
        },
        enabled=True
    )
    print("   ✓ Collector registered")
    
    # Create a training configuration that uses the enhanced metrics collector
    print("\n2️⃣ Creating training configuration...")
    config = TrainingConfig(
        model_name="gpt2",  # Small model for example
        output_dir="./custom_collector_output",
        max_steps=500,
        checkpoint_interval=100,
    )
    
    # Create metrics collector with custom collector enabled
    print("\n3️⃣ Initializing metrics collector with custom collector...")
    metrics_collector = MetricsCollectorV2(
        enabled_collectors={
            DataType.ADAPTER_WEIGHTS,
            DataType.ADAPTER_GRADIENTS,
            DataType.LOSS_LANDSCAPES,  # Our custom collector
        }
    )
    
    print(f"   ✓ Enabled collectors: {metrics_collector.registry.list_enabled()}")
    
    # Demonstrate runtime collector management
    print("\n4️⃣ Runtime collector management:")
    
    # Add another custom collector at runtime
    class SimpleMetricsCollector(DataCollector):
        """A simple collector that just counts steps."""
        
        @property
        def data_type(self) -> DataType:
            return DataType.PARAMETER_NORMS
        
        @property  
        def supported_model_types(self) -> List[str]:
            return ["all"]
        
        def can_collect(self, model: torch.nn.Module, step: int) -> bool:
            return True
        
        def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
            # Simple example: collect parameter norms
            norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    norms[name] = float(param.data.norm())
            
            return {
                "step": step,
                "parameter_norms": norms,
                "total_norm": float(sum(n**2 for n in norms.values())**0.5)
            }
    
    # Add the new collector
    print("   • Adding parameter norms collector at runtime...")
    metrics_collector.add_collector(
        DataType.PARAMETER_NORMS,
        SimpleMetricsCollector,
        config={}
    )
    print(f"   ✓ Updated collectors: {metrics_collector.registry.list_enabled()}")
    
    # Disable a collector
    print("   • Disabling activations collector...")
    metrics_collector.registry.disable(DataType.ACTIVATIONS)
    print(f"   ✓ Active collectors: {metrics_collector.registry.list_enabled()}")
    
    print("\n✅ Custom collector demonstration complete!")
    print("\nKey takeaways:")
    print("• Easy to create custom collectors by inheriting from DataCollector")
    print("• Collectors can be registered globally or added to specific training runs")
    print("• Runtime management allows dynamic enable/disable of collectors")
    print("• Each collector can define its own collection frequency and logic")
    print("• Collected data is automatically saved with checkpoints")


if __name__ == "__main__":
    demonstrate_custom_collector()