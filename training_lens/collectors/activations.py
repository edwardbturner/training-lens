"""Collector for general model activations."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.base import DataCollector, DataType


class ActivationsCollector(DataCollector):
    """Collects activations from specified model points."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.activation_points = self.config.get("activation_points", {})
        self.layer_indices = self.config.get("layer_indices", None)
        self.input_data = None

    @property
    def data_type(self) -> DataType:
        return DataType.ACTIVATIONS

    @property
    def supported_model_types(self) -> List[str]:
        return ["full", "lora", "peft"]

    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        """Check if activations can be collected."""
        return True  # Can collect from any model

    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Collect activations from the model.

        Args:
            model: Model to collect activations from
            step: Current training step
            **kwargs: Additional context (should include input_data)

        Returns:
            Dictionary containing collected activations
        """
        input_data = kwargs.get("input_data")
        if input_data is None:
            # Try to get from config or use dummy data
            input_data = self._get_dummy_input(model)
            if input_data is None:
                return None

        try:
            # Clear previous activations
            self.activations.clear()
            self._cleanup_hooks()

            # Register hooks for activation collection
            self._register_hooks(model)

            # Run forward pass to collect activations
            with torch.no_grad():
                model.eval()
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.to(next(model.parameters()).device)
                    _ = model(input_data)
                elif isinstance(input_data, dict):
                    # Handle tokenized inputs
                    input_data = {
                        k: v.to(next(model.parameters()).device)
                        for k, v in input_data.items()
                        if isinstance(v, torch.Tensor)
                    }
                    _ = model(**input_data)
                model.train()

            # Collect and process activations
            collected_activations = {}
            for name, activation in self.activations.items():
                # Move to CPU and compute statistics
                activation_cpu = activation.cpu()

                collected_activations[name] = {
                    "activation": activation_cpu,
                    "shape": list(activation.shape),
                    "dtype": str(activation.dtype),
                    "statistics": {
                        "mean": activation.mean().item(),
                        "std": activation.std().item(),
                        "norm": torch.norm(activation).item(),
                        "min": activation.min().item(),
                        "max": activation.max().item(),
                        "nonzero_ratio": (activation != 0).float().mean().item(),
                    },
                }

            if collected_activations:
                return {
                    "step": step,
                    "activations": collected_activations,
                    "activation_points": list(collected_activations.keys()),
                    "total_points": len(collected_activations),
                    "input_shape": list(input_data.shape) if isinstance(input_data, torch.Tensor) else "complex",
                    "collection_timestamp": torch.tensor(step, dtype=torch.float32),
                }

        except Exception as e:
            print(f"Warning: Activation collection failed: {e}")
        finally:
            self._cleanup_hooks()

        return None

    def _register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks for activation collection."""
        # Register custom activation points
        for name, module_path in self.activation_points.items():
            try:
                module = self._get_module_by_path(model, module_path)
                handle = module.register_forward_hook(self._make_hook_fn(name))
                self.hooks.append(handle)
            except Exception as e:
                print(f"Warning: Failed to register hook for {name} at {module_path}: {e}")

        # Register standard transformer points if layer indices specified
        if self.layer_indices is not None:
            self._register_standard_transformer_hooks(model)

    def _register_standard_transformer_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks for standard transformer activation points."""
        # Detect model architecture
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama-style architecture
            layers = model.model.layers
            # base_path = "model.layers"  # Not used currently
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-style architecture
            layers = model.transformer.h
            # base_path = "transformer.h"  # Not used currently
        else:
            return  # Unknown architecture

        for layer_idx in self.layer_indices:
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]

            # Register hooks for this layer
            points = [
                (f"layer_{layer_idx}_input", layer),
                (f"layer_{layer_idx}_output", layer),
            ]

            # Try to register attention and MLP hooks
            if hasattr(layer, "self_attn") or hasattr(layer, "attn"):
                attn_module = getattr(layer, "self_attn", getattr(layer, "attn", None))
                if attn_module:
                    points.append((f"layer_{layer_idx}_attention", attn_module))

            if hasattr(layer, "mlp"):
                points.append((f"layer_{layer_idx}_mlp", layer.mlp))

            for point_name, module in points:
                try:
                    handle = module.register_forward_hook(self._make_hook_fn(point_name))
                    self.hooks.append(handle)
                except Exception as e:
                    print(f"Warning: Failed to register hook for {point_name}: {e}")

    def _make_hook_fn(self, name: str):
        """Create a hook function for capturing activations."""

        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # Take first element if tuple (common in transformer layers)
                activation = output[0].detach().clone()
            else:
                activation = output.detach().clone()

            self.activations[name] = activation

        return hook_fn

    def _get_module_by_path(self, model: torch.nn.Module, module_path: str) -> nn.Module:
        """Get module by dot-separated path."""
        module = model
        for part in module_path.split("."):
            module = getattr(module, part)
        return module

    def _get_dummy_input(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """Generate dummy input for the model."""
        try:
            # Try to infer input shape from model
            device = next(model.parameters()).device

            # Common sequence length for language models
            seq_length = self.config.get("dummy_seq_length", 32)
            vocab_size = self.config.get("dummy_vocab_size", 50257)  # GPT-2 vocab size

            # Create dummy token IDs
            dummy_input = torch.randint(0, vocab_size, (1, seq_length), device=device)

            return dummy_input

        except Exception:
            return None

    def _cleanup_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
