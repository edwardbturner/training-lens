"""Collector for LoRA-specific activations (pre-A, post-A/pre-B, post-B)."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..core.base import DataCollector, DataType


class LoRAActivationsCollector(DataCollector):
    """Collects LoRA-specific activations at A and B matrix boundaries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.adapter_name = self.config.get("adapter_name", "default")

    @property
    def data_type(self) -> DataType:
        return DataType.LORA_ACTIVATIONS

    @property
    def supported_model_types(self) -> List[str]:
        return ["lora", "peft"]

    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        """Check if model has LoRA adapters."""
        return self._has_lora_adapters(model)

    def collect(
        self,
        model: torch.nn.Module,
        step: int,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Collect LoRA-specific activations.

        Args:
            model: Model with LoRA adapters
            step: Current training step
            **kwargs: Additional context (should include input_data)

        Returns:
            Dictionary containing LoRA activations
        """
        input_data = kwargs.get('input_data')
        if input_data is None:
            input_data = self._get_dummy_input(model)
            if input_data is None:
                return None

        try:
            # Clear previous activations and hooks
            self.activations.clear()
            self._cleanup_hooks()

            # Register LoRA-specific hooks
            self._register_lora_hooks(model)

            # Run forward pass to collect activations
            with torch.no_grad():
                model.eval()
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.to(next(model.parameters()).device)
                    _ = model(input_data)
                elif isinstance(input_data, dict):
                    input_data = {k: v.to(next(model.parameters()).device)
                                  for k, v in input_data.items()
                                  if isinstance(v, torch.Tensor)}
                    _ = model(**input_data)
                model.train()

            # Process collected activations
            lora_activations = {}

            for name, activation in self.activations.items():
                activation_cpu = activation.cpu()

                # Extract LoRA module name and activation type
                parts = name.split('_')
                module_name = '_'.join(parts[:-1])
                activation_type = parts[-1]

                if module_name not in lora_activations:
                    lora_activations[module_name] = {}

                lora_activations[module_name][activation_type] = {
                    'activation': activation_cpu,
                    'shape': list(activation.shape),
                    'dtype': str(activation.dtype),
                    'statistics': {
                        'mean': activation.mean().item(),
                        'std': activation.std().item(),
                        'norm': torch.norm(activation).item(),
                        'min': activation.min().item(),
                        'max': activation.max().item(),
                    }
                }

            # Compute LoRA-specific metrics
            for module_name, module_activations in lora_activations.items():
                metrics = self._compute_lora_metrics(module_activations)
                lora_activations[module_name]['metrics'] = metrics

            if lora_activations:
                return {
                    'step': step,
                    'adapter_name': self.adapter_name,
                    'lora_activations': lora_activations,
                    'total_modules': len(lora_activations),
                    'input_shape': list(input_data.shape) if isinstance(input_data, torch.Tensor) else "complex",
                    'collection_timestamp': torch.tensor(step, dtype=torch.float32),
                }

        except Exception as e:
            print(f"Warning: LoRA activation collection failed: {e}")
        finally:
            self._cleanup_hooks()

        return None

    def _register_lora_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks for LoRA-specific activation points."""
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (isinstance(module.lora_A, dict) and self.adapter_name in module.lora_A
                        and isinstance(module.lora_B, dict) and self.adapter_name in module.lora_B):

                    try:
                        # Hook input to LoRA path (pre-A)
                        pre_hook = module.register_forward_pre_hook(
                            self._make_pre_lora_hook(name)
                        )
                        self.hooks.append(pre_hook)

                        # Hook A matrix output (post-A, pre-B)
                        lora_A = module.lora_A[self.adapter_name]
                        a_hook = lora_A.register_forward_hook(
                            self._make_post_a_hook(name)
                        )
                        self.hooks.append(a_hook)

                        # Hook B matrix output (post-B)
                        lora_B = module.lora_B[self.adapter_name]
                        b_hook = lora_B.register_forward_hook(
                            self._make_post_b_hook(name)
                        )
                        self.hooks.append(b_hook)

                        # Hook main module output for comparison
                        main_hook = module.register_forward_hook(
                            self._make_main_output_hook(name)
                        )
                        self.hooks.append(main_hook)

                    except Exception as e:
                        print(f"Warning: Failed to register LoRA hooks for {name}: {e}")

    def _make_pre_lora_hook(self, module_name: str):
        """Create hook for input to LoRA path."""
        def hook_fn(module, input):
            if isinstance(input, tuple):
                activation = input[0].detach().clone()
            else:
                activation = input.detach().clone()
            self.activations[f"{module_name}_pre_lora"] = activation
        return hook_fn

    def _make_post_a_hook(self, module_name: str):
        """Create hook for A matrix output."""
        def hook_fn(module, input, output):
            activation = output.detach().clone()
            self.activations[f"{module_name}_post_a"] = activation
        return hook_fn

    def _make_post_b_hook(self, module_name: str):
        """Create hook for B matrix output."""
        def hook_fn(module, input, output):
            activation = output.detach().clone()
            self.activations[f"{module_name}_post_b"] = activation
        return hook_fn

    def _make_main_output_hook(self, module_name: str):
        """Create hook for main module output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activation = output[0].detach().clone()
            else:
                activation = output.detach().clone()
            self.activations[f"{module_name}_main_output"] = activation
        return hook_fn

    def _compute_lora_metrics(self, module_activations: Dict[str, Any]) -> Dict[str, Any]:
        """Compute LoRA-specific metrics from activations."""
        metrics = {}

        # Check if we have the required activations
        required_keys = ['pre_lora', 'post_a', 'post_b']
        available_keys = [key for key in required_keys
                          if key in module_activations]

        if len(available_keys) < 2:
            return metrics

        try:
            # Dimensionality analysis
            if 'pre_lora' in module_activations and 'post_a' in module_activations:
                pre_shape = module_activations['pre_lora']['shape']
                post_a_shape = module_activations['post_a']['shape']

                if len(pre_shape) > 0 and len(post_a_shape) > 0:
                    metrics['dimension_reduction_ratio'] = (
                        pre_shape[-1] / post_a_shape[-1] if post_a_shape[-1] > 0 else 0
                    )

            # Information flow analysis
            if 'pre_lora' in module_activations and 'post_b' in module_activations:
                pre_lora = module_activations['pre_lora']['activation']
                post_b = module_activations['post_b']['activation']

                # Compute correlation between input and final LoRA output
                pre_flat = pre_lora.flatten()
                post_flat = post_b.flatten()

                # Pad to same length for correlation
                max_len = max(len(pre_flat), len(post_flat))
                if len(pre_flat) < max_len:
                    pre_flat = torch.cat([
                        pre_flat,
                        torch.zeros(max_len - len(pre_flat))
                    ])
                if len(post_flat) < max_len:
                    post_flat = torch.cat([
                        post_flat,
                        torch.zeros(max_len - len(post_flat))
                    ])

                correlation = torch.corrcoef(torch.stack([pre_flat, post_flat]))[0, 1]
                metrics['input_output_correlation'] = correlation.item() if not torch.isnan(correlation) else 0.0

            # LoRA contribution analysis
            if 'main_output' in module_activations and 'post_b' in module_activations:
                main_norm = module_activations['main_output']['statistics']['norm']
                lora_norm = module_activations['post_b']['statistics']['norm']

                total_norm = main_norm + lora_norm
                if total_norm > 0:
                    metrics['lora_contribution_ratio'] = lora_norm / total_norm
                    metrics['main_contribution_ratio'] = main_norm / total_norm
                    metrics['lora_to_main_ratio'] = lora_norm / (main_norm + 1e-8)

            # Bottleneck efficiency (how much information preserved through rank reduction)
            if 'pre_lora' in module_activations and 'post_a' in module_activations:
                pre_norm = module_activations['pre_lora']['statistics']['norm']
                post_a_norm = module_activations['post_a']['statistics']['norm']

                metrics['bottleneck_efficiency'] = post_a_norm / (pre_norm + 1e-8)

        except Exception as e:
            print(f"Warning: Failed to compute LoRA metrics: {e}")

        return metrics

    def _has_lora_adapters(self, model: torch.nn.Module) -> bool:
        """Check if model has LoRA adapters."""
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (isinstance(module.lora_A, dict) and self.adapter_name in module.lora_A
                        and isinstance(module.lora_B, dict) and self.adapter_name in module.lora_B):
                    return True
        return False

    def _get_dummy_input(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """Generate dummy input for the model."""
        try:
            device = next(model.parameters()).device
            seq_length = self.config.get("dummy_seq_length", 32)
            vocab_size = self.config.get("dummy_vocab_size", 50257)

            dummy_input = torch.randint(
                0, vocab_size,
                (1, seq_length),
                device=device
            )

            return dummy_input

        except Exception:
            return None

    def _cleanup_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()
