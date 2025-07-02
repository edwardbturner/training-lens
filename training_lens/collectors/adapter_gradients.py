"""Collector for adapter/LoRA gradient vectors."""

from typing import Any, Dict, List, Optional

import torch

from ..core.base import DataCollector, DataType


class AdapterGradientsCollector(DataCollector):
    """Collects gradients for LoRA adapter parameters."""
    
    @property
    def data_type(self) -> DataType:
        return DataType.ADAPTER_GRADIENTS
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["lora", "peft"]
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        """Check if model has LoRA adapters with gradients."""
        return self._has_lora_adapters_with_gradients(model)
    
    def collect(
        self, 
        model: torch.nn.Module, 
        step: int, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Collect LoRA adapter gradients.
        
        Args:
            model: Model with LoRA adapters
            step: Current training step
            **kwargs: Additional context (should include optimizer)
            
        Returns:
            Dictionary containing adapter gradients
        """
        if not self.can_collect(model, step):
            return None
        
        adapter_gradients = {}
        adapter_name = self.config.get("adapter_name", "default")
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (isinstance(module.lora_A, dict) and adapter_name in module.lora_A and
                    isinstance(module.lora_B, dict) and adapter_name in module.lora_B):
                    
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                    
                    gradients_data = {}
                    
                    # Collect A matrix gradients
                    if lora_A.weight.grad is not None:
                        grad_A = lora_A.weight.grad.data.clone().cpu()
                        gradients_data['A_gradient'] = grad_A
                        gradients_data['A_grad_norm'] = torch.norm(grad_A).item()
                        gradients_data['A_grad_mean'] = grad_A.mean().item()
                        gradients_data['A_grad_std'] = grad_A.std().item()
                    
                    # Collect B matrix gradients
                    if lora_B.weight.grad is not None:
                        grad_B = lora_B.weight.grad.data.clone().cpu()
                        gradients_data['B_gradient'] = grad_B
                        gradients_data['B_grad_norm'] = torch.norm(grad_B).item()
                        gradients_data['B_grad_mean'] = grad_B.mean().item()
                        gradients_data['B_grad_std'] = grad_B.std().item()
                    
                    # Compute effective gradient (chain rule for B @ A)
                    if 'A_gradient' in gradients_data and 'B_gradient' in gradients_data:
                        # Effective gradient considering the multiplication B @ A
                        weight_A = lora_A.weight.data
                        weight_B = lora_B.weight.data
                        grad_A = gradients_data['A_gradient']
                        grad_B = gradients_data['B_gradient']
                        
                        # Effective gradient for the combined operation
                        effective_grad = grad_B @ weight_A + weight_B @ grad_A
                        gradients_data['effective_gradient'] = effective_grad.cpu()
                        gradients_data['effective_grad_norm'] = torch.norm(effective_grad).item()
                    
                    # Compute gradient ratios and directions
                    if 'A_grad_norm' in gradients_data and 'B_grad_norm' in gradients_data:
                        gradients_data['grad_norm_ratio'] = (
                            gradients_data['A_grad_norm'] / 
                            (gradients_data['B_grad_norm'] + 1e-8)
                        )
                    
                    # Compute cosine similarity between gradients
                    if 'A_gradient' in gradients_data and 'B_gradient' in gradients_data:
                        grad_A_flat = gradients_data['A_gradient'].flatten()
                        grad_B_flat = gradients_data['B_gradient'].flatten()
                        
                        # Pad to same length for cosine similarity
                        max_len = max(len(grad_A_flat), len(grad_B_flat))
                        if len(grad_A_flat) < max_len:
                            grad_A_flat = torch.cat([
                                grad_A_flat, 
                                torch.zeros(max_len - len(grad_A_flat))
                            ])
                        if len(grad_B_flat) < max_len:
                            grad_B_flat = torch.cat([
                                grad_B_flat, 
                                torch.zeros(max_len - len(grad_B_flat))
                            ])
                        
                        cosine_sim = torch.nn.functional.cosine_similarity(
                            grad_A_flat.unsqueeze(0), 
                            grad_B_flat.unsqueeze(0)
                        )
                        gradients_data['gradient_cosine_similarity'] = cosine_sim.item()
                    
                    if gradients_data:
                        adapter_gradients[name] = gradients_data
        
        if adapter_gradients:
            # Compute global gradient statistics
            all_grad_norms = []
            for module_grads in adapter_gradients.values():
                if 'effective_grad_norm' in module_grads:
                    all_grad_norms.append(module_grads['effective_grad_norm'])
            
            global_stats = {}
            if all_grad_norms:
                global_stats = {
                    'global_grad_norm': sum(all_grad_norms),
                    'mean_module_grad_norm': sum(all_grad_norms) / len(all_grad_norms),
                    'max_module_grad_norm': max(all_grad_norms),
                    'min_module_grad_norm': min(all_grad_norms),
                }
            
            return {
                'step': step,
                'adapter_name': adapter_name,
                'adapter_gradients': adapter_gradients,
                'global_statistics': global_stats,
                'total_adapters': len(adapter_gradients),
                'collection_timestamp': torch.tensor(step, dtype=torch.float32),
            }
        
        return None
    
    def _has_lora_adapters_with_gradients(self, model: torch.nn.Module) -> bool:
        """Check if model has LoRA adapters with gradients."""
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Check if any adapter has gradients
                for adapter_name in getattr(module, 'lora_A', {}):
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                    if (lora_A.weight.grad is not None or 
                        lora_B.weight.grad is not None):
                        return True
        return False