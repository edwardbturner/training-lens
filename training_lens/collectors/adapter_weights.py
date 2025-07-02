"""Collector for adapter/LoRA weight matrices."""

from typing import Any, Dict, List, Optional

import torch

from ..core.base import DataCollector, DataType


class AdapterWeightsCollector(DataCollector):
    """Collects LoRA adapter weight matrices (A and B matrices)."""
    
    @property
    def data_type(self) -> DataType:
        return DataType.ADAPTER_WEIGHTS
    
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
        """Collect LoRA adapter weights.
        
        Args:
            model: Model with LoRA adapters
            step: Current training step
            **kwargs: Additional context
            
        Returns:
            Dictionary containing adapter weights
        """
        if not self.can_collect(model, step):
            return None
        
        adapter_weights = {}
        adapter_name = self.config.get("adapter_name", "default")
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (isinstance(module.lora_A, dict) and adapter_name in module.lora_A and
                    isinstance(module.lora_B, dict) and adapter_name in module.lora_B):
                    
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                    
                    # Extract weight data
                    weights_data = {
                        'A_weight': lora_A.weight.data.clone().cpu(),
                        'B_weight': lora_B.weight.data.clone().cpu(),
                        'shape_A': list(lora_A.weight.shape),
                        'shape_B': list(lora_B.weight.shape),
                        'dtype': str(lora_A.weight.dtype),
                        'rank': lora_A.weight.shape[0],
                    }
                    
                    # Add scaling if available
                    if hasattr(module, 'scaling') and adapter_name in module.scaling:
                        weights_data['scaling'] = float(module.scaling[adapter_name])
                    
                    # Compute effective weight matrix (B @ A)
                    effective_weight = lora_B.weight.data @ lora_A.weight.data
                    weights_data['effective_weight'] = effective_weight.cpu()
                    
                    # Compute statistics
                    weights_data['statistics'] = {
                        'A_norm': torch.norm(lora_A.weight.data).item(),
                        'B_norm': torch.norm(lora_B.weight.data).item(),
                        'effective_norm': torch.norm(effective_weight).item(),
                        'A_mean': lora_A.weight.data.mean().item(),
                        'B_mean': lora_B.weight.data.mean().item(),
                        'A_std': lora_A.weight.data.std().item(),
                        'B_std': lora_B.weight.data.std().item(),
                    }
                    
                    adapter_weights[name] = weights_data
        
        if adapter_weights:
            return {
                'step': step,
                'adapter_name': adapter_name,
                'adapter_weights': adapter_weights,
                'total_adapters': len(adapter_weights),
                'collection_timestamp': torch.tensor(step, dtype=torch.float32),
            }
        
        return None
    
    def _has_lora_adapters(self, model: torch.nn.Module) -> bool:
        """Check if model has LoRA adapters."""
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                return True
        return False