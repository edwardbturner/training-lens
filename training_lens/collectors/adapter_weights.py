"""Collector for adapter/LoRA weight matrices."""

from typing import Any, Dict, List, Optional

import torch

from ..core.base import DataCollector, DataType
from ..utils.lora_utils import get_lora_components_per_layer, LoRAComponentError


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
            **kwargs: Additional context (may include repo_id for external loading)
            
        Returns:
            Dictionary containing adapter weights
        """
        if not self.can_collect(model, step):
            return None
        
        adapter_weights = {}
        adapter_name = self.config.get("adapter_name", "default")
        
        # Try robust external loading if repo_id provided
        repo_id = kwargs.get("repo_id")
        if repo_id:
            try:
                external_components = self._collect_from_repo(repo_id, **kwargs)
                if external_components:
                    return {
                        'step': step,
                        'adapter_name': adapter_name,
                        'adapter_weights': external_components,
                        'total_adapters': len(external_components),
                        'collection_timestamp': torch.tensor(step, dtype=torch.float32),
                        'source': 'external_repo',
                    }
            except LoRAComponentError as e:
                # Fall back to model inspection if external loading fails
                self.logger.warning(f"External LoRA loading failed, using model inspection: {e}")
        
        # Standard model inspection approach
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Handle PEFT-style dictionary adapters
                if (isinstance(module.lora_A, dict) and adapter_name in module.lora_A and
                    isinstance(module.lora_B, dict) and adapter_name in module.lora_B):
                    
                    lora_A = module.lora_A[adapter_name]
                    lora_B = module.lora_B[adapter_name]
                # Handle simple Linear layer adapters
                elif hasattr(module.lora_A, 'weight') and hasattr(module.lora_B, 'weight'):
                    lora_A = module.lora_A
                    lora_B = module.lora_B
                else:
                    continue
                    
                # Extract weight data using the same format as robust utils
                weights_data = {
                    'lora_A': lora_A.weight.data.clone().cpu(),
                    'lora_B': lora_B.weight.data.clone().cpu(),
                    'shape_A': list(lora_A.weight.shape),
                    'shape_B': list(lora_B.weight.shape),
                    'dtype': str(lora_A.weight.dtype),
                    'rank': lora_A.weight.shape[0],
                }
                
                # Add scaling if available
                if hasattr(module, 'scaling'):
                    if isinstance(module.scaling, dict) and adapter_name in module.scaling:
                        weights_data['scaling'] = float(module.scaling[adapter_name])
                    elif hasattr(module, 'scaling') and not isinstance(module.scaling, dict):
                        weights_data['scaling'] = float(module.scaling)
                
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
                'source': 'model_inspection',
            }
        
        return None
    
    def _collect_from_repo(self, repo_id: str, **kwargs) -> Dict[str, Any]:
        """Collect LoRA components from external repository using robust loading.
        
        Args:
            repo_id: HuggingFace repository ID
            **kwargs: Additional arguments (subfolder, revision, etc.)
            
        Returns:
            Dictionary of LoRA components
        """
        subfolder = kwargs.get("subfolder")
        revision = kwargs.get("revision", "main")
        layer_filter = self.config.get("layer_filter")
        
        components = get_lora_components_per_layer(
            repo_id=repo_id,
            subfolder=subfolder,
            revision=revision,
            layer_filter=layer_filter,
            force_download=kwargs.get("force_download", False),
        )
        
        return components
    
    def _has_lora_adapters(self, model: torch.nn.Module) -> bool:
        """Check if model has LoRA adapters."""
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                return True
        return False