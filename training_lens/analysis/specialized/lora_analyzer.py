"""Specialized LoRA activation and parameter analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...utils.helpers import ensure_dir
from ...utils.logging import get_logger

logger = get_logger(__name__)


class LoRAActivationTracker:
    """Specialized tracker for LoRA adapter activations."""
    
    def __init__(self, model: PreTrainedModel, adapter_name: str = "default"):
        """Initialize LoRA activation tracker.
        
        Args:
            model: Model with LoRA adapters
            adapter_name: Name of the LoRA adapter to track
        """
        self.model = model
        self.adapter_name = adapter_name
        self.lora_modules = self._discover_lora_modules()
        self.activation_hooks: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}
        self.activations: Dict[str, Dict[str, torch.Tensor]] = {}
        
        if not self.lora_modules:
            logger.warning("No LoRA modules found in the model")
        else:
            logger.info(f"Found {len(self.lora_modules)} LoRA modules")
    
    def register_lora_hooks(self, track_gradients: bool = False) -> None:
        """Register hooks to capture LoRA-specific activations.
        
        Args:
            track_gradients: Whether to also track gradients
        """
        for module_name, lora_module in self.lora_modules.items():
            self.activation_hooks[module_name] = []
            self.activations[module_name] = {}
            
            # Hook input to LoRA path (pre-A)
            self._register_pre_lora_hook(module_name, lora_module)
            
            # Hook between A and B matrices (post-A, pre-B)
            self._register_mid_lora_hook(module_name, lora_module)
            
            # Hook final LoRA output (post-B)
            self._register_post_lora_hook(module_name, lora_module)
            
            # Hook main path (non-LoRA) for comparison
            self._register_main_path_hook(module_name, lora_module)
            
            if track_gradients:
                self._register_gradient_hooks(module_name, lora_module)
    
    def extract_lora_activations(
        self, 
        input_data: torch.Tensor,
        return_individual_components: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract LoRA activations for given input.
        
        Args:
            input_data: Input tensor
            return_individual_components: Whether to return A and B matrix outputs separately
            
        Returns:
            Dictionary with LoRA module activations
        """
        # Clear previous activations
        for module_name in self.lora_modules:
            self.activations[module_name] = {}
        
        # Run forward pass
        with torch.no_grad():
            self.model(input_data)
        
        results = {}
        for module_name in self.lora_modules:
            module_activations = {}
            
            # Copy activations
            for key, value in self.activations[module_name].items():
                module_activations[key] = value.clone() if isinstance(value, torch.Tensor) else value
            
            # Compute additional metrics
            module_activations.update(self._compute_lora_metrics(module_name))
            
            results[module_name] = module_activations
        
        return results
    
    def analyze_lora_contribution(
        self, 
        input_data: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Analyze the contribution of LoRA adapters vs main model.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Analysis of LoRA vs main path contributions
        """
        activations = self.extract_lora_activations(input_data)
        contributions = {}
        
        for module_name, module_acts in activations.items():
            if 'main_path_output' in module_acts and 'lora_output' in module_acts:
                main_output = module_acts['main_path_output']
                lora_output = module_acts['lora_output']
                
                # Compute relative magnitudes
                main_magnitude = torch.norm(main_output).item()
                lora_magnitude = torch.norm(lora_output).item()
                total_magnitude = main_magnitude + lora_magnitude
                
                contributions[module_name] = {
                    'main_path_contribution': main_magnitude / total_magnitude if total_magnitude > 0 else 0.5,
                    'lora_contribution': lora_magnitude / total_magnitude if total_magnitude > 0 else 0.5,
                    'lora_to_main_ratio': lora_magnitude / main_magnitude if main_magnitude > 0 else float('inf'),
                    'main_magnitude': main_magnitude,
                    'lora_magnitude': lora_magnitude
                }
        
        return contributions
    
    def track_lora_evolution_across_checkpoints(
        self,
        checkpoint_paths: List[Path],
        input_data: torch.Tensor,
        model_loader_fn: callable
    ) -> Dict[str, Any]:
        """Track how LoRA activations evolve across training checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            input_data: Input data for analysis
            model_loader_fn: Function to load model from checkpoint path
            
        Returns:
            Evolution analysis across checkpoints
        """
        evolution_data = {}
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
            
            # Load model at this checkpoint
            model = model_loader_fn(checkpoint_path)
            
            # Create new tracker for this checkpoint
            tracker = LoRAActivationTracker(model, self.adapter_name)
            tracker.register_lora_hooks()
            
            try:
                # Extract activations
                activations = tracker.extract_lora_activations(input_data)
                
                # Analyze contributions
                contributions = tracker.analyze_lora_contribution(input_data)
                
                step_data = {
                    'activations': activations,
                    'contributions': contributions,
                    'checkpoint_path': str(checkpoint_path)
                }
                
                evolution_data[i] = step_data
                
            finally:
                tracker.cleanup()
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Analyze evolution patterns
        analysis = self._analyze_evolution_patterns(evolution_data)
        
        return {
            'checkpoint_data': evolution_data,
            'evolution_analysis': analysis
        }
    
    def compute_lora_rank_utilization(self) -> Dict[str, Dict[str, float]]:
        """Compute how well LoRA adapters utilize their rank capacity.
        
        Returns:
            Rank utilization analysis for each LoRA module
        """
        utilization = {}
        
        for module_name, lora_module in self.lora_modules.items():
            if hasattr(lora_module, 'lora_A') and hasattr(lora_module, 'lora_B'):
                lora_A = lora_module.lora_A[self.adapter_name]
                lora_B = lora_module.lora_B[self.adapter_name]
                
                # Get weight matrices
                A_weight = lora_A.weight.data
                B_weight = lora_B.weight.data
                
                # Compute effective rank using SVD
                with torch.no_grad():
                    # For LoRA: output = B @ A @ input
                    # Effective matrix is B @ A
                    effective_matrix = B_weight @ A_weight
                    
                    # SVD to find effective rank
                    U, S, V = torch.svd(effective_matrix)
                    
                    # Compute rank utilization metrics
                    singular_values = S.cpu().numpy()
                    total_rank = len(singular_values)
                    
                    # Effective rank (participation ratio)
                    normalized_sv = singular_values / np.sum(singular_values) if np.sum(singular_values) > 0 else singular_values
                    effective_rank = np.exp(-np.sum(normalized_sv * np.log(normalized_sv + 1e-10)))
                    
                    # Stable rank (ratio of Frobenius to spectral norm)
                    frobenius_norm = torch.norm(effective_matrix, 'fro').item()
                    spectral_norm = torch.norm(effective_matrix, 2).item()
                    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2) if spectral_norm > 0 else 0
                    
                    utilization[module_name] = {
                        'nominal_rank': int(A_weight.shape[0]),  # LoRA rank
                        'effective_rank': float(effective_rank),
                        'stable_rank': float(stable_rank),
                        'rank_utilization': float(effective_rank / A_weight.shape[0]),
                        'singular_values': singular_values.tolist(),
                        'condition_number': float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-10 else float('inf')
                    }
        
        return utilization
    
    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for module_hooks in self.activation_hooks.values():
            for handle in module_hooks:
                handle.remove()
        self.activation_hooks.clear()
        self.activations.clear()
    
    def _discover_lora_modules(self) -> Dict[str, nn.Module]:
        """Discover all LoRA modules in the model."""
        lora_modules = {}
        
        def find_lora_modules(module, prefix=""):
            for name, child in module.named_children():
                current_path = f"{prefix}.{name}" if prefix else name
                
                # Check if this module has LoRA adapters
                if hasattr(child, 'lora_A') and hasattr(child, 'lora_B'):
                    if (isinstance(child.lora_A, dict) and self.adapter_name in child.lora_A and
                        isinstance(child.lora_B, dict) and self.adapter_name in child.lora_B):
                        lora_modules[current_path] = child
                
                # Recurse into children
                find_lora_modules(child, current_path)
        
        find_lora_modules(self.model)
        return lora_modules
    
    def _register_pre_lora_hook(self, module_name: str, lora_module: nn.Module) -> None:
        """Register hook for input to LoRA path."""
        def hook_fn(module, input, output):
            # Store the input that goes into LoRA path
            if isinstance(input, tuple):
                self.activations[module_name]['pre_lora_input'] = input[0].detach().clone()
            else:
                self.activations[module_name]['pre_lora_input'] = input.detach().clone()
        
        handle = lora_module.register_forward_hook(hook_fn)
        self.activation_hooks[module_name].append(handle)
    
    def _register_mid_lora_hook(self, module_name: str, lora_module: nn.Module) -> None:
        """Register hook between A and B matrices."""
        if hasattr(lora_module, 'lora_A') and self.adapter_name in lora_module.lora_A:
            lora_A = lora_module.lora_A[self.adapter_name]
            
            def hook_fn(module, input, output):
                # This is the output of A matrix (input to B matrix)
                self.activations[module_name]['post_A_pre_B'] = output.detach().clone()
            
            handle = lora_A.register_forward_hook(hook_fn)
            self.activation_hooks[module_name].append(handle)
    
    def _register_post_lora_hook(self, module_name: str, lora_module: nn.Module) -> None:
        """Register hook for final LoRA output."""
        if hasattr(lora_module, 'lora_B') and self.adapter_name in lora_module.lora_B:
            lora_B = lora_module.lora_B[self.adapter_name]
            
            def hook_fn(module, input, output):
                # This is the final LoRA output (output of B matrix)
                self.activations[module_name]['lora_output'] = output.detach().clone()
            
            handle = lora_B.register_forward_hook(hook_fn)
            self.activation_hooks[module_name].append(handle)
    
    def _register_main_path_hook(self, module_name: str, lora_module: nn.Module) -> None:
        """Register hook for main (non-LoRA) path output."""
        # This is tricky - we need to capture the base layer output before LoRA is added
        # We'll use a pre-hook on the module to capture its base computation
        
        def pre_hook_fn(module, input):
            # Store input for later comparison
            if isinstance(input, tuple):
                self.activations[module_name]['main_path_input'] = input[0].detach().clone()
            else:
                self.activations[module_name]['main_path_input'] = input.detach().clone()
        
        def post_hook_fn(module, input, output):
            # The output here includes both main path and LoRA
            # We need to subtract LoRA contribution if possible
            if 'lora_output' in self.activations[module_name]:
                # Estimate main path output by subtracting LoRA
                lora_out = self.activations[module_name]['lora_output']
                if hasattr(lora_module, 'scaling') and self.adapter_name in lora_module.scaling:
                    scaling = lora_module.scaling[self.adapter_name]
                    scaled_lora = lora_out * scaling
                else:
                    scaled_lora = lora_out
                
                # Main path output (approximation)
                if isinstance(output, tuple):
                    main_output = output[0] - scaled_lora
                else:
                    main_output = output - scaled_lora
                
                self.activations[module_name]['main_path_output'] = main_output.detach().clone()
        
        pre_handle = lora_module.register_forward_pre_hook(pre_hook_fn)
        post_handle = lora_module.register_forward_hook(post_hook_fn)
        
        self.activation_hooks[module_name].extend([pre_handle, post_handle])
    
    def _register_gradient_hooks(self, module_name: str, lora_module: nn.Module) -> None:
        """Register hooks to capture gradients (if training)."""
        if hasattr(lora_module, 'lora_A') and self.adapter_name in lora_module.lora_A:
            lora_A = lora_module.lora_A[self.adapter_name]
            
            def grad_hook_A(grad):
                self.activations[module_name]['lora_A_grad'] = grad.detach().clone()
                return grad
            
            if lora_A.weight.requires_grad:
                lora_A.weight.register_hook(grad_hook_A)
        
        if hasattr(lora_module, 'lora_B') and self.adapter_name in lora_module.lora_B:
            lora_B = lora_module.lora_B[self.adapter_name]
            
            def grad_hook_B(grad):
                self.activations[module_name]['lora_B_grad'] = grad.detach().clone()
                return grad
            
            if lora_B.weight.requires_grad:
                lora_B.weight.register_hook(grad_hook_B)
    
    def _compute_lora_metrics(self, module_name: str) -> Dict[str, Any]:
        """Compute additional metrics for LoRA activations."""
        metrics = {}
        module_acts = self.activations[module_name]
        
        # Compute activation magnitudes
        for key, tensor in module_acts.items():
            if isinstance(tensor, torch.Tensor):
                metrics[f"{key}_magnitude"] = torch.norm(tensor).item()
                metrics[f"{key}_mean"] = tensor.mean().item()
                metrics[f"{key}_std"] = tensor.std().item()
        
        # Compute ratios if both paths available
        if 'lora_output' in module_acts and 'main_path_output' in module_acts:
            lora_mag = torch.norm(module_acts['lora_output']).item()
            main_mag = torch.norm(module_acts['main_path_output']).item()
            
            metrics['lora_to_main_magnitude_ratio'] = lora_mag / main_mag if main_mag > 0 else float('inf')
            metrics['total_magnitude'] = lora_mag + main_mag
        
        # Compute A-to-B transformation metrics
        if 'pre_lora_input' in module_acts and 'post_A_pre_B' in module_acts:
            input_tensor = module_acts['pre_lora_input']
            mid_tensor = module_acts['post_A_pre_B']
            
            # Dimensionality change
            metrics['lora_A_dimension_reduction'] = input_tensor.shape[-1] / mid_tensor.shape[-1]
            
            # Information preservation (correlation)
            if input_tensor.numel() > 0 and mid_tensor.numel() > 0:
                # Flatten and compute correlation
                input_flat = input_tensor.flatten()
                mid_flat = mid_tensor.flatten()
                
                # Pad shorter tensor with zeros for correlation computation
                max_len = max(len(input_flat), len(mid_flat))
                if len(input_flat) < max_len:
                    input_flat = torch.cat([input_flat, torch.zeros(max_len - len(input_flat))])
                if len(mid_flat) < max_len:
                    mid_flat = torch.cat([mid_flat, torch.zeros(max_len - len(mid_flat))])
                
                correlation = torch.corrcoef(torch.stack([input_flat, mid_flat]))[0, 1]
                metrics['lora_A_information_preservation'] = correlation.item() if not torch.isnan(correlation) else 0.0
        
        return metrics
    
    def _analyze_evolution_patterns(self, evolution_data: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze how LoRA patterns evolve across checkpoints."""
        if not evolution_data:
            return {"status": "no_data"}
        
        analysis = {
            "contribution_evolution": {},
            "activation_magnitude_evolution": {},
            "rank_utilization_evolution": {}
        }
        
        # Extract evolution patterns for each LoRA module
        for module_name in self.lora_modules.keys():
            contribution_over_time = []
            lora_magnitude_over_time = []
            
            for step_data in evolution_data.values():
                contributions = step_data.get('contributions', {})
                if module_name in contributions:
                    contrib_data = contributions[module_name]
                    contribution_over_time.append(contrib_data['lora_contribution'])
                    lora_magnitude_over_time.append(contrib_data['lora_magnitude'])
            
            if contribution_over_time:
                analysis["contribution_evolution"][module_name] = {
                    "initial_contribution": contribution_over_time[0],
                    "final_contribution": contribution_over_time[-1],
                    "contribution_trend": self._compute_trend(np.array(contribution_over_time)),
                    "mean_contribution": np.mean(contribution_over_time),
                    "contribution_stability": np.std(contribution_over_time)
                }
            
            if lora_magnitude_over_time:
                analysis["activation_magnitude_evolution"][module_name] = {
                    "initial_magnitude": lora_magnitude_over_time[0],
                    "final_magnitude": lora_magnitude_over_time[-1],
                    "magnitude_trend": self._compute_trend(np.array(lora_magnitude_over_time)),
                    "magnitude_growth_rate": (lora_magnitude_over_time[-1] - lora_magnitude_over_time[0]) / len(lora_magnitude_over_time)
                }
        
        return analysis
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return "stable"
        
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


class LoRAParameterAnalyzer:
    """Analyzer for LoRA parameter evolution across checkpoints."""
    
    def __init__(self, adapter_name: str = "default"):
        """Initialize LoRA parameter analyzer.
        
        Args:
            adapter_name: Name of the LoRA adapter to analyze
        """
        self.adapter_name = adapter_name
    
    def analyze_parameter_evolution(
        self,
        checkpoint_paths: List[Path],
        model_loader_fn: callable
    ) -> Dict[str, Any]:
        """Analyze how LoRA parameters evolve during training.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            model_loader_fn: Function to load model from checkpoint
            
        Returns:
            Parameter evolution analysis
        """
        parameter_evolution = {}
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            model = model_loader_fn(checkpoint_path)
            
            # Extract LoRA parameters
            lora_params = self._extract_lora_parameters(model)
            parameter_evolution[i] = {
                'checkpoint_path': str(checkpoint_path),
                'parameters': lora_params
            }
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Analyze evolution patterns
        analysis = self._analyze_parameter_patterns(parameter_evolution)
        
        return {
            'parameter_data': parameter_evolution,
            'evolution_analysis': analysis
        }
    
    def _extract_lora_parameters(self, model: PreTrainedModel) -> Dict[str, Dict[str, Any]]:
        """Extract LoRA parameters from model."""
        lora_params = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if (isinstance(module.lora_A, dict) and self.adapter_name in module.lora_A and
                    isinstance(module.lora_B, dict) and self.adapter_name in module.lora_B):
                    
                    lora_A = module.lora_A[self.adapter_name]
                    lora_B = module.lora_B[self.adapter_name]
                    
                    # Extract parameter statistics
                    A_weight = lora_A.weight.data
                    B_weight = lora_B.weight.data
                    
                    lora_params[name] = {
                        'A_weight_stats': {
                            'mean': A_weight.mean().item(),
                            'std': A_weight.std().item(),
                            'norm': torch.norm(A_weight).item(),
                            'shape': list(A_weight.shape)
                        },
                        'B_weight_stats': {
                            'mean': B_weight.mean().item(),
                            'std': B_weight.std().item(),
                            'norm': torch.norm(B_weight).item(),
                            'shape': list(B_weight.shape)
                        },
                        'combined_stats': {
                            'effective_matrix_norm': torch.norm(B_weight @ A_weight).item(),
                            'rank': A_weight.shape[0]
                        }
                    }
        
        return lora_params
    
    def _analyze_parameter_patterns(self, parameter_evolution: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze parameter evolution patterns."""
        if not parameter_evolution:
            return {"status": "no_data"}
        
        # Get all LoRA module names
        first_checkpoint = next(iter(parameter_evolution.values()))
        module_names = list(first_checkpoint['parameters'].keys())
        
        analysis = {}
        
        for module_name in module_names:
            # Collect parameter evolution for this module
            A_norms = []
            B_norms = []
            effective_norms = []
            
            for step_data in parameter_evolution.values():
                params = step_data['parameters']
                if module_name in params:
                    module_params = params[module_name]
                    A_norms.append(module_params['A_weight_stats']['norm'])
                    B_norms.append(module_params['B_weight_stats']['norm'])
                    effective_norms.append(module_params['combined_stats']['effective_matrix_norm'])
            
            if A_norms and B_norms and effective_norms:
                analysis[module_name] = {
                    'A_matrix_evolution': {
                        'initial_norm': A_norms[0],
                        'final_norm': A_norms[-1],
                        'norm_change': A_norms[-1] - A_norms[0],
                        'trend': self._compute_trend(np.array(A_norms))
                    },
                    'B_matrix_evolution': {
                        'initial_norm': B_norms[0],
                        'final_norm': B_norms[-1],
                        'norm_change': B_norms[-1] - B_norms[0],
                        'trend': self._compute_trend(np.array(B_norms))
                    },
                    'effective_matrix_evolution': {
                        'initial_norm': effective_norms[0],
                        'final_norm': effective_norms[-1],
                        'norm_change': effective_norms[-1] - effective_norms[0],
                        'trend': self._compute_trend(np.array(effective_norms)),
                        'growth_rate': (effective_norms[-1] - effective_norms[0]) / len(effective_norms)
                    }
                }
        
        return analysis
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return "stable"
        
        slope = np.polyfit(np.arange(len(values)), values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"