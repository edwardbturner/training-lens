"""CLI-compatible activation analyzer for checkpoint analysis.

This module provides a CLI-compatible activation analyzer that can work with
checkpoint directories and model names directly, bridging the gap between
the framework-based DataAnalyzer and the CLI expectations.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ...utils.logging import get_logger

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = get_logger(__name__)


class CLIActivationAnalyzer:
    """CLI-compatible activation analyzer for checkpoint analysis.

    This analyzer provides the interface expected by the CLI while being compatible
    with the training-lens framework.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_name: str,
        hf_repo_id: Optional[str] = None,
    ):
        """Initialize the CLI activation analyzer.

        Args:
            checkpoint_dir: Directory containing checkpoints
            model_name: Name/ID of the model to analyze
            hf_repo_id: Optional HuggingFace repository ID
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.activation_data = {}

        if not TRANSFORMERS_AVAILABLE:
            warnings.warn("Transformers not available. Some functionality may be limited.")

        logger.info(f"Initialized CLIActivationAnalyzer for {model_name}")

    def analyze_activation_evolution(
        self,
        input_data: List[torch.Tensor],
        activation_points: Optional[Dict[str, str]] = None,
        checkpoint_steps: Optional[List[int]] = None,
        lora_analysis: bool = False,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Analyze activation evolution across checkpoints.

        Args:
            input_data: List of input tensors to analyze
            activation_points: Custom activation extraction points
            checkpoint_steps: Specific checkpoint steps to analyze
            lora_analysis: Whether to include LoRA-specific analysis
            layer_indices: Specific layer indices to analyze

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting activation evolution analysis...")

        # Find available checkpoints
        available_checkpoints = self._find_checkpoints(checkpoint_steps)

        if not available_checkpoints:
            return {
                "status": "no_checkpoints",
                "message": "No valid checkpoints found"
            }

        # Extract activations from each checkpoint
        for checkpoint_step in available_checkpoints:
            logger.info(f"Processing checkpoint {checkpoint_step}")

            checkpoint_activations = self._extract_checkpoint_activations(
                checkpoint_step=checkpoint_step,
                input_data=input_data,
                activation_points=activation_points,
                layer_indices=layer_indices,
                lora_analysis=lora_analysis,
            )

            if checkpoint_activations:
                self.activation_data[checkpoint_step] = checkpoint_activations

        if not self.activation_data:
            return {
                "status": "no_activations",
                "message": "No activations could be extracted"
            }

        # Analyze evolution patterns
        evolution_analysis = self._analyze_evolution_patterns()

        return {
            "status": "success",
            "checkpoints_analyzed": list(self.activation_data.keys()),
            "activation_points": list(next(iter(self.activation_data.values())).keys()),
            "evolution_patterns": evolution_analysis,
            "summary": {
                "total_checkpoints": len(self.activation_data),
                "total_activation_points": len(next(iter(self.activation_data.values()))),
            }
        }

    def compute_activation_similarities(
        self,
        reference_step: Optional[int] = None,
        similarity_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """Compute similarities between activations across checkpoints.

        Args:
            reference_step: Reference checkpoint step (uses first if None)
            similarity_metric: Similarity metric to use

        Returns:
            Dictionary containing similarity analysis
        """
        if not self.activation_data:
            return {"status": "no_data"}

        steps = sorted(self.activation_data.keys())

        if reference_step is None:
            reference_step = steps[0]

        if reference_step not in self.activation_data:
            logger.warning(f"Reference step {reference_step} not found, using {steps[0]}")
            reference_step = steps[0]

        reference_activations = self.activation_data[reference_step]
        similarities = {}

        for step in steps:
            step_similarities = {}
            step_activations = self.activation_data[step]

            for act_name in reference_activations.keys():
                if act_name in step_activations:
                    similarity = self._compute_similarity(
                        reference_activations[act_name],
                        step_activations[act_name],
                        metric=similarity_metric
                    )
                    step_similarities[act_name] = similarity

            similarities[step] = step_similarities

        # Compute summary statistics
        all_similarities = []
        for step_sims in similarities.values():
            all_similarities.extend(step_sims.values())

        summary = {
            "overall_mean_similarity": np.mean(all_similarities) if all_similarities else 0.0,
            "overall_std_similarity": np.std(all_similarities) if all_similarities else 0.0,
            "reference_step": reference_step,
            "metric_used": similarity_metric,
        }

        return {
            "similarities": similarities,
            "summary": summary,
            "reference_step": reference_step,
        }

    def export_activations(
        self,
        output_dir: Union[str, Path],
        format: str = "npz",
        upload_to_hf: bool = False,
    ) -> Dict[int, str]:
        """Export activation data to files.

        Args:
            output_dir: Directory to save exported data
            format: Export format (npz, pt, json)
            upload_to_hf: Whether to upload to HuggingFace Hub

        Returns:
            Dictionary mapping checkpoint steps to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        for step, activations in self.activation_data.items():
            if format == "npz":
                # Convert tensors to numpy arrays
                numpy_data = {}
                for name, tensor in activations.items():
                    if isinstance(tensor, torch.Tensor):
                        numpy_data[name] = tensor.cpu().numpy()
                    else:
                        numpy_data[name] = np.array(tensor)

                file_path = output_dir / f"activations_step_{step}.npz"
                np.savez_compressed(file_path, **numpy_data)

            elif format == "pt":
                file_path = output_dir / f"activations_step_{step}.pt"
                torch.save(activations, file_path)

            elif format == "json":
                # Convert to JSON-serializable format
                json_data = {}
                for name, tensor in activations.items():
                    if isinstance(tensor, torch.Tensor):
                        json_data[name] = tensor.cpu().numpy().tolist()
                    else:
                        json_data[name] = tensor.tolist() if hasattr(tensor, 'tolist') else tensor

                file_path = output_dir / f"activations_step_{step}.json"
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)

            exported_files[step] = str(file_path)
            logger.info(f"Exported step {step} to {file_path}")

        if upload_to_hf and self.hf_repo_id:
            logger.info(f"Uploading to HuggingFace Hub: {self.hf_repo_id}")
            # TODO: Implement HuggingFace upload
            logger.warning("HuggingFace upload not yet implemented")

        return exported_files

    def _find_checkpoints(self, checkpoint_steps: Optional[List[int]] = None) -> List[int]:
        """Find available checkpoint steps in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            logger.error(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return []

        available_steps = []

        # Look for checkpoint directories (e.g., checkpoint-100, checkpoint-500)
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                try:
                    step = int(item.name.split('-')[1])
                    if checkpoint_steps is None or step in checkpoint_steps:
                        available_steps.append(step)
                except (ValueError, IndexError):
                    continue

        # If no specific steps requested and no checkpoint dirs found, use step numbers
        if not available_steps and checkpoint_steps:
            # Check if the checkpoint files exist directly
            for step in checkpoint_steps:
                checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
                if checkpoint_path.exists():
                    available_steps.append(step)

        # If still no steps, try to find any .pt or .bin files
        if not available_steps:
            for item in self.checkpoint_dir.iterdir():
                if item.suffix in ['.pt', '.bin', '.safetensors']:
                    # Use a default step number
                    available_steps.append(0)
                    break

        return sorted(available_steps)

    def _extract_checkpoint_activations(
        self,
        checkpoint_step: int,
        input_data: List[torch.Tensor],
        activation_points: Optional[Dict[str, str]] = None,
        layer_indices: Optional[List[int]] = None,
        lora_analysis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from a specific checkpoint.

        This is a simplified implementation that generates synthetic activation data
        for demonstration purposes. In a real implementation, this would load the
        actual model checkpoint and extract real activations.
        """
        logger.info(f"Extracting activations from checkpoint {checkpoint_step}")

        # For now, generate synthetic activation data
        # In a real implementation, this would:
        # 1. Load the model from the checkpoint
        # 2. Set up activation hooks
        # 3. Run the input data through the model
        # 4. Collect the activations

        activations = {}

        # Generate synthetic activations based on the parameters
        num_layers = 12 if layer_indices is None else len(layer_indices)
        batch_size = len(input_data)
        hidden_size = 768  # Standard transformer hidden size

        for i in range(num_layers):
            layer_idx = i if layer_indices is None else layer_indices[i]

            # Generate different activation patterns for different checkpoints
            # to simulate training evolution
            base_activations = torch.randn(batch_size, 64, hidden_size)

            # Add checkpoint-specific variations
            checkpoint_factor = 1.0 + (checkpoint_step * 0.01)
            base_activations *= checkpoint_factor

            activations[f"layer_{layer_idx}_output"] = base_activations

            if lora_analysis:
                # Add LoRA-specific activations
                lora_activations = torch.randn(batch_size, 64, 64)  # Lower rank
                activations[f"layer_{layer_idx}_lora"] = lora_activations

        # Add custom activation points if specified
        if activation_points:
            for name, _ in activation_points.items():
                custom_activations = torch.randn(batch_size, 32, hidden_size)
                activations[f"custom_{name}"] = custom_activations

        logger.info(f"Extracted {len(activations)} activation tensors")
        return activations

    def _analyze_evolution_patterns(self) -> Dict[str, Any]:
        """Analyze how activations evolve across checkpoints."""
        if len(self.activation_data) < 2:
            return {"status": "insufficient_data"}

        steps = sorted(self.activation_data.keys())
        patterns = {}

        # Analyze each activation point
        activation_names = list(next(iter(self.activation_data.values())).keys())

        for act_name in activation_names:
            # Compute magnitude evolution
            magnitudes = []
            for step in steps:
                if act_name in self.activation_data[step]:
                    tensor = self.activation_data[step][act_name]
                    magnitude = torch.norm(tensor).item()
                    magnitudes.append(magnitude)

            if len(magnitudes) >= 2:
                # Compute trend
                trend = "increasing" if magnitudes[-1] > magnitudes[0] else "decreasing"

                # Compute stability (inverse of coefficient of variation)
                mean_mag = np.mean(magnitudes)
                std_mag = np.std(magnitudes)
                stability = 1.0 / (std_mag / mean_mag + 1e-8)

                patterns[act_name] = {
                    "magnitudes": magnitudes,
                    "trend": trend,
                    "stability": min(stability, 1.0),  # Cap at 1.0
                    "magnitude_change": (magnitudes[-1] - magnitudes[0]) / magnitudes[0]
                }

        return patterns

    def _compute_similarity(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between two activation tensors."""
        # Flatten tensors for similarity computation
        flat1 = tensor1.view(-1).float()
        flat2 = tensor2.view(-1).float()

        if metric == "cosine":
            similarity = torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0), flat2.unsqueeze(0)
            ).item()
        elif metric == "l2":
            # Convert L2 distance to similarity (higher is more similar)
            distance = torch.norm(flat1 - flat2).item()
            similarity = 1.0 / (1.0 + distance)
        elif metric == "kl_div":
            # Simplified KL divergence (requires normalization)
            prob1 = torch.softmax(flat1, dim=0)
            prob2 = torch.softmax(flat2, dim=0)
            kl_div = torch.nn.functional.kl_div(
                prob1.log(), prob2, reduction='sum'
            ).item()
            similarity = 1.0 / (1.0 + kl_div)
        else:
            logger.warning(f"Unknown similarity metric: {metric}, using cosine")
            similarity = torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0), flat2.unsqueeze(0)
            ).item()

        return similarity
