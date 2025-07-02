"""Activation analysis for tracking activations across training checkpoints."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..integrations.huggingface_integration import HuggingFaceIntegration
from ..utils.helpers import ensure_dir
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ActivationExtractor:
    """Extracts activations at specified points in a model."""

    def __init__(self, model: PreTrainedModel):
        """Initialize activation extractor.

        Args:
            model: The model to extract activations from
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.activation_points: Dict[str, str] = {}

    def register_activation_point(self, name: str, module_path: str) -> None:
        """Register an activation extraction point.

        Args:
            name: Name for this activation point
            module_path: Dot-separated path to the module (e.g., "model.layers.0.mlp")
        """
        try:
            module = self._get_module_by_path(module_path)

            def hook_fn(module, input, output):
                # Store activation (detach to avoid keeping gradients)
                if isinstance(output, tuple):
                    # Take first element if tuple (common in transformer layers)
                    activation = output[0].detach().clone()
                else:
                    activation = output.detach().clone()
                self.activations[name] = activation

            handle = module.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            self.activation_points[name] = module_path
            logger.debug(f"Registered activation point '{name}' at {module_path}")

        except AttributeError as e:
            logger.error(f"Failed to register activation point '{name}' at {module_path}: {e}")
            raise

    def register_lora_activation_points(self, adapter_name: str = "default") -> None:
        """Register activation points for LoRA adapters.

        Args:
            adapter_name: Name of the LoRA adapter
        """
        try:
            # Find all LoRA modules
            lora_modules = self._find_lora_modules(adapter_name)

            for module_name, module in lora_modules.items():
                # Register pre-A matrix (input to LoRA)
                self.register_activation_point(f"lora_{module_name}_pre_A", module_name)

                # Register post-A pre-B (between A and B matrices)
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Hook the A matrix output
                    def make_hook_fn(name_prefix):
                        def hook_fn(module, input, output):
                            self.activations[f"{name_prefix}_post_A_pre_B"] = output.detach().clone()

                        return hook_fn

                    a_handle = module.lora_A[adapter_name].register_forward_hook(make_hook_fn(f"lora_{module_name}"))
                    self.hooks.append(a_handle)

                    logger.debug(f"Registered LoRA activation points for {module_name}")

        except Exception as e:
            logger.error(f"Failed to register LoRA activation points: {e}")
            raise

    def register_standard_transformer_points(self, layer_indices: Optional[List[int]] = None) -> None:
        """Register standard transformer activation points.

        Args:
            layer_indices: Specific layer indices to track (None for all layers)
        """
        # Detect model architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama-style architecture
            layers = self.model.model.layers
            base_path = "model.layers"
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-style architecture
            layers = self.model.transformer.h
            base_path = "transformer.h"
        else:
            logger.warning("Unknown model architecture, cannot register standard points")
            return

        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        for layer_idx in layer_indices:
            if layer_idx >= len(layers):
                logger.warning(f"Layer index {layer_idx} out of range (max: {len(layers)-1})")
                continue

            layer_path = f"{base_path}.{layer_idx}"

            # Register standard points for this layer
            points = {
                f"layer_{layer_idx}_input": layer_path,
                f"layer_{layer_idx}_post_attention": f"{layer_path}.self_attn",
                f"layer_{layer_idx}_post_mlp": f"{layer_path}.mlp",
                f"layer_{layer_idx}_residual": layer_path,  # Final layer output
            }

            for point_name, point_path in points.items():
                try:
                    self.register_activation_point(point_name, point_path)
                except Exception as e:
                    logger.warning(f"Failed to register {point_name}: {e}")

    def extract_activations(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations for given input.

        Args:
            input_data: Input tensor to process

        Returns:
            Dictionary mapping activation point names to activation tensors
        """
        self.activations.clear()

        # Run forward pass to trigger hooks
        with torch.no_grad():
            self.model(input_data)

        # Return copy of activations
        return {name: activation.clone() for name, activation in self.activations.items()}

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.activations.clear()

    def _get_module_by_path(self, module_path: str) -> nn.Module:
        """Get module by dot-separated path."""
        module = self.model
        for part in module_path.split("."):
            module = getattr(module, part)
        return module

    def _find_lora_modules(self, adapter_name: str) -> Dict[str, nn.Module]:
        """Find all LoRA modules in the model."""
        lora_modules = {}

        def find_modules(module, prefix=""):
            for name, child in module.named_children():
                current_path = f"{prefix}.{name}" if prefix else name

                # Check if this is a LoRA module
                if hasattr(child, "lora_A") and hasattr(child, "lora_B"):
                    if adapter_name in child.lora_A:
                        lora_modules[current_path] = child

                # Recurse into children
                find_modules(child, current_path)

        find_modules(self.model)
        return lora_modules


class ActivationAnalyzer:
    """Analyzes activations across training checkpoints."""

    def __init__(
        self, checkpoint_dir: Union[str, Path], model_name: str, hf_repo_id: Optional[str] = None, device: str = "auto"
    ):
        """Initialize activation analyzer.

        Args:
            checkpoint_dir: Directory containing checkpoints
            model_name: Base model name for loading checkpoints
            hf_repo_id: Optional HuggingFace repo for checkpoint storage
            device: Device to run analysis on
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)

        self.hf_integration = None
        if hf_repo_id:
            self.hf_integration = HuggingFaceIntegration(hf_repo_id)

        self.activation_data: Dict[int, Dict[str, torch.Tensor]] = {}

    def analyze_activation_evolution(
        self,
        input_data: Union[torch.Tensor, List[torch.Tensor]],
        activation_points: Dict[str, str],
        checkpoint_steps: Optional[List[int]] = None,
        lora_analysis: bool = False,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Analyze how activations evolve across training checkpoints.

        Args:
            input_data: Input data to analyze (single tensor or list of tensors)
            activation_points: Custom activation points to track
            checkpoint_steps: Specific checkpoints to analyze (None for all)
            lora_analysis: Whether to include LoRA-specific analysis
            layer_indices: Specific transformer layers to analyze

        Returns:
            Comprehensive analysis of activation evolution
        """
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Discover available checkpoints
        available_checkpoints = self._discover_checkpoints()
        if checkpoint_steps is None:
            checkpoint_steps = available_checkpoints
        else:
            checkpoint_steps = [step for step in checkpoint_steps if step in available_checkpoints]

        if not checkpoint_steps:
            raise ValueError("No valid checkpoints found")

        logger.info(f"Analyzing activation evolution across {len(checkpoint_steps)} checkpoints")

        # Extract activations for each checkpoint
        all_activations = {}

        for step in checkpoint_steps:
            logger.info(f"Processing checkpoint {step}")

            # Load model at this checkpoint
            model = self._load_checkpoint_model(step)
            extractor = ActivationExtractor(model)

            try:
                # Register activation points
                for name, path in activation_points.items():
                    extractor.register_activation_point(name, path)

                # Register standard transformer points if requested
                if layer_indices is not None:
                    extractor.register_standard_transformer_points(layer_indices)

                # Register LoRA points if requested
                if lora_analysis:
                    extractor.register_lora_activation_points()

                # Extract activations for all input samples
                step_activations = {}
                for i, input_tensor in enumerate(input_data):
                    input_tensor = input_tensor.to(self.device)
                    sample_activations = extractor.extract_activations(input_tensor)

                    # Store activations for this sample
                    for act_name, act_tensor in sample_activations.items():
                        if act_name not in step_activations:
                            step_activations[act_name] = []
                        step_activations[act_name].append(act_tensor.cpu())

                # Average across samples for each activation point
                averaged_activations = {}
                for act_name, act_list in step_activations.items():
                    if act_list:
                        averaged_activations[act_name] = torch.stack(act_list).mean(dim=0)

                all_activations[step] = averaged_activations

            finally:
                extractor.cleanup()
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Analyze evolution patterns
        analysis = self._analyze_evolution_patterns(all_activations)

        # Store raw activation data for later use
        self.activation_data = all_activations

        return analysis

    def compute_activation_similarities(
        self, reference_step: Optional[int] = None, similarity_metric: str = "cosine"
    ) -> Dict[str, Any]:
        """Compute similarities between activations across checkpoints.

        Args:
            reference_step: Reference checkpoint (None for first checkpoint)
            similarity_metric: Similarity metric to use ('cosine', 'l2', 'kl_div')

        Returns:
            Similarity analysis results
        """
        if not self.activation_data:
            raise ValueError("No activation data available. Run analyze_activation_evolution first.")

        steps = sorted(self.activation_data.keys())
        if reference_step is None:
            reference_step = steps[0]

        if reference_step not in self.activation_data:
            raise ValueError(f"Reference step {reference_step} not found in activation data")

        reference_activations = self.activation_data[reference_step]
        similarities = {}

        for step in steps:
            step_activations = self.activation_data[step]
            step_similarities = {}

            for act_name in reference_activations.keys():
                if act_name in step_activations:
                    ref_act = reference_activations[act_name].flatten()
                    step_act = step_activations[act_name].flatten()

                    # Compute similarity based on metric
                    if similarity_metric == "cosine":
                        similarity = torch.nn.functional.cosine_similarity(
                            ref_act.unsqueeze(0), step_act.unsqueeze(0)
                        ).item()
                    elif similarity_metric == "l2":
                        similarity = -torch.norm(ref_act - step_act).item()  # Negative for similarity
                    elif similarity_metric == "kl_div":
                        # Softmax to convert to probabilities
                        ref_prob = torch.softmax(ref_act, dim=0)
                        step_prob = torch.softmax(step_act, dim=0)
                        similarity = -torch.nn.functional.kl_div(step_prob.log(), ref_prob, reduction="sum").item()
                    else:
                        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

                    step_similarities[act_name] = similarity

            similarities[step] = step_similarities

        return {
            "reference_step": reference_step,
            "similarity_metric": similarity_metric,
            "similarities": similarities,
            "summary": self._summarize_similarities(similarities),
        }

    def export_activations(
        self, output_dir: Union[str, Path], format: str = "npz", upload_to_hf: bool = False
    ) -> Dict[str, Path]:
        """Export activation data for external analysis.

        Args:
            output_dir: Directory to save activation data
            format: Export format ('npz', 'pt', 'json')
            upload_to_hf: Whether to upload to HuggingFace Hub

        Returns:
            Dictionary mapping checkpoint steps to file paths
        """
        if not self.activation_data:
            raise ValueError("No activation data to export")

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        exported_files = {}

        for step, activations in self.activation_data.items():
            if format == "npz":
                # Convert to numpy and save as npz
                numpy_activations = {name: tensor.numpy() for name, tensor in activations.items()}
                file_path = output_dir / f"activations_step_{step}.npz"
                np.savez_compressed(file_path, **numpy_activations)

            elif format == "pt":
                # Save as PyTorch tensors
                file_path = output_dir / f"activations_step_{step}.pt"
                torch.save(activations, file_path)

            elif format == "json":
                # Convert to JSON (limited precision)
                json_activations = {name: tensor.tolist() for name, tensor in activations.items()}
                file_path = output_dir / f"activations_step_{step}.json"
                with open(file_path, "w") as f:
                    json.dump(json_activations, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            exported_files[step] = file_path

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "activation_points": list(next(iter(self.activation_data.values())).keys()),
            "checkpoint_steps": list(self.activation_data.keys()),
            "export_format": format,
        }

        metadata_path = output_dir / "activation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload to HuggingFace if requested
        if upload_to_hf and self.hf_integration:
            try:
                self._upload_activations_to_hf(output_dir)
                logger.info("Successfully uploaded activation data to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to upload to HuggingFace Hub: {e}")

        logger.info(f"Exported activation data for {len(exported_files)} checkpoints to {output_dir}")
        return exported_files

    def _discover_checkpoints(self) -> List[int]:
        """Discover available checkpoint steps."""
        checkpoints = []

        if self.checkpoint_dir.exists():
            for item in self.checkpoint_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.split("-")[1])
                        checkpoints.append(step)
                    except (ValueError, IndexError):
                        continue

        return sorted(checkpoints)

    def _load_checkpoint_model(self, step: int) -> PreTrainedModel:
        """Load model from checkpoint."""
        from transformers import AutoModelForCausalLM

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"

        if not checkpoint_path.exists() and self.hf_integration:
            # Try to download from HuggingFace
            logger.info(f"Downloading checkpoint {step} from HuggingFace")
            checkpoint_path = self.hf_integration.download_checkpoint(step, self.checkpoint_dir.parent)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {step} not found locally or on HuggingFace")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16, device_map={"": self.device}
        )

        return model

    def _analyze_evolution_patterns(self, all_activations: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Analyze patterns in activation evolution."""
        if not all_activations:
            return {"status": "no_data"}

        steps = sorted(all_activations.keys())
        activation_names = list(next(iter(all_activations.values())).keys())

        analysis = {
            "steps_analyzed": steps,
            "activation_points": activation_names,
            "evolution_patterns": {},
            "magnitude_changes": {},
            "variance_analysis": {},
        }

        for act_name in activation_names:
            # Collect activations across all steps
            activations_over_time = []
            for step in steps:
                if act_name in all_activations[step]:
                    activations_over_time.append(all_activations[step][act_name])

            if not activations_over_time:
                continue

            # Stack tensors (steps x ...)
            stacked_activations = torch.stack(activations_over_time)

            # Analyze magnitude changes
            magnitudes = torch.norm(stacked_activations.flatten(1), dim=1)
            analysis["magnitude_changes"][act_name] = {
                "initial_magnitude": magnitudes[0].item(),
                "final_magnitude": magnitudes[-1].item(),
                "magnitude_change": (magnitudes[-1] - magnitudes[0]).item(),
                "magnitude_trend": self._compute_trend(magnitudes.numpy()),
            }

            # Analyze variance patterns
            variances = torch.var(stacked_activations.flatten(1), dim=1)
            analysis["variance_analysis"][act_name] = {
                "initial_variance": variances[0].item(),
                "final_variance": variances[-1].item(),
                "variance_trend": self._compute_trend(variances.numpy()),
            }

            # Analyze evolution patterns (consecutive similarities)
            similarities = []
            for i in range(1, len(activations_over_time)):
                prev_act = activations_over_time[i - 1].flatten()
                curr_act = activations_over_time[i].flatten()
                similarity = torch.nn.functional.cosine_similarity(prev_act.unsqueeze(0), curr_act.unsqueeze(0)).item()
                similarities.append(similarity)

            analysis["evolution_patterns"][act_name] = {
                "mean_similarity": np.mean(similarities) if similarities else 0.0,
                "similarity_trend": self._compute_trend(np.array(similarities)) if similarities else "stable",
                "stability": (
                    "high" if np.mean(similarities) > 0.9 else "medium" if np.mean(similarities) > 0.7 else "low"
                ),
            }

        return analysis

    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend direction from values."""
        if len(values) < 2:
            return "stable"

        # Linear regression to find trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _summarize_similarities(self, similarities: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Summarize similarity patterns across all activation points."""
        all_similarities = []
        activation_summaries = {}

        # Collect all similarities
        for step, step_sims in similarities.items():
            for act_name, sim_value in step_sims.items():
                all_similarities.append(sim_value)

                if act_name not in activation_summaries:
                    activation_summaries[act_name] = []
                activation_summaries[act_name].append(sim_value)

        # Compute overall statistics
        summary = {
            "overall_mean_similarity": np.mean(all_similarities) if all_similarities else 0.0,
            "overall_std_similarity": np.std(all_similarities) if all_similarities else 0.0,
            "activation_point_summaries": {},
        }

        # Per-activation summaries
        for act_name, sims in activation_summaries.items():
            summary["activation_point_summaries"][act_name] = {
                "mean_similarity": np.mean(sims),
                "std_similarity": np.std(sims),
                "min_similarity": np.min(sims),
                "max_similarity": np.max(sims),
                "trend": self._compute_trend(np.array(sims)),
            }

        return summary

    def _upload_activations_to_hf(self, output_dir: Path) -> None:
        """Upload activation data to HuggingFace Hub."""
        if not self.hf_integration:
            raise ValueError("HuggingFace integration not configured")

        # Upload all files in the output directory
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                relative_path = f"activations/{file_path.name}"
                self.hf_integration.upload_file(file_path, relative_path, f"Upload activation data: {file_path.name}")
