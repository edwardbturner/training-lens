"""LoRA-specific analysis for adapter weights and activations."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..core.base import DataAnalyzer, DataType
from ...utils.lora_utils import get_lora_components_per_layer, LoRAComponentError


class LoRAAnalyzer(DataAnalyzer):
    """Analyzes LoRA adapter weights and activations."""

    @property
    def data_type(self) -> DataType:
        return DataType.LORA_ANALYSIS

    @property
    def required_data_types(self) -> List[DataType]:
        return [DataType.ADAPTER_WEIGHTS]

    def can_analyze(self, available_data: Dict[DataType, Any]) -> bool:
        """Check if LoRA analysis can be performed."""
        # Can analyze with adapter weights, optionally enhanced with gradients and activations
        return DataType.ADAPTER_WEIGHTS in available_data

    def analyze(
        self,
        data: Dict[DataType, Any],
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform comprehensive LoRA analysis.

        Args:
            data: Dictionary containing collected data
            output_dir: Optional directory to save outputs
            **kwargs: Additional arguments (may include repo_id for external analysis)

        Returns:
            LoRA analysis results
        """
        # Try robust external analysis if repo_id provided
        repo_id = kwargs.get("repo_id")
        if repo_id:
            try:
                external_analysis = self._analyze_from_repo(repo_id, **kwargs)
                if external_analysis:
                    return external_analysis
            except LoRAComponentError as e:
                self.logger.warning(f"External LoRA analysis failed, using collected data: {e}")

        analysis_results = {
            "adapter_weight_analysis": self._analyze_adapter_weights(data),
            "rank_utilization": self._analyze_rank_utilization(data),
            "adapter_evolution": self._analyze_adapter_evolution(data),
        }

        # Add gradient analysis if available
        if DataType.ADAPTER_GRADIENTS in data:
            analysis_results["gradient_analysis"] = self._analyze_adapter_gradients(data)

        # Add activation analysis if available
        if DataType.LORA_ACTIVATIONS in data:
            analysis_results["activation_analysis"] = self._analyze_lora_activations(data)

        # Cross-analysis if multiple data types available
        if len([dt for dt in [DataType.ADAPTER_WEIGHTS, DataType.ADAPTER_GRADIENTS, DataType.LORA_ACTIVATIONS]
                if dt in data]) > 1:
            analysis_results["cross_analysis"] = self._perform_cross_analysis(data)

        # Save detailed results if output directory provided
        if output_dir:
            self._save_detailed_results(analysis_results, output_dir)

        return analysis_results

    def _analyze_adapter_weights(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze LoRA adapter weight evolution."""
        weight_data = data[DataType.ADAPTER_WEIGHTS]

        if not weight_data:
            return {"status": "no_data"}

        steps = sorted(weight_data.keys())
        analysis = {
            "total_steps": len(steps),
            "step_range": [min(steps), max(steps)],
            "adapter_modules": {},
            "global_trends": {},
        }

        # Analyze each adapter module
        all_module_names = set()
        for step_data in weight_data.values():
            if isinstance(step_data, dict) and 'adapter_weights' in step_data:
                all_module_names.update(step_data['adapter_weights'].keys())

        for module_name in all_module_names:
            module_analysis = self._analyze_single_adapter_module(weight_data, module_name)
            analysis["adapter_modules"][module_name] = module_analysis

        # Compute global trends
        analysis["global_trends"] = self._compute_global_weight_trends(weight_data)

        return analysis

    def _analyze_single_adapter_module(
        self,
        weight_data: Dict[int, Any],
        module_name: str
    ) -> Dict[str, Any]:
        """Analyze evolution of a single LoRA adapter module."""
        steps = sorted(weight_data.keys())

        a_norms = []
        b_norms = []
        effective_norms = []
        ranks = []

        for step in steps:
            step_data = weight_data[step]
            if (isinstance(step_data, dict)
                and 'adapter_weights' in step_data
                    and module_name in step_data['adapter_weights']):

                module_data = step_data['adapter_weights'][module_name]
                stats = module_data.get('statistics', {})

                a_norms.append(stats.get('A_norm', 0))
                b_norms.append(stats.get('B_norm', 0))
                effective_norms.append(stats.get('effective_norm', 0))
                ranks.append(module_data.get('rank', 0))

        if not a_norms:
            return {"status": "no_data"}

        return {
            "A_matrix_evolution": {
                "norms": a_norms,
                "trend": self._compute_trend(np.array(a_norms)),
                "stability": np.std(a_norms) if len(a_norms) > 1 else 0,
                "growth_rate": (a_norms[-1] - a_norms[0]) / len(a_norms) if len(a_norms) > 1 else 0,
            },
            "B_matrix_evolution": {
                "norms": b_norms,
                "trend": self._compute_trend(np.array(b_norms)),
                "stability": np.std(b_norms) if len(b_norms) > 1 else 0,
                "growth_rate": (b_norms[-1] - b_norms[0]) / len(b_norms) if len(b_norms) > 1 else 0,
            },
            "effective_matrix_evolution": {
                "norms": effective_norms,
                "trend": self._compute_trend(np.array(effective_norms)),
                "stability": np.std(effective_norms) if len(effective_norms) > 1 else 0,
                "growth_rate": (effective_norms[-1] - effective_norms[0]) / len(effective_norms) if len(effective_norms) > 1 else 0,
            },
            "rank_info": {
                "rank": ranks[0] if ranks else 0,
                "rank_consistency": len(set(ranks)) == 1 if ranks else True,
            },
            "matrix_balance": self._analyze_matrix_balance(a_norms, b_norms),
        }

    def _analyze_matrix_balance(self, a_norms: List[float], b_norms: List[float]) -> Dict[str, Any]:
        """Analyze balance between A and B matrices."""
        if not a_norms or not b_norms:
            return {"status": "no_data"}

        ratios = [a_norm / (b_norm + 1e-8) for a_norm, b_norm in zip(a_norms, b_norms)]

        return {
            "mean_ratio": np.mean(ratios),
            "ratio_stability": np.std(ratios),
            "ratio_trend": self._compute_trend(np.array(ratios)),
            "balance_classification": self._classify_matrix_balance(np.mean(ratios)),
        }

    def _classify_matrix_balance(self, mean_ratio: float) -> str:
        """Classify the balance between A and B matrices."""
        if 0.5 <= mean_ratio <= 2.0:
            return "balanced"
        elif mean_ratio > 2.0:
            return "A_dominant"
        else:
            return "B_dominant"

    def _analyze_rank_utilization(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze how well adapters utilize their rank capacity."""
        weight_data = data[DataType.ADAPTER_WEIGHTS]

        if not weight_data:
            return {"status": "no_data"}

        utilization_analysis = {}

        # Use the latest checkpoint for rank analysis
        latest_step = max(weight_data.keys())
        latest_data = weight_data[latest_step]

        if isinstance(latest_data, dict) and 'adapter_weights' in latest_data:
            for module_name, module_data in latest_data['adapter_weights'].items():
                if 'A_weight' in module_data and 'B_weight' in module_data:
                    utilization = self._compute_rank_utilization(
                        module_data['A_weight'],
                        module_data['B_weight']
                    )
                    utilization_analysis[module_name] = utilization

        # Compute overall statistics
        if utilization_analysis:
            effective_ranks = [u['effective_rank'] for u in utilization_analysis.values()]
            utilization_ratios = [u['utilization_ratio'] for u in utilization_analysis.values()]

            utilization_analysis["global_statistics"] = {
                "mean_effective_rank": np.mean(effective_ranks),
                "mean_utilization_ratio": np.mean(utilization_ratios),
                "utilization_consistency": np.std(utilization_ratios),
            }

        return utilization_analysis

    def _compute_rank_utilization(self, A_weight: torch.Tensor, B_weight: torch.Tensor) -> Dict[str, Any]:
        """Compute rank utilization metrics for A and B matrices."""
        # Effective matrix is B @ A
        effective_matrix = B_weight @ A_weight

        # SVD to compute effective rank
        U, S, V = torch.svd(effective_matrix)
        singular_values = S.cpu().numpy()

        # Compute metrics
        nominal_rank = A_weight.shape[0]

        # Effective rank (participation ratio)
        normalized_sv = singular_values / (np.sum(singular_values) + 1e-8)
        effective_rank = np.exp(-np.sum(normalized_sv * np.log(normalized_sv + 1e-10)))

        # Stable rank
        frobenius_norm = torch.norm(effective_matrix, 'fro').item()
        spectral_norm = torch.norm(effective_matrix, 2).item()
        stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2 + 1e-8)

        return {
            "nominal_rank": nominal_rank,
            "effective_rank": effective_rank,
            "stable_rank": stable_rank,
            "utilization_ratio": effective_rank / nominal_rank,
            "singular_values": singular_values.tolist(),
            "condition_number": singular_values[0] / (singular_values[-1] + 1e-10),
        }

    def _analyze_adapter_evolution(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze overall adapter evolution patterns."""
        weight_data = data[DataType.ADAPTER_WEIGHTS]

        if not weight_data:
            return {"status": "no_data"}

        steps = sorted(weight_data.keys())

        if len(steps) < 2:
            return {"status": "insufficient_data"}

        # Track global metrics across steps
        total_adapters = []
        total_effective_norms = []

        for step in steps:
            step_data = weight_data[step]
            if isinstance(step_data, dict) and 'adapter_weights' in step_data:
                total_adapters.append(step_data.get('total_adapters', 0))

                # Compute total effective norm
                effective_norms = []
                for module_data in step_data['adapter_weights'].values():
                    stats = module_data.get('statistics', {})
                    effective_norms.append(stats.get('effective_norm', 0))

                total_effective_norms.append(sum(effective_norms))

        evolution_analysis = {
            "adapter_count_evolution": {
                "values": total_adapters,
                "trend": self._compute_trend(np.array(total_adapters)),
                "stability": len(set(total_adapters)) == 1 if total_adapters else True,
            },
            "total_norm_evolution": {
                "values": total_effective_norms,
                "trend": self._compute_trend(np.array(total_effective_norms)),
                "growth_rate": ((total_effective_norms[-1] - total_effective_norms[0])
                                / len(total_effective_norms)) if len(total_effective_norms) > 1 else 0,
            },
            "training_phases": self._identify_training_phases(total_effective_norms),
        }

        return evolution_analysis

    def _identify_training_phases(self, total_norms: List[float]) -> Dict[str, Any]:
        """Identify different phases of training based on norm evolution."""
        if len(total_norms) < 3:
            return {"status": "insufficient_data"}

        # Compute growth rates
        growth_rates = []
        for i in range(1, len(total_norms)):
            rate = total_norms[i] - total_norms[i - 1]
            growth_rates.append(rate)

        # Simple phase detection based on growth rate changes
        phases = []
        current_phase = "initialization"
        phase_start = 0

        threshold = np.std(growth_rates) * 0.5 if growth_rates else 0

        for i, rate in enumerate(growth_rates):
            if abs(rate) < threshold and current_phase != "stabilization":
                phases.append({
                    "phase": current_phase,
                    "start_step": phase_start,
                    "end_step": i,
                    "duration": i - phase_start,
                })
                current_phase = "stabilization"
                phase_start = i
            elif abs(rate) >= threshold and current_phase != "adaptation":
                if current_phase != "initialization":
                    phases.append({
                        "phase": current_phase,
                        "start_step": phase_start,
                        "end_step": i,
                        "duration": i - phase_start,
                    })
                current_phase = "adaptation"
                phase_start = i

        # Add final phase
        phases.append({
            "phase": current_phase,
            "start_step": phase_start,
            "end_step": len(growth_rates),
            "duration": len(growth_rates) - phase_start,
        })

        return {
            "phases": phases,
            "total_phases": len(phases),
            "growth_rates": growth_rates,
        }

    def _analyze_adapter_gradients(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze adapter gradient data."""
        gradient_data = data[DataType.ADAPTER_GRADIENTS]

        if not gradient_data:
            return {"status": "no_data"}

        # Similar analysis structure as weights but for gradients
        return {
            "gradient_evolution": "gradient_analysis_placeholder",
            "gradient_stability": "gradient_stability_placeholder",
        }

    def _analyze_lora_activations(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze LoRA activation data."""
        activation_data = data[DataType.LORA_ACTIVATIONS]

        if not activation_data:
            return {"status": "no_data"}

        # Analyze LoRA-specific activation patterns
        return {
            "activation_flow": "activation_flow_placeholder",
            "bottleneck_analysis": "bottleneck_analysis_placeholder",
        }

    def _perform_cross_analysis(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Perform cross-analysis between different data types."""
        cross_analysis = {}

        # Weight-Gradient correlation
        if (DataType.ADAPTER_WEIGHTS in data
                and DataType.ADAPTER_GRADIENTS in data):
            cross_analysis["weight_gradient_correlation"] = self._analyze_weight_gradient_correlation(
                data[DataType.ADAPTER_WEIGHTS],
                data[DataType.ADAPTER_GRADIENTS]
            )

        # Weight-Activation relationship
        if (DataType.ADAPTER_WEIGHTS in data
                and DataType.LORA_ACTIVATIONS in data):
            cross_analysis["weight_activation_relationship"] = self._analyze_weight_activation_relationship(
                data[DataType.ADAPTER_WEIGHTS],
                data[DataType.LORA_ACTIVATIONS]
            )

        return cross_analysis

    def _analyze_weight_gradient_correlation(self, weight_data: Any, gradient_data: Any) -> Dict[str, Any]:
        """Analyze correlation between weight evolution and gradients."""
        # Placeholder for weight-gradient correlation analysis
        return {"status": "analysis_placeholder"}

    def _analyze_weight_activation_relationship(self, weight_data: Any, activation_data: Any) -> Dict[str, Any]:
        """Analyze relationship between weights and activations."""
        # Placeholder for weight-activation relationship analysis
        return {"status": "analysis_placeholder"}

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

    def _save_detailed_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save detailed LoRA analysis results."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_dir / "lora_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save adapter-specific data
        if "adapter_weight_analysis" in results:
            weight_analysis = results["adapter_weight_analysis"]

            for module_name, module_data in weight_analysis.get("adapter_modules", {}).items():
                # Save evolution data for each module
                safe_module_name = module_name.replace("/", "_").replace(".", "_")

                if "effective_matrix_evolution" in module_data:
                    norms = module_data["effective_matrix_evolution"]["norms"]

                    with open(output_dir / f"{safe_module_name}_norm_evolution.csv", 'w') as f:
                        f.write("step,effective_norm\n")
                        for i, norm in enumerate(norms):
                            f.write(f"{i},{norm}\n")

    def _analyze_from_repo(self, repo_id: str, **kwargs) -> Dict[str, Any]:
        """Perform LoRA analysis from external repository using robust loading.

        Args:
            repo_id: HuggingFace repository ID
            **kwargs: Additional arguments (subfolder, revision, etc.)

        Returns:
            LoRA analysis results
        """
        subfolder = kwargs.get("subfolder")
        revision = kwargs.get("revision", "main")
        layer_filter = kwargs.get("layer_filter")

        # Load components using robust utilities
        components = get_lora_components_per_layer(
            repo_id=repo_id,
            subfolder=subfolder,
            revision=revision,
            layer_filter=layer_filter,
            force_download=kwargs.get("force_download", False),
        )

        if not components:
            return {"status": "no_components_found"}

        # Perform analysis on loaded components
        analysis_results = {
            "source": "external_repo",
            "repo_id": repo_id,
            "revision": revision,
            "total_layers": len(components),
            "layer_analysis": {},
            "global_statistics": {},
        }

        # Analyze each layer
        layer_stats = []
        for layer_name, layer_data in components.items():
            layer_analysis = {
                "rank": layer_data["rank"],
                "scaling": layer_data["scaling"],
                "statistics": layer_data["statistics"],
                "shape_A": layer_data["shape_A"],
                "shape_B": layer_data["shape_B"],
                "dtype": layer_data["dtype"],
            }

            # Add SVD analysis for rank utilization
            if "effective_weight" in layer_data:
                effective_weight = layer_data["effective_weight"]
                try:
                    U, S, Vh = torch.svd(effective_weight)

                    # Compute rank utilization metrics
                    total_singular_values = len(S)
                    effective_rank = torch.sum(S > 0.01 * S[0]).item()
                    rank_utilization = effective_rank / total_singular_values

                    layer_analysis["svd_analysis"] = {
                        "singular_values": S.tolist(),
                        "effective_rank": effective_rank,
                        "rank_utilization": rank_utilization,
                        "condition_number": (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf'),
                    }
                except Exception as e:
                    self.logger.warning(f"SVD analysis failed for {layer_name}: {e}")

            analysis_results["layer_analysis"][layer_name] = layer_analysis
            layer_stats.append(layer_analysis["statistics"])

        # Compute global statistics
        if layer_stats:
            analysis_results["global_statistics"] = {
                "mean_A_norm": np.mean([stats["A_norm"] for stats in layer_stats]),
                "mean_B_norm": np.mean([stats["B_norm"] for stats in layer_stats]),
                "mean_effective_norm": np.mean([stats["effective_norm"] for stats in layer_stats]),
                "total_parameters": sum(
                    np.prod(layer_data["shape_A"]) + np.prod(layer_data["shape_B"])
                    for layer_data in components.values()
                ),
            }

        return analysis_results
