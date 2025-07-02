"""Similarity analysis across different data types and time steps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..core.base import DataAnalyzer, DataType


class SimilarityAnalyzer(DataAnalyzer):
    """Analyzes similarities and correlations across data types and time."""

    @property
    def data_type(self) -> DataType:
        return DataType.SIMILARITY_ANALYSIS

    @property
    def required_data_types(self) -> List[DataType]:
        # Can work with any of these data types
        return [DataType.ADAPTER_WEIGHTS, DataType.ADAPTER_GRADIENTS, DataType.ACTIVATIONS, DataType.LORA_ACTIVATIONS]

    def can_analyze(self, available_data: Dict[DataType, Any]) -> bool:
        """Check if similarity analysis can be performed."""
        # Need at least one data type with multiple time steps
        for data_type in self.required_data_types:
            if data_type in available_data:
                data = available_data[data_type]
                if isinstance(data, dict) and len(data) >= 2:
                    return True
        return False

    def analyze(self, data: Dict[DataType, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Perform comprehensive similarity analysis.

        Args:
            data: Dictionary containing collected data
            output_dir: Optional directory to save outputs

        Returns:
            Similarity analysis results
        """
        analysis_results = {
            "temporal_similarity": self._analyze_temporal_similarity(data),
            "cross_type_similarity": self._analyze_cross_type_similarity(data),
            "reference_similarity": self._analyze_reference_similarity(data),
            "similarity_trends": self._analyze_similarity_trends(data),
        }

        # Add data-type specific similarity analysis
        for data_type in self.required_data_types:
            if data_type in data:
                type_similarity = self._analyze_type_specific_similarity(data_type, data[data_type])
                analysis_results[f"{data_type.value}_similarity"] = type_similarity

        # Save detailed results if output directory provided
        if output_dir:
            self._save_detailed_results(analysis_results, output_dir)

        return analysis_results

    def _analyze_temporal_similarity(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze similarity patterns over time for each data type."""
        temporal_analysis = {}

        for data_type in self.required_data_types:
            if data_type in data and isinstance(data[data_type], dict):
                type_data = data[data_type]
                steps = sorted(type_data.keys())

                if len(steps) >= 2:
                    similarities = self._compute_temporal_similarities(data_type, type_data, steps)
                    temporal_analysis[data_type.value] = similarities

        return temporal_analysis

    def _compute_temporal_similarities(
        self, data_type: DataType, type_data: Dict[int, Any], steps: List[int]
    ) -> Dict[str, Any]:
        """Compute temporal similarities for a specific data type."""
        consecutive_similarities = []
        reference_similarities = []

        # Reference is the first step
        reference_step = steps[0]
        reference_data = type_data[reference_step]

        for i in range(1, len(steps)):
            current_step = steps[i]
            current_data = type_data[current_step]

            # Consecutive similarity (with previous step)
            if i > 0:
                prev_step = steps[i - 1]
                prev_data = type_data[prev_step]

                consecutive_sim = self._compute_data_similarity(data_type, prev_data, current_data)
                if consecutive_sim is not None:
                    consecutive_similarities.append(consecutive_sim)

            # Reference similarity (with first step)
            ref_sim = self._compute_data_similarity(data_type, reference_data, current_data)
            if ref_sim is not None:
                reference_similarities.append(ref_sim)

        return {
            "consecutive_similarities": consecutive_similarities,
            "reference_similarities": reference_similarities,
            "mean_consecutive_similarity": np.mean(consecutive_similarities) if consecutive_similarities else 0,
            "mean_reference_similarity": np.mean(reference_similarities) if reference_similarities else 0,
            "similarity_trend": (
                self._compute_trend(np.array(consecutive_similarities)) if consecutive_similarities else "stable"
            ),
            "stability_score": 1 - np.std(consecutive_similarities) if consecutive_similarities else 1,
        }

    def _compute_data_similarity(self, data_type: DataType, data1: Any, data2: Any) -> Optional[float]:
        """Compute similarity between two data points of the same type."""
        if data_type == DataType.ADAPTER_WEIGHTS:
            return self._compute_weight_similarity(data1, data2)
        elif data_type == DataType.ADAPTER_GRADIENTS:
            return self._compute_gradient_similarity(data1, data2)
        elif data_type in [DataType.ACTIVATIONS, DataType.LORA_ACTIVATIONS]:
            return self._compute_activation_similarity(data1, data2)

        return None

    def _compute_weight_similarity(self, weight_data1: Any, weight_data2: Any) -> Optional[float]:
        """Compute similarity between two weight data points."""
        if (
            not isinstance(weight_data1, dict)
            or "adapter_weights" not in weight_data1
            or not isinstance(weight_data2, dict)
            or "adapter_weights" not in weight_data2
        ):
            return None

        weights1 = weight_data1["adapter_weights"]
        weights2 = weight_data2["adapter_weights"]

        # Find common modules
        common_modules = set(weights1.keys()) & set(weights2.keys())
        if not common_modules:
            return None

        similarities = []

        for module_name in common_modules:
            module1 = weights1[module_name]
            module2 = weights2[module_name]

            # Compare effective weights using cosine similarity
            if "effective_weight" in module1 and "effective_weight" in module2:
                weight1_tensor = module1["effective_weight"]
                weight2_tensor = module2["effective_weight"]

                # Flatten tensors
                flat1 = weight1_tensor.flatten()
                flat2 = weight2_tensor.flatten()

                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

                similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else None

    def _compute_gradient_similarity(self, grad_data1: Any, grad_data2: Any) -> Optional[float]:
        """Compute similarity between two gradient data points."""
        if (
            not isinstance(grad_data1, dict)
            or "adapter_gradients" not in grad_data1
            or not isinstance(grad_data2, dict)
            or "adapter_gradients" not in grad_data2
        ):
            return None

        grads1 = grad_data1["adapter_gradients"]
        grads2 = grad_data2["adapter_gradients"]

        # Find common modules
        common_modules = set(grads1.keys()) & set(grads2.keys())
        if not common_modules:
            return None

        similarities = []

        for module_name in common_modules:
            module1 = grads1[module_name]
            module2 = grads2[module_name]

            # Compare effective gradients
            if "effective_gradient" in module1 and "effective_gradient" in module2:
                grad1_tensor = module1["effective_gradient"]
                grad2_tensor = module2["effective_gradient"]

                # Flatten tensors
                flat1 = grad1_tensor.flatten()
                flat2 = grad2_tensor.flatten()

                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

                similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else None

    def _compute_activation_similarity(self, act_data1: Any, act_data2: Any) -> Optional[float]:
        """Compute similarity between two activation data points."""
        # Handle both regular activations and LoRA activations
        activations1 = None
        activations2 = None

        if isinstance(act_data1, dict):
            if "activations" in act_data1:
                activations1 = act_data1["activations"]
            elif "lora_activations" in act_data1:
                activations1 = act_data1["lora_activations"]

        if isinstance(act_data2, dict):
            if "activations" in act_data2:
                activations2 = act_data2["activations"]
            elif "lora_activations" in act_data2:
                activations2 = act_data2["lora_activations"]

        if not activations1 or not activations2:
            return None

        # Find common activation points
        common_points = set(activations1.keys()) & set(activations2.keys())
        if not common_points:
            return None

        similarities = []

        for point_name in common_points:
            point1 = activations1[point_name]
            point2 = activations2[point_name]

            # Extract statistics for comparison
            if (
                isinstance(point1, dict)
                and isinstance(point2, dict)
                and "statistics" in point1
                and "statistics" in point2
            ):
                stats1 = point1["statistics"]
                stats2 = point2["statistics"]

                # Compare using magnitude similarity
                norm1 = stats1.get("norm", 0)
                norm2 = stats2.get("norm", 0)

                if norm1 > 0 and norm2 > 0:
                    similarity = min(norm1, norm2) / max(norm1, norm2)
                    similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else None

    def _analyze_cross_type_similarity(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze similarities between different data types at the same time steps."""
        available_types = [dt for dt in self.required_data_types if dt in data]

        if len(available_types) < 2:
            return {"status": "insufficient_types"}

        cross_similarities = {}

        # Find common time steps across all available types
        common_steps = None
        for data_type in available_types:
            if isinstance(data[data_type], dict):
                type_steps = set(data[data_type].keys())
                if common_steps is None:
                    common_steps = type_steps
                else:
                    common_steps &= type_steps

        if not common_steps:
            return {"status": "no_common_steps"}

        # Analyze similarities between types at each common step
        for step in sorted(common_steps):
            step_similarities = {}

            for i, type1 in enumerate(available_types):
                for type2 in available_types[i + 1 :]:
                    pair_key = f"{type1.value}_{type2.value}"

                    # This is a simplified cross-type similarity
                    # In practice, you might need domain-specific similarity measures
                    similarity = self._compute_cross_type_similarity(type1, data[type1][step], type2, data[type2][step])

                    if similarity is not None:
                        step_similarities[pair_key] = similarity

            if step_similarities:
                cross_similarities[step] = step_similarities

        # Aggregate cross-type similarities
        if cross_similarities:
            aggregated = self._aggregate_cross_type_similarities(cross_similarities)
            cross_similarities["aggregated"] = aggregated

        return cross_similarities

    def _compute_cross_type_similarity(
        self, type1: DataType, data1: Any, type2: DataType, data2: Any
    ) -> Optional[float]:
        """Compute similarity between different data types (simplified)."""
        # This is a placeholder for cross-type similarity
        # Real implementation would need domain knowledge about relationships

        # For now, return a simple correlation based on available statistics
        stats1 = self._extract_summary_stats(type1, data1)
        stats2 = self._extract_summary_stats(type2, data2)

        if stats1 is not None and stats2 is not None:
            # Simple correlation between summary statistics
            return abs(np.corrcoef([stats1], [stats2])[0, 1])

        return None

    def _extract_summary_stats(self, data_type: DataType, data: Any) -> Optional[float]:
        """Extract a summary statistic from data for cross-type comparison."""
        if data_type == DataType.ADAPTER_WEIGHTS:
            if isinstance(data, dict) and "adapter_weights" in data:
                total_norm = 0
                for module_data in data["adapter_weights"].values():
                    stats = module_data.get("statistics", {})
                    total_norm += stats.get("effective_norm", 0)
                return total_norm

        elif data_type == DataType.ADAPTER_GRADIENTS:
            if isinstance(data, dict) and "global_statistics" in data:
                return data["global_statistics"].get("global_grad_norm", 0)

        elif data_type in [DataType.ACTIVATIONS, DataType.LORA_ACTIVATIONS]:
            if isinstance(data, dict):
                activations_key = "activations" if "activations" in data else "lora_activations"
                if activations_key in data:
                    total_norm = 0
                    for point_data in data[activations_key].values():
                        if isinstance(point_data, dict) and "statistics" in point_data:
                            total_norm += point_data["statistics"].get("norm", 0)
                    return total_norm

        return None

    def _aggregate_cross_type_similarities(self, cross_similarities: Dict[int, Any]) -> Dict[str, Any]:
        """Aggregate cross-type similarities across time steps."""
        all_pairs = set()
        for step_sims in cross_similarities.values():
            if isinstance(step_sims, dict):
                all_pairs.update(step_sims.keys())

        aggregated = {}

        for pair in all_pairs:
            pair_similarities = []
            for step_sims in cross_similarities.values():
                if isinstance(step_sims, dict) and pair in step_sims:
                    pair_similarities.append(step_sims[pair])

            if pair_similarities:
                aggregated[pair] = {
                    "mean_similarity": np.mean(pair_similarities),
                    "std_similarity": np.std(pair_similarities),
                    "trend": self._compute_trend(np.array(pair_similarities)),
                    "values": pair_similarities,
                }

        return aggregated

    def _analyze_reference_similarity(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze similarity to reference points (e.g., initial state, final state)."""
        reference_analysis = {}

        for data_type in self.required_data_types:
            if data_type in data and isinstance(data[data_type], dict):
                type_data = data[data_type]
                steps = sorted(type_data.keys())

                if len(steps) >= 2:
                    ref_analysis = self._compute_reference_similarities(data_type, type_data, steps)
                    reference_analysis[data_type.value] = ref_analysis

        return reference_analysis

    def _compute_reference_similarities(
        self, data_type: DataType, type_data: Dict[int, Any], steps: List[int]
    ) -> Dict[str, Any]:
        """Compute similarities to various reference points."""
        initial_step = steps[0]
        final_step = steps[-1]
        mid_step = steps[len(steps) // 2]

        initial_data = type_data[initial_step]
        final_data = type_data[final_step]
        mid_data = type_data[mid_step]

        similarities_to_initial = []
        similarities_to_final = []
        similarities_to_mid = []

        for step in steps:
            current_data = type_data[step]

            # Similarity to initial state
            sim_initial = self._compute_data_similarity(data_type, initial_data, current_data)
            if sim_initial is not None:
                similarities_to_initial.append(sim_initial)

            # Similarity to final state
            sim_final = self._compute_data_similarity(data_type, final_data, current_data)
            if sim_final is not None:
                similarities_to_final.append(sim_final)

            # Similarity to mid state
            sim_mid = self._compute_data_similarity(data_type, mid_data, current_data)
            if sim_mid is not None:
                similarities_to_mid.append(sim_mid)

        return {
            "to_initial": {
                "similarities": similarities_to_initial,
                "mean": np.mean(similarities_to_initial) if similarities_to_initial else 0,
                "trend": (
                    self._compute_trend(np.array(similarities_to_initial)) if similarities_to_initial else "stable"
                ),
            },
            "to_final": {
                "similarities": similarities_to_final,
                "mean": np.mean(similarities_to_final) if similarities_to_final else 0,
                "trend": self._compute_trend(np.array(similarities_to_final)) if similarities_to_final else "stable",
            },
            "to_mid": {
                "similarities": similarities_to_mid,
                "mean": np.mean(similarities_to_mid) if similarities_to_mid else 0,
                "trend": self._compute_trend(np.array(similarities_to_mid)) if similarities_to_mid else "stable",
            },
        }

    def _analyze_similarity_trends(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze overall trends in similarity patterns."""
        trend_analysis = {}

        for data_type in self.required_data_types:
            if data_type in data:
                trends = self._compute_similarity_trends_for_type(data_type, data[data_type])
                trend_analysis[data_type.value] = trends

        return trend_analysis

    def _compute_similarity_trends_for_type(self, data_type: DataType, type_data: Any) -> Dict[str, Any]:
        """Compute similarity trends for a specific data type."""
        # Placeholder for detailed trend analysis
        return {"trend_analysis": "placeholder"}

    def _analyze_type_specific_similarity(self, data_type: DataType, type_data: Any) -> Dict[str, Any]:
        """Perform data-type specific similarity analysis."""
        if data_type == DataType.ADAPTER_WEIGHTS:
            return self._analyze_weight_specific_similarity(type_data)
        elif data_type == DataType.ADAPTER_GRADIENTS:
            return self._analyze_gradient_specific_similarity(type_data)
        elif data_type in [DataType.ACTIVATIONS, DataType.LORA_ACTIVATIONS]:
            return self._analyze_activation_specific_similarity(type_data)

        return {"status": "unsupported_type"}

    def _analyze_weight_specific_similarity(self, weight_data: Any) -> Dict[str, Any]:
        """Weight-specific similarity analysis."""
        return {"weight_specific": "placeholder"}

    def _analyze_gradient_specific_similarity(self, gradient_data: Any) -> Dict[str, Any]:
        """Gradient-specific similarity analysis."""
        return {"gradient_specific": "placeholder"}

    def _analyze_activation_specific_similarity(self, activation_data: Any) -> Dict[str, Any]:
        """Activation-specific similarity analysis."""
        return {"activation_specific": "placeholder"}

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
        """Save detailed similarity analysis results."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_dir / "similarity_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save similarity matrices if available
        if "temporal_similarity" in results:
            for data_type, type_results in results["temporal_similarity"].items():
                if isinstance(type_results, dict) and "consecutive_similarities" in type_results:
                    similarities = type_results["consecutive_similarities"]

                    safe_type_name = data_type.replace("/", "_").replace(".", "_")
                    with open(output_dir / f"{safe_type_name}_temporal_similarities.csv", "w") as f:
                        f.write("step,similarity\n")
                        for i, sim in enumerate(similarities):
                            f.write(f"{i+1},{sim}\n")
