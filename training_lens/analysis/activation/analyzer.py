"""Activation analysis for the new framework."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.base import DataAnalyzer, DataType


class ActivationAnalyzer(DataAnalyzer):
    """Analyzes activation data collected during training."""

    @property
    def data_type(self) -> DataType:
        return DataType.ACTIVATION_ANALYSIS

    @property
    def required_data_types(self) -> List[DataType]:
        return [DataType.ACTIVATIONS]

    def can_analyze(self, available_data: Dict[DataType, Any]) -> bool:
        """Check if activation analysis can be performed."""
        return DataType.ACTIVATIONS in available_data

    def analyze(self, data: Dict[DataType, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze activation evolution across training steps.

        Args:
            data: Dictionary containing collected data
            output_dir: Optional directory to save outputs

        Returns:
            Activation analysis results
        """
        activation_data = data[DataType.ACTIVATIONS]

        if not activation_data:
            return {"status": "no_data"}

        analysis_results = {
            "activation_evolution": self._analyze_activation_evolution(activation_data),
            "magnitude_trends": self._analyze_magnitude_trends(activation_data),
            "distribution_changes": self._analyze_distribution_changes(activation_data),
            "stability_metrics": self._compute_stability_metrics(activation_data),
        }

        # Save detailed results if output directory provided
        if output_dir:
            self._save_detailed_results(analysis_results, output_dir)

        return analysis_results

    def _analyze_activation_evolution(self, activation_data: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze how activations evolve across training steps."""
        steps = sorted(activation_data.keys())
        evolution_analysis = {
            "total_steps": len(steps),
            "step_range": [min(steps), max(steps)],
            "activation_points": {},
        }

        # Get all activation points
        if steps:
            first_step_data = activation_data[steps[0]]
            if isinstance(first_step_data, dict) and "activations" in first_step_data:
                activation_points = list(first_step_data["activations"].keys())

                for point_name in activation_points:
                    point_evolution = self._analyze_single_activation_point(activation_data, point_name)
                    evolution_analysis["activation_points"][point_name] = point_evolution

        return evolution_analysis

    def _analyze_single_activation_point(self, activation_data: Dict[int, Any], point_name: str) -> Dict[str, Any]:
        """Analyze evolution of a single activation point."""
        steps = sorted(activation_data.keys())
        magnitudes = []
        means = []
        stds = []

        for step in steps:
            step_data = activation_data[step]
            if isinstance(step_data, dict) and "activations" in step_data and point_name in step_data["activations"]:

                point_data = step_data["activations"][point_name]
                stats = point_data.get("statistics", {})

                magnitudes.append(stats.get("norm", 0))
                means.append(stats.get("mean", 0))
                stds.append(stats.get("std", 0))

        if not magnitudes:
            return {"status": "no_data"}

        # Compute trends
        magnitude_trend = self._compute_trend(np.array(magnitudes))
        mean_trend = self._compute_trend(np.array(means))
        std_trend = self._compute_trend(np.array(stds))

        return {
            "magnitude_evolution": {
                "values": magnitudes,
                "trend": magnitude_trend,
                "initial": magnitudes[0],
                "final": magnitudes[-1],
                "change_percent": ((magnitudes[-1] - magnitudes[0]) / magnitudes[0] * 100) if magnitudes[0] != 0 else 0,
            },
            "mean_evolution": {"values": means, "trend": mean_trend, "stability": np.std(means) if means else 0},
            "std_evolution": {"values": stds, "trend": std_trend, "stability": np.std(stds) if stds else 0},
        }

    def _analyze_magnitude_trends(self, activation_data: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze overall magnitude trends across all activation points."""
        steps = sorted(activation_data.keys())
        overall_magnitudes = []

        for step in steps:
            step_data = activation_data[step]
            if isinstance(step_data, dict) and "activations" in step_data:
                step_magnitudes = []
                for point_data in step_data["activations"].values():
                    stats = point_data.get("statistics", {})
                    magnitude = stats.get("norm", 0)
                    step_magnitudes.append(magnitude)

                if step_magnitudes:
                    overall_magnitudes.append(np.mean(step_magnitudes))

        if not overall_magnitudes:
            return {"status": "no_data"}

        trend = self._compute_trend(np.array(overall_magnitudes))

        return {
            "overall_magnitude_trend": trend,
            "magnitude_values": overall_magnitudes,
            "magnitude_stability": np.std(overall_magnitudes),
            "magnitude_range": [min(overall_magnitudes), max(overall_magnitudes)],
        }

    def _analyze_distribution_changes(self, activation_data: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze how activation distributions change over training."""
        steps = sorted(activation_data.keys())
        distribution_analysis = {}

        if len(steps) < 2:
            return {"status": "insufficient_data"}

        # Compare first and last distributions
        first_step = steps[0]
        last_step = steps[-1]

        first_data = activation_data[first_step]
        last_data = activation_data[last_step]

        if (
            isinstance(first_data, dict)
            and "activations" in first_data
            and isinstance(last_data, dict)
            and "activations" in last_data
        ):

            for point_name in first_data["activations"].keys():
                if point_name in last_data["activations"]:
                    first_stats = first_data["activations"][point_name].get("statistics", {})
                    last_stats = last_data["activations"][point_name].get("statistics", {})

                    distribution_analysis[point_name] = {
                        "mean_shift": last_stats.get("mean", 0) - first_stats.get("mean", 0),
                        "std_change": last_stats.get("std", 0) - first_stats.get("std", 0),
                        "range_expansion": (
                            (last_stats.get("max", 0) - last_stats.get("min", 0))
                            - (first_stats.get("max", 0) - first_stats.get("min", 0))
                        ),
                        "sparsity_change": (last_stats.get("nonzero_ratio", 1) - first_stats.get("nonzero_ratio", 1)),
                    }

        return distribution_analysis

    def _compute_stability_metrics(self, activation_data: Dict[int, Any]) -> Dict[str, Any]:
        """Compute stability metrics for activations."""
        steps = sorted(activation_data.keys())

        if len(steps) < 3:
            return {"status": "insufficient_data"}

        # Compute consecutive similarities
        similarities = []

        for i in range(1, len(steps)):
            prev_step = steps[i - 1]
            curr_step = steps[i]

            similarity = self._compute_step_similarity(activation_data[prev_step], activation_data[curr_step])

            if similarity is not None:
                similarities.append(similarity)

        if not similarities:
            return {"status": "no_similarities"}

        stability_score = float(np.mean(similarities))
        stability_trend = self._compute_trend(np.array(similarities))

        return {
            "stability_score": stability_score,
            "stability_trend": stability_trend,
            "similarity_values": similarities,
            "stability_classification": self._classify_stability(stability_score),
        }

    def _compute_step_similarity(self, step1_data: Any, step2_data: Any) -> Optional[float]:
        """Compute similarity between two training steps."""
        if (
            not isinstance(step1_data, dict)
            or "activations" not in step1_data
            or not isinstance(step2_data, dict)
            or "activations" not in step2_data
        ):
            return None

        similarities = []

        for point_name in step1_data["activations"].keys():
            if point_name in step2_data["activations"]:
                # Compare statistics as a simple similarity measure
                stats1 = step1_data["activations"][point_name].get("statistics", {})
                stats2 = step2_data["activations"][point_name].get("statistics", {})

                # Use magnitude similarity as a proxy
                norm1 = stats1.get("norm", 0)
                norm2 = stats2.get("norm", 0)

                if norm1 > 0 and norm2 > 0:
                    similarity = min(norm1, norm2) / max(norm1, norm2)
                    similarities.append(similarity)

        return float(np.mean(similarities)) if similarities else None

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

    def _classify_stability(self, stability_score: float) -> str:
        """Classify stability based on score."""
        if stability_score > 0.9:
            return "very_stable"
        elif stability_score > 0.7:
            return "stable"
        elif stability_score > 0.5:
            return "moderate"
        else:
            return "unstable"

    def _save_detailed_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save detailed analysis results to files."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_dir / "activation_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save visualization-friendly data if available
        if "activation_evolution" in results:
            evolution_data = results["activation_evolution"]

            # Extract time series data for each activation point
            for point_name, point_data in evolution_data.get("activation_points", {}).items():
                if "magnitude_evolution" in point_data:
                    magnitude_values = point_data["magnitude_evolution"]["values"]

                    # Save as CSV for easy plotting
                    with open(output_dir / f"{point_name}_magnitude_evolution.csv", "w") as f:
                        f.write("step,magnitude\n")
                        for i, magnitude in enumerate(magnitude_values):
                            f.write(f"{i},{magnitude}\n")
