"""Convergence analysis across different data types."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.base import DataAnalyzer, DataType


class ConvergenceAnalyzer(DataAnalyzer):
    """Analyzes convergence patterns across multiple data types."""

    @property
    def data_type(self) -> DataType:
        return DataType.CONVERGENCE_ANALYSIS

    @property
    def required_data_types(self) -> List[DataType]:
        # Flexible - can work with any combination of these
        return [DataType.ADAPTER_WEIGHTS, DataType.ADAPTER_GRADIENTS, DataType.ACTIVATIONS]

    def can_analyze(self, available_data: Dict[DataType, Any]) -> bool:
        """Check if convergence analysis can be performed."""
        # Need at least one of the required data types
        return any(dt in available_data for dt in self.required_data_types)

    def analyze(self, data: Dict[DataType, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze convergence patterns across available data types.

        Args:
            data: Dictionary containing collected data
            output_dir: Optional directory to save outputs

        Returns:
            Convergence analysis results
        """
        analysis_results = {
            "convergence_signals": self._detect_convergence_signals(data),
            "stability_analysis": self._analyze_stability_across_types(data),
            "plateau_detection": self._detect_plateaus(data),
            "convergence_timeline": self._analyze_convergence_timeline(data),
        }

        # Add data-type specific convergence analysis
        if DataType.ADAPTER_WEIGHTS in data:
            analysis_results["weight_convergence"] = self._analyze_weight_convergence(data)

        if DataType.ADAPTER_GRADIENTS in data:
            analysis_results["gradient_convergence"] = self._analyze_gradient_convergence(data)

        if DataType.ACTIVATIONS in data:
            analysis_results["activation_convergence"] = self._analyze_activation_convergence(data)

        # Cross-data-type convergence analysis
        available_types = [dt for dt in self.required_data_types if dt in data]
        if len(available_types) > 1:
            analysis_results["cross_type_convergence"] = self._analyze_cross_type_convergence(data, available_types)

        # Save detailed results if output directory provided
        if output_dir:
            self._save_detailed_results(analysis_results, output_dir)

        return analysis_results

    def _detect_convergence_signals(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Detect convergence signals across all available data types."""
        convergence_signals = {}

        for data_type, type_data in data.items():
            if data_type in self.required_data_types and type_data:
                signals = self._extract_convergence_signals_for_type(data_type, type_data)
                convergence_signals[data_type.value] = signals

        # Aggregate signals
        if convergence_signals:
            aggregated = self._aggregate_convergence_signals(convergence_signals)
            convergence_signals["aggregated"] = aggregated

        return convergence_signals

    def _extract_convergence_signals_for_type(self, data_type: DataType, type_data: Any) -> Dict[str, Any]:
        """Extract convergence signals for a specific data type."""
        if not isinstance(type_data, dict):
            return {"status": "invalid_data"}

        steps = sorted(type_data.keys())
        if len(steps) < 3:
            return {"status": "insufficient_data"}

        signals = {}

        if data_type == DataType.ADAPTER_WEIGHTS:
            signals = self._extract_weight_convergence_signals(type_data, steps)
        elif data_type == DataType.ADAPTER_GRADIENTS:
            signals = self._extract_gradient_convergence_signals(type_data, steps)
        elif data_type == DataType.ACTIVATIONS:
            signals = self._extract_activation_convergence_signals(type_data, steps)

        return signals

    def _extract_weight_convergence_signals(self, weight_data: Dict[int, Any], steps: List[int]) -> Dict[str, Any]:
        """Extract convergence signals from weight data."""
        # Track overall norm changes
        total_norms = []

        for step in steps:
            step_data = weight_data[step]
            if isinstance(step_data, dict) and "adapter_weights" in step_data:
                step_norm = 0
                for module_data in step_data["adapter_weights"].values():
                    stats = module_data.get("statistics", {})
                    step_norm += stats.get("effective_norm", 0)
                total_norms.append(step_norm)

        if not total_norms:
            return {"status": "no_norm_data"}

        # Compute convergence metrics
        recent_window = min(10, len(total_norms) // 4)
        if recent_window < 2:
            recent_window = 2

        recent_norms = total_norms[-recent_window:]
        recent_stability = float(np.std(recent_norms))

        # Compute rate of change
        changes = np.diff(total_norms)
        recent_changes = changes[-recent_window:] if len(changes) >= recent_window else changes
        avg_change_rate = float(np.mean(np.abs(recent_changes))) if recent_changes.size > 0 else 0.0

        return {
            "total_norms": total_norms,
            "recent_stability": recent_stability,
            "avg_change_rate": avg_change_rate,
            "convergence_score": self._compute_convergence_score(recent_stability, avg_change_rate),
            "converged": recent_stability < 0.01 and avg_change_rate < 0.001,
        }

    def _extract_gradient_convergence_signals(self, gradient_data: Dict[int, Any], steps: List[int]) -> Dict[str, Any]:
        """Extract convergence signals from gradient data."""
        # Track global gradient norms
        global_grad_norms = []

        for step in steps:
            step_data = gradient_data[step]
            if isinstance(step_data, dict) and "global_statistics" in step_data:
                global_stats = step_data["global_statistics"]
                global_grad_norms.append(global_stats.get("global_grad_norm", 0))

        if not global_grad_norms:
            return {"status": "no_gradient_data"}

        # Convergence analysis based on gradient norms
        recent_window = min(10, len(global_grad_norms) // 4)
        if recent_window < 2:
            recent_window = 2

        recent_norms = global_grad_norms[-recent_window:]
        recent_stability = float(np.std(recent_norms))
        mean_recent_norm = float(np.mean(recent_norms))

        return {
            "global_grad_norms": global_grad_norms,
            "recent_stability": recent_stability,
            "mean_recent_norm": mean_recent_norm,
            "convergence_score": self._compute_convergence_score(recent_stability, mean_recent_norm),
            "converged": recent_stability < 0.01 and mean_recent_norm < 0.1,
        }

    def _extract_activation_convergence_signals(
        self, activation_data: Dict[int, Any], steps: List[int]
    ) -> Dict[str, Any]:
        """Extract convergence signals from activation data."""
        # Track average activation magnitudes
        avg_magnitudes = []

        for step in steps:
            step_data = activation_data[step]
            if isinstance(step_data, dict) and "activations" in step_data:
                step_magnitudes = []
                for point_data in step_data["activations"].values():
                    stats = point_data.get("statistics", {})
                    step_magnitudes.append(stats.get("norm", 0))

                if step_magnitudes:
                    avg_magnitudes.append(np.mean(step_magnitudes))

        if not avg_magnitudes:
            return {"status": "no_activation_data"}

        # Convergence analysis
        recent_window = min(10, len(avg_magnitudes) // 4)
        if recent_window < 2:
            recent_window = 2

        recent_magnitudes = avg_magnitudes[-recent_window:]
        recent_stability = float(np.std(recent_magnitudes))

        changes = np.diff(avg_magnitudes)
        recent_changes = changes[-recent_window:] if len(changes) >= recent_window else changes
        avg_change_rate = float(np.mean(np.abs(recent_changes))) if recent_changes.size > 0 else 0.0

        return {
            "avg_magnitudes": avg_magnitudes,
            "recent_stability": recent_stability,
            "avg_change_rate": avg_change_rate,
            "convergence_score": self._compute_convergence_score(recent_stability, avg_change_rate),
            "converged": recent_stability < 0.05 and avg_change_rate < 0.01,
        }

    def _compute_convergence_score(self, stability: float, change_rate: float) -> float:
        """Compute a convergence score from stability and change rate."""
        # Normalize and combine metrics (higher score = more converged)
        stability_score = max(0, 1 - stability * 10)  # Penalize high variance
        change_score = max(0, 1 - change_rate * 100)  # Penalize high change rates

        return (stability_score + change_score) / 2

    def _aggregate_convergence_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate convergence signals across data types."""
        convergence_scores = []
        converged_flags = []

        for data_type, type_signals in signals.items():
            if isinstance(type_signals, dict):
                score = type_signals.get("convergence_score")
                converged = type_signals.get("converged")

                if score is not None:
                    convergence_scores.append(score)
                if converged is not None:
                    converged_flags.append(converged)

        if not convergence_scores:
            return {"status": "no_scores"}

        return {
            "overall_convergence_score": np.mean(convergence_scores),
            "min_convergence_score": np.min(convergence_scores),
            "max_convergence_score": np.max(convergence_scores),
            "consensus_converged": all(converged_flags) if converged_flags else False,
            "partial_convergence": any(converged_flags) if converged_flags else False,
            "convergence_consistency": np.std(convergence_scores) if len(convergence_scores) > 1 else 0,
        }

    def _analyze_stability_across_types(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze stability patterns across different data types."""
        stability_analysis = {}

        for data_type in self.required_data_types:
            if data_type in data:
                stability = self._compute_stability_for_type(data_type, data[data_type])
                stability_analysis[data_type.value] = stability

        return stability_analysis

    def _compute_stability_for_type(self, data_type: DataType, type_data: Any) -> Dict[str, Any]:
        """Compute stability metrics for a specific data type."""
        if not isinstance(type_data, dict):
            return {"status": "invalid_data"}

        steps = sorted(type_data.keys())
        if len(steps) < 3:
            return {"status": "insufficient_data"}

        # Extract relevant metrics based on data type
        if data_type == DataType.ADAPTER_WEIGHTS:
            return self._compute_weight_stability(type_data, steps)
        elif data_type == DataType.ADAPTER_GRADIENTS:
            return self._compute_gradient_stability(type_data, steps)
        elif data_type == DataType.ACTIVATIONS:
            return self._compute_activation_stability(type_data, steps)

        return {"status": "unsupported_type"}

    def _compute_weight_stability(self, weight_data: Dict[int, Any], steps: List[int]) -> Dict[str, Any]:
        """Compute stability metrics for weights."""
        # Similar to convergence signals but focused on stability
        return {"stability_placeholder": "weight_stability"}

    def _compute_gradient_stability(self, gradient_data: Dict[int, Any], steps: List[int]) -> Dict[str, Any]:
        """Compute stability metrics for gradients."""
        return {"stability_placeholder": "gradient_stability"}

    def _compute_activation_stability(self, activation_data: Dict[int, Any], steps: List[int]) -> Dict[str, Any]:
        """Compute stability metrics for activations."""
        return {"stability_placeholder": "activation_stability"}

    def _detect_plateaus(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Detect plateau regions in training."""
        plateau_analysis = {}

        for data_type in self.required_data_types:
            if data_type in data:
                plateaus = self._detect_plateaus_for_type(data_type, data[data_type])
                plateau_analysis[data_type.value] = plateaus

        return plateau_analysis

    def _detect_plateaus_for_type(self, data_type: DataType, type_data: Any) -> Dict[str, Any]:
        """Detect plateaus for a specific data type."""
        # Placeholder for plateau detection logic
        return {"plateaus": "placeholder"}

    def _analyze_convergence_timeline(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Analyze the timeline of convergence across data types."""
        timeline_analysis = {}

        # Extract convergence points for each data type
        convergence_points = {}

        for data_type in self.required_data_types:
            if data_type in data:
                convergence_point = self._find_convergence_point(data_type, data[data_type])
                if convergence_point is not None:
                    convergence_points[data_type.value] = convergence_point

        if convergence_points:
            timeline_analysis = {
                "convergence_points": convergence_points,
                "earliest_convergence": min(convergence_points.values()),
                "latest_convergence": max(convergence_points.values()),
                "convergence_spread": max(convergence_points.values()) - min(convergence_points.values()),
                "synchronized": len(set(convergence_points.values())) == 1,
            }

        return timeline_analysis

    def _find_convergence_point(self, data_type: DataType, type_data: Any) -> Optional[int]:
        """Find the step where convergence occurred for a data type."""
        # Simplified convergence detection - return midpoint for now
        if isinstance(type_data, dict) and type_data:
            steps = sorted(type_data.keys())
            return steps[len(steps) // 2] if steps else None
        return None

    def _analyze_weight_convergence(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Detailed weight convergence analysis."""
        return {"weight_convergence": "detailed_analysis_placeholder"}

    def _analyze_gradient_convergence(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Detailed gradient convergence analysis."""
        return {"gradient_convergence": "detailed_analysis_placeholder"}

    def _analyze_activation_convergence(self, data: Dict[DataType, Any]) -> Dict[str, Any]:
        """Detailed activation convergence analysis."""
        return {"activation_convergence": "detailed_analysis_placeholder"}

    def _analyze_cross_type_convergence(
        self, data: Dict[DataType, Any], available_types: List[DataType]
    ) -> Dict[str, Any]:
        """Analyze convergence relationships between different data types."""
        cross_analysis = {
            "available_types": [dt.value for dt in available_types],
            "correlation_analysis": {},
            "lead_lag_analysis": {},
        }

        # Placeholder for cross-type analysis
        return cross_analysis

    def _save_detailed_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save detailed convergence analysis results."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_dir / "convergence_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save convergence timeline if available
        if "convergence_timeline" in results:
            timeline = results["convergence_timeline"]

            if "convergence_points" in timeline:
                with open(output_dir / "convergence_timeline.csv", "w") as f:
                    f.write("data_type,convergence_step\n")
                    for data_type, step in timeline["convergence_points"].items():
                        f.write(f"{data_type},{step}\n")
