"""Weight analysis for training insights."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class WeightAnalyzer:
    """Specialized analyzer for weight-related insights."""

    def __init__(self, weight_data: Optional[Dict[str, Any]] = None):
        """Initialize weight analyzer.

        Args:
            weight_data: Optional weight data dictionary
        """
        self.weight_data = weight_data or {}
        self.weight_history = []
        self.layer_weights = {}

        if weight_data:
            self._extract_weight_info()

    def _extract_weight_info(self) -> None:
        """Extract weight information from data."""
        if "weight_stats_history" in self.weight_data:
            self.weight_history = self.weight_data["weight_stats_history"]

        if "layer_weights" in self.weight_data:
            self.layer_weights = self.weight_data["layer_weights"]

    def analyze_weight_evolution(self) -> Dict[str, Any]:
        """Analyze how weights evolve during training.

        Returns:
            Analysis of weight evolution
        """
        if not self.weight_history:
            return {"status": "no_data"}

        analysis = {}

        # Extract time series data
        steps = [entry["step"] for entry in self.weight_history]
        overall_norms = [entry["overall_norm"] for entry in self.weight_history]
        overall_means = [entry["overall_mean"] for entry in self.weight_history]
        overall_stds = [entry["overall_std"] for entry in self.weight_history]

        # Basic statistics
        analysis["norm_statistics"] = {
            "initial_norm": float(overall_norms[0]) if overall_norms else 0.0,
            "final_norm": float(overall_norms[-1]) if overall_norms else 0.0,
            "max_norm": float(np.max(overall_norms)) if overall_norms else 0.0,
            "min_norm": float(np.min(overall_norms)) if overall_norms else 0.0,
            "mean_norm": float(np.mean(overall_norms)) if overall_norms else 0.0,
            "std_norm": float(np.std(overall_norms)) if overall_norms else 0.0,
        }

        # Trend analysis
        if len(overall_norms) > 5:
            analysis["norm_trend"] = self._analyze_trend(np.array(overall_norms))
            analysis["mean_trend"] = self._analyze_trend(np.array(overall_means))
            analysis["std_trend"] = self._analyze_trend(np.array(overall_stds))

        # Stability assessment
        analysis["stability_assessment"] = self._assess_weight_stability(overall_norms)

        # Change detection
        if len(overall_norms) > 10:
            analysis["significant_changes"] = self._detect_significant_changes(steps, overall_norms)

        return analysis

    def analyze_layer_weights(self) -> Dict[str, Any]:
        """Analyze weights at the layer level.

        Returns:
            Layer-wise weight analysis
        """
        if not self.weight_history:
            return {"status": "no_data"}

        # Collect layer data across time
        layer_evolution = {}

        for entry in self.weight_history:
            step = entry["step"]
            layer_norms = entry.get("layer_norms", {})

            for layer_name, norm in layer_norms.items():
                if layer_name not in layer_evolution:
                    layer_evolution[layer_name] = {"steps": [], "norms": []}

                layer_evolution[layer_name]["steps"].append(step)
                layer_evolution[layer_name]["norms"].append(norm)

        # Analyze each layer
        layer_analysis = {}

        for layer_name, data in layer_evolution.items():
            norms = np.array(data["norms"])

            layer_stats = {
                "initial_norm": float(norms[0]) if len(norms) > 0 else 0.0,
                "final_norm": float(norms[-1]) if len(norms) > 0 else 0.0,
                "max_norm": float(np.max(norms)),
                "min_norm": float(np.min(norms)),
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "coefficient_of_variation": float(np.std(norms) / np.mean(norms)) if np.mean(norms) > 0 else 0.0,
            }

            # Layer-specific assessments
            if len(norms) > 5:
                layer_stats["trend"] = self._analyze_trend(norms)
                layer_stats["stability"] = self._assess_layer_stability(norms)

            layer_analysis[layer_name] = layer_stats

        # Cross-layer analysis
        analysis = {
            "layer_analysis": layer_analysis,
            "cross_layer_insights": self._analyze_cross_layer_patterns(layer_analysis),
        }

        return analysis

    def detect_weight_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in weight behavior.

        Returns:
            Detected weight anomalies
        """
        anomalies = {
            "detected_anomalies": [],
            "anomaly_count": 0,
            "severity_score": 0.0,
        }

        if not self.weight_history:
            return anomalies

        # Extract norm sequence
        overall_norms = np.array([entry["overall_norm"] for entry in self.weight_history])

        # Detect sudden jumps
        jumps = self._detect_sudden_jumps(overall_norms)
        anomalies["detected_anomalies"].extend(jumps)

        # Detect plateaus
        plateaus = self._detect_plateaus(overall_norms)
        anomalies["detected_anomalies"].extend(plateaus)

        # Detect extreme values
        outliers = self._detect_outliers(overall_norms)
        anomalies["detected_anomalies"].extend(outliers)

        # Layer-specific anomalies
        layer_anomalies = self._detect_layer_anomalies()
        anomalies["detected_anomalies"].extend(layer_anomalies)

        anomalies["anomaly_count"] = len(anomalies["detected_anomalies"])
        anomalies["severity_score"] = self._calculate_anomaly_severity(anomalies["detected_anomalies"])

        return anomalies

    def generate_weight_report(self) -> Dict[str, Any]:
        """Generate comprehensive weight analysis report.

        Returns:
            Complete weight analysis report
        """
        report = {
            "evolution_analysis": self.analyze_weight_evolution(),
            "layer_analysis": self.analyze_layer_weights(),
            "anomaly_detection": self.detect_weight_anomalies(),
        }

        # Overall assessment
        report["overall_assessment"] = self._generate_overall_assessment(report)

        return report

    def visualize_weight_evolution(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create visualizations of weight evolution.

        Args:
            save_path: Optional path to save visualizations

        Returns:
            Dictionary with plot information
        """
        plots_created = {}

        if not self.weight_history:
            return plots_created

        # Overall weight evolution plot
        fig1 = self._plot_overall_evolution()
        if save_path:
            fig1_path = save_path / "weight_evolution.png"
            fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
            plots_created["weight_evolution"] = fig1_path
        plt.close(fig1)

        # Layer-wise evolution plot
        fig2 = self._plot_layer_evolution()
        if save_path:
            fig2_path = save_path / "layer_weight_evolution.png"
            fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
            plots_created["layer_evolution"] = fig2_path
        plt.close(fig2)

        # Weight distribution plot
        fig3 = self._plot_weight_distributions()
        if save_path:
            fig3_path = save_path / "weight_distributions.png"
            fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
            plots_created["weight_distributions"] = fig3_path
        plt.close(fig3)

        return plots_created

    def _analyze_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in a sequence of values."""
        if len(values) < 2:
            return {"status": "insufficient_data"}

        x = np.arange(len(values))
        trend_coef = np.polyfit(x, values, 1)[0]

        # Calculate R-squared
        y_pred = np.polyval([trend_coef, values[0]], x)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "trend_coefficient": float(trend_coef),
            "trend_direction": "increasing" if trend_coef > 0 else "decreasing",
            "trend_strength": abs(float(trend_coef)),
            "r_squared": float(r_squared),
            "linear_fit_quality": "good" if r_squared > 0.8 else "moderate" if r_squared > 0.5 else "poor",
            "initial_value": float(values[0]),
            "final_value": float(values[-1]),
            "total_change": float(values[-1] - values[0]),
            "percent_change": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0,
        }

    def _assess_weight_stability(self, norms: List[float]) -> Dict[str, Any]:
        """Assess overall weight stability."""
        if not norms:
            return {"status": "no_data"}

        norms_array = np.array(norms)
        mean_norm = np.mean(norms_array)
        std_norm = np.std(norms_array)

        coefficient_of_variation = std_norm / mean_norm if mean_norm > 0 else float("inf")

        # Stability levels
        if coefficient_of_variation < 0.01:
            stability_level = "very_stable"
        elif coefficient_of_variation < 0.05:
            stability_level = "stable"
        elif coefficient_of_variation < 0.1:
            stability_level = "moderate"
        else:
            stability_level = "unstable"

        return {
            "stability_level": stability_level,
            "coefficient_of_variation": float(coefficient_of_variation),
            "relative_std": float(std_norm / mean_norm) if mean_norm > 0 else 0.0,
            "stability_score": max(0.0, 1.0 - coefficient_of_variation),
        }

    def _detect_significant_changes(self, steps: List[int], norms: List[float]) -> List[Dict[str, Any]]:
        """Detect significant changes in weight norms."""
        if len(norms) < 3:
            return []

        changes = []
        norms_array = np.array(norms)

        # Calculate rolling statistics
        window_size = min(5, len(norms) // 3)

        for i in range(window_size, len(norms) - window_size):
            before_window = norms_array[i - window_size : i]
            after_window = norms_array[i : i + window_size]

            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)

            # Detect significant change (more than 2 standard deviations)
            combined_std = np.std(np.concatenate([before_window, after_window]))
            change_magnitude = abs(after_mean - before_mean)

            if change_magnitude > 2 * combined_std and combined_std > 0:
                changes.append(
                    {
                        "step": steps[i],
                        "change_type": "increase" if after_mean > before_mean else "decrease",
                        "magnitude": float(change_magnitude),
                        "relative_magnitude": float(change_magnitude / before_mean) if before_mean > 0 else 0.0,
                        "significance": float(change_magnitude / combined_std),
                    }
                )

        return changes

    def _assess_layer_stability(self, norms: np.ndarray) -> str:
        """Assess stability for a specific layer."""
        if len(norms) == 0:
            return "no_data"

        coefficient_of_variation = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else float("inf")

        if coefficient_of_variation < 0.02:
            return "very_stable"
        elif coefficient_of_variation < 0.05:
            return "stable"
        elif coefficient_of_variation < 0.1:
            return "moderate"
        else:
            return "unstable"

    def _analyze_cross_layer_patterns(self, layer_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across layers."""
        if not layer_analysis:
            return {"status": "no_data"}

        # Collect statistics across layers
        final_norms = []
        stability_scores = []
        trends = []

        for layer_name, stats in layer_analysis.items():
            final_norms.append(stats.get("final_norm", 0))

            if "trend" in stats:
                trends.append(stats["trend"].get("trend_coefficient", 0))

            if "stability" in stats:
                stability_scores.append(
                    1.0
                    if stats["stability"] == "very_stable"
                    else 0.8 if stats["stability"] == "stable" else 0.5 if stats["stability"] == "moderate" else 0.2
                )

        analysis = {
            "layer_count": len(layer_analysis),
            "norm_distribution": {
                "mean_final_norm": float(np.mean(final_norms)) if final_norms else 0.0,
                "std_final_norm": float(np.std(final_norms)) if final_norms else 0.0,
                "norm_range": float(np.max(final_norms) - np.min(final_norms)) if final_norms else 0.0,
            },
            "stability_summary": {
                "mean_stability_score": float(np.mean(stability_scores)) if stability_scores else 0.0,
                "unstable_layer_count": sum(
                    1 for _, stats in layer_analysis.items() if stats.get("stability") == "unstable"
                ),
            },
            "trend_summary": {
                "mean_trend": float(np.mean(trends)) if trends else 0.0,
                "increasing_layers": sum(1 for t in trends if t > 0.001),
                "decreasing_layers": sum(1 for t in trends if t < -0.001),
            },
        }

        return analysis

    def _detect_sudden_jumps(self, norms: np.ndarray) -> List[Dict[str, Any]]:
        """Detect sudden jumps in weight norms."""
        jumps = []

        if len(norms) < 3:
            return jumps

        # Calculate differences
        diffs = np.diff(norms)
        diff_std = np.std(diffs)
        diff_mean = np.mean(diffs)

        # Detect outlier differences
        threshold = 3 * diff_std

        for i, diff in enumerate(diffs):
            if abs(diff - diff_mean) > threshold:
                jumps.append(
                    {
                        "type": "sudden_jump",
                        "step_index": i + 1,
                        "magnitude": float(abs(diff)),
                        "direction": "increase" if diff > 0 else "decrease",
                        "severity": min(1.0, abs(diff - diff_mean) / threshold),
                    }
                )

        return jumps

    def _detect_plateaus(self, norms: np.ndarray) -> List[Dict[str, Any]]:
        """Detect plateau regions in weight norms."""
        plateaus = []

        if len(norms) < 10:
            return plateaus

        window_size = 5
        plateau_threshold = 0.001  # Relative change threshold

        for i in range(len(norms) - window_size + 1):
            window = norms[i : i + window_size]
            relative_change = (np.max(window) - np.min(window)) / np.mean(window) if np.mean(window) > 0 else 0

            if relative_change < plateau_threshold:
                plateaus.append(
                    {
                        "type": "plateau",
                        "start_index": i,
                        "end_index": i + window_size - 1,
                        "length": window_size,
                        "relative_change": float(relative_change),
                        "severity": 1.0 - relative_change / plateau_threshold,
                    }
                )

        return plateaus

    def _detect_outliers(self, norms: np.ndarray) -> List[Dict[str, Any]]:
        """Detect outlier values in weight norms."""
        outliers = []

        if len(norms) < 5:
            return outliers

        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        threshold = 3 * std_norm

        for i, norm in enumerate(norms):
            if abs(norm - mean_norm) > threshold:
                outliers.append(
                    {
                        "type": "outlier",
                        "step_index": i,
                        "value": float(norm),
                        "deviation": float(abs(norm - mean_norm)),
                        "severity": min(1.0, abs(norm - mean_norm) / threshold),
                    }
                )

        return outliers

    def _detect_layer_anomalies(self) -> List[Dict[str, Any]]:
        """Detect layer-specific anomalies."""
        anomalies = []

        if not self.weight_history:
            return anomalies

        # Check for layers with extreme behaviors
        layer_evolution = {}

        for entry in self.weight_history:
            layer_norms = entry.get("layer_norms", {})
            for layer_name, norm in layer_norms.items():
                if layer_name not in layer_evolution:
                    layer_evolution[layer_name] = []
                layer_evolution[layer_name].append(norm)

        # Analyze each layer for anomalies
        for layer_name, norms in layer_evolution.items():
            if len(norms) < 3:
                continue

            norms_array = np.array(norms)

            # Check for extreme growth/decay
            final_ratio = norms_array[-1] / norms_array[0] if norms_array[0] > 0 else 0

            if final_ratio > 10 or final_ratio < 0.1:
                anomalies.append(
                    {
                        "type": "layer_extreme_change",
                        "layer": layer_name,
                        "ratio": float(final_ratio),
                        "change_type": "extreme_growth" if final_ratio > 10 else "extreme_decay",
                        "severity": min(1.0, abs(np.log10(final_ratio)) / 2),
                    }
                )

        return anomalies

    def _calculate_anomaly_severity(self, anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall anomaly severity score."""
        if not anomalies:
            return 0.0

        total_severity = sum(anomaly.get("severity", 0.0) for anomaly in anomalies)
        return min(1.0, total_severity / len(anomalies))

    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall weight health assessment."""
        assessment = {
            "weight_health": "unknown",
            "key_issues": [],
            "recommendations": [],
        }

        # Analyze evolution
        evolution = report.get("evolution_analysis", {})
        stability = evolution.get("stability_assessment", {})

        if stability.get("stability_level") in ["unstable", "moderate"]:
            assessment["key_issues"].append("weight_instability")
            assessment["recommendations"].append("Consider adjusting learning rate or adding weight decay")

        # Analyze layer patterns
        layer_analysis = report.get("layer_analysis", {})
        cross_layer = layer_analysis.get("cross_layer_insights", {})

        if cross_layer.get("stability_summary", {}).get("unstable_layer_count", 0) > 0:
            assessment["key_issues"].append("layer_instability")
            assessment["recommendations"].append("Review layer-specific learning rates or initialization")

        # Analyze anomalies
        anomalies = report.get("anomaly_detection", {})
        if anomalies.get("severity_score", 0) > 0.5:
            assessment["key_issues"].append("weight_anomalies")
            assessment["recommendations"].append("Investigate training instabilities")

        # Overall assessment
        if not assessment["key_issues"]:
            assessment["weight_health"] = "healthy"
        elif len(assessment["key_issues"]) <= 2:
            assessment["weight_health"] = "moderate"
        else:
            assessment["weight_health"] = "poor"

        return assessment

    def _plot_overall_evolution(self):
        """Create overall weight evolution plot."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        steps = [entry["step"] for entry in self.weight_history]
        norms = [entry["overall_norm"] for entry in self.weight_history]
        means = [entry["overall_mean"] for entry in self.weight_history]
        stds = [entry["overall_std"] for entry in self.weight_history]

        # Plot norms
        ax1.plot(steps, norms, "b-", linewidth=2, marker="o", markersize=3)
        ax1.set_ylabel("Weight Norm")
        ax1.set_title("Overall Weight Evolution")
        ax1.grid(True, alpha=0.3)

        # Plot means
        ax2.plot(steps, means, "g-", linewidth=2, marker="s", markersize=3)
        ax2.set_ylabel("Weight Mean")
        ax2.grid(True, alpha=0.3)

        # Plot stds
        ax3.plot(steps, stds, "r-", linewidth=2, marker="^", markersize=3)
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Weight Std")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_layer_evolution(self):
        """Create layer-wise evolution plot."""
        if not self.weight_history:
            return None

        # Collect layer data
        layer_data = {}
        for entry in self.weight_history:
            step = entry["step"]
            layer_norms = entry.get("layer_norms", {})

            for layer_name, norm in layer_norms.items():
                if layer_name not in layer_data:
                    layer_data[layer_name] = {"steps": [], "norms": []}
                layer_data[layer_name]["steps"].append(step)
                layer_data[layer_name]["norms"].append(norm)

        if not layer_data:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))

        for layer_name, data in layer_data.items():
            ax.plot(data["steps"], data["norms"], label=layer_name.split(".")[-1], alpha=0.7, linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Layer Weight Norm")
        ax.set_title("Layer-wise Weight Evolution")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig

    def _plot_weight_distributions(self):
        """Create weight distribution plots."""
        if len(self.weight_history) < 2:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = [entry["step"] for entry in self.weight_history]

        # Plot norm distribution over time
        norms = [entry["overall_norm"] for entry in self.weight_history]
        axes[0, 0].hist(norms, bins=20, alpha=0.7, color="blue")
        axes[0, 0].set_xlabel("Weight Norm")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Weight Norms")

        # Plot mean distribution
        means = [entry["overall_mean"] for entry in self.weight_history]
        axes[0, 1].hist(means, bins=20, alpha=0.7, color="green")
        axes[0, 1].set_xlabel("Weight Mean")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Weight Means")

        # Plot std distribution
        stds = [entry["overall_std"] for entry in self.weight_history]
        axes[1, 0].hist(stds, bins=20, alpha=0.7, color="red")
        axes[1, 0].set_xlabel("Weight Std")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Weight Stds")

        # Plot norm vs step scatter
        axes[1, 1].scatter(steps, norms, alpha=0.6, color="purple")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Weight Norm")
        axes[1, 1].set_title("Weight Norm vs Training Step")

        plt.tight_layout()
        return fig
