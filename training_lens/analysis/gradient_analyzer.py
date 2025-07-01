"""Gradient analysis for training insights."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class GradientAnalyzer:
    """Specialized analyzer for gradient-related insights."""

    def __init__(self, gradient_data: Optional[Dict[str, Any]] = None):
        """Initialize gradient analyzer.

        Args:
            gradient_data: Optional gradient data dictionary
        """
        self.gradient_data = gradient_data or {}
        self.cosine_similarities = []
        self.gradient_norms = []
        self.layer_gradients = {}

        if gradient_data:
            self._extract_gradient_info()

    def _extract_gradient_info(self) -> None:
        """Extract gradient information from data."""
        if "gradient_cosine_similarities" in self.gradient_data:
            self.cosine_similarities = self.gradient_data["gradient_cosine_similarities"]

        if "gradient_norms" in self.gradient_data:
            self.gradient_norms = self.gradient_data["gradient_norms"]

        if "layer_gradients" in self.gradient_data:
            self.layer_gradients = self.gradient_data["layer_gradients"]

    def analyze_gradient_consistency(self) -> Dict[str, Any]:
        """Analyze gradient direction consistency over training.

        Returns:
            Analysis of gradient consistency
        """
        if not self.cosine_similarities:
            return {"status": "no_data"}

        similarities = np.array(self.cosine_similarities)

        # Basic statistics
        analysis = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "median_similarity": float(np.median(similarities)),
        }

        # Consistency assessment
        analysis["consistency_score"] = self._calculate_consistency_score(similarities)
        analysis["consistency_level"] = self._assess_consistency_level(analysis["consistency_score"])

        # Trend analysis
        if len(similarities) > 10:
            trend = self._analyze_similarity_trend(similarities)
            analysis["trend_analysis"] = trend

        # Stability windows
        stability_windows = self._find_stability_windows(similarities)
        analysis["stability_windows"] = stability_windows

        return analysis

    def analyze_gradient_magnitude_evolution(self) -> Dict[str, Any]:
        """Analyze how gradient magnitudes evolve.

        Returns:
            Analysis of gradient magnitude evolution
        """
        if not self.gradient_norms:
            return {"status": "no_data"}

        norms = np.array(self.gradient_norms)

        analysis = {
            "initial_norm": float(norms[0]) if len(norms) > 0 else 0.0,
            "final_norm": float(norms[-1]) if len(norms) > 0 else 0.0,
            "max_norm": float(np.max(norms)),
            "min_norm": float(np.min(norms)),
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
        }

        # Gradient explosion/vanishing detection
        analysis["explosion_risk"] = self._detect_gradient_explosion(norms)
        analysis["vanishing_risk"] = self._detect_gradient_vanishing(norms)

        # Norm evolution trend
        if len(norms) > 5:
            trend = self._analyze_norm_trend(norms)
            analysis["trend_analysis"] = trend

        return analysis

    def analyze_layer_wise_gradients(self) -> Dict[str, Any]:
        """Analyze gradients at the layer level.

        Returns:
            Layer-wise gradient analysis
        """
        if not self.layer_gradients:
            return {"status": "no_data"}

        layer_analysis = {}

        for layer_name, layer_grad_history in self.layer_gradients.items():
            if not layer_grad_history:
                continue

            layer_norms = np.array(layer_grad_history)

            layer_stats = {
                "mean_norm": float(np.mean(layer_norms)),
                "std_norm": float(np.std(layer_norms)),
                "max_norm": float(np.max(layer_norms)),
                "min_norm": float(np.min(layer_norms)),
                "coefficient_of_variation": (
                    float(np.std(layer_norms) / np.mean(layer_norms)) if np.mean(layer_norms) > 0 else 0.0
                ),
            }

            # Layer-specific issues
            layer_stats["gradient_flow_quality"] = self._assess_gradient_flow_quality(layer_norms)

            layer_analysis[layer_name] = layer_stats

        # Cross-layer analysis
        analysis = {
            "layer_analysis": layer_analysis,
            "gradient_flow_summary": self._analyze_cross_layer_flow(layer_analysis),
        }

        return analysis

    def detect_gradient_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in gradient behavior.

        Returns:
            Detected gradient anomalies
        """
        anomalies = {
            "detected_anomalies": [],
            "anomaly_count": 0,
            "severity_score": 0.0,
        }

        # Cosine similarity anomalies
        if self.cosine_similarities:
            similarity_anomalies = self._detect_similarity_anomalies(self.cosine_similarities)
            anomalies["detected_anomalies"].extend(similarity_anomalies)

        # Gradient norm anomalies
        if self.gradient_norms:
            norm_anomalies = self._detect_norm_anomalies(self.gradient_norms)
            anomalies["detected_anomalies"].extend(norm_anomalies)

        # Layer-wise anomalies
        if self.layer_gradients:
            layer_anomalies = self._detect_layer_anomalies(self.layer_gradients)
            anomalies["detected_anomalies"].extend(layer_anomalies)

        anomalies["anomaly_count"] = len(anomalies["detected_anomalies"])
        anomalies["severity_score"] = self._calculate_anomaly_severity(anomalies["detected_anomalies"])

        return anomalies

    def generate_gradient_report(self) -> Dict[str, Any]:
        """Generate comprehensive gradient analysis report.

        Returns:
            Complete gradient analysis report
        """
        report = {
            "consistency_analysis": self.analyze_gradient_consistency(),
            "magnitude_analysis": self.analyze_gradient_magnitude_evolution(),
            "layer_analysis": self.analyze_layer_wise_gradients(),
            "anomaly_detection": self.detect_gradient_anomalies(),
        }

        # Overall assessment
        report["overall_assessment"] = self._generate_overall_assessment(report)

        return report

    def visualize_gradient_evolution(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create visualizations of gradient evolution.

        Args:
            save_path: Optional path to save visualizations

        Returns:
            Dictionary with plot information
        """
        plots_created = {}

        if self.cosine_similarities:
            fig1 = self._plot_cosine_similarities()
            if save_path:
                fig1_path = save_path / "gradient_cosine_similarities.png"
                fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
                plots_created["cosine_similarities"] = fig1_path
            plt.close(fig1)

        if self.gradient_norms:
            fig2 = self._plot_gradient_norms()
            if save_path:
                fig2_path = save_path / "gradient_norms.png"
                fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
                plots_created["gradient_norms"] = fig2_path
            plt.close(fig2)

        if self.layer_gradients:
            fig3 = self._plot_layer_gradients()
            if save_path:
                fig3_path = save_path / "layer_gradient_norms.png"
                fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
                plots_created["layer_gradients"] = fig3_path
            plt.close(fig3)

        return plots_created

    def _calculate_consistency_score(self, similarities: np.ndarray) -> float:
        """Calculate a consistency score from cosine similarities."""
        # Score based on mean similarity and stability (low variance)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Penalize high variance
        consistency_score = mean_sim - (std_sim * 0.5)

        # Normalize to 0-1 range
        return max(0.0, min(1.0, (consistency_score + 1) / 2))

    def _assess_consistency_level(self, score: float) -> str:
        """Assess consistency level from score."""
        if score > 0.8:
            return "very_consistent"
        elif score > 0.6:
            return "consistent"
        elif score > 0.4:
            return "moderately_consistent"
        else:
            return "inconsistent"

    def _analyze_similarity_trend(self, similarities: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in cosine similarities."""
        x = np.arange(len(similarities))

        # Linear trend
        trend_coef = np.polyfit(x, similarities, 1)[0]

        # Moving average trend
        window_size = min(10, len(similarities) // 4)
        if window_size > 1:
            moving_avg = np.convolve(similarities, np.ones(window_size) / window_size, mode="valid")
            recent_trend = moving_avg[-1] - moving_avg[0] if len(moving_avg) > 1 else 0
        else:
            recent_trend = 0

        return {
            "linear_trend_coefficient": float(trend_coef),
            "trend_direction": (
                "increasing" if trend_coef > 0.001 else "decreasing" if trend_coef < -0.001 else "stable"
            ),
            "recent_trend": float(recent_trend),
            "trend_strength": abs(float(trend_coef)),
        }

    def _find_stability_windows(self, similarities: np.ndarray, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Find windows of stable gradient behavior."""
        if len(similarities) < 10:
            return []

        windows = []
        window_size = 10

        for i in range(len(similarities) - window_size + 1):
            window = similarities[i : i + window_size]
            window_std = np.std(window)

            if window_std < threshold:
                windows.append(
                    {
                        "start_step": i,
                        "end_step": i + window_size - 1,
                        "stability_score": float(1.0 / (window_std + 1e-8)),
                        "mean_similarity": float(np.mean(window)),
                    }
                )

        return windows

    def _detect_gradient_explosion(self, norms: np.ndarray) -> Dict[str, Any]:
        """Detect gradient explosion patterns."""
        max_norm = np.max(norms)
        mean_norm = np.mean(norms)

        # Check for sudden spikes
        spikes = []
        for i in range(1, len(norms)):
            if norms[i] > 3 * norms[i - 1] and norms[i] > 5.0:
                spikes.append({"step": i, "norm": float(norms[i])})

        explosion_risk = (
            "high" if max_norm > 10.0 or len(spikes) > 3 else "medium" if max_norm > 5.0 or len(spikes) > 1 else "low"
        )

        return {
            "risk_level": explosion_risk,
            "max_norm": float(max_norm),
            "spike_count": len(spikes),
            "spikes": spikes,
            "norm_ratio": float(max_norm / mean_norm) if mean_norm > 0 else 0.0,
        }

    def _detect_gradient_vanishing(self, norms: np.ndarray) -> Dict[str, Any]:
        """Detect gradient vanishing patterns."""
        min_norm = np.min(norms)
        mean_norm = np.mean(norms)

        # Check for very small gradients
        vanishing_steps = [i for i, norm in enumerate(norms) if norm < 1e-6]

        # Check for decreasing trend
        if len(norms) > 10:
            recent_norms = norms[-10:]
            trend_coef = np.polyfit(range(len(recent_norms)), recent_norms, 1)[0]
            decreasing_trend = trend_coef < -1e-4
        else:
            decreasing_trend = False

        vanishing_risk = (
            "high"
            if min_norm < 1e-7 or len(vanishing_steps) > len(norms) * 0.1
            else "medium" if min_norm < 1e-5 or decreasing_trend else "low"
        )

        return {
            "risk_level": vanishing_risk,
            "min_norm": float(min_norm),
            "vanishing_step_count": len(vanishing_steps),
            "decreasing_trend": decreasing_trend,
            "norm_ratio": float(min_norm / mean_norm) if mean_norm > 0 else 0.0,
        }

    def _analyze_norm_trend(self, norms: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in gradient norms."""
        x = np.arange(len(norms))
        trend_coef = np.polyfit(x, norms, 1)[0]

        return {
            "trend_coefficient": float(trend_coef),
            "trend_direction": "increasing" if trend_coef > 0 else "decreasing",
            "initial_norm": float(norms[0]),
            "final_norm": float(norms[-1]),
            "change_ratio": float(norms[-1] / norms[0]) if norms[0] > 0 else 0.0,
        }

    def _assess_gradient_flow_quality(self, layer_norms: np.ndarray) -> str:
        """Assess gradient flow quality for a layer."""
        if len(layer_norms) == 0:
            return "no_data"

        mean_norm = np.mean(layer_norms)
        std_norm = np.std(layer_norms)
        coefficient_of_variation = std_norm / mean_norm if mean_norm > 0 else float("inf")

        # Good flow: reasonable magnitude, low variation
        if mean_norm > 1e-6 and mean_norm < 10.0 and coefficient_of_variation < 0.5:
            return "good"
        elif mean_norm < 1e-7:
            return "vanishing"
        elif mean_norm > 10.0:
            return "exploding"
        elif coefficient_of_variation > 1.0:
            return "unstable"
        else:
            return "moderate"

    def _analyze_cross_layer_flow(self, layer_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gradient flow across layers."""
        if not layer_analysis:
            return {"status": "no_data"}

        mean_norms = [stats["mean_norm"] for stats in layer_analysis.values()]

        return {
            "overall_flow_quality": self._assess_overall_flow_quality(layer_analysis),
            "norm_variance_across_layers": float(np.var(mean_norms)),
            "problematic_layers": [
                name
                for name, stats in layer_analysis.items()
                if stats["gradient_flow_quality"] in ["vanishing", "exploding"]
            ],
            "layer_count": len(layer_analysis),
        }

    def _assess_overall_flow_quality(self, layer_analysis: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall gradient flow quality."""
        if not layer_analysis:
            return "no_data"

        quality_counts = {}
        for stats in layer_analysis.values():
            quality = stats["gradient_flow_quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        total_layers = len(layer_analysis)
        good_ratio = quality_counts.get("good", 0) / total_layers

        if good_ratio > 0.8:
            return "excellent"
        elif good_ratio > 0.6:
            return "good"
        elif good_ratio > 0.4:
            return "moderate"
        else:
            return "poor"

    def _detect_similarity_anomalies(self, similarities: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in cosine similarities."""
        if len(similarities) < 10:
            return []

        similarities = np.array(similarities)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        anomalies = []
        threshold = 3 * std_sim  # 3-sigma rule

        for i, sim in enumerate(similarities):
            if abs(sim - mean_sim) > threshold:
                anomalies.append(
                    {
                        "type": "cosine_similarity_outlier",
                        "step": i,
                        "value": float(sim),
                        "severity": min(1.0, abs(sim - mean_sim) / threshold),
                    }
                )

        return anomalies

    def _detect_norm_anomalies(self, norms: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in gradient norms."""
        if len(norms) < 10:
            return []

        norms = np.array(norms)
        log_norms = np.log(norms + 1e-8)  # Log scale for better outlier detection
        mean_log = np.mean(log_norms)
        std_log = np.std(log_norms)

        anomalies = []
        threshold = 3 * std_log

        for i, log_norm in enumerate(log_norms):
            if abs(log_norm - mean_log) > threshold:
                anomalies.append(
                    {
                        "type": "gradient_norm_outlier",
                        "step": i,
                        "value": float(norms[i]),
                        "severity": min(1.0, abs(log_norm - mean_log) / threshold),
                    }
                )

        return anomalies

    def _detect_layer_anomalies(self, layer_gradients: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect layer-specific gradient anomalies."""
        anomalies = []

        for layer_name, grad_history in layer_gradients.items():
            if len(grad_history) < 5:
                continue

            layer_norms = np.array(grad_history)

            # Check for sudden changes
            for i in range(1, len(layer_norms)):
                ratio = layer_norms[i] / (layer_norms[i - 1] + 1e-8)
                if ratio > 10 or ratio < 0.1:
                    anomalies.append(
                        {
                            "type": "layer_gradient_sudden_change",
                            "layer": layer_name,
                            "step": i,
                            "ratio": float(ratio),
                            "severity": min(1.0, abs(np.log(ratio)) / 2.3),  # log(10) = 2.3
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
        """Generate overall gradient health assessment."""
        assessment = {
            "gradient_health": "unknown",
            "key_issues": [],
            "recommendations": [],
        }

        # Analyze consistency
        consistency = report.get("consistency_analysis", {})
        if consistency.get("consistency_level") in ["inconsistent", "moderately_consistent"]:
            assessment["key_issues"].append("gradient_inconsistency")
            assessment["recommendations"].append("Consider reducing learning rate or adding gradient clipping")

        # Analyze magnitude issues
        magnitude = report.get("magnitude_analysis", {})
        if magnitude.get("explosion_risk", {}).get("risk_level") == "high":
            assessment["key_issues"].append("gradient_explosion")
            assessment["recommendations"].append("Implement gradient clipping")

        if magnitude.get("vanishing_risk", {}).get("risk_level") == "high":
            assessment["key_issues"].append("gradient_vanishing")
            assessment["recommendations"].append("Consider batch normalization or residual connections")

        # Analyze layer flow
        layer_analysis = report.get("layer_analysis", {})
        flow_summary = layer_analysis.get("gradient_flow_summary", {})
        if flow_summary.get("overall_flow_quality") in ["poor", "moderate"]:
            assessment["key_issues"].append("poor_gradient_flow")
            assessment["recommendations"].append("Review model architecture for gradient flow bottlenecks")

        # Analyze anomalies
        anomalies = report.get("anomaly_detection", {})
        if anomalies.get("severity_score", 0) > 0.5:
            assessment["key_issues"].append("gradient_anomalies")
            assessment["recommendations"].append("Investigate training instabilities")

        # Overall health assessment
        if not assessment["key_issues"]:
            assessment["gradient_health"] = "healthy"
        elif len(assessment["key_issues"]) <= 2:
            assessment["gradient_health"] = "moderate"
        else:
            assessment["gradient_health"] = "poor"

        return assessment

    def _plot_cosine_similarities(self):
        """Create cosine similarity plot."""
        fig, ax = plt.subplots(figsize=(12, 6))

        similarities = self.cosine_similarities
        ax.plot(similarities, linewidth=2, alpha=0.8, color="blue")

        # Add trend line
        if len(similarities) > 10:
            x = np.arange(len(similarities))
            z = np.polyfit(x, similarities, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=1, label="Trend")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Cosine Similarity")
        ax.set_title("Gradient Direction Consistency Over Training")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        return fig

    def _plot_gradient_norms(self):
        """Create gradient norms plot."""
        fig, ax = plt.subplots(figsize=(12, 6))

        norms = self.gradient_norms
        ax.plot(norms, linewidth=2, alpha=0.8, color="green")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Magnitude Evolution")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        return fig

    def _plot_layer_gradients(self):
        """Create layer-wise gradient plot."""
        if not self.layer_gradients:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))

        for layer_name, grad_history in self.layer_gradients.items():
            if grad_history:
                ax.plot(grad_history, label=layer_name.split(".")[-1], alpha=0.7)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Layer-wise Gradient Evolution")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig
