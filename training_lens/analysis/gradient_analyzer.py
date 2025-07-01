"""Gradient analysis for training insights."""

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..utils.logging import get_logger

logger = get_logger(__name__)


class GradientAnalysisResult(NamedTuple):
    """Structured result for gradient analysis."""

    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    median_similarity: float
    consistency_score: float
    consistency_level: str


class GradientAnomaly(NamedTuple):
    """Structured anomaly detection result."""

    type: str
    step: int
    value: float
    severity: float


class GradientAnalyzer:
    """Specialized analyzer for gradient-related insights."""

    def __init__(self, gradient_data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize gradient analyzer.

        Args:
            gradient_data: Optional gradient data dictionary
        """
        self.gradient_data = gradient_data or {}
        self.cosine_similarities: List[float] = []
        self.gradient_norms: List[float] = []
        self.layer_gradients: Dict[str, List[float]] = {}

        if gradient_data:
            try:
                self._extract_gradient_info()
            except Exception as e:
                logger.error(f"Failed to extract gradient info: {e}")
                # Initialize with empty data on error
                self.cosine_similarities = []
                self.gradient_norms = []
                self.layer_gradients = {}

    def _extract_gradient_info(self) -> None:
        """Extract gradient information from data with error handling."""
        try:
            if "gradient_cosine_similarities" in self.gradient_data:
                similarities = self.gradient_data["gradient_cosine_similarities"]
                if isinstance(similarities, list):
                    self.cosine_similarities = [float(s) for s in similarities if s is not None]
                else:
                    logger.warning("gradient_cosine_similarities is not a list")
                    self.cosine_similarities = []

            if "gradient_norms" in self.gradient_data:
                norms = self.gradient_data["gradient_norms"]
                if isinstance(norms, list):
                    self.gradient_norms = [float(n) for n in norms if n is not None]
                else:
                    logger.warning("gradient_norms is not a list")
                    self.gradient_norms = []

            if "layer_gradients" in self.gradient_data:
                layer_data = self.gradient_data["layer_gradients"]
                if isinstance(layer_data, dict):
                    self.layer_gradients = {}
                    for layer_name, grad_history in layer_data.items():
                        if isinstance(grad_history, list):
                            self.layer_gradients[layer_name] = [float(g) for g in grad_history if g is not None]
                        else:
                            logger.warning(f"Layer {layer_name} gradients is not a list")
                else:
                    logger.warning("layer_gradients is not a dictionary")
                    self.layer_gradients = {}

        except Exception as e:
            logger.error(f"Error extracting gradient info: {e}")
            raise ValueError(f"Invalid gradient data format: {e}")

    def analyze_gradient_consistency(self) -> Dict[str, Any]:
        """Analyze gradient direction consistency over training.

        Returns:
            Analysis of gradient consistency with error handling
        """
        try:
            if not self.cosine_similarities:
                return {"status": "no_data", "error": "No cosine similarity data available"}

            similarities = np.array(self.cosine_similarities)

            # Validate data
            if np.any(np.isnan(similarities)) or np.any(np.isinf(similarities)):
                logger.warning("Found NaN or infinite values in cosine similarities, cleaning data")
                similarities = similarities[~np.isnan(similarities) & ~np.isinf(similarities)]

            if len(similarities) == 0:
                return {"status": "no_valid_data", "error": "No valid cosine similarity data after cleaning"}

            # Basic statistics with error handling
            analysis: Dict[str, Any] = {
                "mean_similarity": float(np.mean(similarities)),
                "std_similarity": float(np.std(similarities)),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
                "median_similarity": float(np.median(similarities)),
            }

            # Consistency assessment
            analysis["consistency_score"] = self._calculate_consistency_score(similarities)
            analysis["consistency_level"] = self._assess_consistency_level(analysis["consistency_score"])

            # Trend analysis (only if enough data)
            if len(similarities) > 10:
                try:
                    trend = self._analyze_similarity_trend(similarities)
                    analysis["trend_analysis"] = trend
                except Exception as e:
                    logger.warning(f"Failed to analyze similarity trend: {e}")
                    analysis["trend_analysis"] = {"error": str(e)}

            # Stability windows
            try:
                stability_windows = self._find_stability_windows(similarities)
                analysis["stability_windows"] = stability_windows
            except Exception as e:
                logger.warning(f"Failed to find stability windows: {e}")
                analysis["stability_windows"] = []

            return analysis

        except Exception as e:
            logger.error(f"Error in gradient consistency analysis: {e}")
            return {"status": "error", "error": str(e), "traceback": str(e.__class__.__name__)}

    def analyze_gradient_magnitude_evolution(self) -> Dict[str, Any]:
        """Analyze how gradient magnitudes evolve.

        Returns:
            Analysis of gradient magnitude evolution with error handling
        """
        try:
            if not self.gradient_norms:
                return {"status": "no_data", "error": "No gradient norm data available"}

            norms = np.array(self.gradient_norms)

            # Validate and clean data
            if np.any(np.isnan(norms)) or np.any(np.isinf(norms)):
                logger.warning("Found NaN or infinite values in gradient norms, cleaning data")
                norms = norms[~np.isnan(norms) & ~np.isinf(norms)]

            if len(norms) == 0:
                return {"status": "no_valid_data", "error": "No valid gradient norm data after cleaning"}

            # Ensure all norms are positive
            if np.any(norms <= 0):
                logger.warning("Found non-positive gradient norms, filtering out")
                norms = norms[norms > 0]

            if len(norms) == 0:
                return {"status": "no_valid_data", "error": "No positive gradient norms found"}

            analysis: Dict[str, Any] = {
                "initial_norm": float(norms[0]) if len(norms) > 0 else 0.0,
                "final_norm": float(norms[-1]) if len(norms) > 0 else 0.0,
                "max_norm": float(np.max(norms)),
                "min_norm": float(np.min(norms)),
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
            }

            # Gradient explosion/vanishing detection
            try:
                analysis["explosion_risk"] = self._detect_gradient_explosion(norms)
            except Exception as e:
                logger.warning(f"Failed to detect gradient explosion: {e}")
                analysis["explosion_risk"] = {"error": str(e)}

            try:
                analysis["vanishing_risk"] = self._detect_gradient_vanishing(norms)
            except Exception as e:
                logger.warning(f"Failed to detect gradient vanishing: {e}")
                analysis["vanishing_risk"] = {"error": str(e)}

            # Norm evolution trend
            if len(norms) > 5:
                try:
                    trend = self._analyze_norm_trend(norms)
                    analysis["trend_analysis"] = trend
                except Exception as e:
                    logger.warning(f"Failed to analyze norm trend: {e}")
                    analysis["trend_analysis"] = {"error": str(e)}

            return analysis

        except Exception as e:
            logger.error(f"Error in gradient magnitude analysis: {e}")
            return {"status": "error", "error": str(e), "traceback": str(e.__class__.__name__)}

    def analyze_layer_wise_gradients(self) -> Dict[str, Any]:
        """Analyze gradients at the layer level.

        Returns:
            Layer-wise gradient analysis with error handling
        """
        try:
            if not self.layer_gradients:
                return {"status": "no_data", "error": "No layer gradient data available"}

            layer_analysis: Dict[str, Dict[str, Any]] = {}

            for layer_name, layer_grad_history in self.layer_gradients.items():
                try:
                    if not layer_grad_history:
                        logger.warning(f"No gradient history for layer {layer_name}")
                        continue

                    layer_norms = np.array(layer_grad_history)

                    # Validate and clean layer data
                    if np.any(np.isnan(layer_norms)) or np.any(np.isinf(layer_norms)):
                        logger.warning(f"Found NaN or infinite values in layer {layer_name}, cleaning data")
                        layer_norms = layer_norms[~np.isnan(layer_norms) & ~np.isinf(layer_norms)]

                    if len(layer_norms) == 0:
                        logger.warning(f"No valid data for layer {layer_name} after cleaning")
                        continue

                    layer_stats: Dict[str, Any] = {
                        "mean_norm": float(np.mean(layer_norms)),
                        "std_norm": float(np.std(layer_norms)),
                        "max_norm": float(np.max(layer_norms)),
                        "min_norm": float(np.min(layer_norms)),
                        "coefficient_of_variation": (
                            float(np.std(layer_norms) / np.mean(layer_norms)) if np.mean(layer_norms) > 0 else 0.0
                        ),
                    }

                    # Layer-specific issues
                    try:
                        layer_stats["gradient_flow_quality"] = self._assess_gradient_flow_quality(layer_norms)
                    except Exception as e:
                        logger.warning(f"Failed to assess gradient flow for layer {layer_name}: {e}")
                        layer_stats["gradient_flow_quality"] = "error"

                    layer_analysis[layer_name] = layer_stats

                except Exception as e:
                    logger.error(f"Error analyzing layer {layer_name}: {e}")
                    layer_analysis[layer_name] = {"error": str(e)}

            # Cross-layer analysis
            try:
                analysis: Dict[str, Any] = {
                    "layer_analysis": layer_analysis,
                    "gradient_flow_summary": self._analyze_cross_layer_flow(layer_analysis),
                }
            except Exception as e:
                logger.error(f"Error in cross-layer analysis: {e}")
                analysis = {"layer_analysis": layer_analysis, "gradient_flow_summary": {"error": str(e)}}

            return analysis

        except Exception as e:
            logger.error(f"Error in layer-wise gradient analysis: {e}")
            return {"status": "error", "error": str(e), "traceback": str(e.__class__.__name__)}

    def detect_gradient_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in gradient behavior.

        Returns:
            Detected gradient anomalies with error handling
        """
        try:
            anomalies: Dict[str, Any] = {
                "detected_anomalies": [],
                "anomaly_count": 0,
                "severity_score": 0.0,
            }

            # Cosine similarity anomalies
            if self.cosine_similarities:
                try:
                    similarity_anomalies = self._detect_similarity_anomalies(self.cosine_similarities)
                    anomalies["detected_anomalies"].extend(similarity_anomalies)
                except Exception as e:
                    logger.warning(f"Failed to detect similarity anomalies: {e}")

            # Gradient norm anomalies
            if self.gradient_norms:
                try:
                    norm_anomalies = self._detect_norm_anomalies(self.gradient_norms)
                    anomalies["detected_anomalies"].extend(norm_anomalies)
                except Exception as e:
                    logger.warning(f"Failed to detect norm anomalies: {e}")

            # Layer-wise anomalies
            if self.layer_gradients:
                try:
                    layer_anomalies = self._detect_layer_anomalies(self.layer_gradients)
                    anomalies["detected_anomalies"].extend(layer_anomalies)
                except Exception as e:
                    logger.warning(f"Failed to detect layer anomalies: {e}")

            anomalies["anomaly_count"] = len(anomalies["detected_anomalies"])
            anomalies["severity_score"] = self._calculate_anomaly_severity(anomalies["detected_anomalies"])

            return anomalies

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "status": "error",
                "error": str(e),
                "detected_anomalies": [],
                "anomaly_count": 0,
                "severity_score": 0.0,
            }

    def generate_gradient_report(self) -> Dict[str, Any]:
        """Generate comprehensive gradient analysis report.

        Returns:
            Complete gradient analysis report with error handling
        """
        try:
            report = {
                "consistency_analysis": self.analyze_gradient_consistency(),
                "magnitude_analysis": self.analyze_gradient_magnitude_evolution(),
                "layer_analysis": self.analyze_layer_wise_gradients(),
                "anomaly_detection": self.detect_gradient_anomalies(),
            }

            # Overall assessment
            try:
                report["overall_assessment"] = self._generate_overall_assessment(report)
            except Exception as e:
                logger.warning(f"Failed to generate overall assessment: {e}")
                report["overall_assessment"] = {"error": str(e)}

            return report

        except Exception as e:
            logger.error(f"Error generating gradient report: {e}")
            return {"status": "error", "error": str(e), "traceback": str(e.__class__.__name__)}

    def visualize_gradient_evolution(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create visualizations of gradient evolution.

        Args:
            save_path: Optional path to save visualizations

        Returns:
            Dictionary with plot information and error handling
        """
        plots_created: Dict[str, Any] = {}

        try:
            if self.cosine_similarities:
                try:
                    fig1 = self._plot_cosine_similarities()
                    if save_path:
                        fig1_path = save_path / "gradient_cosine_similarities.png"
                        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
                        plots_created["cosine_similarities"] = fig1_path
                    plt.close(fig1)
                except Exception as e:
                    logger.error(f"Failed to create cosine similarities plot: {e}")
                    plots_created["cosine_similarities"] = {"error": str(e)}

            if self.gradient_norms:
                try:
                    fig2 = self._plot_gradient_norms()
                    if save_path:
                        fig2_path = save_path / "gradient_norms.png"
                        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
                        plots_created["gradient_norms"] = fig2_path
                    plt.close(fig2)
                except Exception as e:
                    logger.error(f"Failed to create gradient norms plot: {e}")
                    plots_created["gradient_norms"] = {"error": str(e)}

            if self.layer_gradients:
                try:
                    fig3 = self._plot_layer_gradients()
                    if save_path and fig3 is not None:
                        fig3_path = save_path / "layer_gradient_norms.png"
                        fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
                        plots_created["layer_gradients"] = fig3_path
                    if fig3 is not None:
                        plt.close(fig3)
                except Exception as e:
                    logger.error(f"Failed to create layer gradients plot: {e}")
                    plots_created["layer_gradients"] = {"error": str(e)}

            return plots_created

        except Exception as e:
            logger.error(f"Error in gradient visualization: {e}")
            return {"error": str(e)}

    def _calculate_consistency_score(self, similarities: np.ndarray) -> float:
        """Calculate a consistency score from cosine similarities."""
        try:
            # Score based on mean similarity and stability (low variance)
            mean_sim = float(np.mean(similarities))
            std_sim = float(np.std(similarities))

            # Penalize high variance
            consistency_score = mean_sim - (std_sim * 0.5)

            # Normalize to 0-1 range
            return max(0.0, min(1.0, (consistency_score + 1) / 2))
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0

    def _assess_consistency_level(self, score: float) -> str:
        """Assess consistency level from score."""
        try:
            if score > 0.8:
                return "very_consistent"
            elif score > 0.6:
                return "consistent"
            elif score > 0.4:
                return "moderately_consistent"
            else:
                return "inconsistent"
        except Exception as e:
            logger.error(f"Error assessing consistency level: {e}")
            return "unknown"

    def _analyze_similarity_trend(self, similarities: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in cosine similarities."""
        try:
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
        except Exception as e:
            logger.error(f"Error analyzing similarity trend: {e}")
            return {"error": str(e)}

    def _find_stability_windows(self, similarities: np.ndarray, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Find windows of stable gradient behavior."""
        try:
            if len(similarities) < 10:
                return []

            windows: List[Dict[str, Any]] = []
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
        except Exception as e:
            logger.error(f"Error finding stability windows: {e}")
            return []

    def _detect_gradient_explosion(self, norms: np.ndarray) -> Dict[str, Any]:
        """Detect gradient explosion patterns."""
        try:
            max_norm = np.max(norms)
            mean_norm = np.mean(norms)

            # Check for sudden spikes
            spikes: List[Dict[str, Any]] = []
            for i in range(1, len(norms)):
                if norms[i] > 3 * norms[i - 1] and norms[i] > 5.0:
                    spikes.append({"step": i, "norm": float(norms[i])})

            explosion_risk = (
                "high"
                if max_norm > 10.0 or len(spikes) > 3
                else "medium" if max_norm > 5.0 or len(spikes) > 1 else "low"
            )

            return {
                "risk_level": explosion_risk,
                "max_norm": float(max_norm),
                "spike_count": len(spikes),
                "spikes": spikes,
                "norm_ratio": float(max_norm / mean_norm) if mean_norm > 0 else 0.0,
            }
        except Exception as e:
            logger.error(f"Error detecting gradient explosion: {e}")
            return {"error": str(e)}

    def _detect_gradient_vanishing(self, norms: np.ndarray) -> Dict[str, Any]:
        """Detect gradient vanishing patterns."""
        try:
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
        except Exception as e:
            logger.error(f"Error detecting gradient vanishing: {e}")
            return {"error": str(e)}

    def _analyze_norm_trend(self, norms: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in gradient norms."""
        try:
            x = np.arange(len(norms))
            trend_coef = np.polyfit(x, norms, 1)[0]

            return {
                "trend_coefficient": float(trend_coef),
                "trend_direction": "increasing" if trend_coef > 0 else "decreasing",
                "initial_norm": float(norms[0]),
                "final_norm": float(norms[-1]),
                "change_ratio": float(norms[-1] / norms[0]) if norms[0] > 0 else 0.0,
            }
        except Exception as e:
            logger.error(f"Error analyzing norm trend: {e}")
            return {"error": str(e)}

    def _assess_gradient_flow_quality(self, layer_norms: np.ndarray) -> str:
        """Assess gradient flow quality for a layer."""
        try:
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
        except Exception as e:
            logger.error(f"Error assessing gradient flow quality: {e}")
            return "error"

    def _analyze_cross_layer_flow(self, layer_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gradient flow across layers."""
        try:
            if not layer_analysis:
                return {"status": "no_data"}

            mean_norms = [stats["mean_norm"] for stats in layer_analysis.values() if "mean_norm" in stats]

            return {
                "overall_flow_quality": self._assess_overall_flow_quality(layer_analysis),
                "norm_variance_across_layers": float(np.var(mean_norms)) if mean_norms else 0.0,
                "problematic_layers": [
                    name
                    for name, stats in layer_analysis.items()
                    if stats.get("gradient_flow_quality") in ["vanishing", "exploding"]
                ],
                "layer_count": len(layer_analysis),
            }
        except Exception as e:
            logger.error(f"Error analyzing cross-layer flow: {e}")
            return {"error": str(e)}

    def _assess_overall_flow_quality(self, layer_analysis: Dict[str, Dict[str, Any]]) -> str:
        """Assess overall gradient flow quality."""
        try:
            if not layer_analysis:
                return "no_data"

            quality_counts: Dict[str, int] = {}
            for stats in layer_analysis.values():
                quality = stats.get("gradient_flow_quality", "unknown")
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
        except Exception as e:
            logger.error(f"Error assessing overall flow quality: {e}")
            return "error"

    def _detect_similarity_anomalies(self, similarities: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in cosine similarities."""
        try:
            if len(similarities) < 10:
                return []

            similarities_array = np.array(similarities)
            mean_sim = float(np.mean(similarities_array))
            std_sim = float(np.std(similarities_array))

            anomalies: List[Dict[str, Any]] = []
            threshold = 3 * std_sim  # 3-sigma rule

            for i, sim in enumerate(similarities_array):
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
        except Exception as e:
            logger.error(f"Error detecting similarity anomalies: {e}")
            return []

    def _detect_norm_anomalies(self, norms: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in gradient norms."""
        try:
            if len(norms) < 10:
                return []

            norms_array = np.array(norms)
            log_norms = np.log(norms_array + 1e-8)  # Log scale for better outlier detection
            mean_log = float(np.mean(log_norms))
            std_log = float(np.std(log_norms))

            anomalies: List[Dict[str, Any]] = []
            threshold = 3 * std_log

            for i, log_norm in enumerate(log_norms):
                if abs(log_norm - mean_log) > threshold:
                    anomalies.append(
                        {
                            "type": "gradient_norm_outlier",
                            "step": i,
                            "value": float(norms_array[i]),
                            "severity": min(1.0, abs(log_norm - mean_log) / threshold),
                        }
                    )

            return anomalies
        except Exception as e:
            logger.error(f"Error detecting norm anomalies: {e}")
            return []

    def _detect_layer_anomalies(self, layer_gradients: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect layer-specific gradient anomalies."""
        try:
            anomalies: List[Dict[str, Any]] = []

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
        except Exception as e:
            logger.error(f"Error detecting layer anomalies: {e}")
            return []

    def _calculate_anomaly_severity(self, anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall anomaly severity score."""
        try:
            if not anomalies:
                return 0.0

            total_severity = sum(anomaly.get("severity", 0.0) for anomaly in anomalies)
            return float(min(1.0, total_severity / len(anomalies)))
        except Exception as e:
            logger.error(f"Error calculating anomaly severity: {e}")
            return 0.0

    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall gradient health assessment."""
        try:
            assessment: Dict[str, Any] = {
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
        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return {"error": str(e)}

    def _plot_cosine_similarities(self) -> Figure:
        """Create cosine similarity plot."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating cosine similarities plot: {e}")
            raise

    def _plot_gradient_norms(self) -> Figure:
        """Create gradient norms plot."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating gradient norms plot: {e}")
            raise

    def _plot_layer_gradients(self) -> Optional[Figure]:
        """Create layer-wise gradient plot."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating layer gradients plot: {e}")
            raise
