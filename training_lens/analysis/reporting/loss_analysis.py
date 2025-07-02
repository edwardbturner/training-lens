"""Loss function analysis for training insights."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ...utils.logging import get_logger

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = get_logger(__name__)


class LossFunctionAnalyzer:
    """Specialized analyzer for loss function evolution during training."""

    def __init__(self, wandb_run_path: Optional[str] = None) -> None:
        """Initialize loss function analyzer.

        Args:
            wandb_run_path: Optional wandb run path (entity/project/run_id)
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for LossFunctionAnalyzer. Install with: pip install wandb")

        self.wandb_run_path = wandb_run_path
        self.training_loss: List[float] = []
        self.eval_loss: List[float] = []
        self.training_steps: List[int] = []
        self.eval_steps: List[int] = []

        if wandb_run_path:
            self._load_wandb_data()

    def _load_wandb_data(self) -> None:
        """Load loss data from wandb run."""
        if not self.wandb_run_path:
            raise ValueError("wandb_run_path is required for loading wandb data")

        try:
            api = wandb.Api()
            run = api.run(self.wandb_run_path)

            # Get run history
            history = run.history()

            # Extract training loss
            if "train/loss" in history.columns:
                train_data = history[["_step", "train/loss"]].dropna()
                self.training_steps = train_data["_step"].tolist()
                self.training_loss = train_data["train/loss"].tolist()
            elif "loss" in history.columns:
                train_data = history[["_step", "loss"]].dropna()
                self.training_steps = train_data["_step"].tolist()
                self.training_loss = train_data["loss"].tolist()

            # Extract eval loss
            if "eval/loss" in history.columns:
                eval_data = history[["_step", "eval/loss"]].dropna()
                self.eval_steps = eval_data["_step"].tolist()
                self.eval_loss = eval_data["eval/loss"].tolist()
            elif "validation_loss" in history.columns:
                eval_data = history[["_step", "validation_loss"]].dropna()
                self.eval_steps = eval_data["_step"].tolist()
                self.eval_loss = eval_data["validation_loss"].tolist()

            logger.info(
                f"Loaded {len(self.training_loss)} training loss points and {len(self.eval_loss)} eval loss points"
            )

        except Exception as e:
            logger.error(f"Failed to load wandb data: {e}")
            raise ValueError(f"Could not load data from wandb run {self.wandb_run_path}: {e}")

    def load_data_from_dict(self, data: Dict[str, Union[List[float], List[int]]]) -> None:
        """Load loss data from dictionary.

        Args:
            data: Dictionary with keys 'training_loss', 'eval_loss', 'training_steps', 'eval_steps'
        """
        self.training_loss = [float(x) for x in data.get("training_loss", [])]
        self.eval_loss = [float(x) for x in data.get("eval_loss", [])]
        self.training_steps = [int(x) for x in data.get("training_steps", list(range(len(self.training_loss))))]
        self.eval_steps = [int(x) for x in data.get("eval_steps", list(range(len(self.eval_loss))))]

        logger.info(f"Loaded {len(self.training_loss)} training loss points and {len(self.eval_loss)} eval loss points")

    def analyze_loss_evolution(self) -> Dict[str, Any]:
        """Analyze loss evolution patterns.

        Returns:
            Comprehensive loss analysis
        """
        try:
            analysis: Dict[str, Any] = {}

            # Training loss analysis
            if self.training_loss:
                analysis["training_loss"] = self._analyze_loss_sequence(
                    self.training_loss, self.training_steps, "training"
                )

            # Eval loss analysis
            if self.eval_loss:
                analysis["eval_loss"] = self._analyze_loss_sequence(self.eval_loss, self.eval_steps, "eval")

            # Comparative analysis
            if self.training_loss and self.eval_loss:
                analysis["comparative_analysis"] = self._analyze_loss_comparison()

            # Convergence analysis
            analysis["convergence_analysis"] = self._analyze_convergence()

            # Overfitting detection
            analysis["overfitting_analysis"] = self._detect_overfitting()

            return analysis

        except Exception as e:
            logger.error(f"Error in loss evolution analysis: {e}")
            return {"status": "error", "error": str(e)}

    def plot_loss_curves(
        self, log_x: bool = False, log_y: bool = False, save_path: Optional[Path] = None, figsize: tuple = (12, 8)
    ) -> Figure:
        """Plot training and evaluation loss curves.

        Args:
            log_x: Whether to use log scale for x-axis
            log_y: Whether to use log scale for y-axis
            save_path: Optional path to save the plot
            figsize: Figure size tuple

        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)

            # Plot training loss
            if self.training_loss and self.training_steps:
                ax.plot(
                    self.training_steps, self.training_loss, label="Training Loss", color="blue", alpha=0.8, linewidth=2
                )

            # Plot eval loss
            if self.eval_loss and self.eval_steps:
                ax.plot(self.eval_steps, self.eval_loss, label="Evaluation Loss", color="red", alpha=0.8, linewidth=2)

            # Set scales
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            # Formatting
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Evaluation Loss Evolution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add smoothed trend lines if enough data
            if len(self.training_loss) > 10:
                self._add_trend_lines(ax, log_x, log_y)

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Loss curve plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating loss plot: {e}")
            raise

    def detect_loss_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in loss behavior.

        Returns:
            Detected loss anomalies
        """
        try:
            anomalies: Dict[str, Any] = {"training_anomalies": [], "eval_anomalies": [], "comparative_anomalies": []}

            # Training loss anomalies
            if self.training_loss:
                anomalies["training_anomalies"] = self._detect_sequence_anomalies(
                    self.training_loss, self.training_steps, "training"
                )

            # Eval loss anomalies
            if self.eval_loss:
                anomalies["eval_anomalies"] = self._detect_sequence_anomalies(self.eval_loss, self.eval_steps, "eval")

            # Comparative anomalies (training vs eval)
            if self.training_loss and self.eval_loss:
                anomalies["comparative_anomalies"] = self._detect_comparative_anomalies()

            return anomalies

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"status": "error", "error": str(e)}

    def generate_loss_report(self) -> Dict[str, Any]:
        """Generate comprehensive loss analysis report.

        Returns:
            Complete loss analysis report
        """
        try:
            report = {
                "loss_evolution": self.analyze_loss_evolution(),
                "anomaly_detection": self.detect_loss_anomalies(),
                "data_summary": {
                    "training_points": len(self.training_loss),
                    "eval_points": len(self.eval_loss),
                    "training_range": (
                        [min(self.training_steps), max(self.training_steps)] if self.training_steps else None
                    ),
                    "eval_range": [min(self.eval_steps), max(self.eval_steps)] if self.eval_steps else None,
                },
            }

            # Overall assessment
            report["overall_assessment"] = self._generate_overall_assessment(report)

            return report

        except Exception as e:
            logger.error(f"Error generating loss report: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_loss_sequence(self, loss_values: List[float], steps: List[int], loss_type: str) -> Dict[str, Any]:
        """Analyze a sequence of loss values."""
        try:
            loss_array = np.array(loss_values)

            analysis: Dict[str, Any] = {
                "initial_loss": float(loss_array[0]),
                "final_loss": float(loss_array[-1]),
                "min_loss": float(np.min(loss_array)),
                "max_loss": float(np.max(loss_array)),
                "mean_loss": float(np.mean(loss_array)),
                "std_loss": float(np.std(loss_array)),
                "loss_reduction": float(loss_array[0] - loss_array[-1]),
                "reduction_percentage": (
                    float((loss_array[0] - loss_array[-1]) / loss_array[0] * 100) if loss_array[0] > 0 else 0.0
                ),
            }

            # Trend analysis
            if len(loss_array) > 5:
                analysis["trend_analysis"] = self._analyze_trend(loss_array, steps)

            # Smoothness analysis
            analysis["smoothness"] = self._analyze_smoothness(loss_array)

            # Learning phases
            analysis["learning_phases"] = self._identify_learning_phases(loss_array)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {loss_type} loss sequence: {e}")
            return {"error": str(e)}

    def _analyze_loss_comparison(self) -> Dict[str, Any]:
        """Compare training and evaluation loss."""
        try:
            # Align loss sequences by interpolating to common steps
            common_steps = sorted(set(self.training_steps + self.eval_steps))

            train_interp = np.interp(common_steps, self.training_steps, self.training_loss)
            eval_interp = np.interp(common_steps, self.eval_steps, self.eval_loss)

            # Calculate gap
            gap = eval_interp - train_interp

            analysis = {
                "mean_gap": float(np.mean(gap)),
                "std_gap": float(np.std(gap)),
                "max_gap": float(np.max(gap)),
                "min_gap": float(np.min(gap)),
                "final_gap": float(gap[-1]),
                "gap_trend": self._analyze_trend(gap, common_steps)["trend_direction"],
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in loss comparison: {e}")
            return {"error": str(e)}

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence properties."""
        try:
            convergence = {}

            # Training convergence
            if self.training_loss:
                convergence["training"] = self._analyze_sequence_convergence(self.training_loss, "training")

            # Eval convergence
            if self.eval_loss:
                convergence["eval"] = self._analyze_sequence_convergence(self.eval_loss, "eval")

            return convergence

        except Exception as e:
            logger.error(f"Error in convergence analysis: {e}")
            return {"error": str(e)}

    def _detect_overfitting(self) -> Dict[str, Any]:
        """Detect overfitting patterns."""
        try:
            if not (self.training_loss and self.eval_loss):
                return {"status": "insufficient_data"}

            # Find point where eval loss starts increasing while training decreases
            overfitting_signals = []

            # Simple heuristic: look for periods where training loss decreases but eval loss increases
            if len(self.training_loss) > 10 and len(self.eval_loss) > 10:
                # Use rolling windows to smooth out noise
                window = min(5, len(self.training_loss) // 4)

                train_smooth = pd.Series(self.training_loss).rolling(window=window).mean()
                eval_smooth = pd.Series(self.eval_loss).rolling(window=window).mean()

                # Find divergence points
                for i in range(window, min(len(train_smooth), len(eval_smooth)) - 1):
                    if (
                        train_smooth.iloc[i] < train_smooth.iloc[i - 1]
                        and eval_smooth.iloc[i] > eval_smooth.iloc[i - 1]
                    ):
                        overfitting_signals.append(
                            {
                                "step": i,
                                "training_loss": float(train_smooth.iloc[i]),
                                "eval_loss": float(eval_smooth.iloc[i]),
                            }
                        )

            # Overall assessment
            final_gap = (
                self.eval_loss[-1] - self.training_loss[-1]
                if len(self.eval_loss) > 0 and len(self.training_loss) > 0
                else 0
            )

            overfitting_risk = (
                "high"
                if final_gap > 0.5 or len(overfitting_signals) > 3
                else "medium"
                if final_gap > 0.2 or len(overfitting_signals) > 1
                else "low"
            )

            return {
                "risk_level": overfitting_risk,
                "overfitting_signals": overfitting_signals,
                "final_gap": float(final_gap),
                "divergence_points": len(overfitting_signals),
            }

        except Exception as e:
            logger.error(f"Error in overfitting detection: {e}")
            return {"error": str(e)}

    def _analyze_trend(self, values: np.ndarray, steps: List[int]) -> Dict[str, Any]:
        """Analyze trend in loss values."""
        try:
            if len(values) < 2:
                return {"trend_direction": "stable", "trend_strength": 0.0}

            # Linear trend
            trend_coef = np.polyfit(steps, values, 1)[0]

            # Exponential trend (if decreasing)
            exp_trend = None
            if values[0] > values[-1]:
                try:
                    # Fit y = a * exp(b * x)
                    log_values = np.log(values + 1e-8)
                    exp_coef = np.polyfit(steps, log_values, 1)
                    exp_trend = exp_coef[0]  # b coefficient
                except Exception:
                    pass

            trend_direction = "decreasing" if trend_coef < -1e-6 else "increasing" if trend_coef > 1e-6 else "stable"

            return {
                "trend_direction": trend_direction,
                "linear_coefficient": float(trend_coef),
                "trend_strength": abs(float(trend_coef)),
                "exponential_coefficient": float(exp_trend) if exp_trend is not None else None,
            }

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"error": str(e)}

    def _analyze_smoothness(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze smoothness of loss curve."""
        try:
            if len(values) < 3:
                return {"smoothness_score": 1.0}

            # Calculate second derivative (curvature)
            second_derivative = np.diff(values, n=2)
            curvature = np.std(second_derivative)

            # Smoothness score (lower curvature = smoother)
            smoothness_score = 1.0 / (1.0 + curvature)

            return {
                "smoothness_score": float(smoothness_score),
                "curvature_std": float(curvature),
                "smoothness_level": (
                    "very_smooth"
                    if smoothness_score > 0.8
                    else "smooth"
                    if smoothness_score > 0.6
                    else "moderate"
                    if smoothness_score > 0.4
                    else "noisy"
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing smoothness: {e}")
            return {"error": str(e)}

    def _identify_learning_phases(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """Identify different learning phases."""
        try:
            if len(values) < 10:
                return []

            phases = []

            # Simple approach: identify periods of fast vs slow learning
            # Calculate rolling rate of change
            window = max(3, len(values) // 10)
            rates = []

            for i in range(window, len(values)):
                rate = (values[i - window] - values[i]) / window  # Loss reduction rate
                rates.append(rate)

            if not rates:
                return []

            rates = np.array(rates)

            # Identify fast learning (high rate) vs slow learning (low rate)
            threshold = np.percentile(rates, 75)

            current_phase = None
            phase_start = 0

            for i, rate in enumerate(rates):
                phase_type = "fast_learning" if rate > threshold else "slow_learning"

                if current_phase != phase_type:
                    if current_phase is not None:
                        phases.append(
                            {
                                "phase": current_phase,
                                "start_step": phase_start,
                                "end_step": i + window - 1,
                                "duration": i + window - 1 - phase_start,
                            }
                        )
                    current_phase = phase_type
                    phase_start = i + window

            # Add final phase
            if current_phase is not None:
                phases.append(
                    {
                        "phase": current_phase,
                        "start_step": phase_start,
                        "end_step": len(values) - 1,
                        "duration": len(values) - 1 - phase_start,
                    }
                )

            return phases

        except Exception as e:
            logger.error(f"Error identifying learning phases: {e}")
            return []

    def _analyze_sequence_convergence(self, values: List[float], seq_type: str) -> Dict[str, Any]:
        """Analyze convergence of a loss sequence."""
        try:
            if len(values) < 10:
                return {"status": "insufficient_data"}

            values_array = np.array(values)

            # Check if loss is still decreasing in recent steps
            recent_portion = 0.2  # Last 20% of training
            recent_start = int(len(values) * (1 - recent_portion))
            recent_values = values_array[recent_start:]

            # Calculate recent trend
            recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

            # Calculate coefficient of variation in recent portion
            recent_cv = np.std(recent_values) / np.mean(recent_values) if np.mean(recent_values) > 0 else float("inf")

            # Convergence assessment
            converged = abs(recent_trend) < 1e-4 and recent_cv < 0.1  # Very small trend  # Low variation

            convergence_quality = (
                "converged"
                if converged
                else (
                    "slow_convergence"
                    if abs(recent_trend) < 1e-3
                    else "still_improving"
                    if recent_trend < -1e-3
                    else "diverging"
                )
            )

            return {
                "converged": converged,
                "convergence_quality": convergence_quality,
                "recent_trend": float(recent_trend),
                "recent_coefficient_of_variation": float(recent_cv),
                "plateau_detection": recent_cv < 0.05,
            }

        except Exception as e:
            logger.error(f"Error analyzing convergence for {seq_type}: {e}")
            return {"error": str(e)}

    def _detect_sequence_anomalies(self, values: List[float], steps: List[int], seq_type: str) -> List[Dict[str, Any]]:
        """Detect anomalies in a loss sequence."""
        try:
            if len(values) < 10:
                return []

            values_array = np.array(values)
            anomalies = []

            # Z-score based outliers
            z_scores = np.abs((values_array - np.mean(values_array)) / np.std(values_array))
            outlier_threshold = 3.0

            for i, (z_score, value, step) in enumerate(zip(z_scores, values_array, steps)):
                if z_score > outlier_threshold:
                    anomalies.append(
                        {
                            "type": "outlier",
                            "step": step,
                            "value": float(value),
                            "z_score": float(z_score),
                            "severity": min(1.0, z_score / 5.0),
                        }
                    )

            # Sudden spikes
            if len(values) > 1:
                ratios = values_array[1:] / (values_array[:-1] + 1e-8)
                for i, (ratio, step) in enumerate(zip(ratios, steps[1:])):
                    if ratio > 2.0:  # Loss doubled
                        anomalies.append(
                            {
                                "type": "sudden_spike",
                                "step": step,
                                "ratio": float(ratio),
                                "severity": min(1.0, ratio / 5.0),
                            }
                        )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies in {seq_type}: {e}")
            return []

    def _detect_comparative_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in training vs eval loss comparison."""
        try:
            anomalies = []

            # Simple check: eval loss much higher than training loss
            if self.training_loss and self.eval_loss:
                final_train = self.training_loss[-1]
                final_eval = self.eval_loss[-1]

                gap_ratio = final_eval / final_train if final_train > 0 else float("inf")

                if gap_ratio > 3.0:  # Eval loss more than 3x training loss
                    anomalies.append(
                        {
                            "type": "large_generalization_gap",
                            "training_loss": float(final_train),
                            "eval_loss": float(final_eval),
                            "gap_ratio": float(gap_ratio),
                            "severity": min(1.0, gap_ratio / 5.0),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting comparative anomalies: {e}")
            return []

    def _generate_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of loss behavior."""
        try:
            assessment = {"training_health": "unknown", "key_findings": [], "recommendations": []}

            # Analyze training progress
            loss_evolution = report.get("loss_evolution", {})

            # Check training loss
            train_analysis = loss_evolution.get("training_loss", {})
            if train_analysis and not train_analysis.get("error"):
                reduction_pct = train_analysis.get("reduction_percentage", 0)
                if reduction_pct > 50:
                    assessment["key_findings"].append("Strong training loss reduction")
                elif reduction_pct > 20:
                    assessment["key_findings"].append("Moderate training loss reduction")
                else:
                    assessment["key_findings"].append("Limited training loss reduction")
                    assessment["recommendations"].append("Consider longer training or learning rate adjustment")

            # Check overfitting
            overfitting = loss_evolution.get("overfitting_analysis", {})
            if overfitting and not overfitting.get("error"):
                risk = overfitting.get("risk_level", "unknown")
                if risk == "high":
                    assessment["key_findings"].append("High overfitting risk detected")
                    assessment["recommendations"].append("Consider regularization or early stopping")
                elif risk == "medium":
                    assessment["key_findings"].append("Moderate overfitting risk")
                    assessment["recommendations"].append("Monitor validation loss closely")

            # Check convergence
            convergence = loss_evolution.get("convergence_analysis", {})
            train_conv = convergence.get("training", {})
            if train_conv and not train_conv.get("error"):
                quality = train_conv.get("convergence_quality", "unknown")
                if quality == "converged":
                    assessment["key_findings"].append("Training has converged")
                elif quality == "still_improving":
                    assessment["key_findings"].append("Training still improving")
                    assessment["recommendations"].append("Consider continuing training")

            # Overall health assessment
            issues = len([f for f in assessment["key_findings"] if "risk" in f or "Limited" in f])
            if issues == 0:
                assessment["training_health"] = "healthy"
            elif issues <= 2:
                assessment["training_health"] = "moderate"
            else:
                assessment["training_health"] = "concerning"

            return assessment

        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return {"error": str(e)}

    def _add_trend_lines(self, ax, log_x: bool, log_y: bool) -> None:
        """Add trend lines to the plot."""
        try:
            # Training trend
            if len(self.training_loss) > 10:
                if log_y:
                    # Exponential fit for log scale
                    try:
                        log_loss = np.log(np.array(self.training_loss) + 1e-8)
                        coef = np.polyfit(self.training_steps, log_loss, 1)
                        trend_line = np.exp(np.polyval(coef, self.training_steps))
                        ax.plot(
                            self.training_steps,
                            trend_line,
                            "--",
                            color="blue",
                            alpha=0.5,
                            linewidth=1,
                            label="Training Trend",
                        )
                    except Exception:
                        pass
                else:
                    # Linear fit
                    coef = np.polyfit(self.training_steps, self.training_loss, 1)
                    trend_line = np.polyval(coef, self.training_steps)
                    ax.plot(
                        self.training_steps,
                        trend_line,
                        "--",
                        color="blue",
                        alpha=0.5,
                        linewidth=1,
                        label="Training Trend",
                    )

            # Eval trend
            if len(self.eval_loss) > 10:
                if log_y:
                    try:
                        log_loss = np.log(np.array(self.eval_loss) + 1e-8)
                        coef = np.polyfit(self.eval_steps, log_loss, 1)
                        trend_line = np.exp(np.polyval(coef, self.eval_steps))
                        ax.plot(
                            self.eval_steps, trend_line, "--", color="red", alpha=0.5, linewidth=1, label="Eval Trend"
                        )
                    except Exception:
                        pass
                else:
                    coef = np.polyfit(self.eval_steps, self.eval_loss, 1)
                    trend_line = np.polyval(coef, self.eval_steps)
                    ax.plot(self.eval_steps, trend_line, "--", color="red", alpha=0.5, linewidth=1, label="Eval Trend")

        except Exception as e:
            logger.warning(f"Could not add trend lines: {e}")
