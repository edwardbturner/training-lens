"""Standard report generation for training analysis."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt

from ..utils.logging import get_logger
from .checkpoint_analyzer import CheckpointAnalyzer
from .gradient_analyzer import GradientAnalyzer
from .weight_analyzer import WeightAnalyzer

logger = get_logger(__name__)


class StandardReports:
    """Generates standard analysis reports for training runs."""

    def __init__(self, checkpoint_analyzer: CheckpointAnalyzer):
        """Initialize with a checkpoint analyzer.

        Args:
            checkpoint_analyzer: Analyzer instance with loaded checkpoints
        """
        self.analyzer = checkpoint_analyzer
        self.report_data = {}

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate high-level executive summary.

        Returns:
            Executive summary dictionary
        """
        if not self.analyzer.checkpoints_info:
            return {"status": "no_data"}

        # Basic training info
        checkpoints = self.analyzer.checkpoints_info
        steps = [cp["step"] for cp in checkpoints]

        # Get final metrics
        final_metadata = checkpoints[-1].get("metadata", {})
        initial_metadata = checkpoints[0].get("metadata", {})

        summary = {
            "training_overview": {
                "total_checkpoints": len(checkpoints),
                "training_steps": max(steps) if steps else 0,
                "training_duration_steps": max(steps) - min(steps) if len(steps) > 1 else 0,
                "checkpoint_frequency": self._calculate_checkpoint_frequency(steps),
            },
            "performance_metrics": {
                "initial_loss": initial_metadata.get("train_loss"),
                "final_loss": final_metadata.get("train_loss"),
                "loss_improvement": self._calculate_loss_improvement(
                    initial_metadata.get("train_loss"), final_metadata.get("train_loss")
                ),
                "training_efficiency": self._assess_training_efficiency(),
            },
            "model_health": {
                "gradient_health": self._assess_gradient_health(),
                "weight_stability": self._assess_weight_stability(),
                "convergence_status": self._assess_convergence(),
            },
            "recommendations": self._generate_recommendations(),
            "generated_at": datetime.now().isoformat(),
        }

        return summary

    def generate_technical_report(self) -> Dict[str, Any]:
        """Generate detailed technical analysis report.

        Returns:
            Comprehensive technical report
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analyzer_version": "0.1.0",
                "checkpoints_analyzed": len(self.analyzer.checkpoints_info),
            },
            "executive_summary": self.generate_executive_summary(),
            "detailed_analysis": {},
        }

        # Generate detailed analyses
        report["detailed_analysis"]["training_dynamics"] = self.analyzer.analyze_training_dynamics()
        report["detailed_analysis"]["gradient_analysis"] = self.analyzer.analyze_gradient_evolution()
        report["detailed_analysis"]["weight_analysis"] = self.analyzer.analyze_weight_evolution()
        report["detailed_analysis"]["overfitting_analysis"] = self.analyzer.detect_overfitting()

        # Add specialized analyses if data is available
        gradient_data = self._collect_gradient_data()
        if gradient_data:
            grad_analyzer = GradientAnalyzer(gradient_data)
            report["detailed_analysis"]["gradient_deep_dive"] = grad_analyzer.generate_gradient_report()

        weight_data = self._collect_weight_data()
        if weight_data:
            weight_analyzer = WeightAnalyzer(weight_data)
            report["detailed_analysis"]["weight_deep_dive"] = weight_analyzer.generate_weight_report()

        return report

    def generate_training_diagnostics(self) -> Dict[str, Any]:
        """Generate training diagnostics and troubleshooting report.

        Returns:
            Diagnostics report with issues and recommendations
        """
        diagnostics = {
            "overall_health": "unknown",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "performance_assessment": {},
        }

        # Check for critical issues
        critical_issues = self._detect_critical_issues()
        diagnostics["critical_issues"] = critical_issues

        # Check for warnings
        warnings = self._detect_warnings()
        diagnostics["warnings"] = warnings

        # Performance assessment
        diagnostics["performance_assessment"] = self._assess_overall_performance()

        # Generate recommendations
        diagnostics["recommendations"] = self._generate_targeted_recommendations(critical_issues, warnings)

        # Overall health score
        diagnostics["overall_health"] = self._calculate_overall_health(critical_issues, warnings)

        return diagnostics

    def generate_comparison_report(
        self,
        baseline_analyzer: CheckpointAnalyzer,
        experiment_name: str = "Experiment",
        baseline_name: str = "Baseline",
    ) -> Dict[str, Any]:
        """Generate comparison report between two training runs.

        Args:
            baseline_analyzer: Analyzer for baseline training run
            experiment_name: Name of the current experiment
            baseline_name: Name of the baseline experiment

        Returns:
            Comparison report
        """
        current_summary = self.generate_executive_summary()
        baseline_reports = StandardReports(baseline_analyzer)
        baseline_summary = baseline_reports.generate_executive_summary()

        comparison = {
            "comparison_metadata": {
                "experiment_name": experiment_name,
                "baseline_name": baseline_name,
                "generated_at": datetime.now().isoformat(),
            },
            "performance_comparison": self._compare_performance(current_summary, baseline_summary),
            "training_comparison": self._compare_training_dynamics(current_summary, baseline_summary),
            "recommendations": self._generate_comparison_recommendations(current_summary, baseline_summary),
        }

        return comparison

    def export_report(
        self,
        report_type: str,
        output_path: Union[str, Path],
        format: str = "json",
        include_plots: bool = False,
    ) -> Path:
        """Export a report to file.

        Args:
            report_type: Type of report ("executive", "technical", "diagnostics")
            output_path: Output file path
            format: Output format ("json", "html", "pdf")
            include_plots: Whether to include visualization plots

        Returns:
            Path to the exported report
        """
        output_path = Path(output_path)

        # Generate report data
        if report_type == "executive":
            report_data = self.generate_executive_summary()
        elif report_type == "technical":
            report_data = self.generate_technical_report()
        elif report_type == "diagnostics":
            report_data = self.generate_training_diagnostics()
        else:
            raise ValueError(f"Unknown report type: {report_type}")

        # Export based on format
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "html":
            html_content = self._generate_html_report(report_data, report_type)
            with open(output_path, "w") as f:
                f.write(html_content)

        elif format == "pdf":
            # This would require additional dependencies like reportlab
            raise NotImplementedError("PDF export not yet implemented")

        # Generate plots if requested
        if include_plots:
            plots_dir = output_path.parent / f"{output_path.stem}_plots"
            plots_dir.mkdir(exist_ok=True)
            self._generate_report_plots(plots_dir)

        logger.info(f"Report exported to {output_path}")
        return output_path

    def _calculate_checkpoint_frequency(self, steps: List[int]) -> Optional[int]:
        """Calculate average checkpoint frequency."""
        if len(steps) < 2:
            return None

        intervals = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
        return int(sum(intervals) / len(intervals))

    def _calculate_loss_improvement(
        self, initial_loss: Optional[float], final_loss: Optional[float]
    ) -> Optional[Dict[str, float]]:
        """Calculate loss improvement metrics."""
        if initial_loss is None or final_loss is None:
            return None

        absolute_improvement = initial_loss - final_loss
        relative_improvement = (absolute_improvement / initial_loss) * 100 if initial_loss > 0 else 0

        return {
            "absolute": absolute_improvement,
            "relative_percent": relative_improvement,
            "improvement_rate": (
                absolute_improvement / len(self.analyzer.checkpoints_info) if self.analyzer.checkpoints_info else 0
            ),
        }

    def _assess_training_efficiency(self) -> str:
        """Assess overall training efficiency."""
        training_dynamics = self.analyzer.analyze_training_dynamics()

        if "training_efficiency" in training_dynamics:
            efficiency = training_dynamics["training_efficiency"]
            score = efficiency.get("efficiency_score", 0)

            if score > 0.8:
                return "excellent"
            elif score > 0.6:
                return "good"
            elif score > 0.4:
                return "moderate"
            else:
                return "poor"

        return "unknown"

    def _assess_gradient_health(self) -> str:
        """Assess gradient health."""
        gradient_analysis = self.analyzer.analyze_gradient_evolution()

        if "gradient_stability" in gradient_analysis:
            stability = gradient_analysis["gradient_stability"]

            if stability in ["very_stable", "stable"]:
                return "healthy"
            elif stability == "moderate":
                return "moderate"
            else:
                return "poor"

        return "unknown"

    def _assess_weight_stability(self) -> str:
        """Assess weight stability."""
        weight_analysis = self.analyzer.analyze_weight_evolution()

        if "weight_stability" in weight_analysis:
            stability = weight_analysis["weight_stability"]

            if stability in ["very_stable", "stable"]:
                return "stable"
            elif stability == "moderate":
                return "moderate"
            else:
                return "unstable"

        return "unknown"

    def _assess_convergence(self) -> str:
        """Assess training convergence."""
        gradient_analysis = self.analyzer.analyze_gradient_evolution()

        if "convergence_analysis" in gradient_analysis:
            convergence = gradient_analysis["convergence_analysis"]
            status = convergence.get("convergence_status", "unknown")
            return status

        return "unknown"

    def _generate_recommendations(self) -> List[str]:
        """Generate general recommendations."""
        recommendations = []

        # Check gradient health
        gradient_health = self._assess_gradient_health()
        if gradient_health == "poor":
            recommendations.append("Consider reducing learning rate or adding gradient clipping")

        # Check weight stability
        weight_stability = self._assess_weight_stability()
        if weight_stability == "unstable":
            recommendations.append("Add weight decay or adjust learning rate schedule")

        # Check convergence
        convergence = self._assess_convergence()
        if convergence == "not_converged":
            recommendations.append("Increase training steps or adjust hyperparameters")

        return recommendations

    def _collect_gradient_data(self) -> Dict[str, Any]:
        """Collect gradient data from all checkpoints."""
        gradient_data = {}
        all_similarities = []

        for cp in self.analyzer.checkpoints_info:
            step = cp["step"]
            metrics = self.analyzer.load_checkpoint_metrics(step)

            if metrics and "gradient_cosine_similarities" in metrics:
                all_similarities.extend(metrics["gradient_cosine_similarities"])

        if all_similarities:
            gradient_data["gradient_cosine_similarities"] = all_similarities

        return gradient_data

    def _collect_weight_data(self) -> Dict[str, Any]:
        """Collect weight data from all checkpoints."""
        weight_data = {}
        all_weight_stats = []

        for cp in self.analyzer.checkpoints_info:
            step = cp["step"]
            metrics = self.analyzer.load_checkpoint_metrics(step)

            if metrics and "weight_stats_history" in metrics:
                all_weight_stats.extend(metrics["weight_stats_history"])

        if all_weight_stats:
            weight_data["weight_stats_history"] = all_weight_stats

        return weight_data

    def _detect_critical_issues(self) -> List[Dict[str, Any]]:
        """Detect critical training issues."""
        issues = []

        # Check for gradient explosion/vanishing
        gradient_analysis = self.analyzer.analyze_gradient_evolution()
        if gradient_analysis.get("gradient_stability") == "unstable":
            issues.append(
                {
                    "type": "gradient_instability",
                    "severity": "critical",
                    "description": "Gradient instability detected",
                    "recommendation": "Implement gradient clipping or reduce learning rate",
                }
            )

        # Check for overfitting
        overfitting = self.analyzer.detect_overfitting()
        if overfitting.get("overfitting_detected"):
            issues.append(
                {
                    "type": "overfitting",
                    "severity": "critical",
                    "description": "Overfitting detected",
                    "recommendation": "Implement early stopping or regularization",
                }
            )

        return issues

    def _detect_warnings(self) -> List[Dict[str, Any]]:
        """Detect training warnings."""
        warnings = []

        # Check training efficiency
        efficiency = self._assess_training_efficiency()
        if efficiency in ["poor", "moderate"]:
            warnings.append(
                {
                    "type": "low_efficiency",
                    "severity": "warning",
                    "description": f"Training efficiency is {efficiency}",
                    "recommendation": "Review hyperparameters and model architecture",
                }
            )

        return warnings

    def _assess_overall_performance(self) -> Dict[str, Any]:
        """Assess overall training performance."""
        return {
            "gradient_health": self._assess_gradient_health(),
            "weight_stability": self._assess_weight_stability(),
            "training_efficiency": self._assess_training_efficiency(),
            "convergence_status": self._assess_convergence(),
        }

    def _generate_targeted_recommendations(
        self, critical_issues: List[Dict[str, Any]], warnings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate targeted recommendations based on issues."""
        recommendations = []

        # Add recommendations from critical issues
        for issue in critical_issues:
            recommendations.append(issue.get("recommendation", ""))

        # Add recommendations from warnings
        for warning in warnings:
            recommendations.append(warning.get("recommendation", ""))

        # Remove duplicates
        return list(set(filter(None, recommendations)))

    def _calculate_overall_health(self, critical_issues: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> str:
        """Calculate overall training health score."""
        if critical_issues:
            return "poor"
        elif len(warnings) > 2:
            return "moderate"
        elif warnings:
            return "good"
        else:
            return "excellent"

    def _compare_performance(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between runs."""
        comparison = {}

        current_perf = current.get("performance_metrics", {})
        baseline_perf = baseline.get("performance_metrics", {})

        # Compare final losses
        current_loss = current_perf.get("final_loss")
        baseline_loss = baseline_perf.get("final_loss")

        if current_loss is not None and baseline_loss is not None:
            improvement = (baseline_loss - current_loss) / baseline_loss * 100
            comparison["loss_improvement_percent"] = improvement
            comparison["loss_comparison"] = "better" if improvement > 0 else "worse"

        return comparison

    def _compare_training_dynamics(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare training dynamics between runs."""
        comparison = {}

        current_overview = current.get("training_overview", {})
        baseline_overview = baseline.get("training_overview", {})

        # Compare training steps
        current_steps = current_overview.get("training_steps", 0)
        baseline_steps = baseline_overview.get("training_steps", 0)

        comparison["steps_difference"] = current_steps - baseline_steps
        comparison["efficiency_comparison"] = "more_efficient" if current_steps < baseline_steps else "less_efficient"

        return comparison

    def _generate_comparison_recommendations(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        # Add comparison-specific recommendations
        performance_comp = self._compare_performance(current, baseline)

        if performance_comp.get("loss_comparison") == "worse":
            recommendations.append("Consider reverting to baseline hyperparameters")
        elif performance_comp.get("loss_comparison") == "better":
            recommendations.append("Current configuration shows improvement over baseline")

        return recommendations

    def _generate_html_report(self, report_data: Dict[str, Any], report_type: str) -> str:
        """Generate HTML version of report."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Lens {report_type.title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .critical {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
                .success {{ background: #d1edff; border: 1px solid #74c0fc; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }}
                .recommendation {{ background: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Training Lens {report_type.title()} Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>📊 Report Data</h2>
                <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        return html_template

    def _generate_report_plots(self, plots_dir: Path) -> None:
        """Generate plots for reports."""
        plots_dir.mkdir(exist_ok=True)

        if not self.analyzer.checkpoints_info:
            return

        # Training overview plot
        self._create_training_overview_plot(plots_dir)

        # Performance metrics plot
        self._create_performance_plot(plots_dir)

    def _create_training_overview_plot(self, plots_dir: Path) -> None:
        """Create training overview plot."""
        steps = []
        losses = []
        learning_rates = []

        for cp in self.analyzer.checkpoints_info:
            metadata = cp.get("metadata", {})
            if "step" in metadata:
                steps.append(metadata["step"])
                losses.append(metadata.get("train_loss"))
                learning_rates.append(metadata.get("learning_rate"))

        if not steps:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Loss plot
        valid_losses = [(s, l) for s, l in zip(steps, losses) if l is not None]
        if valid_losses:
            loss_steps, loss_values = zip(*valid_losses)
            ax1.plot(loss_steps, loss_values, "b-", linewidth=2, marker="o")
            ax1.set_ylabel("Training Loss")
            ax1.set_title("Training Progress Overview")
            ax1.grid(True, alpha=0.3)

        # Learning rate plot
        valid_lrs = [(s, lr) for s, lr in zip(steps, learning_rates) if lr is not None]
        if valid_lrs:
            lr_steps, lr_values = zip(*valid_lrs)
            ax2.plot(lr_steps, lr_values, "g-", linewidth=2, marker="s")
            ax2.set_xlabel("Training Step")
            ax2.set_ylabel("Learning Rate")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "training_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_performance_plot(self, plots_dir: Path) -> None:
        """Create performance metrics plot."""
        # This would create additional performance visualizations
        pass
