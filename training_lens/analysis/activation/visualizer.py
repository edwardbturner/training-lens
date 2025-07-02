"""Visualization tools for activation evolution analysis."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...utils.logging import get_logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Visualization features will be limited.")

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")


logger = get_logger(__name__)


class ActivationVisualizer:
    """Visualization tools for activation evolution analysis."""

    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """Initialize activation visualizer.

        Args:
            style: Plotting style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize

        if PLOTTING_AVAILABLE:
            plt.style.use(style if style in plt.style.available else "default")
            sns.set_palette("husl")

        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    def plot_activation_evolution(
        self,
        activation_data: Dict[int, Dict[str, np.ndarray]],
        activation_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ) -> Union[Figure, None]:
        """Plot activation magnitude evolution across checkpoints.

        Args:
            activation_data: Dictionary mapping checkpoint steps to activation data
            activation_names: Specific activation points to plot
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Figure object if using matplotlib, None if using plotly
        """
        if not PLOTTING_AVAILABLE and not interactive:
            logger.error("Matplotlib not available for static plots")
            return None

        if not PLOTLY_AVAILABLE and interactive:
            logger.error("Plotly not available for interactive plots")
            return None

        # Prepare data
        steps = sorted(activation_data.keys())

        if activation_names is None:
            # Get all activation names from first checkpoint
            first_checkpoint = activation_data[steps[0]]
            activation_names = list(first_checkpoint.keys())

        # Extract magnitude evolution
        evolution_data = {}
        for act_name in activation_names:
            magnitudes = []
            for step in steps:
                if act_name in activation_data[step]:
                    magnitude = np.linalg.norm(activation_data[step][act_name])
                    magnitudes.append(magnitude)
                else:
                    magnitudes.append(np.nan)
            evolution_data[act_name] = magnitudes

        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_evolution_plot(steps, evolution_data, save_path)
        else:
            return self._create_static_evolution_plot(steps, evolution_data, save_path)

    def plot_activation_similarity_heatmap(
        self,
        similarity_data: Dict[int, Dict[str, float]],
        reference_step: int,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ) -> Union[Figure, None]:
        """Plot similarity heatmap showing how activations change relative to reference.

        Args:
            similarity_data: Similarity data from ActivationAnalyzer
            reference_step: Reference checkpoint step
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Figure object
        """
        if not PLOTTING_AVAILABLE and not interactive:
            logger.error("Matplotlib not available for static plots")
            return None

        # Prepare data matrix
        steps = sorted(similarity_data.keys())
        activation_names = list(similarity_data[steps[0]].keys())

        similarity_matrix = np.zeros((len(steps), len(activation_names)))

        for i, step in enumerate(steps):
            for j, act_name in enumerate(activation_names):
                if act_name in similarity_data[step]:
                    similarity_matrix[i, j] = similarity_data[step][act_name]
                else:
                    similarity_matrix[i, j] = np.nan

        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_heatmap(
                similarity_matrix, steps, activation_names, reference_step, save_path
            )
        else:
            return self._create_static_heatmap(similarity_matrix, steps, activation_names, reference_step, save_path)

    def plot_lora_contribution_evolution(
        self,
        lora_data: Dict[int, Dict[str, Dict[str, float]]],
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ) -> Union[Figure, None]:
        """Plot LoRA contribution evolution across checkpoints.

        Args:
            lora_data: LoRA analysis data from LoRAActivationTracker
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Figure object
        """
        if not PLOTTING_AVAILABLE and not interactive:
            logger.error("Matplotlib not available for static plots")
            return None

        # Extract LoRA contribution data
        steps = sorted(lora_data.keys())
        first_checkpoint = lora_data[steps[0]]
        module_names = list(first_checkpoint.keys())

        contribution_data = {}
        for module_name in module_names:
            lora_contributions = []
            main_contributions = []

            for step in steps:
                if module_name in lora_data[step]:
                    contrib_data = lora_data[step][module_name]
                    lora_contributions.append(contrib_data.get("lora_contribution", 0))
                    main_contributions.append(contrib_data.get("main_path_contribution", 0))
                else:
                    lora_contributions.append(np.nan)
                    main_contributions.append(np.nan)

            contribution_data[module_name] = {
                "lora_contribution": lora_contributions,
                "main_path_contribution": main_contributions,
            }

        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_lora_plot(steps, contribution_data, save_path)
        else:
            return self._create_static_lora_plot(steps, contribution_data, save_path)

    def plot_activation_distribution_evolution(
        self,
        activation_data: Dict[int, Dict[str, np.ndarray]],
        activation_name: str,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ) -> Union[Figure, None]:
        """Plot how activation distributions evolve across checkpoints.

        Args:
            activation_data: Activation data across checkpoints
            activation_name: Specific activation point to analyze
            save_path: Path to save the plot
            interactive: Whether to create interactive plot

        Returns:
            Figure object
        """
        if not PLOTTING_AVAILABLE and not interactive:
            logger.error("Matplotlib not available for static plots")
            return None

        # Extract activation distributions
        steps = sorted(activation_data.keys())
        distributions = []
        valid_steps = []

        for step in steps:
            if activation_name in activation_data[step]:
                act_tensor = activation_data[step][activation_name]
                distributions.append(act_tensor.flatten())
                valid_steps.append(step)

        if not distributions:
            logger.error(f"No data found for activation {activation_name}")
            return None

        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_distribution_plot(valid_steps, distributions, activation_name, save_path)
        else:
            return self._create_static_distribution_plot(valid_steps, distributions, activation_name, save_path)

    def create_activation_summary_dashboard(
        self,
        activation_data: Dict[int, Dict[str, np.ndarray]],
        similarity_data: Optional[Dict[int, Dict[str, float]]] = None,
        lora_data: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, None]:
        """Create comprehensive dashboard with multiple activation analysis plots.

        Args:
            activation_data: Activation evolution data
            similarity_data: Optional similarity analysis data
            lora_data: Optional LoRA analysis data
            save_path: Path to save the dashboard

        Returns:
            Figure object or None
        """
        if not PLOTTING_AVAILABLE:
            logger.error("Matplotlib not available for dashboard creation")
            return None

        # Create subplot layout
        n_plots = 2 + (1 if similarity_data else 0) + (1 if lora_data else 0)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0

        # Plot 1: Activation magnitude evolution
        self._plot_magnitude_evolution_subplot(activation_data, axes.flatten()[plot_idx])
        plot_idx += 1

        # Plot 2: Activation variance evolution
        self._plot_variance_evolution_subplot(activation_data, axes.flatten()[plot_idx])
        plot_idx += 1

        # Plot 3: Similarity heatmap (if available)
        if similarity_data:
            self._plot_similarity_heatmap_subplot(similarity_data, axes.flatten()[plot_idx])
            plot_idx += 1

        # Plot 4: LoRA contribution (if available)
        if lora_data:
            self._plot_lora_contribution_subplot(lora_data, axes.flatten()[plot_idx])
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes.flatten())):
            axes.flatten()[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Dashboard saved to {save_path}")

        return fig

    def _create_static_evolution_plot(
        self, steps: List[int], evolution_data: Dict[str, List[float]], save_path: Optional[Union[str, Path]]
    ) -> Figure:
        """Create static matplotlib evolution plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (act_name, magnitudes) in enumerate(evolution_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(steps, magnitudes, marker="o", label=act_name, color=color, linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Activation Magnitude")
        ax.set_title("Activation Magnitude Evolution Across Training")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Evolution plot saved to {save_path}")

        return fig

    def _create_interactive_evolution_plot(
        self, steps: List[int], evolution_data: Dict[str, List[float]], save_path: Optional[Union[str, Path]]
    ):
        """Create interactive plotly evolution plot."""
        fig = go.Figure()

        for act_name, magnitudes in evolution_data.items():
            fig.add_trace(
                go.Scatter(
                    x=steps, y=magnitudes, mode="lines+markers", name=act_name, line=dict(width=2), marker=dict(size=6)
                )
            )

        fig.update_layout(
            title="Activation Magnitude Evolution Across Training",
            xaxis_title="Training Step",
            yaxis_title="Activation Magnitude",
            hovermode="x unified",
            showlegend=True,
            width=1000,
            height=600,
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive evolution plot saved to {save_path}")

        return fig

    def _create_static_heatmap(
        self,
        similarity_matrix: np.ndarray,
        steps: List[int],
        activation_names: List[str],
        reference_step: int,
        save_path: Optional[Union[str, Path]],
    ) -> Figure:
        """Create static matplotlib heatmap."""
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(similarity_matrix.T, cmap="RdYlBu_r", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(step) for step in steps], rotation=45)
        ax.set_yticks(range(len(activation_names)))
        ax.set_yticklabels(activation_names)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Activation Point")
        ax.set_title(f"Activation Similarity to Reference Step {reference_step}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Cosine Similarity")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Heatmap saved to {save_path}")

        return fig

    def _create_interactive_heatmap(
        self,
        similarity_matrix: np.ndarray,
        steps: List[int],
        activation_names: List[str],
        reference_step: int,
        save_path: Optional[Union[str, Path]],
    ):
        """Create interactive plotly heatmap."""
        fig = go.Figure(
            data=go.Heatmap(
                z=similarity_matrix.T,
                x=steps,
                y=activation_names,
                colorscale="RdYlBu_r",
                colorbar=dict(title="Cosine Similarity"),
            )
        )

        fig.update_layout(
            title=f"Activation Similarity to Reference Step {reference_step}",
            xaxis_title="Training Step",
            yaxis_title="Activation Point",
            width=1000,
            height=600,
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive heatmap saved to {save_path}")

        return fig

    def _create_static_lora_plot(
        self,
        steps: List[int],
        contribution_data: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Union[str, Path]],
    ) -> Figure:
        """Create static LoRA contribution plot."""
        n_modules = len(contribution_data)
        n_cols = min(3, n_modules)
        n_rows = (n_modules + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0] * n_cols, self.figsize[1] * n_rows))
        if n_modules == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()

        for i, (module_name, data) in enumerate(contribution_data.items()):
            ax = axes[i]

            ax.plot(steps, data["lora_contribution"], marker="o", label="LoRA", color="red", linewidth=2)
            ax.plot(steps, data["main_path_contribution"], marker="s", label="Main Path", color="blue", linewidth=2)

            ax.set_xlabel("Training Step")
            ax.set_ylabel("Contribution Ratio")
            ax.set_title(f"LoRA vs Main Path: {module_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        # Hide unused subplots
        for i in range(n_modules, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"LoRA plot saved to {save_path}")

        return fig

    def _create_interactive_lora_plot(
        self,
        steps: List[int],
        contribution_data: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Union[str, Path]],
    ):
        """Create interactive LoRA contribution plot."""
        n_modules = len(contribution_data)
        subplot_titles = list(contribution_data.keys())

        fig = make_subplots(rows=(n_modules + 2) // 3, cols=min(3, n_modules), subplot_titles=subplot_titles)

        for i, (module_name, data) in enumerate(contribution_data.items()):
            row = i // 3 + 1
            col = i % 3 + 1

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=data["lora_contribution"],
                    mode="lines+markers",
                    name=f"{module_name} - LoRA",
                    line=dict(color="red", width=2),
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=data["main_path_contribution"],
                    mode="lines+markers",
                    name=f"{module_name} - Main Path",
                    line=dict(color="blue", width=2),
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="LoRA vs Main Path Contribution Evolution", height=400 * ((n_modules + 2) // 3), showlegend=True
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive LoRA plot saved to {save_path}")

        return fig

    def _create_static_distribution_plot(
        self,
        steps: List[int],
        distributions: List[np.ndarray],
        activation_name: str,
        save_path: Optional[Union[str, Path]],
    ) -> Figure:
        """Create static distribution evolution plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create violin plot
        positions = range(len(steps))
        parts = ax.violinplot(distributions, positions=positions, showmeans=True, showmedians=True)

        # Customize colors
        if "bodies" in parts:
            bodies = parts["bodies"]
            # bodies is a list of PolyCollection objects
            for pc in bodies:  # type: ignore
                pc.set_facecolor("lightblue")
                pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(step) for step in steps], rotation=45)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Activation Value")
        ax.set_title(f"Distribution Evolution: {activation_name}")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Distribution plot saved to {save_path}")

        return fig

    def _create_interactive_distribution_plot(
        self,
        steps: List[int],
        distributions: List[np.ndarray],
        activation_name: str,
        save_path: Optional[Union[str, Path]],
    ):
        """Create interactive distribution evolution plot."""
        fig = go.Figure()

        for i, (step, dist) in enumerate(zip(steps, distributions)):
            fig.add_trace(go.Violin(y=dist, name=f"Step {step}", x0=i, box_visible=True, meanline_visible=True))

        fig.update_layout(
            title=f"Distribution Evolution: {activation_name}",
            xaxis_title="Training Step",
            yaxis_title="Activation Value",
            width=1000,
            height=600,
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive distribution plot saved to {save_path}")

        return fig

    def _plot_magnitude_evolution_subplot(self, activation_data: Dict[int, Dict[str, np.ndarray]], ax: Axes) -> None:
        """Plot magnitude evolution in a subplot."""
        steps = sorted(activation_data.keys())
        first_checkpoint = activation_data[steps[0]]
        activation_names = list(first_checkpoint.keys())[:5]  # Limit to first 5 for readability

        for i, act_name in enumerate(activation_names):
            magnitudes = []
            for step in steps:
                if act_name in activation_data[step]:
                    magnitude = np.linalg.norm(activation_data[step][act_name])
                    magnitudes.append(magnitude)
                else:
                    magnitudes.append(np.nan)

            color = self.colors[i % len(self.colors)]
            ax.plot(steps, magnitudes, marker="o", label=act_name, color=color)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Magnitude")
        ax.set_title("Activation Magnitude Evolution")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    def _plot_variance_evolution_subplot(self, activation_data: Dict[int, Dict[str, np.ndarray]], ax: Axes) -> None:
        """Plot variance evolution in a subplot."""
        steps = sorted(activation_data.keys())
        first_checkpoint = activation_data[steps[0]]
        activation_names = list(first_checkpoint.keys())[:5]  # Limit to first 5 for readability

        for i, act_name in enumerate(activation_names):
            variances = []
            for step in steps:
                if act_name in activation_data[step]:
                    variance = np.var(activation_data[step][act_name])
                    variances.append(variance)
                else:
                    variances.append(np.nan)

            color = self.colors[i % len(self.colors)]
            ax.plot(steps, variances, marker="s", label=act_name, color=color)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Variance")
        ax.set_title("Activation Variance Evolution")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    def _plot_similarity_heatmap_subplot(self, similarity_data: Dict[int, Dict[str, float]], ax: Axes) -> None:
        """Plot similarity heatmap in a subplot."""
        steps = sorted(similarity_data.keys())
        activation_names = list(similarity_data[steps[0]].keys())[:10]  # Limit for readability

        similarity_matrix = np.zeros((len(steps), len(activation_names)))

        for i, step in enumerate(steps):
            for j, act_name in enumerate(activation_names):
                if act_name in similarity_data[step]:
                    similarity_matrix[i, j] = similarity_data[step][act_name]

        ax.imshow(similarity_matrix.T, cmap="RdYlBu_r", aspect="auto")

        ax.set_xticks(range(0, len(steps), max(1, len(steps) // 5)))
        ax.set_xticklabels([str(steps[i]) for i in range(0, len(steps), max(1, len(steps) // 5))])
        ax.set_yticks(range(len(activation_names)))
        ax.set_yticklabels([name[:15] + "..." if len(name) > 15 else name for name in activation_names])

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Activation Point")
        ax.set_title("Activation Similarity Heatmap")

    def _plot_lora_contribution_subplot(self, lora_data: Dict[int, Dict[str, Dict[str, float]]], ax: Axes) -> None:
        """Plot LoRA contribution in a subplot."""
        steps = sorted(lora_data.keys())
        first_checkpoint = lora_data[steps[0]]
        module_names = list(first_checkpoint.keys())[:3]  # Limit for readability

        for i, module_name in enumerate(module_names):
            lora_contributions = []
            for step in steps:
                if module_name in lora_data[step]:
                    contrib = lora_data[step][module_name].get("lora_contribution", 0)
                    lora_contributions.append(contrib)
                else:
                    lora_contributions.append(np.nan)

            color = self.colors[i % len(self.colors)]
            ax.plot(steps, lora_contributions, marker="o", label=module_name, color=color)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("LoRA Contribution")
        ax.set_title("LoRA Contribution Evolution")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
