"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Any

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rich_console: bool = True,
) -> logging.Logger:
    """Set up logging configuration for training-lens.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        rich_console: Whether to use rich formatting for console output

    Returns:
        Configured logger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create logger
    logger = logging.getLogger("training_lens")
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    if rich_console:
        console = Console()
        console_handler: logging.Handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        console_format = "%(message)s"
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    console_handler.setLevel(level)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "training_lens") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class TrainingLogger:
    """Enhanced logger for training processes with progress tracking."""

    def __init__(
        self,
        name: str = "training_lens",
        log_file: Optional[Union[str, Path]] = None,
    ):
        self.logger = logging.getLogger(name)
        self.console = Console()

        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def log_training_step(
        self, step: int, loss: float, learning_rate: float, grad_norm: Optional[float] = None, **metrics: Any
    ) -> None:
        """Log training step information."""
        msg = f"Step {step:>6d} | Loss: {loss:.4f} | LR: {learning_rate:.2e}"

        if grad_norm is not None:
            msg += f" | Grad Norm: {grad_norm:.4f}"

        if metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            msg += f" | {metric_str}"

        self.info(msg)

    def log_checkpoint_saved(self, step: int, path: Path) -> None:
        """Log checkpoint save information."""
        self.info(f"Checkpoint saved at step {step}: {path}")

    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        from .helpers import get_gpu_memory_usage, get_memory_usage

        memory = get_memory_usage()
        msg = f"Memory: {memory['rss']} ({memory['percent']})"

        gpu_memory = get_gpu_memory_usage()
        if gpu_memory:
            msg += f" | GPU: {gpu_memory['allocated']} ({gpu_memory['percent']})"

        self.debug(msg)

    def print_banner(self, title: str) -> None:
        """Print a formatted banner."""
        self.console.rule(f"[bold blue]{title}[/bold blue]")

    def print_table(self, data: dict, title: str = "Metrics") -> None:
        """Print data in a formatted table."""
        from rich.table import Table

        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)
