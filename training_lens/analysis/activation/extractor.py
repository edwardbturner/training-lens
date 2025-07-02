"""Activation extraction utilities (from legacy analysis module).

This module provides the ActivationExtractor class for detailed activation extraction.
"""

import importlib.util

# Import the ActivationExtractor class from the original activation_analyzer.py
from pathlib import Path

# Load the original module to extract ActivationExtractor
original_module_path = Path(__file__).parent.parent.parent / "analysis" / "activation_analyzer.py"

if original_module_path.exists():
    spec = importlib.util.spec_from_file_location("original_activation", original_module_path)
    if spec is not None and spec.loader is not None:
        original_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_module)
    else:
        # Fallback if spec creation failed
        original_module = None

    # Extract the ActivationExtractor class
    if original_module is not None and hasattr(original_module, "ActivationExtractor"):
        ActivationExtractor = original_module.ActivationExtractor
    else:
        # Fallback if class not found
        class ActivationExtractor:
            """Fallback ActivationExtractor class."""

            def __init__(self, *args, **kwargs):
                raise NotImplementedError("ActivationExtractor not available")

else:
    # Fallback if file not found
    class _FallbackActivationExtractor:
        """Fallback ActivationExtractor class."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ActivationExtractor not available")

    ActivationExtractor = _FallbackActivationExtractor


__all__ = ["ActivationExtractor"]
