"""LoRA tracking components (from legacy specialized analysis).

This module provides specialized LoRA tracking classes.
"""

# Import classes from the original specialized module
import importlib.util
from pathlib import Path

# Load the original specialized LoRA analyzer
original_module_path = Path(__file__).parent.parent.parent / "analysis" / "specialized" / "lora_analyzer.py"

if original_module_path.exists():
    spec = importlib.util.spec_from_file_location("original_lora", original_module_path)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)

    # Extract the specialized classes
    LoRAActivationTracker = getattr(original_module, "LoRAActivationTracker", None)
    LoRAParameterAnalyzer = getattr(original_module, "LoRAParameterAnalyzer", None)
else:
    LoRAActivationTracker = None
    LoRAParameterAnalyzer = None

# Fallback classes if not found
if LoRAActivationTracker is None:

    class LoRAActivationTracker:
        """Fallback LoRAActivationTracker class."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("LoRAActivationTracker not available")


if LoRAParameterAnalyzer is None:

    class LoRAParameterAnalyzer:
        """Fallback LoRAParameterAnalyzer class."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("LoRAParameterAnalyzer not available")


__all__ = ["LoRAActivationTracker", "LoRAParameterAnalyzer"]
