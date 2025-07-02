"""Shared pytest fixtures and configuration for training-lens tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from transformers import AutoConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Create a mock model with LoRA layers for testing."""
    
    class MockLoRALayer(torch.nn.Module):
        def __init__(self, in_features=768, out_features=768, rank=16):
            super().__init__()
            # Create actual nn.Linear modules as attributes
            self._lora_A_default = torch.nn.Linear(in_features, rank, bias=False)
            self._lora_B_default = torch.nn.Linear(rank, out_features, bias=False)
            
            # Create dict-like access for compatibility
            self.lora_A = {"default": self._lora_A_default}
            self.lora_B = {"default": self._lora_B_default}
            self.scaling = {"default": 1.0}
            
            # Initialize weights
            torch.nn.init.normal_(self._lora_A_default.weight)
            torch.nn.init.zeros_(self._lora_B_default.weight)
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.layers = torch.nn.ModuleList([
                MockLoRALayer() for _ in range(4)
            ])
            self.config = AutoConfig.from_pretrained("bert-base-uncased")
            
        def save_pretrained(self, save_directory):
            """Mock save_pretrained for PEFT compatibility."""
            import os
            os.makedirs(save_directory, exist_ok=True)
            # Save a dummy file to simulate adapter save
            with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
                f.write('{"peft_type": "LORA"}')
                
    return MockModel()


@pytest.fixture
def mock_optimizer(mock_model):
    """Create a mock optimizer for the model."""
    return torch.optim.AdamW(mock_model.parameters(), lr=1e-4)


@pytest.fixture
def sample_checkpoint_metadata() -> Dict[str, Any]:
    """Sample checkpoint metadata for testing."""
    return {
        "step": 100,
        "epoch": 1.0,
        "learning_rate": 1e-4,
        "train_loss": 2.5,
        "eval_loss": 2.3,
        "grad_norm": 1.2,
    }


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """Sample training configuration for testing."""
    return {
        "model_name": "bert-base-uncased",
        "training_method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "learning_rate": 2e-4,
        "max_steps": 1000,
        "checkpoint_interval": 100,
        "output_dir": "./test_output",
        "capture_adapter_weights": True,
        "capture_adapter_gradients": True,
    }


@pytest.fixture
def mock_lora_components() -> Dict[str, Dict[str, torch.Tensor]]:
    """Mock LoRA components for testing collectors."""
    components = {}
    for i in range(4):
        layer_name = f"model.layers.{i}.self_attn.q_proj"
        components[layer_name] = {
            "lora_A": torch.randn(16, 768),  # rank x in_features
            "lora_B": torch.randn(768, 16),  # out_features x rank
            "scaling": torch.tensor(1.0),
        }
    return components