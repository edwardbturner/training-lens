"""Lightweight fixtures for CI testing without heavy dependencies."""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any


@pytest.fixture
def simple_model():
    """Create a simple model without requiring PEFT or Unsloth."""
    
    class SimpleLoRALayer(nn.Module):
        def __init__(self, in_features=768, out_features=768, rank=16):
            super().__init__()
            # Simulate LoRA structure
            self.base_layer = nn.Linear(in_features, out_features)
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.scaling = 1.0
            
            # Initialize
            nn.init.kaiming_uniform_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)
            
        def forward(self, x):
            base_out = self.base_layer(x)
            lora_out = self.lora_B(self.lora_A(x)) * self.scaling
            return base_out + lora_out
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleDict({
                'h': nn.ModuleList([
                    nn.ModuleDict({
                        'self_attn': SimpleLoRALayer(768, 768),
                        'mlp': SimpleLoRALayer(768, 768)
                    }) for _ in range(2)  # Just 2 layers for speed
                ])
            })
            self.config = type('Config', (), {
                'hidden_size': 768,
                'num_hidden_layers': 2,
                'model_type': 'gpt2'
            })()
            
        def save_pretrained(self, path):
            """Mock save_pretrained for compatibility."""
            import os
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
            
        def parameters_with_lora(self):
            """Get parameters that simulate LoRA parameters."""
            for name, param in self.named_parameters():
                if 'lora_' in name:
                    yield param
                    
    return SimpleModel()


@pytest.fixture  
def simple_tokenizer():
    """Create a simple mock tokenizer."""
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "[PAD]"
            self.eos_token = "[EOS]"
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.vocab_size = 30000
            
        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            if isinstance(texts, str):
                texts = [texts]
            
            max_length = kwargs.get('max_length', 128)
            input_ids = []
            
            for text in texts:
                # Mock tokenization: just use character codes
                ids = [ord(c) % self.vocab_size for c in text[:max_length]]
                ids = ids[:max_length]
                # Pad if needed
                if kwargs.get('padding'):
                    ids += [self.pad_token_id] * (max_length - len(ids))
                input_ids.append(ids)
            
            return {
                'input_ids': input_ids,
                'attention_mask': [[1 if id != self.pad_token_id else 0 for id in ids] for ids in input_ids]
            }
            
        def save_pretrained(self, path):
            """Mock save."""
            import os
            os.makedirs(path, exist_ok=True)
            
        def apply_chat_template(self, messages, **kwargs):
            """Simple chat template."""
            text = ""
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n"
            return text
    
    return MockTokenizer()


@pytest.fixture
def simple_optimizer(simple_model):
    """Create optimizer for simple model."""
    # Only optimize LoRA parameters
    lora_params = [p for n, p in simple_model.named_parameters() if 'lora_' in n]
    return torch.optim.Adam(lora_params, lr=1e-4)


@pytest.fixture
def simple_training_config() -> Dict[str, Any]:
    """Minimal training configuration."""
    return {
        "model_name": "gpt2",  # Using standard model name
        "training_method": "lora",
        "lora_r": 8,  # Smaller rank for CI
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "learning_rate": 2e-4,
        "max_steps": 10,  # Very few steps for CI
        "checkpoint_interval": 5,
        "output_dir": "./ci_test_output",
        "capture_adapter_weights": True,
        "capture_adapter_gradients": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
    }


@pytest.fixture
def mock_dataset():
    """Create a tiny mock dataset."""
    return [
        {"text": "Hello world", "labels": [1, 2]},
        {"text": "Test example", "labels": [3, 4]},
        {"text": "Another test", "labels": [5, 6]},
        {"text": "Final example", "labels": [7, 8]},
    ]