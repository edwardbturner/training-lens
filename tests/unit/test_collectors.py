"""Unit tests for data collectors."""

import pytest
import torch

from training_lens.collectors.adapter_weights import AdapterWeightsCollector
from training_lens.collectors.adapter_gradients import AdapterGradientsCollector
from training_lens.core.base import DataType


class TestAdapterWeightsCollector:
    """Test adapter weights collector."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = AdapterWeightsCollector()
        assert collector.data_type == DataType.ADAPTER_WEIGHTS
        assert "lora" in collector.supported_model_types

    def test_can_collect(self, mock_model):
        """Test can_collect method."""
        collector = AdapterWeightsCollector()
        assert collector.can_collect(mock_model, step=100) is True

        # Test with non-LoRA model
        regular_model = torch.nn.Linear(10, 10)
        assert collector.can_collect(regular_model, step=100) is False

    def test_collect_adapter_weights(self, mock_model):
        """Test collecting adapter weights."""
        collector = AdapterWeightsCollector()

        result = collector.collect(mock_model, step=100)

        assert result is not None
        assert result["step"] == 100
        assert result["adapter_name"] == "default"
        assert "adapter_weights" in result
        assert result["source"] == "model_inspection"

        # Check collected weights
        weights = result["adapter_weights"]
        assert len(weights) > 0

        for layer_name, layer_data in weights.items():
            assert "lora_A" in layer_data
            assert "lora_B" in layer_data
            assert "shape_A" in layer_data
            assert "shape_B" in layer_data
            assert "effective_weight" in layer_data
            assert "statistics" in layer_data

            # Verify statistics
            stats = layer_data["statistics"]
            assert "A_norm" in stats
            assert "B_norm" in stats
            assert "effective_norm" in stats


class TestAdapterGradientsCollector:
    """Test adapter gradients collector."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = AdapterGradientsCollector()
        assert collector.data_type == DataType.ADAPTER_GRADIENTS

    def test_collect_gradients_without_optimizer(self, mock_model):
        """Test gradient collection fails without optimizer."""
        collector = AdapterGradientsCollector()

        # Should return None without optimizer
        result = collector.collect(mock_model, step=100)
        assert result is None

    def test_collect_gradients_with_optimizer(self, mock_model, mock_optimizer):
        """Test gradient collection with optimizer."""
        collector = AdapterGradientsCollector()

        # Simulate backward pass to create gradients
        loss = torch.sum(
            torch.stack(
                [
                    torch.sum(layer._lora_A_default.weight) + torch.sum(layer._lora_B_default.weight)
                    for layer in mock_model.base_model.layers
                ]
            )
        )
        loss.backward()

        # Collect gradients - pass optimizer through kwargs
        result = collector.collect(mock_model, step=100, optimizer=mock_optimizer)

        assert result is not None
        assert result["step"] == 100
        assert "adapter_gradients" in result

        gradients = result["adapter_gradients"]
        assert len(gradients) > 0

        for layer_name, grad_data in gradients.items():
            # Check for gradient data (could be A_gradient or lora_A_grad)
            assert "A_gradient" in grad_data or "lora_A_grad" in grad_data
            assert "B_gradient" in grad_data or "lora_B_grad" in grad_data

            # Check for statistics
            assert "A_grad_norm" in grad_data
            assert "B_grad_norm" in grad_data
