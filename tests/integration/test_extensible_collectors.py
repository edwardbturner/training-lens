"""Integration tests for extensible collector system."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

from training_lens.core.base import DataCollector, DataType
from training_lens.core.collector_registry import register_collector
from training_lens.training.metrics_collector_v2 import MetricsCollectorV2
from training_lens.training.config import TrainingConfig, CheckpointMetadata
from training_lens.training.checkpoint_manager import CheckpointManager


class CustomTestCollector(DataCollector):
    """Custom collector for integration testing."""
    
    @property
    def data_type(self) -> DataType:
        return DataType.PARAMETER_NORMS
    
    @property
    def supported_model_types(self) -> List[str]:
        return ["all"]
    
    def can_collect(self, model: torch.nn.Module, step: int) -> bool:
        return step % 50 == 0
    
    def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
        norms = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                norm = float(param.data.norm())
                norms[name] = norm
                total_norm += norm ** 2
        
        return {
            "step": step,
            "parameter_norms": norms,
            "total_norm": float(total_norm ** 0.5),
            "num_parameters": len(norms),
        }
    
    def get_metrics(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics for logging."""
        return {
            "total_param_norm": collected_data.get("total_norm", 0),
            "num_trainable_params": collected_data.get("num_parameters", 0),
        }


class TestExtensibleCollectorIntegration:
    """Integration tests for the extensible collector system."""
    
    def test_full_training_integration(self, temp_dir, mock_model, mock_optimizer):
        """Test collectors work through full training workflow."""
        # Register custom collector
        register_collector(
            DataType.PARAMETER_NORMS,
            CustomTestCollector,
            enabled=True
        )
        
        # Create metrics collector with multiple collectors
        metrics_collector = MetricsCollectorV2(
            enabled_collectors={
                DataType.ADAPTER_WEIGHTS,
                DataType.ADAPTER_GRADIENTS,
                DataType.PARAMETER_NORMS,  # Our custom collector
            }
        )
        
        # Setup with model
        metrics_collector.setup(mock_model, mock_optimizer)
        
        # Simulate training steps
        collected_steps = []
        for step in [50, 100, 150, 200]:
            # Simulate backward pass
            loss = torch.sum(torch.stack([
                torch.sum(layer.lora_A["default"].weight)
                for layer in mock_model.base_model.layers
            ]))
            loss.backward()
            
            # Collect metrics
            metrics = metrics_collector.collect_step_metrics(
                step=step,
                logs={"train_loss": float(loss), "learning_rate": 1e-4}
            )
            
            collected_steps.append(step)
            
            # Verify metrics include custom collector data
            assert "total_param_norm" in metrics
            assert "num_trainable_params" in metrics
            
            # Verify data was collected
            param_norm_data = metrics_collector.get_collected_data(step, DataType.PARAMETER_NORMS)
            assert param_norm_data is not None
            assert param_norm_data["step"] == step
            assert "parameter_norms" in param_norm_data
            
            # Clear gradients
            mock_optimizer.zero_grad()
        
        # Test checkpoint data includes custom collector
        checkpoint_data = metrics_collector.get_checkpoint_data()
        assert "latest_collected_data" in checkpoint_data
        assert "parameter_norms" in checkpoint_data["latest_collected_data"]
        
        # Verify summary includes custom collector info
        summary = metrics_collector.export_metrics_summary()
        assert DataType.PARAMETER_NORMS.value in summary["enabled_collectors"]
    
    def test_runtime_collector_management(self, mock_model, mock_optimizer):
        """Test adding and removing collectors at runtime."""
        # Start with minimal collectors
        metrics_collector = MetricsCollectorV2(
            enabled_collectors={DataType.ADAPTER_WEIGHTS}
        )
        metrics_collector.setup(mock_model, mock_optimizer)
        
        # Collect initial metrics
        metrics1 = metrics_collector.collect_step_metrics(
            step=100,
            logs={"train_loss": 2.5}
        )
        
        # Should not have parameter norms yet
        assert "total_param_norm" not in metrics1
        
        # Add custom collector at runtime
        metrics_collector.add_collector(
            DataType.PARAMETER_NORMS,
            CustomTestCollector
        )
        
        # Collect metrics again
        metrics2 = metrics_collector.collect_step_metrics(
            step=150,
            logs={"train_loss": 2.3}
        )
        
        # Now should have parameter norms
        assert "total_param_norm" in metrics2
        
        # Remove collector
        metrics_collector.remove_collector(DataType.PARAMETER_NORMS)
        
        # Collect metrics again
        metrics3 = metrics_collector.collect_step_metrics(
            step=200,
            logs={"train_loss": 2.1}
        )
        
        # Should not have parameter norms anymore
        assert "total_param_norm" not in metrics3
    
    def test_collector_error_handling(self, mock_model, mock_optimizer):
        """Test that collector errors don't break training."""
        
        class ErrorCollector(DataCollector):
            """Collector that raises errors."""
            
            @property
            def data_type(self) -> DataType:
                return DataType.LOSS_LANDSCAPES
            
            @property
            def supported_model_types(self) -> List[str]:
                return ["all"]
            
            def can_collect(self, model: torch.nn.Module, step: int) -> bool:
                return True
            
            def collect(self, model: torch.nn.Module, step: int, **kwargs) -> Optional[Dict[str, Any]]:
                raise RuntimeError("Test error in collector")
        
        # Register error collector
        register_collector(DataType.LOSS_LANDSCAPES, ErrorCollector, enabled=True)
        
        # Create metrics collector
        metrics_collector = MetricsCollectorV2(
            enabled_collectors={
                DataType.ADAPTER_WEIGHTS,
                DataType.LOSS_LANDSCAPES,  # Error collector
            }
        )
        metrics_collector.setup(mock_model, mock_optimizer)
        
        # Collecting should not raise error
        metrics = metrics_collector.collect_step_metrics(
            step=100,
            logs={"train_loss": 2.5}
        )
        
        # Should still have other metrics
        assert "train_loss" in metrics
        assert metrics["step"] == 100
        
        # Error collector data should be None
        error_data = metrics_collector.get_collected_data(100, DataType.LOSS_LANDSCAPES)
        assert error_data is None
    
    def test_checkpoint_integration(self, temp_dir, mock_model, mock_optimizer):
        """Test that collected data integrates with checkpoint system."""
        # Register custom collector
        register_collector(DataType.PARAMETER_NORMS, CustomTestCollector, enabled=True)
        
        # Create components
        checkpoint_manager = CheckpointManager(output_dir=temp_dir)
        metrics_collector = MetricsCollectorV2(
            enabled_collectors={
                DataType.ADAPTER_WEIGHTS,
                DataType.PARAMETER_NORMS,
            }
        )
        metrics_collector.setup(mock_model, mock_optimizer)
        
        # Collect metrics
        metrics = metrics_collector.collect_step_metrics(
            step=100,
            logs={"train_loss": 2.5}
        )
        
        # Save checkpoint with collector data
        metadata = CheckpointMetadata(
            step=100,
            epoch=1.0,
            learning_rate=1e-4,
            train_loss=2.5,
        )
        
        checkpoint_path = checkpoint_manager.save_lora_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            metadata=metadata,
            additional_data=metrics_collector.get_checkpoint_data()
        )
        
        # Load checkpoint and verify collector data
        loaded = checkpoint_manager.load_checkpoint(step=100)
        assert loaded["additional_data"] is not None
        
        additional_data = loaded["additional_data"]
        assert "latest_collected_data" in additional_data
        assert "parameter_norms" in additional_data["latest_collected_data"]
        assert "adapter_weights" in additional_data["latest_collected_data"]
    
    def test_collector_configuration(self, mock_model, mock_optimizer):
        """Test collector configuration system."""
        # Create metrics collector with custom configs
        collector_configs = {
            DataType.PARAMETER_NORMS: {
                "include_frozen": False,
                "compute_spectral_norm": True,
            }
        }
        
        metrics_collector = MetricsCollectorV2(
            enabled_collectors={DataType.PARAMETER_NORMS},
            collector_configs=collector_configs
        )
        
        # Register collector
        register_collector(DataType.PARAMETER_NORMS, CustomTestCollector)
        
        # Get collector and check config
        collector = metrics_collector.registry.get_collector(DataType.PARAMETER_NORMS)
        assert collector is not None
        assert collector.config["include_frozen"] is False
        assert collector.config["compute_spectral_norm"] is True