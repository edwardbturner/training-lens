"""CI-specific tests for Unsloth compatibility layer without heavy dependencies."""

import pytest
import torch

from training_lens.utils.unsloth_compat import (
    is_bfloat16_supported,
    is_unsloth_available,
    is_peft_available,
    get_backend_info,
)


@pytest.mark.ci
class TestUnslothCompatCI:
    """Test Unsloth compatibility functions in CI environment."""
    
    def test_is_bfloat16_supported_ci(self):
        """Test bfloat16 support detection in CI."""
        # Should return a boolean
        result = is_bfloat16_supported()
        assert isinstance(result, bool)
        
        # In CI, this will depend on the runner's hardware
        # But we can at least verify the function works
    
    def test_backend_availability_ci(self):
        """Test backend availability checks in CI."""
        # PEFT should be available as it's a dependency
        assert is_peft_available() is True
        
        # Unsloth will likely not be available in CI
        unsloth_available = is_unsloth_available()
        assert isinstance(unsloth_available, bool)
        
        # In CI, we expect Unsloth to not be available
        # This is actually what we want to test - that the code works without it
    
    def test_get_backend_info_ci(self):
        """Test backend info retrieval in CI environment."""
        info = get_backend_info()
        
        # Verify structure
        assert isinstance(info, dict)
        assert "unsloth_available" in info
        assert "peft_available" in info
        assert "bfloat16_supported" in info
        assert "cuda_available" in info
        assert "device" in info
        
        # PEFT should be available
        assert info["peft_available"] is True
        
        # Device should be a string
        assert isinstance(info["device"], str)
        
        # Device could be various values depending on environment
        # Just verify it's not empty
        assert len(info["device"]) > 0
    
    def test_backend_info_consistency(self):
        """Test that backend info is consistent across calls."""
        info1 = get_backend_info()
        info2 = get_backend_info()
        
        # Should return the same values
        assert info1 == info2
    
    def test_cuda_detection_ci(self):
        """Test CUDA detection in CI."""
        info = get_backend_info()
        
        # CUDA availability should match PyTorch's detection
        assert info["cuda_available"] == torch.cuda.is_available()
        
        # If CUDA is not available, device should be CPU
        if not info["cuda_available"]:
            assert info["device"] == "cpu"
    
    def test_compatibility_without_unsloth(self):
        """Test that all compatibility functions work regardless of Unsloth availability."""
        # This is the key test for CI - everything should work with or without Unsloth
        
        # All these should work and not raise exceptions
        bfloat16_support = is_bfloat16_supported()
        unsloth_check = is_unsloth_available()
        peft_check = is_peft_available()
        backend_info = get_backend_info()
        
        # Basic sanity checks
        assert isinstance(bfloat16_support, bool)
        assert isinstance(unsloth_check, bool)
        assert isinstance(peft_check, bool)
        assert isinstance(backend_info, dict)
        
        # Unsloth availability should match between functions
        assert unsloth_check == backend_info["unsloth_available"]
    
    def test_device_selection_ci(self):
        """Test device selection logic in CI."""
        info = get_backend_info()
        
        # Device should be one of the valid options
        valid_devices = ["cpu", "cuda", "mps"]
        assert info["device"] in valid_devices
        
        # If CUDA is available, device should be cuda
        if torch.cuda.is_available():
            assert info["device"] == "cuda"
        # If MPS is available (Mac), device should be mps
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert info["device"] == "mps"
        # Otherwise, should be CPU
        else:
            assert info["device"] == "cpu"
    
    def test_bfloat16_on_cpu(self):
        """Test bfloat16 support detection on CPU."""
        # Create a simple test to verify bfloat16 handling
        if is_bfloat16_supported():
            # Should be able to create bfloat16 tensors
            try:
                tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
                assert tensor.dtype == torch.bfloat16
            except Exception:
                # If creation fails, the detection was wrong
                pytest.fail("bfloat16 reported as supported but tensor creation failed")
        else:
            # Should not be able to create bfloat16 tensors on this platform
            pass