"""Unit tests for Unsloth compatibility layer."""


from training_lens.utils.unsloth_compat import (
    get_backend_info,
    is_bfloat16_supported,
    is_peft_available,
    is_unsloth_available,
)


class TestUnslothCompat:
    """Test Unsloth compatibility functions."""

    def test_is_bfloat16_supported(self):
        """Test bfloat16 support detection."""
        # Should return a boolean
        result = is_bfloat16_supported()
        assert isinstance(result, bool)

    def test_backend_availability(self):
        """Test backend availability checks."""
        # Should at least have PEFT available (it's a core dependency now)
        assert is_peft_available() is True

        # Unsloth may or may not be available
        assert isinstance(is_unsloth_available(), bool)

    def test_get_backend_info(self):
        """Test backend info retrieval."""
        info = get_backend_info()

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
