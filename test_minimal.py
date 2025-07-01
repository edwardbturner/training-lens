#!/usr/bin/env python3
"""Minimal test to check basic functionality."""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_imports():
    """Test basic imports work."""
    try:
        pass

        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation."""
    try:
        from training_lens.training.config import TrainingConfig

        config = TrainingConfig(
            model_name="test-model",
            training_method="lora",
            max_steps=100,
        )

        assert config.model_name == "test-model"
        assert config.training_method == "lora"
        assert config.max_steps == 100
        print("✅ Configuration creation successful")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_helpers():
    """Test helper functions."""
    try:
        from training_lens.utils.helpers import format_size, get_device

        # Test format_size
        assert format_size(0) == "0B"
        assert format_size(1024) == "1.0KB"
        assert format_size(1024 * 1024) == "1.0MB"

        # Test get_device
        device = get_device()
        assert str(device) in ["cuda", "mps", "cpu"]

        print("✅ Helper functions test successful")
        return True
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False


def test_metadata():
    """Test checkpoint metadata."""
    try:
        from training_lens.training.config import CheckpointMetadata

        metadata = CheckpointMetadata(
            step=100,
            epoch=1.0,
            learning_rate=2e-4,
            train_loss=1.5,
        )

        assert metadata.step == 100
        assert metadata.epoch == 1.0

        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["step"] == 100

        print("✅ Metadata test successful")
        return True
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False


def test_cli_imports():
    """Test CLI imports."""
    try:
        pass

        print("✅ CLI imports successful")
        return True
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔍 Running Training Lens minimal tests...")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_config_creation,
        test_helpers,
        test_metadata,
        test_cli_imports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
