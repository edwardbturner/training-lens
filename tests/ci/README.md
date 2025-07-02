# CI Tests for Training Lens

This directory contains lightweight tests specifically designed for continuous integration (CI) environments where heavy dependencies like Unsloth may not be available.

## Overview

These tests mirror the functionality of the main unit tests but use simplified fixtures and mock objects that don't require GPU resources or heavy ML frameworks.

## Test Files

- `test_collectors_ci.py` - Tests for data collectors (adapter weights, gradients)
- `test_collector_registry_ci.py` - Tests for the collector registry system
- `test_checkpoint_manager_ci.py` - Tests for checkpoint management
- `test_unsloth_compat_ci.py` - Tests for Unsloth compatibility layer
- `test_config_ci.py` - Tests for configuration handling
- `test_utils_ci.py` - Tests for utility functions
- `test_data_collectors_integration_ci.py` - Integration tests for collectors

## Running CI Tests

### Run all CI tests:
```bash
pytest tests/ci -m ci -v
```

### Run with the provided runner script:
```bash
python tests/ci/run_ci_tests.py
```

### Run specific test file:
```bash
python tests/ci/run_ci_tests.py collectors_ci
```

### Run with coverage:
```bash
python tests/ci/run_ci_tests.py --coverage
```

## Key Differences from Main Tests

1. **Lightweight Fixtures**: Uses `simple_model` and `simple_tokenizer` instead of full Transformers/Unsloth models
2. **No GPU Required**: All tests run on CPU
3. **Fast Execution**: Designed for quick CI runs
4. **Minimal Dependencies**: Only requires core Python packages and PyTorch

## CI Markers

All tests are marked with `@pytest.mark.ci` to allow selective execution in CI pipelines.

## Fixtures

The CI tests use special fixtures defined in `conftest.py`:

- `simple_model`: A lightweight model that simulates LoRA structure without PEFT
- `simple_tokenizer`: A mock tokenizer without Transformers dependency  
- `simple_optimizer`: A basic Adam optimizer for the simple model
- `simple_training_config`: Minimal training configuration for CI

## GitHub Actions Integration

These tests are designed to run in GitHub Actions where:
- Unsloth is not installed
- Only CPU runners are available
- Fast execution is critical
- Test isolation is important