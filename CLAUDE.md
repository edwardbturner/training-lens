# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`training-lens` is a LoRA-focused library for interpreting fine-tune training runs of models. It provides comprehensive monitoring and analysis tools for LoRA adapter training, with deep insights into how LoRA adapters evolve during the fine-tuning process. The library is integrated with Unsloth for optimal LoRA training performance.

## Repository Structure

### Core Directories
- `training_lens/` - Main Python package
  - `training/` - Core training components (wrapper, checkpoint manager, metrics collector)
  - `analysis/` - Analysis and reporting tools for checkpoints and training data
  - `collectors/` - Data collection modules for various metrics
  - `config/` - Configuration modules for training and LoRA
  - `core/` - Core functionality including base classes and registry
  - `integrations/` - External service integrations (HuggingFace, W&B)
  - `cli/` - Command-line interface
  - `utils/` - Utility functions and helpers

### Documentation
- `README.md` - Comprehensive project documentation with usage examples
- `CHANGELOG.md` - Version history
- `specs/` - Technical specifications
- `ai_docs/` - AI memory and implementation notes

### Testing
- `tests/` - Comprehensive test suite
  - `unit/` - Unit tests
  - `integration/` - Integration tests
  - `ci/` - CI-specific tests
  - `fixtures/` - Test fixtures

### Configuration
- `pyproject.toml` - Modern Python packaging configuration
- `pytest.ini` - Test configuration
- `.flake8` - Linting configuration

## Development Status

The project is fully developed with:
- Python 3.10+ support
- Modern packaging with pyproject.toml
- Comprehensive test coverage (100% of tests passing)
- Integration with Unsloth for optimized LoRA training
- Support for HuggingFace Hub and Weights & Biases
- Extensible collector system for custom metrics
- CLI tools for training, analysis, and export

## Key Features

1. **LoRA Training Wrapper**: Comprehensive monitoring during LoRA fine-tuning
2. **Checkpoint Management**: Efficient storage and retrieval of training checkpoints
3. **Metrics Collection**: Extensible system for collecting various training metrics
4. **Analysis Tools**: Deep analysis of LoRA adapter evolution
5. **Integration Support**: HuggingFace Hub and W&B integration
6. **CLI Interface**: Command-line tools for all major operations

## Development Guidelines

### Code Style
- Follow PEP 8 with 120 character line limit
- Use type hints for all function signatures
- Write comprehensive docstrings for all public APIs
- Avoid adding comments unless specifically requested

### Testing
- Run tests with: `pytest`
- Check linting with: `flake8 training_lens/`
- All tests must pass before committing

### Key Commands
- Install: `pip install -e .`
- Run tests: `pytest`
- Lint: `flake8 training_lens/`
- Train: `training-lens train --config config.yaml`
- Analyze: `training-lens analyze checkpoint_path`

## Recent Updates

- Consolidated MetricsCollector implementations (removed V2)
- Fixed all test failures (100% pass rate)
- Added utility modules for I/O and model operations
- Updated unsloth compatibility layer
- Fixed linting issues throughout the codebase

## License

This project is licensed under the MIT License.