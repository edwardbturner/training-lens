# Changelog

All notable changes to Training Lens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Training Lens
- Comprehensive training monitoring and analysis framework
- Real-time gradient cosine similarity tracking
- Weight evolution analysis and visualization
- Training diagnostics and health assessment
- W&B integration for experiment tracking
- HuggingFace Hub integration with automatic checkpoint upload
- CLI interface for training and analysis workflows
- Professional reporting with executive summaries and technical reports
- Data export functionality for external analysis
- Modular architecture with specialized analyzers
- Extensive examples and documentation

### Core Features
- **TrainingWrapper**: Main training orchestrator with monitoring
- **CheckpointManager**: Automatic checkpoint saving and management
- **MetricsCollector**: Real-time metrics collection and analysis
- **CheckpointAnalyzer**: Post-training checkpoint analysis
- **GradientAnalyzer**: Specialized gradient evolution analysis
- **WeightAnalyzer**: Weight distribution and stability analysis
- **StandardReports**: Professional report generation
- **CLI Tools**: Command-line interface for all major operations

### Integrations
- **Weights & Biases**: Experiment tracking and metrics logging
- **HuggingFace Hub**: Model and checkpoint storage in `training_lens_checkpoints/` folder
- **unsloth**: LoRA fine-tuning optimization
- **Multiple Export Formats**: JSON, CSV, NumPy, Parquet

### Analysis Capabilities
- Gradient direction consistency via cosine similarity
- Weight evolution tracking and stability assessment
- Training dynamics analysis and convergence detection
- Overfitting detection and early stopping recommendations
- Anomaly detection in gradients and weights
- Layer-wise analysis for deep insights
- Performance efficiency metrics

### Documentation
- Comprehensive README with installation and usage guides
- Detailed technical specifications
- Example scripts for basic, advanced, and analysis workflows
- CLI reference and configuration guides
- Contributing guidelines and development setup

## [0.1.0] - 2024-01-01

### Added
- Initial project structure and foundation
- Core configuration management
- Basic training wrapper framework
- Essential utility functions

---

**Note**: This project is currently in active development. The API may change between versions until we reach v1.0.0. We recommend pinning to specific versions in production environments.