# Contributing to Training Lens

Thank you for your interest in contributing to Training Lens! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming environment for all contributors. By participating, you are expected to uphold this code.

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/training-lens.git
   cd training-lens
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up git hooks**:
   ```bash
   git config core.hooksPath .githooks
   ```

### Project Structure

```
training-lens/
├── training_lens/          # Main package
│   ├── training/           # Core training components
│   ├── analysis/           # Analysis and reporting
│   ├── integrations/       # External service integrations
│   ├── cli/               # Command-line interface
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── examples/              # Usage examples
├── specs/                 # Technical specifications
└── .github/              # GitHub workflows and templates
```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve documentation and examples
5. **Testing**: Add or improve test coverage

### Areas for Contribution

- **New Analysis Methods**: Additional metrics and insights
- **Visualization Improvements**: Enhanced charts and dashboards
- **Integration Support**: Additional external service integrations
- **Performance Optimizations**: Faster analysis and reduced memory usage
- **Documentation**: Examples, tutorials, and API documentation

### Before You Start

1. **Check existing issues** to see if your idea is already being discussed
2. **Create an issue** to discuss major changes before implementing them
3. **Look for "good first issue"** labels for beginner-friendly tasks

## Style Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run these tools before submitting:

```bash
# Format code
black training_lens/ tests/
isort training_lens/ tests/

# Check linting
flake8 training_lens/ tests/

# Type checking
mypy training_lens/
```

### Coding Standards

1. **Follow PEP 8** style guidelines
2. **Use type hints** for function signatures
3. **Write docstrings** for all public functions and classes
4. **Keep functions focused** and single-purpose
5. **Use descriptive variable names**

### Documentation Style

- Use **Google-style docstrings**
- Include **type information** in docstrings
- Provide **examples** in docstrings when helpful
- Keep **line length** under 88 characters

Example:
```python
def analyze_gradients(
    checkpoint_data: Dict[str, Any],
    window_size: int = 10,
) -> Dict[str, float]:
    """Analyze gradient evolution from checkpoint data.

    Args:
        checkpoint_data: Dictionary containing gradient information
        window_size: Size of analysis window for smoothing

    Returns:
        Dictionary with gradient analysis results

    Examples:
        >>> data = load_checkpoint_data("path/to/checkpoint")
        >>> results = analyze_gradients(data, window_size=5)
        >>> print(results["consistency_score"])
        0.85
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=training_lens --cov-report=html

# Run specific test file
pytest tests/test_basic.py

# Run tests with specific markers
pytest -m "not slow"
```

### Writing Tests

1. **Create test files** in the `tests/` directory
2. **Use descriptive test names** that explain what is being tested
3. **Follow the AAA pattern**: Arrange, Act, Assert
4. **Mock external dependencies** when appropriate
5. **Test edge cases** and error conditions

Example test:
```python
def test_training_config_validation():
    """Test that training configuration validates correctly."""
    # Arrange
    invalid_config = {"training_method": "invalid_method"}

    # Act & Assert
    with pytest.raises(ValueError, match="training_method must be one of"):
        TrainingConfig(**invalid_config)
```

### Test Categories

Use pytest markers to categorize tests:

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take longer to run

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Add or update tests** for your changes

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest
   black training_lens/ tests/
   flake8 training_lens/ tests/
   ```

5. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add gradient consistency analysis

   - Implement cosine similarity tracking
   - Add visualization for gradient evolution
   - Include unit tests for new functionality"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** on GitHub

### Pull Request Guidelines

- **Write a clear title** and description
- **Reference related issues** using "Fixes #123" or "Relates to #456"
- **Include screenshots** for UI changes
- **Update documentation** as needed
- **Ensure CI passes** before requesting review

### Commit Message Format

Use clear, descriptive commit messages:

```
Add gradient consistency analysis

- Implement cosine similarity tracking for gradient vectors
- Add visualization functions for gradient evolution
- Include comprehensive unit tests
- Update documentation with usage examples

Fixes #123
```

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the examples and guides

### Communication

- Be **respectful** and **constructive** in all interactions
- **Ask questions** if anything is unclear
- **Provide context** when reporting issues
- **Share knowledge** and help other contributors

### Recognition

Contributors are recognized in several ways:

- **Contributors list** in the README
- **Release notes** acknowledgments
- **GitHub contributor metrics**

## Development Tips

### Debugging

- Use **logging** instead of print statements
- **Write tests** to reproduce bugs before fixing them
- **Use the debugger** for complex issues

### Performance

- **Profile code** before optimizing
- **Consider memory usage** for large datasets
- **Use appropriate data structures**

### Documentation

- **Update examples** when adding new features
- **Keep docstrings current** with code changes
- **Add type hints** for better IDE support

Thank you for contributing to Training Lens! Your efforts help make machine learning training more transparent and understandable for everyone.