# Contributing to AVA6

Thank you for your interest in contributing to AVA6! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](https://github.com/djflorence/AVA6/blob/main/CODE_OF_CONDUCT.md) to ensure a positive and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Set up the development environment** following the [Installation Guide](../user-guide/installation.md)
4. **Create a new branch** for your feature or bug fix

## Development Workflow

### Setting Up Your Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Making Changes

1. Make your changes in your feature branch
2. Add tests for your changes
3. Run the tests to ensure they pass
4. Update documentation if necessary

### Testing

Run the test suite to ensure your changes don't break existing functionality:

```bash
pytest
```

For more comprehensive testing:

```bash
pytest --cov=src tests/
```

### Code Style

We use Black for code formatting and Ruff for linting. You can format your code with:

```bash
black src/ tests/
ruff check --fix src/ tests/
```

## Pull Request Process

1. **Update your fork** with the latest changes from the main repository
2. **Push your changes** to your fork
3. **Create a pull request** from your branch to the main repository
4. **Describe your changes** in the pull request description
5. **Address any feedback** from reviewers

### Pull Request Template

When creating a pull request, please include:

- A clear description of the changes
- Any related issues (e.g., "Fixes #123")
- Screenshots or examples if applicable
- Confirmation that tests pass

## Documentation

If your changes affect user-facing functionality, please update the documentation accordingly. Documentation is written in Markdown and built with MkDocs.

To preview documentation changes locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then visit `http://localhost:8000` in your browser.

## Release Process

Releases are managed by the core team. If you believe a release is needed, please open an issue to discuss it.

## Getting Help

If you need help with the contribution process or have questions, feel free to:

- Open an issue on GitHub
- Reach out to the maintainers
- Join our community discussions

Thank you for contributing to AVA6! 