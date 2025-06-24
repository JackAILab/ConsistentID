# Claude Memory

## Project Overview
This is an AI image generation and manipulation project focused on diffusion models and transformers.

## Key Commands
- Run tests: `poetry run test` or `poetry run tests`
- Run all tests with coverage: `poetry run pytest`
- Run specific test file: `poetry run pytest tests/test_file.py`
- Run tests with markers: `poetry run pytest -m unit`
- Run linting: `ruff` (already included in dependencies)
- Run type checking: (to be determined - no type checker currently set up)

## Testing Infrastructure
- Testing framework: pytest
- Coverage tool: pytest-cov
- Mocking: pytest-mock
- Test directories: tests/unit/, tests/integration/
- Coverage threshold: 80%
- Coverage reports: HTML (htmlcov/), XML (coverage.xml), and terminal

## Package Manager
- **Poetry** is set up as the package manager
- Dependencies migrated from requirements.txt
- Development dependencies are in the [tool.poetry.group.dev] section
- Run `poetry install` to install all dependencies
- Run `poetry install --only dev` to install only dev dependencies

## Test Markers
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow-running tests

## Important Files
- pyproject.toml: Poetry configuration and pytest settings
- tests/conftest.py: Shared pytest fixtures
- tests/test_setup_validation.py: Validation tests for the infrastructure
- .gitignore: Updated with testing and Poetry entries