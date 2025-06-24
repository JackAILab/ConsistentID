"""
Shared pytest fixtures and configuration for all tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Create a temporary file for testing.
    
    Args:
        temp_dir: The temporary directory fixture
        
    Yields:
        Path: Path to the temporary file
    """
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    yield temp_file


@pytest.fixture
def mock_config() -> dict:
    """
    Provide a mock configuration dictionary for testing.
    
    Returns:
        dict: Mock configuration settings
    """
    return {
        "model_name": "test_model",
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "device": "cpu",
        "seed": 42,
        "output_dir": "/tmp/test_output",
        "cache_dir": "/tmp/test_cache",
    }


@pytest.fixture
def mock_env_vars(monkeypatch) -> dict:
    """
    Set up mock environment variables for testing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        
    Returns:
        dict: Dictionary of environment variables that were set
    """
    env_vars = {
        "TEST_API_KEY": "test_key_12345",
        "TEST_SECRET": "test_secret_67890",
        "TEST_ENDPOINT": "http://localhost:8000",
        "TEST_DEBUG": "true",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """
    Create a sample image file path for testing.
    
    Args:
        temp_dir: The temporary directory fixture
        
    Returns:
        Path: Path to the sample image file
    """
    image_path = temp_dir / "sample_image.jpg"
    # Create a minimal valid JPEG file (1x1 pixel)
    image_data = bytes.fromhex(
        "FFD8FFE000104A46494600010100000100010000FFDB00430008060607060508"
        "0707070909080A0C140D0C0B0B0C1912130F141D1A1F1E1D1A1C1C2024301C1C"
        "1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C1C"
        "1C1C1CFFC00011080001000103012200021101031101FFDA000C030100021103"
        "11003F00F7000000000000FFD9"
    )
    image_path.write_bytes(image_data)
    return image_path


@pytest.fixture
def sample_json_data() -> dict:
    """
    Provide sample JSON data for testing.
    
    Returns:
        dict: Sample JSON data structure
    """
    return {
        "id": "test_123",
        "name": "Test Item",
        "attributes": {
            "color": "blue",
            "size": "medium",
            "tags": ["test", "sample", "fixture"],
        },
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "version": "1.0.0",
        },
    }


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """
    Reset environment to a clean state before each test.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    # Clear any test-specific environment variables
    test_env_prefixes = ["TEST_", "MOCK_", "FAKE_"]
    for key in list(os.environ.keys()):
        if any(key.startswith(prefix) for prefix in test_env_prefixes):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_model_weights(temp_dir: Path) -> Path:
    """
    Create a mock model weights file for testing.
    
    Args:
        temp_dir: The temporary directory fixture
        
    Returns:
        Path: Path to the mock model weights file
    """
    weights_path = temp_dir / "model_weights.pth"
    # Create a dummy file to represent model weights
    weights_path.write_bytes(b"Mock model weights content")
    return weights_path


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture and assert on log messages.
    
    Args:
        caplog: pytest caplog fixture
        
    Returns:
        The caplog fixture configured for testing
    """
    caplog.set_level("DEBUG")
    return caplog


# Marker definitions for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )