"""
Pytest configuration and fixtures for the inference system test suite.

This module provides session and function-level fixtures for testing all components
of the multilingual product inference system, following pytest best practices.
"""

import pytest
import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--aws-config",
        action="store",
        default=None,
        help="Path to AWS test configuration JSON file"
    )


@pytest.fixture(scope="session")
def aws_config(request) -> Dict[str, Any]:
    """
    Load AWS configuration from command line argument or default location.
    
    Args:
        request: Pytest request object
        
    Returns:
        Dictionary containing AWS test configuration
    """
    config_path = request.config.getoption("--aws-config")
    
    if config_path is None:
        # Default to aws_test_config.json in tests directory
        config_path = Path(__file__).parent / "aws_test_config.json"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration if file doesn't exist
        return {
            "aws_profile": "ml-sandbox",
            "aws_region": "us-east-1",
            "custom_deployments": {},
            "timeout_seconds": 60
        }


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the test session.
    
    This ensures all async tests run in the same event loop.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(aws_config):
    """
    Setup test environment for the entire test session.
    
    Args:
        aws_config: AWS configuration dictionary
    """
    # Set environment variables for testing
    os.environ["INFERENCE_TEST_MODE"] = "true"
    os.environ["INFERENCE_LOG_LEVEL"] = "DEBUG"
    
    # Set AWS environment variables from config
    if "aws_profile" in aws_config:
        os.environ["AWS_PROFILE"] = aws_config["aws_profile"]
    if "aws_region" in aws_config:
        os.environ["AWS_REGION"] = aws_config["aws_region"]
        os.environ["AWS_DEFAULT_REGION"] = aws_config["aws_region"]
    
    yield
    
    # Clean up environment variables
    os.environ.pop("INFERENCE_TEST_MODE", None)
    os.environ.pop("INFERENCE_LOG_LEVEL", None)


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interaction")
    config.addinivalue_line("markers", "e2e: End-to-end tests with full system")
    config.addinivalue_line("markers", "aws: Tests requiring AWS services")
    config.addinivalue_line("markers", "slow: Tests that take longer to execute")
    config.addinivalue_line("markers", "milvus: Tests requiring Milvus database")
    config.addinivalue_line("markers", "mock: Tests using mocked dependencies")
    config.addinivalue_line("markers", "performance: Performance and load testing")
    config.addinivalue_line("markers", "multilingual: Tests with multilingual content")
    config.addinivalue_line("markers", "edge_case: Tests with edge cases and error conditions")