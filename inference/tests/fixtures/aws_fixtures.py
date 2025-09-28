"""
AWS-specific fixtures for integration and end-to-end testing.

This module provides fixtures and utilities for testing with actual
AWS services or mocked AWS environments.
"""

import os
import boto3
import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch
from moto import mock_s3, mock_bedrock_runtime
import json

from inference.tests.fixtures.mock_responses import MOCK_AWS_RESPONSES


# AWS Configuration fixtures
@pytest.fixture(scope="session")
def aws_credentials():
    """
    Mock AWS credentials for testing.
    
    Sets up environment variables for AWS testing without
    requiring actual AWS credentials.
    """
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def aws_test_config():
    """
    AWS test configuration with environment-specific settings.
    
    Returns:
        Dictionary containing AWS test configuration
    """
    # Check if we should use real AWS services
    use_real_aws = os.getenv("USE_REAL_AWS", "false").lower() == "true"
    
    config = {
        "use_real_aws": use_real_aws,
        "profile": os.getenv("AWS_PROFILE", "ml-sandbox"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "bedrock_model_id": "us.amazon.nova-pro-v1:0",
        "custom_deployment_name": os.getenv("CUSTOM_DEPLOYMENT_NAME"),
        "s3_bucket": "test-inference-bucket",
        "timeout_seconds": 30
    }
    
    return config


@pytest.fixture(scope="function")
def aws_session(aws_test_config):
    """
    Create AWS session for testing.
    
    Args:
        aws_test_config: AWS test configuration
        
    Returns:
        boto3 Session instance or None if using mocks
    """
    if aws_test_config["use_real_aws"]:
        return boto3.Session(
            profile_name=aws_test_config["profile"],
            region_name=aws_test_config["region"]
        )
    else:
        # Return None when using mocks
        return None


# Bedrock fixtures
@pytest.fixture(scope="function")
def mock_bedrock_client():
    """
    Create a mock Bedrock client for testing.
    
    Returns:
        Mock Bedrock client with predefined responses
    """
    mock_client = Mock()
    
    # Mock invoke_model method
    def mock_invoke_model(modelId, body, **kwargs):
        # Parse the request body to determine response
        try:
            request_data = json.loads(body)
            prompt = request_data.get("prompt", "")
            
            # Generate response based on prompt content
            if "samsung" in prompt.lower():
                response_text = "Samsung"
            elif "apple" in prompt.lower() or "iphone" in prompt.lower():
                response_text = "Apple"
            elif "nike" in prompt.lower():
                response_text = "Nike"
            else:
                response_text = "Unknown"
            
            response_body = json.dumps({
                "completion": f"The brand is {response_text}",
                "stop_reason": "end_turn"
            })
            
        except Exception:
            response_body = json.dumps(MOCK_AWS_RESPONSES["bedrock_success"]["body"])
        
        # Mock response structure
        mock_response = Mock()
        mock_response.read.return_value = response_body.encode()
        
        return {
            "body": mock_response,
            "contentType": "application/json",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "test-request-id"
            }
        }
    
    mock_client.invoke_model = Mock(side_effect=mock_invoke_model)
    
    # Mock invoke_model_with_response_stream method
    def mock_invoke_stream(modelId, body, **kwargs):
        chunks = [
            {"chunk": {"bytes": b'{"completion": "The "}'}},
            {"chunk": {"bytes": b'{"completion": "brand "}'}},
            {"chunk": {"bytes": b'{"completion": "is Samsung"}'}},
        ]
        return iter(chunks)
    
    mock_client.invoke_model_with_response_stream = Mock(side_effect=mock_invoke_stream)
    
    return mock_client


@pytest.fixture(scope="function")
def real_bedrock_client(aws_session, aws_test_config):
    """
    Create a real Bedrock client for integration testing.
    
    Args:
        aws_session: AWS session
        aws_test_config: AWS test configuration
        
    Returns:
        Real Bedrock client or None if using mocks
    """
    if not aws_test_config["use_real_aws"] or aws_session is None:
        return None
    
    try:
        return aws_session.client("bedrock-runtime", region_name=aws_test_config["region"])
    except Exception as e:
        pytest.skip(f"Could not create Bedrock client: {e}")


@pytest.fixture(scope="function")
def bedrock_client(aws_test_config, mock_bedrock_client, real_bedrock_client):
    """
    Provide appropriate Bedrock client based on test configuration.
    
    Args:
        aws_test_config: AWS test configuration
        mock_bedrock_client: Mock Bedrock client
        real_bedrock_client: Real Bedrock client
        
    Returns:
        Bedrock client (mock or real)
    """
    if aws_test_config["use_real_aws"]:
        return real_bedrock_client
    else:
        return mock_bedrock_client


# S3 fixtures
@pytest.fixture(scope="function")
def mock_s3_client():
    """
    Create a mock S3 client for testing.
    
    Returns:
        Mock S3 client with moto
    """
    with mock_s3():
        client = boto3.client("s3", region_name="us-east-1")
        
        # Create test bucket
        client.create_bucket(Bucket="test-inference-bucket")
        
        # Add some test objects
        test_objects = {
            "test-data.json": json.dumps({"test": "data"}),
            "model-config.json": json.dumps({"model": "config"}),
            "embeddings/test-embeddings.json": json.dumps({"embeddings": [0.1, 0.2, 0.3]})
        }
        
        for key, content in test_objects.items():
            client.put_object(
                Bucket="test-inference-bucket",
                Key=key,
                Body=content.encode(),
                ContentType="application/json"
            )
        
        yield client


@pytest.fixture(scope="function")
def real_s3_client(aws_session, aws_test_config):
    """
    Create a real S3 client for integration testing.
    
    Args:
        aws_session: AWS session
        aws_test_config: AWS test configuration
        
    Returns:
        Real S3 client or None if using mocks
    """
    if not aws_test_config["use_real_aws"] or aws_session is None:
        return None
    
    try:
        return aws_session.client("s3", region_name=aws_test_config["region"])
    except Exception as e:
        pytest.skip(f"Could not create S3 client: {e}")


@pytest.fixture(scope="function")
def s3_client(aws_test_config, mock_s3_client, real_s3_client):
    """
    Provide appropriate S3 client based on test configuration.
    
    Args:
        aws_test_config: AWS test configuration
        mock_s3_client: Mock S3 client
        real_s3_client: Real S3 client
        
    Returns:
        S3 client (mock or real)
    """
    if aws_test_config["use_real_aws"]:
        return real_s3_client
    else:
        return mock_s3_client


# AWS service patches for unit testing
@pytest.fixture(scope="function")
def patch_aws_services():
    """
    Patch AWS services for unit testing.
    
    Returns:
        Context manager that patches AWS service creation
    """
    with patch("boto3.client") as mock_boto_client:
        # Configure mock to return appropriate clients
        def mock_client_factory(service_name, **kwargs):
            if service_name == "bedrock-runtime":
                return mock_bedrock_client()
            elif service_name == "s3":
                return mock_s3_client()
            else:
                return Mock()
        
        mock_boto_client.side_effect = mock_client_factory
        yield mock_boto_client


# Custom deployment fixtures for fine-tuned models
@pytest.fixture(scope="function")
def custom_deployment_config():
    """
    Configuration for testing custom model deployments.
    
    Returns:
        Dictionary with custom deployment configuration
    """
    return {
        "deployment_name": os.getenv("CUSTOM_DEPLOYMENT_NAME", "test-deployment"),
        "model_arn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model",
        "endpoint_name": "test-endpoint",
        "region": "us-east-1"
    }


@pytest.fixture(scope="function")
def mock_custom_deployment(custom_deployment_config):
    """
    Mock custom deployment for testing fine-tuned models.
    
    Args:
        custom_deployment_config: Custom deployment configuration
        
    Returns:
        Mock deployment client
    """
    mock_client = Mock()
    
    def mock_invoke_custom_model(deploymentName, body, **kwargs):
        # Simulate custom model response
        response_body = json.dumps({
            "completion": "Custom model prediction: Samsung",
            "confidence": 0.95,
            "model_version": "v1.0"
        })
        
        mock_response = Mock()
        mock_response.read.return_value = response_body.encode()
        
        return {
            "body": mock_response,
            "contentType": "application/json"
        }
    
    mock_client.invoke_model = Mock(side_effect=mock_invoke_custom_model)
    return mock_client


# Environment validation fixtures
@pytest.fixture(scope="function")
def validate_aws_environment(aws_test_config):
    """
    Validate AWS environment for testing.
    
    Args:
        aws_test_config: AWS test configuration
        
    Raises:
        pytest.skip: If AWS environment is not properly configured
    """
    if aws_test_config["use_real_aws"]:
        # Validate AWS credentials
        try:
            session = boto3.Session(profile_name=aws_test_config["profile"])
            sts = session.client("sts")
            sts.get_caller_identity()
        except Exception as e:
            pytest.skip(f"AWS credentials not available: {e}")
        
        # Validate required environment variables
        required_vars = ["AWS_PROFILE", "AWS_REGION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")


# Performance testing fixtures
@pytest.fixture(scope="function")
def aws_performance_config():
    """
    Configuration for AWS performance testing.
    
    Returns:
        Dictionary with performance test configuration
    """
    return {
        "max_concurrent_requests": 10,
        "request_timeout": 30,
        "max_response_time": 5.0,
        "min_throughput": 1.0,  # requests per second
        "test_duration": 60  # seconds
    }


# Error simulation fixtures
@pytest.fixture(scope="function")
def aws_error_simulator():
    """
    Simulator for AWS service errors.
    
    Returns:
        Dictionary of error simulation functions
    """
    from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
    
    def simulate_client_error(error_code="ValidationException", message="Simulated error"):
        raise ClientError(
            error_response={
                "Error": {
                    "Code": error_code,
                    "Message": message
                }
            },
            operation_name="TestOperation"
        )
    
    def simulate_credentials_error():
        raise NoCredentialsError()
    
    def simulate_connection_error():
        raise EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")
    
    return {
        "client_error": simulate_client_error,
        "credentials_error": simulate_credentials_error,
        "connection_error": simulate_connection_error
    }


# Integration test helpers
class AWSTestHelper:
    """
    Helper class for AWS integration testing.
    
    Provides utilities for testing AWS service interactions
    and validating responses.
    """
    
    def __init__(self, aws_config: Dict[str, Any]):
        """
        Initialize AWS test helper.
        
        Args:
            aws_config: AWS configuration dictionary
        """
        self.config = aws_config
        self.use_real_aws = aws_config.get("use_real_aws", False)
    
    async def test_bedrock_inference(self, client, model_id: str, prompt: str) -> Dict[str, Any]:
        """
        Test Bedrock inference with error handling.
        
        Args:
            client: Bedrock client
            model_id: Model ID to test
            prompt: Test prompt
            
        Returns:
            Dictionary with test results
        """
        try:
            body = json.dumps({
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.1
            })
            
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json"
            )
            
            response_body = json.loads(response["body"].read())
            
            return {
                "success": True,
                "response": response_body,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }
    
    def validate_aws_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate AWS service response structure.
        
        Args:
            response: AWS service response
            
        Returns:
            True if response is valid
        """
        if not isinstance(response, dict):
            return False
        
        # Check for common AWS response fields
        required_fields = ["ResponseMetadata"]
        return any(field in response for field in required_fields)
    
    def get_test_model_id(self) -> str:
        """
        Get appropriate model ID for testing.
        
        Returns:
            Model ID string
        """
        if self.use_real_aws:
            return self.config.get("bedrock_model_id", "us.amazon.nova-pro-v1:0")
        else:
            return "test-model-id"


@pytest.fixture(scope="function")
def aws_test_helper(aws_test_config):
    """
    Create AWS test helper instance.
    
    Args:
        aws_test_config: AWS test configuration
        
    Returns:
        AWSTestHelper instance
    """
    return AWSTestHelper(aws_test_config)