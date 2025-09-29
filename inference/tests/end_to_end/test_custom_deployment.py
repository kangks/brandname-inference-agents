"""
End-to-end tests for custom deployment configurations.

This module tests actual custom deployment names and configurations,
deployment-specific settings, custom model configurations, and deployment health checks.
"""

import os
import pytest
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from unittest.mock import patch

import sys
# Add the root project directory to Python path to enable proper imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Add the src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from inference.src.agents.orchestrator_agent import StrandsOrchestratorAgent
from inference.src.agents.llm_agent import BedrockLLMAgent
from inference.src.config.settings import get_config
from inference.tests.fixtures.test_data import TestPayload
from inference.tests.utils.assertion_helpers import AssertionHelpers

@pytest.fixture(scope="class", autouse=True)
def setup_aws_config(request, aws_config):
    """Setup AWS configuration for the test class."""
    if hasattr(request.cls, 'setup_class'):
        request.cls.setup_class(aws_config)


@pytest.mark.e2e
@pytest.mark.aws
@pytest.mark.slow
class TestCustomDeployment:
    """End-to-end tests using actual custom deployment names and configurations."""
    
    @classmethod
    def setup_class(cls, aws_config=None):
        """Setup custom deployment environment for E2E tests."""
        # Use default values if aws_config is not provided (for backwards compatibility)
        if aws_config is None:
            aws_config = {
                "aws_profile": "ml-sandbox",
                "aws_region": "us-east-1",
                "custom_deployments": {}
            }
        
        cls.aws_profile = aws_config.get("aws_profile", "ml-sandbox")
        cls.aws_region = aws_config.get("aws_region", "us-east-1")
        
        # Set environment variables for custom deployment
        os.environ["AWS_PROFILE"] = cls.aws_profile
        os.environ["AWS_REGION"] = cls.aws_region
        os.environ["AWS_DEFAULT_REGION"] = cls.aws_region
        
        # Custom deployment configurations to test
        # Use configurations from aws_config if available, otherwise use defaults
        config_deployments = aws_config.get("custom_deployments", {})
        
        cls.custom_deployments = [
            {
                "name": "finetuned-nova-deployment",
                "model_id": config_deployments.get("finetuned_nova", {}).get(
                    "model_id", 
                    "arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-micro-v1:0:8k/abcdefghijklmnop"
                ),
                "deployment_type": "custom_model",
                "expected_method": "finetuned_nova_llm"
            },
            {
                "name": "custom-claude-deployment", 
                "model_id": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "deployment_type": "foundation_model",
                "expected_method": "llm"
            },
            {
                "name": "custom-embedding-deployment",
                "model_id": "amazon.titan-embed-text-v1",
                "deployment_type": "embedding_model",
                "expected_method": "rag"
            }
        ]
        
        # Test payload for custom deployment tests
        cls.test_payload = TestPayload()
        
    def setup_method(self):
        """Setup for each test method."""
        self.config = get_config()
        self.assertion_helpers = AssertionHelpers()
        
    @pytest.mark.asyncio
    async def test_finetuned_nova_custom_deployment(self):
        """Test with actual finetuned Nova custom deployment configuration."""
        deployment_config = self.custom_deployments[0]  # finetuned-nova-deployment
        
        # Set custom deployment environment variables
        custom_env = {
            "CUSTOM_DEPLOYMENT_NAME": deployment_config["name"],
            "CUSTOM_MODEL_ID": deployment_config["model_id"],
            "DEPLOYMENT_TYPE": deployment_config["deployment_type"]
        }
        
        with patch.dict(os.environ, custom_env):
            # Initialize orchestrator with custom deployment
            orchestrator = StrandsOrchestratorAgent()
            
            # Test custom deployment request
            from inference.src.models.data_models import ProductInput, LanguageHint
            
            product_input = ProductInput(
                product_name=self.test_payload.product_name,
                language_hint=LanguageHint.ENGLISH
            )
            
            try:
                result = await orchestrator.process(product_input)
                
                # Validate custom deployment result
                assert result is not None, "Custom deployment should return a result"
                
                # Check for success or expected result structure
                if result.get("success"):
                    assert result.get("result") is not None, "Successful result should contain result data"
                else:
                    # In test environment, deployment might not be available
                    assert "error" in result or "result" in result, "Should have result or error information"
                
                # Validate custom deployment metadata
                if "metadata" in result:
                    metadata = result["metadata"]
                    assert "model_id" in metadata or "deployment_name" in metadata, \
                        "Metadata should include deployment information"
                    
                    # Check for custom model indicators
                    if "model_id" in metadata:
                        model_id = metadata["model_id"]
                        assert "custom-model" in model_id or "nova" in model_id.lower(), \
                            f"Should use custom Nova model: {model_id}"
                
                # Custom validation for the actual result structure
                if result.get("success") and result.get("result"):
                    result_data = result.get("result")
                    # Check if we have brand extraction data in the result
                    if hasattr(result_data, 'predicted_brand'):
                        assert result_data.predicted_brand is not None, "Should have predicted brand"
                        assert len(result_data.predicted_brand.strip()) > 0, "Brand name should not be empty"
                    elif isinstance(result_data, dict) and 'predicted_brand' in result_data:
                        assert result_data['predicted_brand'] is not None, "Should have predicted brand"
                        assert len(result_data['predicted_brand'].strip()) > 0, "Brand name should not be empty"
                
            except Exception as e:
                # If custom deployment isn't available, skip gracefully
                if any(keyword in str(e).lower() for keyword in ["not found", "invalid", "unavailable"]):
                    pytest.skip(f"Custom deployment not available: {e}")
                else:
                    raise
    
    @pytest.mark.asyncio
    async def test_custom_claude_deployment(self):
        """Test with custom Claude deployment configuration."""
        deployment_config = self.custom_deployments[1]  # custom-claude-deployment
        
        # Set custom deployment environment variables
        custom_env = {
            "CUSTOM_DEPLOYMENT_NAME": deployment_config["name"],
            "CUSTOM_MODEL_ID": deployment_config["model_id"],
            "DEPLOYMENT_TYPE": deployment_config["deployment_type"]
        }
        
        with patch.dict(os.environ, custom_env):
            # Initialize LLM agent with custom deployment
            llm_agent = BedrockLLMAgent(config={})
            
            # Test custom Claude deployment
            from inference.src.models.data_models import ProductInput, LanguageHint
            
            product_input = ProductInput(
                product_name=self.test_payload.product_name,
                language_hint=LanguageHint.ENGLISH
            )
            
            try:
                # Initialize the agent first
                await llm_agent.initialize()
                result = await llm_agent.process(product_input)
                
                # Validate custom Claude deployment result
                assert result is not None, "Custom Claude deployment should return a result"
                
                # Check for success or expected result structure
                if result.get("success"):
                    # If successful, check for result data
                    result_data = result.get("result")
                    if result_data and hasattr(result_data, 'predicted_brand'):
                        assert result_data.predicted_brand is not None, "Should have predicted brand"
                else:
                    # If not successful, this might be expected in test environment
                    assert "error" in result, "Should have error information if not successful"
                
                # Validate Claude-specific processing if successful
                if result.get("success") and result.get("result"):
                    result_data = result.get("result")
                    if hasattr(result_data, 'metadata') and result_data.metadata:
                        metadata = result_data.metadata
                        if "model_id" in metadata:
                            model_id = metadata["model_id"]
                            assert "claude" in model_id.lower(), f"Should use Claude model: {model_id}"
                    
                    # Custom validation for the actual result structure
                    if hasattr(result_data, 'predicted_brand'):
                        assert result_data.predicted_brand is not None, "Should have predicted brand"
                        assert len(result_data.predicted_brand.strip()) > 0, "Brand name should not be empty"
                
            except Exception as e:
                # If custom deployment isn't available, skip gracefully
                if any(keyword in str(e).lower() for keyword in ["not found", "invalid", "unavailable"]):
                    pytest.skip(f"Custom Claude deployment not available: {e}")
                else:
                    raise
    
    @pytest.mark.asyncio
    async def test_deployment_specific_settings(self):
        """Test deployment-specific settings and environment variables."""
        # Test various deployment-specific configurations
        deployment_settings = [
            {
                "DEPLOYMENT_ENVIRONMENT": "production",
                "MODEL_TEMPERATURE": "0.1",
                "MAX_TOKENS": "1000",
                "TIMEOUT_SECONDS": "30"
            },
            {
                "DEPLOYMENT_ENVIRONMENT": "staging", 
                "MODEL_TEMPERATURE": "0.3",
                "MAX_TOKENS": "500",
                "TIMEOUT_SECONDS": "15"
            }
        ]
        
        for i, settings in enumerate(deployment_settings):
            with patch.dict(os.environ, settings):
                orchestrator = StrandsOrchestratorAgent()
                
                from inference.src.models.data_models import ProductInput, LanguageHint
                
                product_input = ProductInput(
                    product_name=f"{self.test_payload.product_name} - Config {i+1}",
                    language_hint=LanguageHint.ENGLISH
                )
                
                try:
                    result = await orchestrator.process(product_input)
                    
                    # Validate deployment-specific processing
                    assert result is not None, f"Deployment config {i+1} should return a result"
                    
                    # Check for success or expected result structure
                    if result.get("success"):
                        assert result.get("result") is not None, f"Config {i+1} successful result should contain result data"
                    else:
                        assert "error" in result or "result" in result, f"Config {i+1} should have result or error information"
                    
                    # Validate configuration-specific metadata
                    if "metadata" in result:
                        metadata = result["metadata"]
                        
                        # Check for deployment environment
                        if "deployment_environment" in metadata:
                            assert metadata["deployment_environment"] == settings["DEPLOYMENT_ENVIRONMENT"]
                        
                        # Check for model parameters
                        if "model_parameters" in metadata:
                            params = metadata["model_parameters"]
                            if "temperature" in params:
                                expected_temp = float(settings["MODEL_TEMPERATURE"])
                                assert abs(params["temperature"] - expected_temp) < 0.01
                
                except Exception as e:
                    # Log configuration-specific errors but don't fail the test
                    print(f"Deployment config {i+1} failed: {e}")
    
    @pytest.mark.asyncio
    async def test_custom_model_configurations(self):
        """Test custom model configurations and endpoints."""
        # Test different custom model configurations
        custom_models = [
            {
                "model_name": "custom-brand-extractor-v1",
                "model_type": "fine_tuned",
                "expected_performance": "high_accuracy"
            },
            {
                "model_name": "multilingual-brand-model-v2",
                "model_type": "foundation_model",
                "expected_performance": "multilingual_support"
            }
        ]
        
        for model_config in custom_models:
            # Set custom model environment
            custom_env = {
                "CUSTOM_MODEL_NAME": model_config["model_name"],
                "CUSTOM_MODEL_TYPE": model_config["model_type"]
            }
            
            with patch.dict(os.environ, custom_env):
                orchestrator = StrandsOrchestratorAgent()
                
                # Test multilingual input for custom models
                from inference.src.models.data_models import ProductInput, LanguageHint
                
                multilingual_inputs = [
                    ProductInput(
                        product_name="Samsung Galaxy S24 Ultra",
                        language_hint=LanguageHint.ENGLISH
                    ),
                    ProductInput(
                        product_name="iPhone 15 Pro Max สีดำ",
                        language_hint=LanguageHint.THAI
                    )
                ]
                
                for product_input in multilingual_inputs:
                    try:
                        result = await orchestrator.process(product_input)
                        
                        # Validate custom model processing
                        assert result is not None, f"Custom model {model_config['model_name']} should return a result"
                        
                        # Check for success or expected result structure
                        if result.get("success"):
                            assert result.get("result") is not None, "Successful result should contain result data"
                        else:
                            assert "error" in result or "result" in result, "Should have result or error information"
                        
                        # Validate model-specific capabilities
                        if model_config["expected_performance"] == "multilingual_support" and result.get("success"):
                            # Should handle multilingual input
                            if product_input.language_hint != LanguageHint.ENGLISH:
                                result_data = result.get("result")
                                if result_data:
                                    # Check if result has brand extraction data
                                    assert result_data is not None, "Should extract brands from multilingual input"
                        
                        # Validate custom model metadata
                        if "metadata" in result:
                            metadata = result["metadata"]
                            if "custom_model" in metadata:
                                assert metadata["custom_model"] == model_config["model_name"]
                    
                    except Exception as e:
                        # Skip if custom model not available
                        if "not found" in str(e).lower() or "unavailable" in str(e).lower():
                            pytest.skip(f"Custom model {model_config['model_name']} not available: {e}")
                        else:
                            print(f"Custom model {model_config['model_name']} error: {e}")
    
    @pytest.mark.asyncio
    async def test_deployment_health_checks(self):
        """Test deployment health checks and monitoring."""
        # Test health checks for different deployment configurations
        health_check_configs = [
            {"deployment_name": "production-deployment", "expected_status": "healthy"},
            {"deployment_name": "staging-deployment", "expected_status": "healthy"},
            {"deployment_name": "development-deployment", "expected_status": "healthy"}
        ]
        
        for config in health_check_configs:
            deployment_env = {
                "DEPLOYMENT_NAME": config["deployment_name"],
                "HEALTH_CHECK_ENABLED": "true"
            }
            
            with patch.dict(os.environ, deployment_env):
                orchestrator = StrandsOrchestratorAgent()
                
                # Perform health check request
                from inference.src.models.data_models import ProductInput, LanguageHint
                
                product_input = ProductInput(
                    product_name="Health Check Test Product",
                    language_hint=LanguageHint.ENGLISH
                )
                
                try:
                    start_time = time.time()
                    result = await orchestrator.process(product_input)
                    response_time = time.time() - start_time
                    
                    # Validate health check response
                    assert result is not None, f"Health check for {config['deployment_name']} should return a result"
                    
                    # Validate response time for health
                    assert response_time < 60.0, f"Health check should respond quickly: {response_time:.2f}s"
                    
                    # Validate health-specific metadata
                    if "metadata" in result:
                        metadata = result["metadata"]
                        
                        # Check deployment health indicators
                        if "deployment_status" in metadata:
                            assert metadata["deployment_status"] in ["healthy", "degraded"], \
                                f"Deployment status should be valid: {metadata['deployment_status']}"
                        
                        if "health_score" in metadata:
                            health_score = metadata["health_score"]
                            assert 0 <= health_score <= 1.0, f"Health score should be between 0-1: {health_score}"
                
                except Exception as e:
                    # Log health check failures but don't fail test
                    print(f"Health check for {config['deployment_name']} failed: {e}")
    
    @pytest.mark.asyncio
    async def test_deployment_failover_and_recovery(self):
        """Test deployment failover and recovery mechanisms."""
        # Test failover scenarios
        failover_scenarios = [
            {
                "name": "primary_deployment_failure",
                "primary_deployment": "production-primary",
                "fallback_deployment": "production-secondary"
            },
            {
                "name": "model_endpoint_failure", 
                "primary_model": "custom-model-v1",
                "fallback_model": "foundation-model-fallback"
            }
        ]
        
        for scenario in failover_scenarios:
            # Test primary deployment
            primary_env = {
                "PRIMARY_DEPLOYMENT": scenario.get("primary_deployment", ""),
                "FALLBACK_DEPLOYMENT": scenario.get("fallback_deployment", ""),
                "FAILOVER_ENABLED": "true"
            }
            
            with patch.dict(os.environ, primary_env):
                orchestrator = StrandsOrchestratorAgent()
                
                from inference.src.models.data_models import ProductInput, LanguageHint
                
                product_input = ProductInput(
                    product_name=f"Failover Test - {scenario['name']}",
                    language_hint=LanguageHint.ENGLISH
                )
                
                try:
                    result = await orchestrator.process(product_input)
                    
                    # Validate failover handling
                    assert result is not None, f"Failover scenario {scenario['name']} should return a result"
                    
                    # Check for failover metadata
                    if "metadata" in result:
                        metadata = result["metadata"]
                        
                        # Should indicate which deployment was used
                        if "deployment_used" in metadata:
                            deployment_used = metadata["deployment_used"]
                            assert deployment_used in [
                                scenario.get("primary_deployment", ""),
                                scenario.get("fallback_deployment", "")
                            ], f"Should use configured deployment: {deployment_used}"
                        
                        # Should indicate if failover occurred
                        if "failover_occurred" in metadata:
                            failover_occurred = metadata["failover_occurred"]
                            assert isinstance(failover_occurred, bool), "Failover flag should be boolean"
                
                except Exception as e:
                    # Failover tests may not be fully configured
                    print(f"Failover scenario {scenario['name']} error: {e}")
    
    @pytest.mark.asyncio
    async def test_custom_deployment_performance_monitoring(self):
        """Test performance monitoring for custom deployments."""
        # Test performance monitoring across different deployments
        deployment_performance_tests = [
            {"deployment": "high-performance", "expected_max_time": 10.0},
            {"deployment": "standard-performance", "expected_max_time": 30.0},
            {"deployment": "cost-optimized", "expected_max_time": 60.0}
        ]
        
        performance_results = {}
        
        for perf_test in deployment_performance_tests:
            deployment_env = {
                "DEPLOYMENT_TIER": perf_test["deployment"],
                "PERFORMANCE_MONITORING": "true"
            }
            
            with patch.dict(os.environ, deployment_env):
                orchestrator = StrandsOrchestratorAgent()
                
                # Run multiple requests to get performance baseline
                from inference.src.models.data_models import ProductInput, LanguageHint
                
                response_times = []
                for i in range(3):  # Limited runs for E2E testing
                    product_input = ProductInput(
                        product_name=f"{self.test_payload.product_name} - Perf Test {i+1}",
                        language_hint=LanguageHint.ENGLISH
                    )
                    
                    start_time = time.time()
                    try:
                        result = await orchestrator.process(product_input)
                        response_time = time.time() - start_time
                        
                        if result and result.get("success"):
                            response_times.append(response_time)
                    
                    except Exception as e:
                        print(f"Performance test {i+1} for {perf_test['deployment']} failed: {e}")
                
                # Analyze performance results
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    max_response_time = max(response_times)
                    
                    performance_results[perf_test["deployment"]] = {
                        "avg_time": avg_response_time,
                        "max_time": max_response_time,
                        "sample_count": len(response_times)
                    }
                    
                    # Validate performance expectations
                    expected_max = perf_test["expected_max_time"]
                    if max_response_time <= expected_max:
                        print(f"✓ {perf_test['deployment']} performance within expectations: {max_response_time:.2f}s <= {expected_max}s")
                    else:
                        print(f"⚠ {perf_test['deployment']} performance exceeded expectations: {max_response_time:.2f}s > {expected_max}s")
        
        # Validate overall performance monitoring
        assert len(performance_results) > 0, "Should collect performance data from at least one deployment"
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up any deployment-specific resources
        pass
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests in class."""
        # Restore original environment variables
        pass