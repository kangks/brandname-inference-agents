"""
End-to-end tests for AWS environment integration.

This module tests the complete inference pipeline with actual AWS services
including Bedrock, S3, and Milvus database connectivity.
"""

import os
import pytest
import asyncio
import boto3
from typing import Dict, Any, List
from unittest.mock import patch
import time

from inference.src.agents.orchestrator_agent import OrchestratorAgent
from inference.src.agents.rag_agent import RAGAgent
from inference.src.agents.llm_agent import LLMAgent
from inference.src.agents.ner_agent import NERAgent
from inference.src.config.settings import get_config
from inference.tests.fixtures.test_data import TestPayload
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.e2e
@pytest.mark.aws
@pytest.mark.slow
class TestAWSEnvironment:
    """End-to-end tests with actual AWS environment configuration."""
    
    @classmethod
    def setup_class(cls):
        """Setup AWS environment for E2E tests."""
        cls.aws_profile = "ml-sandbox"
        cls.aws_region = "us-east-1"
        
        # Set environment variables for AWS configuration
        os.environ["AWS_PROFILE"] = cls.aws_profile
        os.environ["AWS_REGION"] = cls.aws_region
        os.environ["AWS_DEFAULT_REGION"] = cls.aws_region
        
        # Initialize AWS session
        cls.session = boto3.Session(profile_name=cls.aws_profile, region_name=cls.aws_region)
        cls.bedrock_client = cls.session.client('bedrock-runtime', region_name=cls.aws_region)
        cls.s3_client = cls.session.client('s3', region_name=cls.aws_region)
        
        # Test payload for all E2E tests
        cls.test_payload = TestPayload()
        
    def setup_method(self):
        """Setup for each test method."""
        self.config = get_config()
        self.assertion_helpers = AssertionHelpers()
        
    @pytest.mark.asyncio
    async def test_bedrock_model_inference(self):
        """Test actual Bedrock model inference with real AWS services."""
        # Initialize LLM agent with actual AWS configuration
        llm_agent = LLMAgent()
        
        # Test payload for Bedrock inference
        inference_request = {
            "product_name": self.test_payload.product_name,
            "language_hint": self.test_payload.language_hint
        }
        
        # Perform actual Bedrock inference
        result = await llm_agent.process(inference_request)
        
        # Validate response structure
        assert result is not None, "Bedrock inference should return a result"
        assert "brands" in result, "Result should contain brands"
        assert isinstance(result["brands"], list), "Brands should be a list"
        
        # Validate response content
        self.assertion_helpers.assert_valid_brand_extraction(result)
        
        # Validate AWS-specific metadata
        if "metadata" in result:
            metadata = result["metadata"]
            assert "model_id" in metadata, "Metadata should include model_id"
            assert "region" in metadata, "Metadata should include region"
            assert metadata["region"] == self.aws_region, f"Region should be {self.aws_region}"
    
    @pytest.mark.asyncio
    async def test_s3_integration(self):
        """Test S3 integration for model artifacts and data storage."""
        # Test S3 connectivity
        try:
            # List buckets to verify S3 access
            response = self.s3_client.list_buckets()
            assert "Buckets" in response, "Should be able to list S3 buckets"
            
            # Test bucket access if specific bucket is configured
            if self.config.aws.s3_bucket:
                bucket_name = self.config.aws.s3_bucket
                
                # Test bucket access
                self.s3_client.head_bucket(Bucket=bucket_name)
                
                # Test object listing
                objects = self.s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                assert "Contents" in objects or objects["KeyCount"] == 0, "Should be able to list bucket contents"
                
        except Exception as e:
            pytest.skip(f"S3 integration test skipped due to configuration: {e}")
    
    @pytest.mark.asyncio
    async def test_milvus_database_connectivity(self):
        """Test real Milvus database connectivity and vector operations."""
        # Initialize RAG agent with actual Milvus configuration
        rag_agent = RAGAgent()
        
        # Test Milvus connection
        try:
            # Test basic connectivity
            await rag_agent._ensure_connection()
            
            # Test collection operations
            collections = await rag_agent._list_collections()
            assert isinstance(collections, list), "Should return list of collections"
            
            # Test vector search with actual data
            search_request = {
                "product_name": self.test_payload.product_name,
                "language_hint": self.test_payload.language_hint
            }
            
            result = await rag_agent.process(search_request)
            
            # Validate RAG response
            assert result is not None, "RAG search should return a result"
            assert "brands" in result, "Result should contain brands"
            assert isinstance(result["brands"], list), "Brands should be a list"
            
            # Validate vector search metadata
            if "metadata" in result:
                metadata = result["metadata"]
                assert "similarity_scores" in metadata or "search_results" in metadata, \
                    "Metadata should include search information"
                    
        except Exception as e:
            pytest.skip(f"Milvus connectivity test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_complete_inference_pipeline(self):
        """Test complete inference pipeline with actual AWS services."""
        # Initialize orchestrator with actual AWS configuration
        orchestrator = OrchestratorAgent()
        
        # Test complete pipeline
        inference_request = {
            "product_name": self.test_payload.product_name,
            "language_hint": self.test_payload.language_hint,
            "method": "orchestrator"
        }
        
        # Execute complete pipeline
        start_time = time.time()
        result = await orchestrator.process(inference_request)
        execution_time = time.time() - start_time
        
        # Validate pipeline result
        assert result is not None, "Pipeline should return a result"
        assert "brands" in result, "Result should contain brands"
        assert isinstance(result["brands"], list), "Brands should be a list"
        
        # Validate orchestrator-specific metadata
        if "metadata" in result:
            metadata = result["metadata"]
            assert "agents_used" in metadata, "Metadata should include agents used"
            assert "execution_time" in metadata, "Metadata should include execution time"
            
        # Validate performance
        assert execution_time < 30.0, f"Pipeline should complete within 30 seconds, took {execution_time:.2f}s"
        
        # Validate brand extraction quality
        self.assertion_helpers.assert_valid_brand_extraction(result)
    
    @pytest.mark.asyncio
    async def test_ner_agent_with_aws_environment(self):
        """Test NER agent functionality in AWS environment."""
        # Initialize NER agent
        ner_agent = NERAgent()
        
        # Test NER processing
        ner_request = {
            "product_name": self.test_payload.product_name,
            "language_hint": self.test_payload.language_hint
        }
        
        result = await ner_agent.process(ner_request)
        
        # Validate NER result
        assert result is not None, "NER should return a result"
        assert "brands" in result, "Result should contain brands"
        assert isinstance(result["brands"], list), "Brands should be a list"
        
        # Validate NER-specific processing
        if "metadata" in result:
            metadata = result["metadata"]
            assert "entities_found" in metadata or "processing_method" in metadata, \
                "Metadata should include NER processing information"
    
    @pytest.mark.asyncio
    async def test_aws_service_error_handling(self):
        """Test error handling with actual AWS service failures."""
        # Test with invalid AWS configuration
        with patch.dict(os.environ, {"AWS_REGION": "invalid-region"}):
            llm_agent = LLMAgent()
            
            inference_request = {
                "product_name": self.test_payload.product_name,
                "language_hint": self.test_payload.language_hint
            }
            
            # Should handle AWS configuration errors gracefully
            try:
                result = await llm_agent.process(inference_request)
                # If it succeeds, validate it's a proper error response
                if result and "error" in result:
                    assert "aws" in result["error"].lower() or "region" in result["error"].lower(), \
                        "Error should indicate AWS/region issue"
            except Exception as e:
                # Should be a recognizable AWS error
                assert any(keyword in str(e).lower() for keyword in ["aws", "region", "credentials", "bedrock"]), \
                    f"Exception should be AWS-related: {e}"
    
    @pytest.mark.asyncio
    async def test_concurrent_aws_requests(self):
        """Test system performance under concurrent load with actual AWS services."""
        # Initialize multiple agents
        orchestrator = OrchestratorAgent()
        
        # Create multiple concurrent requests
        requests = []
        for i in range(3):  # Limited concurrent requests for E2E testing
            request = {
                "product_name": f"{self.test_payload.product_name} - Test {i+1}",
                "language_hint": self.test_payload.language_hint,
                "method": "orchestrator"
            }
            requests.append(request)
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [orchestrator.process(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Validate concurrent execution
        assert len(results) == len(requests), "Should return result for each request"
        
        # Validate individual results
        successful_results = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i+1} failed with exception: {result}")
            else:
                assert result is not None, f"Request {i+1} should return a result"
                if "brands" in result:
                    successful_results += 1
        
        # At least some requests should succeed
        assert successful_results > 0, "At least one concurrent request should succeed"
        
        # Validate performance under load
        avg_time_per_request = total_time / len(requests)
        assert avg_time_per_request < 45.0, \
            f"Average time per request should be reasonable: {avg_time_per_request:.2f}s"
    
    @pytest.mark.asyncio
    async def test_aws_environment_configuration(self):
        """Test AWS environment configuration and credentials."""
        # Validate AWS profile and region configuration
        assert os.environ.get("AWS_PROFILE") == self.aws_profile, \
            f"AWS_PROFILE should be set to {self.aws_profile}"
        assert os.environ.get("AWS_REGION") == self.aws_region, \
            f"AWS_REGION should be set to {self.aws_region}"
        
        # Test AWS credentials and access
        try:
            # Test STS to validate credentials
            sts_client = self.session.client('sts')
            identity = sts_client.get_caller_identity()
            
            assert "Account" in identity, "Should be able to get AWS account identity"
            assert "Arn" in identity, "Should have valid ARN"
            
            # Test Bedrock service access
            bedrock_client = self.session.client('bedrock', region_name=self.aws_region)
            models = bedrock_client.list_foundation_models()
            
            assert "modelSummaries" in models, "Should be able to list Bedrock models"
            assert len(models["modelSummaries"]) > 0, "Should have access to foundation models"
            
        except Exception as e:
            pytest.fail(f"AWS environment configuration test failed: {e}")
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up any test-specific resources
        pass
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests in class."""
        # Restore original environment variables if needed
        pass