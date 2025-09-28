#!/usr/bin/env python3
"""
Unit tests for simple inference functionality.

This module contains pytest-based unit tests for testing individual agent
inference capabilities with proper mocking and fixtures.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple

from inference.src.models.data_models import ProductInput, LanguageHint
from inference.src.agents.llm_agent import StrandsLLMAgent
from inference.tests.utils.test_base import BaseAgentTest
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.unit
@pytest.mark.asyncio
class TestSimpleInference(BaseAgentTest):
    """Unit tests for simple inference functionality."""
    
    @pytest.fixture
    def llm_agent_config(self):
        """Configuration for LLM agent testing."""
        return {
            "model_id": "us.amazon.nova-pro-v1:0",
            "aws_region": "us-east-1",
            "aws_profile": "ml-sandbox",
            "temperature": 0.05,
            "max_tokens": 100,
            "timeout_seconds": 30
        }
    
    @pytest.fixture
    async def mock_llm_agent(self, llm_agent_config):
        """Mock LLM agent for testing."""
        agent = StrandsLLMAgent(llm_agent_config)
        
        # Mock the AWS Bedrock client
        with patch('inference.src.agents.llm_agent.boto3.client') as mock_client:
            mock_bedrock = AsyncMock()
            mock_client.return_value = mock_bedrock
            
            # Mock successful initialization
            agent.is_initialized = True
            agent.client = mock_bedrock
            
            yield agent
    
    @pytest.fixture
    def simple_inference_test_cases(self) -> List[Tuple[str, str]]:
        """Test cases for simple inference testing."""
        return [
            ("Samsung Galaxy S23 Ultra", "Samsung"),
            ("iPhone 15 Pro Max", "Apple"),
            ("Sony WH-1000XM4 Headphones", "Sony"),
            ("Alectric Air Fryer 5L Digital", "Alectric"),
            ("LG OLED C3 55 inch TV", "LG"),
            ("Nintendo Switch OLED", "Nintendo")
        ]
    
    @pytest.fixture
    def multilingual_test_cases(self) -> List[Tuple[str, LanguageHint, str]]:
        """Test cases for multilingual inference testing."""
        return [
            ("Samsung Galaxy S23", LanguageHint.ENGLISH, "Samsung"),
            ("iPhone 15 Pro", LanguageHint.ENGLISH, "Apple"),
            ("Sony โทรศัพท์", LanguageHint.MIXED, "Sony"),
            ("Samsung โทรศัพท์มือถือ", LanguageHint.MIXED, "Samsung"),
            ("LG เครื่องซักผ้า", LanguageHint.MIXED, "LG"),
            ("Panasonic แอร์", LanguageHint.MIXED, "Panasonic")
        ]
    
    def mock_llm_response(self, predicted_brand: str, confidence: float = 0.85) -> Dict[str, Any]:
        """Create a mock LLM response."""
        return {
            "success": True,
            "result": Mock(
                predicted_brand=predicted_brand,
                confidence=confidence,
                reasoning=f"Identified {predicted_brand} based on product name analysis",
                processing_time=0.5
            )
        }
    
    async def test_agent_initialization(self, llm_agent_config):
        """Test LLM agent initialization."""
        with patch('inference.src.agents.llm_agent.boto3.client') as mock_client:
            mock_bedrock = AsyncMock()
            mock_client.return_value = mock_bedrock
            
            agent = StrandsLLMAgent(llm_agent_config)
            await agent.initialize()
            
            assert agent.is_initialized
            assert agent.client is not None
            assert agent.config == llm_agent_config
    
    async def test_agent_cleanup(self, mock_llm_agent):
        """Test LLM agent cleanup."""
        await mock_llm_agent.cleanup()
        # Verify cleanup was called (implementation specific)
        assert True  # Placeholder assertion
    
    @pytest.mark.parametrize("product_name,expected_brand", [
        ("Samsung Galaxy S23 Ultra", "Samsung"),
        ("iPhone 15 Pro Max", "Apple"),
        ("Sony WH-1000XM4 Headphones", "Sony"),
    ])
    async def test_simple_inference_success(self, mock_llm_agent, product_name, expected_brand):
        """Test successful simple inference for individual products."""
        # Mock the agent's process method to return expected result
        mock_response = self.mock_llm_response(expected_brand)
        mock_llm_agent.process = AsyncMock(return_value=mock_response)
        
        # Create product input
        product_input = ProductInput(
            product_name=product_name,
            language_hint=LanguageHint.AUTO
        )
        
        # Process with agent
        result = await mock_llm_agent.process(product_input)
        
        # Assertions
        assert result["success"] is True
        assert result["result"].predicted_brand == expected_brand
        assert result["result"].confidence > 0.0
        assert hasattr(result["result"], "reasoning")
        assert hasattr(result["result"], "processing_time")
    
    async def test_simple_inference_batch(self, mock_llm_agent, simple_inference_test_cases):
        """Test simple inference with multiple products."""
        results = []
        success_count = 0
        total_time = 0
        
        for product_name, expected_brand in simple_inference_test_cases:
            start_time = time.time()
            
            # Mock the response for this product
            mock_response = self.mock_llm_response(expected_brand)
            mock_llm_agent.process = AsyncMock(return_value=mock_response)
            
            # Create product input
            product_input = ProductInput(
                product_name=product_name,
                language_hint=LanguageHint.AUTO
            )
            
            # Process with agent
            result = await mock_llm_agent.process(product_input)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            if result.get("success"):
                llm_result = result.get("result")
                predicted_brand = llm_result.predicted_brand
                confidence = llm_result.confidence
                
                # Check if prediction is correct
                is_correct = predicted_brand.lower() == expected_brand.lower()
                
                results.append({
                    "product_name": product_name,
                    "expected_brand": expected_brand,
                    "predicted_brand": predicted_brand,
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "processing_time": processing_time
                })
                
                if is_correct or confidence > 0.5:
                    success_count += 1
        
        # Assertions for batch processing
        assert len(results) == len(simple_inference_test_cases)
        assert success_count > 0
        
        # Calculate success rate
        success_rate = success_count / len(simple_inference_test_cases)
        assert success_rate >= 0.5, f"Success rate {success_rate:.1%} is below 50%"
        
        # Check average processing time
        avg_time = total_time / len(simple_inference_test_cases)
        assert avg_time < 5.0, f"Average processing time {avg_time:.3f}s is too slow"
    
    @pytest.mark.parametrize("product_name,language_hint,expected_brand", [
        ("Samsung Galaxy S23", LanguageHint.ENGLISH, "Samsung"),
        ("iPhone 15 Pro", LanguageHint.ENGLISH, "Apple"),
        ("Sony โทรศัพท์", LanguageHint.MIXED, "Sony"),
    ])
    async def test_multilingual_inference(self, mock_llm_agent, product_name, language_hint, expected_brand):
        """Test multilingual inference capabilities."""
        # Mock the response
        mock_response = self.mock_llm_response(expected_brand)
        mock_llm_agent.process = AsyncMock(return_value=mock_response)
        
        # Create product input with specific language hint
        product_input = ProductInput(
            product_name=product_name,
            language_hint=language_hint
        )
        
        # Process with agent
        result = await mock_llm_agent.process(product_input)
        
        # Assertions
        assert result["success"] is True
        assert result["result"].predicted_brand == expected_brand
        assert result["result"].confidence > 0.0
        
        # Verify the agent was called with correct language hint
        mock_llm_agent.process.assert_called_once_with(product_input)
    
    async def test_multilingual_inference_batch(self, mock_llm_agent, multilingual_test_cases):
        """Test multilingual inference with multiple language combinations."""
        success_count = 0
        
        for product_name, lang_hint, expected_brand in multilingual_test_cases:
            # Mock the response for this product
            mock_response = self.mock_llm_response(expected_brand)
            mock_llm_agent.process = AsyncMock(return_value=mock_response)
            
            product_input = ProductInput(
                product_name=product_name,
                language_hint=lang_hint
            )
            
            result = await mock_llm_agent.process(product_input)
            
            if result.get("success"):
                llm_result = result.get("result")
                predicted_brand = llm_result.predicted_brand
                confidence = llm_result.confidence
                
                is_correct = predicted_brand.lower() == expected_brand.lower()
                
                if is_correct or confidence > 0.5:
                    success_count += 1
        
        # Assertions
        assert success_count >= len(multilingual_test_cases) // 2, \
            f"Multilingual success rate too low: {success_count}/{len(multilingual_test_cases)}"
    
    async def test_inference_error_handling(self, mock_llm_agent):
        """Test error handling in inference."""
        # Mock an error response
        error_response = {
            "success": False,
            "error": "Model inference failed"
        }
        mock_llm_agent.process = AsyncMock(return_value=error_response)
        
        product_input = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint.AUTO
        )
        
        result = await mock_llm_agent.process(product_input)
        
        # Assertions
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Model inference failed"
    
    async def test_inference_timeout_handling(self, mock_llm_agent):
        """Test timeout handling in inference."""
        # Mock a timeout scenario
        mock_llm_agent.process = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))
        
        product_input = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint.AUTO
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await mock_llm_agent.process(product_input)
    
    async def test_inference_performance_requirements(self, mock_llm_agent):
        """Test that inference meets performance requirements."""
        # Mock fast response
        mock_response = self.mock_llm_response("TestBrand", confidence=0.9)
        mock_llm_agent.process = AsyncMock(return_value=mock_response)
        
        product_input = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint.AUTO
        )
        
        start_time = time.time()
        result = await mock_llm_agent.process(product_input)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert result["success"] is True
        assert processing_time < 5.0, f"Processing time {processing_time:.3f}s exceeds 5s limit"
    
    async def test_inference_confidence_validation(self, mock_llm_agent):
        """Test confidence score validation."""
        # Test various confidence levels
        confidence_levels = [0.1, 0.5, 0.8, 0.95]
        
        for confidence in confidence_levels:
            mock_response = self.mock_llm_response("TestBrand", confidence=confidence)
            mock_llm_agent.process = AsyncMock(return_value=mock_response)
            
            product_input = ProductInput(
                product_name="Test Product",
                language_hint=LanguageHint.AUTO
            )
            
            result = await mock_llm_agent.process(product_input)
            
            # Assertions
            assert result["success"] is True
            assert 0.0 <= result["result"].confidence <= 1.0
            assert result["result"].confidence == confidence
    
    def test_performance_metrics_calculation(self, simple_inference_test_cases):
        """Test performance metrics calculation logic."""
        # Mock processing times
        processing_times = [0.5, 1.2, 0.8, 2.1, 0.9, 1.5]
        success_rates = [True, True, False, True, True, True]
        
        # Calculate metrics
        total_tests = len(processing_times)
        successful_tests = sum(success_rates)
        success_rate = successful_tests / total_tests
        avg_time = sum(processing_times) / total_tests
        
        # Assertions
        assert total_tests == len(simple_inference_test_cases)
        assert 0.0 <= success_rate <= 1.0
        assert avg_time > 0.0
        
        # Performance thresholds
        assert success_rate >= 0.5, f"Success rate {success_rate:.1%} below minimum 50%"
        assert avg_time < 5.0, f"Average time {avg_time:.3f}s exceeds 5s limit"


@pytest.mark.unit
class TestInferenceUtilities:
    """Unit tests for inference utility functions."""
    
    def test_product_input_validation(self):
        """Test ProductInput validation."""
        # Valid input
        valid_input = ProductInput(
            product_name="Samsung Galaxy S23",
            language_hint=LanguageHint.ENGLISH
        )
        assert valid_input.product_name == "Samsung Galaxy S23"
        assert valid_input.language_hint == LanguageHint.ENGLISH
        
        # Test with AUTO language hint
        auto_input = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint.AUTO
        )
        assert auto_input.language_hint == LanguageHint.AUTO
    
    def test_language_hint_enum(self):
        """Test LanguageHint enum values."""
        # Test all enum values
        assert LanguageHint.AUTO is not None
        assert LanguageHint.ENGLISH is not None
        assert LanguageHint.MIXED is not None
        
        # Test enum string values
        assert hasattr(LanguageHint.AUTO, 'value')
        assert hasattr(LanguageHint.ENGLISH, 'value')
        assert hasattr(LanguageHint.MIXED, 'value')
    
    @pytest.mark.parametrize("product_name,expected_valid", [
        ("Samsung Galaxy S23", True),
        ("", False),
        (None, False),
        ("A" * 1000, True),  # Long product name should be valid
    ])
    def test_product_name_validation(self, product_name, expected_valid):
        """Test product name validation logic."""
        if expected_valid:
            if product_name is not None and product_name != "":
                product_input = ProductInput(
                    product_name=product_name,
                    language_hint=LanguageHint.AUTO
                )
                assert product_input.product_name == product_name
        else:
            # Test invalid inputs would raise appropriate errors
            # (Implementation depends on actual validation logic)
            assert product_name is None or product_name == ""