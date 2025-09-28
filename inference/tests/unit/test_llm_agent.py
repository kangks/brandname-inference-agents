"""
Unit tests for LLM (Large Language Model) agents.

This module contains comprehensive unit tests for LLM agent implementations,
including StrandsLLMAgent, EnhancedStrandsLLMAgent, and FinetunedNovaLLMAgent,
with mocked AWS Bedrock dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from inference.src.agents.llm_agent import (
    StrandsLLMAgent, 
    EnhancedStrandsLLMAgent, 
    FinetunedNovaLLMAgent
)
from inference.src.models.data_models import (
    ProductInput, 
    LLMResult, 
    LanguageHint
)
from inference.src.agents.base_agent import AgentError, AgentInitializationError
from inference.tests.utils.test_base import BaseAgentTest


class TestLLMAgent(BaseAgentTest):
    """
    Unit tests for LLM agent implementations extending BaseAgentTest.
    
    Tests LLM agent initialization, prompt engineering, model inference, response parsing,
    AWS service error handling, rate limiting, and different model configurations.
    """
    
    @pytest.fixture
    def llm_config(self):
        """Standard LLM agent configuration for testing."""
        return {
            "model_id": "amazon.nova-pro-v1:0",
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.9,
            "confidence_threshold": 0.5,
            "max_text_length": 1000,
            "timeout_seconds": 30
        }
    
    @pytest.fixture
    def mock_strands_agent(self):
        """Mock Strands Agent for testing."""
        mock_agent = Mock()
        
        # Mock successful brand extraction responses
        def mock_call(prompt):
            if "Samsung" in prompt:
                return "Samsung"
            elif "Apple" in prompt or "iPhone" in prompt:
                return "Apple"
            elif "Sony" in prompt:
                return "Sony"
            else:
                return "Unknown"
        
        mock_agent.__call__ = mock_call
        return mock_agent
    
    @pytest.fixture
    def enhanced_llm_config(self):
        """Configuration for enhanced LLM agent."""
        return {
            "model_id": "amazon.nova-pro-v1:0",
            "max_tokens": 1000,
            "temperature": 0.1,
            "use_context_enhancement": True,
            "context_weight": 0.3,
            "enable_reasoning_analysis": True
        }
    
    @pytest.fixture
    def finetuned_config(self):
        """Configuration for fine-tuned Nova LLM agent."""
        return {
            "custom_deployment_name": "test-deployment",
            "aws_region": "us-east-1",
            "aws_profile": "test-profile",
            "temperature": 0.05,
            "confidence_threshold": 0.6
        }
    
    @pytest.fixture
    def mock_bedrock_client(self):
        """Mock AWS Bedrock client for testing."""
        mock_client = Mock()
        
        # Mock get_custom_model_deployment response
        mock_client.get_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/test-deployment"
        }
        
        return mock_client

    # Initialization Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_initialization_success(self, llm_config, mock_strands_agent):
        """Test successful LLM agent initialization."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.strands_agent is not None
            assert agent.model_id == "amazon.nova-pro-v1:0"
            assert agent.temperature == 0.1
            assert agent.timeout_seconds == 30
    
    @pytest.mark.asyncio
    async def test_llm_agent_initialization_strands_failure(self, llm_config):
        """Test LLM agent initialization when Strands Agent creation fails."""
        with patch('strands.Agent', side_effect=Exception("Strands initialization failed")):
            agent = StrandsLLMAgent(llm_config)
            
            with pytest.raises(AgentInitializationError) as exc_info:
                await agent.initialize()
            
            assert "Failed to initialize Strands LLM agent" in str(exc_info.value)
            assert not agent.is_initialized()
    
    @pytest.mark.asyncio
    async def test_llm_agent_validation_failure(self, llm_config):
        """Test LLM agent initialization when validation fails."""
        mock_agent = Mock()
        mock_agent.return_value = ""  # Empty response should cause validation failure
        
        with patch('strands.Agent', return_value=mock_agent):
            agent = StrandsLLMAgent(llm_config)
            
            with pytest.raises(AgentInitializationError) as exc_info:
                await agent.initialize()
            
            assert "Agent returned empty response during validation" in str(exc_info.value)
            assert not agent.is_initialized()

    # Processing Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_process_valid_input(self, llm_config, mock_strands_agent):
        """Test LLM agent processing with valid input."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is True
            assert result["agent_type"] == "llm"
            assert result["error"] is None
            assert isinstance(result["result"], LLMResult)
    
    @pytest.mark.asyncio
    async def test_llm_agent_infer_brand_success(self, llm_config, mock_strands_agent):
        """Test successful brand inference."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            result = await agent.infer_brand("Samsung Galaxy S23")
            
            assert isinstance(result, LLMResult)
            assert result.predicted_brand == "Samsung"
            assert result.confidence > 0.0
            assert result.processing_time > 0.0
            assert result.model_id == "amazon.nova-pro-v1:0"
            assert result.reasoning is not None
    
    @pytest.mark.asyncio
    async def test_llm_agent_infer_brand_with_context(self, llm_config, mock_strands_agent):
        """Test brand inference with additional context."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            context = "This is a smartphone from a Korean manufacturer"
            result = await agent.infer_brand("Galaxy S23", context=context)
            
            assert isinstance(result, LLMResult)
            assert result.predicted_brand != "Unknown"  # Should infer something with context
    
    @pytest.mark.asyncio
    async def test_llm_agent_process_empty_input(self, llm_config, mock_strands_agent):
        """Test LLM agent processing with empty input."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            result = await agent.infer_brand("")
            
            assert isinstance(result, LLMResult)
            # Should handle empty input gracefully
    
    @pytest.mark.asyncio
    async def test_llm_agent_process_long_text_truncation(self, llm_config, mock_strands_agent):
        """Test LLM agent handling of text that exceeds max length."""
        config = llm_config.copy()
        config["max_text_length"] = 50
        
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(config)
            await agent.initialize()
            
            long_text = "Samsung Galaxy S23 " * 20  # Much longer than 50 chars
            result = await agent.infer_brand(long_text)
            
            assert isinstance(result, LLMResult)
            # Should process without error despite truncation
    
    @pytest.mark.asyncio
    async def test_llm_agent_process_not_initialized(self, llm_config):
        """Test LLM agent processing when not initialized."""
        agent = StrandsLLMAgent(llm_config)
        # Don't initialize the agent
        
        with pytest.raises(AgentError) as exc_info:
            await agent.infer_brand("Samsung Galaxy S23")
        
        assert "Agent not initialized" in str(exc_info.value)

    # Prompt Engineering Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_prompt_template_selection(self, llm_config, mock_strands_agent):
        """Test prompt template selection based on text characteristics."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test English text
            english_prompt = agent._build_prompt("Samsung Galaxy S23")
            assert "Samsung Galaxy S23" in english_prompt
            
            # Test mixed Thai-English text
            mixed_prompt = agent._build_prompt("Samsung โทรศัพท์")
            assert "Samsung โทรศัพท์" in mixed_prompt
            # Should use multilingual template for mixed text
    
    @pytest.mark.asyncio
    async def test_llm_agent_language_detection(self, llm_config, mock_strands_agent):
        """Test language composition detection."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test mixed language detection
            lang_info = agent._detect_language_composition("Samsung โทรศัพท์")
            
            assert lang_info["is_mixed"] is True
            assert lang_info["thai_ratio"] > 0.1
            assert lang_info["english_ratio"] > 0.1
            
            # Test primarily English
            lang_info = agent._detect_language_composition("Samsung Galaxy S23")
            assert lang_info["is_primarily_thai"] is False
    
    @pytest.mark.asyncio
    async def test_llm_agent_prompt_with_context(self, llm_config, mock_strands_agent):
        """Test prompt building with additional context."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            context = "This is a smartphone product"
            prompt = agent._build_prompt("Galaxy S23", context=context)
            
            assert "Galaxy S23" in prompt
            assert context in prompt

    # Response Parsing Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_response_parsing_success(self, llm_config, mock_strands_agent):
        """Test successful response parsing."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test parsing clear brand response
            brand, reasoning = agent._extract_brand_from_response("Samsung")
            assert brand == "Samsung"
            assert reasoning == "Samsung"
            
            # Test parsing response with explanation
            brand, reasoning = agent._extract_brand_from_response("The brand is Apple based on the iPhone model")
            assert brand == "Apple"
            assert "iPhone" in reasoning
    
    @pytest.mark.asyncio
    async def test_llm_agent_response_parsing_edge_cases(self, llm_config, mock_strands_agent):
        """Test response parsing with edge cases."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test empty response
            brand, reasoning = agent._extract_brand_from_response("")
            assert brand == "Unknown"
            
            # Test "unknown" response
            brand, reasoning = agent._extract_brand_from_response("unknown")
            assert brand == "Unknown"
            
            # Test response with no clear brand
            brand, reasoning = agent._extract_brand_from_response("I cannot determine the brand")
            assert brand == "Unknown"
    
    @pytest.mark.asyncio
    async def test_llm_agent_brand_parsing_patterns(self, llm_config, mock_strands_agent):
        """Test brand name parsing with various response patterns."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test different response formats
            test_cases = [
                ("Samsung", "Samsung"),
                ("Brand: Samsung", "Samsung"),
                ("The answer is Apple", "Apple"),
                ("Brand name: Sony", "Sony"),
                ("SAMSUNG", "SAMSUNG"),
            ]
            
            for response, expected_brand in test_cases:
                brand = agent._parse_brand_from_text(response)
                assert brand == expected_brand

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_strands_invocation_error(self, llm_config):
        """Test LLM agent handling of Strands invocation errors."""
        mock_agent = Mock()
        mock_agent.side_effect = Exception("Model invocation failed")
        
        with patch('strands.Agent', return_value=mock_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "Agent invocation failed" in result["error"]
            assert result["result"] is None
    
    @pytest.mark.asyncio
    async def test_llm_agent_timeout_handling(self, llm_config):
        """Test LLM agent timeout handling."""
        config = llm_config.copy()
        config["timeout_seconds"] = 0.1  # Very short timeout
        
        # Mock agent that takes too long
        mock_agent = Mock()
        def slow_response(prompt):
            import time
            time.sleep(1)  # Longer than timeout
            return "Samsung"
        
        mock_agent.__call__ = slow_response
        
        with patch('strands.Agent', return_value=mock_agent):
            agent = StrandsLLMAgent(config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "timed out" in result["error"]
    
    @pytest.mark.asyncio
    async def test_llm_agent_aws_service_failure_simulation(self, llm_config):
        """Test LLM agent handling of simulated AWS service failures."""
        mock_agent = Mock()
        mock_agent.side_effect = Exception("ServiceUnavailable: The service is temporarily unavailable")
        
        with patch('strands.Agent', return_value=mock_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "ServiceUnavailable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_llm_agent_rate_limiting_simulation(self, llm_config):
        """Test LLM agent handling of simulated rate limiting."""
        mock_agent = Mock()
        mock_agent.side_effect = Exception("ThrottlingException: Rate exceeded")
        
        with patch('strands.Agent', return_value=mock_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "ThrottlingException" in result["error"]

    # Confidence Scoring Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_confidence_calculation(self, llm_config, mock_strands_agent):
        """Test LLM agent confidence calculation with various scenarios."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test high confidence scenario
            confidence = agent._calculate_confidence(
                "Samsung", 
                "The brand is clearly Samsung based on the Galaxy model name",
                "Samsung Galaxy S23"
            )
            assert confidence > 0.5
            
            # Test low confidence scenario
            confidence = agent._calculate_confidence(
                "Unknown",
                "I'm not sure about the brand",
                "Generic product"
            )
            assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_llm_agent_confidence_factors(self, llm_config, mock_strands_agent):
        """Test individual confidence factors."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Test brand name in original text boost
            confidence = agent._calculate_confidence(
                "Samsung",
                "Brand is Samsung",
                "Samsung Galaxy S23"  # Brand appears in original
            )
            high_confidence = confidence
            
            confidence = agent._calculate_confidence(
                "Samsung",
                "Brand is Samsung", 
                "Unknown product"  # Brand doesn't appear in original
            )
            low_confidence = confidence
            
            assert high_confidence > low_confidence

    # Resource Management Tests
    
    @pytest.mark.asyncio
    async def test_llm_agent_cleanup(self, llm_config, mock_strands_agent):
        """Test LLM agent resource cleanup."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.strands_agent is not None
            
            await agent.cleanup()
            
            assert not agent.is_initialized()
            assert agent.strands_agent is None
    
    @pytest.mark.asyncio
    async def test_llm_agent_health_check_success(self, llm_config, mock_strands_agent):
        """Test successful LLM agent health check."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Health check should pass without raising exceptions
            await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_llm_agent_health_check_failure(self, llm_config):
        """Test LLM agent health check failure scenarios."""
        agent = StrandsLLMAgent(llm_config)
        # Don't initialize the agent
        
        with pytest.raises(RuntimeError):
            await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_llm_agent_concurrent_processing(self, llm_config, mock_strands_agent):
        """Test LLM agent handling concurrent processing requests."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            # Create multiple concurrent processing tasks
            tasks = []
            for i in range(5):
                product_input = ProductInput(
                    product_name=f"Samsung Galaxy S{20 + i}",
                    language_hint=LanguageHint.EN
                )
                task = agent.process(product_input)
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for result in results:
                assert result["success"] is True
                assert result["agent_type"] == "llm"

    # Parameterized Tests for Different Models and Configurations
    
    @pytest.mark.parametrize("model_id,temperature,top_p", [
        ("amazon.nova-pro-v1:0", 0.1, 0.9),
        ("amazon.nova-lite-v1:0", 0.2, 0.8),
        ("anthropic.claude-3-sonnet-20240229-v1:0", 0.0, 1.0),
    ])
    @pytest.mark.asyncio
    async def test_llm_agent_different_models(self, llm_config, mock_strands_agent, 
                                             model_id, temperature, top_p):
        """Test LLM agent with different model configurations."""
        config = llm_config.copy()
        config["model_id"] = model_id
        config["temperature"] = temperature
        config["top_p"] = top_p
        
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(config)
            await agent.initialize()
            
            assert agent.model_id == model_id
            assert agent.temperature == temperature
            assert agent.top_p == top_p
    
    @pytest.mark.parametrize("product_name,expected_brand", [
        ("Samsung Galaxy S23", "Samsung"),
        ("iPhone 15 Pro Max", "Apple"),  # Should be mapped to Apple
        ("Sony WH-1000XM4", "Sony"),
        ("Generic USB Cable", "Unknown"),  # No clear brand
    ])
    @pytest.mark.asyncio
    async def test_llm_agent_brand_extraction_scenarios(self, llm_config, mock_strands_agent,
                                                       product_name, expected_brand):
        """Test LLM agent brand extraction with various product scenarios."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = StrandsLLMAgent(llm_config)
            await agent.initialize()
            
            result = await agent.infer_brand(product_name)
            
            assert isinstance(result, LLMResult)
            if expected_brand != "Unknown":
                assert result.predicted_brand == expected_brand
            else:
                # For unknown cases, just ensure we get a result
                assert result.predicted_brand is not None


class TestEnhancedStrandsLLMAgent(BaseAgentTest):
    """
    Specific unit tests for EnhancedStrandsLLMAgent functionality.
    
    Tests enhanced prompt engineering, context-aware inference, and improved confidence scoring.
    """
    
    @pytest.fixture
    def enhanced_config(self):
        """Configuration for enhanced LLM agent testing."""
        return {
            "model_id": "amazon.nova-pro-v1:0",
            "use_context_enhancement": True,
            "context_weight": 0.3,
            "enable_reasoning_analysis": True
        }
    
    @pytest.mark.asyncio
    async def test_enhanced_llm_agent_initialization(self, enhanced_config, mock_strands_agent):
        """Test enhanced LLM agent initialization with additional features."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = EnhancedStrandsLLMAgent(enhanced_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.use_context_enhancement is True
            assert agent.context_weight == 0.3
            assert agent.enable_reasoning_analysis is True
    
    @pytest.mark.asyncio
    async def test_enhanced_llm_agent_context_enhancement(self, enhanced_config, mock_strands_agent):
        """Test context-aware confidence enhancement."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = EnhancedStrandsLLMAgent(enhanced_config)
            await agent.initialize()
            
            # Test confidence enhancement with context
            enhanced_confidence = agent._enhance_confidence_with_context(
                0.6, "Samsung", "This Samsung product is from Korea"
            )
            
            # Should boost confidence when brand appears in context
            assert enhanced_confidence > 0.6
    
    @pytest.mark.asyncio
    async def test_enhanced_llm_agent_inference_with_context(self, enhanced_config, mock_strands_agent):
        """Test enhanced inference with context awareness."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            agent = EnhancedStrandsLLMAgent(enhanced_config)
            await agent.initialize()
            
            context = "This is a Samsung smartphone from the Galaxy series"
            result = await agent.infer_brand("Galaxy S23", context=context)
            
            assert isinstance(result, LLMResult)
            assert result.predicted_brand == "Samsung"
            # Should have enhanced confidence due to context
            assert result.confidence > 0.5


class TestFinetunedNovaLLMAgent(BaseAgentTest):
    """
    Specific unit tests for FinetunedNovaLLMAgent functionality.
    
    Tests fine-tuned model integration, custom deployment handling, and specialized prompting.
    """
    
    @pytest.fixture
    def finetuned_config(self):
        """Configuration for fine-tuned Nova agent testing."""
        return {
            "custom_deployment_name": "test-deployment",
            "aws_region": "us-east-1",
            "aws_profile": "test-profile",
            "temperature": 0.05,
            "confidence_threshold": 0.6
        }
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_initialization_success(self, finetuned_config, mock_strands_agent, mock_bedrock_client):
        """Test successful fine-tuned Nova agent initialization."""
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_bedrock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.custom_deployment_name == "test-deployment"
            assert agent.temperature == 0.05
            assert agent.confidence_threshold == 0.6
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_deployment_arn_retrieval(self, finetuned_config, mock_strands_agent, mock_bedrock_client):
        """Test custom model deployment ARN retrieval."""
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_bedrock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            
            # Test ARN retrieval
            arn = await agent._get_custom_model_deployment_arn()
            
            assert arn == "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/test-deployment"
            mock_bedrock_client.get_custom_model_deployment.assert_called_once_with(
                customModelDeploymentIdentifier="test-deployment"
            )
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_aws_error_handling(self, finetuned_config, mock_strands_agent):
        """Test fine-tuned Nova agent AWS error handling."""
        from botocore.exceptions import ClientError
        
        mock_client = Mock()
        mock_client.get_custom_model_deployment.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFound", "Message": "Deployment not found"}},
            "GetCustomModelDeployment"
        )
        
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            
            with pytest.raises(AgentInitializationError) as exc_info:
                await agent.initialize()
            
            assert "Failed to get custom model deployment ARN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_specialized_prompting(self, finetuned_config, mock_strands_agent, mock_bedrock_client):
        """Test specialized prompting for fine-tuned model."""
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_bedrock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            await agent.initialize()
            
            # Test simplified prompt for fine-tuned model
            prompt = agent._build_prompt("Samsung Galaxy S23")
            
            # Should use simplified format for fine-tuned model
            assert "Product title:" in prompt
            assert "Brand name:" in prompt
            assert len(prompt) < 200  # Should be more concise than regular prompts
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_confidence_boost(self, finetuned_config, mock_strands_agent, mock_bedrock_client):
        """Test confidence boost for fine-tuned model predictions."""
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_bedrock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            await agent.initialize()
            
            result = await agent.infer_brand("Samsung Galaxy S23")
            
            assert isinstance(result, LLMResult)
            # Fine-tuned model should have higher confidence threshold
            if result.predicted_brand != "Unknown":
                assert result.confidence >= 0.6  # Higher threshold for fine-tuned model
    
    @pytest.mark.asyncio
    async def test_finetuned_nova_agent_system_prompt(self, finetuned_config, mock_strands_agent, mock_bedrock_client):
        """Test specialized system prompt for fine-tuned model."""
        with patch('strands.Agent', return_value=mock_strands_agent), \
             patch('boto3.Session') as mock_session:
            
            mock_session.return_value.client.return_value = mock_bedrock_client
            
            agent = FinetunedNovaLLMAgent(finetuned_config)
            
            system_prompt = agent._get_finetuned_system_prompt()
            
            assert "specialized brand extraction model" in system_prompt
            assert "fine-tuned specifically" in system_prompt
            assert "ONLY the brand name" in system_prompt