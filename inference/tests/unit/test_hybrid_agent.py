"""
Unit tests for Hybrid agents.

This module contains comprehensive unit tests for Hybrid agent implementations,
including SequentialHybridAgent and OptimizedHybridAgent, testing sequential processing,
result aggregation, and confidence scoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from inference.src.agents.hybrid_agent import SequentialHybridAgent, OptimizedHybridAgent
from inference.src.models.data_models import (
    ProductInput, 
    HybridResult, 
    NERResult,
    RAGResult,
    LLMResult,
    EntityResult,
    EntityType,
    SimilarProduct,
    LanguageHint
)
from inference.src.agents.base_agent import AgentError, AgentInitializationError
from inference.tests.utils.test_base import BaseAgentTest


class TestHybridAgent(BaseAgentTest):
    """
    Unit tests for Hybrid agent implementations extending BaseAgentTest.
    
    Tests sequential processing, pipeline execution, result aggregation, confidence scoring,
    error handling, fallback mechanisms, and dynamic agent registration.
    """
    
    @pytest.fixture
    def hybrid_config(self):
        """Standard hybrid agent configuration for testing."""
        return {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "ner_confidence_threshold": 0.8,
            "rag_confidence_threshold": 0.8,
            "ner_weight": 0.3,
            "rag_weight": 0.4,
            "llm_weight": 0.3,
            "use_early_termination": False,
            "use_context_enhancement": True,
            "max_pipeline_time": 60.0
        }
    
    @pytest.fixture
    def mock_ner_agent(self):
        """Mock NER agent for testing."""
        mock_agent = Mock()
        
        # Mock successful NER result
        mock_result = NERResult(
            entities=[
                EntityResult(
                    entity_type=EntityType.BRAND,
                    text="Samsung",
                    confidence=0.9,
                    start_pos=0,
                    end_pos=7
                )
            ],
            confidence=0.9,
            processing_time=0.1,
            model_used="test_ner_model"
        )
        
        mock_agent.extract_entities = AsyncMock(return_value=mock_result)
        return mock_agent
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Mock RAG agent for testing."""
        mock_agent = Mock()
        
        # Mock successful RAG result
        mock_result = RAGResult(
            predicted_brand="Samsung",
            similar_products=[
                SimilarProduct(
                    product_name="Samsung Galaxy S22",
                    brand="Samsung",
                    category="Electronics",
                    sub_category="Smartphones",
                    similarity_score=0.95
                )
            ],
            confidence=0.85,
            processing_time=0.2,
            embedding_model="test_embedding_model"
        )
        
        mock_agent.retrieve_and_infer = AsyncMock(return_value=mock_result)
        return mock_agent
    
    @pytest.fixture
    def mock_llm_agent(self):
        """Mock LLM agent for testing."""
        mock_agent = Mock()
        
        # Mock successful LLM result
        mock_result = LLMResult(
            predicted_brand="Samsung",
            reasoning="The product name clearly indicates Samsung as the brand",
            confidence=0.8,
            processing_time=0.3,
            model_id="test_llm_model"
        )
        
        mock_agent.infer_brand = AsyncMock(return_value=mock_result)
        return mock_agent
    
    @pytest.fixture
    def optimized_config(self):
        """Configuration for optimized hybrid agent."""
        return {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "use_dynamic_thresholds": True,
            "use_stage_selection": True,
            "performance_mode": "balanced",
            "min_confidence_threshold": 0.6,
            "max_confidence_threshold": 0.9
        }

    # Initialization Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_initialization_success(self, hybrid_config):
        """Test successful hybrid agent initialization."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        assert agent.is_initialized()
        assert agent.enable_ner_stage is True
        assert agent.enable_rag_stage is True
        assert agent.enable_llm_stage is True
        assert agent.ner_weight == 0.3
        assert agent.rag_weight == 0.4
        assert agent.llm_weight == 0.3
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_initialization_invalid_config(self):
        """Test hybrid agent initialization with invalid configuration."""
        # Invalid config: no stages enabled
        invalid_config = {
            "enable_ner_stage": False,
            "enable_rag_stage": False,
            "enable_llm_stage": False
        }
        
        agent = SequentialHybridAgent(invalid_config)
        
        with pytest.raises(AgentInitializationError) as exc_info:
            await agent.initialize()
        
        assert "at least one pipeline stage must be enabled" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_initialization_invalid_weights(self):
        """Test hybrid agent initialization with invalid weight configuration."""
        # Invalid config: weights don't sum to 1.0
        invalid_config = {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "ner_weight": 0.5,
            "rag_weight": 0.5,
            "llm_weight": 0.5  # Total = 1.5, should be ~1.0
        }
        
        agent = SequentialHybridAgent(invalid_config)
        
        with pytest.raises(AgentInitializationError) as exc_info:
            await agent.initialize()
        
        assert "weights must sum to 1.0" in str(exc_info.value).lower()

    # Sequential Processing Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_process_valid_input(self, hybrid_config, mock_ner_agent, mock_rag_agent, mock_llm_agent):
        """Test hybrid agent processing with valid input."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        product_input = ProductInput(
            product_name="Samsung Galaxy S23",
            language_hint=LanguageHint.EN
        )
        
        result = await agent.process(product_input)
        
        assert result["success"] is True
        assert result["agent_type"] == "hybrid"
        assert result["error"] is None
        assert isinstance(result["result"], HybridResult)
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_sequential_pipeline_execution(self, hybrid_config, mock_ner_agent, mock_rag_agent, mock_llm_agent):
        """Test sequential pipeline execution through all stages."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent,
            rag_agent=mock_rag_agent,
            llm_agent=mock_llm_agent
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
        assert "NER" in result.pipeline_steps
        assert "RAG" in result.pipeline_steps
        assert "LLM" in result.pipeline_steps
        assert result.confidence > 0.0
        assert result.processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_ner_stage_execution(self, hybrid_config, mock_ner_agent):
        """Test NER stage execution in isolation."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        ner_result, context = await agent._execute_ner_stage(
            "Samsung Galaxy S23", mock_ner_agent, ""
        )
        
        assert isinstance(ner_result, NERResult)
        assert ner_result.confidence == 0.9
        assert len(ner_result.entities) == 1
        assert ner_result.entities[0].text == "Samsung"
        assert "Samsung" in context  # Context should be enhanced
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_rag_stage_execution(self, hybrid_config, mock_rag_agent):
        """Test RAG stage execution in isolation."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        rag_result, context = await agent._execute_rag_stage(
            "Samsung Galaxy S23", mock_rag_agent, ""
        )
        
        assert isinstance(rag_result, RAGResult)
        assert rag_result.predicted_brand == "Samsung"
        assert rag_result.confidence == 0.85
        assert len(rag_result.similar_products) == 1
        assert "Samsung" in context  # Context should be enhanced
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_llm_stage_execution(self, hybrid_config, mock_llm_agent):
        """Test LLM stage execution in isolation."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        context = "Previous stages detected Samsung brand"
        llm_result, updated_context = await agent._execute_llm_stage(
            "Samsung Galaxy S23", mock_llm_agent, context
        )
        
        assert isinstance(llm_result, LLMResult)
        assert llm_result.predicted_brand == "Samsung"
        assert llm_result.confidence == 0.8
        assert llm_result.reasoning is not None
        # LLM stage doesn't modify context in current implementation
        assert updated_context == context

    # Context Enhancement Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_context_enhancement_with_ner(self, hybrid_config):
        """Test context enhancement with NER findings."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Create NER result with multiple entities
        ner_result = NERResult(
            entities=[
                EntityResult(EntityType.BRAND, "Samsung", 0.9, 0, 7),
                EntityResult(EntityType.PRODUCT, "Galaxy", 0.8, 8, 14)
            ],
            confidence=0.85,
            processing_time=0.1,
            model_used="test_model"
        )
        
        context = agent._enhance_context_with_ner("", ner_result)
        
        assert "Samsung" in context
        assert "brand entities" in context.lower()
        assert "Galaxy" in context
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_context_enhancement_with_rag(self, hybrid_config):
        """Test context enhancement with RAG findings."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Create RAG result with similar products
        rag_result = RAGResult(
            predicted_brand="Samsung",
            similar_products=[
                SimilarProduct("Samsung Galaxy S22", "Samsung", "Electronics", "Smartphones", 0.95),
                SimilarProduct("Samsung Note 20", "Samsung", "Electronics", "Smartphones", 0.90)
            ],
            confidence=0.85,
            processing_time=0.2,
            embedding_model="test_model"
        )
        
        context = agent._enhance_context_with_rag("", rag_result)
        
        assert "Samsung" in context
        assert "RAG suggests" in context or "Similar products" in context

    # Result Aggregation Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_result_combination_all_stages(self, hybrid_config, mock_ner_agent, mock_rag_agent, mock_llm_agent):
        """Test result combination when all stages execute successfully."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Mock results from all stages
        ner_result = NERResult(
            entities=[EntityResult(EntityType.BRAND, "Samsung", 0.9, 0, 7)],
            confidence=0.9, processing_time=0.1, model_used="ner_model"
        )
        rag_result = RAGResult(
            predicted_brand="Samsung", similar_products=[], confidence=0.85,
            processing_time=0.2, embedding_model="rag_model"
        )
        llm_result = LLMResult(
            predicted_brand="Samsung", reasoning="Clear brand", confidence=0.8,
            processing_time=0.3, model_id="llm_model"
        )
        
        combined_result = agent._combine_pipeline_results(
            ner_result, rag_result, llm_result, ["NER", "RAG", "LLM"], 0.0
        )
        
        assert isinstance(combined_result, HybridResult)
        assert combined_result.predicted_brand == "Samsung"
        assert combined_result.confidence > 0.0
        assert combined_result.ner_contribution > 0.0
        assert combined_result.rag_contribution > 0.0
        assert combined_result.llm_contribution > 0.0
        assert len(combined_result.pipeline_steps) == 3
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_result_combination_partial_stages(self, hybrid_config):
        """Test result combination when only some stages execute."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Only NER and RAG results, no LLM
        ner_result = NERResult(
            entities=[EntityResult(EntityType.BRAND, "Samsung", 0.9, 0, 7)],
            confidence=0.9, processing_time=0.1, model_used="ner_model"
        )
        rag_result = RAGResult(
            predicted_brand="Samsung", similar_products=[], confidence=0.85,
            processing_time=0.2, embedding_model="rag_model"
        )
        
        combined_result = agent._combine_pipeline_results(
            ner_result, rag_result, None, ["NER", "RAG"], 0.0
        )
        
        assert isinstance(combined_result, HybridResult)
        assert combined_result.predicted_brand == "Samsung"
        assert combined_result.ner_contribution > 0.0
        assert combined_result.rag_contribution > 0.0
        assert combined_result.llm_contribution == 0.0  # No LLM stage
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_result_combination_conflicting_brands(self, hybrid_config):
        """Test result combination when stages predict different brands."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Different brand predictions from each stage
        ner_result = NERResult(
            entities=[EntityResult(EntityType.BRAND, "Samsung", 0.7, 0, 7)],
            confidence=0.7, processing_time=0.1, model_used="ner_model"
        )
        rag_result = RAGResult(
            predicted_brand="Apple", similar_products=[], confidence=0.8,
            processing_time=0.2, embedding_model="rag_model"
        )
        llm_result = LLMResult(
            predicted_brand="Samsung", reasoning="Brand analysis", confidence=0.9,
            processing_time=0.3, model_id="llm_model"
        )
        
        combined_result = agent._combine_pipeline_results(
            ner_result, rag_result, llm_result, ["NER", "RAG", "LLM"], 0.0
        )
        
        assert isinstance(combined_result, HybridResult)
        # Should select brand with highest weighted confidence
        assert combined_result.predicted_brand in ["Samsung", "Apple"]
        assert combined_result.confidence > 0.0

    # Early Termination Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_early_termination_ner(self, mock_ner_agent):
        """Test early termination after NER stage with high confidence."""
        config = {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "ner_confidence_threshold": 0.8,
            "use_early_termination": True,
            "ner_weight": 0.3,
            "rag_weight": 0.4,
            "llm_weight": 0.3
        }
        
        # Mock high-confidence NER result
        high_confidence_ner = NERResult(
            entities=[EntityResult(EntityType.BRAND, "Samsung", 0.95, 0, 7)],
            confidence=0.95, processing_time=0.1, model_used="ner_model"
        )
        mock_ner_agent.extract_entities = AsyncMock(return_value=high_confidence_ner)
        
        agent = SequentialHybridAgent(config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent,
            rag_agent=None,  # Should not be called due to early termination
            llm_agent=None
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
        assert "NER" in result.pipeline_steps
        assert "RAG" not in result.pipeline_steps  # Should terminate early
        assert "LLM" not in result.pipeline_steps
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_early_termination_rag(self, mock_ner_agent, mock_rag_agent):
        """Test early termination after RAG stage with high confidence."""
        config = {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "ner_confidence_threshold": 0.8,
            "rag_confidence_threshold": 0.8,
            "use_early_termination": True,
            "ner_weight": 0.3,
            "rag_weight": 0.4,
            "llm_weight": 0.3
        }
        
        # Mock low-confidence NER but high-confidence RAG
        low_confidence_ner = NERResult(
            entities=[], confidence=0.3, processing_time=0.1, model_used="ner_model"
        )
        high_confidence_rag = RAGResult(
            predicted_brand="Samsung", similar_products=[], confidence=0.9,
            processing_time=0.2, embedding_model="rag_model"
        )
        
        mock_ner_agent.extract_entities = AsyncMock(return_value=low_confidence_ner)
        mock_rag_agent.retrieve_and_infer = AsyncMock(return_value=high_confidence_rag)
        
        agent = SequentialHybridAgent(config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent,
            rag_agent=mock_rag_agent,
            llm_agent=None  # Should not be called due to early termination
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
        assert "NER" in result.pipeline_steps
        assert "RAG" in result.pipeline_steps
        assert "LLM" not in result.pipeline_steps  # Should terminate early

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_stage_failure_handling(self, hybrid_config, mock_rag_agent, mock_llm_agent):
        """Test hybrid agent handling of individual stage failures."""
        # Mock NER agent that fails
        failing_ner_agent = Mock()
        failing_ner_agent.extract_entities = AsyncMock(side_effect=Exception("NER failed"))
        
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=failing_ner_agent,
            rag_agent=mock_rag_agent,
            llm_agent=mock_llm_agent
        )
        
        # Should continue with other stages despite NER failure
        assert isinstance(result, HybridResult)
        assert "RAG" in result.pipeline_steps
        assert "LLM" in result.pipeline_steps
        # NER should not be in pipeline steps due to failure
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_all_stages_failure(self, hybrid_config):
        """Test hybrid agent handling when all stages fail."""
        # Mock agents that all fail
        failing_ner = Mock()
        failing_ner.extract_entities = AsyncMock(side_effect=Exception("NER failed"))
        
        failing_rag = Mock()
        failing_rag.retrieve_and_infer = AsyncMock(side_effect=Exception("RAG failed"))
        
        failing_llm = Mock()
        failing_llm.infer_brand = AsyncMock(side_effect=Exception("LLM failed"))
        
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=failing_ner,
            rag_agent=failing_rag,
            llm_agent=failing_llm
        )
        
        # Should return result with "Unknown" brand and low confidence
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Unknown"
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_process_not_initialized(self, hybrid_config):
        """Test hybrid agent processing when not initialized."""
        agent = SequentialHybridAgent(hybrid_config)
        # Don't initialize the agent
        
        with pytest.raises(AgentError) as exc_info:
            await agent.hybrid_inference("Samsung Galaxy S23")
        
        assert "Agent not initialized" in str(exc_info.value)

    # Resource Management Tests
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_cleanup(self, hybrid_config):
        """Test hybrid agent resource cleanup."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        assert agent.is_initialized()
        
        await agent.cleanup()
        
        assert not agent.is_initialized()
        assert agent.current_ner_agent is None
        assert agent.current_rag_agent is None
        assert agent.current_llm_agent is None
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_health_check_success(self, hybrid_config):
        """Test successful hybrid agent health check."""
        agent = SequentialHybridAgent(hybrid_config)
        await agent.initialize()
        
        # Health check should pass without raising exceptions
        await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_hybrid_agent_health_check_failure(self, hybrid_config):
        """Test hybrid agent health check failure scenarios."""
        agent = SequentialHybridAgent(hybrid_config)
        # Don't initialize the agent
        
        with pytest.raises(RuntimeError):
            await agent._perform_health_check()

    # Parameterized Tests
    
    @pytest.mark.parametrize("ner_weight,rag_weight,llm_weight", [
        (0.5, 0.3, 0.2),  # NER-heavy
        (0.2, 0.6, 0.2),  # RAG-heavy
        (0.2, 0.2, 0.6),  # LLM-heavy
        (0.33, 0.33, 0.34),  # Balanced
    ])
    @pytest.mark.asyncio
    async def test_hybrid_agent_different_weight_configurations(self, hybrid_config, mock_ner_agent, 
                                                               mock_rag_agent, mock_llm_agent,
                                                               ner_weight, rag_weight, llm_weight):
        """Test hybrid agent with different weight configurations."""
        config = hybrid_config.copy()
        config["ner_weight"] = ner_weight
        config["rag_weight"] = rag_weight
        config["llm_weight"] = llm_weight
        
        agent = SequentialHybridAgent(config)
        await agent.initialize()
        
        assert agent.ner_weight == ner_weight
        assert agent.rag_weight == rag_weight
        assert agent.llm_weight == llm_weight
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent,
            rag_agent=mock_rag_agent,
            llm_agent=mock_llm_agent
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
    
    @pytest.mark.parametrize("enable_ner,enable_rag,enable_llm", [
        (True, False, False),   # NER only
        (False, True, False),   # RAG only
        (False, False, True),   # LLM only
        (True, True, False),    # NER + RAG
        (True, False, True),    # NER + LLM
        (False, True, True),    # RAG + LLM
    ])
    @pytest.mark.asyncio
    async def test_hybrid_agent_stage_combinations(self, hybrid_config, mock_ner_agent, 
                                                  mock_rag_agent, mock_llm_agent,
                                                  enable_ner, enable_rag, enable_llm):
        """Test hybrid agent with different stage combinations."""
        config = hybrid_config.copy()
        config["enable_ner_stage"] = enable_ner
        config["enable_rag_stage"] = enable_rag
        config["enable_llm_stage"] = enable_llm
        
        agent = SequentialHybridAgent(config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent if enable_ner else None,
            rag_agent=mock_rag_agent if enable_rag else None,
            llm_agent=mock_llm_agent if enable_llm else None
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
        
        # Check that only enabled stages are in pipeline steps
        if enable_ner:
            assert "NER" in result.pipeline_steps
        if enable_rag:
            assert "RAG" in result.pipeline_steps
        if enable_llm:
            assert "LLM" in result.pipeline_steps


class TestOptimizedHybridAgent(BaseAgentTest):
    """
    Specific unit tests for OptimizedHybridAgent functionality.
    
    Tests enhanced decision making, dynamic thresholding, stage selection,
    and performance optimizations.
    """
    
    @pytest.fixture
    def optimized_config(self):
        """Configuration for optimized hybrid agent testing."""
        return {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "use_dynamic_thresholds": True,
            "use_stage_selection": True,
            "performance_mode": "balanced",
            "min_confidence_threshold": 0.6,
            "max_confidence_threshold": 0.9,
            "ner_weight": 0.3,
            "rag_weight": 0.4,
            "llm_weight": 0.3
        }
    
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_initialization(self, optimized_config):
        """Test optimized hybrid agent initialization with enhanced features."""
        agent = OptimizedHybridAgent(optimized_config)
        await agent.initialize()
        
        assert agent.is_initialized()
        assert agent.use_dynamic_thresholds is True
        assert agent.use_stage_selection is True
        assert agent.performance_mode == "balanced"
    
    @pytest.mark.parametrize("performance_mode,expected_early_termination", [
        ("fast", True),      # Fast mode should enable early termination
        ("balanced", True),  # Balanced mode should enable early termination
        ("accurate", False), # Accurate mode should disable early termination
    ])
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_performance_modes(self, optimized_config, 
                                                           performance_mode, expected_early_termination):
        """Test optimized hybrid agent with different performance modes."""
        config = optimized_config.copy()
        config["performance_mode"] = performance_mode
        
        agent = OptimizedHybridAgent(config)
        await agent.initialize()
        
        assert agent.performance_mode == performance_mode
        assert agent.use_early_termination == expected_early_termination
    
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_stage_selection(self, optimized_config):
        """Test dynamic stage selection based on input characteristics."""
        agent = OptimizedHybridAgent(optimized_config)
        await agent.initialize()
        
        # Test stage selection for short simple text
        short_stages = agent._select_optimal_stages("Samsung S23")
        assert isinstance(short_stages, dict)
        assert "ner" in short_stages
        assert "rag" in short_stages
        assert "llm" in short_stages
        
        # Test stage selection for mixed language text
        mixed_stages = agent._select_optimal_stages("Samsung โทรศัพท์")
        assert isinstance(mixed_stages, dict)
        # Mixed language should enable LLM
        assert mixed_stages["llm"] is True
    
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_mixed_language_detection(self, optimized_config):
        """Test mixed language detection for stage selection."""
        agent = OptimizedHybridAgent(optimized_config)
        await agent.initialize()
        
        # Test mixed Thai-English
        assert agent._detect_mixed_language("Samsung โทรศัพท์") is True
        
        # Test pure English
        assert agent._detect_mixed_language("Samsung Galaxy S23") is False
        
        # Test pure Thai
        assert agent._detect_mixed_language("โทรศัพท์มือถือ") is False
    
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_dynamic_confidence_adjustment(self, optimized_config):
        """Test dynamic confidence adjustment based on context."""
        agent = OptimizedHybridAgent(optimized_config)
        await agent.initialize()
        
        # Test confidence boost for multiple stages
        adjusted_confidence = agent._adjust_confidence_dynamically(
            0.7, "Samsung Galaxy S23", ["NER", "RAG", "LLM"]
        )
        assert adjusted_confidence > 0.7  # Should boost for multiple stages
        
        # Test confidence reduction for very short text
        adjusted_confidence = agent._adjust_confidence_dynamically(
            0.7, "S23", ["LLM"]
        )
        assert adjusted_confidence < 0.7  # Should reduce for very short text
        
        # Test confidence boost for brand indicators
        adjusted_confidence = agent._adjust_confidence_dynamically(
            0.7, "Samsung® Galaxy", ["LLM"]
        )
        assert adjusted_confidence > 0.7  # Should boost for brand indicators
    
    @pytest.mark.asyncio
    async def test_optimized_hybrid_agent_result_optimization(self, optimized_config, mock_ner_agent, mock_rag_agent, mock_llm_agent):
        """Test result optimization with dynamic adjustments."""
        agent = OptimizedHybridAgent(optimized_config)
        await agent.initialize()
        
        result = await agent.hybrid_inference(
            "Samsung Galaxy S23",
            ner_agent=mock_ner_agent,
            rag_agent=mock_rag_agent,
            llm_agent=mock_llm_agent
        )
        
        assert isinstance(result, HybridResult)
        assert result.predicted_brand == "Samsung"
        # Confidence should be optimized based on context
        assert result.confidence > 0.0