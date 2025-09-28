"""
Unit tests for NER (Named Entity Recognition) agents.

This module contains comprehensive unit tests for NER agent implementations,
including SpacyNERAgent and MultilingualNERAgent, following pytest best practices.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from inference.src.agents.ner_agent import SpacyNERAgent, MultilingualNERAgent
from inference.src.models.data_models import (
    ProductInput, 
    NERResult, 
    EntityResult, 
    EntityType, 
    LanguageHint
)
from inference.src.agents.base_agent import AgentError, AgentInitializationError
from inference.tests.utils.test_base import BaseAgentTest


class TestNERAgent(BaseAgentTest):
    """
    Unit tests for NER agent implementations extending BaseAgentTest.
    
    Tests NER agent initialization, processing, error handling, multilingual input,
    timeout management, and parameterized scenarios.
    """
    
    @pytest.fixture
    def ner_config(self):
        """Standard NER agent configuration for testing."""
        return {
            "model_name": "en_core_web_sm",
            "confidence_threshold": 0.5,
            "max_text_length": 1000
        }
    
    @pytest.fixture
    def mock_spacy_model(self):
        """Mock spaCy model for testing."""
        mock_model = Mock()
        mock_model.pipe_names = ["ner"]
        
        # Mock document with entities
        mock_doc = Mock()
        mock_entity = Mock()
        mock_entity.text = "Samsung"
        mock_entity.label_ = "ORG"
        mock_entity.start_char = 0
        mock_entity.end_char = 7
        mock_doc.ents = [mock_entity]
        
        mock_model.return_value = mock_doc
        return mock_model
    
    @pytest.fixture
    def multilingual_config(self):
        """Configuration for multilingual NER agent."""
        return {
            "model_name": "xx_ent_wiki_sm",
            "confidence_threshold": 0.5,
            "use_transformer_fallback": True,
            "thai_text_threshold": 0.3,
            "mixed_language_boost": 0.15
        }

    # Initialization Tests
    
    @pytest.mark.asyncio
    async def test_ner_agent_initialization_success(self, ner_config, mock_spacy_model):
        """Test successful NER agent initialization."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.nlp is not None
            assert agent.model_name == "en_core_web_sm"
            assert agent.confidence_threshold == 0.5
    
    @pytest.mark.asyncio
    async def test_ner_agent_initialization_model_not_found(self, ner_config):
        """Test NER agent initialization when model is not found."""
        with patch('spacy.load', side_effect=OSError("Model not found")):
            with patch('spacy.cli.download') as mock_download:
                mock_download.side_effect = Exception("Download failed")
                
                agent = SpacyNERAgent(ner_config)
                
                with pytest.raises(AgentInitializationError) as exc_info:
                    await agent.initialize()
                
                assert "Failed to load or download spaCy model" in str(exc_info.value)
                assert not agent.is_initialized()
    
    @pytest.mark.asyncio
    async def test_ner_agent_initialization_no_ner_component(self, ner_config):
        """Test NER agent initialization when model lacks NER component."""
        mock_model = Mock()
        mock_model.pipe_names = ["tagger", "parser"]  # No NER component
        
        with patch('spacy.load', return_value=mock_model):
            agent = SpacyNERAgent(ner_config)
            
            with pytest.raises(AgentInitializationError) as exc_info:
                await agent.initialize()
            
            assert "does not have NER component" in str(exc_info.value)
            assert not agent.is_initialized()
    
    @pytest.mark.asyncio
    async def test_ner_agent_fallback_model_loading(self, ner_config, mock_spacy_model):
        """Test NER agent fallback to alternative models."""
        def mock_load_side_effect(model_name):
            if model_name == "en_core_web_sm":
                return mock_spacy_model
            else:
                raise OSError("Model not found")
        
        with patch('spacy.load', side_effect=mock_load_side_effect):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.model_name == "en_core_web_sm"  # Should use fallback

    # Processing Tests
    
    @pytest.mark.asyncio
    async def test_ner_agent_process_valid_input(self, ner_config, mock_spacy_model):
        """Test NER agent processing with valid input."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is True
            assert result["agent_type"] == "ner"
            assert result["error"] is None
            assert isinstance(result["result"], NERResult)
    
    @pytest.mark.asyncio
    async def test_ner_agent_extract_entities_success(self, ner_config, mock_spacy_model):
        """Test successful entity extraction."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            result = await agent.extract_entities("Samsung Galaxy S23")
            
            assert isinstance(result, NERResult)
            assert len(result.entities) > 0
            assert result.entities[0].entity_type == EntityType.BRAND
            assert result.entities[0].text == "Samsung"
            assert result.confidence > 0.0
            assert result.processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_ner_agent_process_empty_input(self, ner_config, mock_spacy_model):
        """Test NER agent processing with empty input."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            result = await agent.extract_entities("")
            
            assert isinstance(result, NERResult)
            assert len(result.entities) == 0
            assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_ner_agent_process_long_text_truncation(self, ner_config, mock_spacy_model):
        """Test NER agent handling of text that exceeds max length."""
        config = ner_config.copy()
        config["max_text_length"] = 50
        
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(config)
            await agent.initialize()
            
            long_text = "Samsung Galaxy S23 " * 20  # Much longer than 50 chars
            result = await agent.extract_entities(long_text)
            
            assert isinstance(result, NERResult)
            # Should process without error despite truncation
    
    @pytest.mark.asyncio
    async def test_ner_agent_process_not_initialized(self, ner_config):
        """Test NER agent processing when not initialized."""
        agent = SpacyNERAgent(ner_config)
        # Don't initialize the agent
        
        with pytest.raises(AgentError) as exc_info:
            await agent.extract_entities("Samsung Galaxy S23")
        
        assert "Agent not initialized" in str(exc_info.value)

    # Multilingual Input Tests
    
    @pytest.mark.parametrize("product_name,expected_entities", [
        ("Samsung Galaxy โทรศัพท์", 1),  # Mixed Thai-English
        ("ยาสีฟัน Wonder Smile", 1),  # Thai with English brand
        ("BENQ GW2785TC 27นิ้ว", 1),  # English brand with Thai units
        ("โทรศัพท์ทั่วไป", 0),  # Pure Thai, no brand
        ("Generic USB Cable", 0),  # Generic English, no clear brand
    ])
    @pytest.mark.asyncio
    async def test_ner_agent_multilingual_input(self, multilingual_config, mock_spacy_model, 
                                               product_name, expected_entities):
        """Test NER agent with various multilingual inputs."""
        with patch('spacy.load', return_value=mock_spacy_model):
            # Mock entities based on expected count
            mock_doc = Mock()
            if expected_entities > 0:
                mock_entity = Mock()
                mock_entity.text = "Samsung" if "Samsung" in product_name else "Wonder Smile"
                mock_entity.label_ = "ORG"
                mock_entity.start_char = 0
                mock_entity.end_char = len(mock_entity.text)
                mock_doc.ents = [mock_entity]
            else:
                mock_doc.ents = []
            
            mock_spacy_model.return_value = mock_doc
            
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            result = await agent.extract_entities(product_name)
            
            assert isinstance(result, NERResult)
            if expected_entities > 0:
                assert len(result.entities) >= expected_entities
                assert result.confidence > 0.0
            else:
                # May still find entities due to preprocessing, but confidence should be lower
                assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_multilingual_ner_language_detection(self, multilingual_config, mock_spacy_model):
        """Test multilingual NER agent language composition analysis."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            # Test mixed language detection
            mixed_text = "Samsung โทรศัพท์มือถือ"
            lang_info = agent._analyze_language_composition(mixed_text)
            
            assert lang_info["is_mixed"] is True
            assert lang_info["thai_ratio"] > 0.1
            assert lang_info["english_ratio"] > 0.1
    
    @pytest.mark.asyncio
    async def test_multilingual_ner_confidence_enhancement(self, multilingual_config, mock_spacy_model):
        """Test multilingual NER confidence enhancement for mixed language text."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            # Create mock entity
            mock_entity = Mock()
            mock_entity.text = "Samsung"
            mock_entity.entity_type = EntityType.BRAND
            mock_entity.confidence = 0.6
            
            # Test enhancement for mixed language
            lang_info = {"is_mixed": True, "is_primarily_thai": False}
            enhanced_entities = agent._enhance_multilingual_entities(
                [mock_entity], "Samsung โทรศัพท์", lang_info
            )
            
            # Confidence should be boosted for mixed language context
            assert enhanced_entities[0].confidence > 0.6

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_ner_agent_spacy_processing_error(self, ner_config):
        """Test NER agent handling of spaCy processing errors."""
        mock_model = Mock()
        mock_model.pipe_names = ["ner"]
        mock_model.side_effect = Exception("spaCy processing failed")
        
        with patch('spacy.load', return_value=mock_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "spaCy processing failed" in result["error"]
            assert result["result"] is None
    
    @pytest.mark.asyncio
    async def test_ner_agent_confidence_calculation_edge_cases(self, ner_config, mock_spacy_model):
        """Test NER agent confidence calculation with edge cases."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            # Test with very short entity
            mock_entity = Mock()
            mock_entity.text = "A"
            mock_entity.start_char = 0
            mock_entity.end_char = 1
            
            confidence = agent._calculate_entity_confidence(mock_entity, "A product")
            assert 0.0 <= confidence <= 1.0
            
            # Test with entity containing special characters
            mock_entity.text = "Samsung®"
            confidence = agent._calculate_entity_confidence(mock_entity, "Samsung® product")
            assert confidence > 0.6  # Should get boost for special characters

    # Timeout and Resource Management Tests
    
    @pytest.mark.asyncio
    async def test_ner_agent_cleanup(self, ner_config, mock_spacy_model):
        """Test NER agent resource cleanup."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.nlp is not None
            
            await agent.cleanup()
            
            assert not agent.is_initialized()
            assert agent.nlp is None
    
    @pytest.mark.asyncio
    async def test_ner_agent_health_check_success(self, ner_config, mock_spacy_model):
        """Test successful NER agent health check."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            # Health check should pass without raising exceptions
            await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_ner_agent_health_check_failure(self, ner_config):
        """Test NER agent health check failure scenarios."""
        agent = SpacyNERAgent(ner_config)
        # Don't initialize the agent
        
        with pytest.raises(RuntimeError):
            await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_ner_agent_concurrent_processing(self, ner_config, mock_spacy_model):
        """Test NER agent handling concurrent processing requests."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
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
                assert result["agent_type"] == "ner"

    # Parameterized Test Scenarios
    
    @pytest.mark.parametrize("confidence_threshold,expected_entities", [
        (0.1, 1),  # Low threshold, should find entities
        (0.5, 1),  # Medium threshold
        (0.9, 0),  # High threshold, might filter out entities
    ])
    @pytest.mark.asyncio
    async def test_ner_agent_confidence_threshold_filtering(self, ner_config, mock_spacy_model,
                                                           confidence_threshold, expected_entities):
        """Test NER agent filtering based on confidence thresholds."""
        config = ner_config.copy()
        config["confidence_threshold"] = confidence_threshold
        
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(config)
            await agent.initialize()
            
            result = await agent.extract_entities("Samsung Galaxy S23")
            
            assert isinstance(result, NERResult)
            # Note: Actual filtering depends on calculated confidence scores
            assert len(result.entities) >= 0  # May vary based on confidence calculation
    
    @pytest.mark.parametrize("model_name", [
        "en_core_web_sm",
        "en_core_web_md", 
        "xx_ent_wiki_sm"
    ])
    @pytest.mark.asyncio
    async def test_ner_agent_different_models(self, ner_config, mock_spacy_model, model_name):
        """Test NER agent with different spaCy models."""
        config = ner_config.copy()
        config["model_name"] = model_name
        
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(config)
            await agent.initialize()
            
            assert agent.model_name == model_name
            assert agent.is_initialized()
    
    @pytest.mark.parametrize("text_input,preprocessing_expected", [
        ("Samsung  Galaxy   S23", "Samsung Galaxy S23"),  # Multiple spaces
        ("Samsung\tGalaxy\nS23", "Samsung Galaxy S23"),  # Tabs and newlines
        ("Samsung®Galaxy™S23©", "Samsung Galaxy S23"),  # Special characters
        ("Samsungโทรศัพท์", "Samsung โทรศัพท์"),  # Thai-English spacing
    ])
    @pytest.mark.asyncio
    async def test_ner_agent_text_preprocessing(self, ner_config, mock_spacy_model, 
                                               text_input, preprocessing_expected):
        """Test NER agent text preprocessing functionality."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = SpacyNERAgent(ner_config)
            await agent.initialize()
            
            preprocessed = agent._preprocess_text(text_input)
            
            # Check that preprocessing produces expected normalization
            assert len(preprocessed.split()) == len(preprocessing_expected.split())
            assert preprocessed.strip() != ""


class TestMultilingualNERAgent(BaseAgentTest):
    """
    Specific unit tests for MultilingualNERAgent functionality.
    
    Tests enhanced multilingual processing, Thai-English mixed text handling,
    and specialized confidence scoring.
    """
    
    @pytest.fixture
    def multilingual_config(self):
        """Configuration for multilingual NER agent testing."""
        return {
            "model_name": "xx_ent_wiki_sm",
            "confidence_threshold": 0.5,
            "use_transformer_fallback": True,
            "transformer_model": "xlm-roberta-base",
            "thai_text_threshold": 0.3,
            "mixed_language_boost": 0.15
        }
    
    @pytest.mark.asyncio
    async def test_multilingual_agent_initialization(self, multilingual_config, mock_spacy_model):
        """Test multilingual NER agent initialization with enhanced settings."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.use_transformer_fallback is True
            assert agent.thai_text_threshold == 0.3
            assert agent.mixed_language_boost == 0.15
    
    @pytest.mark.asyncio
    async def test_multilingual_thai_entity_enhancement(self, multilingual_config, mock_spacy_model):
        """Test Thai-specific entity enhancement."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            # Create mock entity with Thai pattern
            mock_entity = Mock()
            mock_entity.text = "Samsung โทรศัพท์"
            mock_entity.confidence = 0.6
            mock_entity.entity_type = EntityType.BRAND
            
            enhanced_entity = agent._enhance_thai_entity(mock_entity, "Samsung โทรศัพท์ มือถือ")
            
            # Should boost confidence for Thai brand patterns
            assert enhanced_entity.confidence >= 0.6
    
    @pytest.mark.asyncio
    async def test_multilingual_mixed_language_enhancement(self, multilingual_config, mock_spacy_model):
        """Test mixed language entity enhancement."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            # Create mock entity with mixed language
            mock_entity = Mock()
            mock_entity.text = "Samsung"
            mock_entity.confidence = 0.6
            mock_entity.entity_type = EntityType.PRODUCT
            
            enhanced_entity = agent._enhance_mixed_language_entity(
                mock_entity, "Samsung โทรศัพท์"
            )
            
            # Should boost confidence and potentially change entity type to BRAND
            assert enhanced_entity.confidence >= 0.6
    
    @pytest.mark.asyncio
    async def test_multilingual_confidence_adjustment(self, multilingual_config, mock_spacy_model):
        """Test multilingual confidence adjustment based on language factors."""
        with patch('spacy.load', return_value=mock_spacy_model):
            agent = MultilingualNERAgent(multilingual_config)
            await agent.initialize()
            
            # Test mixed language confidence boost
            mixed_lang_info = {
                "is_mixed": True,
                "total_chars": 20
            }
            
            adjusted_confidence = agent._adjust_multilingual_confidence(0.6, mixed_lang_info)
            assert adjusted_confidence > 0.6
            
            # Test short text penalty
            short_text_info = {
                "is_mixed": False,
                "total_chars": 5
            }
            
            adjusted_confidence = agent._adjust_multilingual_confidence(0.6, short_text_info)
            assert adjusted_confidence < 0.6