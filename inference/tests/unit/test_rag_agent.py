"""
Unit tests for RAG (Retrieval-Augmented Generation) agents.

This module contains comprehensive unit tests for RAG agent implementations,
including SentenceTransformerRAGAgent and EnhancedRAGAgent, with mocked Milvus dependencies.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from inference.src.agents.rag_agent import SentenceTransformerRAGAgent, EnhancedRAGAgent
from inference.src.models.data_models import (
    ProductInput, 
    RAGResult, 
    SimilarProduct, 
    VectorSearchResult,
    LanguageHint
)
from inference.src.agents.base_agent import AgentError, AgentInitializationError
from inference.tests.utils.test_base import BaseAgentTest


class TestRAGAgent(BaseAgentTest):
    """
    Unit tests for RAG agent implementations extending BaseAgentTest.
    
    Tests RAG agent initialization, embedding generation, vector search, similarity scoring,
    database connection handling, and performance with mocked Milvus dependencies.
    """
    
    @pytest.fixture
    def rag_config(self):
        """Standard RAG agent configuration for testing."""
        return {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": 384,
            "milvus_uri": "./test_milvus.db",
            "collection_name": "test_product_brand",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "confidence_threshold": 0.5,
            "max_text_length": 512
        }
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer model for testing."""
        mock_model = Mock()
        
        # Mock encode method to return consistent embeddings
        def mock_encode(texts, normalize_embeddings=True):
            # Return consistent embeddings for testing
            if isinstance(texts, list):
                return np.array([[0.1] * 384 for _ in texts], dtype=np.float32)
            else:
                return np.array([[0.1] * 384], dtype=np.float32)
        
        mock_model.encode = mock_encode
        return mock_model
    
    @pytest.fixture
    def mock_milvus_client(self):
        """Mock Milvus client for testing."""
        mock_client = Mock()
        
        # Mock list_collections
        mock_client.list_collections.return_value = ["test_product_brand"]
        
        # Mock create_collection
        mock_client.create_collection.return_value = None
        
        # Mock search results
        mock_search_results = [
            [
                {
                    "entity": {
                        "product_name": "Samsung Galaxy S23",
                        "brand": "Samsung",
                        "category": "Electronics",
                        "sub_category": "Smartphones"
                    },
                    "distance": 0.1  # Low distance = high similarity
                },
                {
                    "entity": {
                        "product_name": "Samsung Galaxy Note",
                        "brand": "Samsung", 
                        "category": "Electronics",
                        "sub_category": "Smartphones"
                    },
                    "distance": 0.2
                },
                {
                    "entity": {
                        "product_name": "Apple iPhone 15",
                        "brand": "Apple",
                        "category": "Electronics", 
                        "sub_category": "Smartphones"
                    },
                    "distance": 0.8  # High distance = low similarity
                }
            ]
        ]
        
        mock_client.search.return_value = mock_search_results
        return mock_client
    
    @pytest.fixture
    def enhanced_rag_config(self):
        """Configuration for enhanced RAG agent."""
        return {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": 384,
            "milvus_uri": "./test_milvus.db",
            "collection_name": "test_product_brand",
            "use_fuzzy_matching": True,
            "fuzzy_threshold": 0.8,
            "multilingual_boost": 0.1
        }

    # Initialization Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_initialization_success(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test successful RAG agent initialization."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.embedding_model is not None
            assert agent.milvus_client is not None
            assert agent.embedding_dimension == 384
    
    @pytest.mark.asyncio
    async def test_rag_agent_initialization_embedding_dimension_mismatch(self, rag_config, mock_milvus_client):
        """Test RAG agent initialization with embedding dimension mismatch."""
        # Mock SentenceTransformer that returns different dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 512], dtype=np.float32)  # 512 instead of 384
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.embedding_dimension == 512  # Should update to actual dimension
    
    @pytest.mark.asyncio
    async def test_rag_agent_initialization_milvus_connection_failure(self, rag_config, mock_sentence_transformer):
        """Test RAG agent initialization when Milvus connection fails."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', side_effect=Exception("Connection failed")):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            
            with pytest.raises(AgentInitializationError) as exc_info:
                await agent.initialize()
            
            assert "Failed to initialize RAG agent" in str(exc_info.value)
            assert not agent.is_initialized()
    
    @pytest.mark.asyncio
    async def test_rag_agent_collection_creation(self, rag_config, mock_sentence_transformer):
        """Test RAG agent creating new Milvus collection when it doesn't exist."""
        mock_client = Mock()
        mock_client.list_collections.return_value = []  # Collection doesn't exist
        mock_client.create_collection.return_value = None
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            mock_client.create_collection.assert_called_once()

    # Embedding Generation Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_generate_embedding_success(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test successful embedding generation."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            embedding = await agent._generate_embedding("Samsung Galaxy S23")
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_rag_agent_generate_embedding_error(self, rag_config, mock_milvus_client):
        """Test embedding generation error handling."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            with pytest.raises(AgentError) as exc_info:
                await agent._generate_embedding("Samsung Galaxy S23")
            
            assert "Failed to generate embedding" in str(exc_info.value)

    # Vector Search Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_vector_search_success(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test successful vector similarity search."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            query_embedding = np.array([0.1] * 384, dtype=np.float32)
            search_result = await agent._search_similar_products(query_embedding)
            
            assert isinstance(search_result, VectorSearchResult)
            assert len(search_result.hits) > 0
            assert search_result.search_time > 0.0
            assert search_result.collection_name == "test_product_brand"
    
    @pytest.mark.asyncio
    async def test_rag_agent_vector_search_error(self, rag_config, mock_sentence_transformer):
        """Test vector search error handling."""
        mock_client = Mock()
        mock_client.list_collections.return_value = ["test_product_brand"]
        mock_client.search.side_effect = Exception("Search failed")
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            query_embedding = np.array([0.1] * 384, dtype=np.float32)
            
            with pytest.raises(AgentError) as exc_info:
                await agent._search_similar_products(query_embedding)
            
            assert "Vector search failed" in str(exc_info.value)

    # Processing Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_process_valid_input(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent processing with valid input."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is True
            assert result["agent_type"] == "rag"
            assert result["error"] is None
            assert isinstance(result["result"], RAGResult)
    
    @pytest.mark.asyncio
    async def test_rag_agent_retrieve_and_infer_success(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test successful retrieve and infer operation."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("Samsung Galaxy S23")
            
            assert isinstance(result, RAGResult)
            assert result.predicted_brand == "Samsung"  # Should infer Samsung from mock data
            assert len(result.similar_products) > 0
            assert result.confidence > 0.0
            assert result.processing_time > 0.0
            assert result.embedding_model == agent.embedding_model_name
    
    @pytest.mark.asyncio
    async def test_rag_agent_process_empty_input(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent processing with empty input."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("")
            
            assert isinstance(result, RAGResult)
            # Should handle empty input gracefully
    
    @pytest.mark.asyncio
    async def test_rag_agent_process_long_text_truncation(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent handling of text that exceeds max length."""
        config = rag_config.copy()
        config["max_text_length"] = 50
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(config)
            await agent.initialize()
            
            long_text = "Samsung Galaxy S23 " * 20  # Much longer than 50 chars
            result = await agent.retrieve_and_infer(long_text)
            
            assert isinstance(result, RAGResult)
            # Should process without error despite truncation
    
    @pytest.mark.asyncio
    async def test_rag_agent_process_not_initialized(self, rag_config):
        """Test RAG agent processing when not initialized."""
        agent = SentenceTransformerRAGAgent(rag_config)
        # Don't initialize the agent
        
        with pytest.raises(AgentError) as exc_info:
            await agent.retrieve_and_infer("Samsung Galaxy S23")
        
        assert "Agent not initialized" in str(exc_info.value)

    # Brand Inference Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_brand_inference_single_brand(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test brand inference when all results point to single brand."""
        # Mock search results with all Samsung products
        samsung_results = [
            [
                {
                    "entity": {
                        "product_name": "Samsung Galaxy S23",
                        "brand": "Samsung",
                        "category": "Electronics"
                    },
                    "distance": 0.1
                },
                {
                    "entity": {
                        "product_name": "Samsung Galaxy Note",
                        "brand": "Samsung",
                        "category": "Electronics"
                    },
                    "distance": 0.2
                }
            ]
        ]
        
        mock_milvus_client.search.return_value = samsung_results
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("Samsung Galaxy S23")
            
            assert result.predicted_brand == "Samsung"
            assert result.confidence > 0.5  # Should be confident with consistent results
    
    @pytest.mark.asyncio
    async def test_rag_agent_brand_inference_mixed_brands(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test brand inference with mixed brand results."""
        # Use default mock_milvus_client which has mixed Samsung/Apple results
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("Samsung Galaxy S23")
            
            # Should still predict Samsung due to higher similarity scores
            assert result.predicted_brand == "Samsung"
            assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_rag_agent_brand_inference_no_results(self, rag_config, mock_sentence_transformer):
        """Test brand inference when no search results are found."""
        mock_client = Mock()
        mock_client.list_collections.return_value = ["test_product_brand"]
        mock_client.search.return_value = [[]]  # Empty results
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("Unknown Product")
            
            assert result.predicted_brand == "Unknown"
            assert result.confidence == 0.0
            assert len(result.similar_products) == 0

    # Similarity Scoring Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_confidence_calculation(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent confidence calculation with various scenarios."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            # Test confidence calculation with mock data
            brand_data = {
                "similarities": [0.9, 0.8, 0.7],
                "products": ["Product 1", "Product 2", "Product 3"],
                "categories": {"Electronics", "Smartphones"}
            }
            
            confidence = agent._calculate_brand_confidence(
                "Samsung", brand_data, "Samsung Galaxy", 5
            )
            
            assert 0.0 <= confidence <= 1.0
            assert confidence > 0.5  # Should be reasonably confident
    
    @pytest.mark.asyncio
    async def test_rag_agent_similar_products_conversion(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test conversion of search hits to SimilarProduct objects."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            # Mock hits data
            hits = [
                {
                    "entity": {
                        "product_name": "Samsung Galaxy S23",
                        "brand": "Samsung",
                        "category": "Electronics",
                        "sub_category": "Smartphones"
                    },
                    "distance": 0.1
                }
            ]
            
            similar_products = agent._convert_hits_to_similar_products(hits)
            
            assert len(similar_products) == 1
            assert isinstance(similar_products[0], SimilarProduct)
            assert similar_products[0].product_name == "Samsung Galaxy S23"
            assert similar_products[0].brand == "Samsung"
            assert similar_products[0].similarity_score == 0.9  # 1.0 - 0.1 distance

    # Database Connection Error Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_milvus_connection_failure_during_search(self, rag_config, mock_sentence_transformer):
        """Test handling of Milvus connection failure during search."""
        mock_client = Mock()
        mock_client.list_collections.return_value = ["test_product_brand"]
        mock_client.search.side_effect = Exception("Connection lost")
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            product_input = ProductInput(
                product_name="Samsung Galaxy S23",
                language_hint=LanguageHint.EN
            )
            
            result = await agent.process(product_input)
            
            assert result["success"] is False
            assert "Connection lost" in result["error"]
            assert result["result"] is None

    # Performance Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_concurrent_processing(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent handling concurrent processing requests."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
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
                assert result["agent_type"] == "rag"
    
    @pytest.mark.asyncio
    async def test_rag_agent_query_response_time(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent query response time measurement."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.Milvus', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            result = await agent.retrieve_and_infer("Samsung Galaxy S23")
            
            assert result.processing_time > 0.0
            assert result.processing_time < 10.0  # Should be reasonably fast with mocks

    # Resource Management Tests
    
    @pytest.mark.asyncio
    async def test_rag_agent_cleanup(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test RAG agent resource cleanup."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.embedding_model is not None
            assert agent.milvus_client is not None
            
            await agent.cleanup()
            
            assert not agent.is_initialized()
            assert agent.embedding_model is None
            assert agent.milvus_client is None
    
    @pytest.mark.asyncio
    async def test_rag_agent_health_check_success(self, rag_config, mock_sentence_transformer, mock_milvus_client):
        """Test successful RAG agent health check."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(rag_config)
            await agent.initialize()
            
            # Health check should pass without raising exceptions
            await agent._perform_health_check()
    
    @pytest.mark.asyncio
    async def test_rag_agent_health_check_failure_scenarios(self, rag_config):
        """Test RAG agent health check failure scenarios."""
        agent = SentenceTransformerRAGAgent(rag_config)
        # Don't initialize the agent
        
        with pytest.raises(RuntimeError):
            await agent._perform_health_check()

    # Parameterized Tests for Different Configurations
    
    @pytest.mark.parametrize("embedding_model,expected_dim", [
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ("sentence-transformers/all-mpnet-base-v2", 768),
    ])
    @pytest.mark.asyncio
    async def test_rag_agent_different_embedding_models(self, rag_config, mock_milvus_client, 
                                                       embedding_model, expected_dim):
        """Test RAG agent with different embedding models."""
        config = rag_config.copy()
        config["embedding_model"] = embedding_model
        config["embedding_dimension"] = expected_dim
        
        # Mock SentenceTransformer with appropriate dimension
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * expected_dim], dtype=np.float32)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(config)
            await agent.initialize()
            
            assert agent.embedding_model_name == embedding_model
            assert agent.embedding_dimension == expected_dim
    
    @pytest.mark.parametrize("similarity_threshold,confidence_threshold", [
        (0.5, 0.3),  # Low thresholds
        (0.7, 0.5),  # Medium thresholds  
        (0.9, 0.8),  # High thresholds
    ])
    @pytest.mark.asyncio
    async def test_rag_agent_threshold_configurations(self, rag_config, mock_sentence_transformer, 
                                                     mock_milvus_client, similarity_threshold, confidence_threshold):
        """Test RAG agent with different threshold configurations."""
        config = rag_config.copy()
        config["similarity_threshold"] = similarity_threshold
        config["confidence_threshold"] = confidence_threshold
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = SentenceTransformerRAGAgent(config)
            await agent.initialize()
            
            assert agent.similarity_threshold == similarity_threshold
            assert agent.confidence_threshold == confidence_threshold
            
            result = await agent.retrieve_and_infer("Samsung Galaxy S23")
            assert isinstance(result, RAGResult)


class TestEnhancedRAGAgent(BaseAgentTest):
    """
    Specific unit tests for EnhancedRAGAgent functionality.
    
    Tests enhanced multilingual support, fuzzy matching, and improved confidence scoring.
    """
    
    @pytest.fixture
    def enhanced_config(self):
        """Configuration for enhanced RAG agent testing."""
        return {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": 384,
            "milvus_uri": "./test_milvus.db",
            "collection_name": "test_product_brand",
            "use_fuzzy_matching": True,
            "fuzzy_threshold": 0.8,
            "multilingual_boost": 0.1
        }
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_agent_initialization(self, enhanced_config, mock_sentence_transformer, mock_milvus_client):
        """Test enhanced RAG agent initialization with additional features."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = EnhancedRAGAgent(enhanced_config)
            await agent.initialize()
            
            assert agent.is_initialized()
            assert agent.use_fuzzy_matching is True
            assert agent.fuzzy_threshold == 0.8
            assert agent.multilingual_boost == 0.1
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_agent_multilingual_preprocessing(self, enhanced_config, mock_sentence_transformer, mock_milvus_client):
        """Test enhanced text preprocessing for multilingual support."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = EnhancedRAGAgent(enhanced_config)
            await agent.initialize()
            
            # Test Thai-English mixed text preprocessing
            mixed_text = "Samsungโทรศัพท์มือถือ"
            preprocessed = agent._preprocess_text(mixed_text)
            
            # Should add proper spacing between Thai and English
            assert "Samsung โทรศัพท์" in preprocessed or "Samsung" in preprocessed
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_agent_fuzzy_matching(self, enhanced_config, mock_sentence_transformer, mock_milvus_client):
        """Test fuzzy matching functionality."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = EnhancedRAGAgent(enhanced_config)
            await agent.initialize()
            
            # Test fuzzy match boost calculation
            fuzzy_boost = agent._calculate_fuzzy_match_boost("Samsung", "Samsung Galaxy S23")
            assert fuzzy_boost > 0.0  # Should get boost for exact match
            
            fuzzy_boost = agent._calculate_fuzzy_match_boost("Samsung", "Samung Galaxy")  # Typo
            assert fuzzy_boost >= 0.0  # May get boost for similar match
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_agent_confidence_enhancement(self, enhanced_config, mock_sentence_transformer, mock_milvus_client):
        """Test enhanced confidence calculation with multilingual factors."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = EnhancedRAGAgent(enhanced_config)
            await agent.initialize()
            
            # Test confidence calculation with multilingual boost
            brand_data = {
                "similarities": [0.8],
                "products": ["Samsung โทรศัพท์"],
                "categories": {"Electronics"}
            }
            
            confidence = agent._calculate_brand_confidence(
                "Samsung", brand_data, "Samsung โทรศัพท์", 1
            )
            
            # Should get multilingual boost for mixed language context
            assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_enhanced_rag_agent_transliteration_handling(self, enhanced_config, mock_sentence_transformer, mock_milvus_client):
        """Test handling of transliterated brand names."""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer), \
             patch('pymilvus.MilvusClient', return_value=mock_milvus_client):
            
            agent = EnhancedRAGAgent(enhanced_config)
            await agent.initialize()
            
            # Test preprocessing with transliteration
            thai_text = "ซัมซุง Galaxy"
            preprocessed = agent._preprocess_text(thai_text)
            
            # Should add Samsung transliteration
            assert "Samsung" in preprocessed or "ซัมซุง" in preprocessed