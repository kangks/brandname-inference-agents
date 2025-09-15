"""
Retrieval-Augmented Generation (RAG) Agent implementation.

This module implements vector similarity search using sentence transformers and Milvus
for brand inference, following PEP 8 coding standards and referencing patterns from
the notebook implementation.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, connections
from pymilvus.exceptions import MilvusException

from ..models.data_models import (
    ProductInput,
    RAGResult,
    SimilarProduct,
    VectorSearchResult
)
from .base_agent import RAGAgent, AgentError, AgentInitializationError


class SentenceTransformerRAGAgent(RAGAgent):
    """
    RAG agent using SentenceTransformers and Milvus for vector similarity search.
    
    Implements multilingual embedding generation and vector database retrieval
    for brand inference based on product similarity patterns from the notebook.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize SentenceTransformer RAG agent.
        
        Args:
            config: Configuration dictionary containing model and database settings
        """
        super().__init__("sentence_transformer_rag", config)
        
        # Model configuration
        self.embedding_model_name = self.get_config_value(
            "embedding_model", 
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embedding_dimension = self.get_config_value("embedding_dimension", 384)
        
        # Milvus configuration
        self.milvus_uri = self.get_config_value("milvus_uri", "./milvus_rag.db")
        self.collection_name = self.get_config_value("collection_name", "product_brand")
        self.top_k = self.get_config_value("top_k", 5)
        
        # Inference configuration
        self.similarity_threshold = self.get_config_value("similarity_threshold", 0.7)
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.5)
        self.max_text_length = self.get_config_value("max_text_length", 512)
        
        # Initialize components
        self.embedding_model = None
        self.milvus_client = None
        
        # Brand inference weights for confidence calculation
        self.similarity_weight = self.get_config_value("similarity_weight", 0.6)
        self.frequency_weight = self.get_config_value("frequency_weight", 0.3)
        self.diversity_weight = self.get_config_value("diversity_weight", 0.1)
    
    async def initialize(self) -> None:
        """Initialize embedding model and Milvus connection."""
        try:
            # Initialize SentenceTransformer
            
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                self.embedding_model_name
            )
            
            # Verify embedding dimension
            test_embedding = self.embedding_model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            if actual_dim != self.embedding_dimension:
                self.logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {actual_dim}. Updating configuration."
                )
                self.embedding_dimension = actual_dim
            
            # Initialize Milvus client
            
            self.logger.info(f"Connecting to Milvus: {self.milvus_uri}")
            self.milvus_client = MilvusClient(self.milvus_uri)
            
            # Verify collection exists or create it
            await self._ensure_collection_exists()
            
            self.set_initialized(True)
            self.logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            self.set_initialized(False)
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to initialize RAG agent: {str(e)}",
                e
            )
    
    async def _ensure_collection_exists(self) -> None:
        """Ensure Milvus collection exists with proper schema."""
        try:
            # Check if collection exists
            collections = self.milvus_client.list_collections()
            
            if self.collection_name not in collections:
                self.logger.info(f"Creating Milvus collection: {self.collection_name}")
                
                # Create collection with schema matching notebook patterns
                self.milvus_client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.embedding_dimension,
                    description="Product brand similarity search for RAG inference"
                )
                
                self.logger.info(f"Created collection '{self.collection_name}'")
            else:
                self.logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to setup Milvus collection: {str(e)}",
                e
            )
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input and perform RAG inference.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing RAG results
        """
        start_time = time.time()
        
        try:
            result = await self.retrieve_and_infer(input_data.product_name)
            
            return {
                "agent_type": "rag",
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"RAG processing failed: {str(e)}")
            
            return {
                "agent_type": "rag",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def retrieve_and_infer(self, product_name: str) -> RAGResult:
        """
        Retrieve similar products and infer brand using vector similarity.
        
        Args:
            product_name: Input product name text
            
        Returns:
            RAGResult with brand prediction and similar products
        """
        start_time = time.time()
        
        if not self.embedding_model or not self.milvus_client:
            raise AgentError(self.agent_name, "Agent not initialized")
        
        # Preprocess and validate input
        cleaned_text = self._preprocess_text(product_name)
        
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = cleaned_text[:self.max_text_length]
            self.logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        # Generate embedding for query
        query_embedding = await self._generate_embedding(cleaned_text)
        
        # Search for similar products in Milvus
        search_result = await self._search_similar_products(query_embedding)
        
        # Infer brand from similar products
        predicted_brand, confidence = self._infer_brand_from_results(
            search_result.hits, 
            cleaned_text
        )
        
        # Convert hits to SimilarProduct objects
        similar_products = self._convert_hits_to_similar_products(search_result.hits)
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            predicted_brand=predicted_brand,
            similar_products=similar_products,
            confidence=confidence,
            processing_time=processing_time,
            embedding_model=self.embedding_model_name
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common product name patterns for better embedding
        # Remove excessive punctuation that might hurt embeddings
        text = re.sub(r'[^\w\s\u0E00-\u0E7F\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        try:
            # Run embedding generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode([text], normalize_embeddings=True)
            )
            
            return embedding[0].astype(np.float32)
            
        except Exception as e:
            raise AgentError(
                self.agent_name,
                f"Failed to generate embedding: {str(e)}"
            )
    
    async def _search_similar_products(self, query_embedding: np.ndarray) -> VectorSearchResult:
        """
        Search for similar products in Milvus vector database.
        
        Args:
            query_embedding: Query vector for similarity search
            
        Returns:
            VectorSearchResult with search hits and metadata
        """
        search_start = time.time()
        
        try:
            # Perform vector similarity search following notebook patterns
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding.tolist()],
                limit=self.top_k,
                output_fields=["product_name", "brand", "category", "sub_category"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}}
            )
            
            search_time = time.time() - search_start
            
            # Extract hits from search results
            hits = search_results[0] if search_results else []
            
            return VectorSearchResult(
                hits=hits,
                query_vector=query_embedding.tolist(),
                search_time=search_time,
                collection_name=self.collection_name
            )
            
        except Exception as e:
            raise AgentError(
                self.agent_name,
                f"Vector search failed: {str(e)}"
            )
    
    def _infer_brand_from_results(
        self, 
        hits: List[Dict[str, Any]], 
        query_text: str
    ) -> Tuple[str, float]:
        """
        Infer brand from search results using similarity and frequency analysis.
        
        Args:
            hits: Search results from Milvus
            query_text: Original query text for context
            
        Returns:
            Tuple of (predicted_brand, confidence_score)
        """
        if not hits:
            return "Unknown", 0.0
        
        # Collect brand candidates with their similarity scores
        brand_candidates = {}
        
        for hit in hits:
            try:
                # Extract brand and similarity score
                brand = hit.get("entity", {}).get("brand", "Unknown")
                similarity = 1.0 - hit.get("distance", 1.0)  # Convert distance to similarity
                
                if brand and brand != "Unknown":
                    if brand not in brand_candidates:
                        brand_candidates[brand] = {
                            "similarities": [],
                            "products": [],
                            "categories": set()
                        }
                    
                    brand_candidates[brand]["similarities"].append(similarity)
                    brand_candidates[brand]["products"].append(
                        hit.get("entity", {}).get("product_name", "")
                    )
                    
                    category = hit.get("entity", {}).get("category", "")
                    if category:
                        brand_candidates[brand]["categories"].add(category)
                        
            except (KeyError, AttributeError) as e:
                self.logger.warning(f"Error processing hit: {e}")
                continue
        
        if not brand_candidates:
            return "Unknown", 0.0
        
        # Calculate confidence for each brand candidate
        best_brand = None
        best_confidence = 0.0
        
        for brand, data in brand_candidates.items():
            confidence = self._calculate_brand_confidence(
                brand, 
                data, 
                query_text, 
                len(hits)
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_brand = brand
        
        return best_brand or "Unknown", best_confidence
    
    def _calculate_brand_confidence(
        self, 
        brand: str, 
        brand_data: Dict[str, Any], 
        query_text: str, 
        total_hits: int
    ) -> float:
        """
        Calculate confidence score for a brand candidate.
        
        Args:
            brand: Brand name
            brand_data: Dictionary with similarities, products, categories
            query_text: Original query text
            total_hits: Total number of search hits
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        similarities = brand_data["similarities"]
        products = brand_data["products"]
        categories = brand_data["categories"]
        
        # Similarity component (weighted average of top similarities)
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        similarity_score = (avg_similarity * 0.7 + max_similarity * 0.3)
        
        # Frequency component (how often this brand appears in results)
        frequency_score = len(similarities) / total_hits
        
        # Diversity component (variety of product categories)
        diversity_score = min(1.0, len(categories) / 3.0)  # Normalize to max 3 categories
        
        # Text matching component (brand name appears in query)
        text_match_score = 0.0
        if brand.lower() in query_text.lower():
            text_match_score = 0.2
        
        # Combine components with weights
        confidence = (
            similarity_score * self.similarity_weight +
            frequency_score * self.frequency_weight +
            diversity_score * self.diversity_weight +
            text_match_score
        )
        
        # Apply similarity threshold
        if max_similarity < self.similarity_threshold:
            confidence *= 0.5  # Reduce confidence for low similarity
        
        return min(1.0, confidence)
    
    def _convert_hits_to_similar_products(
        self, 
        hits: List[Dict[str, Any]]
    ) -> List[SimilarProduct]:
        """
        Convert Milvus search hits to SimilarProduct objects.
        
        Args:
            hits: Raw search hits from Milvus
            
        Returns:
            List of SimilarProduct objects
        """
        similar_products = []
        
        for hit in hits:
            try:
                entity = hit.get("entity", {})
                similarity = 1.0 - hit.get("distance", 1.0)  # Convert distance to similarity
                
                similar_product = SimilarProduct(
                    product_name=entity.get("product_name", ""),
                    brand=entity.get("brand", "Unknown"),
                    category=entity.get("category", ""),
                    sub_category=entity.get("sub_category", ""),
                    similarity_score=similarity
                )
                
                similar_products.append(similar_product)
                
            except (KeyError, AttributeError, ValueError) as e:
                self.logger.warning(f"Error converting hit to SimilarProduct: {e}")
                continue
        
        return similar_products
    
    async def cleanup(self) -> None:
        """Clean up RAG agent resources."""
        if self.milvus_client:
            try:
                # Close Milvus connection if needed
                # Note: MilvusClient doesn't have explicit close method in lite version
                self.milvus_client = None
            except Exception as e:
                self.logger.warning(f"Error closing Milvus connection: {e}")
        
        if self.embedding_model:
            # Clear embedding model reference
            self.embedding_model = None
        
        self.set_initialized(False)
        self.logger.info("RAG agent cleaned up")
    
    async def _perform_health_check(self) -> None:
        """Perform RAG agent specific health check."""
        await super()._perform_health_check()
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")
        
        if not self.milvus_client:
            raise RuntimeError("Milvus client not connected")
        
        # Test embedding generation
        try:
            test_embedding = await self._generate_embedding("test product")
            if test_embedding.shape[0] != self.embedding_dimension:
                raise RuntimeError("Embedding dimension mismatch")
        except Exception as e:
            raise RuntimeError(f"Embedding generation test failed: {e}")
        
        # Test Milvus connection
        try:
            collections = self.milvus_client.list_collections()
            if self.collection_name not in collections:
                raise RuntimeError(f"Collection {self.collection_name} not found")
        except Exception as e:
            raise RuntimeError(f"Milvus connection test failed: {e}")


class EnhancedRAGAgent(SentenceTransformerRAGAgent):
    """
    Enhanced RAG agent with additional features for better multilingual support.
    
    Extends the base RAG agent with improved text preprocessing, fuzzy matching,
    and enhanced confidence scoring for Thai-English mixed text.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize enhanced RAG agent.
        
        Args:
            config: Configuration dictionary with enhanced settings
        """
        super().__init__(config)
        self.agent_name = "enhanced_rag"
        
        # Enhanced configuration
        self.use_fuzzy_matching = self.get_config_value("use_fuzzy_matching", True)
        self.fuzzy_threshold = self.get_config_value("fuzzy_threshold", 0.8)
        self.multilingual_boost = self.get_config_value("multilingual_boost", 0.1)
        
        # Language detection patterns
        self.thai_pattern = r'[ก-๙]'
        self.english_pattern = r'[a-zA-Z]'
    
    def _preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing for multilingual support.
        
        Args:
            text: Raw input text
            
        Returns:
            Enhanced cleaned and normalized text
        """
        if not text:
            return ""
        
        # Apply base preprocessing
        text = super()._preprocess_text(text)
        
        # Enhanced multilingual preprocessing
        import re
        
        # Normalize Thai-English spacing
        text = re.sub(r'([a-zA-Z])([ก-๙])', r'\1 \2', text)
        text = re.sub(r'([ก-๙])([a-zA-Z])', r'\1 \2', text)
        
        # Handle common transliteration patterns
        transliteration_map = {
            'ซัมซุง': 'Samsung',
            'ไอโฟน': 'iPhone',
            'แอปเปิล': 'Apple',
            'โซนี่': 'Sony',
            'แอลจี': 'LG'
        }
        
        for thai, english in transliteration_map.items():
            if thai in text and english not in text:
                text = f"{text} {english}"
        
        return text.strip()
    
    def _calculate_brand_confidence(
        self, 
        brand: str, 
        brand_data: Dict[str, Any], 
        query_text: str, 
        total_hits: int
    ) -> float:
        """
        Enhanced confidence calculation with multilingual factors.
        
        Args:
            brand: Brand name
            brand_data: Dictionary with similarities, products, categories
            query_text: Original query text
            total_hits: Total number of search hits
            
        Returns:
            Enhanced confidence score between 0.0 and 1.0
        """
        # Get base confidence
        base_confidence = super()._calculate_brand_confidence(
            brand, brand_data, query_text, total_hits
        )
        
        # Apply multilingual enhancements
        enhanced_confidence = base_confidence
        
        # Detect mixed language context
        import re
        has_thai = bool(re.search(self.thai_pattern, query_text))
        has_english = bool(re.search(self.english_pattern, query_text))
        
        if has_thai and has_english:
            # Mixed language context often indicates international brands
            enhanced_confidence = min(1.0, enhanced_confidence + self.multilingual_boost)
        
        # Fuzzy matching boost
        if self.use_fuzzy_matching:
            fuzzy_boost = self._calculate_fuzzy_match_boost(brand, query_text)
            enhanced_confidence = min(1.0, enhanced_confidence + fuzzy_boost)
        
        return enhanced_confidence
    
    def _calculate_fuzzy_match_boost(self, brand: str, query_text: str) -> float:
        """
        Calculate confidence boost based on fuzzy string matching.
        
        Args:
            brand: Brand name to match
            query_text: Query text to search in
            
        Returns:
            Confidence boost value
        """
        try:
            from difflib import SequenceMatcher
            
            # Check for fuzzy matches of brand name in query
            brand_lower = brand.lower()
            query_lower = query_text.lower()
            
            # Direct substring match
            if brand_lower in query_lower:
                return 0.15
            
            # Fuzzy matching for partial matches
            words = query_lower.split()
            max_similarity = 0.0
            
            for word in words:
                if len(word) >= 3:  # Only check meaningful words
                    similarity = SequenceMatcher(None, brand_lower, word).ratio()
                    max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= self.fuzzy_threshold:
                return 0.1 * max_similarity
            
            return 0.0
            
        except ImportError:
            # difflib not available, skip fuzzy matching
            return 0.0
        except Exception as e:
            self.logger.warning(f"Fuzzy matching error: {e}")
            return 0.0