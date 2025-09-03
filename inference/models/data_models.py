"""
Core data models for multilingual product inference system.

This module defines the data structures used throughout the inference pipeline,
following PEP 8 standards and referencing patterns from the notebook implementation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum


class EntityType(Enum):
    """Entity types for NER processing."""
    
    BRAND = "BRAND"
    CATEGORY = "CATEGORY"
    VARIANT = "VARIANT"
    PRODUCT = "PRODUCT"
    ORG = "ORG"


class LanguageHint(Enum):
    """Language hints for multilingual processing."""
    
    ENGLISH = "en"
    THAI = "th"
    MIXED = "mixed"
    AUTO = "auto"


@dataclass
class ProductInput:
    """Input data structure for product inference."""
    
    product_name: str
    language_hint: Optional[LanguageHint] = LanguageHint.AUTO
    
    def __post_init__(self) -> None:
        """Validate input data."""
        if not self.product_name or not self.product_name.strip():
            raise ValueError("Product name cannot be empty")


@dataclass
class EntityResult:
    """Result structure for individual entity extraction."""
    
    entity_type: EntityType
    text: str
    confidence: float
    start_pos: int
    end_pos: int
    
    def __post_init__(self) -> None:
        """Validate entity result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.start_pos < 0 or self.end_pos < self.start_pos:
            raise ValueError("Invalid position values")


@dataclass
class NERResult:
    """Result structure for NER agent processing."""
    
    entities: List[EntityResult]
    confidence: float
    processing_time: float
    model_used: str = "spacy_default"
    
    def __post_init__(self) -> None:
        """Validate NER result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
    
    def get_brands(self) -> List[EntityResult]:
        """Extract brand entities from results."""
        return [entity for entity in self.entities if entity.entity_type == EntityType.BRAND]


@dataclass
class SimilarProduct:
    """Structure for similar product information in RAG results."""
    
    product_name: str
    brand: str
    category: str
    sub_category: str
    similarity_score: float
    
    def __post_init__(self) -> None:
        """Validate similar product data."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class RAGResult:
    """Result structure for RAG agent processing."""
    
    predicted_brand: str
    similar_products: List[SimilarProduct]
    confidence: float
    processing_time: float
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __post_init__(self) -> None:
        """Validate RAG result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass
class LLMResult:
    """Result structure for LLM agent processing."""
    
    predicted_brand: str
    reasoning: str
    confidence: float
    processing_time: float
    model_id: str = "amazon.nova-pro-v1:0"
    
    def __post_init__(self) -> None:
        """Validate LLM result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass
class HybridResult:
    """Result structure for hybrid agent processing."""
    
    predicted_brand: str
    ner_contribution: float
    rag_contribution: float
    llm_contribution: float
    confidence: float
    processing_time: float
    pipeline_steps: List[str]
    
    def __post_init__(self) -> None:
        """Validate hybrid result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        
        # Validate contribution weights sum to approximately 1.0
        total_contribution = self.ner_contribution + self.rag_contribution + self.llm_contribution
        if not (0.99 <= total_contribution <= 1.01):
            raise ValueError("Contribution weights must sum to 1.0")


@dataclass
class InferenceResult:
    """Complete inference result from orchestrator agent."""
    
    input_product: str
    ner_result: Optional[NERResult]
    rag_result: Optional[RAGResult]
    llm_result: Optional[LLMResult]
    hybrid_result: Optional[HybridResult]
    best_prediction: str
    best_confidence: float
    best_method: str
    total_processing_time: float
    
    def __post_init__(self) -> None:
        """Validate inference result data."""
        if not (0.0 <= self.best_confidence <= 1.0):
            raise ValueError("Best confidence must be between 0.0 and 1.0")
        if self.total_processing_time < 0:
            raise ValueError("Total processing time cannot be negative")
        
        # Ensure at least one result is present
        results = [self.ner_result, self.rag_result, self.llm_result, self.hybrid_result]
        if not any(results):
            raise ValueError("At least one inference result must be present")


@dataclass
class TrainingDataPoint:
    """Training data structure based on notebook format."""
    
    product_name: str
    brand: str
    category: str
    sub_category: str
    shop_id: Optional[str] = None
    product_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate training data."""
        required_fields = [self.product_name, self.brand, self.category, self.sub_category]
        if not all(field and field.strip() for field in required_fields):
            raise ValueError("All required fields must be non-empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching notebook structure."""
        return {
            "product_name": self.product_name,
            "brand": self.brand,
            "category": self.category,
            "sub_category": self.sub_category,
            "shop_id": self.shop_id,
            "product_id": self.product_id
        }


@dataclass
class VectorSearchResult:
    """Result structure for vector database searches."""
    
    hits: List[Dict[str, Any]]
    query_vector: List[float]
    search_time: float
    collection_name: str
    
    def __post_init__(self) -> None:
        """Validate vector search result."""
        if self.search_time < 0:
            raise ValueError("Search time cannot be negative")


@dataclass
class AgentHealth:
    """Health status structure for agent monitoring."""
    
    agent_name: str
    is_healthy: bool
    last_check: float
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate health data."""
        if self.response_time is not None and self.response_time < 0:
            raise ValueError("Response time cannot be negative")