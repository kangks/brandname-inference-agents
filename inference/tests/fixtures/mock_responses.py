"""
Mock response data for inference system testing.

This module provides standardized mock responses for all agent types
and API endpoints, ensuring consistent testing across the system.
"""

import time
from typing import Dict, Any, List
from dataclasses import asdict

from inference.src.models.data_models import (
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    EntityResult,
    EntityType,
    SimilarProduct,
    AgentHealth
)


# Mock NER Results
MOCK_NER_RESULTS = {
    "samsung_success": NERResult(
        entities=[
            EntityResult(
                entity_type=EntityType.BRAND,
                text="Samsung",
                confidence=0.95,
                start_pos=0,
                end_pos=7
            ),
            EntityResult(
                entity_type=EntityType.PRODUCT,
                text="Galaxy S24 Ultra",
                confidence=0.88,
                start_pos=8,
                end_pos=24
            )
        ],
        confidence=0.92,
        processing_time=0.15,
        model_used="en_core_web_sm"
    ),
    
    "apple_success": NERResult(
        entities=[
            EntityResult(
                entity_type=EntityType.BRAND,
                text="Apple",
                confidence=0.98,
                start_pos=0,
                end_pos=5
            ),
            EntityResult(
                entity_type=EntityType.PRODUCT,
                text="iPhone",
                confidence=0.95,
                start_pos=6,
                end_pos=12
            )
        ],
        confidence=0.96,
        processing_time=0.12,
        model_used="en_core_web_sm"
    ),
    
    "no_entities": NERResult(
        entities=[],
        confidence=0.0,
        processing_time=0.08,
        model_used="en_core_web_sm"
    ),
    
    "low_confidence": NERResult(
        entities=[
            EntityResult(
                entity_type=EntityType.BRAND,
                text="UnknownBrand",
                confidence=0.45,
                start_pos=0,
                end_pos=12
            )
        ],
        confidence=0.45,
        processing_time=0.18,
        model_used="en_core_web_sm"
    )
}


# Mock RAG Results
MOCK_RAG_RESULTS = {
    "samsung_success": RAGResult(
        similar_products=[
            SimilarProduct(
                product_name="Samsung Galaxy S24 Ultra 256GB",
                brand="Samsung",
                category="Electronics",
                sub_category="Smartphones",
                similarity_score=0.95
            ),
            SimilarProduct(
                product_name="Samsung Galaxy S23 Ultra",
                brand="Samsung",
                category="Electronics",
                sub_category="Smartphones",
                similarity_score=0.88
            )
        ],
        predicted_brand="Samsung",
        confidence=0.91,
        processing_time=0.25,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    ),
    
    "apple_success": RAGResult(
        similar_products=[
            SimilarProduct(
                product_name="iPhone 15 Pro Max 256GB",
                brand="Apple",
                category="Electronics",
                sub_category="Smartphones",
                similarity_score=0.93
            ),
            SimilarProduct(
                product_name="iPhone 14 Pro Max",
                brand="Apple",
                category="Electronics",
                sub_category="Smartphones",
                similarity_score=0.85
            )
        ],
        predicted_brand="Apple",
        confidence=0.89,
        processing_time=0.22,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    ),
    
    "no_matches": RAGResult(
        similar_products=[],
        predicted_brand="Unknown",
        confidence=0.0,
        processing_time=0.18,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    ),
    
    "low_similarity": RAGResult(
        similar_products=[
            SimilarProduct(
                product_name="Generic Product",
                brand="Generic",
                category="Unknown",
                sub_category="Unknown",
                similarity_score=0.45
            )
        ],
        predicted_brand="Generic",
        confidence=0.45,
        processing_time=0.28,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    )
}


# Mock LLM Results
MOCK_LLM_RESULTS = {
    "samsung_success": LLMResult(
        predicted_brand="Samsung",
        reasoning="The product name clearly indicates Samsung as the brand, with 'Galaxy S24 Ultra' being a well-known Samsung smartphone model. The confidence is high due to the explicit brand mention and recognizable product line.",
        confidence=0.94,
        processing_time=0.45,
        model_id="us.amazon.nova-pro-v1:0"
    ),
    
    "apple_success": LLMResult(
        predicted_brand="Apple",
        reasoning="The product name contains 'iPhone' which is exclusively an Apple product. The model '15 Pro Max' is consistent with Apple's naming convention for their premium smartphone line.",
        confidence=0.96,
        processing_time=0.42,
        model_id="us.amazon.nova-pro-v1:0"
    ),
    
    "uncertain": LLMResult(
        predicted_brand="Unknown",
        reasoning="The product description does not contain clear brand indicators. While there are product features mentioned, no specific brand name or distinctive product identifiers are present.",
        confidence=0.25,
        processing_time=0.38,
        model_id="us.amazon.nova-pro-v1:0"
    ),
    
    "nike_success": LLMResult(
        predicted_brand="Nike",
        reasoning="The product name includes 'Nike Air Max 270', which is a distinctive Nike product line. Air Max is a well-established Nike brand for athletic footwear.",
        confidence=0.92,
        processing_time=0.41,
        model_id="us.amazon.nova-pro-v1:0"
    )
}


# Mock Hybrid Results
MOCK_HYBRID_RESULTS = {
    "samsung_success": HybridResult(
        final_prediction="Samsung",
        confidence=0.93,
        processing_time=0.82,
        stage_results={
            "ner": {"brand": "Samsung", "confidence": 0.95},
            "rag": {"brand": "Samsung", "confidence": 0.91},
            "llm": {"brand": "Samsung", "confidence": 0.94}
        },
        stages_used=["ner", "rag", "llm"]
    ),
    
    "apple_success": HybridResult(
        final_prediction="Apple",
        confidence=0.94,
        processing_time=0.79,
        stage_results={
            "ner": {"brand": "Apple", "confidence": 0.98},
            "rag": {"brand": "Apple", "confidence": 0.89},
            "llm": {"brand": "Apple", "confidence": 0.96}
        },
        stages_used=["ner", "rag", "llm"]
    ),
    
    "partial_stages": HybridResult(
        final_prediction="Nike",
        confidence=0.88,
        processing_time=0.55,
        stage_results={
            "ner": {"brand": "Nike", "confidence": 0.85},
            "llm": {"brand": "Nike", "confidence": 0.92}
        },
        stages_used=["ner", "llm"]
    ),
    
    "low_confidence": HybridResult(
        final_prediction="Unknown",
        confidence=0.35,
        processing_time=0.65,
        stage_results={
            "ner": {"brand": "Unknown", "confidence": 0.20},
            "rag": {"brand": "Generic", "confidence": 0.45},
            "llm": {"brand": "Unknown", "confidence": 0.40}
        },
        stages_used=["ner", "rag", "llm"]
    )
}


# Mock Agent Health Results
MOCK_HEALTH_RESULTS = {
    "healthy_agent": AgentHealth(
        agent_name="test_agent",
        is_healthy=True,
        last_check=time.time(),
        error_message=None,
        response_time=0.05
    ),
    
    "unhealthy_agent": AgentHealth(
        agent_name="test_agent",
        is_healthy=False,
        last_check=time.time(),
        error_message="Agent initialization failed",
        response_time=None
    ),
    
    "slow_agent": AgentHealth(
        agent_name="test_agent",
        is_healthy=True,
        last_check=time.time(),
        error_message=None,
        response_time=2.5
    )
}


# Mock Agent Response Structures
MOCK_AGENT_RESPONSES = {
    "ner_success": {
        "agent_type": "ner",
        "result": MOCK_NER_RESULTS["samsung_success"],
        "success": True,
        "error": None
    },
    
    "ner_failure": {
        "agent_type": "ner",
        "result": None,
        "success": False,
        "error": "NER model initialization failed"
    },
    
    "rag_success": {
        "agent_type": "rag",
        "result": MOCK_RAG_RESULTS["samsung_success"],
        "success": True,
        "error": None
    },
    
    "rag_failure": {
        "agent_type": "rag",
        "result": None,
        "success": False,
        "error": "Milvus database connection failed"
    },
    
    "llm_success": {
        "agent_type": "llm",
        "result": MOCK_LLM_RESULTS["samsung_success"],
        "success": True,
        "error": None
    },
    
    "llm_failure": {
        "agent_type": "llm",
        "result": None,
        "success": False,
        "error": "AWS Bedrock service unavailable"
    },
    
    "hybrid_success": {
        "agent_type": "hybrid",
        "result": MOCK_HYBRID_RESULTS["samsung_success"],
        "success": True,
        "error": None
    },
    
    "hybrid_failure": {
        "agent_type": "hybrid",
        "result": None,
        "success": False,
        "error": "Multiple stage failures in hybrid processing"
    }
}


# Mock API Responses
MOCK_API_RESPONSES = {
    "inference_success": {
        "status": "success",
        "method": "orchestrator",
        "result": {
            "predicted_brand": "Samsung",
            "confidence": 0.93,
            "processing_time": 0.82,
            "method_used": "orchestrator"
        },
        "timestamp": time.time()
    },
    
    "inference_error": {
        "status": "error",
        "method": "invalid_method",
        "error": "Invalid method 'invalid_method'. Valid methods: orchestrator, ner, rag, llm, hybrid, simple",
        "available_methods": ["orchestrator", "ner", "rag", "llm", "hybrid", "simple"],
        "timestamp": time.time()
    },
    
    "health_check_success": {
        "status": "healthy",
        "service": "inference-api",
        "environment": "test",
        "orchestrator_status": "healthy",
        "available_methods": ["orchestrator", "ner", "rag", "llm", "hybrid", "simple"],
        "individual_agents_count": 6,
        "timestamp": time.time()
    },
    
    "health_check_degraded": {
        "status": "degraded",
        "service": "inference-api",
        "environment": "test",
        "orchestrator_status": "healthy",
        "available_methods": ["orchestrator", "ner", "llm", "simple"],
        "individual_agents_count": 4,
        "timestamp": time.time(),
        "warnings": ["RAG agent unavailable", "Hybrid agent unavailable"]
    }
}


# Mock AWS Service Responses
MOCK_AWS_RESPONSES = {
    "bedrock_success": {
        "body": {
            "completion": "Based on the product name 'Samsung Galaxy S24 Ultra', I can confidently identify this as a Samsung product. The Galaxy series is Samsung's flagship smartphone line.",
            "stop_reason": "end_turn"
        },
        "contentType": "application/json"
    },
    
    "bedrock_error": {
        "Error": {
            "Code": "ValidationException",
            "Message": "The provided model identifier is invalid"
        }
    },
    
    "s3_success": {
        "Body": b'{"product_data": "mock_data"}',
        "ContentType": "application/json"
    },
    
    "s3_error": {
        "Error": {
            "Code": "NoSuchKey",
            "Message": "The specified key does not exist"
        }
    }
}


# Mock Milvus Responses
MOCK_MILVUS_RESPONSES = {
    "search_success": [
        [
            {
                "id": 1,
                "distance": 0.15,  # Lower distance = higher similarity
                "entity": {
                    "product_name": "Samsung Galaxy S24 Ultra 256GB",
                    "brand": "Samsung",
                    "category": "Electronics",
                    "sub_category": "Smartphones"
                }
            },
            {
                "id": 2,
                "distance": 0.22,
                "entity": {
                    "product_name": "Samsung Galaxy S23 Ultra",
                    "brand": "Samsung",
                    "category": "Electronics",
                    "sub_category": "Smartphones"
                }
            }
        ]
    ],
    
    "search_no_results": [[]],
    
    "insert_success": {
        "insert_count": 1,
        "ids": [1]
    },
    
    "query_success": [
        {
            "id": 1,
            "product_name": "Samsung Galaxy S24 Ultra",
            "brand": "Samsung",
            "category": "Electronics"
        }
    ]
}


# Helper functions for accessing mock data
def get_mock_agent_response(agent_type: str, success: bool = True) -> Dict[str, Any]:
    """
    Get a mock agent response for testing.
    
    Args:
        agent_type: Type of agent (ner, rag, llm, hybrid)
        success: Whether to return success or failure response
        
    Returns:
        Mock agent response dictionary
    """
    key = f"{agent_type}_{'success' if success else 'failure'}"
    return MOCK_AGENT_RESPONSES.get(key, MOCK_AGENT_RESPONSES["ner_success"])


def get_mock_result(result_type: str, scenario: str = "samsung_success") -> Any:
    """
    Get a mock result object for testing.
    
    Args:
        result_type: Type of result (ner, rag, llm, hybrid)
        scenario: Specific scenario to return
        
    Returns:
        Mock result object
    """
    result_maps = {
        "ner": MOCK_NER_RESULTS,
        "rag": MOCK_RAG_RESULTS,
        "llm": MOCK_LLM_RESULTS,
        "hybrid": MOCK_HYBRID_RESULTS
    }
    
    result_map = result_maps.get(result_type, MOCK_NER_RESULTS)
    return result_map.get(scenario, list(result_map.values())[0])


def get_mock_api_response(response_type: str) -> Dict[str, Any]:
    """
    Get a mock API response for testing.
    
    Args:
        response_type: Type of API response
        
    Returns:
        Mock API response dictionary
    """
    return MOCK_API_RESPONSES.get(response_type, MOCK_API_RESPONSES["inference_success"])


def get_mock_aws_response(service: str, success: bool = True) -> Dict[str, Any]:
    """
    Get a mock AWS service response for testing.
    
    Args:
        service: AWS service name (bedrock, s3)
        success: Whether to return success or error response
        
    Returns:
        Mock AWS response dictionary
    """
    key = f"{service}_{'success' if success else 'error'}"
    return MOCK_AWS_RESPONSES.get(key, MOCK_AWS_RESPONSES["bedrock_success"])


def get_mock_milvus_response(operation: str, success: bool = True) -> Any:
    """
    Get a mock Milvus response for testing.
    
    Args:
        operation: Milvus operation (search, insert, query)
        success: Whether to return successful response
        
    Returns:
        Mock Milvus response
    """
    if success:
        key = f"{operation}_success"
    else:
        key = f"{operation}_no_results" if operation == "search" else f"{operation}_success"
    
    return MOCK_MILVUS_RESPONSES.get(key, MOCK_MILVUS_RESPONSES["search_success"])


# Response validation helpers
def validate_agent_response_structure(response: Dict[str, Any]) -> bool:
    """
    Validate that an agent response has the expected structure.
    
    Args:
        response: Agent response to validate
        
    Returns:
        True if structure is valid
    """
    required_keys = ["agent_type", "result", "success", "error"]
    return all(key in response for key in required_keys)


def validate_api_response_structure(response: Dict[str, Any]) -> bool:
    """
    Validate that an API response has the expected structure.
    
    Args:
        response: API response to validate
        
    Returns:
        True if structure is valid
    """
    required_keys = ["status", "timestamp"]
    return all(key in response for key in required_keys)


# Mock response generators for dynamic testing
class MockResponseGenerator:
    """
    Generator for creating dynamic mock responses.
    
    Provides methods to generate mock responses with varying
    characteristics for comprehensive testing.
    """
    
    @staticmethod
    def generate_ner_result(brand: str = "TestBrand", confidence: float = 0.8) -> NERResult:
        """Generate a mock NER result with specified parameters."""
        return NERResult(
            entities=[
                EntityResult(
                    entity_type=EntityType.BRAND,
                    text=brand,
                    confidence=confidence,
                    start_pos=0,
                    end_pos=len(brand)
                )
            ],
            confidence=confidence,
            processing_time=0.15,
            model_used="test_ner_model"
        )
    
    @staticmethod
    def generate_rag_result(brand: str = "TestBrand", confidence: float = 0.8) -> RAGResult:
        """Generate a mock RAG result with specified parameters."""
        return RAGResult(
            similar_products=[
                SimilarProduct(
                    product_name=f"{brand} Test Product",
                    brand=brand,
                    category="Test Category",
                    sub_category="Test Sub Category",
                    similarity_score=confidence
                )
            ],
            predicted_brand=brand,
            confidence=confidence,
            processing_time=0.25,
            embedding_model="test_embedding_model"
        )
    
    @staticmethod
    def generate_llm_result(brand: str = "TestBrand", confidence: float = 0.8) -> LLMResult:
        """Generate a mock LLM result with specified parameters."""
        return LLMResult(
            predicted_brand=brand,
            reasoning=f"Test reasoning for {brand} brand prediction",
            confidence=confidence,
            processing_time=0.45,
            model_id="test_llm_model"
        )
    
    @staticmethod
    def generate_agent_response(agent_type: str, success: bool = True, 
                              brand: str = "TestBrand", confidence: float = 0.8) -> Dict[str, Any]:
        """Generate a complete mock agent response."""
        if not success:
            return {
                "agent_type": agent_type,
                "result": None,
                "success": False,
                "error": f"Simulated {agent_type} agent failure"
            }
        
        # Generate appropriate result based on agent type
        if agent_type == "ner":
            result = MockResponseGenerator.generate_ner_result(brand, confidence)
        elif agent_type == "rag":
            result = MockResponseGenerator.generate_rag_result(brand, confidence)
        elif agent_type == "llm":
            result = MockResponseGenerator.generate_llm_result(brand, confidence)
        else:
            result = MockResponseGenerator.generate_ner_result(brand, confidence)
        
        return {
            "agent_type": agent_type,
            "result": result,
            "success": True,
            "error": None
        }