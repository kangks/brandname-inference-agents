"""
Assertion helpers for inference system testing.

This module provides specialized assertion functions for testing
inference results, agent behavior, and system responses.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
import pytest

from inference.src.models.data_models import (
    ProductInput,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    EntityResult,
    EntityType,
    SimilarProduct
)


class LogCapture:
    """Helper class for capturing log messages during tests."""
    
    def __init__(self):
        self.logs = []
        self.handler = None
        
    def get_logs(self):
        """Get captured log messages."""
        return self.logs
        
    def clear_logs(self):
        """Clear captured log messages."""
        self.logs.clear()


class AssertionHelpers:
    """Main assertion helpers class with utility methods."""
    
    @contextmanager
    def capture_logs(self, logger_name: str = None, level: int = logging.INFO):
        """
        Context manager to capture log messages during test execution.
        
        Args:
            logger_name: Name of logger to capture (None for root logger)
            level: Minimum log level to capture
            
        Returns:
            LogCapture instance
        """
        log_capture = LogCapture()
        
        class TestLogHandler(logging.Handler):
            def emit(self, record):
                log_capture.logs.append(self.format(record))
        
        # Setup handler
        handler = TestLogHandler()
        handler.setLevel(level)
        
        # Get logger
        logger = logging.getLogger(logger_name)
        original_level = logger.level
        
        # Add handler and adjust level
        logger.addHandler(handler)
        if logger.level > level:
            logger.setLevel(level)
        
        try:
            yield log_capture
        finally:
            # Cleanup
            logger.removeHandler(handler)
            logger.setLevel(original_level)
    
    def assert_valid_brand_extraction(self, result: Dict[str, Any]):
        """
        Assert that a brand extraction result is valid.
        
        Args:
            result: Brand extraction result to validate
        """
        assert isinstance(result, dict), "Result must be a dictionary"
        assert "brands" in result, "Result must contain 'brands' field"
        
        brands = result["brands"]
        assert isinstance(brands, list), "Brands must be a list"
        
        # Validate each brand entry
        for brand in brands:
            if isinstance(brand, str):
                assert len(brand.strip()) > 0, "Brand name cannot be empty"
            elif isinstance(brand, dict):
                assert "name" in brand or "text" in brand, "Brand dict must have name or text"
                brand_name = brand.get("name") or brand.get("text")
                assert len(brand_name.strip()) > 0, "Brand name cannot be empty"


class InferenceAssertions:
    """
    Specialized assertions for inference system testing.
    
    Provides domain-specific assertion methods for validating
    inference results, agent responses, and system behavior.
    """
    
    @staticmethod
    def assert_valid_agent_result(result: Dict[str, Any], agent_type: str = None):
        """
        Assert that an agent result has the expected structure.
        
        Args:
            result: Agent result dictionary
            agent_type: Expected agent type (optional)
        """
        # Check basic structure
        assert isinstance(result, dict), "Result must be a dictionary"
        
        required_keys = ["agent_type", "result", "success", "error"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Check agent type if specified
        if agent_type:
            assert result["agent_type"] == agent_type, \
                f"Expected agent_type '{agent_type}', got '{result['agent_type']}'"
        
        # Check success/error consistency
        if result["success"]:
            assert result["result"] is not None, "Successful result should have data"
            assert result["error"] is None, "Successful result should not have error"
        else:
            assert result["error"] is not None, "Failed result should have error message"
    
    @staticmethod
    def assert_valid_ner_result(ner_result: NERResult):
        """
        Assert that a NER result is valid.
        
        Args:
            ner_result: NER result to validate
        """
        assert isinstance(ner_result, NERResult), "Must be NERResult instance"
        
        # Check confidence range
        assert 0.0 <= ner_result.confidence <= 1.0, \
            f"Confidence {ner_result.confidence} not in range [0.0, 1.0]"
        
        # Check processing time
        assert ner_result.processing_time >= 0, \
            f"Processing time {ner_result.processing_time} cannot be negative"
        
        # Check entities
        assert isinstance(ner_result.entities, list), "Entities must be a list"
        
        for entity in ner_result.entities:
            InferenceAssertions.assert_valid_entity_result(entity)
        
        # Check model used
        assert isinstance(ner_result.model_used, str), "Model used must be string"
        assert len(ner_result.model_used) > 0, "Model used cannot be empty"
    
    @staticmethod
    def assert_valid_entity_result(entity: EntityResult):
        """
        Assert that an entity result is valid.
        
        Args:
            entity: Entity result to validate
        """
        assert isinstance(entity, EntityResult), "Must be EntityResult instance"
        
        # Check entity type
        assert isinstance(entity.entity_type, EntityType), "Must be valid EntityType"
        
        # Check text
        assert isinstance(entity.text, str), "Entity text must be string"
        assert len(entity.text) > 0, "Entity text cannot be empty"
        
        # Check confidence
        assert 0.0 <= entity.confidence <= 1.0, \
            f"Entity confidence {entity.confidence} not in range [0.0, 1.0]"
        
        # Check positions
        assert entity.start_pos >= 0, "Start position cannot be negative"
        assert entity.end_pos >= entity.start_pos, \
            "End position must be >= start position"
    
    @staticmethod
    def assert_valid_rag_result(rag_result: RAGResult):
        """
        Assert that a RAG result is valid.
        
        Args:
            rag_result: RAG result to validate
        """
        assert isinstance(rag_result, RAGResult), "Must be RAGResult instance"
        
        # Check confidence range
        assert 0.0 <= rag_result.confidence <= 1.0, \
            f"Confidence {rag_result.confidence} not in range [0.0, 1.0]"
        
        # Check processing time
        assert rag_result.processing_time >= 0, \
            f"Processing time {rag_result.processing_time} cannot be negative"
        
        # Check similar products
        assert isinstance(rag_result.similar_products, list), \
            "Similar products must be a list"
        
        for product in rag_result.similar_products:
            InferenceAssertions.assert_valid_similar_product(product)
        
        # Check predicted brand
        if rag_result.predicted_brand:
            assert isinstance(rag_result.predicted_brand, str), \
                "Predicted brand must be string"
        
        # Check embedding model
        assert isinstance(rag_result.embedding_model, str), \
            "Embedding model must be string"
    
    @staticmethod
    def assert_valid_similar_product(product: SimilarProduct):
        """
        Assert that a similar product is valid.
        
        Args:
            product: Similar product to validate
        """
        assert isinstance(product, SimilarProduct), "Must be SimilarProduct instance"
        
        # Check required string fields
        string_fields = ["product_name", "brand", "category", "sub_category"]
        for field in string_fields:
            value = getattr(product, field)
            assert isinstance(value, str), f"{field} must be string"
            assert len(value) > 0, f"{field} cannot be empty"
        
        # Check similarity score
        assert 0.0 <= product.similarity_score <= 1.0, \
            f"Similarity score {product.similarity_score} not in range [0.0, 1.0]"
    
    @staticmethod
    def assert_valid_llm_result(llm_result: LLMResult):
        """
        Assert that an LLM result is valid.
        
        Args:
            llm_result: LLM result to validate
        """
        assert isinstance(llm_result, LLMResult), "Must be LLMResult instance"
        
        # Check confidence range
        assert 0.0 <= llm_result.confidence <= 1.0, \
            f"Confidence {llm_result.confidence} not in range [0.0, 1.0]"
        
        # Check processing time
        assert llm_result.processing_time >= 0, \
            f"Processing time {llm_result.processing_time} cannot be negative"
        
        # Check predicted brand
        assert isinstance(llm_result.predicted_brand, str), \
            "Predicted brand must be string"
        
        # Check reasoning
        assert isinstance(llm_result.reasoning, str), \
            "Reasoning must be string"
        
        # Check model ID
        assert isinstance(llm_result.model_id, str), \
            "Model ID must be string"
        assert len(llm_result.model_id) > 0, "Model ID cannot be empty"
    
    @staticmethod
    def assert_valid_hybrid_result(hybrid_result: HybridResult):
        """
        Assert that a Hybrid result is valid.
        
        Args:
            hybrid_result: Hybrid result to validate
        """
        assert isinstance(hybrid_result, HybridResult), "Must be HybridResult instance"
        
        # Check confidence range
        assert 0.0 <= hybrid_result.confidence <= 1.0, \
            f"Confidence {hybrid_result.confidence} not in range [0.0, 1.0]"
        
        # Check processing time
        assert hybrid_result.processing_time >= 0, \
            f"Processing time {hybrid_result.processing_time} cannot be negative"
        
        # Check final prediction
        assert isinstance(hybrid_result.final_prediction, str), \
            "Final prediction must be string"
        
        # Check stage results
        assert isinstance(hybrid_result.stage_results, dict), \
            "Stage results must be dictionary"
        
        # Check stages used
        assert isinstance(hybrid_result.stages_used, list), \
            "Stages used must be list"
        
        # Validate that all stages used have results
        for stage in hybrid_result.stages_used:
            assert stage in hybrid_result.stage_results, \
                f"Stage '{stage}' used but no result found"
    
    @staticmethod
    def assert_confidence_in_range(result: Union[NERResult, RAGResult, LLMResult, HybridResult],
                                 min_confidence: float = 0.0, 
                                 max_confidence: float = 1.0):
        """
        Assert that result confidence is within specified range.
        
        Args:
            result: Result with confidence attribute
            min_confidence: Minimum acceptable confidence
            max_confidence: Maximum acceptable confidence
        """
        confidence = result.confidence
        assert min_confidence <= confidence <= max_confidence, \
            f"Confidence {confidence} not in range [{min_confidence}, {max_confidence}]"
    
    @staticmethod
    def assert_processing_time_reasonable(result: Union[NERResult, RAGResult, LLMResult, HybridResult],
                                        max_time: float = 30.0):
        """
        Assert that processing time is reasonable.
        
        Args:
            result: Result with processing_time attribute
            max_time: Maximum acceptable processing time in seconds
        """
        processing_time = result.processing_time
        assert 0 <= processing_time <= max_time, \
            f"Processing time {processing_time}s not reasonable (max: {max_time}s)"
    
    @staticmethod
    def assert_brand_prediction_consistent(results: List[Union[NERResult, RAGResult, LLMResult]],
                                         tolerance: float = 0.8):
        """
        Assert that brand predictions are consistent across multiple results.
        
        Args:
            results: List of results to check for consistency
            tolerance: Minimum fraction of results that should agree
        """
        if len(results) < 2:
            return  # Cannot check consistency with fewer than 2 results
        
        # Extract brand predictions
        brands = []
        for result in results:
            if isinstance(result, NERResult):
                brand_entities = result.get_brands()
                if brand_entities:
                    brands.append(brand_entities[0].text)
            elif isinstance(result, (RAGResult, LLMResult)):
                if result.predicted_brand:
                    brands.append(result.predicted_brand)
        
        if not brands:
            return  # No brands to check
        
        # Find most common brand
        brand_counts = {}
        for brand in brands:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        most_common_brand = max(brand_counts.items(), key=lambda x: x[1])
        consistency_ratio = most_common_brand[1] / len(brands)
        
        assert consistency_ratio >= tolerance, \
            f"Brand prediction consistency {consistency_ratio:.2f} below tolerance {tolerance}"


class PerformanceAssertions:
    """
    Assertions for performance testing.
    
    Provides methods to assert performance characteristics
    and timing requirements.
    """
    
    @staticmethod
    def assert_response_time_within_limit(start_time: float, end_time: float, 
                                        max_time: float):
        """
        Assert that response time is within acceptable limits.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            max_time: Maximum acceptable time in seconds
        """
        elapsed_time = end_time - start_time
        assert elapsed_time <= max_time, \
            f"Response time {elapsed_time:.3f}s exceeds limit {max_time}s"
    
    @staticmethod
    def assert_throughput_meets_requirement(num_requests: int, total_time: float,
                                          min_throughput: float):
        """
        Assert that throughput meets minimum requirements.
        
        Args:
            num_requests: Number of requests processed
            total_time: Total time taken in seconds
            min_throughput: Minimum requests per second
        """
        actual_throughput = num_requests / total_time
        assert actual_throughput >= min_throughput, \
            f"Throughput {actual_throughput:.2f} req/s below minimum {min_throughput} req/s"
    
    @staticmethod
    def assert_memory_usage_reasonable(memory_usage_mb: float, max_memory_mb: float):
        """
        Assert that memory usage is within reasonable limits.
        
        Args:
            memory_usage_mb: Actual memory usage in MB
            max_memory_mb: Maximum acceptable memory usage in MB
        """
        assert memory_usage_mb <= max_memory_mb, \
            f"Memory usage {memory_usage_mb:.1f}MB exceeds limit {max_memory_mb}MB"


class APIAssertions:
    """
    Assertions for API response testing.
    
    Provides methods to validate API responses and HTTP behavior.
    """
    
    @staticmethod
    def assert_http_status_code(response, expected_status: int):
        """
        Assert HTTP status code matches expected value.
        
        Args:
            response: HTTP response object
            expected_status: Expected status code
        """
        actual_status = getattr(response, 'status_code', None)
        assert actual_status == expected_status, \
            f"Expected status {expected_status}, got {actual_status}"
    
    @staticmethod
    def assert_response_has_json(response):
        """
        Assert that response contains valid JSON.
        
        Args:
            response: HTTP response object
        """
        try:
            json_data = response.json()
            assert isinstance(json_data, dict), "Response JSON must be a dictionary"
        except Exception as e:
            pytest.fail(f"Response does not contain valid JSON: {e}")
    
    @staticmethod
    def assert_api_response_structure(response_data: Dict[str, Any], 
                                    required_fields: List[str]):
        """
        Assert that API response has required structure.
        
        Args:
            response_data: Response data dictionary
            required_fields: List of required field names
        """
        assert isinstance(response_data, dict), "Response must be a dictionary"
        
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
    
    @staticmethod
    def assert_error_response_format(response_data: Dict[str, Any]):
        """
        Assert that error response has proper format.
        
        Args:
            response_data: Error response data
        """
        required_fields = ["status", "error", "method"]
        APIAssertions.assert_api_response_structure(response_data, required_fields)
        
        assert response_data["status"] == "error", \
            f"Expected status 'error', got '{response_data['status']}'"
        
        assert isinstance(response_data["error"], str), \
            "Error message must be string"
        
        assert len(response_data["error"]) > 0, \
            "Error message cannot be empty"
    
    @staticmethod
    def assert_success_response_format(response_data: Dict[str, Any]):
        """
        Assert that success response has proper format.
        
        Args:
            response_data: Success response data
        """
        required_fields = ["status", "result", "method"]
        APIAssertions.assert_api_response_structure(response_data, required_fields)
        
        assert response_data["status"] == "success", \
            f"Expected status 'success', got '{response_data['status']}'"
        
        assert response_data["result"] is not None, \
            "Success response must have result data"


class AsyncAssertions:
    """
    Assertions for async operation testing.
    
    Provides methods to test async behavior and timing.
    """
    
    @staticmethod
    async def assert_async_completes_within(coro, max_time: float):
        """
        Assert that async operation completes within time limit.
        
        Args:
            coro: Coroutine to test
            max_time: Maximum time in seconds
        """
        start_time = time.time()
        await coro
        elapsed_time = time.time() - start_time
        
        assert elapsed_time <= max_time, \
            f"Async operation took {elapsed_time:.3f}s, expected <= {max_time}s"
    
    @staticmethod
    async def assert_async_raises(expected_exception, coro):
        """
        Assert that async operation raises expected exception.
        
        Args:
            expected_exception: Expected exception type
            coro: Coroutine that should raise exception
        """
        with pytest.raises(expected_exception):
            await coro
    
    @staticmethod
    async def assert_async_timeout(coro, timeout: float):
        """
        Assert that async operation times out as expected.
        
        Args:
            coro: Coroutine that should timeout
            timeout: Timeout duration in seconds
        """
        import asyncio
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coro, timeout=timeout)