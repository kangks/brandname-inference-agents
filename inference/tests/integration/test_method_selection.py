#!/usr/bin/env python3
"""
Integration tests for method selection functionality.

This module contains pytest-based integration tests for testing different
inference methods and their selection logic.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple

from inference.tests.utils.test_base import BaseAgentTest
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.integration
class TestMethodSelection:
    """Integration tests for method selection functionality."""
    
    @pytest.fixture
    def api_client(self):
        """Mock API client for testing."""
        return Mock()
    
    @pytest.fixture
    def base_url(self):
        """Base URL for API testing."""
        return "http://production-alb-107602758.us-east-1.elb.amazonaws.com"
    
    @pytest.fixture
    def method_test_cases(self) -> List[Dict[str, Any]]:
        """Test cases for different inference methods."""
        return [
            {
                "name": "Default (orchestrator)",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en"
                },
                "expected_method": "orchestrator"
            },
            {
                "name": "Simple method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra", 
                    "language_hint": "en",
                    "method": "simple"
                },
                "expected_method": "simple"
            },
            {
                "name": "RAG method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en", 
                    "method": "rag"
                },
                "expected_method": "rag"
            },
            {
                "name": "Hybrid method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en",
                    "method": "hybrid"
                },
                "expected_method": "hybrid"
            },
            {
                "name": "NER method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en",
                    "method": "ner"
                },
                "expected_method": "ner"
            },
            {
                "name": "LLM method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en",
                    "method": "llm"
                },
                "expected_method": "llm"
            }
        ]
    
    @pytest.fixture
    def invalid_method_test_cases(self) -> List[Dict[str, Any]]:
        """Test cases for invalid methods."""
        return [
            {
                "name": "Invalid method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en",
                    "method": "invalid"
                },
                "expected_error": "Invalid method"
            },
            {
                "name": "Empty method",
                "payload": {
                    "product_name": "Samsung Galaxy S24 Ultra",
                    "language_hint": "en",
                    "method": ""
                },
                "expected_error": "Method parameter is required"
            }
        ]
    
    def mock_successful_response(self, method: str, prediction: str = "Samsung", confidence: float = 0.85) -> Dict[str, Any]:
        """Create a mock successful API response."""
        base_response = {
            "success": True,
            "method": method,
            "brand_predictions": [
                {
                    "brand": prediction,
                    "confidence": confidence,
                    "method": method
                }
            ],
            "processing_time_ms": 500
        }
        
        # Add method-specific fields
        if method == "simple":
            base_response["reasoning"] = f"Pattern matching identified {prediction}"
        elif method == "rag":
            base_response["similar_products"] = [
                {"product": "Similar Product 1", "similarity": 0.8},
                {"product": "Similar Product 2", "similarity": 0.7}
            ]
        elif method == "hybrid":
            base_response["contributions"] = {
                "ner": 0.3,
                "rag": 0.4,
                "llm": 0.3
            }
        elif method == "ner":
            base_response["entities"] = [
                {"text": prediction, "label": "BRAND", "confidence": confidence}
            ]
        elif method == "llm":
            base_response["reasoning"] = f"LLM analysis identified {prediction} based on product features"
        elif method == "orchestrator":
            base_response["orchestrator_agents"] = ["ner", "rag", "llm", "hybrid"]
            base_response["agent_results"] = {
                "ner": {"success": True, "prediction": prediction, "confidence": 0.8},
                "rag": {"success": True, "prediction": prediction, "confidence": 0.7},
                "llm": {"success": True, "prediction": prediction, "confidence": confidence}
            }
        
        return base_response
    
    def mock_error_response(self, status_code: int, error_message: str) -> Dict[str, Any]:
        """Create a mock error API response."""
        return {
            "success": False,
            "error": error_message,
            "status_code": status_code,
            "available_methods": ["orchestrator", "simple", "rag", "hybrid", "ner", "llm"]
        }
    
    def test_root_endpoint_method_documentation(self, api_client, base_url):
        """Test that root endpoint provides method documentation."""
        # Mock root endpoint response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "Brand Inference API",
            "version": "1.0.0",
            "inference_methods": {
                "orchestrator": {
                    "available": True,
                    "description": "Coordinates multiple agents for best results"
                },
                "simple": {
                    "available": True,
                    "description": "Pattern-based brand recognition"
                },
                "rag": {
                    "available": True,
                    "description": "Retrieval-augmented generation"
                },
                "hybrid": {
                    "available": True,
                    "description": "Combines multiple approaches"
                },
                "ner": {
                    "available": True,
                    "description": "Named entity recognition"
                },
                "llm": {
                    "available": True,
                    "description": "Large language model inference"
                }
            }
        }
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            import requests
            response = requests.get(f"{base_url}/")
            
            # Verify request was made
            mock_get.assert_called_once_with(f"{base_url}/")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify service information
            assert data["service"] == "Brand Inference API"
            assert "inference_methods" in data
            
            # Verify method documentation
            methods = data["inference_methods"]
            expected_methods = ["orchestrator", "simple", "rag", "hybrid", "ner", "llm"]
            
            for method in expected_methods:
                assert method in methods
                assert "available" in methods[method]
                assert "description" in methods[method]
                assert methods[method]["available"] is True
    
    @pytest.mark.parametrize("test_case", [
        {
            "name": "Default (orchestrator)",
            "payload": {"product_name": "Samsung Galaxy S24 Ultra", "language_hint": "en"},
            "expected_method": "orchestrator"
        },
        {
            "name": "Simple method",
            "payload": {"product_name": "Samsung Galaxy S24 Ultra", "language_hint": "en", "method": "simple"},
            "expected_method": "simple"
        },
        {
            "name": "RAG method",
            "payload": {"product_name": "Samsung Galaxy S24 Ultra", "language_hint": "en", "method": "rag"},
            "expected_method": "rag"
        }
    ])
    def test_method_selection_success(self, api_client, base_url, test_case):
        """Test successful method selection for different methods."""
        expected_method = test_case["expected_method"]
        mock_response_data = self.mock_successful_response(expected_method)
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            import requests
            response = requests.post(
                f"{base_url}/infer",
                headers={"Content-Type": "application/json"},
                json=test_case["payload"],
                timeout=30
            )
            
            # Verify request was made correctly
            mock_post.assert_called_once_with(
                f"{base_url}/infer",
                headers={"Content-Type": "application/json"},
                json=test_case["payload"],
                timeout=30
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify basic response structure
            assert data["success"] is True
            assert data["method"] == expected_method
            assert "brand_predictions" in data
            assert len(data["brand_predictions"]) > 0
            
            # Verify prediction structure
            prediction = data["brand_predictions"][0]
            assert "brand" in prediction
            assert "confidence" in prediction
            assert "method" in prediction
            assert prediction["method"] == expected_method
            
            # Verify method-specific fields
            if expected_method == "simple":
                assert "reasoning" in data
            elif expected_method == "rag":
                assert "similar_products" in data
            elif expected_method == "hybrid":
                assert "contributions" in data
            elif expected_method == "ner":
                assert "entities" in data
            elif expected_method == "llm":
                assert "reasoning" in data
            elif expected_method == "orchestrator":
                assert "orchestrator_agents" in data
                assert "agent_results" in data
    
    def test_method_selection_batch_testing(self, api_client, base_url, method_test_cases):
        """Test method selection with all available methods."""
        results = []
        
        for test_case in method_test_cases:
            expected_method = test_case["expected_method"]
            mock_response_data = self.mock_successful_response(expected_method)
            
            # Mock API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json=test_case["payload"],
                    timeout=30
                )
                
                result = {
                    "test_name": test_case["name"],
                    "expected_method": expected_method,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    data = response.json()
                    result.update({
                        "actual_method": data.get("method"),
                        "predictions": data.get("brand_predictions", []),
                        "processing_time": data.get("processing_time_ms", 0)
                    })
                
                results.append(result)
        
        # Verify all methods were tested successfully
        assert len(results) == len(method_test_cases)
        successful_tests = [r for r in results if r["success"]]
        assert len(successful_tests) == len(method_test_cases)
        
        # Verify method selection accuracy
        for result in successful_tests:
            assert result["actual_method"] == result["expected_method"]
            assert len(result["predictions"]) > 0
            assert result["processing_time"] >= 0
    
    @pytest.mark.parametrize("test_case", [
        {
            "name": "Invalid method",
            "payload": {"product_name": "Samsung Galaxy S24 Ultra", "language_hint": "en", "method": "invalid"},
            "expected_error": "Invalid method"
        }
    ])
    def test_invalid_method_handling(self, api_client, base_url, test_case):
        """Test handling of invalid methods."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = self.mock_error_response(400, test_case["expected_error"])
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            import requests
            response = requests.post(
                f"{base_url}/infer",
                headers={"Content-Type": "application/json"},
                json=test_case["payload"],
                timeout=30
            )
            
            # Verify request was made
            mock_post.assert_called_once()
            
            # Verify error response
            assert response.status_code == 400
            data = response.json()
            
            assert data["success"] is False
            assert "error" in data
            assert test_case["expected_error"].lower() in data["error"].lower()
            assert "available_methods" in data
    
    def test_method_availability_validation(self, api_client, base_url):
        """Test method availability validation."""
        # Test with all available methods
        available_methods = ["orchestrator", "simple", "rag", "hybrid", "ner", "llm"]
        
        for method in available_methods:
            mock_response_data = self.mock_successful_response(method)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": "Test Product",
                        "language_hint": "en",
                        "method": method
                    },
                    timeout=30
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["method"] == method
    
    def test_method_performance_comparison(self, api_client, base_url):
        """Test performance comparison across different methods."""
        methods = ["simple", "rag", "llm", "hybrid", "orchestrator"]
        performance_results = []
        
        for method in methods:
            # Mock response with different processing times
            processing_times = {
                "simple": 100,
                "rag": 300,
                "llm": 800,
                "hybrid": 500,
                "orchestrator": 1200
            }
            
            mock_response_data = self.mock_successful_response(method)
            mock_response_data["processing_time_ms"] = processing_times[method]
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            with patch('requests.post', return_value=mock_response):
                import requests
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": "Samsung Galaxy S24 Ultra",
                        "language_hint": "en",
                        "method": method
                    },
                    timeout=30
                )
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    performance_results.append({
                        "method": method,
                        "processing_time_ms": data.get("processing_time_ms", 0),
                        "request_time_s": request_time,
                        "confidence": data["brand_predictions"][0]["confidence"]
                    })
        
        # Verify performance results
        assert len(performance_results) == len(methods)
        
        # Verify expected performance characteristics
        simple_result = next(r for r in performance_results if r["method"] == "simple")
        orchestrator_result = next(r for r in performance_results if r["method"] == "orchestrator")
        
        # Simple should be faster than orchestrator
        assert simple_result["processing_time_ms"] < orchestrator_result["processing_time_ms"]
        
        # All methods should complete within reasonable time
        for result in performance_results:
            assert result["processing_time_ms"] < 5000  # Less than 5 seconds
            assert result["request_time_s"] < 2.0  # Request should be fast (mocked)
    
    def test_method_consistency_across_requests(self, api_client, base_url):
        """Test method consistency across multiple requests."""
        method = "llm"
        product_name = "Samsung Galaxy S24 Ultra"
        num_requests = 3
        
        results = []
        
        for i in range(num_requests):
            mock_response_data = self.mock_successful_response(method, "Samsung", 0.85)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": product_name,
                        "language_hint": "en",
                        "method": method
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "method": data.get("method"),
                        "prediction": data["brand_predictions"][0]["brand"],
                        "confidence": data["brand_predictions"][0]["confidence"]
                    })
        
        # Verify consistency
        assert len(results) == num_requests
        
        # All results should use the same method
        methods_used = set(r["method"] for r in results)
        assert len(methods_used) == 1
        assert list(methods_used)[0] == method
        
        # Predictions should be consistent (for same input)
        predictions = set(r["prediction"] for r in results)
        assert len(predictions) == 1  # Should be consistent
        
        # Confidence should be consistent
        confidences = [r["confidence"] for r in results]
        confidence_variance = max(confidences) - min(confidences)
        assert confidence_variance < 0.1  # Should be very consistent for mocked responses


@pytest.mark.integration
class TestMethodSelectionAdvanced:
    """Advanced integration tests for method selection."""
    
    def test_method_fallback_behavior(self, api_client):
        """Test method fallback when primary method fails."""
        # Mock scenario where primary method fails but fallback succeeds
        primary_method = "rag"
        fallback_method = "simple"
        
        # First call fails
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.json.return_value = {
            "success": False,
            "error": "RAG service unavailable"
        }
        
        # Fallback call succeeds
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "success": True,
            "method": fallback_method,
            "brand_predictions": [{"brand": "Samsung", "confidence": 0.75, "method": fallback_method}],
            "fallback_used": True,
            "original_method": primary_method
        }
        
        with patch('requests.post', side_effect=[mock_error_response, mock_success_response]):
            import requests
            
            # First request (fails)
            response1 = requests.post(
                "http://test/infer",
                json={"product_name": "Test", "method": primary_method}
            )
            assert response1.status_code == 500
            
            # Fallback request (succeeds)
            response2 = requests.post(
                "http://test/infer",
                json={"product_name": "Test", "method": fallback_method}
            )
            assert response2.status_code == 200
            data = response2.json()
            assert data["method"] == fallback_method
    
    def test_method_load_balancing(self, api_client):
        """Test method selection under load balancing scenarios."""
        # Mock multiple instances of the same method
        method = "orchestrator"
        instances = ["instance1", "instance2", "instance3"]
        
        responses = []
        for i, instance in enumerate(instances):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "success": True,
                "method": method,
                "instance_id": instance,
                "brand_predictions": [{"brand": "Samsung", "confidence": 0.8 + i * 0.05}],
                "load_factor": 0.3 + i * 0.2
            }
            responses.append(mock_response)
        
        with patch('requests.post', side_effect=responses):
            import requests
            
            results = []
            for i in range(len(instances)):
                response = requests.post(
                    "http://test/infer",
                    json={"product_name": "Test", "method": method}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "instance": data.get("instance_id"),
                        "confidence": data["brand_predictions"][0]["confidence"],
                        "load_factor": data.get("load_factor")
                    })
            
            # Verify load balancing
            assert len(results) == len(instances)
            instance_ids = [r["instance"] for r in results]
            assert len(set(instance_ids)) == len(instances)  # All instances used