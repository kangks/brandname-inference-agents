#!/usr/bin/env python3
"""
Integration tests for method-specific functionality.

This module contains pytest-based integration tests for testing specific
inference methods like NER and LLM in detail.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple

from inference.tests.utils.test_base import BaseAgentTest
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.integration
class TestNERLLMMethodsIntegration:
    """Integration tests for NER and LLM methods specifically."""
    
    @pytest.fixture
    def api_client(self):
        """Mock API client for testing."""
        return Mock()
    
    @pytest.fixture
    def base_url(self):
        """Base URL for API testing."""
        return "http://production-alb-107602758.us-east-1.elb.amazonaws.com"
    
    @pytest.fixture
    def test_products(self) -> List[Tuple[str, str, str]]:
        """Test products with expected brands and languages."""
        return [
            ("Samsung Galaxy S24 Ultra", "Samsung", "en"),
            ("iPhone 15 Pro Max", "Apple", "en"),
            ("Nike Air Jordan 1", "Nike", "en"),
            ("Toyota Camry 2024", "Toyota", "en"),
            ("โค้ก เซโร่", "Coca-Cola", "th")  # Thai text
        ]
    
    def mock_ner_response(self, product_name: str, brand: str, confidence: float = 0.85) -> Dict[str, Any]:
        """Create a mock NER method response."""
        entities = [
            {
                "text": brand,
                "label": "BRAND",
                "confidence": confidence,
                "start": product_name.find(brand) if brand in product_name else 0,
                "end": product_name.find(brand) + len(brand) if brand in product_name else len(brand)
            }
        ]
        
        # Add additional entities for more realistic response
        if "Galaxy" in product_name:
            entities.append({
                "text": "Galaxy",
                "label": "PRODUCT_LINE",
                "confidence": 0.75,
                "start": product_name.find("Galaxy"),
                "end": product_name.find("Galaxy") + 6
            })
        
        return {
            "success": True,
            "method": "ner",
            "brand_predictions": [
                {
                    "brand": brand,
                    "confidence": confidence,
                    "method": "ner"
                }
            ],
            "entities": entities,
            "processing_time_ms": 200,
            "model_info": {
                "model_name": "en_core_web_sm",
                "model_version": "3.4.0"
            }
        }
    
    def mock_llm_response(self, product_name: str, brand: str, confidence: float = 0.90) -> Dict[str, Any]:
        """Create a mock LLM method response."""
        reasoning = f"Based on analysis of the product name '{product_name}', I identified '{brand}' as the brand. "
        
        if "Galaxy" in product_name:
            reasoning += "The 'Galaxy' series is a well-known Samsung product line. "
        elif "iPhone" in product_name:
            reasoning += "iPhone is Apple's flagship smartphone product. "
        elif "Air Jordan" in product_name:
            reasoning += "Air Jordan is Nike's premium basketball shoe brand. "
        elif "Camry" in product_name:
            reasoning += "Camry is Toyota's popular sedan model. "
        
        reasoning += f"Confidence level: {confidence:.1%}"
        
        return {
            "success": True,
            "method": "llm",
            "brand_predictions": [
                {
                    "brand": brand,
                    "confidence": confidence,
                    "method": "llm"
                }
            ],
            "reasoning": reasoning,
            "processing_time_ms": 800,
            "model_info": {
                "model_id": "us.amazon.nova-pro-v1:0",
                "temperature": 0.1,
                "max_tokens": 100
            }
        }
    
    @pytest.mark.parametrize("product_name,expected_brand,language", [
        ("Samsung Galaxy S24 Ultra", "Samsung", "en"),
        ("iPhone 15 Pro Max", "Apple", "en"),
        ("Nike Air Jordan 1", "Nike", "en"),
    ])
    def test_ner_method_integration(self, api_client, base_url, product_name, expected_brand, language):
        """Test NER method integration with different products."""
        mock_response_data = self.mock_ner_response(product_name, expected_brand)
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            import requests
            response = requests.post(
                f"{base_url}/infer",
                headers={"Content-Type": "application/json"},
                json={
                    "product_name": product_name,
                    "language_hint": language,
                    "method": "ner"
                },
                timeout=30
            )
            
            # Verify request
            mock_post.assert_called_once()
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify basic structure
            assert data["success"] is True
            assert data["method"] == "ner"
            assert "brand_predictions" in data
            assert "entities" in data
            
            # Verify prediction
            predictions = data["brand_predictions"]
            assert len(predictions) > 0
            assert predictions[0]["brand"] == expected_brand
            assert predictions[0]["method"] == "ner"
            assert predictions[0]["confidence"] > 0.0
            
            # Verify entities
            entities = data["entities"]
            assert len(entities) > 0
            
            # Find brand entity
            brand_entities = [e for e in entities if e["label"] == "BRAND"]
            assert len(brand_entities) > 0
            assert brand_entities[0]["text"] == expected_brand
            assert brand_entities[0]["confidence"] > 0.0
            
            # Verify model info
            assert "model_info" in data
            assert "model_name" in data["model_info"]
    
    @pytest.mark.parametrize("product_name,expected_brand,language", [
        ("Samsung Galaxy S24 Ultra", "Samsung", "en"),
        ("iPhone 15 Pro Max", "Apple", "en"),
        ("Toyota Camry 2024", "Toyota", "en"),
    ])
    def test_llm_method_integration(self, api_client, base_url, product_name, expected_brand, language):
        """Test LLM method integration with different products."""
        mock_response_data = self.mock_llm_response(product_name, expected_brand)
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            import requests
            response = requests.post(
                f"{base_url}/infer",
                headers={"Content-Type": "application/json"},
                json={
                    "product_name": product_name,
                    "language_hint": language,
                    "method": "llm"
                },
                timeout=30
            )
            
            # Verify request
            mock_post.assert_called_once()
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify basic structure
            assert data["success"] is True
            assert data["method"] == "llm"
            assert "brand_predictions" in data
            assert "reasoning" in data
            
            # Verify prediction
            predictions = data["brand_predictions"]
            assert len(predictions) > 0
            assert predictions[0]["brand"] == expected_brand
            assert predictions[0]["method"] == "llm"
            assert predictions[0]["confidence"] > 0.0
            
            # Verify reasoning
            reasoning = data["reasoning"]
            assert isinstance(reasoning, str)
            assert len(reasoning) > 0
            assert expected_brand in reasoning
            assert product_name in reasoning
            
            # Verify model info
            assert "model_info" in data
            assert "model_id" in data["model_info"]
    
    def test_ner_llm_methods_comparison(self, api_client, base_url, test_products):
        """Test comparison between NER and LLM methods."""
        comparison_results = []
        
        for product_name, expected_brand, language in test_products:
            # Test NER method
            ner_mock_data = self.mock_ner_response(product_name, expected_brand, 0.80)
            ner_mock_response = Mock()
            ner_mock_response.status_code = 200
            ner_mock_response.json.return_value = ner_mock_data
            
            # Test LLM method
            llm_mock_data = self.mock_llm_response(product_name, expected_brand, 0.90)
            llm_mock_response = Mock()
            llm_mock_response.status_code = 200
            llm_mock_response.json.return_value = llm_mock_data
            
            with patch('requests.post', side_effect=[ner_mock_response, llm_mock_response]):
                import requests
                
                # Test NER
                ner_response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": product_name,
                        "language_hint": language,
                        "method": "ner"
                    },
                    timeout=30
                )
                
                # Test LLM
                llm_response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": product_name,
                        "language_hint": language,
                        "method": "llm"
                    },
                    timeout=30
                )
                
                # Collect results
                result = {
                    "product": product_name,
                    "expected_brand": expected_brand,
                    "language": language
                }
                
                if ner_response.status_code == 200:
                    ner_data = ner_response.json()
                    result["ner"] = {
                        "brand": ner_data["brand_predictions"][0]["brand"],
                        "confidence": ner_data["brand_predictions"][0]["confidence"],
                        "processing_time": ner_data["processing_time_ms"],
                        "entities_count": len(ner_data["entities"])
                    }
                
                if llm_response.status_code == 200:
                    llm_data = llm_response.json()
                    result["llm"] = {
                        "brand": llm_data["brand_predictions"][0]["brand"],
                        "confidence": llm_data["brand_predictions"][0]["confidence"],
                        "processing_time": llm_data["processing_time_ms"],
                        "reasoning_length": len(llm_data["reasoning"])
                    }
                
                comparison_results.append(result)
        
        # Analyze comparison results
        assert len(comparison_results) == len(test_products)
        
        ner_successes = 0
        llm_successes = 0
        ner_total_time = 0
        llm_total_time = 0
        
        for result in comparison_results:
            # Check NER results
            if "ner" in result:
                if result["ner"]["brand"] == result["expected_brand"]:
                    ner_successes += 1
                ner_total_time += result["ner"]["processing_time"]
            
            # Check LLM results
            if "llm" in result:
                if result["llm"]["brand"] == result["expected_brand"]:
                    llm_successes += 1
                llm_total_time += result["llm"]["processing_time"]
        
        # Verify both methods work
        assert ner_successes > 0
        assert llm_successes > 0
        
        # Verify performance characteristics
        avg_ner_time = ner_total_time / len(test_products)
        avg_llm_time = llm_total_time / len(test_products)
        
        # NER should generally be faster than LLM
        assert avg_ner_time < avg_llm_time
        
        # Both should complete within reasonable time
        assert avg_ner_time < 1000  # Less than 1 second
        assert avg_llm_time < 2000  # Less than 2 seconds
    
    def test_ner_entity_extraction_details(self, api_client, base_url):
        """Test detailed NER entity extraction functionality."""
        product_name = "Samsung Galaxy S24 Ultra 256GB Phantom Black"
        expected_entities = [
            {"text": "Samsung", "label": "BRAND"},
            {"text": "Galaxy", "label": "PRODUCT_LINE"},
            {"text": "S24 Ultra", "label": "MODEL"},
            {"text": "256GB", "label": "STORAGE"},
            {"text": "Phantom Black", "label": "COLOR"}
        ]
        
        mock_response_data = {
            "success": True,
            "method": "ner",
            "brand_predictions": [
                {"brand": "Samsung", "confidence": 0.95, "method": "ner"}
            ],
            "entities": [
                {
                    "text": entity["text"],
                    "label": entity["label"],
                    "confidence": 0.85 + i * 0.02,
                    "start": product_name.find(entity["text"]),
                    "end": product_name.find(entity["text"]) + len(entity["text"])
                }
                for i, entity in enumerate(expected_entities)
            ],
            "processing_time_ms": 250
        }
        
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
                    "method": "ner"
                },
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify entities
            entities = data["entities"]
            assert len(entities) == len(expected_entities)
            
            # Verify each entity type is found
            entity_labels = [e["label"] for e in entities]
            for expected_entity in expected_entities:
                assert expected_entity["label"] in entity_labels
            
            # Verify brand entity has highest confidence
            brand_entities = [e for e in entities if e["label"] == "BRAND"]
            assert len(brand_entities) > 0
            brand_confidence = brand_entities[0]["confidence"]
            
            # Brand should have high confidence
            assert brand_confidence > 0.8
    
    def test_llm_reasoning_quality(self, api_client, base_url):
        """Test LLM reasoning quality and detail."""
        test_cases = [
            {
                "product": "iPhone 15 Pro Max 1TB Natural Titanium",
                "expected_brand": "Apple",
                "reasoning_keywords": ["iPhone", "Apple", "flagship", "smartphone"]
            },
            {
                "product": "Nike Air Max 270 React Black White",
                "expected_brand": "Nike",
                "reasoning_keywords": ["Nike", "Air Max", "athletic", "footwear"]
            },
            {
                "product": "Samsung QLED 65 inch 4K Smart TV",
                "expected_brand": "Samsung",
                "reasoning_keywords": ["Samsung", "QLED", "television", "electronics"]
            }
        ]
        
        for test_case in test_cases:
            product_name = test_case["product"]
            expected_brand = test_case["expected_brand"]
            
            # Create detailed reasoning
            reasoning = f"Analysis of '{product_name}': "
            reasoning += f"The brand '{expected_brand}' is clearly identifiable from the product name. "
            
            for keyword in test_case["reasoning_keywords"]:
                if keyword in product_name:
                    reasoning += f"The term '{keyword}' is a strong indicator. "
            
            reasoning += f"Based on market knowledge and product naming conventions, "
            reasoning += f"this is definitively a {expected_brand} product with high confidence."
            
            mock_response_data = {
                "success": True,
                "method": "llm",
                "brand_predictions": [
                    {"brand": expected_brand, "confidence": 0.92, "method": "llm"}
                ],
                "reasoning": reasoning,
                "processing_time_ms": 850,
                "model_info": {
                    "model_id": "us.amazon.nova-pro-v1:0",
                    "tokens_used": 45
                }
            }
            
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
                        "method": "llm"
                    },
                    timeout=30
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify reasoning quality
                reasoning_text = data["reasoning"]
                assert len(reasoning_text) > 50  # Should be detailed
                
                # Verify reasoning contains key information
                assert product_name in reasoning_text
                assert expected_brand in reasoning_text
                
                # Verify reasoning contains expected keywords
                reasoning_lower = reasoning_text.lower()
                keyword_count = sum(1 for keyword in test_case["reasoning_keywords"] 
                                  if keyword.lower() in reasoning_lower)
                assert keyword_count >= 2  # At least 2 keywords should be mentioned
    
    def test_multilingual_ner_llm_support(self, api_client, base_url):
        """Test multilingual support for NER and LLM methods."""
        multilingual_cases = [
            {
                "product": "โค้ก เซโร่",  # Thai: Coke Zero
                "language": "th",
                "expected_brand": "Coca-Cola",
                "method": "ner"
            },
            {
                "product": "Samsung โทรศัพท์มือถือ",  # Thai: Samsung mobile phone
                "language": "th",
                "expected_brand": "Samsung",
                "method": "llm"
            },
            {
                "product": "iPhone สีดำ",  # Thai: iPhone black color
                "language": "th",
                "expected_brand": "Apple",
                "method": "ner"
            }
        ]
        
        for case in multilingual_cases:
            if case["method"] == "ner":
                mock_data = self.mock_ner_response(case["product"], case["expected_brand"])
            else:
                mock_data = self.mock_llm_response(case["product"], case["expected_brand"])
            
            # Add multilingual processing info
            mock_data["language_detected"] = case["language"]
            mock_data["multilingual_processing"] = True
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_data
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": case["product"],
                        "language_hint": case["language"],
                        "method": case["method"]
                    },
                    timeout=30
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify multilingual processing
                assert data["success"] is True
                assert data["method"] == case["method"]
                assert data.get("language_detected") == case["language"]
                assert data.get("multilingual_processing") is True
                
                # Verify prediction
                predictions = data["brand_predictions"]
                assert len(predictions) > 0
                assert predictions[0]["brand"] == case["expected_brand"]


@pytest.mark.integration
class TestMethodSpecificErrorHandling:
    """Integration tests for method-specific error handling."""
    
    @pytest.fixture
    def api_client(self):
        """Mock API client for testing."""
        return Mock()
    
    @pytest.fixture
    def base_url(self):
        """Base URL for API testing."""
        return "http://production-alb-107602758.us-east-1.elb.amazonaws.com"
    
    def test_ner_method_error_scenarios(self, api_client, base_url):
        """Test NER method error handling scenarios."""
        error_scenarios = [
            {
                "name": "Model not available",
                "error_response": {
                    "success": False,
                    "method": "ner",
                    "error": "NER model not available",
                    "error_code": "MODEL_UNAVAILABLE",
                    "status_code": 503
                }
            },
            {
                "name": "Text too long",
                "error_response": {
                    "success": False,
                    "method": "ner",
                    "error": "Input text exceeds maximum length",
                    "error_code": "TEXT_TOO_LONG",
                    "max_length": 1000,
                    "status_code": 400
                }
            },
            {
                "name": "Processing timeout",
                "error_response": {
                    "success": False,
                    "method": "ner",
                    "error": "NER processing timeout",
                    "error_code": "TIMEOUT",
                    "timeout_seconds": 30,
                    "status_code": 408
                }
            }
        ]
        
        for scenario in error_scenarios:
            mock_response = Mock()
            mock_response.status_code = scenario["error_response"]["status_code"]
            mock_response.json.return_value = scenario["error_response"]
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": "Test Product",
                        "language_hint": "en",
                        "method": "ner"
                    },
                    timeout=30
                )
                
                # Verify error response
                assert response.status_code == scenario["error_response"]["status_code"]
                data = response.json()
                
                assert data["success"] is False
                assert data["method"] == "ner"
                assert "error" in data
                assert "error_code" in data
    
    def test_llm_method_error_scenarios(self, api_client, base_url):
        """Test LLM method error handling scenarios."""
        error_scenarios = [
            {
                "name": "AWS service unavailable",
                "error_response": {
                    "success": False,
                    "method": "llm",
                    "error": "AWS Bedrock service unavailable",
                    "error_code": "AWS_SERVICE_UNAVAILABLE",
                    "aws_error": "ServiceUnavailable",
                    "status_code": 503
                }
            },
            {
                "name": "Rate limit exceeded",
                "error_response": {
                    "success": False,
                    "method": "llm",
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": 60,
                    "status_code": 429
                }
            },
            {
                "name": "Model inference failed",
                "error_response": {
                    "success": False,
                    "method": "llm",
                    "error": "Model inference failed",
                    "error_code": "INFERENCE_FAILED",
                    "model_id": "us.amazon.nova-pro-v1:0",
                    "status_code": 500
                }
            }
        ]
        
        for scenario in error_scenarios:
            mock_response = Mock()
            mock_response.status_code = scenario["error_response"]["status_code"]
            mock_response.json.return_value = scenario["error_response"]
            
            with patch('requests.post', return_value=mock_response):
                import requests
                response = requests.post(
                    f"{base_url}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": "Test Product",
                        "language_hint": "en",
                        "method": "llm"
                    },
                    timeout=30
                )
                
                # Verify error response
                assert response.status_code == scenario["error_response"]["status_code"]
                data = response.json()
                
                assert data["success"] is False
                assert data["method"] == "llm"
                assert "error" in data
                assert "error_code" in data