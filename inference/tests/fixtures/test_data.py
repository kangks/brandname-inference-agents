"""
Standardized test payloads and data for inference system testing.

This module provides consistent test data sets for all test scenarios,
including multilingual content, edge cases, and performance testing data.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import pytest

from inference.src.models.data_models import ProductInput, LanguageHint


@dataclass
class TestPayload:
    """Simple test payload class for E2E tests."""
    product_name: str = "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2"
    language_hint: str = "en"
    method: str = "orchestrator"


# Standard test payloads for consistent testing
STANDARD_TEST_PAYLOADS = {
    "basic_english": ProductInput(
        product_name="Samsung Galaxy S24 Ultra 256GB Titanium Black",
        language_hint=LanguageHint.ENGLISH
    ),
    
    "basic_thai": ProductInput(
        product_name="โค้กเซโร่ 325 มล. แพ็ค 6 กระป๋อง",
        language_hint=LanguageHint.THAI
    ),
    
    "mixed_language": ProductInput(
        product_name="iPhone 15 Pro Max สีทิเทเนียมธรรมชาติ 256GB",
        language_hint=LanguageHint.MIXED
    ),
    
    "auto_detect": ProductInput(
        product_name="Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
        language_hint=LanguageHint.AUTO
    ),
    
    "brand_heavy": ProductInput(
        product_name="Nike Air Max 270 React Black White Running Shoes",
        language_hint=LanguageHint.ENGLISH
    ),
    
    "category_heavy": ProductInput(
        product_name="Wireless Bluetooth Headphones with Noise Cancellation",
        language_hint=LanguageHint.ENGLISH
    ),
    
    "minimal": ProductInput(
        product_name="Apple iPhone",
        language_hint=LanguageHint.ENGLISH
    ),
    
    "complex": ProductInput(
        product_name="Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones with Auto Noise Canceling Optimizer, Crystal Clear Hands-Free Calling, and Alexa Voice Control, Silver",
        language_hint=LanguageHint.ENGLISH
    )
}


# Edge case test payloads for error handling testing
# Note: These are created as dictionaries to avoid validation at module load time
EDGE_CASE_PAYLOADS = {
    "empty_string": {
        "product_name": "",
        "language_hint": "auto"
    },
    
    "whitespace_only": {
        "product_name": "   ",
        "language_hint": "auto"
    },
    
    "single_character": ProductInput(
        product_name="A",
        language_hint=LanguageHint.AUTO
    ),
    
    "very_long": ProductInput(
        product_name="A" * 1000,
        language_hint=LanguageHint.AUTO
    ),
    
    "special_characters": ProductInput(
        product_name="!@#$%^&*()_+-=[]{}|;:,.<>?",
        language_hint=LanguageHint.AUTO
    ),
    
    "numbers_only": ProductInput(
        product_name="1234567890",
        language_hint=LanguageHint.AUTO
    ),
    
    "unicode_emoji": ProductInput(
        product_name="🚀🎯💡🔥⚡ Product Name",
        language_hint=LanguageHint.AUTO
    ),
    
    "mixed_scripts": ProductInput(
        product_name="Product Название товара 商品名称 製品名",
        language_hint=LanguageHint.AUTO
    ),
    
    "leading_trailing_spaces": ProductInput(
        product_name="   Product Name   ",
        language_hint=LanguageHint.AUTO
    ),
    
    "multiple_spaces": ProductInput(
        product_name="Product    Name    With    Spaces",
        language_hint=LanguageHint.AUTO
    )
}


# Performance test payloads for load testing
PERFORMANCE_TEST_PAYLOADS = {
    "small_batch": [
        ProductInput(f"Test Product {i}", LanguageHint.AUTO) 
        for i in range(10)
    ],
    
    "medium_batch": [
        ProductInput(f"Performance Test Product {i} with longer name", LanguageHint.AUTO) 
        for i in range(100)
    ],
    
    "large_batch": [
        ProductInput(f"Large Scale Performance Test Product {i} with extended description", LanguageHint.AUTO) 
        for i in range(1000)
    ],
    
    "mixed_language_batch": [
        ProductInput(f"Mixed Language Product {i} สินค้าทดสอบ {i}", LanguageHint.MIXED) 
        for i in range(50)
    ]
}


# Brand-specific test payloads for accuracy testing
BRAND_SPECIFIC_PAYLOADS = {
    "samsung": [
        ProductInput("Samsung Galaxy S24 Ultra", LanguageHint.ENGLISH),
        ProductInput("Samsung QLED 4K Smart TV", LanguageHint.ENGLISH),
        ProductInput("Samsung Galaxy Buds Pro", LanguageHint.ENGLISH),
        ProductInput("Samsung Refrigerator French Door", LanguageHint.ENGLISH)
    ],
    
    "apple": [
        ProductInput("iPhone 15 Pro Max", LanguageHint.ENGLISH),
        ProductInput("MacBook Pro M3", LanguageHint.ENGLISH),
        ProductInput("iPad Air 5th Generation", LanguageHint.ENGLISH),
        ProductInput("AirPods Pro 2nd Generation", LanguageHint.ENGLISH)
    ],
    
    "nike": [
        ProductInput("Nike Air Max 270", LanguageHint.ENGLISH),
        ProductInput("Nike Dri-FIT Running Shirt", LanguageHint.ENGLISH),
        ProductInput("Nike Air Jordan 1", LanguageHint.ENGLISH),
        ProductInput("Nike React Infinity Run", LanguageHint.ENGLISH)
    ],
    
    "thai_brands": [
        ProductInput("เซเว่น อีเลฟเว่น กาแฟ", LanguageHint.THAI),
        ProductInput("โลตัส เฟรช มาร์ท", LanguageHint.THAI),
        ProductInput("บิ๊กซี ซูเปอร์เซ็นเตอร์", LanguageHint.THAI),
        ProductInput("ซีพี เฟรช มาร์ท", LanguageHint.THAI)
    ]
}


# Method-specific test payloads for testing different inference methods
METHOD_SPECIFIC_PAYLOADS = {
    "ner_focused": [
        ProductInput("Apple iPhone 15 Pro Max", LanguageHint.ENGLISH),
        ProductInput("Samsung Galaxy S24 Ultra", LanguageHint.ENGLISH),
        ProductInput("Nike Air Max 270 React", LanguageHint.ENGLISH),
        ProductInput("Sony WH-1000XM5 Headphones", LanguageHint.ENGLISH)
    ],
    
    "rag_focused": [
        ProductInput("Wireless Bluetooth Headphones", LanguageHint.ENGLISH),
        ProductInput("Smart TV 4K QLED", LanguageHint.ENGLISH),
        ProductInput("Running Shoes Athletic", LanguageHint.ENGLISH),
        ProductInput("Smartphone 256GB Storage", LanguageHint.ENGLISH)
    ],
    
    "llm_focused": [
        ProductInput("Premium noise-canceling wireless headphones with advanced features", LanguageHint.ENGLISH),
        ProductInput("High-performance smartphone with professional camera system", LanguageHint.ENGLISH),
        ProductInput("Comfortable running shoes designed for long-distance athletes", LanguageHint.ENGLISH),
        ProductInput("Smart home appliance with energy-efficient technology", LanguageHint.ENGLISH)
    ],
    
    "hybrid_focused": [
        ProductInput("iPhone 15 Pro Max with advanced camera system and titanium design", LanguageHint.ENGLISH),
        ProductInput("Samsung QLED 4K Smart TV with quantum dot technology", LanguageHint.ENGLISH),
        ProductInput("Nike Air Max 270 React running shoes with responsive cushioning", LanguageHint.ENGLISH),
        ProductInput("Sony WH-1000XM5 wireless headphones with industry-leading noise cancellation", LanguageHint.ENGLISH)
    ]
}


@dataclass
class TestDataSet:
    """
    Container for organized test data sets.
    
    Provides structured access to different categories of test data
    with metadata and validation.
    """
    
    name: str
    description: str
    payloads: List[ProductInput]
    expected_success_rate: float = 1.0
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


# Organized test data sets
TEST_DATA_SETS = {
    "standard": TestDataSet(
        name="standard",
        description="Standard test payloads for basic functionality testing",
        payloads=list(STANDARD_TEST_PAYLOADS.values()),
        expected_success_rate=1.0
    ),
    
    "edge_cases": TestDataSet(
        name="edge_cases",
        description="Edge case payloads for error handling testing",
        payloads=list(EDGE_CASE_PAYLOADS.values()),
        expected_success_rate=0.3,  # Many edge cases expected to fail
        timeout_seconds=10.0
    ),
    
    "performance_small": TestDataSet(
        name="performance_small",
        description="Small batch for performance testing",
        payloads=PERFORMANCE_TEST_PAYLOADS["small_batch"],
        expected_success_rate=1.0,
        timeout_seconds=60.0
    ),
    
    "performance_medium": TestDataSet(
        name="performance_medium",
        description="Medium batch for performance testing",
        payloads=PERFORMANCE_TEST_PAYLOADS["medium_batch"],
        expected_success_rate=1.0,
        timeout_seconds=300.0
    ),
    
    "multilingual": TestDataSet(
        name="multilingual",
        description="Mixed language content for multilingual testing",
        payloads=[
            STANDARD_TEST_PAYLOADS["basic_thai"],
            STANDARD_TEST_PAYLOADS["mixed_language"],
            STANDARD_TEST_PAYLOADS["auto_detect"]
        ] + PERFORMANCE_TEST_PAYLOADS["mixed_language_batch"],
        expected_success_rate=0.9
    ),
    
    "brand_accuracy": TestDataSet(
        name="brand_accuracy",
        description="Brand-specific payloads for accuracy testing",
        payloads=[
            payload for brand_payloads in BRAND_SPECIFIC_PAYLOADS.values()
            for payload in brand_payloads
        ],
        expected_success_rate=0.95
    )
}


# Pytest fixtures for test data
@pytest.fixture(scope="function")
def standard_test_payload():
    """Get a standard test payload."""
    return STANDARD_TEST_PAYLOADS["auto_detect"]


@pytest.fixture(scope="function", params=list(STANDARD_TEST_PAYLOADS.keys()))
def all_standard_payloads(request):
    """Parametrized fixture for all standard test payloads."""
    return STANDARD_TEST_PAYLOADS[request.param]


@pytest.fixture(scope="function", params=list(EDGE_CASE_PAYLOADS.keys()))
def all_edge_case_payloads(request):
    """Parametrized fixture for all edge case payloads."""
    return EDGE_CASE_PAYLOADS[request.param]


@pytest.fixture(scope="function")
def performance_test_batch():
    """Get a batch of payloads for performance testing."""
    return PERFORMANCE_TEST_PAYLOADS["small_batch"]


@pytest.fixture(scope="function", params=["samsung", "apple", "nike"])
def brand_specific_payloads(request):
    """Parametrized fixture for brand-specific test payloads."""
    return BRAND_SPECIFIC_PAYLOADS[request.param]


@pytest.fixture(scope="function", params=["ner_focused", "rag_focused", "llm_focused", "hybrid_focused"])
def method_specific_payloads(request):
    """Parametrized fixture for method-specific test payloads."""
    return METHOD_SPECIFIC_PAYLOADS[request.param]


@pytest.fixture(scope="function", params=list(TEST_DATA_SETS.keys()))
def test_data_sets(request):
    """Parametrized fixture for organized test data sets."""
    return TEST_DATA_SETS[request.param]


# Helper functions for test data access
def get_test_payload(payload_name: str) -> ProductInput:
    """
    Get a specific test payload by name.
    
    Args:
        payload_name: Name of the payload to retrieve
        
    Returns:
        ProductInput instance
        
    Raises:
        KeyError: If payload name not found
    """
    if payload_name in STANDARD_TEST_PAYLOADS:
        return STANDARD_TEST_PAYLOADS[payload_name]
    elif payload_name in EDGE_CASE_PAYLOADS:
        return EDGE_CASE_PAYLOADS[payload_name]
    else:
        raise KeyError(f"Test payload '{payload_name}' not found")


def get_test_data_set(set_name: str) -> TestDataSet:
    """
    Get a specific test data set by name.
    
    Args:
        set_name: Name of the test data set
        
    Returns:
        TestDataSet instance
        
    Raises:
        KeyError: If set name not found
    """
    if set_name not in TEST_DATA_SETS:
        raise KeyError(f"Test data set '{set_name}' not found")
    
    return TEST_DATA_SETS[set_name]


def get_payloads_by_language(language_hint: LanguageHint) -> List[ProductInput]:
    """
    Get test payloads filtered by language hint.
    
    Args:
        language_hint: Language hint to filter by
        
    Returns:
        List of ProductInput instances matching the language hint
    """
    matching_payloads = []
    
    for payload in STANDARD_TEST_PAYLOADS.values():
        if payload.language_hint == language_hint:
            matching_payloads.append(payload)
    
    return matching_payloads


def get_payloads_for_method_testing(method_name: str) -> List[ProductInput]:
    """
    Get test payloads optimized for testing a specific method.
    
    Args:
        method_name: Name of the inference method (ner, rag, llm, hybrid)
        
    Returns:
        List of ProductInput instances optimized for the method
    """
    method_key = f"{method_name}_focused"
    
    if method_key in METHOD_SPECIFIC_PAYLOADS:
        return METHOD_SPECIFIC_PAYLOADS[method_key]
    else:
        # Return standard payloads as fallback
        return list(STANDARD_TEST_PAYLOADS.values())


# Expected results mapping for validation
EXPECTED_BRANDS = {
    "Samsung Galaxy S24 Ultra 256GB Titanium Black": "Samsung",
    "iPhone 15 Pro Max สีทิเทเนียมธรรมชาติ 256GB": "Apple",
    "Nike Air Max 270 React Black White Running Shoes": "Nike",
    "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones": "Sony",
    "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2": "Alectric",
    "Apple iPhone": "Apple",
    "Samsung Galaxy Buds Pro": "Samsung"
}


def get_expected_brand(product_name: str) -> str:
    """
    Get expected brand for a given product name.
    
    Args:
        product_name: Product name to look up
        
    Returns:
        Expected brand name or "Unknown" if not found
    """
    return EXPECTED_BRANDS.get(product_name, "Unknown")