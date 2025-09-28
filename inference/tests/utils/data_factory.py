"""
Test data factory for generating test data programmatically.

This module provides factories for creating test data, mock responses,
and test scenarios for comprehensive testing of the inference system.
"""

import random
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from inference.src.models.data_models import (
    ProductInput,
    LanguageHint,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    EntityResult,
    EntityType,
    SimilarProduct
)


class TestScenario(Enum):
    """Test scenario types for different testing needs."""
    
    BASIC = "basic"
    MULTILINGUAL = "multilingual"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"
    ERROR_SIMULATION = "error_simulation"


@dataclass
class TestCase:
    """
    Test case data structure.
    
    Represents a complete test case with input, expected output,
    and metadata for test execution.
    """
    
    name: str
    input_data: ProductInput
    expected_success: bool
    expected_brand: Optional[str] = None
    expected_confidence_min: float = 0.0
    expected_confidence_max: float = 1.0
    scenario_type: TestScenario = TestScenario.BASIC
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class TestDataFactory:
    """
    Factory for generating test data programmatically.
    
    Provides methods to create various types of test data including
    inputs, expected outputs, and complete test scenarios.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test data factory.
        
        Args:
            seed: Random seed for reproducible test data generation
        """
        if seed is not None:
            random.seed(seed)
        
        # Predefined test data sets
        self.brand_names = [
            "Samsung", "Apple", "Sony", "Nike", "Adidas", "Toyota", "Honda",
            "LG", "Huawei", "Xiaomi", "Microsoft", "Google", "Amazon",
            "Coca-Cola", "Pepsi", "McDonald's", "Starbucks", "Alectric"
        ]
        
        self.product_categories = [
            "Electronics", "Clothing", "Automotive", "Food & Beverage",
            "Home Appliances", "Sports Equipment", "Beauty", "Books"
        ]
        
        self.thai_brands = [
            "เซเว่น", "โลตัส", "บิ๊กซี", "เทสโก้", "ซีพี", "ไทยยูเนี่ยน",
            "กรุงไทย", "กสิกรไทย", "ไทยพาณิชย์"
        ]
        
        self.mixed_language_products = [
            "iPhone 15 Pro Max สีทิเทเนียมธรรมชาติ 256GB",
            "Samsung Galaxy S24 Ultra โทรศัพท์มือถือ",
            "Nike Air Max 270 รองเท้าผ้าใบ สีดำ",
            "โค้กเซโร่ Coca-Cola Zero 325ml",
            "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2"
        ]
    
    def create_basic_product_input(self, product_name: str = None, 
                                 language_hint: LanguageHint = LanguageHint.AUTO) -> ProductInput:
        """
        Create a basic product input for testing.
        
        Args:
            product_name: Product name (random if None)
            language_hint: Language hint for processing
            
        Returns:
            ProductInput instance
        """
        if product_name is None:
            brand = random.choice(self.brand_names)
            category = random.choice(self.product_categories)
            product_name = f"{brand} {category} Model {random.randint(100, 999)}"
        
        return ProductInput(
            product_name=product_name,
            language_hint=language_hint
        )
    
    def create_multilingual_product_input(self) -> ProductInput:
        """
        Create a multilingual product input for testing.
        
        Returns:
            ProductInput with mixed language content
        """
        product_name = random.choice(self.mixed_language_products)
        return ProductInput(
            product_name=product_name,
            language_hint=LanguageHint.MIXED
        )
    
    def create_edge_case_product_input(self) -> ProductInput:
        """
        Create edge case product input for testing error handling.
        
        Returns:
            ProductInput with edge case data
        """
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "A" * 1000,  # Very long string
            "!@#$%^&*()",  # Special characters only
            "123456789",  # Numbers only
            "   Leading and trailing spaces   ",
            "Multiple    spaces    between    words"
        ]
        
        product_name = random.choice(edge_cases)
        return ProductInput(
            product_name=product_name,
            language_hint=LanguageHint.AUTO
        )
    
    def create_ner_result(self, confidence: float = None, 
                         num_entities: int = None) -> NERResult:
        """
        Create a mock NER result for testing.
        
        Args:
            confidence: Overall confidence (random if None)
            num_entities: Number of entities (random if None)
            
        Returns:
            NERResult instance
        """
        if confidence is None:
            confidence = random.uniform(0.5, 0.95)
        
        if num_entities is None:
            num_entities = random.randint(1, 3)
        
        entities = []
        for i in range(num_entities):
            entity = EntityResult(
                entity_type=random.choice(list(EntityType)),
                text=random.choice(self.brand_names),
                confidence=random.uniform(0.6, 0.9),
                start_pos=i * 10,
                end_pos=(i * 10) + len(random.choice(self.brand_names))
            )
            entities.append(entity)
        
        return NERResult(
            entities=entities,
            confidence=confidence,
            processing_time=random.uniform(0.1, 0.5),
            model_used="test_ner_model"
        )
    
    def create_rag_result(self, confidence: float = None,
                         num_similar: int = None) -> RAGResult:
        """
        Create a mock RAG result for testing.
        
        Args:
            confidence: Overall confidence (random if None)
            num_similar: Number of similar products (random if None)
            
        Returns:
            RAGResult instance
        """
        if confidence is None:
            confidence = random.uniform(0.6, 0.9)
        
        if num_similar is None:
            num_similar = random.randint(1, 5)
        
        similar_products = []
        for i in range(num_similar):
            brand = random.choice(self.brand_names)
            category = random.choice(self.product_categories)
            
            similar_product = SimilarProduct(
                product_name=f"{brand} {category} {i+1}",
                brand=brand,
                category=category,
                sub_category=f"Sub-{category}",
                similarity_score=random.uniform(0.7, 0.95)
            )
            similar_products.append(similar_product)
        
        return RAGResult(
            similar_products=similar_products,
            predicted_brand=random.choice(self.brand_names),
            confidence=confidence,
            processing_time=random.uniform(0.2, 0.8),
            embedding_model="test_embedding_model"
        )
    
    def create_llm_result(self, confidence: float = None,
                         predicted_brand: str = None) -> LLMResult:
        """
        Create a mock LLM result for testing.
        
        Args:
            confidence: Overall confidence (random if None)
            predicted_brand: Predicted brand (random if None)
            
        Returns:
            LLMResult instance
        """
        if confidence is None:
            confidence = random.uniform(0.7, 0.95)
        
        if predicted_brand is None:
            predicted_brand = random.choice(self.brand_names)
        
        reasoning_templates = [
            f"The product name clearly indicates {predicted_brand} brand based on naming patterns.",
            f"Analysis of product features suggests {predicted_brand} as the most likely brand.",
            f"Brand identification confidence is high for {predicted_brand} based on context clues.",
            f"Product characteristics align with {predicted_brand} brand specifications."
        ]
        
        return LLMResult(
            predicted_brand=predicted_brand,
            reasoning=random.choice(reasoning_templates),
            confidence=confidence,
            processing_time=random.uniform(0.3, 1.2),
            model_id="test_llm_model"
        )
    
    def create_hybrid_result(self, confidence: float = None,
                           stages_used: List[str] = None) -> HybridResult:
        """
        Create a mock Hybrid result for testing.
        
        Args:
            confidence: Overall confidence (random if None)
            stages_used: List of stages used (default if None)
            
        Returns:
            HybridResult instance
        """
        if confidence is None:
            confidence = random.uniform(0.8, 0.95)
        
        if stages_used is None:
            stages_used = ["ner", "rag", "llm"]
        
        predicted_brand = random.choice(self.brand_names)
        
        stage_results = {}
        for stage in stages_used:
            stage_results[stage] = {
                "brand": predicted_brand,
                "confidence": random.uniform(0.7, 0.9)
            }
        
        return HybridResult(
            final_prediction=predicted_brand,
            confidence=confidence,
            processing_time=sum(random.uniform(0.1, 0.5) for _ in stages_used),
            stage_results=stage_results,
            stages_used=stages_used
        )
    
    def create_test_case(self, scenario: TestScenario = TestScenario.BASIC,
                        name: str = None) -> TestCase:
        """
        Create a complete test case based on scenario type.
        
        Args:
            scenario: Type of test scenario
            name: Test case name (auto-generated if None)
            
        Returns:
            TestCase instance
        """
        if name is None:
            name = f"test_{scenario.value}_{random.randint(1000, 9999)}"
        
        if scenario == TestScenario.BASIC:
            input_data = self.create_basic_product_input()
            expected_success = True
            expected_brand = self.brand_names[0]  # Predictable for basic tests
            
        elif scenario == TestScenario.MULTILINGUAL:
            input_data = self.create_multilingual_product_input()
            expected_success = True
            expected_brand = None  # May vary based on language processing
            
        elif scenario == TestScenario.EDGE_CASE:
            input_data = self.create_edge_case_product_input()
            expected_success = False  # Edge cases may fail
            expected_brand = None
            
        elif scenario == TestScenario.PERFORMANCE:
            # Large product name for performance testing
            large_name = " ".join(self.brand_names * 10)
            input_data = ProductInput(product_name=large_name)
            expected_success = True
            expected_brand = None
            
        elif scenario == TestScenario.ERROR_SIMULATION:
            input_data = self.create_basic_product_input()
            expected_success = False  # Simulated error scenario
            expected_brand = None
            
        else:
            # Default to basic scenario
            input_data = self.create_basic_product_input()
            expected_success = True
            expected_brand = None
        
        return TestCase(
            name=name,
            input_data=input_data,
            expected_success=expected_success,
            expected_brand=expected_brand,
            scenario_type=scenario,
            metadata={"created_at": time.time()}
        )
    
    def create_test_suite(self, num_cases: int = 10,
                         scenarios: List[TestScenario] = None) -> List[TestCase]:
        """
        Create a complete test suite with multiple test cases.
        
        Args:
            num_cases: Number of test cases to create
            scenarios: List of scenarios to include (all if None)
            
        Returns:
            List of TestCase instances
        """
        if scenarios is None:
            scenarios = list(TestScenario)
        
        test_cases = []
        for i in range(num_cases):
            scenario = random.choice(scenarios)
            test_case = self.create_test_case(
                scenario=scenario,
                name=f"test_case_{i+1:03d}_{scenario.value}"
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def create_performance_test_data(self, size_category: str = "medium") -> List[ProductInput]:
        """
        Create test data for performance testing.
        
        Args:
            size_category: Size category (small, medium, large, xlarge)
            
        Returns:
            List of ProductInput instances for performance testing
        """
        size_configs = {
            "small": {"count": 10, "name_length": 50},
            "medium": {"count": 100, "name_length": 100},
            "large": {"count": 1000, "name_length": 200},
            "xlarge": {"count": 5000, "name_length": 500}
        }
        
        config = size_configs.get(size_category, size_configs["medium"])
        
        test_data = []
        for i in range(config["count"]):
            # Create product name of specified length
            base_name = f"Performance Test Product {i+1}"
            padding = "A" * max(0, config["name_length"] - len(base_name))
            product_name = base_name + " " + padding
            
            test_input = ProductInput(
                product_name=product_name,
                language_hint=LanguageHint.AUTO
            )
            test_data.append(test_input)
        
        return test_data
    
    def create_error_test_scenarios(self) -> List[TestCase]:
        """
        Create test cases specifically for error handling scenarios.
        
        Returns:
            List of TestCase instances for error testing
        """
        error_scenarios = [
            # Empty/invalid inputs
            TestCase(
                name="empty_product_name",
                input_data=ProductInput(product_name=""),
                expected_success=False
            ),
            TestCase(
                name="whitespace_only",
                input_data=ProductInput(product_name="   "),
                expected_success=False
            ),
            TestCase(
                name="very_long_input",
                input_data=ProductInput(product_name="A" * 10000),
                expected_success=False,
                timeout_seconds=5.0  # Should timeout or handle gracefully
            ),
            TestCase(
                name="special_characters_only",
                input_data=ProductInput(product_name="!@#$%^&*()"),
                expected_success=False
            ),
            TestCase(
                name="unicode_edge_cases",
                input_data=ProductInput(product_name="🚀🎯💡🔥⚡"),
                expected_success=True,  # Should handle Unicode
                expected_confidence_min=0.0,
                expected_confidence_max=0.5  # Low confidence expected
            )
        ]
        
        return error_scenarios


# Global factory instance
_global_factory: Optional[TestDataFactory] = None


def get_test_data_factory(seed: Optional[int] = None) -> TestDataFactory:
    """
    Get the global test data factory instance.
    
    Args:
        seed: Random seed for reproducible data generation
        
    Returns:
        Global TestDataFactory instance
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = TestDataFactory(seed=seed)
    
    return _global_factory


def reset_test_data_factory():
    """Reset the global test data factory."""
    global _global_factory
    _global_factory = None