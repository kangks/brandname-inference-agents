"""
Simple inference agent for basic brand prediction without external dependencies.

This module provides a lightweight inference capability that works without
external services, using basic pattern matching for demonstration purposes.
"""

import re
import time
from typing import Dict, Any, List, Optional
import logging

from ..models.data_models import ProductInput, LanguageHint
from .base_agent import BaseAgent, AgentError


class SimpleInferenceAgent(BaseAgent):
    """
    Simple inference agent providing basic brand prediction.
    
    Uses pattern matching and keyword detection to provide immediate
    brand predictions without external dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize simple inference agent."""
        super().__init__("simple_inference", config or {})
        
        # Load brand patterns
        self.brand_patterns = self._load_brand_patterns()
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.6)
    
    async def initialize(self) -> None:
        """Initialize the simple agent."""
        try:
            # Simple initialization - no external dependencies
            self.set_initialized(True)
            self.logger.info("Simple inference agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simple agent: {str(e)}")
            raise AgentError(self.agent_name, f"Initialization failed: {str(e)}", e)
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input and return brand prediction.
        
        Args:
            input_data: Product input data
            
        Returns:
            Dictionary containing inference results in orchestrator format
        """
        start_time = time.time()
        
        try:
            if not self._is_initialized:
                raise AgentError(self.agent_name, "Agent not initialized")
            
            # Perform simple inference
            prediction_result = await self.predict_brand(
                input_data.product_name,
                input_data.language_hint
            )
            
            processing_time = time.time() - start_time
            
            # Create a simple result object that matches LLMResult structure
            from ..models.data_models import LLMResult
            
            simple_result = LLMResult(
                predicted_brand=prediction_result["predicted_brand"],
                reasoning=f"Pattern matching using method: {prediction_result['method']}",
                confidence=prediction_result["confidence"],
                processing_time=processing_time,
                model_id="simple_pattern_matcher"
            )
            
            return {
                "agent_type": "simple",
                "result": simple_result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Simple inference failed: {str(e)}")
            
            return {
                "agent_type": "simple",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def predict_brand(
        self, 
        product_name: str, 
        language_hint: LanguageHint = LanguageHint.AUTO
    ) -> Dict[str, Any]:
        """
        Predict brand from product name using simple pattern matching.
        
        Args:
            product_name: Input product name
            language_hint: Language hint for processing
            
        Returns:
            Dictionary containing prediction results
        """
        if not product_name or not product_name.strip():
            raise AgentError(self.agent_name, "Product name cannot be empty")
        
        original_name = product_name.strip()
        product_name_lower = original_name.lower()
        
        # Find brand matches with confidence scoring
        best_match = None
        best_confidence = 0.0
        best_method = "none"
        
        for brand_name, patterns in self.brand_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, product_name_lower, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    confidence = self._calculate_match_confidence(
                        match, pattern, product_name_lower, brand_name
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = brand_name
                        best_method = "pattern_match"
        
        # If we found a good match, return it
        if best_match and best_confidence >= self.confidence_threshold:
            return {
                "predicted_brand": best_match,
                "confidence": best_confidence,
                "method": best_method
            }
        
        # Try fuzzy matching for partial matches
        fuzzy_result = self._try_fuzzy_matching(product_name_lower)
        if fuzzy_result["confidence"] > best_confidence:
            return fuzzy_result
        
        # Default prediction for unknown brands
        return {
            "predicted_brand": "Unknown",
            "confidence": 0.1,  # Very low confidence for unknown
            "method": "default"
        }
    
    def _calculate_match_confidence(
        self, 
        match: re.Match, 
        pattern: str, 
        product_name: str, 
        brand_name: str
    ) -> float:
        """Calculate confidence score for a pattern match."""
        base_confidence = 0.7
        
        # Boost confidence for exact brand name matches
        if brand_name.lower() in product_name:
            base_confidence += 0.2
        
        # Boost confidence for longer matches
        match_length = len(match.group())
        if match_length > 4:
            base_confidence += 0.1
        
        # Boost confidence for word boundary matches
        if r'\b' in pattern:
            base_confidence += 0.1
        
        # Boost confidence for specific product model patterns
        if re.search(r'\d+', match.group()):  # Contains numbers (model numbers)
            base_confidence += 0.05
        
        return min(0.95, base_confidence)  # Cap at 0.95
    
    def _try_fuzzy_matching(self, product_name: str) -> Dict[str, Any]:
        """Try fuzzy matching for partial brand name matches."""
        words = product_name.split()
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            for brand_name in self.brand_patterns.keys():
                # Simple similarity check
                similarity = self._calculate_similarity(word, brand_name.lower())
                
                if similarity > 0.8:  # High similarity threshold
                    confidence = similarity * 0.6  # Lower confidence for fuzzy matches
                    return {
                        "predicted_brand": brand_name,
                        "confidence": confidence,
                        "method": "fuzzy_match"
                    }
        
        return {
            "predicted_brand": "Unknown",
            "confidence": 0.0,
            "method": "none"
        }
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate simple similarity between two words."""
        if word1 == word2:
            return 1.0
        
        # Simple Jaccard similarity using character bigrams
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(word1)
        bigrams2 = get_bigrams(word2)
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _load_brand_patterns(self) -> Dict[str, List[str]]:
        """Load brand patterns for matching."""
        return {
            "Apple": [r"\biphone\b", r"\bipad\b", r"\bmac\b", r"\bapple\b", r"\bairpods\b", r"\bwatch\b.*apple", r"\bmacbook\b"],
            "Samsung": [r"\bsamsung\b", r"\bgalaxy\b", r"\bnote\b.*samsung", r"\bs\d+\b", r"\bgalaxy\s+s\d+"],
            "Nike": [r"\bnike\b", r"\bair jordan\b", r"\bjordan\b", r"\bair max\b", r"\bdunk\b", r"\bair force\b"],
            "Adidas": [r"\badidas\b", r"\byeezy\b", r"\bultraboost\b", r"\bstan smith\b", r"\boriginals\b"],
            "Sony": [r"\bsony\b", r"\bplaystation\b", r"\bps\d\b", r"\bbravia\b", r"\bxperia\b", r"\bwalkman\b"],
            "Microsoft": [r"\bmicrosoft\b", r"\bsurface\b", r"\bxbox\b", r"\bwindows\b", r"\boffice\b"],
            "Google": [r"\bgoogle\b", r"\bpixel\b", r"\bandroid\b", r"\bchrome\b", r"\bnest\b"],
            "Toyota": [r"\btoyota\b", r"\bcamry\b", r"\bcorolla\b", r"\bprius\b", r"\brav4\b", r"\bhighlander\b"],
            "Honda": [r"\bhonda\b", r"\baccord\b", r"\bcivic\b", r"\bcr-v\b", r"\bpilot\b"],
            "Coca-Cola": [r"\bcoca.?cola\b", r"\bcoke\b", r"\bโค้ก\b", r"\bzero\b.*coke", r"\bdiet.*coke"],
            "Pepsi": [r"\bpepsi\b", r"\bเป๊ปซี่\b", r"\bmax\b.*pepsi"],
            "Colgate": [r"\bcolgate\b", r"\btotal\b.*colgate", r"ยาสีฟัน.*colgate"],
            "LG": [r"\blg\b", r"\boled\b.*lg", r"\bnanocell\b"],
            "Huawei": [r"\bhuawei\b", r"\bmate\b", r"\bp\d+\b.*huawei"],
            "Xiaomi": [r"\bxiaomi\b", r"\bmi\b", r"\bredmi\b", r"\bpoco\b"],
            "BMW": [r"\bbmw\b", r"\bseries\b.*\d", r"\bx\d\b"],
            "Mercedes": [r"\bmercedes\b", r"\bbenz\b", r"\bc-class\b", r"\be-class\b"],
            "Audi": [r"\baudi\b", r"\ba\d\b", r"\bq\d\b", r"\brs\d\b"],
            "Starbucks": [r"\bstarbucks\b", r"\bfrappuccino\b"],
            "McDonald's": [r"\bmcdonald\b", r"\bmcdonalds\b", r"\bbig mac\b", r"\bmcflurry\b"]
        }
    
    async def cleanup(self) -> None:
        """Clean up simple agent resources."""
        self.brand_patterns.clear()
        self.set_initialized(False)
        self.logger.info("Simple inference agent cleaned up")
    
    async def _perform_health_check(self) -> None:
        """Perform simple agent health check."""
        await super()._perform_health_check()
        
        # Test basic functionality
        try:
            test_result = await self.predict_brand("iPhone 15 Pro Max", LanguageHint.EN)
            if not test_result:
                raise RuntimeError("Basic inference test failed")
        except Exception as e:
            raise RuntimeError(f"Health check inference failed: {e}")


# Factory function
def create_simple_agent(config: Optional[Dict[str, Any]] = None) -> SimpleInferenceAgent:
    """
    Create a simple inference agent.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured simple inference agent
    """
    return SimpleInferenceAgent(config)