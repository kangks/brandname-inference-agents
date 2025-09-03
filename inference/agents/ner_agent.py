"""
Named Entity Recognition (NER) Agent implementation.

This module implements spaCy-based NER agent with custom entity recognition
following PEP 8 coding standards and referencing patterns from the notebook.
Supports multilingual Thai-English mixed text processing.
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio

try:
    import spacy
    from spacy.lang.xx import MultiLanguage
except ImportError:
    spacy = None

from ..models.data_models import (
    ProductInput,
    NERResult,
    EntityResult,
    EntityType,
    LanguageHint
)
from .base_agent import NERAgent, AgentError, AgentInitializationError


class SpacyNERAgent(NERAgent):
    """
    spaCy-based Named Entity Recognition agent.
    
    Implements multilingual entity extraction for brand, category, and variant
    identification from product names, following patterns from the reference notebook.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize spaCy NER agent.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__("spacy_ner", config)
        
        self.nlp = None
        self.model_name = self.get_config_value("model_name", "xx_ent_wiki_sm")
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.5)
        self.max_text_length = self.get_config_value("max_text_length", 1000)
        
        # Entity type mappings from spaCy to our custom types
        self.entity_mappings = {
            "ORG": EntityType.BRAND,
            "PRODUCT": EntityType.PRODUCT,
            "BRAND": EntityType.BRAND,
            "CATEGORY": EntityType.CATEGORY,
            "VARIANT": EntityType.VARIANT,
            "PERSON": EntityType.BRAND,  # Sometimes brand names are person names
        }
        
        # Brand indicators for confidence scoring (from notebook patterns)
        self.brand_indicators = {
            "en": ["brand", "by", "from", "made by", "®", "™", "©"],
            "th": ["แบรนด์", "ยี่ห้อ", "จาก", "โดย", "ของ"],
            "mixed": ["brand", "แบรนด์", "ยี่ห้อ", "®", "™"]
        }
    
    async def initialize(self) -> None:
        """Initialize spaCy model and resources."""
        try:
            if spacy is None:
                raise AgentInitializationError(
                    self.agent_name,
                    "spaCy library not installed. Please install with: pip install spacy"
                )
            
            # Try to load the model
            try:
                self.nlp = spacy.load(self.model_name)
                self.logger.info(f"Loaded spaCy model: {self.model_name}")
            except OSError as e:
                self.logger.warning(f"Model {self.model_name} not found: {str(e)}")
                # Try fallback models
                fallback_models = ["en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"]
                model_loaded = False
                
                for fallback_model in fallback_models:
                    if fallback_model != self.model_name:
                        try:
                            self.nlp = spacy.load(fallback_model)
                            self.model_name = fallback_model
                            self.logger.info(f"Loaded fallback spaCy model: {fallback_model}")
                            model_loaded = True
                            break
                        except OSError:
                            continue
                
                if not model_loaded:
                    # Try to download the original model
                    try:
                        self.logger.info(f"Attempting to download model: {self.model_name}")
                        await self._download_model()
                        self.nlp = spacy.load(self.model_name)
                        self.logger.info(f"Downloaded and loaded spaCy model: {self.model_name}")
                    except Exception as download_error:
                        raise AgentInitializationError(
                            self.agent_name,
                            f"Failed to load or download spaCy model {self.model_name}: {str(download_error)}"
                        )
            
            # Verify model has NER component
            if "ner" not in self.nlp.pipe_names:
                raise AgentInitializationError(
                    self.agent_name,
                    f"Model {self.model_name} does not have NER component"
                )
            
            self.set_initialized(True)
            self.logger.info(f"spaCy NER agent initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            self.set_initialized(False)
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to initialize spaCy NER agent: {str(e)}",
                e
            )
    
    async def _download_model(self) -> None:
        """Download spaCy model if not available."""
        try:
            import spacy.cli
            
            # Run download in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                spacy.cli.download,
                self.model_name
            )
            self.logger.info(f"Downloaded spaCy model: {self.model_name}")
            
        except Exception as e:
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to download model {self.model_name}: {str(e)}",
                e
            )
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input and extract entities.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing NER results
        """
        start_time = time.time()
        
        try:
            result = await self.extract_entities(input_data.product_name)
            
            return {
                "agent_type": "ner",
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"NER processing failed: {str(e)}")
            
            return {
                "agent_type": "ner",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def extract_entities(self, product_name: str) -> NERResult:
        """
        Extract entities from product name using spaCy NER.
        
        Args:
            product_name: Input product name text
            
        Returns:
            NERResult with extracted entities and confidence scores
        """
        start_time = time.time()
        
        if not self.nlp:
            raise AgentError(self.agent_name, "Agent not initialized")
        
        # Validate and preprocess input
        cleaned_text = self._preprocess_text(product_name)
        
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = cleaned_text[:self.max_text_length]
            self.logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        # Run spaCy NER processing
        doc = self.nlp(cleaned_text)
        
        # Extract entities with confidence scoring
        entities = []
        for ent in doc.ents:
            entity_result = self._process_entity(ent, cleaned_text)
            if entity_result and entity_result.confidence >= self.confidence_threshold:
                entities.append(entity_result)
        
        # Calculate overall confidence based on entity quality and context
        overall_confidence = self._calculate_overall_confidence(entities, cleaned_text)
        
        processing_time = time.time() - start_time
        
        return NERResult(
            entities=entities,
            confidence=overall_confidence,
            processing_time=processing_time,
            model_used=self.model_name
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better NER performance.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common patterns that might confuse NER
        # Handle mixed Thai-English text patterns
        text = re.sub(r'([a-zA-Z])([ก-๙])', r'\1 \2', text)  # Space between English and Thai
        text = re.sub(r'([ก-๙])([a-zA-Z])', r'\1 \2', text)  # Space between Thai and English
        
        # Handle common product name patterns
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Space between letters and numbers
        
        return text
    
    def _process_entity(self, ent, text: str) -> Optional[EntityResult]:
        """
        Process individual spaCy entity and convert to EntityResult.
        
        Args:
            ent: spaCy entity object
            text: Original text for context
            
        Returns:
            EntityResult or None if entity should be filtered out
        """
        # Map spaCy entity type to our custom types
        entity_type = self.entity_mappings.get(ent.label_, None)
        
        if not entity_type:
            # Skip unmapped entity types
            return None
        
        # Calculate confidence based on entity characteristics
        confidence = self._calculate_entity_confidence(ent, text)
        
        # Filter out very short or low-quality entities
        if len(ent.text.strip()) < 2 or confidence < 0.1:
            return None
        
        return EntityResult(
            entity_type=entity_type,
            text=ent.text.strip(),
            confidence=confidence,
            start_pos=ent.start_char,
            end_pos=ent.end_char
        )
    
    def _calculate_entity_confidence(self, ent, text: str) -> float:
        """
        Calculate confidence score for individual entity.
        
        Args:
            ent: spaCy entity object
            text: Full text context
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.6  # Base confidence for spaCy entities
        
        # Boost confidence for certain patterns
        confidence_boosts = 0.0
        
        # Length-based confidence (longer entities often more reliable)
        if len(ent.text) >= 3:
            confidence_boosts += 0.1
        if len(ent.text) >= 6:
            confidence_boosts += 0.1
        
        # Position-based confidence (entities at start often brands)
        if ent.start_char < len(text) * 0.3:  # First 30% of text
            confidence_boosts += 0.1
        
        # Pattern-based confidence
        entity_lower = ent.text.lower()
        
        # Check for brand indicators in context
        context_window = 50
        start_pos = max(0, ent.start_char - context_window)
        end_pos = min(len(text), ent.end_char + context_window)
        context = text[start_pos:end_pos].lower()
        
        for lang_indicators in self.brand_indicators.values():
            for indicator in lang_indicators:
                if indicator in context:
                    confidence_boosts += 0.15
                    break
        
        # Capitalization patterns (proper nouns often brands)
        if ent.text[0].isupper() and len(ent.text) > 1:
            confidence_boosts += 0.1
        
        # Mixed language detection (common in Thai-English product names)
        has_thai = bool(re.search(r'[ก-๙]', ent.text))
        has_english = bool(re.search(r'[a-zA-Z]', ent.text))
        if has_thai and has_english:
            confidence_boosts += 0.1
        
        # Special characters that indicate brands
        if any(char in ent.text for char in ['®', '™', '©']):
            confidence_boosts += 0.2
        
        final_confidence = min(1.0, base_confidence + confidence_boosts)
        return final_confidence
    
    def _calculate_overall_confidence(self, entities: List[EntityResult], text: str) -> float:
        """
        Calculate overall confidence for the NER result.
        
        Args:
            entities: List of extracted entities
            text: Original text
            
        Returns:
            Overall confidence score between 0.0 and 1.0
        """
        if not entities:
            return 0.0
        
        # Base confidence from average entity confidence
        avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # Adjust based on number and quality of entities
        entity_count_factor = min(1.0, len(entities) / 3.0)  # Optimal around 3 entities
        
        # Boost if we found brand entities specifically
        brand_entities = [e for e in entities if e.entity_type == EntityType.BRAND]
        brand_boost = 0.2 if brand_entities else 0.0
        
        # Text coverage factor (entities should cover reasonable portion of text)
        total_entity_length = sum(len(e.text) for e in entities)
        coverage_ratio = total_entity_length / len(text) if text else 0
        coverage_factor = min(1.0, coverage_ratio * 2)  # Optimal around 50% coverage
        
        overall_confidence = (
            avg_entity_confidence * 0.5 +
            entity_count_factor * 0.2 +
            coverage_factor * 0.1 +
            brand_boost
        )
        
        return min(1.0, overall_confidence)
    
    async def cleanup(self) -> None:
        """Clean up spaCy resources."""
        if self.nlp:
            # spaCy doesn't require explicit cleanup, but we can clear the reference
            self.nlp = None
        
        self.set_initialized(False)
        self.logger.info("spaCy NER agent cleaned up")
    
    async def _perform_health_check(self) -> None:
        """Perform NER agent specific health check."""
        await super()._perform_health_check()
        
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded")
        
        # Test with a simple example
        test_text = "Samsung Galaxy S23"
        doc = self.nlp(test_text)
        
        if not hasattr(doc, 'ents'):
            raise RuntimeError("spaCy NER pipeline not functioning correctly")


class MultilingualNERAgent(SpacyNERAgent):
    """
    Enhanced NER agent with better multilingual support.
    
    Extends SpacyNERAgent with improved Thai-English mixed text processing
    and fallback to transformer-based models for better multilingual performance.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize multilingual NER agent.
        
        Args:
            config: Configuration dictionary with multilingual settings
        """
        super().__init__(config)
        self.agent_name = "multilingual_ner"
        
        # Transformer model fallback for better multilingual support
        self.use_transformer_fallback = self.get_config_value("use_transformer_fallback", False)
        self.transformer_model = self.get_config_value(
            "transformer_model", 
            "xlm-roberta-base"
        )
        
        # Language-specific processing settings
        self.thai_text_threshold = self.get_config_value("thai_text_threshold", 0.3)
        self.mixed_language_boost = self.get_config_value("mixed_language_boost", 0.15)
    
    async def extract_entities(self, product_name: str) -> NERResult:
        """
        Extract entities with enhanced multilingual processing.
        
        Args:
            product_name: Input product name text
            
        Returns:
            NERResult with multilingual entity extraction
        """
        # Detect language composition
        language_info = self._analyze_language_composition(product_name)
        
        # Use base spaCy processing
        result = await super().extract_entities(product_name)
        
        # Apply multilingual enhancements
        enhanced_entities = self._enhance_multilingual_entities(
            result.entities, 
            product_name, 
            language_info
        )
        
        # Adjust confidence based on multilingual factors
        enhanced_confidence = self._adjust_multilingual_confidence(
            result.confidence, 
            language_info
        )
        
        return NERResult(
            entities=enhanced_entities,
            confidence=enhanced_confidence,
            processing_time=result.processing_time,
            model_used=f"{result.model_used}_multilingual"
        )
    
    def _analyze_language_composition(self, text: str) -> Dict[str, Any]:
        """
        Analyze language composition of the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language composition information
        """
        thai_chars = len(re.findall(r'[ก-๙]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s+', '', text))
        
        thai_ratio = thai_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        is_mixed = thai_ratio > 0.1 and english_ratio > 0.1
        is_primarily_thai = thai_ratio > self.thai_text_threshold
        is_primarily_english = english_ratio > 0.7
        
        return {
            "thai_ratio": thai_ratio,
            "english_ratio": english_ratio,
            "is_mixed": is_mixed,
            "is_primarily_thai": is_primarily_thai,
            "is_primarily_english": is_primarily_english,
            "total_chars": total_chars
        }
    
    def _enhance_multilingual_entities(
        self, 
        entities: List[EntityResult], 
        text: str, 
        language_info: Dict[str, Any]
    ) -> List[EntityResult]:
        """
        Enhance entity extraction for multilingual text.
        
        Args:
            entities: Original entities from spaCy
            text: Original text
            language_info: Language composition analysis
            
        Returns:
            Enhanced list of entities
        """
        enhanced_entities = []
        
        for entity in entities:
            # Boost confidence for entities in mixed-language context
            if language_info["is_mixed"]:
                entity.confidence = min(1.0, entity.confidence + self.mixed_language_boost)
            
            # Apply language-specific enhancements
            if language_info["is_primarily_thai"]:
                entity = self._enhance_thai_entity(entity, text)
            elif language_info["is_mixed"]:
                entity = self._enhance_mixed_language_entity(entity, text)
            
            enhanced_entities.append(entity)
        
        return enhanced_entities
    
    def _enhance_thai_entity(self, entity: EntityResult, text: str) -> EntityResult:
        """
        Apply Thai-specific entity enhancements.
        
        Args:
            entity: Original entity result
            text: Full text context
            
        Returns:
            Enhanced entity result
        """
        # Thai brand names often have specific patterns
        thai_brand_patterns = [
            r'[ก-๙]+\s*[a-zA-Z]+',  # Thai followed by English
            r'[a-zA-Z]+\s*[ก-๙]+',  # English followed by Thai
        ]
        
        for pattern in thai_brand_patterns:
            if re.search(pattern, entity.text):
                entity.confidence = min(1.0, entity.confidence + 0.1)
                break
        
        return entity
    
    def _enhance_mixed_language_entity(self, entity: EntityResult, text: str) -> EntityResult:
        """
        Apply mixed-language entity enhancements.
        
        Args:
            entity: Original entity result
            text: Full text context
            
        Returns:
            Enhanced entity result
        """
        # Mixed language entities are often transliterations or brand names
        has_thai = bool(re.search(r'[ก-๙]', entity.text))
        has_english = bool(re.search(r'[a-zA-Z]', entity.text))
        
        if has_thai and has_english:
            # This is likely a transliterated brand name
            entity.confidence = min(1.0, entity.confidence + 0.15)
            if entity.entity_type != EntityType.BRAND:
                entity.entity_type = EntityType.BRAND
        
        return entity
    
    def _adjust_multilingual_confidence(
        self, 
        base_confidence: float, 
        language_info: Dict[str, Any]
    ) -> float:
        """
        Adjust overall confidence based on multilingual factors.
        
        Args:
            base_confidence: Original confidence score
            language_info: Language composition analysis
            
        Returns:
            Adjusted confidence score
        """
        adjusted_confidence = base_confidence
        
        # Mixed language text often indicates product names with brands
        if language_info["is_mixed"]:
            adjusted_confidence = min(1.0, adjusted_confidence + 0.1)
        
        # Very short text might be less reliable
        if language_info["total_chars"] < 10:
            adjusted_confidence *= 0.8
        
        return adjusted_confidence