"""
Hybrid Agent implementation with sequential processing.

This module implements sequential pipeline processing (NER → RAG → LLM) following
PEP 8 coding standards and referencing patterns from the notebook implementation
for result combination and confidence aggregation.
"""

import time
import asyncio
import re
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..models.data_models import (
    ProductInput,
    HybridResult,
    NERResult,
    RAGResult,
    LLMResult,
    EntityType
)
from .base_agent import HybridAgent, NERAgent, RAGAgent, LLMAgent, AgentError, AgentInitializationError


class SequentialHybridAgent(HybridAgent):
    """
    Sequential hybrid agent implementing NER → RAG → LLM pipeline.
    
    Processes product names through multiple inference stages, combining results
    with confidence-based weighting and enhanced context passing between stages.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize sequential hybrid agent.
        
        Args:
            config: Configuration dictionary containing pipeline settings
        """
        super().__init__("sequential_hybrid", config)
        
        # Pipeline configuration
        self.enable_ner_stage = self.get_config_value("enable_ner_stage", True)
        self.enable_rag_stage = self.get_config_value("enable_rag_stage", True)
        self.enable_llm_stage = self.get_config_value("enable_llm_stage", True)
        
        # Confidence thresholds for early termination
        self.ner_confidence_threshold = self.get_config_value("ner_confidence_threshold", 0.8)
        self.rag_confidence_threshold = self.get_config_value("rag_confidence_threshold", 0.8)
        
        # Contribution weights for final confidence calculation
        self.ner_weight = self.get_config_value("ner_weight", 0.3)
        self.rag_weight = self.get_config_value("rag_weight", 0.4)
        self.llm_weight = self.get_config_value("llm_weight", 0.3)
        
        # Pipeline optimization settings
        self.use_early_termination = self.get_config_value("use_early_termination", False)
        self.use_context_enhancement = self.get_config_value("use_context_enhancement", True)
        self.max_pipeline_time = self.get_config_value("max_pipeline_time", 60.0)
        
        # Agent references (will be set during processing)
        self.current_ner_agent = None
        self.current_rag_agent = None
        self.current_llm_agent = None
    
    async def initialize(self) -> None:
        """Initialize hybrid agent (no specific resources needed)."""
        try:
            # Validate configuration
            self._validate_configuration()
            
            self.set_initialized(True)
            self.logger.info("Sequential hybrid agent initialized successfully")
            
        except Exception as e:
            self.set_initialized(False)
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to initialize hybrid agent: {str(e)}",
                e
            )
    
    def _validate_configuration(self) -> None:
        """Validate hybrid agent configuration."""
        # Ensure at least one stage is enabled
        if not any([self.enable_ner_stage, self.enable_rag_stage, self.enable_llm_stage]):
            raise ValueError("At least one pipeline stage must be enabled")
        
        # Validate weights sum to approximately 1.0
        total_weight = self.ner_weight + self.rag_weight + self.llm_weight
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Pipeline weights must sum to 1.0, got {total_weight}")
        
        # Validate thresholds
        if not (0.0 <= self.ner_confidence_threshold <= 1.0):
            raise ValueError("NER confidence threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.rag_confidence_threshold <= 1.0):
            raise ValueError("RAG confidence threshold must be between 0.0 and 1.0")
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input through hybrid pipeline.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing hybrid results
        """
        start_time = time.time()
        
        try:
            # Use empty agent dict since we'll receive agents as parameters
            result = await self.hybrid_inference(input_data.product_name)
            
            return {
                "agent_type": "hybrid",
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Hybrid processing failed: {str(e)}")
            
            return {
                "agent_type": "hybrid",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def hybrid_inference(
        self,
        product_name: str,
        ner_agent: Optional[NERAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        llm_agent: Optional[LLMAgent] = None
    ) -> HybridResult:
        """
        Perform hybrid inference using sequential pipeline processing.
        
        Args:
            product_name: Input product name text
            ner_agent: Optional NER agent for pipeline
            rag_agent: Optional RAG agent for pipeline
            llm_agent: Optional LLM agent for pipeline
            
        Returns:
            HybridResult with combined inference results
        """
        start_time = time.time()
        
        if not self._is_initialized:
            raise AgentError(self.agent_name, "Agent not initialized")
        
        # Store agent references
        self.current_ner_agent = ner_agent
        self.current_rag_agent = rag_agent
        self.current_llm_agent = llm_agent
        
        # Initialize pipeline state
        pipeline_steps = []
        context = ""
        
        # Stage results
        ner_result = None
        rag_result = None
        llm_result = None
        
        try:
            # Stage 1: NER Processing
            if self.enable_ner_stage and ner_agent:
                ner_result, context = await self._execute_ner_stage(
                    product_name, ner_agent, context
                )
                pipeline_steps.append("NER")
                
                # Early termination check
                if (self.use_early_termination and 
                    ner_result and 
                    ner_result.confidence >= self.ner_confidence_threshold):
                    
                    brand = self._extract_brand_from_ner(ner_result)
                    if brand != "Unknown":
                        return self._create_early_termination_result(
                            brand, ner_result, None, None, pipeline_steps, start_time
                        )
            
            # Stage 2: RAG Processing
            if self.enable_rag_stage and rag_agent:
                rag_result, context = await self._execute_rag_stage(
                    product_name, rag_agent, context
                )
                pipeline_steps.append("RAG")
                
                # Early termination check
                if (self.use_early_termination and 
                    rag_result and 
                    rag_result.confidence >= self.rag_confidence_threshold):
                    
                    return self._create_early_termination_result(
                        rag_result.predicted_brand, ner_result, rag_result, None, 
                        pipeline_steps, start_time
                    )
            
            # Stage 3: LLM Processing
            if self.enable_llm_stage and llm_agent:
                llm_result, context = await self._execute_llm_stage(
                    product_name, llm_agent, context
                )
                pipeline_steps.append("LLM")
            
            # Combine results from all stages
            final_result = self._combine_pipeline_results(
                ner_result, rag_result, llm_result, pipeline_steps, start_time
            )
            
            return final_result
            
        except asyncio.TimeoutError:
            raise AgentError(
                self.agent_name,
                f"Pipeline processing timed out after {self.max_pipeline_time} seconds"
            )
        except Exception as e:
            raise AgentError(
                self.agent_name,
                f"Pipeline processing failed: {str(e)}"
            )
    
    async def _execute_ner_stage(
        self, 
        product_name: str, 
        ner_agent: NERAgent, 
        context: str
    ) -> Tuple[Optional[NERResult], str]:
        """
        Execute NER stage of the pipeline.
        
        Args:
            product_name: Input product name
            ner_agent: NER agent instance
            context: Current pipeline context
            
        Returns:
            Tuple of (NER result, updated context)
        """
        try:
            self.logger.debug("Executing NER stage")
            
            # Execute NER inference
            ner_result = await ner_agent.extract_entities(product_name)
            
            # Update context with NER findings
            if self.use_context_enhancement:
                context = self._enhance_context_with_ner(context, ner_result)
            
            self.logger.debug(f"NER stage completed with confidence: {ner_result.confidence}")
            return ner_result, context
            
        except Exception as e:
            self.logger.warning(f"NER stage failed: {str(e)}")
            return None, context
    
    async def _execute_rag_stage(
        self, 
        product_name: str, 
        rag_agent: RAGAgent, 
        context: str
    ) -> Tuple[Optional[RAGResult], str]:
        """
        Execute RAG stage of the pipeline.
        
        Args:
            product_name: Input product name
            rag_agent: RAG agent instance
            context: Current pipeline context
            
        Returns:
            Tuple of (RAG result, updated context)
        """
        try:
            self.logger.debug("Executing RAG stage")
            
            # Execute RAG inference
            rag_result = await rag_agent.retrieve_and_infer(product_name)
            
            # Update context with RAG findings
            if self.use_context_enhancement:
                context = self._enhance_context_with_rag(context, rag_result)
            
            self.logger.debug(f"RAG stage completed with confidence: {rag_result.confidence}")
            return rag_result, context
            
        except Exception as e:
            self.logger.warning(f"RAG stage failed: {str(e)}")
            return None, context
    
    async def _execute_llm_stage(
        self, 
        product_name: str, 
        llm_agent: LLMAgent, 
        context: str
    ) -> Tuple[Optional[LLMResult], str]:
        """
        Execute LLM stage of the pipeline.
        
        Args:
            product_name: Input product name
            llm_agent: LLM agent instance
            context: Current pipeline context
            
        Returns:
            Tuple of (LLM result, updated context)
        """
        try:
            self.logger.debug("Executing LLM stage")
            
            # Execute LLM inference with enhanced context
            llm_result = await llm_agent.infer_brand(product_name, context)
            
            self.logger.debug(f"LLM stage completed with confidence: {llm_result.confidence}")
            return llm_result, context
            
        except Exception as e:
            self.logger.warning(f"LLM stage failed: {str(e)}")
            return None, context
    
    def _enhance_context_with_ner(self, context: str, ner_result: NERResult) -> str:
        """
        Enhance pipeline context with NER findings.
        
        Args:
            context: Current context string
            ner_result: NER processing result
            
        Returns:
            Enhanced context string
        """
        if not ner_result or not ner_result.entities:
            return context
        
        # Extract brand entities
        brand_entities = [e for e in ner_result.entities if e.entity_type == EntityType.BRAND]
        
        if brand_entities:
            brands = [e.text for e in brand_entities]
            brand_context = f"Detected brand entities: {', '.join(brands)}. "
            context = f"{context}{brand_context}".strip()
        
        # Add other relevant entities
        other_entities = [e for e in ner_result.entities if e.entity_type != EntityType.BRAND]
        if other_entities:
            entities_text = [f"{e.entity_type.value}: {e.text}" for e in other_entities]
            entity_context = f"Other entities: {', '.join(entities_text)}. "
            context = f"{context}{entity_context}".strip()
        
        return context
    
    def _enhance_context_with_rag(self, context: str, rag_result: RAGResult) -> str:
        """
        Enhance pipeline context with RAG findings.
        
        Args:
            context: Current context string
            rag_result: RAG processing result
            
        Returns:
            Enhanced context string
        """
        if not rag_result:
            return context
        
        # Add predicted brand from RAG
        if rag_result.predicted_brand != "Unknown":
            brand_context = f"RAG suggests brand: {rag_result.predicted_brand}. "
            context = f"{context}{brand_context}".strip()
        
        # Add similar products information
        if rag_result.similar_products:
            top_products = rag_result.similar_products[:3]  # Top 3 similar products
            similar_brands = list(set([p.brand for p in top_products if p.brand != "Unknown"]))
            
            if similar_brands:
                similar_context = f"Similar products from brands: {', '.join(similar_brands)}. "
                context = f"{context}{similar_context}".strip()
        
        return context
    
    def _extract_brand_from_ner(self, ner_result: NERResult) -> str:
        """
        Extract the most confident brand from NER results.
        
        Args:
            ner_result: NER processing result
            
        Returns:
            Brand name or "Unknown"
        """
        if not ner_result or not ner_result.entities:
            return "Unknown"
        
        # Find brand entities
        brand_entities = [e for e in ner_result.entities if e.entity_type == EntityType.BRAND]
        
        if not brand_entities:
            return "Unknown"
        
        # Return the brand with highest confidence
        best_brand = max(brand_entities, key=lambda e: e.confidence)
        return best_brand.text
    
    def _create_early_termination_result(
        self,
        predicted_brand: str,
        ner_result: Optional[NERResult],
        rag_result: Optional[RAGResult],
        llm_result: Optional[LLMResult],
        pipeline_steps: List[str],
        start_time: float
    ) -> HybridResult:
        """
        Create hybrid result for early termination scenarios.
        
        Args:
            predicted_brand: Final predicted brand
            ner_result: NER result (if available)
            rag_result: RAG result (if available)
            llm_result: LLM result (if available)
            pipeline_steps: Completed pipeline steps
            start_time: Pipeline start time
            
        Returns:
            HybridResult for early termination
        """
        processing_time = time.time() - start_time
        
        # Calculate contributions based on available results
        ner_contribution = 1.0 if ner_result else 0.0
        rag_contribution = 1.0 if rag_result else 0.0
        llm_contribution = 0.0  # LLM not executed in early termination
        
        # Normalize contributions
        total_contribution = ner_contribution + rag_contribution + llm_contribution
        if total_contribution > 0:
            ner_contribution /= total_contribution
            rag_contribution /= total_contribution
            llm_contribution /= total_contribution
        
        # Use the confidence from the terminating stage
        if rag_result:
            confidence = rag_result.confidence
        elif ner_result:
            confidence = ner_result.confidence
        else:
            confidence = 0.5
        
        return HybridResult(
            predicted_brand=predicted_brand,
            ner_contribution=ner_contribution,
            rag_contribution=rag_contribution,
            llm_contribution=llm_contribution,
            confidence=confidence,
            processing_time=processing_time,
            pipeline_steps=pipeline_steps
        )
    
    def _combine_pipeline_results(
        self,
        ner_result: Optional[NERResult],
        rag_result: Optional[RAGResult],
        llm_result: Optional[LLMResult],
        pipeline_steps: List[str],
        start_time: float
    ) -> HybridResult:
        """
        Combine results from all pipeline stages.
        
        Args:
            ner_result: NER processing result
            rag_result: RAG processing result
            llm_result: LLM processing result
            pipeline_steps: Completed pipeline steps
            start_time: Pipeline start time
            
        Returns:
            Combined HybridResult
        """
        processing_time = time.time() - start_time
        
        # Collect brand candidates with their confidence scores
        candidates = []
        
        if ner_result:
            brand = self._extract_brand_from_ner(ner_result)
            if brand != "Unknown":
                candidates.append({
                    "brand": brand,
                    "confidence": ner_result.confidence,
                    "weight": self.ner_weight,
                    "source": "NER"
                })
        
        if rag_result and rag_result.predicted_brand != "Unknown":
            candidates.append({
                "brand": rag_result.predicted_brand,
                "confidence": rag_result.confidence,
                "weight": self.rag_weight,
                "source": "RAG"
            })
        
        if llm_result and llm_result.predicted_brand != "Unknown":
            candidates.append({
                "brand": llm_result.predicted_brand,
                "confidence": llm_result.confidence,
                "weight": self.llm_weight,
                "source": "LLM"
            })
        
        # Select best brand and calculate final confidence
        if not candidates:
            predicted_brand = "Unknown"
            final_confidence = 0.0
        else:
            # Calculate weighted confidence for each candidate
            brand_scores = {}
            for candidate in candidates:
                brand = candidate["brand"]
                weighted_score = candidate["confidence"] * candidate["weight"]
                
                if brand in brand_scores:
                    brand_scores[brand] += weighted_score
                else:
                    brand_scores[brand] = weighted_score
            
            # Select brand with highest weighted score
            predicted_brand = max(brand_scores.keys(), key=lambda b: brand_scores[b])
            final_confidence = brand_scores[predicted_brand]
        
        # Calculate stage contributions
        ner_contribution = self.ner_weight if ner_result else 0.0
        rag_contribution = self.rag_weight if rag_result else 0.0
        llm_contribution = self.llm_weight if llm_result else 0.0
        
        # Normalize contributions based on executed stages
        total_executed_weight = ner_contribution + rag_contribution + llm_contribution
        if total_executed_weight > 0:
            ner_contribution /= total_executed_weight
            rag_contribution /= total_executed_weight
            llm_contribution /= total_executed_weight
        else:
            # If no stages executed, set equal contributions that sum to 1.0
            ner_contribution = 1.0 / 3.0
            rag_contribution = 1.0 / 3.0
            llm_contribution = 1.0 / 3.0
        
        return HybridResult(
            predicted_brand=predicted_brand,
            ner_contribution=ner_contribution,
            rag_contribution=rag_contribution,
            llm_contribution=llm_contribution,
            confidence=final_confidence,
            processing_time=processing_time,
            pipeline_steps=pipeline_steps
        )
    
    async def cleanup(self) -> None:
        """Clean up hybrid agent resources."""
        # Clear agent references
        self.current_ner_agent = None
        self.current_rag_agent = None
        self.current_llm_agent = None
        
        self.set_initialized(False)
        self.logger.info("Sequential hybrid agent cleaned up")
    
    async def _perform_health_check(self) -> None:
        """Perform hybrid agent specific health check."""
        await super()._perform_health_check()
        
        # Validate configuration is still valid
        try:
            self._validate_configuration()
        except Exception as e:
            raise RuntimeError(f"Configuration validation failed: {e}")


class OptimizedHybridAgent(SequentialHybridAgent):
    """
    Optimized hybrid agent with enhanced decision making.
    
    Extends the sequential hybrid agent with improved stage selection,
    dynamic thresholding, and performance optimizations.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize optimized hybrid agent.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        super().__init__(config)
        self.agent_name = "optimized_hybrid"
        
        # Optimization settings
        self.use_dynamic_thresholds = self.get_config_value("use_dynamic_thresholds", True)
        self.use_stage_selection = self.get_config_value("use_stage_selection", True)
        self.performance_mode = self.get_config_value("performance_mode", "balanced")  # fast, balanced, accurate
        
        # Dynamic threshold parameters
        self.min_confidence_threshold = self.get_config_value("min_confidence_threshold", 0.6)
        self.max_confidence_threshold = self.get_config_value("max_confidence_threshold", 0.9)
        
        # Performance mode configurations
        self._configure_performance_mode()
    
    def _configure_performance_mode(self) -> None:
        """Configure agent based on performance mode."""
        if self.performance_mode == "fast":
            # Prioritize speed
            self.use_early_termination = True
            self.ner_confidence_threshold = 0.7
            self.rag_confidence_threshold = 0.7
            self.max_pipeline_time = 10.0
            
        elif self.performance_mode == "accurate":
            # Prioritize accuracy
            self.use_early_termination = False
            self.ner_confidence_threshold = 0.95
            self.rag_confidence_threshold = 0.95
            self.max_pipeline_time = 120.0
            
        else:  # balanced
            # Balance speed and accuracy
            self.use_early_termination = True
            self.ner_confidence_threshold = 0.8
            self.rag_confidence_threshold = 0.8
            self.max_pipeline_time = 60.0
    
    async def hybrid_inference(
        self,
        product_name: str,
        ner_agent: Optional[NERAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        llm_agent: Optional[LLMAgent] = None
    ) -> HybridResult:
        """
        Perform optimized hybrid inference with dynamic stage selection.
        
        Args:
            product_name: Input product name text
            ner_agent: Optional NER agent for pipeline
            rag_agent: Optional RAG agent for pipeline
            llm_agent: Optional LLM agent for pipeline
            
        Returns:
            Optimized HybridResult
        """
        # Apply stage selection optimization
        if self.use_stage_selection:
            selected_stages = self._select_optimal_stages(product_name)
            
            # Temporarily override stage settings
            original_ner = self.enable_ner_stage
            original_rag = self.enable_rag_stage
            original_llm = self.enable_llm_stage
            
            self.enable_ner_stage = selected_stages["ner"]
            self.enable_rag_stage = selected_stages["rag"]
            self.enable_llm_stage = selected_stages["llm"]
            
            try:
                result = await super().hybrid_inference(product_name, ner_agent, rag_agent, llm_agent)
                
                # Apply post-processing optimizations
                optimized_result = self._optimize_result(result, product_name)
                return optimized_result
                
            finally:
                # Restore original settings
                self.enable_ner_stage = original_ner
                self.enable_rag_stage = original_rag
                self.enable_llm_stage = original_llm
        else:
            return await super().hybrid_inference(product_name, ner_agent, rag_agent, llm_agent)
    
    def _select_optimal_stages(self, product_name: str) -> Dict[str, bool]:
        """
        Select optimal pipeline stages based on input characteristics.
        
        Args:
            product_name: Input product name
            
        Returns:
            Dictionary indicating which stages to enable
        """
        # Analyze input characteristics
        text_length = len(product_name)
        has_mixed_language = self._detect_mixed_language(product_name)
        has_special_chars = bool(re.search(r'[®™©]', product_name))
        
        # Default: enable all stages
        stages = {"ner": True, "rag": True, "llm": True}
        
        # Optimization rules based on input characteristics
        if text_length < 20 and not has_mixed_language:
            # Short, simple text - NER might be sufficient
            if self.performance_mode == "fast":
                stages["llm"] = False
        
        if has_special_chars:
            # Text with brand indicators - NER likely to be effective
            stages["ner"] = True
        
        if has_mixed_language:
            # Mixed language - LLM likely most effective
            stages["llm"] = True
            if self.performance_mode == "fast":
                stages["ner"] = False
        
        return stages
    
    def _detect_mixed_language(self, text: str) -> bool:
        """
        Detect if text contains mixed languages.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if mixed language detected
        """
        import re
        
        has_thai = bool(re.search(r'[ก-๙]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        return has_thai and has_english
    
    def _optimize_result(self, result: HybridResult, product_name: str) -> HybridResult:
        """
        Apply post-processing optimizations to the result.
        
        Args:
            result: Original hybrid result
            product_name: Original product name
            
        Returns:
            Optimized hybrid result
        """
        # Apply dynamic confidence adjustment
        if self.use_dynamic_thresholds:
            adjusted_confidence = self._adjust_confidence_dynamically(
                result.confidence, 
                product_name,
                result.pipeline_steps
            )
            
            # Create optimized result
            return HybridResult(
                predicted_brand=result.predicted_brand,
                ner_contribution=result.ner_contribution,
                rag_contribution=result.rag_contribution,
                llm_contribution=result.llm_contribution,
                confidence=adjusted_confidence,
                processing_time=result.processing_time,
                pipeline_steps=result.pipeline_steps
            )
        
        return result
    
    def _adjust_confidence_dynamically(
        self, 
        base_confidence: float, 
        product_name: str, 
        pipeline_steps: List[str]
    ) -> float:
        """
        Dynamically adjust confidence based on context.
        
        Args:
            base_confidence: Original confidence score
            product_name: Input product name
            pipeline_steps: Executed pipeline steps
            
        Returns:
            Adjusted confidence score
        """
        adjusted_confidence = base_confidence
        
        # Boost confidence if multiple stages agree
        if len(pipeline_steps) >= 2:
            adjusted_confidence = min(1.0, adjusted_confidence + 0.1)
        
        # Reduce confidence for very short or very long text
        text_length = len(product_name)
        if text_length < 10 or text_length > 200:
            adjusted_confidence *= 0.9
        
        # Boost confidence for text with clear brand indicators
        if any(indicator in product_name.lower() for indicator in ['brand', 'แบรนด์', '®', '™']):
            adjusted_confidence = min(1.0, adjusted_confidence + 0.05)
        
        return adjusted_confidence