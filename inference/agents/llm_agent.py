"""
Large Language Model (LLM) Agent implementation using AWS Bedrock.

This module implements Nova Pro inference client using AWS Bedrock SDK with PEP 8
compliant code, referencing patterns from the notebook implementation for multilingual
brand extraction with proper prompt engineering.
"""

import json
import time
import re
import asyncio
from typing import Dict, Any, Optional, List
import logging

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
except ImportError:
    boto3 = None
    ClientError = Exception
    BotoCoreError = Exception

from ..models.data_models import (
    ProductInput,
    LLMResult,
    LanguageHint
)
from .base_agent import LLMAgent, AgentError, AgentInitializationError


class BedrockLLMAgent(LLMAgent):
    """
    AWS Bedrock LLM agent using Nova Pro for brand inference.
    
    Implements multilingual brand extraction using fine-tuned Nova Pro models
    via AWS Bedrock, following patterns from the reference notebook.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Bedrock LLM agent.
        
        Args:
            config: Configuration dictionary containing AWS and model settings
        """
        super().__init__("bedrock_llm", config)
        
        # AWS configuration
        self.aws_profile = self.get_config_value("aws_profile", "ml-sandbox")
        self.aws_region = self.get_config_value("aws_region", "us-east-1")
        
        # Model configuration
        self.model_id = self.get_config_value("model_id", "amazon.nova-pro-v1:0")
        self.max_tokens = self.get_config_value("max_tokens", 1000)
        self.temperature = self.get_config_value("temperature", 0.1)
        self.top_p = self.get_config_value("top_p", 0.9)
        
        # Inference configuration
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.5)
        self.max_text_length = self.get_config_value("max_text_length", 1000)
        self.timeout_seconds = self.get_config_value("timeout_seconds", 30)
        
        # Initialize AWS clients
        self.bedrock_runtime = None
        self.session = None
        
        # Prompt templates for different scenarios
        self.base_prompt_template = self._get_base_prompt_template()
        self.multilingual_prompt_template = self._get_multilingual_prompt_template()
    
    def _get_base_prompt_template(self) -> str:
        """
        Get base prompt template for brand extraction.
        
        Returns:
            Formatted prompt template string
        """
        return """<system>
You are a helpful assistant that helps extract brand names from product titles.

Your task is to identify the brand name from the given product title. Follow these guidelines:
1. Extract only the brand name, not product models or variants
2. Return "Unknown" if no clear brand can be identified
3. For mixed language titles, identify the brand regardless of language
4. Be consistent with brand name formatting (e.g., "Samsung" not "samsung")
5. If multiple brands appear, return the primary/main brand

Examples:
- "Samsung Galaxy S23" → "Samsung"
- "iPhone 15 Pro Max" → "Apple"
- "Sony WH-1000XM4 Headphones" → "Sony"
- "Generic USB Cable" → "Unknown"
</system>

<question>
Extract the brand name from this product title: "{product_name}"

Respond with only the brand name, nothing else.
</question>"""
    
    def _get_multilingual_prompt_template(self) -> str:
        """
        Get multilingual prompt template for Thai-English mixed text.
        
        Returns:
            Multilingual prompt template string
        """
        return """<system>
You are a helpful assistant that helps extract brand names from multilingual product titles, especially Thai-English mixed text.

Your task is to identify the brand name from the given product title. Follow these guidelines:
1. Handle Thai-English mixed text (e.g., "Samsung โทรศัพท์" or "ยาสีฟัน Wonder Smile")
2. Extract only the brand name, not product categories or descriptions
3. Return "Unknown" if no clear brand can be identified
4. Recognize transliterations (e.g., "ซัมซุง" = "Samsung")
5. Be consistent with international brand name formatting

Examples:
- "Samsung Galaxy โทรศัพท์มือถือ" → "Samsung"
- "ยาสีฟัน Wonder Smile toothpaste kid วันเดอร์สไมล์" → "Wonder Smile"
- "BENQ GW2785TC 27นิ้ว FHD Monitor" → "BENQ"
- "โทรศัพท์ทั่วไป" → "Unknown"
</system>

<question>
Extract the brand name from this multilingual product title: "{product_name}"

The title may contain Thai and English text mixed together. Focus on identifying the brand name.

Respond with only the brand name, nothing else.
</question>"""
    
    async def initialize(self) -> None:
        """Initialize AWS Bedrock client and validate configuration."""
        try:
            if boto3 is None:
                raise AgentInitializationError(
                    self.agent_name,
                    "boto3 library not installed. Please install with: pip install boto3"
                )
            
            # Initialize AWS session with profile
            self.logger.info(f"Initializing AWS session with profile: {self.aws_profile}")
            self.session = boto3.Session(profile_name=self.aws_profile)
            
            # Initialize Bedrock Runtime client
            self.bedrock_runtime = self.session.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region
            )
            
            # Validate AWS credentials and model access
            await self._validate_aws_access()
            
            self.set_initialized(True)
            self.logger.info("Bedrock LLM agent initialized successfully")
            
        except Exception as e:
            self.set_initialized(False)
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to initialize Bedrock LLM agent: {str(e)}",
                e
            )
    
    async def _validate_aws_access(self) -> None:
        """Validate AWS credentials and Bedrock model access."""
        try:
            # Test AWS credentials by making a simple call
            loop = asyncio.get_event_loop()
            
            # Create a minimal test payload
            test_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "Hello"}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 10,
                    "temperature": 0.1
                }
            }
            
            # Test model access with a simple call
            await loop.run_in_executor(
                None,
                lambda: self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(test_payload)
                )
            )
            
            self.logger.info(f"Successfully validated access to model: {self.model_id}")
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDeniedException':
                raise AgentInitializationError(
                    self.agent_name,
                    f"Access denied to Bedrock model {self.model_id}. "
                    f"Check AWS permissions and model availability."
                )
            elif error_code == 'ValidationException':
                raise AgentInitializationError(
                    self.agent_name,
                    f"Invalid model ID: {self.model_id}. "
                    f"Please check the model ID and region."
                )
            else:
                raise AgentInitializationError(
                    self.agent_name,
                    f"AWS Bedrock validation failed: {str(e)}"
                )
        except Exception as e:
            raise AgentInitializationError(
                self.agent_name,
                f"Failed to validate AWS Bedrock access: {str(e)}"
            )
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input and perform LLM inference.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing LLM results
        """
        start_time = time.time()
        
        try:
            result = await self.infer_brand(input_data.product_name)
            
            return {
                "agent_type": "llm",
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"LLM processing failed: {str(e)}")
            
            return {
                "agent_type": "llm",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def infer_brand(self, product_name: str, context: Optional[str] = None) -> LLMResult:
        """
        Infer brand using Nova Pro model via AWS Bedrock.
        
        Args:
            product_name: Input product name text
            context: Optional context for enhanced inference
            
        Returns:
            LLMResult with brand prediction and reasoning
        """
        start_time = time.time()
        
        if not self.bedrock_runtime:
            raise AgentError(self.agent_name, "Agent not initialized")
        
        # Preprocess and validate input
        cleaned_text = self._preprocess_text(product_name)
        
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = cleaned_text[:self.max_text_length]
            self.logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        # Select appropriate prompt template
        prompt = self._build_prompt(cleaned_text, context)
        
        # Create Bedrock payload
        payload = self._create_bedrock_payload(prompt)
        
        # Invoke model with timeout
        response_data = await self._invoke_model_with_timeout(payload)
        
        # Extract and process response
        predicted_brand, reasoning = self._extract_brand_from_response(response_data)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(predicted_brand, reasoning, cleaned_text)
        
        processing_time = time.time() - start_time
        
        return LLMResult(
            predicted_brand=predicted_brand,
            reasoning=reasoning,
            confidence=confidence,
            processing_time=processing_time,
            model_id=self.model_id
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for LLM inference.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle special characters that might confuse the model
        text = re.sub(r'[^\w\s\u0E00-\u0E7F\-\(\)\.\,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_language_composition(self, text: str) -> Dict[str, Any]:
        """
        Detect language composition of the text.
        
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
        is_primarily_thai = thai_ratio > 0.5
        
        return {
            "thai_ratio": thai_ratio,
            "english_ratio": english_ratio,
            "is_mixed": is_mixed,
            "is_primarily_thai": is_primarily_thai,
            "total_chars": total_chars
        }
    
    def _build_prompt(self, product_name: str, context: Optional[str] = None) -> str:
        """
        Build appropriate prompt based on text characteristics.
        
        Args:
            product_name: Cleaned product name
            context: Optional additional context
            
        Returns:
            Formatted prompt string
        """
        # Detect language composition
        lang_info = self._detect_language_composition(product_name)
        
        # Choose template based on language composition
        if lang_info["is_mixed"] or lang_info["is_primarily_thai"]:
            template = self.multilingual_prompt_template
        else:
            template = self.base_prompt_template
        
        # Format prompt with product name
        prompt = template.format(product_name=product_name)
        
        # Add context if provided
        if context:
            context_section = f"\n\nAdditional Context:\n{context}\n"
            # Insert context before the question
            prompt = prompt.replace("<question>", f"{context_section}<question>")
        
        return prompt
    
    def _create_bedrock_payload(self, prompt: str) -> Dict[str, Any]:
        """
        Create Bedrock API payload following notebook patterns.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Bedrock API payload dictionary
        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p
            }
        }
    
    async def _invoke_model_with_timeout(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Bedrock model with timeout handling.
        
        Args:
            payload: Bedrock API payload
            
        Returns:
            Response data from Bedrock
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Invoke model with timeout
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.bedrock_runtime.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(payload)
                    )
                ),
                timeout=self.timeout_seconds
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read().decode("utf-8"))
            return response_body
            
        except asyncio.TimeoutError:
            raise AgentError(
                self.agent_name,
                f"Model inference timed out after {self.timeout_seconds} seconds"
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise AgentError(
                self.agent_name,
                f"Bedrock API error ({error_code}): {error_message}"
            )
        except Exception as e:
            raise AgentError(
                self.agent_name,
                f"Model invocation failed: {str(e)}"
            )
    
    def _extract_brand_from_response(self, response_data: Dict[str, Any]) -> tuple[str, str]:
        """
        Extract brand name and reasoning from Bedrock response.
        
        Args:
            response_data: Response data from Bedrock
            
        Returns:
            Tuple of (predicted_brand, reasoning)
        """
        try:
            # Extract content from response following Nova response structure
            content = response_data.get("output", {}).get("message", {}).get("content", [])
            
            if not content:
                return "Unknown", "No response content received"
            
            # Get text from first content item
            response_text = content[0].get("text", "").strip()
            
            if not response_text:
                return "Unknown", "Empty response received"
            
            # Extract brand name from response
            predicted_brand = self._parse_brand_from_text(response_text)
            
            # Use the full response as reasoning
            reasoning = response_text
            
            return predicted_brand, reasoning
            
        except (KeyError, IndexError, AttributeError) as e:
            self.logger.warning(f"Error parsing response: {e}")
            return "Unknown", f"Failed to parse response: {str(e)}"
    
    def _parse_brand_from_text(self, response_text: str) -> str:
        """
        Parse brand name from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Extracted brand name
        """
        # Clean the response
        response_text = response_text.strip()
        
        # Handle common response patterns
        if response_text.lower() in ["unknown", "none", "n/a", "not found"]:
            return "Unknown"
        
        # Extract brand name using various patterns
        brand_patterns = [
            r'^([A-Za-z][A-Za-z0-9\s&\-\.]+?)(?:\s|$)',  # Brand at start
            r'brand[:\s]+([A-Za-z][A-Za-z0-9\s&\-\.]+?)(?:\s|$)',  # After "brand:"
            r'answer[:\s]+([A-Za-z][A-Za-z0-9\s&\-\.]+?)(?:\s|$)',  # After "answer:"
        ]
        
        for pattern in brand_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                brand = match.group(1).strip()
                # Clean up the brand name
                brand = re.sub(r'\s+', ' ', brand)
                brand = brand.strip('.,!?')
                
                # Validate brand name
                if len(brand) >= 2 and not brand.lower() in ["the", "and", "or", "is", "are"]:
                    return brand
        
        # If no pattern matches, try to extract the first meaningful word/phrase
        words = response_text.split()
        if words:
            # Look for capitalized words (likely brand names)
            for word in words:
                if word[0].isupper() and len(word) >= 2:
                    # Check if it's not a common word
                    if word.lower() not in ["the", "brand", "name", "is", "product", "title"]:
                        return word
        
        # Fallback: return first line if it looks like a brand
        first_line = response_text.split('\n')[0].strip()
        if len(first_line) <= 50 and len(first_line) >= 2:
            return first_line
        
        return "Unknown"
    
    def _calculate_confidence(self, predicted_brand: str, reasoning: str, original_text: str) -> float:
        """
        Calculate confidence score for the brand prediction.
        
        Args:
            predicted_brand: Predicted brand name
            reasoning: LLM reasoning text
            original_text: Original product name
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if predicted_brand == "Unknown":
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Brand name quality factors
        if len(predicted_brand) >= 3:
            confidence += 0.1
        
        if predicted_brand[0].isupper():
            confidence += 0.1
        
        # Text matching factors
        if predicted_brand.lower() in original_text.lower():
            confidence += 0.2
        
        # Reasoning quality factors
        reasoning_lower = reasoning.lower()
        
        # Check for confident language in reasoning
        confident_phrases = ["clearly", "obviously", "definitely", "brand is", "brand name is"]
        if any(phrase in reasoning_lower for phrase in confident_phrases):
            confidence += 0.1
        
        # Check for uncertainty indicators
        uncertain_phrases = ["might be", "could be", "possibly", "maybe", "not sure"]
        if any(phrase in reasoning_lower for phrase in uncertain_phrases):
            confidence -= 0.2
        
        # Length and structure of reasoning
        if len(reasoning) > 20:  # Detailed reasoning usually indicates confidence
            confidence += 0.05
        
        # Brand name appears multiple times in reasoning
        brand_mentions = reasoning_lower.count(predicted_brand.lower())
        if brand_mentions > 1:
            confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    async def cleanup(self) -> None:
        """Clean up Bedrock LLM agent resources."""
        if self.bedrock_runtime:
            # Bedrock client doesn't require explicit cleanup
            self.bedrock_runtime = None
        
        if self.session:
            self.session = None
        
        self.set_initialized(False)
        self.logger.info("Bedrock LLM agent cleaned up")
    
    async def _perform_health_check(self) -> None:
        """Perform LLM agent specific health check."""
        await super()._perform_health_check()
        
        if not self.bedrock_runtime:
            raise RuntimeError("Bedrock client not initialized")
        
        # Test with a simple inference
        try:
            test_result = await self.infer_brand("Samsung Galaxy S23")
            if not isinstance(test_result, LLMResult):
                raise RuntimeError("Health check inference failed")
        except Exception as e:
            raise RuntimeError(f"Health check inference failed: {e}")


class EnhancedBedrockLLMAgent(BedrockLLMAgent):
    """
    Enhanced Bedrock LLM agent with additional features.
    
    Extends the base Bedrock agent with improved prompt engineering,
    context-aware inference, and enhanced multilingual support.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize enhanced Bedrock LLM agent.
        
        Args:
            config: Configuration dictionary with enhanced settings
        """
        super().__init__(config)
        self.agent_name = "enhanced_bedrock_llm"
        
        # Enhanced configuration
        self.use_context_enhancement = self.get_config_value("use_context_enhancement", True)
        self.context_weight = self.get_config_value("context_weight", 0.3)
        self.enable_reasoning_analysis = self.get_config_value("enable_reasoning_analysis", True)
    
    async def infer_brand(self, product_name: str, context: Optional[str] = None) -> LLMResult:
        """
        Enhanced brand inference with context awareness.
        
        Args:
            product_name: Input product name text
            context: Optional context for enhanced inference
            
        Returns:
            Enhanced LLMResult with improved confidence scoring
        """
        # Get base result
        result = await super().infer_brand(product_name, context)
        
        # Apply enhancements
        if self.use_context_enhancement and context:
            enhanced_confidence = self._enhance_confidence_with_context(
                result.confidence,
                result.predicted_brand,
                context
            )
            
            # Create enhanced result
            return LLMResult(
                predicted_brand=result.predicted_brand,
                reasoning=result.reasoning,
                confidence=enhanced_confidence,
                processing_time=result.processing_time,
                model_id=f"{result.model_id}_enhanced"
            )
        
        return result
    
    def _enhance_confidence_with_context(
        self, 
        base_confidence: float, 
        predicted_brand: str, 
        context: str
    ) -> float:
        """
        Enhance confidence score using additional context.
        
        Args:
            base_confidence: Original confidence score
            predicted_brand: Predicted brand name
            context: Additional context information
            
        Returns:
            Enhanced confidence score
        """
        if not context or predicted_brand == "Unknown":
            return base_confidence
        
        context_lower = context.lower()
        brand_lower = predicted_brand.lower()
        
        # Context validation boost
        context_boost = 0.0
        
        # Brand appears in context
        if brand_lower in context_lower:
            context_boost += 0.15
        
        # Context contains brand-related keywords
        brand_keywords = ["brand", "manufacturer", "company", "made by"]
        if any(keyword in context_lower for keyword in brand_keywords):
            context_boost += 0.05
        
        # Apply context weight
        enhanced_confidence = base_confidence + (context_boost * self.context_weight)
        
        return min(1.0, enhanced_confidence)