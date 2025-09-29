"""
Base agent interface and abstract classes for the inference system.

This module defines the common interface that all inference agents must implement,
following PEP 8 standards and providing a consistent API across agent types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time
import asyncio
import uuid

from ..models.data_models import (
    ProductInput,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    AgentHealth
)


class BaseAgent(ABC):
    """Abstract base class for all inference agents."""
    
    def __init__(self, agent_name: str, config: Dict[str, Any]) -> None:
        """
        Initialize base agent with configuration.
        
        Args:
            agent_name: Unique identifier for the agent
            config: Configuration dictionary for the agent
        """
        self.agent_name = agent_name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self._is_initialized = False
        self._last_health_check = 0.0
        
        # Initialize monitoring components
        self._inference_logger = None
        self._metrics_collector = None
        self._setup_monitoring()
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent resources (models, connections, etc.)."""
        pass
    
    @abstractmethod
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process product input and return inference results.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing inference results
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        pass
    
    async def health_check(self) -> AgentHealth:
        """
        Perform health check on the agent.
        
        Returns:
            AgentHealth object with current status
        """
        start_time = time.time()
        
        try:
            # Basic health check - ensure agent is initialized
            if not self._is_initialized:
                return AgentHealth(
                    agent_name=self.agent_name,
                    is_healthy=False,
                    last_check=start_time,
                    error_message="Agent not initialized",
                    response_time=None
                )
            
            # Perform agent-specific health check
            await self._perform_health_check()
            
            response_time = time.time() - start_time
            self._last_health_check = start_time
            
            return AgentHealth(
                agent_name=self.agent_name,
                is_healthy=True,
                last_check=start_time,
                error_message=None,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Health check failed for {self.agent_name}: {str(e)}")
            
            return AgentHealth(
                agent_name=self.agent_name,
                is_healthy=False,
                last_check=start_time,
                error_message=str(e),
                response_time=response_time
            )
    
    async def _perform_health_check(self) -> None:
        """
        Perform agent-specific health check operations.
        
        Override this method in subclasses to implement custom health checks.
        """
        # Default implementation - just check if we can access config
        if not self.config:
            raise RuntimeError("Agent configuration is missing")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set_initialized(self, status: bool) -> None:
        """Set agent initialization status."""
        self._is_initialized = status
        if status:
            self.logger.info(f"Agent {self.agent_name} initialized successfully")
        else:
            self.logger.warning(f"Agent {self.agent_name} initialization failed")
    
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._is_initialized
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring components for the agent."""
        try:
            # Import monitoring components (lazy import to avoid circular dependencies)
            from ..monitoring.logger import get_inference_logger
            from ..monitoring.cloudwatch_integration import create_metrics_collector
            
            self._inference_logger = get_inference_logger()
            self._metrics_collector = create_metrics_collector()
            
        except ImportError as e:
            self.logger.warning(f"Monitoring components not available: {str(e)}")
    
    async def process_with_monitoring(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process input with comprehensive monitoring and logging.
        
        Args:
            input_data: Product input data structure
            
        Returns:
            Dictionary containing inference results with monitoring data
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Setup monitoring context
        if self._inference_logger:
            async with self._inference_logger.operation_context(
                request_id=request_id,
                agent_name=self.agent_name,
                operation="process",
                product_name=input_data.product_name,
                language_hint=input_data.language_hint.value
            ):
                try:
                    # Process the input
                    result = await self.process(input_data)
                    
                    # Log successful processing
                    processing_time = time.time() - start_time
                    
                    if self._inference_logger:
                        self._inference_logger.log_agent_decision(
                            agent_name=self.agent_name,
                            decision=str(result),
                            confidence=getattr(result, 'confidence', 0.0),
                            reasoning=f"Processed {input_data.product_name} successfully",
                            input_data={
                                "product_name": input_data.product_name,
                                "language_hint": input_data.language_hint.value
                            },
                            processing_time=processing_time
                        )
                    
                    # Record metrics
                    if self._metrics_collector:
                        self._record_processing_metrics(result, processing_time, request_id)
                    
                    return {
                        "result": result,
                        "request_id": request_id,
                        "processing_time": processing_time,
                        "agent_name": self.agent_name
                    }
                    
                except Exception as e:
                    # Log error
                    if self._inference_logger:
                        self._inference_logger.log_operation_error(
                            self._inference_logger._get_current_context(),
                            e
                        )
                    
                    # Record error metrics
                    if self._metrics_collector:
                        self._metrics_collector.record_error_metrics(
                            error_type=type(e).__name__,
                            agent_name=self.agent_name
                        )
                    
                    raise
        else:
            # Fallback without monitoring
            result = await self.process(input_data)
            processing_time = time.time() - start_time
            
            return {
                "result": result,
                "request_id": request_id,
                "processing_time": processing_time,
                "agent_name": self.agent_name
            }
    
    def _record_processing_metrics(
        self,
        result: Any,
        processing_time: float,
        request_id: str
    ) -> None:
        """
        Record processing metrics for the agent.
        
        Args:
            result: Processing result
            processing_time: Time taken to process
            request_id: Request identifier
        """
        if not self._metrics_collector:
            return
        
        # Record latency metric
        self._metrics_collector.add_metric(
            metric_name=f"{self.agent_name.upper()}Latency",
            value=processing_time * 1000,  # Convert to milliseconds
            unit="Milliseconds",
            dimensions={
                "AgentName": self.agent_name,
                "RequestId": request_id
            }
        )
        
        # Record confidence metric if available
        if hasattr(result, 'confidence'):
            self._metrics_collector.add_metric(
                metric_name=f"{self.agent_name.upper()}Confidence",
                value=result.confidence,
                unit="None",
                dimensions={
                    "AgentName": self.agent_name,
                    "RequestId": request_id
                }
            )
        
        # Record success metric
        self._metrics_collector.add_metric(
            metric_name="ProcessingSuccess",
            value=1.0,
            unit="Count",
            dimensions={
                "AgentName": self.agent_name,
                "RequestId": request_id
            }
        )


class NERAgent(BaseAgent):
    """Abstract base class for Named Entity Recognition agents."""
    
    @abstractmethod
    async def extract_entities(self, product_name: str) -> NERResult:
        """
        Extract entities from product name.
        
        Args:
            product_name: Input product name text
            
        Returns:
            NERResult with extracted entities and confidence scores
        """
        pass


class RAGAgent(BaseAgent):
    """Abstract base class for Retrieval-Augmented Generation agents."""
    
    @abstractmethod
    async def retrieve_and_infer(self, product_name: str) -> RAGResult:
        """
        Retrieve similar products and infer brand.
        
        Args:
            product_name: Input product name text
            
        Returns:
            RAGResult with brand prediction and similar products
        """
        pass


class LLMAgent(BaseAgent):
    """Abstract base class for Large Language Model agents."""
    
    @abstractmethod
    async def infer_brand(self, product_name: str, context: Optional[str] = None) -> LLMResult:
        """
        Infer brand using language model.
        
        Args:
            product_name: Input product name text
            context: Optional context for enhanced inference
            
        Returns:
            LLMResult with brand prediction and reasoning
        """
        pass


class HybridAgent(BaseAgent):
    """Abstract base class for hybrid inference agents."""
    
    @abstractmethod
    async def hybrid_inference(
        self,
        product_name: str,
        ner_agent: Optional[NERAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        llm_agent: Optional[LLMAgent] = None
    ) -> HybridResult:
        """
        Perform hybrid inference using multiple approaches.
        
        Args:
            product_name: Input product name text
            ner_agent: Optional NER agent for pipeline
            rag_agent: Optional RAG agent for pipeline
            llm_agent: Optional LLM agent for pipeline
            
        Returns:
            HybridResult with combined inference results
        """
        pass


class OrchestratorAgent(BaseAgent):
    """Abstract base class for orchestrator agents."""
    
    @abstractmethod
    async def orchestrate_inference(
        self,
        input_data: ProductInput,
        agents: Dict[str, BaseAgent]
    ) -> Dict[str, Any]:
        """
        Orchestrate inference across multiple agents.
        
        Args:
            input_data: Product input data
            agents: Dictionary of available agents
            
        Returns:
            Complete inference results from all agents
        """
        pass


class AgentError(Exception):
    """Base exception class for agent-related errors."""
    
    def __init__(self, agent_name: str, message: str, original_error: Optional[Exception] = None):
        """
        Initialize agent error.
        
        Args:
            agent_name: Name of the agent that raised the error
            message: Error message
            original_error: Original exception that caused this error
        """
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}': {message}")


class AgentTimeoutError(AgentError):
    """Exception raised when agent operations timeout."""
    
    def __init__(self, agent_name: str, timeout_seconds: float):
        """
        Initialize timeout error.
        
        Args:
            agent_name: Name of the agent that timed out
            timeout_seconds: Timeout duration in seconds
        """
        super().__init__(
            agent_name,
            f"Operation timed out after {timeout_seconds} seconds"
        )


class AgentInitializationError(AgentError):
    """Exception raised when agent initialization fails."""
    
    def __init__(self, agent_name: str, message: str, original_error: Optional[Exception] = None):
        """
        Initialize initialization error.
        
        Args:
            agent_name: Name of the agent that failed to initialize
            message: Error message
            original_error: Original exception that caused the failure
        """
        super().__init__(agent_name, f"Initialization failed: {message}", original_error)