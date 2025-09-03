"""
Orchestrator agent implementation using strands-agents SDK.

This module implements the orchestrator agent that coordinates parallel execution
of multiple inference agents, following PEP 8 standards and providing comprehensive
error handling and timeout management.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
import logging
from concurrent.futures import TimeoutError as ConcurrentTimeoutError

try:
    from strands_agents import Agent, AgentMessage, AgentContext
    from strands_agents.exceptions import AgentError as StrandsAgentError
except ImportError:
    # Fallback for development without strands-agents
    class Agent:
        pass
    
    class AgentMessage:
        def __init__(self, content: Any, sender: str = "system"):
            self.content = content
            self.sender = sender
    
    class AgentContext:
        def __init__(self):
            self.messages = []
    
    class StrandsAgentError(Exception):
        pass

from ..models.data_models import (
    ProductInput,
    InferenceResult,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    AgentHealth
)
from ..config.settings import SystemConfig, get_config
from .base_agent import (
    BaseAgent,
    OrchestratorAgent,
    NERAgent,
    RAGAgent,
    LLMAgent,
    HybridAgent,
    AgentError,
    AgentTimeoutError,
    AgentInitializationError
)
from .registry import get_agent_registry, initialize_default_agents


class StrandsOrchestratorAgent(OrchestratorAgent, Agent):
    """
    Orchestrator agent implementation using strands-agents framework.
    
    Coordinates parallel execution of multiple inference agents with proper
    error handling, timeout management, and result aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize orchestrator agent.
        
        Args:
            config: Optional configuration dictionary
        """
        # Initialize base classes
        BaseAgent.__init__(self, "orchestrator", config or {})
        Agent.__init__(self)
        
        self.system_config = get_config()
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_timeouts: Dict[str, float] = {}
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        
        # Configure timeouts and thresholds
        self.default_timeout = self.get_config_value("timeout_seconds", 30.0)
        self.max_parallel_agents = self.get_config_value("max_parallel_agents", 4)
        self.confidence_threshold = self.get_config_value("confidence_threshold", 0.5)
        self.circuit_breaker_threshold = self.get_config_value("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.get_config_value("circuit_breaker_timeout", 60.0)
        
        self.logger.info("Orchestrator agent initialized with strands-agents framework")
        
        # Initialize monitoring
        self._setup_orchestrator_monitoring()
    
    async def initialize(self) -> None:
        """Initialize orchestrator and register available agents."""
        try:
            # Initialize circuit breaker states
            self._initialize_circuit_breakers()
            
            # Register default agents automatically
            await self._register_default_agents()
            
            # Initialize registered agents
            await self._initialize_agents()
            
            self.set_initialized(True)
            self.logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {str(e)}")
            raise AgentInitializationError("orchestrator", str(e), e)
    
    async def cleanup(self) -> None:
        """Clean up orchestrator and all registered agents."""
        cleanup_tasks = []
        
        for agent_name, agent in self.agents.items():
            try:
                cleanup_tasks.append(agent.cleanup())
            except Exception as e:
                self.logger.warning(f"Error scheduling cleanup for {agent_name}: {str(e)}")
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.agents.clear()
        
        # Clean up global registry
        from .registry import cleanup_default_agents
        await cleanup_default_agents()
        
        self.logger.info("Orchestrator cleanup completed")
    
    def register_agent(
        self,
        agent_name: str,
        agent: BaseAgent,
        timeout: Optional[float] = None
    ) -> None:
        """
        Register an inference agent with the orchestrator.
        
        Args:
            agent_name: Unique identifier for the agent
            agent: Agent instance to register
            timeout: Optional custom timeout for this agent
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' is already registered")
        
        self.agents[agent_name] = agent
        self.agent_timeouts[agent_name] = timeout or self.default_timeout
        
        # Initialize circuit breaker for this agent
        self.circuit_breaker_states[agent_name] = {
            "failure_count": 0,
            "last_failure_time": 0.0,
            "state": "closed"  # closed, open, half-open
        }
        
        self.logger.info(f"Registered agent '{agent_name}' with timeout {self.agent_timeouts[agent_name]}s")
    
    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an inference agent.
        
        Args:
            agent_name: Name of the agent to unregister
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.agent_timeouts[agent_name]
            del self.circuit_breaker_states[agent_name]
            self.logger.info(f"Unregistered agent '{agent_name}'")
    
    async def orchestrate_inference(
        self,
        input_data: ProductInput,
        agents: Optional[Dict[str, BaseAgent]] = None
    ) -> InferenceResult:
        """
        Orchestrate inference across multiple agents in parallel.
        
        Args:
            input_data: Product input data
            agents: Optional specific agents to use (defaults to all registered)
            
        Returns:
            Complete inference results from all agents
        """
        start_time = time.time()
        
        try:
            # Use provided agents or all registered agents
            target_agents = agents or self.agents
            
            if not target_agents:
                raise AgentError("orchestrator", "No agents available for inference")
            
            self.logger.info(f"Starting orchestrated inference for: {input_data.product_name}")
            
            # Execute agents in parallel with timeout and error handling
            results = await self._execute_parallel_inference(input_data, target_agents)
            
            # Aggregate results and determine best prediction
            inference_result = await self._aggregate_results(
                input_data.product_name,
                results,
                time.time() - start_time
            )
            
            self.logger.info(
                f"Orchestrated inference completed in {inference_result.total_processing_time:.3f}s, "
                f"best method: {inference_result.best_method} "
                f"(confidence: {inference_result.best_confidence:.3f})"
            )
            
            return inference_result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Orchestrated inference failed after {total_time:.3f}s: {str(e)}")
            raise AgentError("orchestrator", f"Inference orchestration failed: {str(e)}", e)
    
    async def _execute_parallel_inference(
        self,
        input_data: ProductInput,
        target_agents: Dict[str, BaseAgent]
    ) -> Dict[str, Any]:
        """
        Execute inference across multiple agents in parallel.
        
        Args:
            input_data: Product input data
            target_agents: Dictionary of agents to execute
            
        Returns:
            Dictionary of agent results
        """
        # Create tasks for parallel execution
        tasks = {}
        
        for agent_name, agent in target_agents.items():
            # Check circuit breaker state
            if not self._is_agent_available(agent_name):
                self.logger.warning(f"Agent '{agent_name}' is circuit-broken, skipping")
                continue
            
            # Create task with timeout
            timeout = self.agent_timeouts.get(agent_name, self.default_timeout)
            task = asyncio.create_task(
                self._execute_single_agent(agent_name, agent, input_data, timeout)
            )
            tasks[agent_name] = task
        
        if not tasks:
            raise AgentError("orchestrator", "No available agents for execution")
        
        # Wait for all tasks to complete or timeout
        results = {}
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results and handle exceptions
        for (agent_name, task), result in zip(tasks.items(), completed_tasks):
            if isinstance(result, Exception):
                self._handle_agent_failure(agent_name, result)
                results[agent_name] = None
            else:
                self._handle_agent_success(agent_name)
                results[agent_name] = result
        
        return results
    
    async def _execute_single_agent(
        self,
        agent_name: str,
        agent: BaseAgent,
        input_data: ProductInput,
        timeout: float
    ) -> Any:
        """
        Execute a single agent with timeout and error handling.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            input_data: Input data for processing
            timeout: Timeout in seconds
            
        Returns:
            Agent processing result
        """
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.process(input_data),
                timeout=timeout
            )
            
            self.logger.debug(f"Agent '{agent_name}' completed successfully")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Agent '{agent_name}' timed out after {timeout}s"
            self.logger.warning(error_msg)
            raise AgentTimeoutError(agent_name, timeout)
        
        except Exception as e:
            error_msg = f"Agent '{agent_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            raise AgentError(agent_name, str(e), e)
    
    async def _aggregate_results(
        self,
        input_product: str,
        results: Dict[str, Any],
        total_time: float
    ) -> InferenceResult:
        """
        Aggregate results from multiple agents and determine best prediction.
        
        Args:
            input_product: Original input product name
            results: Dictionary of agent results (wrapped in process format)
            total_time: Total processing time
            
        Returns:
            Aggregated inference result
        """
        # Extract individual results from wrapped format
        # Each agent returns {"agent_type": "...", "result": actual_result, "success": bool}
        ner_result = None
        rag_result = None
        llm_result = None
        hybrid_result = None
        simple_result = None
        
        # Track agent results with detailed predictions
        agent_results = {}
        
        for agent_name, agent_response in results.items():
            if isinstance(agent_response, dict) and agent_response.get("success"):
                actual_result = agent_response.get("result")
                
                # Build detailed result for this agent
                agent_detail = {
                    "success": True,
                    "prediction": None,
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "error": None
                }
                
                if agent_name == "ner":
                    ner_result = actual_result
                    if ner_result and ner_result.entities:
                        # Find best brand entity
                        brand_entities = [e for e in ner_result.entities if e.entity_type.value == "BRAND"]
                        if brand_entities:
                            best_brand = max(brand_entities, key=lambda x: x.confidence)
                            agent_detail["prediction"] = best_brand.text
                            agent_detail["confidence"] = best_brand.confidence
                        agent_detail["processing_time"] = ner_result.processing_time
                        agent_detail["entities_count"] = len(ner_result.entities)
                        
                elif agent_name == "rag":
                    rag_result = actual_result
                    if rag_result:
                        agent_detail["prediction"] = rag_result.predicted_brand
                        agent_detail["confidence"] = rag_result.confidence
                        agent_detail["processing_time"] = rag_result.processing_time
                        agent_detail["similar_products_count"] = len(rag_result.similar_products)
                        
                elif agent_name == "llm":
                    llm_result = actual_result
                    if llm_result:
                        agent_detail["prediction"] = llm_result.predicted_brand
                        agent_detail["confidence"] = llm_result.confidence
                        agent_detail["processing_time"] = llm_result.processing_time
                        agent_detail["reasoning"] = llm_result.reasoning[:100] + "..." if len(llm_result.reasoning) > 100 else llm_result.reasoning
                        
                elif agent_name == "hybrid":
                    hybrid_result = actual_result
                    if hybrid_result:
                        agent_detail["prediction"] = hybrid_result.predicted_brand
                        agent_detail["confidence"] = hybrid_result.confidence
                        agent_detail["processing_time"] = hybrid_result.processing_time
                        agent_detail["ner_contribution"] = hybrid_result.ner_contribution
                        agent_detail["rag_contribution"] = hybrid_result.rag_contribution
                        agent_detail["llm_contribution"] = hybrid_result.llm_contribution
                        
                elif agent_name == "simple":
                    # Simple agent returns LLMResult-like structure
                    simple_result = actual_result
                    if simple_result:
                        agent_detail["prediction"] = simple_result.predicted_brand
                        agent_detail["confidence"] = simple_result.confidence
                        agent_detail["processing_time"] = simple_result.processing_time
                        if hasattr(simple_result, 'reasoning'):
                            agent_detail["reasoning"] = simple_result.reasoning[:100] + "..." if len(simple_result.reasoning) > 100 else simple_result.reasoning
                
                agent_results[agent_name] = agent_detail
                
            else:
                # Agent failed or returned None
                error_msg = "Unknown error"
                if isinstance(agent_response, dict):
                    error_msg = agent_response.get("error", "Agent returned no result")
                elif agent_response is None:
                    error_msg = "Agent returned None"
                
                agent_results[agent_name] = {
                    "success": False,
                    "prediction": None,
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "error": error_msg
                }
        
        # Determine best prediction based on confidence scores
        best_prediction, best_confidence, best_method = self._select_best_prediction(
            ner_result, rag_result, llm_result, hybrid_result, simple_result
        )
        
        return InferenceResult(
            input_product=input_product,
            ner_result=ner_result,
            rag_result=rag_result,
            llm_result=llm_result,
            hybrid_result=hybrid_result,
            best_prediction=best_prediction,
            best_confidence=best_confidence,
            best_method=best_method,
            total_processing_time=total_time,
            agent_results=agent_results
        )
    
    def _select_best_prediction(
        self,
        ner_result: Optional[NERResult],
        rag_result: Optional[RAGResult],
        llm_result: Optional[LLMResult],
        hybrid_result: Optional[HybridResult],
        simple_result: Optional[LLMResult] = None
    ) -> Tuple[str, float, str]:
        """
        Select the best prediction based on confidence scores.
        
        Args:
            ner_result: NER agent result
            rag_result: RAG agent result
            llm_result: LLM agent result
            hybrid_result: Hybrid agent result
            
        Returns:
            Tuple of (best_prediction, best_confidence, best_method)
        """
        candidates = []
        
        # Collect valid predictions with confidence scores
        if ner_result and ner_result.entities:
            # Use highest confidence brand entity
            brand_entities = [e for e in ner_result.entities if e.entity_type.value == "BRAND"]
            if brand_entities:
                best_brand = max(brand_entities, key=lambda x: x.confidence)
                candidates.append((best_brand.text, best_brand.confidence, "ner"))
        
        if rag_result and rag_result.confidence >= self.confidence_threshold:
            candidates.append((rag_result.predicted_brand, rag_result.confidence, "rag"))
        
        if llm_result and llm_result.confidence >= self.confidence_threshold:
            candidates.append((llm_result.predicted_brand, llm_result.confidence, "llm"))
        
        if hybrid_result and hybrid_result.confidence >= self.confidence_threshold:
            candidates.append((hybrid_result.predicted_brand, hybrid_result.confidence, "hybrid"))
        
        if simple_result and simple_result.confidence >= self.confidence_threshold:
            candidates.append((simple_result.predicted_brand, simple_result.confidence, "simple"))
        
        # Select best candidate or return default
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1])
            return best_candidate
        else:
            # If no confident predictions, use simple result as fallback if available
            if simple_result:
                return simple_result.predicted_brand, simple_result.confidence, "simple"
            # No predictions available at all
            return "unknown", 0.0, "none"
    
    async def _register_default_agents(self) -> None:
        """Register default agents from the global registry."""
        try:
            self.logger.info("Registering default agents...")
            
            # Initialize default agents through registry
            default_agents = await initialize_default_agents()
            
            # Register each agent with the orchestrator
            for agent_name, agent in default_agents.items():
                self.register_agent(agent_name, agent)
            
            self.logger.info(f"Registered {len(default_agents)} default agents")
            
        except Exception as e:
            self.logger.warning(f"Failed to register some default agents: {str(e)}")
            # Continue with whatever agents were successfully registered
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breaker states for all agents."""
        for agent_name in self.agents.keys():
            self.circuit_breaker_states[agent_name] = {
                "failure_count": 0,
                "last_failure_time": 0.0,
                "state": "closed"
            }
    
    async def _initialize_agents(self) -> None:
        """Initialize all registered agents."""
        initialization_tasks = []
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'initialize'):
                initialization_tasks.append(
                    self._safe_initialize_agent(agent_name, agent)
                )
        
        if initialization_tasks:
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check for initialization failures
            for (agent_name, _), result in zip(self.agents.items(), results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to initialize agent '{agent_name}': {str(result)}")
                    # Mark agent as unavailable
                    self.circuit_breaker_states[agent_name]["state"] = "open"
    
    async def _safe_initialize_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """
        Safely initialize an agent with error handling.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance to initialize
        """
        try:
            await agent.initialize()
            self.logger.info(f"Agent '{agent_name}' initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent '{agent_name}': {str(e)}")
            raise AgentInitializationError(agent_name, str(e), e)
    
    def _is_agent_available(self, agent_name: str) -> bool:
        """
        Check if an agent is available (not circuit-broken).
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            True if agent is available, False otherwise
        """
        circuit_state = self.circuit_breaker_states.get(agent_name, {})
        state = circuit_state.get("state", "closed")
        
        if state == "closed":
            return True
        elif state == "open":
            # Check if we should try half-open
            last_failure = circuit_state.get("last_failure_time", 0.0)
            if time.time() - last_failure > self.circuit_breaker_timeout:
                circuit_state["state"] = "half-open"
                return True
            return False
        elif state == "half-open":
            return True
        
        return False
    
    def _handle_agent_failure(self, agent_name: str, error: Exception) -> None:
        """
        Handle agent failure and update circuit breaker state.
        
        Args:
            agent_name: Name of the failed agent
            error: Exception that caused the failure
        """
        circuit_state = self.circuit_breaker_states.get(agent_name, {})
        circuit_state["failure_count"] = circuit_state.get("failure_count", 0) + 1
        circuit_state["last_failure_time"] = time.time()
        
        # Open circuit breaker if threshold exceeded
        if circuit_state["failure_count"] >= self.circuit_breaker_threshold:
            circuit_state["state"] = "open"
            self.logger.warning(
                f"Circuit breaker opened for agent '{agent_name}' "
                f"after {circuit_state['failure_count']} failures"
            )
        
        self.logger.error(f"Agent '{agent_name}' failed: {str(error)}")
    
    def _handle_agent_success(self, agent_name: str) -> None:
        """
        Handle agent success and reset circuit breaker if needed.
        
        Args:
            agent_name: Name of the successful agent
        """
        circuit_state = self.circuit_breaker_states.get(agent_name, {})
        
        # Reset circuit breaker on success
        if circuit_state.get("state") in ["half-open", "open"]:
            circuit_state["state"] = "closed"
            circuit_state["failure_count"] = 0
            self.logger.info(f"Circuit breaker reset for agent '{agent_name}'")
    
    async def get_agent_health(self) -> Dict[str, AgentHealth]:
        """
        Get health status of all registered agents.
        
        Returns:
            Dictionary mapping agent names to their health status
        """
        health_results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'health_check'):
                    health = await agent.health_check()
                else:
                    # Basic health check for agents without health_check method
                    health = AgentHealth(
                        agent_name=agent_name,
                        is_healthy=self._is_agent_available(agent_name),
                        last_check=time.time(),
                        error_message=None if self._is_agent_available(agent_name) else "Circuit breaker open",
                        response_time=None
                    )
                
                health_results[agent_name] = health
                
            except Exception as e:
                health_results[agent_name] = AgentHealth(
                    agent_name=agent_name,
                    is_healthy=False,
                    last_check=time.time(),
                    error_message=str(e),
                    response_time=None
                )
        
        return health_results
    
    async def process(self, input_data: ProductInput) -> Dict[str, Any]:
        """
        Process input through orchestrated inference (BaseAgent interface).
        
        Args:
            input_data: Product input data
            
        Returns:
            Dictionary containing complete inference results
        """
        result = await self.orchestrate_inference(input_data)
        
        # Extract entities from NER result if available
        entities = []
        if result.ner_result and result.ner_result.entities:
            entities = [
                {
                    "text": entity.text,
                    "label": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "start": entity.start_pos,
                    "end": entity.end_pos
                }
                for entity in result.ner_result.entities
            ]
        
        # Use agent results from the InferenceResult, or build default if not available
        agent_results = result.agent_results or {}
        
        # Convert InferenceResult to dictionary format expected by server
        return {
            "product_name": input_data.product_name,
            "language": input_data.language_hint.value,
            "brand_predictions": [
                {
                    "brand": result.best_prediction,
                    "confidence": result.best_confidence,
                    "method": result.best_method
                }
            ],
            "entities": entities,
            "processing_time_ms": int(result.total_processing_time * 1000),
            "agent_used": "orchestrator",
            "orchestrator_agents": list(self.agents.keys()),
            "agent_results": agent_results,
            "timestamp": time.time()
        }
    
    async def _perform_health_check(self) -> None:
        """Perform orchestrator-specific health check."""
        # Check if we have any available agents
        available_agents = sum(1 for name in self.agents.keys() if self._is_agent_available(name))
        
        if available_agents == 0:
            raise RuntimeError("No agents are currently available")
        
        # Check agent health
        health_results = await self.get_agent_health()
        healthy_agents = sum(1 for health in health_results.values() if health.is_healthy)
        
        if healthy_agents == 0:
            raise RuntimeError("No agents are currently healthy")
        
        self.logger.debug(f"Health check passed: {healthy_agents}/{len(self.agents)} agents healthy")
    
    def _setup_orchestrator_monitoring(self) -> None:
        """Setup monitoring specific to orchestrator agent."""
        try:
            from ..monitoring.logger import get_inference_logger
            from ..monitoring.cloudwatch_integration import create_metrics_collector
            
            self._orchestrator_logger = get_inference_logger()
            self._orchestrator_metrics = create_metrics_collector()
            
        except ImportError as e:
            self.logger.warning(f"Orchestrator monitoring not available: {str(e)}")
            self._orchestrator_logger = None
            self._orchestrator_metrics = None
    
    async def orchestrate_inference_with_monitoring(
        self,
        input_data: ProductInput,
        agents: Optional[Dict[str, BaseAgent]] = None
    ) -> InferenceResult:
        """
        Orchestrate inference with comprehensive monitoring.
        
        Args:
            input_data: Product input data
            agents: Optional specific agents to use
            
        Returns:
            Complete inference results with monitoring data
        """
        request_id = str(uuid.uuid4())
        
        if self._orchestrator_logger:
            async with self._orchestrator_logger.operation_context(
                request_id=request_id,
                agent_name="orchestrator",
                operation="orchestrate_inference",
                product_name=input_data.product_name,
                language_hint=input_data.language_hint.value
            ):
                result = await self.orchestrate_inference(input_data, agents)
                
                # Log inference result
                self._orchestrator_logger.log_inference_result(result, request_id)
                
                # Record orchestrator metrics
                if self._orchestrator_metrics:
                    self._orchestrator_metrics.record_inference_metrics(result, request_id)
                
                return result
        else:
            return await self.orchestrate_inference(input_data, agents)


# Factory function for creating orchestrator instances
def create_orchestrator_agent(config: Optional[Dict[str, Any]] = None) -> StrandsOrchestratorAgent:
    """
    Factory function to create a configured orchestrator agent.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured orchestrator agent instance
    """
    return StrandsOrchestratorAgent(config)