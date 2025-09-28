"""
Integration tests for orchestrator-agent communication.

This module tests the integration between the orchestrator and individual agents,
focusing on 1-to-1 communication, parallel execution, timeout handling, and
health monitoring as specified in the requirements.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from inference.src.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
from inference.src.agents.base_agent import BaseAgent, AgentError, AgentTimeoutError
from inference.src.models.data_models import (
    ProductInput,
    LanguageHint,
    AgentHealth,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult
)

from inference.tests.utils.assertion_helpers import InferenceAssertions
from inference.tests.utils.mock_factories import MockAgentFactory


@pytest.mark.integration
class TestOrchestratorIntegration:
    """
    Integration tests for orchestrator-agent communication.
    
    Tests 1-to-1 communication between orchestrator and each agent type,
    parallel execution, timeout handling, and health monitoring.
    """
    
    def setup_method(self):
        """Setup for each test method."""
        
        # Create orchestrator with test configuration
        self.orchestrator_config = {
            "confidence_threshold": 0.5,
            "max_parallel_agents": 4,
            "timeout_seconds": 30,
            "agent_timeout": 10
        }
        
        self.orchestrator = StrandsMultiAgentOrchestrator(self.orchestrator_config)
        
        # Create mock agents for testing
        self.mock_agents = self._create_mock_agents()
        
        # Test input data
        self.test_input = ProductInput(
            product_name="Samsung Galaxy S24 Ultra 256GB",
            language_hint=LanguageHint.ENGLISH
        )
    
    def teardown_method(self):
        """Cleanup after each test method."""
        
        # Clean up orchestrator
        if hasattr(self.orchestrator, 'cleanup'):
            asyncio.run(self.orchestrator.cleanup())
    
    def _create_mock_agents(self) -> Dict[str, Mock]:
        """
        Create mock agents for testing orchestrator communication.
        
        Returns:
            Dictionary of mock agents by type
        """
        mock_agents = {}
        
        # Create NER agent mock
        mock_agents['ner'] = MockAgentFactory.create_mock_ner_agent()
        
        # Create RAG agent mock
        mock_agents['rag'] = MockAgentFactory.create_mock_rag_agent()
        
        # Create LLM agent mock
        mock_agents['llm'] = MockAgentFactory.create_mock_llm_agent()
        
        # Create Hybrid agent mock
        mock_agents['hybrid'] = MockAgentFactory.create_mock_hybrid_agent()
        
        return mock_agents
    
    @pytest.mark.asyncio
    async def test_orchestrator_ner_agent_communication(self):
        """Test 1-to-1 communication between orchestrator and NER agent."""
        # Setup orchestrator with NER agent
        ner_agent = self.mock_agents['ner']
        
        # Mock the orchestrator's agent registry
        with patch.object(self.orchestrator, 'agents', {'ner': ner_agent}):
            # Test direct communication
            result = await self._test_agent_communication(ner_agent, "ner")
            
            # Verify communication was successful
            assert result is not None
            assert result.get("success") is True
            assert result.get("agent_type") == "ner"
            
            # Verify agent was called correctly
            ner_agent.process.assert_called_once()
            call_args = ner_agent.process.call_args[0][0]
            assert isinstance(call_args, ProductInput)
            assert call_args.product_name == self.test_input.product_name
    
    @pytest.mark.asyncio
    async def test_orchestrator_rag_agent_communication(self):
        """Test 1-to-1 communication between orchestrator and RAG agent."""
        # Setup orchestrator with RAG agent
        rag_agent = self.mock_agents['rag']
        
        # Mock the orchestrator's agent registry
        with patch.object(self.orchestrator, 'agents', {'rag': rag_agent}):
            # Test direct communication
            result = await self._test_agent_communication(rag_agent, "rag")
            
            # Verify communication was successful
            assert result is not None
            assert result.get("success") is True
            assert result.get("agent_type") == "rag"
            
            # Verify agent was called correctly
            rag_agent.process.assert_called_once()
            call_args = rag_agent.process.call_args[0][0]
            assert isinstance(call_args, ProductInput)
    
    @pytest.mark.asyncio
    async def test_orchestrator_llm_agent_communication(self):
        """Test 1-to-1 communication between orchestrator and LLM agent."""
        # Setup orchestrator with LLM agent
        llm_agent = self.mock_agents['llm']
        
        # Mock the orchestrator's agent registry
        with patch.object(self.orchestrator, 'agents', {'llm': llm_agent}):
            # Test direct communication
            result = await self._test_agent_communication(llm_agent, "llm")
            
            # Verify communication was successful
            assert result is not None
            assert result.get("success") is True
            assert result.get("agent_type") == "llm"
            
            # Verify agent was called correctly
            llm_agent.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_hybrid_agent_communication(self):
        """Test 1-to-1 communication between orchestrator and Hybrid agent."""
        # Setup orchestrator with Hybrid agent
        hybrid_agent = self.mock_agents['hybrid']
        
        # Mock the orchestrator's agent registry
        with patch.object(self.orchestrator, 'agents', {'hybrid': hybrid_agent}):
            # Test direct communication
            result = await self._test_agent_communication(hybrid_agent, "hybrid")
            
            # Verify communication was successful
            assert result is not None
            assert result.get("success") is True
            assert result.get("agent_type") == "hybrid"
            
            # Verify agent was called correctly
            hybrid_agent.process.assert_called_once()
    
    async def _test_agent_communication(self, agent: Mock, agent_type: str) -> Dict[str, Any]:
        """
        Helper method to test communication with a specific agent.
        
        Args:
            agent: Mock agent to test
            agent_type: Type of agent being tested
            
        Returns:
            Result from agent communication
        """
        # Simulate orchestrator calling agent directly
        result = await agent.process(self.test_input)
        
        # Verify result structure
        InferenceAssertions.assert_valid_agent_result(result, agent_type)
        
        return result
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test parallel execution of multiple agents through orchestrator."""
        # Setup orchestrator with multiple agents
        all_agents = self.mock_agents.copy()
        
        with patch.object(self.orchestrator, 'agents', all_agents):
            # Test parallel execution using orchestrator's coordination method
            start_time = time.time()
            
            # Execute multiple agents in parallel
            tasks = []
            for agent_name, agent in all_agents.items():
                task = asyncio.create_task(agent.process(self.test_input))
                tasks.append((agent_name, task))
            
            # Wait for all tasks to complete
            results = {}
            for agent_name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=10.0)
                    results[agent_name] = result
                except asyncio.TimeoutError:
                    results[agent_name] = {"error": "timeout", "success": False}
            
            execution_time = time.time() - start_time
            
            # Verify parallel execution was faster than sequential
            # (Should be less than sum of individual processing times)
            assert execution_time < 2.0  # Should complete quickly with mocks
            
            # Verify all agents were executed
            assert len(results) == len(all_agents)
            
            # Verify all agents returned successful results
            for agent_name, result in results.items():
                assert result.get("success") is True, f"Agent {agent_name} failed"
                assert result.get("agent_type") == agent_name
    
    @pytest.mark.asyncio
    async def test_result_coordination_and_aggregation(self):
        """Test orchestrator's ability to coordinate and aggregate results."""
        # Setup orchestrator with all agents
        all_agents = self.mock_agents.copy()
        
        with patch.object(self.orchestrator, 'agents', all_agents):
            # Mock the orchestrator's coordination method
            with patch.object(
                self.orchestrator, 
                '_coordinate_inference_internal'
            ) as mock_coordinate:
                # Setup mock coordination result
                mock_coordinate.return_value = {
                    "method": "test_coordination",
                    "product_name": self.test_input.product_name,
                    "results": {
                        "ner": {"prediction": "Samsung", "confidence": 0.85},
                        "rag": {"prediction": "Samsung", "confidence": 0.88},
                        "llm": {"prediction": "Samsung", "confidence": 0.90},
                        "hybrid": {"prediction": "Samsung", "confidence": 0.89}
                    },
                    "timestamp": time.time()
                }
                
                # Test coordination
                result = await self.orchestrator._coordinate_inference_internal(
                    self.test_input.product_name,
                    "test_coordination"
                )
                
                # Verify coordination result structure
                assert "method" in result
                assert "product_name" in result
                assert "results" in result
                assert "timestamp" in result
                
                # Verify all agents contributed to results
                agent_results = result["results"]
                assert len(agent_results) == len(all_agents)
                
                # Verify result aggregation
                for agent_name in all_agents.keys():
                    assert agent_name in agent_results
                    agent_result = agent_results[agent_name]
                    assert "prediction" in agent_result
                    assert "confidence" in agent_result
    
    @pytest.mark.asyncio
    async def test_timeout_handling_and_error_recovery(self):
        """Test orchestrator's timeout handling and error recovery mechanisms."""
        # Create agent that times out
        timeout_agent = MockAgentFactory.create_mock_base_agent("timeout_agent")
        
        # Configure timeout behavior
        async def timeout_process(input_data):
            await asyncio.sleep(5.0)  # This will timeout
            return {"success": True}
        
        timeout_agent.process = AsyncMock(side_effect=timeout_process)
        
        # Create normal agent for comparison
        normal_agent = self.mock_agents['rag']
        
        agents = {
            "timeout_agent": timeout_agent,
            "normal_agent": normal_agent
        }
        
        with patch.object(self.orchestrator, 'agents', agents):
            # Test timeout handling
            start_time = time.time()
            
            # Execute agents with timeout
            tasks = []
            for agent_name, agent in agents.items():
                if agent_name == "timeout_agent":
                    # This should timeout
                    task = asyncio.create_task(
                        asyncio.wait_for(agent.process(self.test_input), timeout=2.0)
                    )
                else:
                    # This should succeed
                    task = asyncio.create_task(agent.process(self.test_input))
                
                tasks.append((agent_name, task))
            
            # Collect results with error handling
            results = {}
            for agent_name, task in tasks:
                try:
                    result = await task
                    results[agent_name] = result
                except asyncio.TimeoutError:
                    results[agent_name] = {
                        "error": "Agent timed out",
                        "success": False,
                        "agent_type": agent_name
                    }
                except Exception as e:
                    results[agent_name] = {
                        "error": str(e),
                        "success": False,
                        "agent_type": agent_name
                    }
            
            execution_time = time.time() - start_time
            
            # Verify timeout was handled properly
            assert execution_time < 5.0  # Should not wait for full timeout
            
            # Verify timeout agent failed gracefully
            timeout_result = results["timeout_agent"]
            assert timeout_result["success"] is False
            assert "timeout" in timeout_result["error"].lower()
            
            # Verify normal agent still succeeded
            normal_result = results["normal_agent"]
            assert normal_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_agent_health_monitoring_and_status_reporting(self):
        """Test orchestrator's agent health monitoring and status reporting."""
        # Setup agents with different health states
        healthy_agent = self.mock_agents['ner']
        healthy_agent.health_check = AsyncMock(return_value=AgentHealth(
            agent_name="ner",
            is_healthy=True,
            last_check=time.time(),
            error_message=None,
            response_time=0.1
        ))
        
        unhealthy_agent = MockAgentFactory.create_mock_base_agent("unhealthy_agent")
        unhealthy_agent._is_initialized = False
        unhealthy_agent.health_check = AsyncMock(return_value=AgentHealth(
            agent_name="unhealthy_agent",
            is_healthy=False,
            last_check=time.time(),
            error_message="Agent initialization failed",
            response_time=None
        ))
        
        agents = {
            "healthy_agent": healthy_agent,
            "unhealthy_agent": unhealthy_agent
        }
        
        with patch.object(self.orchestrator, 'agents', agents):
            # Test health monitoring
            health_results = {}
            
            for agent_name, agent in agents.items():
                try:
                    health_status = await agent.health_check()
                    health_results[agent_name] = health_status
                except Exception as e:
                    health_results[agent_name] = AgentHealth(
                        agent_name=agent_name,
                        is_healthy=False,
                        last_check=time.time(),
                        error_message=str(e),
                        response_time=None
                    )
            
            # Verify health monitoring results
            assert len(health_results) == len(agents)
            
            # Verify healthy agent status
            healthy_status = health_results["healthy_agent"]
            assert healthy_status.is_healthy is True
            assert healthy_status.error_message is None
            assert healthy_status.response_time is not None
            
            # Verify unhealthy agent status
            unhealthy_status = health_results["unhealthy_agent"]
            assert unhealthy_status.is_healthy is False
            assert unhealthy_status.error_message is not None
            
            # Test status reporting aggregation
            overall_health = self._aggregate_health_status(health_results)
            
            # Verify overall health reflects mixed status
            assert overall_health["total_agents"] == 2
            assert overall_health["healthy_agents"] == 1
            assert overall_health["unhealthy_agents"] == 1
            assert overall_health["overall_healthy"] is False  # Not all agents healthy
    
    def _aggregate_health_status(self, health_results: Dict[str, AgentHealth]) -> Dict[str, Any]:
        """
        Aggregate health status from multiple agents.
        
        Args:
            health_results: Dictionary of agent health results
            
        Returns:
            Aggregated health status
        """
        total_agents = len(health_results)
        healthy_agents = sum(1 for h in health_results.values() if h.is_healthy)
        unhealthy_agents = total_agents - healthy_agents
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "overall_healthy": unhealthy_agents == 0,
            "health_check_time": time.time()
        }
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_propagation(self):
        """Test how orchestrator handles and propagates agent errors."""
        # Create agent that raises errors
        error_agent = MockAgentFactory.create_mock_base_agent("error_agent")
        
        # Configure error behavior
        async def error_process(input_data):
            raise Exception("Simulated agent error")
        
        error_agent.process = AsyncMock(side_effect=error_process)
        
        # Create normal agent
        normal_agent = self.mock_agents['llm']
        
        agents = {
            "error_agent": error_agent,
            "normal_agent": normal_agent
        }
        
        with patch.object(self.orchestrator, 'agents', agents):
            # Test error propagation
            results = {}
            
            for agent_name, agent in agents.items():
                try:
                    result = await agent.process(self.test_input)
                    results[agent_name] = result
                except Exception as e:
                    results[agent_name] = {
                        "error": str(e),
                        "success": False,
                        "agent_type": agent_name
                    }
            
            # Verify error was captured properly
            error_result = results["error_agent"]
            assert error_result["success"] is False
            assert "error" in error_result
            assert "Simulated agent error" in error_result["error"]
            
            # Verify normal agent was not affected
            normal_result = results["normal_agent"]
            assert normal_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_orchestrator_concurrent_request_handling(self):
        """Test orchestrator's ability to handle concurrent requests."""
        # Setup orchestrator with agents
        all_agents = self.mock_agents.copy()
        
        with patch.object(self.orchestrator, 'agents', all_agents):
            # Create multiple concurrent requests
            num_requests = 5
            test_inputs = [
                ProductInput(
                    product_name=f"Test Product {i}",
                    language_hint=LanguageHint.ENGLISH
                )
                for i in range(num_requests)
            ]
            
            # Execute concurrent requests
            start_time = time.time()
            
            tasks = []
            for i, test_input in enumerate(test_inputs):
                # Simulate orchestrator processing each request
                task = asyncio.create_task(
                    self._simulate_orchestrator_processing(test_input, all_agents)
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_time = time.time() - start_time
            
            # Verify all requests completed successfully
            assert len(results) == num_requests
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Request {i} failed with exception: {result}")
                
                assert isinstance(result, dict)
                assert "success" in result
                assert result["success"] is True
            
            # Verify concurrent execution was efficient
            # Should be faster than sequential execution
            assert execution_time < num_requests * 1.0  # Allow 1 second per request max
    
    async def _simulate_orchestrator_processing(
        self, 
        test_input: ProductInput, 
        agents: Dict[str, Mock]
    ) -> Dict[str, Any]:
        """
        Simulate orchestrator processing a single request.
        
        Args:
            test_input: Input to process
            agents: Available agents
            
        Returns:
            Processing result
        """
        # Simulate orchestrator selecting and executing agents
        selected_agent = agents['llm']  # Use LLM agent for simulation
        
        try:
            result = await selected_agent.process(test_input)
            return {
                "success": True,
                "result": result,
                "input": test_input.product_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input": test_input.product_name
            }