"""
Unit tests for Orchestrator agents.

This module contains comprehensive unit tests for Orchestrator agent implementations,
including StrandsMultiAgentOrchestrator, testing coordination, agent management,
and multiagent capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

from inference.src.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
from inference.src.models.data_models import (
    ProductInput, 
    InferenceResult,
    AgentHealth,
    LanguageHint
)
from inference.src.agents.base_agent import AgentError, AgentInitializationError
from inference.tests.utils.test_base import BaseAgentTest


class TestOrchestratorAgent(BaseAgentTest):
    """
    Unit tests for Orchestrator agent implementations extending BaseAgentTest.
    
    Tests coordination, agent management, multiagent capabilities, dynamic agent registration,
    configuration management, and error handling.
    """
    
    @pytest.fixture
    def orchestrator_config(self):
        """Standard orchestrator configuration for testing."""
        return {
            "confidence_threshold": 0.5,
            "max_parallel_agents": 4,
            "coordination_timeout": 30.0,
            "enable_multiagent": True
        }
    
    @pytest.fixture
    def mock_strands_agent(self):
        """Mock Strands Agent for orchestrator testing."""
        mock_agent = Mock()
        
        # Mock agent call responses
        def mock_call(prompt):
            if "Samsung" in prompt:
                return "Samsung"
            elif "Apple" in prompt or "iPhone" in prompt:
                return "Apple"
            else:
                return "Unknown"
        
        mock_agent.__call__ = mock_call
        mock_agent.name = "TestAgent"
        return mock_agent
    
    @pytest.fixture
    def mock_swarm_class(self):
        """Mock Swarm class for multiagent testing."""
        mock_swarm_class = Mock()
        mock_swarm_instance = Mock()
        
        # Mock swarm execution
        mock_swarm_instance.run.return_value = {
            "result": "Samsung",
            "confidence": 0.8,
            "agents_used": 3
        }
        
        mock_swarm_class.return_value = mock_swarm_instance
        return mock_swarm_class
    
    @pytest.fixture
    def mock_graph_builder(self):
        """Mock GraphBuilder for agent graph testing."""
        mock_builder = Mock()
        
        # Mock graph building and execution
        mock_builder.build.return_value = Mock()
        mock_builder.execute.return_value = {
            "result": "Samsung",
            "confidence": 0.85,
            "execution_path": ["ner", "rag", "llm"]
        }
        
        return mock_builder

    # Initialization Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_success(self, orchestrator_config):
        """Test successful orchestrator initialization."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            assert orchestrator.config == orchestrator_config
            assert orchestrator.confidence_threshold == 0.5
            assert orchestrator.max_parallel_agents == 4
            assert len(orchestrator.specialized_agents) == 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_tools(self, orchestrator_config):
        """Test orchestrator initialization with proper tools setup."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Verify that tools are properly set up
            assert hasattr(orchestrator, 'create_ner_agent')
            assert hasattr(orchestrator, 'create_rag_agent')
            assert hasattr(orchestrator, 'create_llm_agent')
            assert hasattr(orchestrator, 'coordinate_inference')

    # Agent Creation Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_create_ner_agent(self, orchestrator_config):
        """Test NER agent creation through orchestrator."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            agent_id = orchestrator.create_ner_agent()
            
            assert agent_id.startswith("ner_agent_")
            assert agent_id in orchestrator.specialized_agents
            assert orchestrator.specialized_agents[agent_id] is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_create_rag_agent(self, orchestrator_config):
        """Test RAG agent creation through orchestrator."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            agent_id = orchestrator.create_rag_agent()
            
            assert agent_id.startswith("rag_agent_")
            assert agent_id in orchestrator.specialized_agents
            assert orchestrator.specialized_agents[agent_id] is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_create_llm_agent(self, orchestrator_config):
        """Test LLM agent creation through orchestrator."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            agent_id = orchestrator.create_llm_agent()
            
            assert agent_id.startswith("llm_agent_")
            assert agent_id in orchestrator.specialized_agents
            assert orchestrator.specialized_agents[agent_id] is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_create_finetuned_nova_agent(self, orchestrator_config):
        """Test fine-tuned Nova agent creation through orchestrator."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            agent_id = orchestrator.create_finetuned_nova_agent()
            
            assert agent_id.startswith("finetuned_nova_agent_")
            assert agent_id in orchestrator.specialized_agents
            assert orchestrator.specialized_agents[agent_id] is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_create_hybrid_agent(self, orchestrator_config):
        """Test hybrid agent creation through orchestrator."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            agent_id = orchestrator.create_hybrid_agent()
            
            assert agent_id.startswith("hybrid_agent_")
            assert agent_id in orchestrator.specialized_agents
            assert orchestrator.specialized_agents[agent_id] is not None

    # Coordination Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_coordinate_inference_swarm(self, orchestrator_config, mock_strands_agent):
        """Test inference coordination using swarm method."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create some agents first
            orchestrator.create_ner_agent()
            orchestrator.create_rag_agent()
            orchestrator.create_llm_agent()
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "swarm"
            )
            
            assert isinstance(result, dict)
            assert result["method"] == "swarm"
            assert result["product_name"] == "Samsung Galaxy S23"
            assert "results" in result
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_orchestrator_coordinate_inference_graph(self, orchestrator_config, mock_strands_agent):
        """Test inference coordination using agent graph method."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create some agents first
            orchestrator.create_ner_agent()
            orchestrator.create_rag_agent()
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "graph"
            )
            
            assert isinstance(result, dict)
            assert result["method"] == "agent_graph"
            assert result["product_name"] == "Samsung Galaxy S23"
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_orchestrator_coordinate_inference_enhanced(self, orchestrator_config, mock_strands_agent):
        """Test inference coordination using enhanced coordination method."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create some agents first
            orchestrator.create_finetuned_nova_agent()
            orchestrator.create_llm_agent()
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "enhanced"
            )
            
            assert isinstance(result, dict)
            assert result["method"] == "enhanced"
            assert result["product_name"] == "Samsung Galaxy S23"
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_orchestrator_coordinate_inference_no_agents(self, orchestrator_config, mock_strands_agent):
        """Test inference coordination when no agents are available."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Don't create any specialized agents
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "swarm"
            )
            
            # Should fall back to some default behavior
            assert isinstance(result, dict)
            assert result["product_name"] == "Samsung Galaxy S23"

    # Brand Parsing Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_brand_parsing_success(self, orchestrator_config, mock_strands_agent):
        """Test successful brand parsing from agent responses."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Test parsing clear brand response
            brand = orchestrator._parse_brand_from_response("Samsung", "test_agent")
            assert brand == "Samsung"
            
            # Test parsing response with explanation
            brand = orchestrator._parse_brand_from_response(
                "The brand is Apple based on the iPhone model", "test_agent"
            )
            assert brand == "Apple"
    
    @pytest.mark.asyncio
    async def test_orchestrator_brand_parsing_edge_cases(self, orchestrator_config, mock_strands_agent):
        """Test brand parsing with edge cases."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Test empty response
            brand = orchestrator._parse_brand_from_response("", "test_agent")
            assert brand == "Unknown"
            
            # Test response with no clear brand
            brand = orchestrator._parse_brand_from_response(
                "I cannot determine the brand", "test_agent"
            )
            assert brand == "Unknown"
            
            # Test response with common brand names
            brand = orchestrator._parse_brand_from_response("Samsung Galaxy", "test_agent")
            assert brand == "Samsung"
    
    @pytest.mark.asyncio
    async def test_orchestrator_direct_brand_extraction(self, orchestrator_config, mock_strands_agent):
        """Test direct brand extraction from product names."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Test extraction from clear product names
            brand = orchestrator._extract_brand_from_product_name("Samsung Galaxy S23")
            assert brand == "Samsung"
            
            brand = orchestrator._extract_brand_from_product_name("Alectric Smart Fan")
            assert brand == "Alectric"
            
            # Test extraction failure
            brand = orchestrator._extract_brand_from_product_name("generic product")
            assert brand == "Unknown"

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_coordination_error_handling(self, orchestrator_config):
        """Test orchestrator error handling during coordination."""
        with patch('strands.Agent') as mock_agent_class:
            # Mock agent that raises exception
            mock_agent = Mock()
            mock_agent.side_effect = Exception("Agent failed")
            mock_agent_class.return_value = mock_agent
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create agents that will fail
            orchestrator.create_ner_agent()
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "swarm"
            )
            
            # Should handle errors gracefully
            assert isinstance(result, dict)
            assert "results" in result
            # Some agents may have errors in their results
    
    @pytest.mark.asyncio
    async def test_orchestrator_swarm_coordination_failure(self, orchestrator_config, mock_strands_agent):
        """Test orchestrator handling of swarm coordination failures."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create agents
            orchestrator.create_ner_agent()
            
            # Mock swarm failure by patching the coordination method
            with patch.object(orchestrator, '_coordinate_with_swarm', side_effect=Exception("Swarm failed")):
                with pytest.raises(AgentError) as exc_info:
                    await orchestrator._coordinate_inference_internal("Samsung Galaxy S23", "swarm")
                
                assert "Swarm coordination failed" in str(exc_info.value)

    # Agent Management Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_multiple_agent_creation(self, orchestrator_config):
        """Test creating multiple agents of the same type."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create multiple NER agents
            agent_id_1 = orchestrator.create_ner_agent()
            agent_id_2 = orchestrator.create_ner_agent()
            
            assert agent_id_1 != agent_id_2
            assert agent_id_1 in orchestrator.specialized_agents
            assert agent_id_2 in orchestrator.specialized_agents
            assert len(orchestrator.specialized_agents) == 2
    
    @pytest.mark.asyncio
    async def test_orchestrator_agent_registry_integration(self, orchestrator_config, mock_strands_agent):
        """Test orchestrator integration with agent registry."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Mock agent registry
            mock_registry = {
                "ner_agent": Mock(),
                "rag_agent": Mock(),
                "llm_agent": Mock()
            }
            
            # Set up mock registry agents
            for agent_name, agent in mock_registry.items():
                agent.process = AsyncMock(return_value={
                    "success": True,
                    "result": Mock(predicted_brand="Samsung", confidence=0.8),
                    "error": None
                })
            
            orchestrator.agents = mock_registry
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "enhanced"
            )
            
            assert isinstance(result, dict)
            assert "results" in result

    # Configuration Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_configuration_validation(self, orchestrator_config):
        """Test orchestrator configuration validation."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            # Test valid configuration
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            assert orchestrator.confidence_threshold == 0.5
            assert orchestrator.max_parallel_agents == 4
            
            # Test configuration with missing values (should use defaults)
            minimal_config = {}
            orchestrator = StrandsMultiAgentOrchestrator(minimal_config)
            assert orchestrator.confidence_threshold == 0.5  # Default value
    
    @pytest.mark.asyncio
    async def test_orchestrator_dynamic_configuration_update(self, orchestrator_config):
        """Test dynamic configuration updates."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Update configuration
            orchestrator.confidence_threshold = 0.7
            orchestrator.max_parallel_agents = 6
            
            assert orchestrator.confidence_threshold == 0.7
            assert orchestrator.max_parallel_agents == 6

    # Multiagent Capability Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_multiagent_availability_check(self, orchestrator_config):
        """Test multiagent capability availability check."""
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Check multiagent availability (depends on imports)
            assert hasattr(orchestrator, 'MULTIAGENT_AVAILABLE')
            # The actual value depends on whether strands.multiagent is available
    
    @pytest.mark.asyncio
    async def test_orchestrator_swarm_configuration(self, orchestrator_config, mock_strands_agent):
        """Test swarm configuration and setup."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create agents for swarm
            orchestrator.create_ner_agent()
            orchestrator.create_rag_agent()
            
            result = await orchestrator._coordinate_inference_internal(
                "Samsung Galaxy S23", "swarm"
            )
            
            assert isinstance(result, dict)
            assert "swarm_config" in result
            assert result["swarm_config"]["nodes"] >= 2

    # Performance Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_concurrent_coordination(self, orchestrator_config, mock_strands_agent):
        """Test orchestrator handling concurrent coordination requests."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create agents
            orchestrator.create_ner_agent()
            orchestrator.create_rag_agent()
            
            # Create multiple concurrent coordination tasks
            tasks = []
            for i in range(3):
                task = orchestrator._coordinate_inference_internal(
                    f"Samsung Galaxy S{20 + i}", "swarm"
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for result in results:
                assert isinstance(result, dict)
                assert "results" in result
    
    @pytest.mark.asyncio
    async def test_orchestrator_agent_limit_enforcement(self, orchestrator_config):
        """Test orchestrator enforcement of agent limits."""
        config = orchestrator_config.copy()
        config["max_parallel_agents"] = 2
        
        with patch('strands.Agent') as mock_agent_class:
            mock_agent_class.return_value = Mock()
            
            orchestrator = StrandsMultiAgentOrchestrator(config)
            
            # Create agents up to limit
            agent_ids = []
            for i in range(5):  # Try to create more than limit
                agent_id = orchestrator.create_ner_agent()
                agent_ids.append(agent_id)
            
            # All agents should be created (limit applies to parallel execution, not creation)
            assert len(orchestrator.specialized_agents) == 5

    # Parameterized Tests
    
    @pytest.mark.parametrize("coordination_method", [
        "swarm",
        "graph", 
        "enhanced",
        "unknown_method"  # Should fall back to enhanced
    ])
    @pytest.mark.asyncio
    async def test_orchestrator_coordination_methods(self, orchestrator_config, mock_strands_agent, coordination_method):
        """Test orchestrator with different coordination methods."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create some agents
            orchestrator.create_ner_agent()
            orchestrator.create_rag_agent()
            
            if coordination_method == "workflow":
                # Workflow method should raise an error in current implementation
                with pytest.raises(AgentError):
                    await orchestrator._coordinate_inference_internal(
                        "Samsung Galaxy S23", coordination_method
                    )
            else:
                result = await orchestrator._coordinate_inference_internal(
                    "Samsung Galaxy S23", coordination_method
                )
                
                assert isinstance(result, dict)
                assert result["product_name"] == "Samsung Galaxy S23"
    
    @pytest.mark.parametrize("product_name,expected_brand_hint", [
        ("Samsung Galaxy S23", "Samsung"),
        ("iPhone 15 Pro", "Apple"),
        ("Sony WH-1000XM4", "Sony"),
        ("Alectric Smart Fan", "Alectric"),
        ("Generic Product", "Unknown"),
    ])
    @pytest.mark.asyncio
    async def test_orchestrator_brand_extraction_scenarios(self, orchestrator_config, mock_strands_agent,
                                                          product_name, expected_brand_hint):
        """Test orchestrator brand extraction with various product scenarios."""
        with patch('strands.Agent', return_value=mock_strands_agent):
            orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Create agents
            orchestrator.create_ner_agent()
            orchestrator.create_llm_agent()
            
            result = await orchestrator._coordinate_inference_internal(product_name, "enhanced")
            
            assert isinstance(result, dict)
            assert result["product_name"] == product_name
            
            # Check if any agent predicted the expected brand
            if expected_brand_hint != "Unknown":
                results = result.get("results", {})
                brand_predictions = [
                    r.get("prediction") for r in results.values() 
                    if isinstance(r, dict) and r.get("prediction")
                ]
                # At least one agent should predict the expected brand or extract it directly
                assert any(expected_brand_hint in str(pred) for pred in brand_predictions) or \
                       orchestrator._extract_brand_from_product_name(product_name) == expected_brand_hint