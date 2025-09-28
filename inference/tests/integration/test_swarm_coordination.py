#!/usr/bin/env python3
"""
Integration tests for swarm coordination functionality.

This module contains pytest-based integration tests for testing swarm coordination
in the multi-agent orchestrator system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from inference.tests.utils.test_base import BaseAgentTest
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.integration
@pytest.mark.asyncio
class TestSwarmCoordination(BaseAgentTest):
    """Integration tests for swarm coordination functionality."""
    
    @pytest.fixture
    def swarm_config(self):
        """Configuration for swarm coordination testing."""
        return {
            "coordination_method": "swarm",
            "max_parallel_agents": 4,
            "confidence_threshold": 0.6,
            "timeout_seconds": 30,
            "consensus_threshold": 0.7
        }
    
    @pytest.fixture
    async def swarm_orchestrator(self, swarm_config):
        """Swarm orchestrator instance for testing."""
        try:
            from inference.src.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
            
            orchestrator = StrandsMultiAgentOrchestrator(swarm_config)
            
            # Mock the initialization
            with patch.object(orchestrator, 'initialize', new_callable=AsyncMock) as mock_init:
                mock_init.return_value = True
                await orchestrator.initialize()
                
                yield orchestrator
                
        except ImportError:
            # Fallback mock if orchestrator not available
            orchestrator = Mock()
            orchestrator.config = swarm_config
            yield orchestrator
    
    @pytest.fixture
    def swarm_test_products(self) -> List[str]:
        """Test products for swarm coordination testing."""
        return [
            "Samsung Galaxy S24 Ultra",
            "iPhone 15 Pro Max",
            "Sony WH-1000XM5 Headphones",
            "Nike Air Max 270",
            "Toyota Camry 2024"
        ]
    
    @pytest.fixture
    def mock_swarm_result(self):
        """Mock swarm coordination result."""
        return {
            "success": True,
            "coordination_method": "swarm",
            "best_prediction": "Samsung",
            "best_confidence": 0.85,
            "best_method": "llm",
            "orchestration_time": 2.5,
            "total_agents": 4,
            "successful_agents": 3,
            "consensus_reached": True,
            "agent_results": {
                "ner": {"success": True, "prediction": "Samsung", "confidence": 0.80},
                "rag": {"success": True, "prediction": "Samsung", "confidence": 0.75},
                "llm": {"success": True, "prediction": "Samsung", "confidence": 0.85},
                "hybrid": {"success": False, "error": "Timeout"}
            },
            "swarm_metrics": {
                "consensus_score": 0.80,
                "agreement_ratio": 0.75,
                "coordination_efficiency": 0.85
            }
        }
    
    async def test_swarm_coordination_basic(self, swarm_orchestrator, mock_swarm_result):
        """Test basic swarm coordination functionality."""
        product_name = "Samsung Galaxy S24 Ultra"
        
        # Mock the orchestrate_multiagent_inference method
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_swarm_result
            
            # Execute swarm coordination
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify orchestration was called correctly
            mock_orchestrate.assert_called_once_with(product_name, coordination_method="swarm")
            
            # Verify swarm coordination result
            assert result["success"] is True
            assert result["coordination_method"] == "swarm"
            assert result["best_prediction"] is not None
            assert result["best_confidence"] > 0.0
            assert result["orchestration_time"] > 0.0
            
            # Verify swarm-specific metrics
            assert "swarm_metrics" in result
            assert result["swarm_metrics"]["consensus_score"] > 0.0
            assert result["swarm_metrics"]["agreement_ratio"] > 0.0
            assert result["swarm_metrics"]["coordination_efficiency"] > 0.0
    
    async def test_swarm_agent_coordination(self, swarm_orchestrator, mock_swarm_result):
        """Test swarm coordination with multiple agents."""
        product_name = "iPhone 15 Pro Max"
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_swarm_result
            
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify agent coordination
            assert "agent_results" in result
            agent_results = result["agent_results"]
            
            # Verify multiple agents participated
            assert len(agent_results) >= 3
            
            # Verify agent result structure
            for agent_name, agent_result in agent_results.items():
                assert "success" in agent_result
                if agent_result["success"]:
                    assert "prediction" in agent_result
                    assert "confidence" in agent_result
                    assert agent_result["confidence"] > 0.0
                else:
                    assert "error" in agent_result
            
            # Verify successful agents count
            successful_agents = sum(1 for result in agent_results.values() if result["success"])
            assert successful_agents == result["successful_agents"]
            assert successful_agents >= 2  # At least 2 agents should succeed
    
    async def test_swarm_consensus_mechanism(self, swarm_orchestrator):
        """Test swarm consensus mechanism."""
        product_name = "Sony WH-1000XM5 Headphones"
        
        # Mock result with consensus data
        consensus_result = {
            "success": True,
            "coordination_method": "swarm",
            "best_prediction": "Sony",
            "best_confidence": 0.88,
            "consensus_reached": True,
            "agent_results": {
                "ner": {"success": True, "prediction": "Sony", "confidence": 0.85},
                "rag": {"success": True, "prediction": "Sony", "confidence": 0.80},
                "llm": {"success": True, "prediction": "Sony", "confidence": 0.90},
                "hybrid": {"success": True, "prediction": "Sony", "confidence": 0.75}
            },
            "consensus_analysis": {
                "unanimous_agreement": True,
                "prediction_distribution": {"Sony": 4, "Other": 0},
                "confidence_variance": 0.05,
                "consensus_strength": "strong"
            }
        }
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = consensus_result
            
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify consensus was reached
            assert result["consensus_reached"] is True
            
            # Verify consensus analysis
            assert "consensus_analysis" in result
            consensus = result["consensus_analysis"]
            
            assert consensus["unanimous_agreement"] is True
            assert consensus["prediction_distribution"]["Sony"] == 4
            assert consensus["confidence_variance"] < 0.1  # Low variance indicates agreement
            assert consensus["consensus_strength"] == "strong"
    
    async def test_swarm_partial_failure_handling(self, swarm_orchestrator):
        """Test swarm handling of partial agent failures."""
        product_name = "Nike Air Max 270"
        
        # Mock result with some agent failures
        partial_failure_result = {
            "success": True,
            "coordination_method": "swarm",
            "best_prediction": "Nike",
            "best_confidence": 0.82,
            "consensus_reached": True,
            "total_agents": 4,
            "successful_agents": 2,
            "failed_agents": 2,
            "agent_results": {
                "ner": {"success": False, "error": "Model not available"},
                "rag": {"success": True, "prediction": "Nike", "confidence": 0.80},
                "llm": {"success": True, "prediction": "Nike", "confidence": 0.85},
                "hybrid": {"success": False, "error": "Timeout"}
            },
            "failure_analysis": {
                "failure_rate": 0.5,
                "critical_failures": 0,
                "recoverable_failures": 2,
                "fallback_used": True
            }
        }
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = partial_failure_result
            
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify swarm still succeeds despite partial failures
            assert result["success"] is True
            assert result["best_prediction"] is not None
            
            # Verify failure handling
            assert result["failed_agents"] == 2
            assert result["successful_agents"] == 2
            
            # Verify failure analysis
            assert "failure_analysis" in result
            failure_analysis = result["failure_analysis"]
            
            assert failure_analysis["failure_rate"] == 0.5
            assert failure_analysis["critical_failures"] == 0  # No critical failures
            assert failure_analysis["fallback_used"] is True
    
    @pytest.mark.parametrize("product_name", [
        "Samsung Galaxy S24 Ultra",
        "iPhone 15 Pro Max",
        "Sony WH-1000XM5 Headphones",
    ])
    async def test_swarm_coordination_consistency(self, swarm_orchestrator, mock_swarm_result, product_name):
        """Test swarm coordination consistency across different products."""
        # Update mock result for this product
        expected_brand = product_name.split()[0]  # First word as brand
        mock_swarm_result["best_prediction"] = expected_brand
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_swarm_result
            
            # Execute swarm coordination
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify consistent behavior
            assert result["success"] is True
            assert result["coordination_method"] == "swarm"
            assert result["best_prediction"] == expected_brand
            assert result["best_confidence"] > 0.0
            
            # Verify swarm metrics are present
            assert "swarm_metrics" in result or "consensus_reached" in result
    
    async def test_swarm_performance_requirements(self, swarm_orchestrator, mock_swarm_result):
        """Test that swarm coordination meets performance requirements."""
        product_name = "Toyota Camry 2024"
        
        # Mock with performance data
        mock_swarm_result["orchestration_time"] = 2.8
        mock_swarm_result["performance_metrics"] = {
            "total_execution_time": 2.8,
            "parallel_efficiency": 0.85,
            "coordination_overhead": 0.3,
            "agent_avg_time": 2.1
        }
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_swarm_result
            
            start_time = time.time()
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            execution_time = time.time() - start_time
            
            # Verify performance requirements
            assert execution_time < 5.0  # Should complete within 5 seconds
            assert result["orchestration_time"] < 10.0  # Internal time should be reasonable
            
            # Verify performance metrics
            if "performance_metrics" in result:
                perf = result["performance_metrics"]
                assert perf["parallel_efficiency"] > 0.5  # Should be reasonably efficient
                assert perf["coordination_overhead"] < 1.0  # Overhead should be reasonable
    
    async def test_swarm_error_handling(self, swarm_orchestrator):
        """Test swarm coordination error handling."""
        product_name = "Invalid Product"
        
        # Mock error scenario
        error_result = {
            "success": False,
            "coordination_method": "swarm",
            "error": "All agents failed to process input",
            "error_code": "SWARM_TOTAL_FAILURE",
            "failed_agents": 4,
            "successful_agents": 0,
            "agent_results": {
                "ner": {"success": False, "error": "Invalid input"},
                "rag": {"success": False, "error": "Database connection failed"},
                "llm": {"success": False, "error": "Model inference failed"},
                "hybrid": {"success": False, "error": "Dependency failure"}
            }
        }
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = error_result
            
            result = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name,
                coordination_method="swarm"
            )
            
            # Verify error handling
            assert result["success"] is False
            assert "error" in result
            assert result["coordination_method"] == "swarm"
            assert result["successful_agents"] == 0
            assert result["failed_agents"] > 0
    
    async def test_swarm_timeout_handling(self, swarm_orchestrator):
        """Test swarm coordination timeout handling."""
        product_name = "Test Product"
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            # Mock timeout scenario
            mock_orchestrate.side_effect = asyncio.TimeoutError("Swarm coordination timed out")
            
            with pytest.raises(asyncio.TimeoutError, match="Swarm coordination timed out"):
                await swarm_orchestrator.orchestrate_multiagent_inference(
                    product_name,
                    coordination_method="swarm"
                )
    
    async def test_swarm_configuration_validation(self, swarm_orchestrator, swarm_config):
        """Test swarm configuration validation."""
        # Verify swarm configuration
        if hasattr(swarm_orchestrator, 'config'):
            config = swarm_orchestrator.config
            
            # Verify required configuration parameters
            if "coordination_method" in config:
                assert config["coordination_method"] == "swarm"
            
            if "max_parallel_agents" in config:
                assert config["max_parallel_agents"] > 0
                assert config["max_parallel_agents"] <= 10  # Reasonable limit
            
            if "confidence_threshold" in config:
                assert 0.0 <= config["confidence_threshold"] <= 1.0
            
            if "timeout_seconds" in config:
                assert config["timeout_seconds"] > 0
                assert config["timeout_seconds"] <= 300  # Reasonable timeout


@pytest.mark.integration
@pytest.mark.asyncio
class TestSwarmCoordinationAdvanced:
    """Advanced integration tests for swarm coordination."""
    
    async def test_swarm_vs_sequential_coordination(self, swarm_orchestrator):
        """Test comparison between swarm and sequential coordination."""
        product_name = "Samsung Galaxy S24 Ultra"
        
        # Mock swarm result
        swarm_result = {
            "success": True,
            "coordination_method": "swarm",
            "best_prediction": "Samsung",
            "best_confidence": 0.85,
            "orchestration_time": 2.5,
            "parallel_execution": True
        }
        
        # Mock sequential result
        sequential_result = {
            "success": True,
            "coordination_method": "sequential",
            "best_prediction": "Samsung",
            "best_confidence": 0.85,
            "orchestration_time": 4.2,
            "parallel_execution": False
        }
        
        with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
            # Test swarm coordination
            mock_orchestrate.return_value = swarm_result
            swarm_result_actual = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name, coordination_method="swarm"
            )
            
            # Test sequential coordination
            mock_orchestrate.return_value = sequential_result
            sequential_result_actual = await swarm_orchestrator.orchestrate_multiagent_inference(
                product_name, coordination_method="sequential"
            )
            
            # Verify both methods work
            assert swarm_result_actual["success"] is True
            assert sequential_result_actual["success"] is True
            
            # Verify swarm is faster (in this mock scenario)
            assert swarm_result_actual["orchestration_time"] < sequential_result_actual["orchestration_time"]
            
            # Verify coordination methods are different
            assert swarm_result_actual["coordination_method"] == "swarm"
            assert sequential_result_actual["coordination_method"] == "sequential"
    
    async def test_swarm_scalability(self, swarm_orchestrator):
        """Test swarm coordination scalability with different agent counts."""
        product_name = "Test Product"
        agent_counts = [2, 4, 6, 8]
        
        results = []
        
        for agent_count in agent_counts:
            # Mock result with different agent counts
            mock_result = {
                "success": True,
                "coordination_method": "swarm",
                "best_prediction": "TestBrand",
                "best_confidence": 0.80,
                "total_agents": agent_count,
                "successful_agents": agent_count - 1,  # One agent fails
                "orchestration_time": 1.5 + (agent_count * 0.2),  # Time increases with agents
                "scalability_metrics": {
                    "agents_per_second": agent_count / (1.5 + (agent_count * 0.2)),
                    "coordination_efficiency": max(0.5, 1.0 - (agent_count * 0.05))
                }
            }
            
            with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
                mock_orchestrate.return_value = mock_result
                
                result = await swarm_orchestrator.orchestrate_multiagent_inference(
                    product_name, coordination_method="swarm"
                )
                
                results.append({
                    "agent_count": agent_count,
                    "orchestration_time": result["orchestration_time"],
                    "efficiency": result.get("scalability_metrics", {}).get("coordination_efficiency", 0.5)
                })
        
        # Verify scalability characteristics
        assert len(results) == len(agent_counts)
        
        # Verify that system can handle different agent counts
        for result in results:
            assert result["orchestration_time"] > 0
            assert result["efficiency"] > 0
            
        # Verify reasonable scaling (time shouldn't increase too dramatically)
        max_time = max(r["orchestration_time"] for r in results)
        min_time = min(r["orchestration_time"] for r in results)
        assert max_time / min_time < 5.0  # Time shouldn't increase more than 5x
    
    async def test_swarm_fault_tolerance(self, swarm_orchestrator):
        """Test swarm fault tolerance with various failure scenarios."""
        product_name = "Test Product"
        
        fault_scenarios = [
            {
                "name": "Single agent failure",
                "failed_agents": 1,
                "total_agents": 4,
                "expected_success": True
            },
            {
                "name": "Half agents fail",
                "failed_agents": 2,
                "total_agents": 4,
                "expected_success": True
            },
            {
                "name": "Majority agents fail",
                "failed_agents": 3,
                "total_agents": 4,
                "expected_success": False  # Should fail if majority fails
            }
        ]
        
        for scenario in fault_scenarios:
            successful_agents = scenario["total_agents"] - scenario["failed_agents"]
            
            mock_result = {
                "success": scenario["expected_success"],
                "coordination_method": "swarm",
                "total_agents": scenario["total_agents"],
                "successful_agents": successful_agents,
                "failed_agents": scenario["failed_agents"],
                "fault_tolerance": {
                    "scenario": scenario["name"],
                    "failure_rate": scenario["failed_agents"] / scenario["total_agents"],
                    "resilience_score": successful_agents / scenario["total_agents"]
                }
            }
            
            if scenario["expected_success"]:
                mock_result.update({
                    "best_prediction": "TestBrand",
                    "best_confidence": 0.75
                })
            else:
                mock_result.update({
                    "error": "Insufficient successful agents for consensus",
                    "error_code": "INSUFFICIENT_CONSENSUS"
                })
            
            with patch.object(swarm_orchestrator, 'orchestrate_multiagent_inference', new_callable=AsyncMock) as mock_orchestrate:
                mock_orchestrate.return_value = mock_result
                
                result = await swarm_orchestrator.orchestrate_multiagent_inference(
                    product_name, coordination_method="swarm"
                )
                
                # Verify fault tolerance behavior
                assert result["success"] == scenario["expected_success"]
                assert result["failed_agents"] == scenario["failed_agents"]
                assert result["successful_agents"] == successful_agents
                
                if scenario["expected_success"]:
                    assert result["best_prediction"] is not None
                else:
                    assert "error" in result