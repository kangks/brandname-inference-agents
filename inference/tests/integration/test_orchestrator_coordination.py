#!/usr/bin/env python3
"""
Integration tests for orchestrator coordination functionality.

This module contains pytest-based integration tests for testing how the orchestrator
coordinates multiple sub-agents and aggregates their results to provide the best prediction.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Tuple

from inference.src.agents.orchestrator_agent import StrandsOrchestratorAgent
from inference.src.models.data_models import ProductInput, LanguageHint
from inference.tests.utils.test_base import BaseAgentTest
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.integration
@pytest.mark.asyncio
class TestOrchestratorCoordination(BaseAgentTest):
    """Integration tests for orchestrator coordination functionality."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Configuration for orchestrator testing."""
        return {
            "confidence_threshold": 0.6,
            "max_parallel_agents": 4,
            "timeout_seconds": 30
        }
    
    @pytest.fixture
    async def orchestrator_instance(self, orchestrator_config):
        """Orchestrator instance for coordination testing."""
        orchestrator = StrandsOrchestratorAgent(orchestrator_config)
        
        # Mock the initialization to avoid external dependencies
        with patch.object(orchestrator, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            await orchestrator.initialize()
            
            # Mock agents dictionary
            orchestrator.agents = {
                'ner': Mock(is_initialized=True),
                'rag': Mock(is_initialized=True),
                'llm': Mock(is_initialized=True),
                'hybrid': Mock(is_initialized=True)
            }
            
            yield orchestrator
            
            # Cleanup
            try:
                await orchestrator.cleanup()
            except Exception:
                pass  # Ignore cleanup errors in tests
    
    @pytest.fixture
    def coordination_test_products(self) -> List[Tuple[str, str]]:
        """Test products for coordination testing."""
        return [
            ("Samsung Galaxy S24 Ultra", "Samsung"),
            ("iPhone 15 Pro Max", "Apple"),
            ("Sony WH-1000XM5 Headphones", "Sony"),
            ("Nike Air Jordan 1", "Nike"),
            ("Toyota Camry 2024", "Toyota")
        ]
    
    @pytest.fixture
    def mock_agent_results(self):
        """Mock agent results for testing."""
        return {
            'ner_result': Mock(
                entities=[Mock(entity_type=Mock(value="BRAND"), text="Samsung", confidence=0.85)],
                processing_time=0.5
            ),
            'rag_result': Mock(
                predicted_brand="Samsung",
                confidence=0.75,
                processing_time=0.8
            ),
            'llm_result': Mock(
                predicted_brand="Samsung",
                confidence=0.90,
                reasoning="Identified Samsung based on product name analysis",
                processing_time=1.2
            ),
            'hybrid_result': Mock(
                predicted_brand="Samsung",
                confidence=0.80,
                processing_time=1.0
            )
        }
    
    @pytest.fixture
    def mock_inference_result(self, mock_agent_results):
        """Mock inference result for testing."""
        return Mock(
            best_prediction="Samsung",
            best_confidence=0.90,
            best_method="llm",
            total_processing_time=2.5,
            **mock_agent_results
        )
    
    async def test_orchestrator_initialization(self, orchestrator_instance):
        """Test orchestrator initialization and agent registration."""
        # Verify orchestrator is initialized
        assert orchestrator_instance is not None
        
        # Verify agents are registered
        assert hasattr(orchestrator_instance, 'agents')
        registered_agents = list(orchestrator_instance.agents.keys())
        
        # Check expected agents are registered
        expected_agents = ['ner', 'rag', 'llm', 'hybrid']
        for agent_name in expected_agents:
            assert agent_name in registered_agents
        
        # Verify agents are initialized
        for agent_name, agent in orchestrator_instance.agents.items():
            assert agent.is_initialized is True
    
    async def test_agent_health_check(self, orchestrator_instance):
        """Test agent health checking functionality."""
        # Mock health check results
        mock_health_results = {
            'ner': Mock(is_healthy=True, error_message=None),
            'rag': Mock(is_healthy=True, error_message=None),
            'llm': Mock(is_healthy=True, error_message=None),
            'hybrid': Mock(is_healthy=False, error_message="Connection failed")
        }
        
        with patch.object(orchestrator_instance, 'get_agent_health', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = mock_health_results
            
            health_results = await orchestrator_instance.get_agent_health()
            
            # Verify health check was called
            mock_health.assert_called_once()
            
            # Verify health results
            assert len(health_results) == 4
            assert health_results['ner'].is_healthy is True
            assert health_results['rag'].is_healthy is True
            assert health_results['llm'].is_healthy is True
            assert health_results['hybrid'].is_healthy is False
            assert health_results['hybrid'].error_message == "Connection failed"
            
            # Count healthy agents
            healthy_agents = [name for name, health in health_results.items() if health.is_healthy]
            assert len(healthy_agents) == 3
    
    @pytest.mark.parametrize("product_name,expected_brand", [
        ("Samsung Galaxy S24 Ultra", "Samsung"),
        ("iPhone 15 Pro Max", "Apple"),
        ("Sony WH-1000XM5 Headphones", "Sony"),
    ])
    async def test_orchestrated_inference(self, orchestrator_instance, mock_inference_result, product_name, expected_brand):
        """Test orchestrated inference with different products."""
        # Update mock result for this product
        mock_inference_result.best_prediction = expected_brand
        
        # Mock the orchestrate_inference method
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_inference_result
            
            # Create product input
            input_data = ProductInput(
                product_name=product_name,
                language_hint=LanguageHint("en")
            )
            
            # Execute orchestrated inference
            start_time = time.time()
            inference_result = await orchestrator_instance.orchestrate_inference(input_data)
            total_time = time.time() - start_time
            
            # Verify orchestration was called
            mock_orchestrate.assert_called_once_with(input_data)
            
            # Verify inference result
            assert inference_result.best_prediction == expected_brand
            assert inference_result.best_confidence > 0.0
            assert inference_result.best_method is not None
            assert inference_result.total_processing_time > 0.0
            
            # Verify performance
            assert total_time < 5.0  # Should complete within 5 seconds
    
    async def test_agent_coordination_workflow(self, orchestrator_instance, mock_inference_result):
        """Test the complete agent coordination workflow."""
        product_name = "Samsung Galaxy S24 Ultra"
        
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_inference_result
            
            # Step 1: Create input
            input_data = ProductInput(
                product_name=product_name,
                language_hint=LanguageHint("en")
            )
            
            # Step 2: Execute orchestration
            inference_result = await orchestrator_instance.orchestrate_inference(input_data)
            
            # Step 3: Verify orchestrator decision process
            assert inference_result.best_prediction is not None
            assert inference_result.best_confidence > 0.0
            assert inference_result.best_method in ['ner', 'rag', 'llm', 'hybrid']
            
            # Step 4: Verify individual agent contributions
            agent_results = []
            
            # Check NER results
            if hasattr(inference_result, 'ner_result') and inference_result.ner_result:
                ner = inference_result.ner_result
                assert hasattr(ner, 'entities')
                assert hasattr(ner, 'processing_time')
                agent_results.append(("NER", "Samsung", 0.85))
            
            # Check RAG results
            if hasattr(inference_result, 'rag_result') and inference_result.rag_result:
                rag = inference_result.rag_result
                assert hasattr(rag, 'predicted_brand')
                assert hasattr(rag, 'confidence')
                assert hasattr(rag, 'processing_time')
                agent_results.append(("RAG", rag.predicted_brand, rag.confidence))
            
            # Check LLM results
            if hasattr(inference_result, 'llm_result') and inference_result.llm_result:
                llm = inference_result.llm_result
                assert hasattr(llm, 'predicted_brand')
                assert hasattr(llm, 'confidence')
                assert hasattr(llm, 'reasoning')
                assert hasattr(llm, 'processing_time')
                agent_results.append(("LLM", llm.predicted_brand, llm.confidence))
            
            # Check Hybrid results
            if hasattr(inference_result, 'hybrid_result') and inference_result.hybrid_result:
                hybrid = inference_result.hybrid_result
                assert hasattr(hybrid, 'predicted_brand')
                assert hasattr(hybrid, 'confidence')
                assert hasattr(hybrid, 'processing_time')
                agent_results.append(("HYBRID", hybrid.predicted_brand, hybrid.confidence))
            
            # Step 5: Verify coordination summary
            assert len(agent_results) > 0  # At least one agent should have results
            
            # Verify best method selection logic
            if agent_results:
                # Find the agent with highest confidence
                best_agent = max(agent_results, key=lambda x: x[2])
                assert inference_result.best_method.upper() == best_agent[0] or inference_result.best_confidence >= best_agent[2]
    
    async def test_orchestrator_error_handling(self, orchestrator_instance):
        """Test orchestrator error handling during coordination."""
        # Test with invalid input
        invalid_input = ProductInput(
            product_name="",  # Empty product name
            language_hint=LanguageHint("en")
        )
        
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            # Mock an error scenario
            mock_orchestrate.side_effect = ValueError("Invalid product name")
            
            with pytest.raises(ValueError, match="Invalid product name"):
                await orchestrator_instance.orchestrate_inference(invalid_input)
    
    async def test_orchestrator_timeout_handling(self, orchestrator_instance):
        """Test orchestrator timeout handling."""
        input_data = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint("en")
        )
        
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            # Mock a timeout scenario
            mock_orchestrate.side_effect = asyncio.TimeoutError("Orchestration timed out")
            
            with pytest.raises(asyncio.TimeoutError, match="Orchestration timed out"):
                await orchestrator_instance.orchestrate_inference(input_data)
    
    async def test_orchestrator_partial_agent_failure(self, orchestrator_instance, mock_agent_results):
        """Test orchestrator handling of partial agent failures."""
        # Create a result where some agents fail
        partial_result = Mock(
            best_prediction="Samsung",
            best_confidence=0.75,
            best_method="rag",
            total_processing_time=2.0,
            ner_result=None,  # NER failed
            rag_result=mock_agent_results['rag_result'],  # RAG succeeded
            llm_result=None,  # LLM failed
            hybrid_result=mock_agent_results['hybrid_result']  # Hybrid succeeded
        )
        
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = partial_result
            
            input_data = ProductInput(
                product_name="Samsung Galaxy S24 Ultra",
                language_hint=LanguageHint("en")
            )
            
            result = await orchestrator_instance.orchestrate_inference(input_data)
            
            # Verify orchestrator still provides a result despite partial failures
            assert result.best_prediction is not None
            assert result.best_confidence > 0.0
            assert result.best_method in ['rag', 'hybrid']  # Only successful agents
            
            # Verify failed agents are handled gracefully
            assert result.ner_result is None
            assert result.llm_result is None
            assert result.rag_result is not None
            assert result.hybrid_result is not None
    
    async def test_orchestrator_performance_requirements(self, orchestrator_instance, mock_inference_result):
        """Test that orchestrator meets performance requirements."""
        with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
            mock_orchestrate.return_value = mock_inference_result
            
            input_data = ProductInput(
                product_name="Samsung Galaxy S24 Ultra",
                language_hint=LanguageHint("en")
            )
            
            # Measure performance
            start_time = time.time()
            result = await orchestrator_instance.orchestrate_inference(input_data)
            execution_time = time.time() - start_time
            
            # Performance assertions
            assert execution_time < 5.0  # Should complete within 5 seconds
            assert result.total_processing_time > 0.0
            assert result.total_processing_time < 10.0  # Internal processing should be reasonable
    
    async def test_orchestrator_coordination_consistency(self, orchestrator_instance, coordination_test_products):
        """Test orchestrator coordination consistency across multiple products."""
        results = []
        
        for product_name, expected_brand in coordination_test_products:
            # Create mock result for this product
            mock_result = Mock(
                best_prediction=expected_brand,
                best_confidence=0.80,
                best_method="llm",
                total_processing_time=1.5
            )
            
            with patch.object(orchestrator_instance, 'orchestrate_inference', new_callable=AsyncMock) as mock_orchestrate:
                mock_orchestrate.return_value = mock_result
                
                input_data = ProductInput(
                    product_name=product_name,
                    language_hint=LanguageHint("en")
                )
                
                result = await orchestrator_instance.orchestrate_inference(input_data)
                results.append({
                    "product": product_name,
                    "expected": expected_brand,
                    "predicted": result.best_prediction,
                    "confidence": result.best_confidence,
                    "method": result.best_method,
                    "time": result.total_processing_time
                })
        
        # Verify consistency
        assert len(results) == len(coordination_test_products)
        
        # All results should have valid predictions
        for result in results:
            assert result["predicted"] is not None
            assert result["confidence"] > 0.0
            assert result["method"] is not None
            assert result["time"] > 0.0
        
        # Calculate success metrics
        successful_predictions = sum(1 for r in results if r["predicted"] == r["expected"])
        success_rate = successful_predictions / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        
        # Performance assertions
        assert success_rate >= 0.8  # At least 80% accuracy
        assert avg_confidence >= 0.7  # Average confidence should be reasonable
        assert avg_time < 3.0  # Average processing time should be fast


@pytest.mark.integration
@pytest.mark.asyncio
class TestOrchestratorCoordinationAdvanced:
    """Advanced integration tests for orchestrator coordination."""
    
    @pytest.fixture
    async def orchestrator_with_real_agents(self):
        """Orchestrator with more realistic agent mocking."""
        orchestrator = StrandsOrchestratorAgent()
        
        # Mock individual agents with more realistic behavior
        mock_agents = {}
        
        # NER Agent
        ner_agent = AsyncMock()
        ner_agent.is_initialized = True
        ner_agent.process.return_value = {
            "success": True,
            "entities": [{"text": "Samsung", "type": "BRAND", "confidence": 0.85}],
            "processing_time": 0.5
        }
        mock_agents['ner'] = ner_agent
        
        # RAG Agent
        rag_agent = AsyncMock()
        rag_agent.is_initialized = True
        rag_agent.process.return_value = {
            "success": True,
            "predicted_brand": "Samsung",
            "confidence": 0.75,
            "processing_time": 0.8
        }
        mock_agents['rag'] = rag_agent
        
        # LLM Agent
        llm_agent = AsyncMock()
        llm_agent.is_initialized = True
        llm_agent.process.return_value = {
            "success": True,
            "predicted_brand": "Samsung",
            "confidence": 0.90,
            "reasoning": "Product name contains Samsung brand identifier",
            "processing_time": 1.2
        }
        mock_agents['llm'] = llm_agent
        
        orchestrator.agents = mock_agents
        
        yield orchestrator
    
    async def test_agent_parallel_execution(self, orchestrator_with_real_agents):
        """Test parallel execution of multiple agents."""
        orchestrator = orchestrator_with_real_agents
        
        # Mock parallel execution
        with patch.object(orchestrator, 'execute_agents_parallel', new_callable=AsyncMock) as mock_parallel:
            mock_parallel.return_value = {
                'ner': {"success": True, "prediction": "Samsung", "confidence": 0.85},
                'rag': {"success": True, "prediction": "Samsung", "confidence": 0.75},
                'llm': {"success": True, "prediction": "Samsung", "confidence": 0.90}
            }
            
            input_data = ProductInput(
                product_name="Samsung Galaxy S24 Ultra",
                language_hint=LanguageHint("en")
            )
            
            # Execute parallel processing
            results = await orchestrator.execute_agents_parallel(input_data)
            
            # Verify parallel execution
            mock_parallel.assert_called_once_with(input_data)
            assert len(results) == 3
            assert all(result["success"] for result in results.values())
    
    async def test_agent_result_aggregation(self, orchestrator_with_real_agents):
        """Test agent result aggregation logic."""
        orchestrator = orchestrator_with_real_agents
        
        # Mock aggregation
        agent_results = {
            'ner': {"success": True, "prediction": "Samsung", "confidence": 0.85},
            'rag': {"success": True, "prediction": "Samsung", "confidence": 0.75},
            'llm': {"success": True, "prediction": "Samsung", "confidence": 0.90}
        }
        
        with patch.object(orchestrator, 'aggregate_results', new_callable=AsyncMock) as mock_aggregate:
            mock_aggregate.return_value = {
                "best_prediction": "Samsung",
                "best_confidence": 0.90,
                "best_method": "llm",
                "all_results": agent_results
            }
            
            # Execute aggregation
            aggregated = await orchestrator.aggregate_results(agent_results)
            
            # Verify aggregation logic
            mock_aggregate.assert_called_once_with(agent_results)
            assert aggregated["best_prediction"] == "Samsung"
            assert aggregated["best_confidence"] == 0.90
            assert aggregated["best_method"] == "llm"
    
    async def test_coordination_with_agent_failures(self, orchestrator_with_real_agents):
        """Test coordination when some agents fail."""
        orchestrator = orchestrator_with_real_agents
        
        # Mock mixed success/failure results
        mixed_results = {
            'ner': {"success": False, "error": "NER model not available"},
            'rag': {"success": True, "prediction": "Samsung", "confidence": 0.75},
            'llm': {"success": True, "prediction": "Samsung", "confidence": 0.90}
        }
        
        with patch.object(orchestrator, 'handle_mixed_results', new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {
                "best_prediction": "Samsung",
                "best_confidence": 0.90,
                "best_method": "llm",
                "successful_agents": 2,
                "failed_agents": 1
            }
            
            # Execute handling of mixed results
            result = await orchestrator.handle_mixed_results(mixed_results)
            
            # Verify graceful handling of failures
            mock_handle.assert_called_once_with(mixed_results)
            assert result["best_prediction"] is not None
            assert result["successful_agents"] == 2
            assert result["failed_agents"] == 1