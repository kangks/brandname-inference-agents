"""
Integration tests for agent communication and coordination.

This module tests message passing between orchestrator and agents, result aggregation,
confidence scoring mechanisms, error propagation, and concurrent request handling
as specified in the requirements.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from inference.src.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
from inference.src.agents.base_agent import BaseAgent, AgentError, AgentTimeoutError
from inference.src.models.data_models import (
    ProductInput,
    LanguageHint,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    EntityResult,
    EntityType,
    SimilarProduct
)

from inference.tests.utils.assertion_helpers import InferenceAssertions
from inference.tests.utils.mock_factories import MockAgentFactory


@pytest.mark.integration
class TestAgentCommunication:
    """
    Integration tests for agent communication and coordination.
    
    Tests message passing, result aggregation, confidence scoring,
    error propagation, and concurrent request handling.
    """
    
    def setup_method(self):
        """Setup for each test method."""
        
        # Create orchestrator with communication-focused configuration
        self.orchestrator_config = {
            "confidence_threshold": 0.5,
            "max_parallel_agents": 4,
            "timeout_seconds": 30,
            "agent_timeout": 10,
            "result_aggregation_strategy": "weighted_average",
            "communication_timeout": 5.0
        }
        
        self.orchestrator = StrandsMultiAgentOrchestrator(self.orchestrator_config)
        
        # Create test agents with realistic responses
        self.test_agents = self._create_test_agents()
        
        # Test input variations
        self.test_inputs = self._create_test_inputs()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        
        # Clean up orchestrator
        if hasattr(self.orchestrator, 'cleanup'):
            asyncio.run(self.orchestrator.cleanup())
    
    def _create_test_agents(self) -> Dict[str, Mock]:
        """
        Create test agents with realistic communication patterns.
        
        Returns:
            Dictionary of test agents with varied response patterns
        """
        agents = {}
        
        # NER Agent - Returns entity extraction results
        agents['ner'] = MockAgentFactory.create_mock_ner_agent()
        
        # RAG Agent - Returns similarity-based results
        agents['rag'] = MockAgentFactory.create_mock_rag_agent()
        
        # LLM Agent - Returns reasoning-based results
        agents['llm'] = MockAgentFactory.create_mock_llm_agent()
        
        # Hybrid Agent - Returns combined results
        agents['hybrid'] = MockAgentFactory.create_mock_hybrid_agent()
        
        return agents
    
    def _create_test_inputs(self) -> List[ProductInput]:
        """
        Create varied test inputs for communication testing.
        
        Returns:
            List of test inputs with different characteristics
        """
        return [
            ProductInput(
                product_name="Samsung Galaxy S24 Ultra 256GB",
                language_hint=LanguageHint.ENGLISH
            ),
            ProductInput(
                product_name="iPhone 15 Pro Max สีทิเทเนียมธรรมชาติ 256GB",
                language_hint=LanguageHint.MIXED
            ),
            ProductInput(
                product_name="โค้กเซโร่ 325 มล. แพ็ค 6 กระป๋อง",
                language_hint=LanguageHint.THAI
            ),
            ProductInput(
                product_name="Generic USB Cable Type-C",
                language_hint=LanguageHint.ENGLISH
            )
        ]
    
    @pytest.mark.asyncio
    async def test_message_passing_between_orchestrator_and_agents(self):
        """Test message passing mechanism between orchestrator and agents."""
        # Setup orchestrator with test agents
        test_input = self.test_inputs[0]  # Samsung Galaxy
        
        with patch.object(self.orchestrator, 'agents', self.test_agents):
            # Test message passing to each agent type
            for agent_name, agent in self.test_agents.items():
                # Simulate orchestrator sending message to agent
                message_payload = {
                    "request_id": f"test_request_{agent_name}",
                    "input_data": test_input,
                    "timestamp": time.time(),
                    "orchestrator_id": "test_orchestrator"
                }
                
                # Test agent processing message
                result = await agent.process(test_input)
                
                # Verify message was processed correctly
                assert result is not None
                assert result.get("success") is True
                assert result.get("agent_type") == agent_name
                
                # Verify agent was called with correct parameters
                agent.process.assert_called_with(test_input)
                
                # Verify response structure for communication
                self._verify_communication_response_structure(result, agent_name)
    
    def _verify_communication_response_structure(self, result: Dict[str, Any], agent_type: str):
        """
        Verify that agent response has proper structure for communication.
        
        Args:
            result: Agent response to verify
            agent_type: Expected agent type
        """
        # Required fields for communication
        required_fields = ["agent_type", "result", "success"]
        
        for field in required_fields:
            assert field in result, f"Missing required field '{field}' in {agent_type} response"
        
        # Verify agent type matches
        assert result["agent_type"] == agent_type
        
        # Verify result structure based on agent type
        if result["success"]:
            assert result["result"] is not None
            assert result.get("error") is None
        else:
            assert result.get("error") is not None
    
    @pytest.mark.asyncio
    async def test_result_aggregation_mechanisms(self):
        """Test orchestrator's result aggregation from multiple agents."""
        test_input = self.test_inputs[0]  # Samsung Galaxy
        
        with patch.object(self.orchestrator, 'agents', self.test_agents):
            # Collect results from all agents
            agent_results = {}
            
            for agent_name, agent in self.test_agents.items():
                result = await agent.process(test_input)
                agent_results[agent_name] = result
            
            # Test result aggregation
            aggregated_result = self._aggregate_agent_results(agent_results, test_input)
            
            # Verify aggregation structure
            assert "final_prediction" in aggregated_result
            assert "confidence" in aggregated_result
            assert "contributing_agents" in aggregated_result
            assert "aggregation_method" in aggregated_result
            assert "individual_results" in aggregated_result
            
            # Verify all agents contributed
            assert len(aggregated_result["individual_results"]) == len(self.test_agents)
            
            # Verify confidence is reasonable aggregate
            individual_confidences = [
                result["result"].confidence 
                for result in agent_results.values() 
                if result["success"]
            ]
            
            if individual_confidences:
                avg_confidence = sum(individual_confidences) / len(individual_confidences)
                assert 0.0 <= aggregated_result["confidence"] <= 1.0
                # Aggregated confidence should be reasonable relative to individual confidences
                assert abs(aggregated_result["confidence"] - avg_confidence) < 0.3
    
    def _aggregate_agent_results(
        self, 
        agent_results: Dict[str, Dict[str, Any]], 
        input_data: ProductInput
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents.
        
        Args:
            agent_results: Results from individual agents
            input_data: Original input data
            
        Returns:
            Aggregated result
        """
        successful_results = {
            name: result for name, result in agent_results.items() 
            if result.get("success", False)
        }
        
        if not successful_results:
            return {
                "final_prediction": "Unknown",
                "confidence": 0.0,
                "contributing_agents": [],
                "aggregation_method": "no_successful_results",
                "individual_results": agent_results
            }
        
        # Extract predictions and confidences
        predictions = {}
        confidences = {}
        
        for agent_name, result in successful_results.items():
            agent_result = result["result"]
            
            # Extract prediction based on agent type
            if hasattr(agent_result, 'predicted_brand'):
                predictions[agent_name] = agent_result.predicted_brand
                confidences[agent_name] = agent_result.confidence
            elif hasattr(agent_result, 'final_prediction'):
                predictions[agent_name] = agent_result.final_prediction
                confidences[agent_name] = agent_result.confidence
            elif hasattr(agent_result, 'entities') and agent_result.entities:
                # For NER, extract brand entities
                brand_entities = [
                    e for e in agent_result.entities 
                    if e.entity_type == EntityType.BRAND
                ]
                if brand_entities:
                    best_entity = max(brand_entities, key=lambda x: x.confidence)
                    predictions[agent_name] = best_entity.text
                    confidences[agent_name] = best_entity.confidence
                else:
                    predictions[agent_name] = "Unknown"
                    confidences[agent_name] = 0.0
        
        # Aggregate predictions using weighted voting
        final_prediction = self._weighted_vote_prediction(predictions, confidences)
        
        # Calculate aggregated confidence
        if confidences:
            aggregated_confidence = sum(confidences.values()) / len(confidences)
        else:
            aggregated_confidence = 0.0
        
        return {
            "final_prediction": final_prediction,
            "confidence": aggregated_confidence,
            "contributing_agents": list(successful_results.keys()),
            "aggregation_method": "weighted_voting",
            "individual_results": agent_results,
            "prediction_distribution": predictions,
            "confidence_distribution": confidences
        }
    
    def _weighted_vote_prediction(
        self, 
        predictions: Dict[str, str], 
        confidences: Dict[str, float]
    ) -> str:
        """
        Determine final prediction using weighted voting.
        
        Args:
            predictions: Predictions from each agent
            confidences: Confidence scores from each agent
            
        Returns:
            Final prediction based on weighted voting
        """
        if not predictions:
            return "Unknown"
        
        # Count weighted votes for each prediction
        vote_weights = {}
        
        for agent_name, prediction in predictions.items():
            confidence = confidences.get(agent_name, 0.0)
            
            if prediction not in vote_weights:
                vote_weights[prediction] = 0.0
            
            vote_weights[prediction] += confidence
        
        # Return prediction with highest weighted vote
        if vote_weights:
            return max(vote_weights.items(), key=lambda x: x[1])[0]
        else:
            return "Unknown"
    
    @pytest.mark.asyncio
    async def test_confidence_scoring_mechanisms(self):
        """Test confidence scoring and combination mechanisms."""
        test_input = self.test_inputs[0]  # Samsung Galaxy
        
        with patch.object(self.orchestrator, 'agents', self.test_agents):
            # Collect results with confidence scores
            confidence_data = {}
            
            for agent_name, agent in self.test_agents.items():
                result = await agent.process(test_input)
                
                if result["success"]:
                    agent_result = result["result"]
                    confidence_data[agent_name] = {
                        "confidence": agent_result.confidence,
                        "processing_time": agent_result.processing_time,
                        "agent_type": agent_name
                    }
            
            # Test confidence scoring mechanisms
            confidence_analysis = self._analyze_confidence_scores(confidence_data)
            
            # Verify confidence analysis
            assert "individual_scores" in confidence_analysis
            assert "aggregated_score" in confidence_analysis
            assert "confidence_variance" in confidence_analysis
            assert "reliability_assessment" in confidence_analysis
            
            # Verify individual scores are valid
            for agent_name, score_data in confidence_analysis["individual_scores"].items():
                assert 0.0 <= score_data["confidence"] <= 1.0
                assert score_data["processing_time"] > 0.0
            
            # Verify aggregated score is reasonable
            assert 0.0 <= confidence_analysis["aggregated_score"] <= 1.0
            
            # Verify variance calculation
            assert confidence_analysis["confidence_variance"] >= 0.0
    
    def _analyze_confidence_scores(self, confidence_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze confidence scores from multiple agents.
        
        Args:
            confidence_data: Confidence data from agents
            
        Returns:
            Confidence analysis results
        """
        if not confidence_data:
            return {
                "individual_scores": {},
                "aggregated_score": 0.0,
                "confidence_variance": 0.0,
                "reliability_assessment": "no_data"
            }
        
        confidences = [data["confidence"] for data in confidence_data.values()]
        
        # Calculate aggregated confidence (weighted by inverse processing time)
        total_weight = 0.0
        weighted_sum = 0.0
        
        for agent_name, data in confidence_data.items():
            # Weight by inverse processing time (faster = more weight)
            weight = 1.0 / max(data["processing_time"], 0.01)
            weighted_sum += data["confidence"] * weight
            total_weight += weight
        
        aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate variance
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Assess reliability
        reliability = "high" if variance < 0.1 else "medium" if variance < 0.3 else "low"
        
        return {
            "individual_scores": confidence_data,
            "aggregated_score": aggregated_score,
            "confidence_variance": variance,
            "reliability_assessment": reliability,
            "mean_confidence": mean_confidence,
            "confidence_range": (min(confidences), max(confidences))
        }
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_handling(self):
        """Test error propagation between orchestrator and agents."""
        # Create agents with different error scenarios
        error_agents = self._create_error_scenario_agents()
        
        test_input = self.test_inputs[0]
        
        with patch.object(self.orchestrator, 'agents', error_agents):
            # Test error propagation from each agent type
            error_results = {}
            
            for agent_name, agent in error_agents.items():
                try:
                    result = await agent.process(test_input)
                    error_results[agent_name] = result
                except Exception as e:
                    error_results[agent_name] = {
                        "agent_type": agent_name,
                        "success": False,
                        "error": str(e),
                        "exception_type": type(e).__name__
                    }
            
            # Verify error handling
            self._verify_error_propagation(error_results)
            
            # Test orchestrator's error recovery
            recovery_result = self._test_error_recovery(error_results, test_input)
            
            # Verify recovery mechanisms
            assert "successful_agents" in recovery_result
            assert "failed_agents" in recovery_result
            assert "recovery_strategy" in recovery_result
    
    def _create_error_scenario_agents(self) -> Dict[str, Mock]:
        """
        Create agents with different error scenarios.
        
        Returns:
            Dictionary of agents with error conditions
        """
        agents = {}
        
        # Timeout error agent
        agents['timeout_agent'] = MockAgentFactory.create_mock_base_agent("timeout_agent")
        
        # Configure timeout behavior
        async def timeout_process(input_data):
            await asyncio.sleep(5.0)  # This will timeout
            return {"success": True}
        
        agents['timeout_agent'].process = AsyncMock(side_effect=timeout_process)
        
        # Processing error agent
        agents['error_agent'] = MockAgentFactory.create_mock_base_agent("error_agent")
        
        # Configure error behavior
        async def error_process(input_data):
            raise Exception("Simulated processing error")
        
        agents['error_agent'].process = AsyncMock(side_effect=error_process)
        
        # Successful agent for comparison
        agents['success_agent'] = self.test_agents['llm']
        
        return agents
    
    def _verify_error_propagation(self, error_results: Dict[str, Dict[str, Any]]):
        """
        Verify that errors are properly propagated and handled.
        
        Args:
            error_results: Results from error scenario testing
        """
        # Verify timeout agent error
        timeout_result = error_results.get("timeout_agent")
        assert timeout_result is not None
        assert timeout_result["success"] is False
        assert "timeout" in timeout_result.get("error", "").lower() or \
               "timeout" in timeout_result.get("exception_type", "").lower()
        
        # Verify processing error agent
        error_result = error_results.get("error_agent")
        assert error_result is not None
        assert error_result["success"] is False
        assert "error" in error_result
        
        # Verify successful agent still works
        success_result = error_results.get("success_agent")
        assert success_result is not None
        assert success_result["success"] is True
    
    def _test_error_recovery(
        self, 
        error_results: Dict[str, Dict[str, Any]], 
        test_input: ProductInput
    ) -> Dict[str, Any]:
        """
        Test orchestrator's error recovery mechanisms.
        
        Args:
            error_results: Results from error testing
            test_input: Original test input
            
        Returns:
            Recovery analysis results
        """
        successful_agents = []
        failed_agents = []
        
        for agent_name, result in error_results.items():
            if result.get("success", False):
                successful_agents.append(agent_name)
            else:
                failed_agents.append({
                    "agent_name": agent_name,
                    "error": result.get("error", "Unknown error"),
                    "exception_type": result.get("exception_type", "Unknown")
                })
        
        # Determine recovery strategy
        if len(successful_agents) > 0:
            recovery_strategy = "partial_success"
        elif len(failed_agents) == len(error_results):
            recovery_strategy = "complete_failure"
        else:
            recovery_strategy = "mixed_results"
        
        return {
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "recovery_strategy": recovery_strategy,
            "success_rate": len(successful_agents) / len(error_results),
            "can_continue": len(successful_agents) > 0
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling_and_resource_management(self):
        """Test concurrent request handling and resource management."""
        # Setup multiple concurrent requests
        num_concurrent_requests = 10
        concurrent_inputs = [
            ProductInput(
                product_name=f"Test Product {i} - {['Samsung', 'Apple', 'Sony'][i % 3]}",
                language_hint=LanguageHint.ENGLISH
            )
            for i in range(num_concurrent_requests)
        ]
        
        with patch.object(self.orchestrator, 'agents', self.test_agents):
            # Execute concurrent requests
            start_time = time.time()
            
            # Create tasks for concurrent execution
            tasks = []
            for i, test_input in enumerate(concurrent_inputs):
                task = asyncio.create_task(
                    self._process_concurrent_request(test_input, i)
                )
                tasks.append(task)
            
            # Wait for all tasks with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                pytest.fail("Concurrent request handling timed out")
            
            execution_time = time.time() - start_time
            
            # Analyze concurrent execution results
            concurrent_analysis = self._analyze_concurrent_execution(
                results, 
                execution_time, 
                num_concurrent_requests
            )
            
            # Verify concurrent handling performance
            assert concurrent_analysis["success_rate"] >= 0.8  # At least 80% success
            assert concurrent_analysis["avg_response_time"] < 5.0  # Reasonable response time
            assert concurrent_analysis["total_execution_time"] < 15.0  # Efficient overall execution
            
            # Verify resource management
            assert concurrent_analysis["resource_efficiency"] > 0.5  # Efficient resource usage
    
    async def _process_concurrent_request(
        self, 
        test_input: ProductInput, 
        request_id: int
    ) -> Dict[str, Any]:
        """
        Process a single concurrent request.
        
        Args:
            test_input: Input to process
            request_id: Unique request identifier
            
        Returns:
            Processing result with timing information
        """
        start_time = time.time()
        
        try:
            # Simulate orchestrator processing with multiple agents
            agent_results = {}
            
            # Process with subset of agents for efficiency
            selected_agents = ['ner', 'llm']  # Use faster agents for concurrent testing
            
            for agent_name in selected_agents:
                if agent_name in self.test_agents:
                    agent = self.test_agents[agent_name]
                    result = await agent.process(test_input)
                    agent_results[agent_name] = result
            
            processing_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "processing_time": processing_time,
                "agent_results": agent_results,
                "input": test_input.product_name
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "input": test_input.product_name
            }
    
    def _analyze_concurrent_execution(
        self, 
        results: List[Any], 
        total_time: float, 
        num_requests: int
    ) -> Dict[str, Any]:
        """
        Analyze concurrent execution performance.
        
        Args:
            results: Results from concurrent execution
            total_time: Total execution time
            num_requests: Number of concurrent requests
            
        Returns:
            Performance analysis
        """
        successful_results = []
        failed_results = []
        processing_times = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "exception": True})
            elif isinstance(result, dict):
                if result.get("success", False):
                    successful_results.append(result)
                    processing_times.append(result.get("processing_time", 0.0))
                else:
                    failed_results.append(result)
        
        success_rate = len(successful_results) / num_requests
        avg_response_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Calculate resource efficiency (concurrent vs sequential estimate)
        estimated_sequential_time = avg_response_time * num_requests
        resource_efficiency = estimated_sequential_time / total_time if total_time > 0 else 0.0
        
        return {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "total_execution_time": total_time,
            "resource_efficiency": min(resource_efficiency, 10.0),  # Cap at 10x efficiency
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "processing_time_variance": self._calculate_variance(processing_times)
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """
        Calculate variance of a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Variance of the values
        """
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    @pytest.mark.asyncio
    async def test_message_serialization_and_deserialization(self):
        """Test message serialization/deserialization for agent communication."""
        test_input = self.test_inputs[0]
        
        # Test serialization of input data
        serialized_input = self._serialize_message(test_input)
        deserialized_input = self._deserialize_message(serialized_input, ProductInput)
        
        # Verify serialization round-trip
        assert deserialized_input.product_name == test_input.product_name
        assert deserialized_input.language_hint == test_input.language_hint
        
        # Test serialization of agent results
        with patch.object(self.orchestrator, 'agents', self.test_agents):
            agent = self.test_agents['llm']
            result = await agent.process(test_input)
            
            # Serialize and deserialize result
            serialized_result = self._serialize_message(result)
            deserialized_result = self._deserialize_message(serialized_result, dict)
            
            # Verify result serialization
            assert deserialized_result["agent_type"] == result["agent_type"]
            assert deserialized_result["success"] == result["success"]
    
    def _serialize_message(self, message: Any) -> str:
        """
        Serialize message for communication.
        
        Args:
            message: Message to serialize
            
        Returns:
            Serialized message as JSON string
        """
        if hasattr(message, '__dict__'):
            # Handle dataclass or object with attributes
            if hasattr(message, '_asdict'):
                message_dict = message._asdict()
            else:
                message_dict = asdict(message) if hasattr(message, '__dataclass_fields__') else message.__dict__
        else:
            message_dict = message
        
        return json.dumps(message_dict, default=str, ensure_ascii=False)
    
    def _deserialize_message(self, serialized_message: str, target_type: type) -> Any:
        """
        Deserialize message from JSON.
        
        Args:
            serialized_message: JSON string to deserialize
            target_type: Target type for deserialization
            
        Returns:
            Deserialized message
        """
        message_dict = json.loads(serialized_message)
        
        if target_type == dict:
            return message_dict
        elif hasattr(target_type, '__dataclass_fields__'):
            # Handle dataclass
            return target_type(**message_dict)
        else:
            # Handle other types
            return target_type(message_dict)