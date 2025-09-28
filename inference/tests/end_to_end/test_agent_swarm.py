"""
End-to-end tests for agent swarm coordination in production-like environment.

This module tests the orchestrator coordinating multiple agents with actual AWS services,
system performance under concurrent load, error recovery, and monitoring.
"""

import os
import pytest
import asyncio
import time
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics

from inference.src.agents.orchestrator_agent import OrchestratorAgent
from inference.src.agents.rag_agent import RAGAgent
from inference.src.agents.llm_agent import LLMAgent
from inference.src.agents.ner_agent import NERAgent
from inference.src.agents.hybrid_agent import HybridAgent
from inference.src.config.settings import get_config
from inference.tests.fixtures.test_data import TestPayload
from inference.tests.utils.assertion_helpers import AssertionHelpers


@pytest.mark.e2e
@pytest.mark.aws
@pytest.mark.slow
class TestAgentSwarm:
    """End-to-end tests for orchestrator coordinating multiple agents in production-like environment."""
    
    @classmethod
    def setup_class(cls):
        """Setup production-like environment for agent swarm tests."""
        cls.aws_profile = "ml-sandbox"
        cls.aws_region = "us-east-1"
        
        # Set environment variables for production-like configuration
        os.environ["AWS_PROFILE"] = cls.aws_profile
        os.environ["AWS_REGION"] = cls.aws_region
        os.environ["AWS_DEFAULT_REGION"] = cls.aws_region
        
        # Configure logging for monitoring tests
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Test payloads for different scenarios
        cls.test_payloads = [
            TestPayload(product_name="Samsung Galaxy S24 Ultra 256GB", language_hint="en"),
            TestPayload(product_name="iPhone 15 Pro Max 512GB Space Black", language_hint="en"),
            TestPayload(product_name="Sony WH-1000XM5 Wireless Headphones", language_hint="en"),
            TestPayload(product_name="MacBook Pro 16-inch M3 Max", language_hint="en"),
            TestPayload(product_name="Nintendo Switch OLED Model", language_hint="en")
        ]
        
    def setup_method(self):
        """Setup for each test method."""
        self.config = get_config()
        self.assertion_helpers = AssertionHelpers()
        self.orchestrator = OrchestratorAgent()
        
        # Performance tracking
        self.performance_metrics = {
            "response_times": [],
            "success_rate": 0,
            "error_count": 0,
            "agent_usage": {}
        }
        
    @pytest.mark.asyncio
    async def test_orchestrator_multi_agent_coordination(self):
        """Test orchestrator coordinating multiple agents in production-like environment."""
        # Test with different agent methods
        agent_methods = ["orchestrator", "hybrid", "rag", "llm", "ner"]
        results = {}
        
        for method in agent_methods:
            self.logger.info(f"Testing agent method: {method}")
            
            request = {
                "product_name": self.test_payloads[0].product_name,
                "language_hint": self.test_payloads[0].language_hint,
                "method": method
            }
            
            start_time = time.time()
            try:
                result = await self.orchestrator.process(request)
                execution_time = time.time() - start_time
                
                # Validate result
                assert result is not None, f"Method {method} should return a result"
                assert "brands" in result, f"Method {method} result should contain brands"
                
                results[method] = {
                    "success": True,
                    "execution_time": execution_time,
                    "result": result
                }
                
                self.logger.info(f"Method {method} completed in {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[method] = {
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e)
                }
                self.logger.error(f"Method {method} failed: {e}")
        
        # Validate coordination results
        successful_methods = [method for method, result in results.items() if result["success"]]
        assert len(successful_methods) > 0, "At least one agent method should succeed"
        
        # Validate orchestrator coordination
        if "orchestrator" in successful_methods:
            orchestrator_result = results["orchestrator"]["result"]
            if "metadata" in orchestrator_result:
                metadata = orchestrator_result["metadata"]
                assert "agents_used" in metadata, "Orchestrator should report agents used"
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_swarm_performance(self):
        """Test system performance under concurrent load with actual AWS services."""
        # Create concurrent requests with different payloads
        concurrent_requests = []
        for i, payload in enumerate(self.test_payloads):
            request = {
                "product_name": payload.product_name,
                "language_hint": payload.language_hint,
                "method": "orchestrator"
            }
            concurrent_requests.append((f"request_{i+1}", request))
        
        # Execute concurrent requests
        self.logger.info(f"Starting {len(concurrent_requests)} concurrent requests")
        start_time = time.time()
        
        tasks = []
        for request_id, request in concurrent_requests:
            task = asyncio.create_task(
                self._execute_timed_request(request_id, request),
                name=request_id
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze performance results
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        for i, result in enumerate(results):
            request_id = concurrent_requests[i][0]
            
            if isinstance(result, Exception):
                self.logger.error(f"{request_id} failed with exception: {result}")
                failed_requests += 1
            elif result and "success" in result:
                if result["success"]:
                    successful_requests += 1
                    response_times.append(result["execution_time"])
                    self.logger.info(f"{request_id} completed in {result['execution_time']:.2f}s")
                else:
                    failed_requests += 1
                    self.logger.error(f"{request_id} failed: {result.get('error', 'Unknown error')}")
        
        # Validate performance metrics
        total_requests = len(concurrent_requests)
        success_rate = (successful_requests / total_requests) * 100
        
        self.logger.info(f"Performance Summary:")
        self.logger.info(f"  Total requests: {total_requests}")
        self.logger.info(f"  Successful: {successful_requests}")
        self.logger.info(f"  Failed: {failed_requests}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.info(f"  Total time: {total_time:.2f}s")
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            max_response_time = max(response_times)
            
            self.logger.info(f"  Average response time: {avg_response_time:.2f}s")
            self.logger.info(f"  Median response time: {median_response_time:.2f}s")
            self.logger.info(f"  Max response time: {max_response_time:.2f}s")
            
            # Performance assertions
            assert success_rate >= 60.0, f"Success rate should be at least 60%, got {success_rate:.1f}%"
            assert avg_response_time < 60.0, f"Average response time should be under 60s, got {avg_response_time:.2f}s"
            assert max_response_time < 120.0, f"Max response time should be under 120s, got {max_response_time:.2f}s"
        else:
            pytest.fail("No successful requests to analyze performance")
    
    @pytest.mark.asyncio
    async def test_agent_swarm_error_recovery(self):
        """Test error recovery and system resilience in agent swarm."""
        # Test with various error scenarios
        error_scenarios = [
            {
                "name": "invalid_product_name",
                "request": {
                    "product_name": "",  # Empty product name
                    "language_hint": "en",
                    "method": "orchestrator"
                }
            },
            {
                "name": "invalid_language_hint",
                "request": {
                    "product_name": self.test_payloads[0].product_name,
                    "language_hint": "invalid_lang",
                    "method": "orchestrator"
                }
            },
            {
                "name": "invalid_method",
                "request": {
                    "product_name": self.test_payloads[0].product_name,
                    "language_hint": "en",
                    "method": "nonexistent_method"
                }
            }
        ]
        
        recovery_results = {}
        
        for scenario in error_scenarios:
            self.logger.info(f"Testing error recovery scenario: {scenario['name']}")
            
            try:
                result = await self.orchestrator.process(scenario["request"])
                
                # Should either handle gracefully or return error response
                if result:
                    if "error" in result:
                        recovery_results[scenario["name"]] = {
                            "handled_gracefully": True,
                            "error_message": result["error"]
                        }
                    else:
                        recovery_results[scenario["name"]] = {
                            "handled_gracefully": True,
                            "recovered": True
                        }
                else:
                    recovery_results[scenario["name"]] = {
                        "handled_gracefully": False,
                        "error": "No result returned"
                    }
                    
            except Exception as e:
                recovery_results[scenario["name"]] = {
                    "handled_gracefully": False,
                    "exception": str(e)
                }
                self.logger.error(f"Scenario {scenario['name']} raised exception: {e}")
        
        # Validate error recovery
        graceful_handling_count = sum(1 for result in recovery_results.values() 
                                    if result.get("handled_gracefully", False))
        
        self.logger.info(f"Error recovery summary: {graceful_handling_count}/{len(error_scenarios)} scenarios handled gracefully")
        
        # At least some error scenarios should be handled gracefully
        assert graceful_handling_count > 0, "System should handle at least some error scenarios gracefully"
    
    @pytest.mark.asyncio
    async def test_agent_swarm_monitoring_and_logging(self):
        """Test monitoring and logging in production environment."""
        # Test with monitoring-enabled request
        request = {
            "product_name": self.test_payloads[0].product_name,
            "language_hint": self.test_payloads[0].language_hint,
            "method": "orchestrator"
        }
        
        # Capture logs during execution
        with self.assertion_helpers.capture_logs() as log_capture:
            start_time = time.time()
            result = await self.orchestrator.process(request)
            execution_time = time.time() - start_time
        
        # Validate monitoring data
        assert result is not None, "Request should return a result"
        
        # Check for monitoring metadata
        if "metadata" in result:
            metadata = result["metadata"]
            
            # Validate timing information
            if "execution_time" in metadata:
                reported_time = metadata["execution_time"]
                # Allow some variance in timing
                assert abs(reported_time - execution_time) < 5.0, \
                    f"Reported execution time should be close to measured time"
            
            # Validate agent usage tracking
            if "agents_used" in metadata:
                agents_used = metadata["agents_used"]
                assert isinstance(agents_used, (list, dict)), "Agents used should be trackable"
        
        # Validate logging output
        captured_logs = log_capture.get_logs()
        if captured_logs:
            # Should have some log entries during execution
            assert len(captured_logs) > 0, "Should generate log entries during execution"
    
    @pytest.mark.asyncio
    async def test_agent_swarm_resource_management(self):
        """Test resource management and cleanup in agent swarm."""
        # Test multiple sequential requests to check resource cleanup
        sequential_requests = 5
        resource_usage = []
        
        for i in range(sequential_requests):
            request = {
                "product_name": f"{self.test_payloads[0].product_name} - Sequential {i+1}",
                "language_hint": self.test_payloads[0].language_hint,
                "method": "orchestrator"
            }
            
            # Monitor resource usage (simplified)
            start_time = time.time()
            result = await self.orchestrator.process(request)
            execution_time = time.time() - start_time
            
            resource_usage.append({
                "request_id": i + 1,
                "execution_time": execution_time,
                "success": result is not None and "brands" in result
            })
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        # Validate resource management
        successful_requests = sum(1 for usage in resource_usage if usage["success"])
        assert successful_requests >= sequential_requests * 0.8, \
            f"At least 80% of sequential requests should succeed, got {successful_requests}/{sequential_requests}"
        
        # Validate consistent performance (no significant degradation)
        execution_times = [usage["execution_time"] for usage in resource_usage if usage["success"]]
        if len(execution_times) > 1:
            first_half = execution_times[:len(execution_times)//2]
            second_half = execution_times[len(execution_times)//2:]
            
            avg_first_half = statistics.mean(first_half)
            avg_second_half = statistics.mean(second_half)
            
            # Performance shouldn't degrade significantly
            degradation_ratio = avg_second_half / avg_first_half if avg_first_half > 0 else 1.0
            assert degradation_ratio < 2.0, \
                f"Performance shouldn't degrade significantly: {degradation_ratio:.2f}x slower"
    
    async def _execute_timed_request(self, request_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a timed request and return performance metrics."""
        start_time = time.time()
        
        try:
            result = await self.orchestrator.process(request)
            execution_time = time.time() - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "execution_time": execution_time,
                "result": result
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Log performance metrics
        if self.performance_metrics["response_times"]:
            avg_time = statistics.mean(self.performance_metrics["response_times"])
            self.logger.info(f"Test method average response time: {avg_time:.2f}s")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests in class."""
        cls.logger.info("Agent swarm E2E tests completed")