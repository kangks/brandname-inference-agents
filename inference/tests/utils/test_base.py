"""
Base test classes for agent unit testing.

This module provides base classes and common functionality for testing
individual inference agents in isolation with proper setup and teardown.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
from abc import ABC, abstractmethod

from inference.src.agents.base_agent import BaseAgent
from inference.src.models.data_models import ProductInput, LanguageHint


class BaseAgentTest(ABC):
    """
    Base class for all agent unit tests.
    
    Provides common setup, teardown, and utility methods for testing
    individual agents in isolation with mocked dependencies.
    """
    
    def __init__(self):
        """Initialize base test class."""
        self.agent: Optional[BaseAgent] = None
        self.agent_config: Dict[str, Any] = {}
        self.mock_dependencies: Dict[str, Mock] = {}
        self.test_timeout: float = 30.0
    
    def setup_method(self):
        """
        Setup for each test method.
        
        Override this method in subclasses to provide agent-specific setup.
        """
        # Setup default agent configuration
        self.agent_config = self.get_default_config()
        
        # Setup mock dependencies
        self.mock_dependencies = self.setup_mock_dependencies()
        
        # Initialize agent (will be implemented by subclasses)
        self.agent = self.create_agent()
    
    def teardown_method(self):
        """
        Cleanup after each test method.
        
        Ensures proper cleanup of resources and mocks.
        """
        if self.agent:
            # Run cleanup in async context if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, schedule cleanup
                    asyncio.create_task(self.agent.cleanup())
                else:
                    # If not in async context, run cleanup synchronously
                    loop.run_until_complete(self.agent.cleanup())
            except Exception as e:
                # Log cleanup errors but don't fail tests
                print(f"Warning: Agent cleanup failed: {e}")
        
        # Clear references
        self.agent = None
        self.mock_dependencies.clear()
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the agent being tested.
        
        Returns:
            Dictionary containing default agent configuration
        """
        pass
    
    @abstractmethod
    def setup_mock_dependencies(self) -> Dict[str, Mock]:
        """
        Setup mock dependencies for the agent.
        
        Returns:
            Dictionary of mock objects for agent dependencies
        """
        pass
    
    @abstractmethod
    def create_agent(self) -> BaseAgent:
        """
        Create the agent instance for testing.
        
        Returns:
            Agent instance configured for testing
        """
        pass
    
    async def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent is not None
        
        # Test initialization
        await self.agent.initialize()
        assert self.agent._is_initialized is True
        
        # Test that agent has required attributes
        assert hasattr(self.agent, 'agent_name')
        assert hasattr(self.agent, 'config')
        assert hasattr(self.agent, 'logger')
    
    async def test_agent_process_valid_input(self):
        """Test agent processing with valid input."""
        if not self.agent._is_initialized:
            await self.agent.initialize()
        
        # Create valid test input
        test_input = ProductInput(
            product_name="Test Product Name",
            language_hint=LanguageHint.ENGLISH
        )
        
        # Process input
        result = await self.agent.process(test_input)
        
        # Validate result structure
        assert isinstance(result, dict)
        assert "agent_type" in result
        assert "result" in result
        assert "success" in result
        assert "error" in result
        
        # If successful, validate result content
        if result["success"]:
            assert result["result"] is not None
            assert result["error"] is None
        else:
            assert result["error"] is not None
    
    async def test_agent_process_invalid_input(self):
        """Test agent error handling with invalid input."""
        if not self.agent._is_initialized:
            await self.agent.initialize()
        
        # Test with empty product name
        invalid_input = ProductInput(
            product_name="",
            language_hint=LanguageHint.ENGLISH
        )
        
        # Should handle gracefully
        result = await self.agent.process(invalid_input)
        
        # Should return error result
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert result["error"] is not None
    
    async def test_agent_timeout_handling(self):
        """Test agent timeout scenarios."""
        if not self.agent._is_initialized:
            await self.agent.initialize()
        
        # Create test input
        test_input = ProductInput(
            product_name="Test Product",
            language_hint=LanguageHint.ENGLISH
        )
        
        # Mock a timeout scenario by patching the process method
        original_process = self.agent.process
        
        async def timeout_process(input_data):
            await asyncio.sleep(self.test_timeout + 1)  # Exceed timeout
            return await original_process(input_data)
        
        with patch.object(self.agent, 'process', side_effect=timeout_process):
            # Test with timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self.agent.process(test_input), 
                    timeout=1.0  # Short timeout for testing
                )
    
    async def test_agent_health_check(self):
        """Test agent health check functionality."""
        if not self.agent._is_initialized:
            await self.agent.initialize()
        
        # Perform health check
        health = await self.agent.health_check()
        
        # Validate health check result
        assert hasattr(health, 'agent_name')
        assert hasattr(health, 'is_healthy')
        assert hasattr(health, 'last_check')
        assert health.agent_name == self.agent.agent_name
        
        # For initialized agent, should be healthy
        assert health.is_healthy is True
    
    async def test_agent_cleanup(self):
        """Test agent cleanup functionality."""
        if not self.agent._is_initialized:
            await self.agent.initialize()
        
        # Ensure agent is initialized
        assert self.agent._is_initialized is True
        
        # Perform cleanup
        await self.agent.cleanup()
        
        # Verify cleanup (implementation may vary by agent)
        # At minimum, should not raise exceptions
        assert True  # If we get here, cleanup succeeded
    
    # Utility methods for subclasses
    
    def create_test_input(self, product_name: str = "Test Product", 
                         language_hint: LanguageHint = LanguageHint.AUTO) -> ProductInput:
        """
        Create a test ProductInput instance.
        
        Args:
            product_name: Product name for testing
            language_hint: Language hint for testing
            
        Returns:
            ProductInput instance for testing
        """
        return ProductInput(
            product_name=product_name,
            language_hint=language_hint
        )
    
    def assert_valid_result_structure(self, result: Dict[str, Any]):
        """
        Assert that a result has the expected structure.
        
        Args:
            result: Result dictionary to validate
        """
        assert isinstance(result, dict)
        assert "agent_type" in result
        assert "result" in result
        assert "success" in result
        assert "error" in result
        
        if result["success"]:
            assert result["result"] is not None
            assert result["error"] is None
        else:
            assert result["error"] is not None
    
    def assert_processing_time_reasonable(self, result: Dict[str, Any], max_time: float = 10.0):
        """
        Assert that processing time is reasonable.
        
        Args:
            result: Result dictionary containing processing time
            max_time: Maximum acceptable processing time in seconds
        """
        if result.get("success") and result.get("result"):
            processing_time = getattr(result["result"], "processing_time", None)
            if processing_time is not None:
                assert 0 <= processing_time <= max_time, f"Processing time {processing_time}s exceeds maximum {max_time}s"


class AsyncTestMixin:
    """
    Mixin class providing utilities for async testing.
    
    Provides common async testing patterns and utilities.
    """
    
    async def run_with_timeout(self, coro, timeout: float = 30.0):
        """
        Run a coroutine with timeout.
        
        Args:
            coro: Coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            Result of the coroutine
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        return await asyncio.wait_for(coro, timeout=timeout)
    
    async def assert_completes_within(self, coro, max_time: float):
        """
        Assert that a coroutine completes within the specified time.
        
        Args:
            coro: Coroutine to test
            max_time: Maximum time in seconds
        """
        start_time = time.time()
        await coro
        elapsed = time.time() - start_time
        assert elapsed <= max_time, f"Operation took {elapsed}s, expected <= {max_time}s"
    
    async def assert_raises_async(self, exception_type, coro):
        """
        Assert that an async operation raises a specific exception.
        
        Args:
            exception_type: Expected exception type
            coro: Coroutine that should raise the exception
        """
        with pytest.raises(exception_type):
            await coro


class MockAgentTest(BaseAgentTest):
    """
    Test class for mock agents used in testing.
    
    Provides a concrete implementation for testing mock agents
    or as a template for other agent tests.
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for mock agent."""
        return {
            "confidence_threshold": 0.5,
            "timeout_seconds": 30,
            "max_text_length": 1000
        }
    
    def setup_mock_dependencies(self) -> Dict[str, Mock]:
        """Setup mock dependencies for mock agent."""
        return {
            "external_service": Mock(),
            "database": Mock(),
            "model": Mock()
        }
    
    def create_agent(self) -> BaseAgent:
        """Create mock agent for testing."""
        # This would create a mock agent - implementation depends on specific needs
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.agent_name = "mock_test_agent"
        mock_agent.config = self.agent_config
        mock_agent._is_initialized = False
        
        # Setup async methods
        mock_agent.initialize = AsyncMock()
        mock_agent.process = AsyncMock()
        mock_agent.cleanup = AsyncMock()
        mock_agent.health_check = AsyncMock()
        
        return mock_agent


class PerformanceTestMixin:
    """
    Mixin class for performance testing utilities.
    
    Provides methods for measuring and asserting performance characteristics.
    """
    
    async def measure_processing_time(self, agent: BaseAgent, input_data: ProductInput) -> float:
        """
        Measure processing time for an agent.
        
        Args:
            agent: Agent to test
            input_data: Input data for processing
            
        Returns:
            Processing time in seconds
        """
        start_time = time.time()
        await agent.process(input_data)
        return time.time() - start_time
    
    async def assert_performance_within_bounds(self, agent: BaseAgent, input_data: ProductInput, 
                                             max_time: float, min_time: float = 0.0):
        """
        Assert that agent performance is within specified bounds.
        
        Args:
            agent: Agent to test
            input_data: Input data for processing
            max_time: Maximum acceptable processing time
            min_time: Minimum expected processing time
        """
        processing_time = await self.measure_processing_time(agent, input_data)
        assert min_time <= processing_time <= max_time, \
            f"Processing time {processing_time}s not within bounds [{min_time}, {max_time}]"
    
    async def benchmark_agent(self, agent: BaseAgent, input_data: ProductInput, 
                            iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark agent performance over multiple iterations.
        
        Args:
            agent: Agent to benchmark
            input_data: Input data for processing
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance statistics
        """
        times = []
        
        for _ in range(iterations):
            processing_time = await self.measure_processing_time(agent, input_data)
            times.append(processing_time)
        
        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "iterations": iterations
        }