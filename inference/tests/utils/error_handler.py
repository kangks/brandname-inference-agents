"""
Test error handling utilities.

This module provides consistent error handling and reporting for tests,
including error simulation, logging, and recovery mechanisms.
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Callable, Type
from unittest.mock import Mock
from contextlib import asynccontextmanager
import pytest


class TestErrorHandler:
    """
    Centralized error handling for test suites.
    
    Provides consistent error handling, logging, and recovery
    mechanisms for all test scenarios.
    """
    
    def __init__(self, logger_name: str = "test_error_handler"):
        """
        Initialize test error handler.
        
        Args:
            logger_name: Name for the error handler logger
        """
        self.logger = logging.getLogger(logger_name)
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def handle_test_failure(self, test_name: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle test failures with detailed context and logging.
        
        Args:
            test_name: Name of the failed test
            error: Exception that caused the failure
            context: Additional context information
            
        Returns:
            Dictionary containing error details and context
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error record
        error_record = {
            "test_name": test_name,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": pytest.current_time() if hasattr(pytest, 'current_time') else None
        }
        
        # Add to history
        self.error_history.append(error_record)
        
        # Log the error
        self.logger.error(
            f"Test failure in {test_name}: {error_type}: {error_message}",
            extra={"context": context}
        )
        
        return error_record
    
    def handle_async_error(self, test_name: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle async test errors with special consideration for async patterns.
        
        Args:
            test_name: Name of the failed test
            error: Exception that caused the failure
            context: Additional context information
            
        Returns:
            Dictionary containing error details and context
        """
        # Add async-specific context
        async_context = context or {}
        async_context.update({
            "is_async": True,
            "event_loop_running": False,
            "pending_tasks": 0
        })
        
        try:
            loop = asyncio.get_event_loop()
            async_context["event_loop_running"] = loop.is_running()
            
            # Count pending tasks
            all_tasks = asyncio.all_tasks(loop)
            async_context["pending_tasks"] = len(all_tasks)
            
        except RuntimeError:
            # No event loop running
            pass
        
        return self.handle_test_failure(test_name, error, async_context)
    
    def handle_timeout_error(self, test_name: str, timeout_duration: float,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle timeout errors with specific timeout context.
        
        Args:
            test_name: Name of the test that timed out
            timeout_duration: Duration of the timeout
            context: Additional context information
            
        Returns:
            Dictionary containing timeout error details
        """
        timeout_context = context or {}
        timeout_context.update({
            "timeout_duration": timeout_duration,
            "error_category": "timeout"
        })
        
        timeout_error = asyncio.TimeoutError(f"Test timed out after {timeout_duration}s")
        return self.handle_test_failure(test_name, timeout_error, timeout_context)
    
    def handle_assertion_error(self, test_name: str, assertion_error: AssertionError,
                             expected: Any = None, actual: Any = None,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle assertion errors with expected vs actual value context.
        
        Args:
            test_name: Name of the test with assertion failure
            assertion_error: The assertion error
            expected: Expected value
            actual: Actual value
            context: Additional context information
            
        Returns:
            Dictionary containing assertion error details
        """
        assertion_context = context or {}
        assertion_context.update({
            "expected": expected,
            "actual": actual,
            "error_category": "assertion"
        })
        
        return self.handle_test_failure(test_name, assertion_error, assertion_context)
    
    def cleanup_test_resources(self, resources: List[Any]) -> List[str]:
        """
        Clean up test resources with error handling.
        
        Args:
            resources: List of resources to clean up
            
        Returns:
            List of cleanup errors (empty if all successful)
        """
        cleanup_errors = []
        
        for i, resource in enumerate(resources):
            try:
                if hasattr(resource, 'cleanup'):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        # Handle async cleanup
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Schedule cleanup task
                                asyncio.create_task(resource.cleanup())
                            else:
                                # Run cleanup synchronously
                                loop.run_until_complete(resource.cleanup())
                        except Exception as e:
                            cleanup_errors.append(f"Async cleanup failed for resource {i}: {e}")
                    else:
                        # Handle sync cleanup
                        resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, '__exit__'):
                    resource.__exit__(None, None, None)
                    
            except Exception as e:
                cleanup_errors.append(f"Cleanup failed for resource {i}: {e}")
                self.logger.warning(f"Resource cleanup failed: {e}")
        
        return cleanup_errors
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered during testing.
        
        Returns:
            Dictionary containing error statistics and summary
        """
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts.copy(),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            "recent_errors": self.error_history[-5:] if self.error_history else []
        }
    
    def reset_error_tracking(self):
        """Reset error tracking for a new test session."""
        self.error_counts.clear()
        self.error_history.clear()
    
    @asynccontextmanager
    async def async_error_context(self, test_name: str, context: Optional[Dict[str, Any]] = None):
        """
        Async context manager for handling errors in async tests.
        
        Args:
            test_name: Name of the test
            context: Additional context information
        """
        try:
            yield
        except Exception as e:
            self.handle_async_error(test_name, e, context)
            raise
    
    def create_error_simulator(self, error_type: Type[Exception], 
                             error_message: str = "Simulated error") -> Callable:
        """
        Create a function that simulates specific errors for testing.
        
        Args:
            error_type: Type of exception to simulate
            error_message: Message for the simulated error
            
        Returns:
            Function that raises the specified error
        """
        def simulate_error(*args, **kwargs):
            raise error_type(error_message)
        
        return simulate_error
    
    def create_async_error_simulator(self, error_type: Type[Exception],
                                   error_message: str = "Simulated async error") -> Callable:
        """
        Create an async function that simulates specific errors for testing.
        
        Args:
            error_type: Type of exception to simulate
            error_message: Message for the simulated error
            
        Returns:
            Async function that raises the specified error
        """
        async def simulate_async_error(*args, **kwargs):
            raise error_type(error_message)
        
        return simulate_async_error
    
    def create_timeout_simulator(self, timeout_duration: float) -> Callable:
        """
        Create an async function that simulates timeout scenarios.
        
        Args:
            timeout_duration: Duration to sleep before completing
            
        Returns:
            Async function that sleeps for the specified duration
        """
        async def simulate_timeout(*args, **kwargs):
            await asyncio.sleep(timeout_duration)
            return "Completed after timeout simulation"
        
        return simulate_timeout


class ErrorAssertions:
    """
    Custom assertion methods for error testing.
    
    Provides specialized assertions for testing error conditions
    and error handling behavior.
    """
    
    @staticmethod
    def assert_error_type(error: Exception, expected_type: Type[Exception]):
        """
        Assert that an error is of the expected type.
        
        Args:
            error: The error to check
            expected_type: Expected error type
        """
        assert isinstance(error, expected_type), \
            f"Expected {expected_type.__name__}, got {type(error).__name__}"
    
    @staticmethod
    def assert_error_message_contains(error: Exception, expected_substring: str):
        """
        Assert that an error message contains a specific substring.
        
        Args:
            error: The error to check
            expected_substring: Substring that should be in the error message
        """
        error_message = str(error)
        assert expected_substring in error_message, \
            f"Expected '{expected_substring}' in error message: '{error_message}'"
    
    @staticmethod
    async def assert_async_raises(expected_exception: Type[Exception], coro):
        """
        Assert that an async operation raises a specific exception.
        
        Args:
            expected_exception: Expected exception type
            coro: Coroutine that should raise the exception
        """
        with pytest.raises(expected_exception):
            await coro
    
    @staticmethod
    async def assert_timeout_occurs(coro, timeout: float):
        """
        Assert that an async operation times out.
        
        Args:
            coro: Coroutine that should timeout
            timeout: Timeout duration
        """
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coro, timeout=timeout)
    
    @staticmethod
    def assert_no_errors_in_result(result: Dict[str, Any]):
        """
        Assert that a result dictionary indicates no errors.
        
        Args:
            result: Result dictionary to check
        """
        assert result.get("success", False) is True, \
            f"Expected successful result, got error: {result.get('error')}"
        assert result.get("error") is None, \
            f"Expected no error, got: {result.get('error')}"
    
    @staticmethod
    def assert_error_in_result(result: Dict[str, Any], expected_error_type: str = None):
        """
        Assert that a result dictionary indicates an error.
        
        Args:
            result: Result dictionary to check
            expected_error_type: Optional expected error type
        """
        assert result.get("success", True) is False, \
            "Expected error result, but got success"
        assert result.get("error") is not None, \
            "Expected error message, but got None"
        
        if expected_error_type:
            error_message = result.get("error", "")
            assert expected_error_type.lower() in error_message.lower(), \
                f"Expected error type '{expected_error_type}' in error message: '{error_message}'"


# Global error handler instance for test session
_global_error_handler: Optional[TestErrorHandler] = None


def get_test_error_handler() -> TestErrorHandler:
    """
    Get the global test error handler instance.
    
    Returns:
        Global TestErrorHandler instance
    """
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = TestErrorHandler()
    
    return _global_error_handler


def reset_test_error_handler():
    """Reset the global test error handler."""
    global _global_error_handler
    
    if _global_error_handler:
        _global_error_handler.reset_error_tracking()
    
    _global_error_handler = None