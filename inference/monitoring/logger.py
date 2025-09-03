"""
Comprehensive logging infrastructure for agent operations and decisions.

This module provides structured logging with context tracking, performance metrics,
and decision audit trails following PEP 8 standards.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from threading import local
import traceback

from ..models.data_models import (
    ProductInput,
    InferenceResult,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult
)


@dataclass
class LogContext:
    """Context information for structured logging."""
    
    request_id: str
    agent_name: str
    operation: str
    start_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    
    operation: str
    duration_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class InferenceLogger:
    """
    Structured logger for inference operations with context tracking.
    
    Provides comprehensive logging for agent operations, decisions,
    performance metrics, and error tracking.
    """
    
    def __init__(self, logger_name: str = "inference") -> None:
        """
        Initialize inference logger.
        
        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self._local = local()
        self._context_stack: List[LogContext] = []
        
    def _get_current_context(self) -> Optional[LogContext]:
        """Get current logging context from thread-local storage."""
        return getattr(self._local, 'context', None)
    
    def _set_current_context(self, context: Optional[LogContext]) -> None:
        """Set current logging context in thread-local storage."""
        self._local.context = context
    
    @contextmanager
    def operation_context(
        self,
        request_id: str,
        agent_name: str,
        operation: str,
        **metadata
    ):
        """
        Context manager for tracking operation lifecycle.
        
        Args:
            request_id: Unique identifier for the request
            agent_name: Name of the agent performing the operation
            operation: Name of the operation being performed
            **metadata: Additional metadata for the operation
        """
        context = LogContext(
            request_id=request_id,
            agent_name=agent_name,
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        
        # Store previous context
        previous_context = self._get_current_context()
        self._set_current_context(context)
        self._context_stack.append(context)
        
        try:
            self.log_operation_start(context)
            yield context
            
        except Exception as e:
            self.log_operation_error(context, e)
            raise
            
        finally:
            duration = time.time() - context.start_time
            self.log_operation_end(context, duration)
            
            # Restore previous context
            self._context_stack.pop()
            self._set_current_context(previous_context)
    
    def log_operation_start(self, context: LogContext) -> None:
        """Log the start of an operation."""
        self._log_structured(
            level=logging.INFO,
            message=f"Starting {context.operation}",
            event_type="operation_start",
            context=context.to_dict()
        )
    
    def log_operation_end(self, context: LogContext, duration: float) -> None:
        """Log the end of an operation with performance metrics."""
        metrics = PerformanceMetrics(
            operation=context.operation,
            duration_ms=duration * 1000
        )
        
        self._log_structured(
            level=logging.INFO,
            message=f"Completed {context.operation} in {duration:.3f}s",
            event_type="operation_end",
            context=context.to_dict(),
            performance=metrics.to_dict()
        )
    
    def log_operation_error(self, context: LogContext, error: Exception) -> None:
        """Log an operation error with full context."""
        self._log_structured(
            level=logging.ERROR,
            message=f"Error in {context.operation}: {str(error)}",
            event_type="operation_error",
            context=context.to_dict(),
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        )
    
    def log_agent_decision(
        self,
        agent_name: str,
        decision: str,
        confidence: float,
        reasoning: str,
        input_data: Dict[str, Any],
        **metadata
    ) -> None:
        """
        Log agent decision with reasoning for audit trail.
        
        Args:
            agent_name: Name of the agent making the decision
            decision: The decision made by the agent
            confidence: Confidence score for the decision
            reasoning: Explanation of the decision reasoning
            input_data: Input data that led to the decision
            **metadata: Additional metadata
        """
        context = self._get_current_context()
        
        self._log_structured(
            level=logging.INFO,
            message=f"Agent decision: {decision}",
            event_type="agent_decision",
            context=context.to_dict() if context else None,
            decision={
                "agent_name": agent_name,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "input_data": input_data,
                "metadata": metadata
            }
        )
    
    def log_inference_result(
        self,
        result: InferenceResult,
        request_id: str
    ) -> None:
        """
        Log complete inference result for analysis.
        
        Args:
            result: Complete inference result
            request_id: Request identifier
        """
        self._log_structured(
            level=logging.INFO,
            message=f"Inference completed: {result.best_method} -> {result.best_prediction}",
            event_type="inference_result",
            request_id=request_id,
            result={
                "input_product": result.input_product,
                "best_prediction": result.best_prediction,
                "best_confidence": result.best_confidence,
                "best_method": result.best_method,
                "total_processing_time": result.total_processing_time,
                "methods_used": {
                    "ner": result.ner_result is not None,
                    "rag": result.rag_result is not None,
                    "llm": result.llm_result is not None,
                    "hybrid": result.hybrid_result is not None
                }
            }
        )
    
    def log_model_performance(
        self,
        model_name: str,
        operation: str,
        latency_ms: float,
        throughput_ops_per_sec: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        **metadata
    ) -> None:
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            operation: Operation performed
            latency_ms: Latency in milliseconds
            throughput_ops_per_sec: Throughput in operations per second
            memory_usage_mb: Memory usage in MB
            **metadata: Additional performance metadata
        """
        context = self._get_current_context()
        
        self._log_structured(
            level=logging.INFO,
            message=f"Model performance: {model_name} {operation}",
            event_type="model_performance",
            context=context.to_dict() if context else None,
            performance={
                "model_name": model_name,
                "operation": operation,
                "latency_ms": latency_ms,
                "throughput_ops_per_sec": throughput_ops_per_sec,
                "memory_usage_mb": memory_usage_mb,
                "metadata": metadata
            }
        )
    
    def log_circuit_breaker_event(
        self,
        agent_name: str,
        event_type: str,
        failure_count: int,
        threshold: int,
        **metadata
    ) -> None:
        """
        Log circuit breaker state changes.
        
        Args:
            agent_name: Name of the agent
            event_type: Type of circuit breaker event (open, close, half_open)
            failure_count: Current failure count
            threshold: Failure threshold
            **metadata: Additional metadata
        """
        self._log_structured(
            level=logging.WARNING if event_type == "open" else logging.INFO,
            message=f"Circuit breaker {event_type} for {agent_name}",
            event_type="circuit_breaker",
            circuit_breaker={
                "agent_name": agent_name,
                "event_type": event_type,
                "failure_count": failure_count,
                "threshold": threshold,
                "metadata": metadata
            }
        )
    
    def log_health_check(
        self,
        agent_name: str,
        is_healthy: bool,
        response_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Log health check results.
        
        Args:
            agent_name: Name of the agent
            is_healthy: Whether the agent is healthy
            response_time_ms: Health check response time
            error_message: Error message if unhealthy
            **metadata: Additional health metadata
        """
        level = logging.INFO if is_healthy else logging.WARNING
        
        self._log_structured(
            level=level,
            message=f"Health check {agent_name}: {'healthy' if is_healthy else 'unhealthy'}",
            event_type="health_check",
            health={
                "agent_name": agent_name,
                "is_healthy": is_healthy,
                "response_time_ms": response_time_ms,
                "error_message": error_message,
                "metadata": metadata
            }
        )
    
    def log_resource_usage(
        self,
        component: str,
        cpu_percent: float,
        memory_mb: float,
        disk_usage_percent: Optional[float] = None,
        network_io_mb: Optional[float] = None,
        **metadata
    ) -> None:
        """
        Log system resource usage.
        
        Args:
            component: Component name
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            disk_usage_percent: Disk usage percentage
            network_io_mb: Network I/O in MB
            **metadata: Additional resource metadata
        """
        self._log_structured(
            level=logging.DEBUG,
            message=f"Resource usage {component}",
            event_type="resource_usage",
            resources={
                "component": component,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "disk_usage_percent": disk_usage_percent,
                "network_io_mb": network_io_mb,
                "metadata": metadata
            }
        )
    
    def _log_structured(
        self,
        level: int,
        message: str,
        event_type: str,
        **kwargs
    ) -> None:
        """
        Log structured data with consistent format.
        
        Args:
            level: Logging level
            message: Log message
            event_type: Type of event being logged
            **kwargs: Additional structured data
        """
        log_data = {
            "timestamp": time.time(),
            "message": message,
            "event_type": event_type,
            **kwargs
        }
        
        # Add current context if available
        context = self._get_current_context()
        if context and "context" not in kwargs:
            log_data["context"] = context.to_dict()
        
        # Log as JSON for structured parsing
        self.logger.log(level, json.dumps(log_data, default=str))


def setup_structured_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    enable_cloudwatch: bool = False
) -> InferenceLogger:
    """
    Setup structured logging for the inference system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json, text)
        enable_cloudwatch: Whether to enable CloudWatch logging
        
    Returns:
        Configured InferenceLogger instance
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    if log_format == "json":
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
    else:
        # Human-readable formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup CloudWatch handler if enabled
    if enable_cloudwatch:
        try:
            import boto3
            from watchtower import CloudWatchLogsHandler
            
            cloudwatch_handler = CloudWatchLogsHandler(
                log_group="multilingual-inference",
                stream_name="inference-logs",
                boto3_client=boto3.client('logs')
            )
            cloudwatch_handler.setFormatter(formatter)
            root_logger.addHandler(cloudwatch_handler)
            
        except ImportError:
            logging.warning("CloudWatch logging requested but watchtower not available")
        except Exception as e:
            logging.warning(f"Failed to setup CloudWatch logging: {str(e)}")
    
    # Create and return inference logger
    return InferenceLogger()


# Global logger instance
_global_logger: Optional[InferenceLogger] = None


def get_inference_logger() -> InferenceLogger:
    """Get the global inference logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_structured_logging()
    return _global_logger