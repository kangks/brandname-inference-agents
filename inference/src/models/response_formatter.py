"""
Response Formatter for consistent API responses.

This module provides standardized response formatting for the inference API,
ensuring consistent terminology and structure across all endpoints.
"""

import time
from typing import Dict, List, Optional, Any, Union
import logging


logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats API responses with consistent terminology and structure.
    
    Ensures all API responses use "method" terminology consistently and
    follow a standardized format for errors, successes, and health checks.
    """
    
    @staticmethod
    def format_error_response(
        status_code: int,
        error_message: str,
        method: Optional[str] = None,
        available_methods: Optional[List[str]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error responses with consistent terminology.
        
        Args:
            status_code: HTTP status code
            error_message: Error message to include
            method: Method that was requested (if applicable)
            available_methods: List of available methods (if applicable)
            request_id: Request identifier (if available)
            
        Returns:
            Standardized error response dictionary
        """
        response = {
            "status": "error",
            "error": error_message,
            "status_code": status_code,
            "timestamp": time.time()
        }
        
        if method is not None:
            response["method"] = method
        
        if available_methods is not None:
            response["available_methods"] = available_methods
        
        if request_id is not None:
            response["request_id"] = request_id
        
        logger.debug(f"Formatted error response: status_code={status_code}, method={method}")
        return response
    
    @staticmethod
    def format_success_response(
        result: Dict[str, Any],
        method: str,
        request_id: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Format successful inference responses with consistent structure.
        
        Args:
            result: Inference result data
            method: Method used for inference
            request_id: Request identifier (if available)
            processing_time: Processing time in seconds (if available)
            
        Returns:
            Standardized success response dictionary
        """
        response = {
            "status": "success",
            "method": method,
            "result": result,
            "timestamp": time.time()
        }
        
        if request_id is not None:
            response["request_id"] = request_id
        
        if processing_time is not None:
            response["processing_time_ms"] = int(processing_time * 1000)
        
        logger.debug(f"Formatted success response: method={method}, request_id={request_id}")
        return response
    
    @staticmethod
    def format_health_response(
        orchestrator_status: Dict[str, Any],
        individual_agents: Dict[str, Any],
        service_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format health check responses with consistent terminology.
        
        Args:
            orchestrator_status: Status information for orchestrator
            individual_agents: Status information for individual agents
            service_info: Additional service information (optional)
            
        Returns:
            Standardized health response dictionary
        """
        # Determine overall health status
        orchestrator_healthy = orchestrator_status.get("status") == "available"
        agents_count = len(individual_agents)
        
        # Calculate available methods from individual agents
        available_methods = []
        if orchestrator_healthy:
            available_methods.append("orchestrator")
        
        # Add individual agent methods
        for agent_name, agent_info in individual_agents.items():
            if agent_info.get("is_available", False):
                available_methods.append(agent_name)
        
        response = {
            "status": "healthy" if (orchestrator_healthy or agents_count > 0) else "unhealthy",
            "service": "multilingual-inference-orchestrator",
            "timestamp": time.time(),
            "orchestrator_status": orchestrator_status.get("status", "unknown"),
            "available_methods": available_methods,
            "individual_agents_count": agents_count,
            "orchestrator_agents_count": orchestrator_status.get("agents_count", 0)
        }
        
        # Add service information if provided
        if service_info:
            response.update(service_info)
        
        # Add backward compatibility fields
        response["agents_count"] = agents_count  # For backward compatibility
        
        logger.debug(f"Formatted health response: available_methods={len(available_methods)}, agents_count={agents_count}")
        return response
    
    @staticmethod
    def format_root_response(
        available_methods: List[str],
        service_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format root endpoint response with consistent terminology.
        
        Args:
            available_methods: List of currently available methods
            service_info: Additional service information (optional)
            
        Returns:
            Standardized root response dictionary
        """
        response = {
            "status": "healthy",  # Add status field for consistency
            "service": "multilingual-inference-orchestrator",
            "version": "1.0.0",
            "endpoints": {
                "health": "GET /health",
                "inference": "POST /infer"
            },
            "available_methods": available_methods,
            "method_descriptions": {
                "orchestrator": "Use all available agents with best result selection",
                "simple": "Basic pattern matching without external dependencies",
                "rag": "Vector similarity search using sentence transformers",
                "hybrid": "Sequential pipeline combining NER, RAG, and LLM",
                "ner": "Named entity recognition for brand extraction",
                "llm": "Large language model inference",
                "finetuned_nova_llm": "Fine-tuned Nova model specialized for brand extraction"
            },
            "request_format": {
                "product_name": "string (required)",
                "language_hint": "string (optional: en, th, mixed, auto)",
                "method": f"string (optional: {', '.join(available_methods)})"
            },
            "timestamp": time.time()
        }
        
        # Add service information if provided
        if service_info:
            response.update(service_info)
        
        logger.debug(f"Formatted root response: available_methods={len(available_methods)}")
        return response
    
    @staticmethod
    def format_validation_error(
        field_name: str,
        field_value: Any,
        valid_values: Optional[List[str]] = None,
        error_type: str = "invalid_value"
    ) -> Dict[str, Any]:
        """
        Format validation error responses.
        
        Args:
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            valid_values: List of valid values (if applicable)
            error_type: Type of validation error
            
        Returns:
            Standardized validation error response
        """
        if error_type == "missing_field":
            error_message = f"Missing required field '{field_name}'"
        elif error_type == "invalid_value" and valid_values:
            error_message = f"Invalid {field_name} '{field_value}'. Valid {field_name}s: {', '.join(valid_values)}"
        elif error_type == "invalid_value":
            error_message = f"Invalid {field_name} '{field_value}'"
        else:
            error_message = f"Validation error for field '{field_name}': {error_type}"
        
        return ResponseFormatter.format_error_response(
            status_code=400,
            error_message=error_message,
            method=field_value if field_name == "method" else None,
            available_methods=valid_values if field_name == "method" else None
        )
    
    @staticmethod
    def format_service_unavailable_error(
        method: str,
        reason: str,
        available_methods: List[str]
    ) -> Dict[str, Any]:
        """
        Format service unavailable error responses.
        
        Args:
            method: Method that is unavailable
            reason: Reason for unavailability
            available_methods: List of currently available methods
            
        Returns:
            Standardized service unavailable error response
        """
        error_message = f"Method '{method}' not available: {reason}. Available methods: {', '.join(available_methods)}"
        
        return ResponseFormatter.format_error_response(
            status_code=503,
            error_message=error_message,
            method=method,
            available_methods=available_methods
        )
    
    @staticmethod
    def format_internal_error(
        error: Exception,
        method: Optional[str] = None,
        request_id: Optional[str] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Format internal server error responses.
        
        Args:
            error: Exception that occurred
            method: Method being processed (if applicable)
            request_id: Request identifier (if available)
            include_details: Whether to include detailed error information
            
        Returns:
            Standardized internal error response
        """
        if include_details:
            error_message = f"Internal server error: {str(error)}"
        else:
            error_message = "Internal server error occurred"
        
        response = ResponseFormatter.format_error_response(
            status_code=500,
            error_message=error_message,
            method=method,
            request_id=request_id
        )
        
        if include_details:
            response["error_type"] = type(error).__name__
        
        logger.error(f"Internal error formatted: {type(error).__name__}: {str(error)}")
        return response
    
    @staticmethod
    def add_cors_headers(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add CORS headers to response (for future use).
        
        Args:
            response: Response dictionary to modify
            
        Returns:
            Response with CORS headers added
        """
        # This method is for future extensibility
        # CORS headers are typically handled at the HTTP server level
        response["_cors_enabled"] = True
        return response
    
    @staticmethod
    def validate_response_format(response: Dict[str, Any]) -> bool:
        """
        Validate that a response follows the expected format.
        
        Args:
            response: Response dictionary to validate
            
        Returns:
            True if response format is valid, False otherwise
        """
        required_fields = ["status", "timestamp"]
        
        # Check required fields
        for field in required_fields:
            if field not in response:
                logger.warning(f"Response missing required field: {field}")
                return False
        
        # Check status values
        valid_statuses = ["success", "error", "healthy", "unhealthy"]
        if response["status"] not in valid_statuses:
            logger.warning(f"Invalid status value: {response['status']}")
            return False
        
        # Check error responses have error field
        if response["status"] == "error" and "error" not in response:
            logger.warning("Error response missing error field")
            return False
        
        # Check success responses have result field
        if response["status"] == "success" and "result" not in response:
            logger.warning("Success response missing result field")
            return False
        
        return True