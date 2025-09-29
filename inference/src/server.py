"""
Simple HTTP server for the multilingual product inference system.

This module provides a basic HTTP server for running inference agents
in containerized environments.
"""

import asyncio
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from .config.settings import get_config, setup_logging
from .config.method_mapper import MethodAgentMapper
from .models.data_models import ProductInput, LanguageHint
from .models.response_formatter import ResponseFormatter
from .agents.orchestrator_agent import create_orchestrator_agent
from .agents.registry import get_agent_registry, initialize_default_agents


logger = logging.getLogger(__name__)


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for inference endpoints."""
    
    def __init__(self, *args, orchestrator=None, individual_agents=None, **kwargs):
        self.orchestrator = orchestrator
        self.individual_agents = individual_agents or {}
        
        # Initialize method mapper with agent registry
        agent_registry = get_agent_registry()
        self.method_mapper = MethodAgentMapper(agent_registry)
        
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/health':
                self._handle_health_check()
            elif parsed_path.path == '/':
                self._handle_root()
            else:
                self._send_error(404, "Not Found")
                
        except Exception as e:
            logger.error(f"Error handling GET request: {str(e)}")
            self._send_error(500, "Internal Server Error")
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/infer':
                self._handle_inference()
            else:
                self._send_error(404, "Not Found")
                
        except Exception as e:
            logger.error(f"Error handling POST request: {str(e)}")
            self._send_error(500, "Internal Server Error")
    
    def _handle_health_check(self):
        """Handle health check endpoint."""
        try:
            config = get_config()
            
            # Prepare orchestrator status
            orchestrator_status = {}
            if hasattr(self, 'orchestrator') and self.orchestrator:
                orchestrator_status = {
                    "status": "available",
                    "agents_count": len(getattr(self.orchestrator, 'agents', []))
                }
            else:
                orchestrator_status = {
                    "status": "not_initialized",
                    "agents_count": 0
                }
            
            # Prepare individual agents status
            individual_agents = {}
            for agent_name, agent in getattr(self, 'individual_agents', {}).items():
                individual_agents[agent_name] = {
                    "is_available": agent is not None,
                    "is_initialized": agent.is_initialized() if agent else False
                }
            
            # Additional service information
            service_info = {
                "environment": config.environment.value,
                "aws_region": config.aws.region
            }
            
            # Format health response using ResponseFormatter
            health_response = ResponseFormatter.format_health_response(
                orchestrator_status=orchestrator_status,
                individual_agents=individual_agents,
                service_info=service_info
            )
            
            self._send_json_response(200, health_response)
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            error_response = ResponseFormatter.format_internal_error(
                error=e,
                include_details=False
            )
            self._send_json_response(500, error_response)
    
    def _handle_root(self):
        """Handle root endpoint."""
        try:
            # Get available methods dynamically
            available_methods = self.method_mapper.get_valid_methods(include_orchestrator=True)
            
            # Format root response using ResponseFormatter
            response = ResponseFormatter.format_root_response(
                available_methods=available_methods
            )
            
            self._send_json_response(200, response)
            
        except Exception as e:
            logger.error(f"Root endpoint failed: {str(e)}")
            error_response = ResponseFormatter.format_internal_error(
                error=e,
                include_details=False
            )
            self._send_json_response(500, error_response)
    
    def _handle_inference(self):
        """Handle inference endpoint."""
        request_id = f"req-{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                error_response = ResponseFormatter.format_validation_error(
                    field_name="request_body",
                    field_value="empty",
                    error_type="missing_field"
                )
                self._send_json_response(400, error_response)
                return
            
            body = self.rfile.read(content_length)
            try:
                request_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                error_response = ResponseFormatter.format_error_response(
                    status_code=400,
                    error_message="Invalid JSON format in request body",
                    request_id=request_id
                )
                self._send_json_response(400, error_response)
                return
            
            # Validate required fields
            if 'product_name' not in request_data:
                error_response = ResponseFormatter.format_validation_error(
                    field_name="product_name",
                    field_value=None,
                    error_type="missing_field"
                )
                error_response["request_id"] = request_id
                self._send_json_response(400, error_response)
                return
            
            product_name = request_data['product_name']
            language_hint = request_data.get('language_hint', 'auto')
            method = request_data.get('method', 'orchestrator')  # Default to orchestrator
            
            # Convert language hint
            try:
                lang_hint = LanguageHint(language_hint)
            except ValueError:
                lang_hint = LanguageHint.AUTO
            
            # Validate method parameter using MethodAgentMapper
            is_valid, error_message = self.method_mapper.validate_method(method, include_orchestrator=True)
            if not is_valid:
                error_response = ResponseFormatter.format_validation_error(
                    field_name="method",
                    field_value=method,
                    valid_values=self.method_mapper.get_valid_methods(include_orchestrator=True),
                    error_type="invalid_value"
                )
                error_response["request_id"] = request_id
                self._send_json_response(400, error_response)
                return
            
            # Create product input
            product_input = ProductInput(
                product_name=product_name,
                language_hint=lang_hint
            )
            
            # Handle different inference methods
            if method == 'orchestrator':
                # Use orchestrator with all available agents
                result = self._handle_orchestrator_inference_sync(product_input, method, request_id)
            else:
                # Use specific agent method
                result = self._handle_specific_agent_inference_sync(product_input, method, request_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format success response if result doesn't already have proper format
            if not isinstance(result, dict) or "status" not in result:
                formatted_result = ResponseFormatter.format_success_response(
                    result=result,
                    method=method,
                    request_id=request_id,
                    processing_time=processing_time
                )
                self._send_json_response(200, formatted_result)
            else:
                # Result already formatted, just add missing fields if needed
                if "request_id" not in result:
                    result["request_id"] = request_id
                if "processing_time_ms" not in result and processing_time:
                    result["processing_time_ms"] = int(processing_time * 1000)
                self._send_json_response(200, result)
            
        except Exception as e:
            logger.error(f"Inference request failed: {str(e)}")
            error_response = ResponseFormatter.format_internal_error(
                error=e,
                method=request_data.get('method') if 'request_data' in locals() else None,
                request_id=request_id,
                include_details=False
            )
            self._send_json_response(500, error_response)
    
    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str, method: Optional[str] = None, request_id: Optional[str] = None):
        """Send error response using ResponseFormatter."""
        error_response = ResponseFormatter.format_error_response(
            status_code=status_code,
            error_message=message,
            method=method,
            request_id=request_id
        )
        self._send_json_response(status_code, error_response)
    

    

    
    def _handle_orchestrator_inference_sync(self, product_input: ProductInput, method: str, request_id: str) -> Dict[str, Any]:
        """Handle orchestrator inference with all available agents."""
        logger.info(f"Checking orchestrator availability...")
        logger.info(f"  - Has orchestrator attr: {hasattr(self, 'orchestrator')}")
        logger.info(f"  - Orchestrator exists: {self.orchestrator is not None if hasattr(self, 'orchestrator') else False}")
        logger.info(f"  - Has agents attr: {hasattr(self.orchestrator, 'agents') if hasattr(self, 'orchestrator') and self.orchestrator else False}")
        logger.info(f"  - Agents count: {len(self.orchestrator.agents) if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'agents') else 0}")
        
        if (hasattr(self, 'orchestrator') and self.orchestrator and 
            hasattr(self.orchestrator, 'agents') and len(self.orchestrator.agents) > 0):
            
            logger.info(f"Using orchestrator with {len(self.orchestrator.agents)} agents for inference")
            try:
                # Use orchestrator for inference (run in thread)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_orchestrator_inference, product_input)
                    result = future.result(timeout=30)  # 30 second timeout
                
                # Add orchestrator metadata to result
                if isinstance(result, dict):
                    result['method'] = method
                    result['orchestrator_agents'] = list(self.orchestrator.agents.keys())
                    result['request_id'] = request_id
                
                return result
                
            except Exception as e:
                logger.error(f"Orchestrator inference failed: {str(e)}")
                # Return error response for orchestrator failure
                return ResponseFormatter.format_internal_error(
                    error=e,
                    method=method,
                    request_id=request_id,
                    include_details=False
                )
        
        # Return status response when orchestrator is not available
        available_methods = self.method_mapper.get_valid_methods(include_orchestrator=False)
        return ResponseFormatter.format_service_unavailable_error(
            method=method,
            reason="Orchestrator service is running but no agents are available for inference",
            available_methods=available_methods
        )
    
    def _handle_specific_agent_inference_sync(self, product_input: ProductInput, method: str, request_id: str) -> Dict[str, Any]:
        """Handle inference using a specific agent method."""
        logger.info(f"Using specific agent method: {method}")
        
        # Check if the requested method is available
        # First try to get from method mapper (registry), then fallback to individual agents
        agent = self.method_mapper.get_agent_for_method(method)
        if agent is None and method in self.individual_agents:
            agent = self.individual_agents[method]
        
        if agent is None:
            available_methods = self.method_mapper.get_valid_methods(include_orchestrator=True)
            # Also add any individual agents not in registry
            for agent_name in self.individual_agents.keys():
                if agent_name not in available_methods:
                    available_methods.append(agent_name)
            
            return ResponseFormatter.format_service_unavailable_error(
                method=method,
                reason="Method not available or agent not initialized",
                available_methods=available_methods
            )
        
        try:
            # Use the agent obtained from method mapper
            
            # Run agent inference in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_agent_inference, agent, product_input)
                agent_result = future.result(timeout=30)  # 30 second timeout
            
            # Convert agent result to standard response format
            if agent_result and agent_result.get("success"):
                result_data = agent_result.get("result")
                
                # Create standardized response
                response = {
                    "product_name": product_input.product_name,
                    "language": product_input.language_hint.value,
                    "method": method,
                    "request_id": request_id,
                    "timestamp": time.time()
                }
                
                # Add method-specific result formatting
                if method == "simple":
                    response.update({
                        "brand_predictions": [{
                            "brand": result_data.predicted_brand,
                            "confidence": result_data.confidence,
                            "method": method
                        }],
                        "reasoning": getattr(result_data, 'reasoning', ''),
                        "processing_time_ms": int(result_data.processing_time * 1000)
                    })
                elif method == "rag":
                    response.update({
                        "brand_predictions": [{
                            "brand": result_data.predicted_brand,
                            "confidence": result_data.confidence,
                            "method": method
                        }],
                        "similar_products": [
                            {
                                "product_name": p.product_name,
                                "brand": p.brand,
                                "similarity_score": p.similarity_score
                            } for p in result_data.similar_products[:3]  # Top 3
                        ],
                        "processing_time_ms": int(result_data.processing_time * 1000),
                        "embedding_model": result_data.embedding_model
                    })
                elif method == "hybrid":
                    response.update({
                        "brand_predictions": [{
                            "brand": result_data.predicted_brand,
                            "confidence": result_data.confidence,
                            "method": method
                        }],
                        "pipeline_steps": result_data.pipeline_steps,
                        "contributions": {
                            "ner": result_data.ner_contribution,
                            "rag": result_data.rag_contribution,
                            "llm": result_data.llm_contribution
                        },
                        "processing_time_ms": int(result_data.processing_time * 1000)
                    })
                elif method in ["ner", "llm", "finetuned_nova_llm"]:
                    # Handle NER, LLM, and Fine-tuned Nova agents
                    if hasattr(result_data, 'predicted_brand'):
                        brand = result_data.predicted_brand
                        confidence = result_data.confidence
                    elif hasattr(result_data, 'entities') and result_data.entities:
                        # NER result - extract best brand entity
                        brand_entities = [e for e in result_data.entities if e.entity_type.value == "BRAND"]
                        if brand_entities:
                            best_entity = max(brand_entities, key=lambda x: x.confidence)
                            brand = best_entity.text
                            confidence = best_entity.confidence
                        else:
                            brand = "Unknown"
                            confidence = 0.0
                    else:
                        brand = "Unknown"
                        confidence = 0.0
                    
                    response.update({
                        "brand_predictions": [{
                            "brand": brand,
                            "confidence": confidence,
                            "method": method
                        }],
                        "processing_time_ms": int(getattr(result_data, 'processing_time', 0) * 1000)
                    })
                    
                    if method == "ner" and hasattr(result_data, 'entities'):
                        response["entities"] = [
                            {
                                "text": e.text,
                                "type": e.entity_type.value,
                                "confidence": e.confidence,
                                "start": e.start_pos,
                                "end": e.end_pos
                            } for e in result_data.entities
                        ]
                    elif method in ["llm", "finetuned_nova_llm"] and hasattr(result_data, 'reasoning'):
                        response["reasoning"] = result_data.reasoning
                        
                    # Add special metadata for fine-tuned Nova
                    if method == "finetuned_nova_llm":
                        response["model_type"] = "fine_tuned_nova"
                        response["specialization"] = "brand_extraction"
                
                return response
            else:
                # Agent failed
                error_msg = agent_result.get("error", "Unknown agent error") if agent_result else "Agent returned no result"
                return ResponseFormatter.format_error_response(
                    status_code=500,
                    error_message=f"Method '{method}' processing failed: {error_msg}",
                    method=method,
                    request_id=request_id
                )
                
        except Exception as e:
            logger.error(f"Specific method inference failed for {method}: {str(e)}")
            return ResponseFormatter.format_internal_error(
                error=e,
                method=method,
                request_id=request_id,
                include_details=False
            )
    
    def _sync_agent_inference(self, agent, product_input: ProductInput) -> Dict[str, Any]:
        """Run specific agent inference synchronously."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(agent.process(product_input))
            return result
        finally:
            loop.close()
    
    def _sync_orchestrator_inference(self, product_input: ProductInput) -> Dict[str, Any]:
        """Run orchestrator inference synchronously."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.orchestrator.process(product_input))
            return result
        finally:
            loop.close()
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")


def create_handler_class(orchestrator=None, individual_agents=None):
    """Create handler class with orchestrator and individual agents."""
    class Handler(InferenceHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, orchestrator=orchestrator, individual_agents=individual_agents, **kwargs)
    return Handler


async def initialize_server() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Initialize the inference server."""
    try:
        # Load configuration and setup logging
        config = get_config()
        setup_logging(config)
        
        logger.info("Initializing multilingual inference server...")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"AWS Region: {config.aws.region}")
        
        # Initialize agent registry for individual agents
        individual_agents = None
        try:
            individual_agents = await initialize_default_agents()
            logger.info(f"Individual agents initialized: {list(individual_agents.keys())}")
            
            # Log which agents are available
            for agent_name in individual_agents.keys():
                if "finetuned" in agent_name.lower():
                    logger.info(f"✅ Fine-tuned agent available: {agent_name}")
                else:
                    logger.info(f"📝 Agent available: {agent_name}")
                    
        except Exception as e:
            logger.warning(f"Could not initialize individual agents: {str(e)}")
        
        # Create and initialize orchestrator with default agents
        orchestrator = None
        try:
            orchestrator = create_orchestrator_agent()
            await orchestrator.initialize()
            logger.info(f"Orchestrator agent initialized with {len(orchestrator.agents)} specialized agents")
        except Exception as e:
            logger.warning(f"Could not initialize orchestrator agent: {str(e)}")
            logger.info("Server will run without orchestrator capability")
        
        return orchestrator, individual_agents
        
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        raise


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the HTTP server."""
    try:
        # Initialize server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        orchestrator, individual_agents = loop.run_until_complete(initialize_server())
        
        # Create handler class with orchestrator and individual agents
        handler_class = create_handler_class(orchestrator, individual_agents)
        
        # Create and start server
        server = HTTPServer((host, port), handler_class)
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  GET  / - Service info")
        logger.info("  GET  /health - Health check")
        logger.info("  POST /infer - Product inference")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            server.shutdown()
            
            # Cleanup orchestrator if available
            if orchestrator:
                try:
                    loop.run_until_complete(orchestrator.cleanup())
                    logger.info("Orchestrator cleanup completed")
                except Exception as e:
                    logger.warning(f"Orchestrator cleanup failed: {str(e)}")
            
            logger.info("Server stopped")
            
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    # Get host and port from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    
    run_server(host, port)