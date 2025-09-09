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
from .models.data_models import ProductInput, LanguageHint
from .agents.orchestrator_agent import create_orchestrator_agent


logger = logging.getLogger(__name__)


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for inference endpoints."""
    
    def __init__(self, *args, orchestrator=None, **kwargs):
        self.orchestrator = orchestrator
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
            
            health_status = {
                "status": "healthy",
                "service": "multilingual-inference-orchestrator",
                "environment": config.environment.value,
                "aws_region": config.aws.region,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Check if orchestrator is available
            if hasattr(self, 'orchestrator') and self.orchestrator:
                health_status["orchestrator"] = "available"
                health_status["agents_count"] = len(getattr(self.orchestrator, 'agents', []))
            else:
                health_status["orchestrator"] = "not_initialized"
                health_status["agents_count"] = 0
            

            
            self._send_json_response(200, health_status)
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            error_response = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
            self._send_json_response(500, error_response)
    
    def _handle_root(self):
        """Handle root endpoint."""
        response = {
            "service": "multilingual-inference-orchestrator",
            "version": "1.0.0",
            "endpoints": {
                "health": "GET /health",
                "inference": "POST /infer"
            },
            "inference_methods": {
                "orchestrator": "Use all available agents with best result selection",
                "simple": "Basic pattern matching without external dependencies",
                "rag": "Vector similarity search using sentence transformers",
                "hybrid": "Sequential pipeline combining NER, RAG, and LLM",
                "ner": "Named entity recognition for brand extraction",
                "llm": "Large language model inference"
            },
            "request_format": {
                "product_name": "string (required)",
                "language_hint": "string (optional: en, th, mixed, auto)",
                "method": "string (optional: orchestrator, simple, rag, hybrid, ner, llm)"
            }
        }
        self._send_json_response(200, response)
    
    def _handle_inference(self):
        """Handle inference endpoint."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return
            
            body = self.rfile.read(content_length)
            try:
                request_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
                return
            
            # Validate request
            if 'product_name' not in request_data:
                self._send_error(400, "Missing 'product_name' field")
                return
            
            product_name = request_data['product_name']
            language_hint = request_data.get('language_hint', 'auto')
            method = request_data.get('method', 'orchestrator')  # Default to orchestrator
            
            # Convert language hint
            try:
                lang_hint = LanguageHint(language_hint)
            except ValueError:
                lang_hint = LanguageHint.AUTO
            
            # Validate method parameter
            valid_methods = ['orchestrator', 'simple', 'rag', 'hybrid', 'ner', 'llm']
            if method not in valid_methods:
                self._send_error(400, f"Invalid method '{method}'. Valid methods: {', '.join(valid_methods)}")
                return
            
            # Create product input
            product_input = ProductInput(
                product_name=product_name,
                language_hint=lang_hint
            )
            
            # Handle different inference methods
            if method == 'orchestrator':
                # Use orchestrator with all available agents
                result = self._handle_orchestrator_inference_sync(product_input)
            else:
                # Use specific agent method
                result = self._handle_specific_agent_inference_sync(product_input, method)
            
            self._send_json_response(200, result)
            return

            
        except Exception as e:
            logger.error(f"Inference request failed: {str(e)}")
            error_response = {
                "status": "error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
            self._send_json_response(500, error_response)
    
    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        error_data = {
            "error": message,
            "status_code": status_code,
            "timestamp": asyncio.get_event_loop().time()
        }
        self._send_json_response(status_code, error_data)
    

    

    
    def _handle_orchestrator_inference_sync(self, product_input: ProductInput) -> Dict[str, Any]:
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
                    result['agent_used'] = 'orchestrator'
                    result['orchestrator_agents'] = list(self.orchestrator.agents.keys())
                    result['method'] = 'orchestrator'
                
                return result
                
            except Exception as e:
                logger.error(f"Orchestrator inference failed: {str(e)}")
                # Return error response for orchestrator failure
                return {
                    "status": "error",
                    "error": f"Orchestrator inference failed: {str(e)}",
                    "method": "orchestrator",
                    "timestamp": time.time()
                }
        
        # Return status response when orchestrator is not available
        return {
            "input": {
                "product_name": product_input.product_name,
                "language_hint": product_input.language_hint.value
            },
            "status": "ready",
            "message": "Orchestrator service is running but no agents are available for inference",
            "orchestrator_status": "initialized",
            "registered_agents": len(getattr(self.orchestrator, 'agents', [])) if hasattr(self, 'orchestrator') and self.orchestrator else 0,
            "available_agents": list(getattr(self.orchestrator, 'agents', {}).keys()) if hasattr(self, 'orchestrator') and self.orchestrator else [],
            "method": "orchestrator",
            "next_steps": [
                "Check agent dependencies (spaCy, sentence-transformers, boto3)",
                "Verify AWS credentials for LLM agent",
                "Check Milvus database connectivity for RAG agent",
                "Review agent configuration in settings"
            ],
            "timestamp": time.time()
        }
    
    def _handle_specific_agent_inference_sync(self, product_input: ProductInput, method: str) -> Dict[str, Any]:
        """Handle inference using a specific agent method."""
        logger.info(f"Using specific agent method: {method}")
        
        if not (hasattr(self, 'orchestrator') and self.orchestrator and 
                hasattr(self.orchestrator, 'agents')):
            return {
                "status": "error",
                "error": "Orchestrator not available for agent access",
                "method": method,
                "timestamp": time.time()
            }
        
        # Check if the requested agent is available
        if method not in self.orchestrator.agents:
            available_agents = list(self.orchestrator.agents.keys())
            return {
                "status": "error",
                "error": f"Agent '{method}' not available. Available agents: {available_agents}",
                "method": method,
                "available_agents": available_agents,
                "timestamp": time.time()
            }
        
        try:
            # Get the specific agent
            agent = self.orchestrator.agents[method]
            
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
                    "agent_used": method,
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
                elif method in ["ner", "llm"]:
                    # Handle NER and LLM agents
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
                    elif method == "llm" and hasattr(result_data, 'reasoning'):
                        response["reasoning"] = result_data.reasoning
                
                return response
            else:
                # Agent failed
                error_msg = agent_result.get("error", "Unknown agent error") if agent_result else "Agent returned no result"
                return {
                    "status": "error",
                    "error": f"Agent '{method}' failed: {error_msg}",
                    "method": method,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Specific agent inference failed for {method}: {str(e)}")
            return {
                "status": "error",
                "error": f"Agent '{method}' inference failed: {str(e)}",
                "method": method,
                "timestamp": time.time()
            }
    
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


def create_handler_class(orchestrator=None):
    """Create handler class with orchestrator instance."""
    class Handler(InferenceHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, orchestrator=orchestrator, **kwargs)
    return Handler


async def initialize_server() -> Optional[Any]:
    """Initialize the inference server."""
    try:
        # Load configuration and setup logging
        config = get_config()
        setup_logging(config)
        
        logger.info("Initializing multilingual inference server...")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"AWS Region: {config.aws.region}")
        
        # Create and initialize orchestrator with default agents
        orchestrator = None
        try:
            orchestrator = create_orchestrator_agent()
            await orchestrator.initialize()
            logger.info(f"Orchestrator agent initialized with {len(orchestrator.agents)} agents")
        except Exception as e:
            logger.warning(f"Could not initialize orchestrator agent: {str(e)}")
            logger.info("Server will run without inference capability")
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Server initialization failed: {str(e)}")
        raise


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the HTTP server."""
    try:
        # Initialize server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        orchestrator = loop.run_until_complete(initialize_server())
        
        # Create handler class with orchestrator
        handler_class = create_handler_class(orchestrator)
        
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