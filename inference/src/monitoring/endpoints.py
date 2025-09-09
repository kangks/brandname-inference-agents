"""
HTTP endpoints for health checks and diagnostics.

This module provides REST API endpoints for monitoring agent health,
running diagnostics, and accessing system metrics following PEP 8 standards.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import logging
from dataclasses import asdict

try:
    from fastapi import FastAPI, HTTPException, Query, Path
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    BaseModel = None

from .health_checker import HealthChecker, HealthStatus, SystemHealth
from .diagnostics import DiagnosticsCollector, SystemDiagnostics
from .cloudwatch_integration import MetricsCollector, CloudWatchMonitor
from .logger import get_inference_logger
from ..agents.base_agent import BaseAgent


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    
    status: str
    timestamp: float
    components: Dict[str, Any]
    system_metrics: Optional[Dict[str, Any]] = None


class DiagnosticsResponse(BaseModel):
    """Response model for diagnostics endpoints."""
    
    overall_status: str
    timestamp: float
    tests: List[Dict[str, Any]]
    system_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class MonitoringEndpoints:
    """
    HTTP endpoints for monitoring and diagnostics.
    
    Provides REST API endpoints for health checks, diagnostics,
    and system monitoring with proper error handling.
    """
    
    def __init__(self) -> None:
        """Initialize monitoring endpoints."""
        self.logger = get_inference_logger()
        self.health_checker = HealthChecker()
        self.diagnostics_collector = DiagnosticsCollector()
        self.metrics_collector = MetricsCollector()
        self.cloudwatch_monitor = CloudWatchMonitor()
        
        # Create FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Multilingual Inference Monitoring API",
                description="Health checks and diagnostics for the inference system",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("FastAPI not available - HTTP endpoints disabled")
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent for monitoring.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.health_checker.register_agent(name, agent)
        self.diagnostics_collector.register_agent(name, agent)
        self.logger.info(f"Registered agent '{name}' for monitoring endpoints")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        if not self.app:
            return
        
        @self.app.get("/health", response_model=Dict[str, Any])
        async def health_check():
            """Get system health status."""
            try:
                health = await self.health_checker.check_system_health()
                return health.to_dict()
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health/{agent_name}", response_model=Dict[str, Any])
        async def agent_health_check(agent_name: str = Path(..., description="Agent name")):
            """Get health status for a specific agent."""
            try:
                if agent_name not in self.health_checker.agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
                
                agent = self.health_checker.agents[agent_name]
                health = await self.health_checker._check_agent_health(agent_name, agent)
                return health.to_dict()
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Agent health check failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/diagnostics", response_model=Dict[str, Any])
        async def full_diagnostics():
            """Run comprehensive system diagnostics."""
            try:
                diagnostics = await self.diagnostics_collector.run_full_diagnostics()
                return diagnostics.to_dict()
            except Exception as e:
                self.logger.error(f"Diagnostics failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/diagnostics/{agent_name}", response_model=Dict[str, Any])
        async def agent_diagnostics(agent_name: str = Path(..., description="Agent name")):
            """Run diagnostics for a specific agent."""
            try:
                diagnostics = await self.diagnostics_collector.run_agent_diagnostics(agent_name)
                return diagnostics
            except Exception as e:
                self.logger.error(f"Agent diagnostics failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/diagnostics/inference", response_model=Dict[str, Any])
        async def inference_diagnostics(
            product_name: str = Query(..., description="Product name to test"),
            language_hint: Optional[str] = Query(None, description="Language hint")
        ):
            """Run diagnostics for a specific inference request."""
            try:
                diagnostics = await self.diagnostics_collector.run_inference_diagnostics(
                    product_name, language_hint
                )
                return diagnostics
            except Exception as e:
                self.logger.error(f"Inference diagnostics failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics", response_model=Dict[str, Any])
        async def get_metrics():
            """Get current metrics buffer."""
            try:
                metrics = self.metrics_collector.get_buffered_metrics()
                return {
                    "metrics_count": len(metrics),
                    "metrics": metrics,
                    "namespace": self.metrics_collector.namespace
                }
            except Exception as e:
                self.logger.error(f"Failed to get metrics: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/metrics/flush", response_model=Dict[str, Any])
        async def flush_metrics():
            """Flush metrics to CloudWatch."""
            try:
                success = await self.metrics_collector.flush_metrics()
                return {
                    "success": success,
                    "message": "Metrics flushed successfully" if success else "Failed to flush metrics"
                }
            except Exception as e:
                self.logger.error(f"Failed to flush metrics: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health/trends", response_model=Dict[str, Any])
        async def health_trends(
            hours: int = Query(24, description="Number of hours to analyze", ge=1, le=168)
        ):
            """Get health trends over time."""
            try:
                trends = self.health_checker.get_health_trends(hours)
                return trends
            except Exception as e:
                self.logger.error(f"Failed to get health trends: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cloudwatch/dashboard", response_model=Dict[str, Any])
        async def create_dashboard(
            dashboard_name: str = Query("multilingual-inference", description="Dashboard name")
        ):
            """Create CloudWatch dashboard."""
            try:
                success = await self.cloudwatch_monitor.create_dashboard(dashboard_name)
                return {
                    "success": success,
                    "dashboard_name": dashboard_name,
                    "message": "Dashboard created successfully" if success else "Failed to create dashboard"
                }
            except Exception as e:
                self.logger.error(f"Failed to create dashboard: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cloudwatch/alarms", response_model=Dict[str, Any])
        async def create_alarms():
            """Create default CloudWatch alarms."""
            try:
                alarms = self.cloudwatch_monitor.get_default_alarms()
                success = await self.cloudwatch_monitor.create_alarms(alarms)
                return {
                    "success": success,
                    "alarms_count": len(alarms),
                    "message": f"Created {len(alarms)} alarms" if success else "Failed to create alarms"
                }
            except Exception as e:
                self.logger.error(f"Failed to create alarms: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status", response_model=Dict[str, Any])
        async def system_status():
            """Get comprehensive system status."""
            try:
                # Get health status
                health = await self.health_checker.check_system_health()
                
                # Get basic diagnostics
                diagnostics = await self.diagnostics_collector.run_full_diagnostics()
                
                # Get metrics info
                metrics_info = {
                    "buffered_metrics": len(self.metrics_collector.metrics_buffer),
                    "namespace": self.metrics_collector.namespace
                }
                
                return {
                    "timestamp": time.time(),
                    "health": health.to_dict(),
                    "diagnostics_summary": {
                        "overall_status": diagnostics.overall_status,
                        "tests_count": len(diagnostics.tests),
                        "failed_tests": len([t for t in diagnostics.tests if t.status == "fail"])
                    },
                    "metrics": metrics_info,
                    "registered_agents": list(self.health_checker.agents.keys())
                }
            except Exception as e:
                self.logger.error(f"Failed to get system status: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))


def create_monitoring_app() -> Optional[FastAPI]:
    """
    Create FastAPI application for monitoring endpoints.
    
    Returns:
        FastAPI application instance or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        logging.warning("FastAPI not available - monitoring endpoints disabled")
        return None
    
    endpoints = MonitoringEndpoints()
    return endpoints.app


def create_monitoring_endpoints() -> MonitoringEndpoints:
    """
    Create monitoring endpoints instance.
    
    Returns:
        MonitoringEndpoints instance
    """
    return MonitoringEndpoints()


# Standalone health check function for simple deployments
async def simple_health_check(agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
    """
    Simple health check function that doesn't require FastAPI.
    
    Args:
        agents: Dictionary of agents to check
        
    Returns:
        Health status dictionary
    """
    health_checker = HealthChecker()
    
    # Register agents
    for name, agent in agents.items():
        health_checker.register_agent(name, agent)
    
    # Run health check
    health = await health_checker.check_system_health()
    
    return health.to_dict()


# Standalone diagnostics function for simple deployments
async def simple_diagnostics(agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
    """
    Simple diagnostics function that doesn't require FastAPI.
    
    Args:
        agents: Dictionary of agents to check
        
    Returns:
        Diagnostics results dictionary
    """
    diagnostics_collector = DiagnosticsCollector()
    
    # Register agents
    for name, agent in agents.items():
        diagnostics_collector.register_agent(name, agent)
    
    # Run diagnostics
    diagnostics = await diagnostics_collector.run_full_diagnostics()
    
    return diagnostics.to_dict()