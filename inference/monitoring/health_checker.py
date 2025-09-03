"""
Health check endpoints and monitoring for agent services.

This module provides comprehensive health checking capabilities for all
inference agents and system components following PEP 8 standards.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import psutil
import aiohttp

from ..models.data_models import AgentHealth
from ..agents.base_agent import BaseAgent


class HealthStatus(Enum):
    """Health status enumeration."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    
    component_name: str
    status: HealthStatus
    last_check: float
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class SystemHealth:
    """Overall system health information."""
    
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    check_timestamp: float
    system_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "check_timestamp": self.check_timestamp,
            "system_metrics": self.system_metrics
        }


class HealthChecker:
    """
    Comprehensive health checker for inference system components.
    
    Provides health monitoring for agents, external services, and system resources
    with configurable check intervals and thresholds.
    """
    
    def __init__(
        self,
        check_timeout: float = 10.0,
        degraded_threshold: float = 0.7,
        unhealthy_threshold: float = 0.5
    ) -> None:
        """
        Initialize health checker.
        
        Args:
            check_timeout: Timeout for individual health checks
            degraded_threshold: Threshold for degraded status (fraction of healthy components)
            unhealthy_threshold: Threshold for unhealthy status (fraction of healthy components)
        """
        self.check_timeout = check_timeout
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self.external_services: Dict[str, Callable] = {}
        self.custom_checks: Dict[str, Callable] = {}
        
        # Health check history for trend analysis
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 100
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent for health monitoring.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.agents[name] = agent
        self.logger.info(f"Registered agent '{name}' for health monitoring")
    
    def register_external_service(
        self,
        name: str,
        check_function: Callable[[], Any]
    ) -> None:
        """
        Register an external service health check.
        
        Args:
            name: Service name
            check_function: Async function that performs the health check
        """
        self.external_services[name] = check_function
        self.logger.info(f"Registered external service '{name}' for health monitoring")
    
    def register_custom_check(
        self,
        name: str,
        check_function: Callable[[], Any]
    ) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Check name
            check_function: Async function that performs the check
        """
        self.custom_checks[name] = check_function
        self.logger.info(f"Registered custom check '{name}' for health monitoring")
    
    async def check_system_health(self) -> SystemHealth:
        """
        Perform comprehensive system health check.
        
        Returns:
            SystemHealth object with complete health information
        """
        start_time = time.time()
        components = {}
        
        # Check all registered agents
        for agent_name, agent in self.agents.items():
            components[f"agent_{agent_name}"] = await self._check_agent_health(agent_name, agent)
        
        # Check external services
        for service_name, check_func in self.external_services.items():
            components[f"service_{service_name}"] = await self._check_external_service(
                service_name, check_func
            )
        
        # Check custom components
        for check_name, check_func in self.custom_checks.items():
            components[f"custom_{check_name}"] = await self._check_custom_component(
                check_name, check_func
            )
        
        # Add system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Determine overall health status
        overall_status = self._calculate_overall_status(components)
        
        # Create system health object
        system_health = SystemHealth(
            overall_status=overall_status,
            components=components,
            check_timestamp=start_time,
            system_metrics=system_metrics
        )
        
        # Store in history
        self._store_health_history(system_health)
        
        self.logger.info(
            f"System health check completed: {overall_status.value} "
            f"({len([c for c in components.values() if c.status == HealthStatus.HEALTHY])}"
            f"/{len(components)} components healthy)"
        )
        
        return system_health
    
    async def _check_agent_health(self, agent_name: str, agent: BaseAgent) -> ComponentHealth:
        """
        Check health of a specific agent.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            
        Returns:
            ComponentHealth for the agent
        """
        start_time = time.time()
        
        try:
            # Use agent's built-in health check if available
            if hasattr(agent, 'health_check'):
                health_result = await asyncio.wait_for(
                    agent.health_check(),
                    timeout=self.check_timeout
                )
                
                response_time = (time.time() - start_time) * 1000
                
                return ComponentHealth(
                    component_name=f"agent_{agent_name}",
                    status=HealthStatus.HEALTHY if health_result.is_healthy else HealthStatus.UNHEALTHY,
                    last_check=start_time,
                    response_time_ms=response_time,
                    error_message=health_result.error_message,
                    metadata={
                        "agent_type": type(agent).__name__,
                        "agent_response_time": health_result.response_time
                    }
                )
            else:
                # Basic health check - just verify agent is accessible
                response_time = (time.time() - start_time) * 1000
                
                return ComponentHealth(
                    component_name=f"agent_{agent_name}",
                    status=HealthStatus.HEALTHY,
                    last_check=start_time,
                    response_time_ms=response_time,
                    metadata={"agent_type": type(agent).__name__}
                )
                
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component_name=f"agent_{agent_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=f"Health check timed out after {self.check_timeout}s"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component_name=f"agent_{agent_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_external_service(
        self,
        service_name: str,
        check_function: Callable
    ) -> ComponentHealth:
        """
        Check health of an external service.
        
        Args:
            service_name: Name of the service
            check_function: Function to perform the health check
            
        Returns:
            ComponentHealth for the service
        """
        start_time = time.time()
        
        try:
            # Execute the check function with timeout
            result = await asyncio.wait_for(
                check_function(),
                timeout=self.check_timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Interpret result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                metadata = None
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                metadata = result.get("metadata")
            else:
                status = HealthStatus.HEALTHY  # Assume healthy if no exception
                metadata = {"result": str(result)}
            
            return ComponentHealth(
                component_name=f"service_{service_name}",
                status=status,
                last_check=start_time,
                response_time_ms=response_time,
                metadata=metadata
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component_name=f"service_{service_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=f"Service check timed out after {self.check_timeout}s"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component_name=f"service_{service_name}",
                status=HealthStatus.UNHEALTHY,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    async def _check_custom_component(
        self,
        check_name: str,
        check_function: Callable
    ) -> ComponentHealth:
        """
        Check health of a custom component.
        
        Args:
            check_name: Name of the check
            check_function: Function to perform the check
            
        Returns:
            ComponentHealth for the component
        """
        # Similar to external service check but with different naming
        return await self._check_external_service(check_name, check_function)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system-level metrics.
        
        Returns:
            Dictionary of system metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except Exception:
                network_metrics = None
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "percent": (disk.used / disk.total) * 100
                },
                "network": network_metrics
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """
        Calculate overall system health status based on component health.
        
        Args:
            components: Dictionary of component health information
            
        Returns:
            Overall system health status
        """
        if not components:
            return HealthStatus.UNKNOWN
        
        healthy_count = sum(1 for comp in components.values() if comp.status == HealthStatus.HEALTHY)
        total_count = len(components)
        healthy_ratio = healthy_count / total_count
        
        if healthy_ratio >= self.degraded_threshold:
            return HealthStatus.HEALTHY
        elif healthy_ratio >= self.unhealthy_threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
    
    def _store_health_history(self, health: SystemHealth) -> None:
        """
        Store health check result in history.
        
        Args:
            health: SystemHealth object to store
        """
        self.health_history.append(health)
        
        # Trim history if it exceeds maximum size
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get health trends over the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_checks = [
            h for h in self.health_history 
            if h.check_timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return {"error": "No health data available for the specified period"}
        
        # Calculate trends
        status_counts = {}
        component_trends = {}
        
        for health in recent_checks:
            # Overall status trend
            status = health.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Component trends
            for comp_name, comp_health in health.components.items():
                if comp_name not in component_trends:
                    component_trends[comp_name] = []
                component_trends[comp_name].append({
                    "timestamp": health.check_timestamp,
                    "status": comp_health.status.value,
                    "response_time_ms": comp_health.response_time_ms
                })
        
        return {
            "period_hours": hours,
            "total_checks": len(recent_checks),
            "overall_status_distribution": status_counts,
            "component_trends": component_trends,
            "latest_check": recent_checks[-1].to_dict() if recent_checks else None
        }


# Health check functions for common services
async def check_aws_bedrock_health() -> Dict[str, Any]:
    """Check AWS Bedrock service health."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        client = boto3.client('bedrock-runtime')
        
        # Simple API call to check connectivity
        response = client.list_foundation_models()
        
        return {
            "status": "healthy",
            "metadata": {
                "models_available": len(response.get('modelSummaries', [])),
                "service": "bedrock"
            }
        }
        
    except NoCredentialsError:
        return {
            "status": "unhealthy",
            "metadata": {"error": "AWS credentials not configured"}
        }
    except ClientError as e:
        return {
            "status": "unhealthy", 
            "metadata": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "metadata": {"error": str(e)}
        }


async def check_milvus_health(host: str = "localhost", port: int = 19530) -> Dict[str, Any]:
    """Check Milvus vector database health."""
    try:
        from pymilvus import connections, utility
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        
        # Check if server is healthy
        if utility.get_server_version():
            return {
                "status": "healthy",
                "metadata": {
                    "host": host,
                    "port": port,
                    "service": "milvus"
                }
            }
        else:
            return {
                "status": "unhealthy",
                "metadata": {"error": "Unable to get server version"}
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "metadata": {"error": str(e)}
        }


async def check_http_endpoint_health(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Check HTTP endpoint health."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "metadata": {
                            "url": url,
                            "status_code": response.status,
                            "response_time_ms": timeout * 1000  # Approximate
                        }
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "metadata": {
                            "url": url,
                            "status_code": response.status,
                            "error": f"HTTP {response.status}"
                        }
                    }
                    
    except Exception as e:
        return {
            "status": "unhealthy",
            "metadata": {
                "url": url,
                "error": str(e)
            }
        }