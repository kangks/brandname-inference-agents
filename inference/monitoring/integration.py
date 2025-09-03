"""
Integration module for setting up comprehensive monitoring across the inference system.

This module provides easy setup and configuration of all monitoring components
following PEP 8 standards and best practices.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import time

from .logger import setup_structured_logging, InferenceLogger
from .health_checker import (
    HealthChecker, 
    check_aws_bedrock_health,
    check_milvus_health,
    check_http_endpoint_health
)
from .diagnostics import DiagnosticsCollector
from .cloudwatch_integration import CloudWatchMonitor, MetricsCollector
from .endpoints import MonitoringEndpoints
from ..agents.base_agent import BaseAgent


class MonitoringIntegration:
    """
    Comprehensive monitoring integration for the inference system.
    
    Provides centralized setup and management of all monitoring components
    including logging, health checks, diagnostics, and CloudWatch integration.
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        enable_cloudwatch: bool = True,
        enable_endpoints: bool = True,
        aws_region: str = "us-east-1",
        aws_profile: str = "ml-sandbox"
    ) -> None:
        """
        Initialize monitoring integration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_cloudwatch: Whether to enable CloudWatch integration
            enable_endpoints: Whether to enable HTTP monitoring endpoints
            aws_region: AWS region for CloudWatch
            aws_profile: AWS profile name
        """
        self.log_level = log_level
        self.enable_cloudwatch = enable_cloudwatch
        self.enable_endpoints = enable_endpoints
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        
        # Initialize components
        self.logger: Optional[InferenceLogger] = None
        self.health_checker: Optional[HealthChecker] = None
        self.diagnostics_collector: Optional[DiagnosticsCollector] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.cloudwatch_monitor: Optional[CloudWatchMonitor] = None
        self.endpoints: Optional[MonitoringEndpoints] = None
        
        self.setup_logger = logging.getLogger(__name__)
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize all monitoring components."""
        try:
            self.setup_logger.info("Initializing monitoring integration...")
            
            # Setup structured logging
            self.logger = setup_structured_logging(
                log_level=self.log_level,
                log_format="json",
                enable_cloudwatch=self.enable_cloudwatch
            )
            
            # Initialize health checker
            self.health_checker = HealthChecker()
            
            # Initialize diagnostics collector
            self.diagnostics_collector = DiagnosticsCollector()
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            
            # Initialize CloudWatch monitor if enabled
            if self.enable_cloudwatch:
                self.cloudwatch_monitor = CloudWatchMonitor(
                    region_name=self.aws_region,
                    profile_name=self.aws_profile
                )
            
            # Initialize HTTP endpoints if enabled
            if self.enable_endpoints:
                self.endpoints = MonitoringEndpoints()
            
            # Setup external service health checks
            await self._setup_external_health_checks()
            
            self._is_initialized = True
            self.setup_logger.info("Monitoring integration initialized successfully")
            
        except Exception as e:
            self.setup_logger.error(f"Failed to initialize monitoring integration: {str(e)}")
            raise
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent with all monitoring components.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        if not self._is_initialized:
            raise RuntimeError("Monitoring integration not initialized")
        
        # Register with health checker
        if self.health_checker:
            self.health_checker.register_agent(name, agent)
        
        # Register with diagnostics collector
        if self.diagnostics_collector:
            self.diagnostics_collector.register_agent(name, agent)
        
        # Register with endpoints
        if self.endpoints:
            self.endpoints.register_agent(name, agent)
        
        self.setup_logger.info(f"Registered agent '{name}' with monitoring integration")
    
    async def setup_cloudwatch_resources(self) -> Dict[str, bool]:
        """
        Setup CloudWatch dashboards and alarms.
        
        Returns:
            Dictionary with setup results
        """
        if not self.cloudwatch_monitor:
            return {"error": "CloudWatch monitor not initialized"}
        
        results = {}
        
        try:
            # Create dashboard
            dashboard_success = await self.cloudwatch_monitor.create_dashboard(
                "multilingual-inference-dashboard"
            )
            results["dashboard"] = dashboard_success
            
            # Create alarms
            alarms = self.cloudwatch_monitor.get_default_alarms()
            alarms_success = await self.cloudwatch_monitor.create_alarms(alarms)
            results["alarms"] = alarms_success
            results["alarms_count"] = len(alarms)
            
            self.setup_logger.info(f"CloudWatch resources setup: {results}")
            
        except Exception as e:
            self.setup_logger.error(f"Failed to setup CloudWatch resources: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def run_system_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive system health check.
        
        Returns:
            System health results
        """
        if not self.health_checker:
            return {"error": "Health checker not initialized"}
        
        try:
            health = await self.health_checker.check_system_health()
            return health.to_dict()
        except Exception as e:
            self.setup_logger.error(f"System health check failed: {str(e)}")
            return {"error": str(e)}
    
    async def run_system_diagnostics(self) -> Dict[str, Any]:
        """
        Run comprehensive system diagnostics.
        
        Returns:
            System diagnostics results
        """
        if not self.diagnostics_collector:
            return {"error": "Diagnostics collector not initialized"}
        
        try:
            diagnostics = await self.diagnostics_collector.run_full_diagnostics()
            return diagnostics.to_dict()
        except Exception as e:
            self.setup_logger.error(f"System diagnostics failed: {str(e)}")
            return {"error": str(e)}
    
    async def publish_metrics_to_cloudwatch(self) -> bool:
        """
        Publish buffered metrics to CloudWatch.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cloudwatch_monitor or not self.metrics_collector:
            return False
        
        try:
            metrics = self.metrics_collector.metrics_buffer.copy()
            if metrics:
                success = await self.cloudwatch_monitor.publish_metrics(
                    metrics, 
                    self.metrics_collector.namespace
                )
                if success:
                    self.metrics_collector.metrics_buffer.clear()
                return success
            return True
        except Exception as e:
            self.setup_logger.error(f"Failed to publish metrics: {str(e)}")
            return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get status of all monitoring components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            "initialized": self._is_initialized,
            "components": {
                "logger": self.logger is not None,
                "health_checker": self.health_checker is not None,
                "diagnostics_collector": self.diagnostics_collector is not None,
                "metrics_collector": self.metrics_collector is not None,
                "cloudwatch_monitor": self.cloudwatch_monitor is not None,
                "endpoints": self.endpoints is not None
            },
            "configuration": {
                "log_level": self.log_level,
                "enable_cloudwatch": self.enable_cloudwatch,
                "enable_endpoints": self.enable_endpoints,
                "aws_region": self.aws_region,
                "aws_profile": self.aws_profile
            },
            "registered_agents": (
                list(self.health_checker.agents.keys()) 
                if self.health_checker else []
            )
        }
    
    async def _setup_external_health_checks(self) -> None:
        """Setup health checks for external services."""
        if not self.health_checker:
            return
        
        # AWS Bedrock health check
        self.health_checker.register_external_service(
            "aws_bedrock",
            check_aws_bedrock_health
        )
        
        # Milvus health check
        self.health_checker.register_external_service(
            "milvus",
            lambda: check_milvus_health("localhost", 19530)
        )
        
        self.setup_logger.info("External service health checks configured")
    
    async def start_monitoring_loop(self, interval_seconds: int = 60) -> None:
        """
        Start continuous monitoring loop.
        
        Args:
            interval_seconds: Interval between monitoring cycles
        """
        self.setup_logger.info(f"Starting monitoring loop (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Run health check
                if self.health_checker:
                    await self.health_checker.check_system_health()
                
                # Publish metrics to CloudWatch
                if self.enable_cloudwatch:
                    await self.publish_metrics_to_cloudwatch()
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.setup_logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            # Flush any remaining metrics
            if self.metrics_collector:
                await self.metrics_collector.flush_metrics()
            
            # Publish final metrics to CloudWatch
            if self.enable_cloudwatch:
                await self.publish_metrics_to_cloudwatch()
            
            self.setup_logger.info("Monitoring integration cleanup completed")
            
        except Exception as e:
            self.setup_logger.error(f"Error during monitoring cleanup: {str(e)}")


# Factory function for easy setup
def create_monitoring_integration(
    log_level: str = "INFO",
    enable_cloudwatch: bool = True,
    enable_endpoints: bool = True,
    aws_region: str = "us-east-1",
    aws_profile: str = "ml-sandbox"
) -> MonitoringIntegration:
    """
    Create and configure monitoring integration.
    
    Args:
        log_level: Logging level
        enable_cloudwatch: Whether to enable CloudWatch
        enable_endpoints: Whether to enable HTTP endpoints
        aws_region: AWS region
        aws_profile: AWS profile name
        
    Returns:
        Configured MonitoringIntegration instance
    """
    return MonitoringIntegration(
        log_level=log_level,
        enable_cloudwatch=enable_cloudwatch,
        enable_endpoints=enable_endpoints,
        aws_region=aws_region,
        aws_profile=aws_profile
    )


# Convenience function for quick setup
async def setup_monitoring(
    agents: Dict[str, BaseAgent],
    log_level: str = "INFO",
    enable_cloudwatch: bool = True,
    enable_endpoints: bool = True
) -> MonitoringIntegration:
    """
    Quick setup function for monitoring integration.
    
    Args:
        agents: Dictionary of agents to monitor
        log_level: Logging level
        enable_cloudwatch: Whether to enable CloudWatch
        enable_endpoints: Whether to enable HTTP endpoints
        
    Returns:
        Initialized MonitoringIntegration instance
    """
    # Create monitoring integration
    monitoring = create_monitoring_integration(
        log_level=log_level,
        enable_cloudwatch=enable_cloudwatch,
        enable_endpoints=enable_endpoints
    )
    
    # Initialize
    await monitoring.initialize()
    
    # Register agents
    for name, agent in agents.items():
        monitoring.register_agent(name, agent)
    
    # Setup CloudWatch resources if enabled
    if enable_cloudwatch:
        await monitoring.setup_cloudwatch_resources()
    
    return monitoring