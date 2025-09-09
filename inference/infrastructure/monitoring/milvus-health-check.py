#!/usr/bin/env python3
"""
Health monitoring script for Milvus vector database on ECS ARM64 platform.
Uses ml-sandbox AWS profile and us-east-1 region.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
import boto3
import requests
from pymilvus import connections, utility
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MilvusHealthMonitor:
    """Health monitoring for Milvus vector database on ECS ARM64."""
    
    def __init__(
        self,
        milvus_host: str = "milvus.multilingual-inference.local",
        milvus_port: int = 19530,
        metrics_port: int = 9091,
        aws_profile: str = "ml-sandbox",
        aws_region: str = "us-east-1"
    ) -> None:
        """Initialize Milvus health monitor."""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.metrics_port = metrics_port
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=aws_profile)
        self.ecs_client = session.client('ecs', region_name=aws_region)
        self.cloudwatch = session.client('cloudwatch', region_name=aws_region)
        
        self.cluster_name = "multilingual-inference-cluster"
        self.service_name = "multilingual-inference-milvus"
        
    async def check_milvus_connection(self) -> Dict[str, Any]:
        """Check Milvus database connection and basic operations."""
        try:
            # Connect to Milvus
            connections.connect(
                alias="health_check",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10
            )
            
            # Check if connection is healthy
            is_healthy = utility.get_server_version(using="health_check")
            
            # Disconnect
            connections.disconnect("health_check")
            
            return {
                "status": "healthy",
                "version": is_healthy,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("Milvus connection failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def check_metrics_endpoint(self) -> Dict[str, Any]:
        """Check Milvus metrics endpoint availability."""
        try:
            response = requests.get(
                f"http://{self.milvus_host}:{self.metrics_port}/healthz",
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error("Metrics endpoint check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get ECS service status and task health."""
        try:
            response = self.ecs_client.describe_services(
                cluster=self.cluster_name,
                services=[self.service_name]
            )
            
            if not response['services']:
                return {
                    "status": "not_found",
                    "timestamp": time.time()
                }
            
            service = response['services'][0]
            
            return {
                "status": service['status'],
                "running_count": service['runningCount'],
                "desired_count": service['desiredCount'],
                "pending_count": service['pendingCount'],
                "platform_version": service.get('platformVersion', 'unknown'),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("ECS service status check failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def restart_service_if_unhealthy(self) -> bool:
        """Restart ECS service if health checks fail."""
        try:
            # Force new deployment to restart tasks
            response = self.ecs_client.update_service(
                cluster=self.cluster_name,
                service=self.service_name,
                forceNewDeployment=True
            )
            
            logger.info(
                "Service restart initiated",
                service=self.service_name,
                deployment_id=response['service']['deployments'][0]['id']
            )
            
            return True
            
        except Exception as e:
            logger.error("Service restart failed", error=str(e))
            return False
    
    async def publish_metrics(self, health_data: Dict[str, Any]) -> None:
        """Publish health metrics to CloudWatch."""
        try:
            metrics = []
            
            # Milvus connection health
            if 'milvus_connection' in health_data:
                connection_status = health_data['milvus_connection']
                metrics.append({
                    'MetricName': 'MilvusConnectionHealth',
                    'Value': 1 if connection_status['status'] == 'healthy' else 0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Service', 'Value': 'milvus'},
                        {'Name': 'Platform', 'Value': 'ARM64'}
                    ]
                })
            
            # Metrics endpoint health
            if 'metrics_endpoint' in health_data:
                metrics_status = health_data['metrics_endpoint']
                metrics.append({
                    'MetricName': 'MilvusMetricsEndpointHealth',
                    'Value': 1 if metrics_status['status'] == 'healthy' else 0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Service', 'Value': 'milvus'},
                        {'Name': 'Platform', 'Value': 'ARM64'}
                    ]
                })
                
                if 'response_time' in metrics_status:
                    metrics.append({
                        'MetricName': 'MilvusMetricsResponseTime',
                        'Value': metrics_status['response_time'],
                        'Unit': 'Seconds',
                        'Dimensions': [
                            {'Name': 'Service', 'Value': 'milvus'},
                            {'Name': 'Platform', 'Value': 'ARM64'}
                        ]
                    })
            
            # ECS service health
            if 'ecs_service' in health_data:
                service_status = health_data['ecs_service']
                if 'running_count' in service_status and 'desired_count' in service_status:
                    metrics.append({
                        'MetricName': 'MilvusServiceRunningTasks',
                        'Value': service_status['running_count'],
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Service', 'Value': 'milvus'},
                            {'Name': 'Platform', 'Value': 'ARM64'}
                        ]
                    })
                    
                    metrics.append({
                        'MetricName': 'MilvusServiceDesiredTasks',
                        'Value': service_status['desired_count'],
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Service', 'Value': 'milvus'},
                            {'Name': 'Platform', 'Value': 'ARM64'}
                        ]
                    })
            
            # Publish metrics to CloudWatch
            if metrics:
                self.cloudwatch.put_metric_data(
                    Namespace='MultilingualInference/Milvus',
                    MetricData=metrics
                )
                
                logger.info("Health metrics published to CloudWatch", metric_count=len(metrics))
            
        except Exception as e:
            logger.error("Failed to publish metrics", error=str(e))
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        logger.info("Starting Milvus health check")
        
        health_data = {
            "timestamp": time.time(),
            "milvus_connection": await self.check_milvus_connection(),
            "metrics_endpoint": await self.check_metrics_endpoint(),
            "ecs_service": await self.get_service_status()
        }
        
        # Determine overall health
        is_healthy = (
            health_data["milvus_connection"]["status"] == "healthy" and
            health_data["metrics_endpoint"]["status"] == "healthy" and
            health_data["ecs_service"]["status"] == "ACTIVE"
        )
        
        health_data["overall_status"] = "healthy" if is_healthy else "unhealthy"
        
        # Log health status
        logger.info(
            "Health check completed",
            overall_status=health_data["overall_status"],
            milvus_connection=health_data["milvus_connection"]["status"],
            metrics_endpoint=health_data["metrics_endpoint"]["status"],
            ecs_service=health_data["ecs_service"]["status"]
        )
        
        # Publish metrics
        await self.publish_metrics(health_data)
        
        # Restart service if unhealthy (with backoff)
        if not is_healthy:
            logger.warning("Milvus service is unhealthy, considering restart")
            # Add restart logic with proper backoff and limits
            
        return health_data


async def main() -> None:
    """Main health monitoring loop."""
    monitor = MilvusHealthMonitor()
    
    while True:
        try:
            health_data = await monitor.run_health_check()
            
            # Sleep for 60 seconds between checks
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Health monitoring stopped by user")
            break
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            await asyncio.sleep(30)  # Shorter sleep on error


if __name__ == "__main__":
    asyncio.run(main())