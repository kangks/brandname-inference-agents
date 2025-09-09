"""
CloudWatch integration for system monitoring and alerting.

This module provides CloudWatch dashboards, metrics collection, and alerting
for the inference system following PEP 8 standards.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

from ..models.data_models import InferenceResult, AgentHealth


@dataclass
class MetricData:
    """CloudWatch metric data point."""
    
    metric_name: str
    value: float
    unit: str
    timestamp: float
    dimensions: Optional[Dict[str, str]] = None
    
    def to_cloudwatch_format(self) -> Dict[str, Any]:
        """Convert to CloudWatch metric format."""
        metric_data = {
            'MetricName': self.metric_name,
            'Value': self.value,
            'Unit': self.unit,
            'Timestamp': datetime.fromtimestamp(self.timestamp)
        }
        
        if self.dimensions:
            metric_data['Dimensions'] = [
                {'Name': key, 'Value': value}
                for key, value in self.dimensions.items()
            ]
        
        return metric_data


@dataclass
class AlarmConfiguration:
    """CloudWatch alarm configuration."""
    
    alarm_name: str
    metric_name: str
    namespace: str
    statistic: str
    threshold: float
    comparison_operator: str
    evaluation_periods: int
    period: int
    alarm_description: Optional[str] = None
    dimensions: Optional[Dict[str, str]] = None
    
    def to_cloudwatch_format(self) -> Dict[str, Any]:
        """Convert to CloudWatch alarm format."""
        alarm_config = {
            'AlarmName': self.alarm_name,
            'AlarmDescription': self.alarm_description or f'Alarm for {self.metric_name}',
            'MetricName': self.metric_name,
            'Namespace': self.namespace,
            'Statistic': self.statistic,
            'Threshold': self.threshold,
            'ComparisonOperator': self.comparison_operator,
            'EvaluationPeriods': self.evaluation_periods,
            'Period': self.period,
            'TreatMissingData': 'notBreaching'
        }
        
        if self.dimensions:
            alarm_config['Dimensions'] = [
                {'Name': key, 'Value': value}
                for key, value in self.dimensions.items()
            ]
        
        return alarm_config


class MetricsCollector:
    """
    Collects and formats metrics for CloudWatch publishing.
    
    Provides structured metric collection for inference operations,
    agent performance, and system health monitoring.
    """
    
    def __init__(self, namespace: str = "MultilingualInference") -> None:
        """
        Initialize metrics collector.
        
        Args:
            namespace: CloudWatch namespace for metrics
        """
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer: List[MetricData] = []
        self.max_buffer_size = 20  # CloudWatch limit
    
    def record_inference_metrics(
        self,
        result: InferenceResult,
        request_id: str
    ) -> None:
        """
        Record metrics from inference result.
        
        Args:
            result: Inference result to extract metrics from
            request_id: Request identifier for dimensions
        """
        timestamp = time.time()
        
        # Overall inference metrics
        self.add_metric(
            metric_name="InferenceLatency",
            value=result.total_processing_time * 1000,  # Convert to milliseconds
            unit="Milliseconds",
            timestamp=timestamp,
            dimensions={
                "BestMethod": result.best_method,
                "RequestId": request_id
            }
        )
        
        self.add_metric(
            metric_name="InferenceConfidence",
            value=result.best_confidence,
            unit="None",
            timestamp=timestamp,
            dimensions={
                "BestMethod": result.best_method,
                "RequestId": request_id
            }
        )
        
        # Method-specific metrics
        if result.ner_result:
            self.add_metric(
                metric_name="NERLatency",
                value=result.ner_result.processing_time * 1000,
                unit="Milliseconds",
                timestamp=timestamp,
                dimensions={"Method": "NER"}
            )
            
            self.add_metric(
                metric_name="NERConfidence",
                value=result.ner_result.confidence,
                unit="None",
                timestamp=timestamp,
                dimensions={"Method": "NER"}
            )
        
        if result.rag_result:
            self.add_metric(
                metric_name="RAGLatency",
                value=result.rag_result.processing_time * 1000,
                unit="Milliseconds",
                timestamp=timestamp,
                dimensions={"Method": "RAG"}
            )
            
            self.add_metric(
                metric_name="RAGConfidence",
                value=result.rag_result.confidence,
                unit="None",
                timestamp=timestamp,
                dimensions={"Method": "RAG"}
            )
        
        if result.llm_result:
            self.add_metric(
                metric_name="LLMLatency",
                value=result.llm_result.processing_time * 1000,
                unit="Milliseconds",
                timestamp=timestamp,
                dimensions={"Method": "LLM"}
            )
            
            self.add_metric(
                metric_name="LLMConfidence",
                value=result.llm_result.confidence,
                unit="None",
                timestamp=timestamp,
                dimensions={"Method": "LLM"}
            )
        
        if result.hybrid_result:
            self.add_metric(
                metric_name="HybridLatency",
                value=result.hybrid_result.processing_time * 1000,
                unit="Milliseconds",
                timestamp=timestamp,
                dimensions={"Method": "Hybrid"}
            )
            
            self.add_metric(
                metric_name="HybridConfidence",
                value=result.hybrid_result.confidence,
                unit="None",
                timestamp=timestamp,
                dimensions={"Method": "Hybrid"}
            )
    
    def record_agent_health_metrics(
        self,
        health_results: Dict[str, AgentHealth]
    ) -> None:
        """
        Record agent health metrics.
        
        Args:
            health_results: Dictionary of agent health results
        """
        timestamp = time.time()
        
        for agent_name, health in health_results.items():
            # Health status (1 for healthy, 0 for unhealthy)
            self.add_metric(
                metric_name="AgentHealth",
                value=1.0 if health.is_healthy else 0.0,
                unit="None",
                timestamp=timestamp,
                dimensions={"AgentName": agent_name}
            )
            
            # Response time if available
            if health.response_time is not None:
                self.add_metric(
                    metric_name="AgentHealthCheckLatency",
                    value=health.response_time * 1000,
                    unit="Milliseconds",
                    timestamp=timestamp,
                    dimensions={"AgentName": agent_name}
                )
    
    def record_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        component: str = "System"
    ) -> None:
        """
        Record system resource metrics.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
            component: Component name for dimensions
        """
        timestamp = time.time()
        
        self.add_metric(
            metric_name="CPUUtilization",
            value=cpu_percent,
            unit="Percent",
            timestamp=timestamp,
            dimensions={"Component": component}
        )
        
        self.add_metric(
            metric_name="MemoryUtilization",
            value=memory_percent,
            unit="Percent",
            timestamp=timestamp,
            dimensions={"Component": component}
        )
        
        self.add_metric(
            metric_name="DiskUtilization",
            value=disk_percent,
            unit="Percent",
            timestamp=timestamp,
            dimensions={"Component": component}
        )
    
    def record_error_metrics(
        self,
        error_type: str,
        agent_name: Optional[str] = None,
        error_count: int = 1
    ) -> None:
        """
        Record error metrics.
        
        Args:
            error_type: Type of error
            agent_name: Optional agent name
            error_count: Number of errors (default: 1)
        """
        timestamp = time.time()
        dimensions = {"ErrorType": error_type}
        
        if agent_name:
            dimensions["AgentName"] = agent_name
        
        self.add_metric(
            metric_name="ErrorCount",
            value=float(error_count),
            unit="Count",
            timestamp=timestamp,
            dimensions=dimensions
        )
    
    def add_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        timestamp: Optional[float] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a metric to the collection buffer.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Metric unit
            timestamp: Optional timestamp (defaults to current time)
            dimensions: Optional dimensions dictionary
        """
        if timestamp is None:
            timestamp = time.time()
        
        metric = MetricData(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            dimensions=dimensions
        )
        
        self.metrics_buffer.append(metric)
        
        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.max_buffer_size:
            asyncio.create_task(self.flush_metrics())
    
    async def flush_metrics(self) -> bool:
        """
        Flush metrics buffer (placeholder for CloudWatch publishing).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.metrics_buffer:
            return True
        
        try:
            # Log metrics for now (CloudWatch publishing would happen here)
            self.logger.info(f"Flushing {len(self.metrics_buffer)} metrics to CloudWatch")
            
            for metric in self.metrics_buffer:
                self.logger.debug(f"Metric: {metric.metric_name} = {metric.value} {metric.unit}")
            
            # Clear buffer
            self.metrics_buffer.clear()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {str(e)}")
            return False
    
    def get_buffered_metrics(self) -> List[Dict[str, Any]]:
        """
        Get buffered metrics in CloudWatch format.
        
        Returns:
            List of metrics in CloudWatch format
        """
        return [metric.to_cloudwatch_format() for metric in self.metrics_buffer]


class CloudWatchMonitor:
    """
    CloudWatch integration for monitoring and alerting.
    
    Provides CloudWatch dashboard creation, metric publishing,
    and alarm management for the inference system.
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        profile_name: str = "ml-sandbox"
    ) -> None:
        """
        Initialize CloudWatch monitor.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name
        """
        self.region_name = region_name
        self.profile_name = profile_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        self.cloudwatch_client = None
        self.logs_client = None
        
        if BOTO3_AVAILABLE:
            try:
                session = boto3.Session(profile_name=profile_name)
                self.cloudwatch_client = session.client('cloudwatch', region_name=region_name)
                self.logs_client = session.client('logs', region_name=region_name)
                self.logger.info(f"CloudWatch monitor initialized for region {region_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CloudWatch clients: {str(e)}")
        else:
            self.logger.warning("boto3 not available - CloudWatch integration disabled")
    
    async def publish_metrics(self, metrics: List[MetricData], namespace: str) -> bool:
        """
        Publish metrics to CloudWatch.
        
        Args:
            metrics: List of metrics to publish
            namespace: CloudWatch namespace
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cloudwatch_client:
            self.logger.warning("CloudWatch client not available")
            return False
        
        if not metrics:
            return True
        
        try:
            # Convert metrics to CloudWatch format
            metric_data = [metric.to_cloudwatch_format() for metric in metrics]
            
            # Publish in batches (CloudWatch limit is 20 metrics per request)
            batch_size = 20
            
            for i in range(0, len(metric_data), batch_size):
                batch = metric_data[i:i + batch_size]
                
                response = self.cloudwatch_client.put_metric_data(
                    Namespace=namespace,
                    MetricData=batch
                )
                
                self.logger.debug(f"Published {len(batch)} metrics to CloudWatch")
            
            self.logger.info(f"Successfully published {len(metrics)} metrics to CloudWatch")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish metrics to CloudWatch: {str(e)}")
            return False
    
    async def create_dashboard(self, dashboard_name: str) -> bool:
        """
        Create CloudWatch dashboard for inference monitoring.
        
        Args:
            dashboard_name: Name of the dashboard to create
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cloudwatch_client:
            self.logger.warning("CloudWatch client not available")
            return False
        
        try:
            dashboard_body = self._generate_dashboard_config()
            
            response = self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            self.logger.info(f"Created CloudWatch dashboard: {dashboard_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create CloudWatch dashboard: {str(e)}")
            return False
    
    async def create_alarms(self, alarms: List[AlarmConfiguration]) -> bool:
        """
        Create CloudWatch alarms.
        
        Args:
            alarms: List of alarm configurations
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cloudwatch_client:
            self.logger.warning("CloudWatch client not available")
            return False
        
        try:
            for alarm in alarms:
                alarm_config = alarm.to_cloudwatch_format()
                
                response = self.cloudwatch_client.put_metric_alarm(**alarm_config)
                
                self.logger.info(f"Created CloudWatch alarm: {alarm.alarm_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create CloudWatch alarms: {str(e)}")
            return False
    
    def _generate_dashboard_config(self) -> Dict[str, Any]:
        """
        Generate CloudWatch dashboard configuration.
        
        Returns:
            Dashboard configuration dictionary
        """
        return {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["MultilingualInference", "InferenceLatency", "BestMethod", "ner"],
                            [".", ".", ".", "rag"],
                            [".", ".", ".", "llm"],
                            [".", ".", ".", "hybrid"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region_name,
                        "title": "Inference Latency by Method",
                        "yAxis": {
                            "left": {
                                "min": 0
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["MultilingualInference", "InferenceConfidence", "BestMethod", "ner"],
                            [".", ".", ".", "rag"],
                            [".", ".", ".", "llm"],
                            [".", ".", ".", "hybrid"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region_name,
                        "title": "Inference Confidence by Method",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": 1
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["MultilingualInference", "AgentHealth", "AgentName", "ner"],
                            [".", ".", ".", "rag"],
                            [".", ".", ".", "llm"],
                            [".", ".", ".", "hybrid"],
                            [".", ".", ".", "orchestrator"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region_name,
                        "title": "Agent Health Status",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": 1
                            }
                        }
                    }
                },
                {
                    "type": "metric",
                    "x": 8,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["MultilingualInference", "ErrorCount", "ErrorType", "timeout"],
                            [".", ".", ".", "initialization"],
                            [".", ".", ".", "processing"],
                            [".", ".", ".", "circuit_breaker"]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region_name,
                        "title": "Error Count by Type"
                    }
                },
                {
                    "type": "metric",
                    "x": 16,
                    "y": 6,
                    "width": 8,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["MultilingualInference", "CPUUtilization", "Component", "System"],
                            [".", "MemoryUtilization", ".", "."],
                            [".", "DiskUtilization", ".", "."]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region_name,
                        "title": "System Resource Utilization",
                        "yAxis": {
                            "left": {
                                "min": 0,
                                "max": 100
                            }
                        }
                    }
                }
            ]
        }
    
    def get_default_alarms(self) -> List[AlarmConfiguration]:
        """
        Get default alarm configurations for the inference system.
        
        Returns:
            List of default alarm configurations
        """
        return [
            AlarmConfiguration(
                alarm_name="HighInferenceLatency",
                metric_name="InferenceLatency",
                namespace="MultilingualInference",
                statistic="Average",
                threshold=5000.0,  # 5 seconds
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=2,
                period=300,
                alarm_description="Alert when inference latency is high"
            ),
            AlarmConfiguration(
                alarm_name="LowInferenceConfidence",
                metric_name="InferenceConfidence",
                namespace="MultilingualInference",
                statistic="Average",
                threshold=0.5,
                comparison_operator="LessThanThreshold",
                evaluation_periods=3,
                period=300,
                alarm_description="Alert when inference confidence is consistently low"
            ),
            AlarmConfiguration(
                alarm_name="AgentUnhealthy",
                metric_name="AgentHealth",
                namespace="MultilingualInference",
                statistic="Average",
                threshold=0.5,
                comparison_operator="LessThanThreshold",
                evaluation_periods=2,
                period=300,
                alarm_description="Alert when agents are unhealthy"
            ),
            AlarmConfiguration(
                alarm_name="HighErrorRate",
                metric_name="ErrorCount",
                namespace="MultilingualInference",
                statistic="Sum",
                threshold=10.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=2,
                period=300,
                alarm_description="Alert when error rate is high"
            ),
            AlarmConfiguration(
                alarm_name="HighCPUUtilization",
                metric_name="CPUUtilization",
                namespace="MultilingualInference",
                statistic="Average",
                threshold=80.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=3,
                period=300,
                alarm_description="Alert when CPU utilization is high"
            ),
            AlarmConfiguration(
                alarm_name="HighMemoryUtilization",
                metric_name="MemoryUtilization",
                namespace="MultilingualInference",
                statistic="Average",
                threshold=85.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=3,
                period=300,
                alarm_description="Alert when memory utilization is high"
            )
        ]


# Factory functions
def create_metrics_collector(namespace: str = "MultilingualInference") -> MetricsCollector:
    """
    Create a metrics collector instance.
    
    Args:
        namespace: CloudWatch namespace
        
    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(namespace)


def create_cloudwatch_monitor(
    region_name: str = "us-east-1",
    profile_name: str = "ml-sandbox"
) -> CloudWatchMonitor:
    """
    Create a CloudWatch monitor instance.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name
        
    Returns:
        CloudWatchMonitor instance
    """
    return CloudWatchMonitor(region_name, profile_name)