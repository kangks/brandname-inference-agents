"""
Diagnostic endpoints and troubleshooting tools for inference issues.

This module provides comprehensive diagnostics for debugging inference problems,
performance analysis, and system troubleshooting following PEP 8 standards.
"""

import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import psutil
import gc

from ..models.data_models import (
    ProductInput,
    InferenceResult,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    LanguageHint
)
from ..agents.base_agent import BaseAgent


@dataclass
class DiagnosticTest:
    """Individual diagnostic test result."""
    
    test_name: str
    status: str  # "pass", "fail", "warning", "skip"
    duration_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SystemDiagnostics:
    """Complete system diagnostic results."""
    
    timestamp: float
    overall_status: str
    tests: List[DiagnosticTest]
    system_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "tests": [test.to_dict() for test in self.tests],
            "system_info": self.system_info,
            "performance_metrics": self.performance_metrics
        }


class DiagnosticsCollector:
    """
    Comprehensive diagnostics collector for inference system troubleshooting.
    
    Provides detailed diagnostic information for debugging inference issues,
    performance problems, and system configuration issues.
    """
    
    def __init__(self) -> None:
        """Initialize diagnostics collector."""
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self.test_samples = [
            "iPhone 13 Pro Max",
            "Samsung Galaxy S21",
            "MacBook Pro M1",
            "Nike Air Jordan",
            "Coca Cola 330ml",
            "โทรศัพท์ Samsung Galaxy",  # Thai text
            "iPhone สีดำ 128GB",  # Mixed Thai-English
        ]
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent for diagnostics.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.agents[name] = agent
        self.logger.info(f"Registered agent '{name}' for diagnostics")
    
    async def run_full_diagnostics(self) -> SystemDiagnostics:
        """
        Run comprehensive system diagnostics.
        
        Returns:
            SystemDiagnostics with complete diagnostic results
        """
        start_time = time.time()
        tests = []
        
        self.logger.info("Starting full system diagnostics")
        
        # System information tests
        tests.extend(await self._run_system_info_tests())
        
        # Configuration tests
        tests.extend(await self._run_configuration_tests())
        
        # Agent connectivity tests
        tests.extend(await self._run_agent_tests())
        
        # External service tests
        tests.extend(await self._run_external_service_tests())
        
        # Performance tests
        tests.extend(await self._run_performance_tests())
        
        # Memory and resource tests
        tests.extend(await self._run_resource_tests())
        
        # Inference accuracy tests
        tests.extend(await self._run_inference_tests())
        
        # Determine overall status
        overall_status = self._calculate_overall_status(tests)
        
        # Collect system information
        system_info = await self._collect_system_info()
        
        # Collect performance metrics
        performance_metrics = await self._collect_performance_metrics()
        
        diagnostics = SystemDiagnostics(
            timestamp=start_time,
            overall_status=overall_status,
            tests=tests,
            system_info=system_info,
            performance_metrics=performance_metrics
        )
        
        self.logger.info(
            f"Diagnostics completed in {time.time() - start_time:.2f}s: {overall_status}"
        )
        
        return diagnostics
    
    async def run_agent_diagnostics(self, agent_name: str) -> Dict[str, Any]:
        """
        Run diagnostics for a specific agent.
        
        Args:
            agent_name: Name of the agent to diagnose
            
        Returns:
            Dictionary with agent-specific diagnostic results
        """
        if agent_name not in self.agents:
            return {
                "error": f"Agent '{agent_name}' not registered for diagnostics"
            }
        
        agent = self.agents[agent_name]
        tests = []
        
        # Agent health test
        tests.append(await self._test_agent_health(agent_name, agent))
        
        # Agent initialization test
        tests.append(await self._test_agent_initialization(agent_name, agent))
        
        # Agent processing test
        tests.append(await self._test_agent_processing(agent_name, agent))
        
        # Agent performance test
        tests.append(await self._test_agent_performance(agent_name, agent))
        
        return {
            "agent_name": agent_name,
            "tests": [test.to_dict() for test in tests],
            "overall_status": self._calculate_overall_status(tests)
        }
    
    async def run_inference_diagnostics(
        self,
        product_name: str,
        language_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run diagnostics for a specific inference request.
        
        Args:
            product_name: Product name to test
            language_hint: Optional language hint
            
        Returns:
            Dictionary with inference-specific diagnostic results
        """
        start_time = time.time()
        
        try:
            # Create product input
            lang_hint = LanguageHint.AUTO
            if language_hint:
                try:
                    lang_hint = LanguageHint(language_hint)
                except ValueError:
                    pass
            
            product_input = ProductInput(
                product_name=product_name,
                language_hint=lang_hint
            )
            
            # Test each agent individually
            agent_results = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    agent_start = time.time()
                    result = await agent.process(product_input)
                    agent_duration = time.time() - agent_start
                    
                    agent_results[agent_name] = {
                        "status": "success",
                        "duration_ms": agent_duration * 1000,
                        "result": result
                    }
                    
                except Exception as e:
                    agent_results[agent_name] = {
                        "status": "error",
                        "duration_ms": (time.time() - agent_start) * 1000,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
            
            # Analyze results
            analysis = self._analyze_inference_results(agent_results)
            
            return {
                "input": {
                    "product_name": product_name,
                    "language_hint": lang_hint.value
                },
                "total_duration_ms": (time.time() - start_time) * 1000,
                "agent_results": agent_results,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "input": {
                    "product_name": product_name,
                    "language_hint": language_hint
                },
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _run_system_info_tests(self) -> List[DiagnosticTest]:
        """Run system information tests."""
        tests = []
        
        # Python version test
        start_time = time.time()
        try:
            import sys
            python_version = sys.version
            
            if sys.version_info >= (3, 13):
                status = "pass"
                message = f"Python version: {python_version}"
            else:
                status = "warning"
                message = f"Python version {python_version} (recommended: 3.13+)"
            
            tests.append(DiagnosticTest(
                test_name="python_version",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={"version": python_version}
            ))
            
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="python_version",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Failed to check Python version",
                error=str(e)
            ))
        
        # Virtual environment test
        start_time = time.time()
        try:
            import sys
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            if in_venv:
                status = "pass"
                message = "Running in virtual environment"
            else:
                status = "warning"
                message = "Not running in virtual environment (recommended: .venv)"
            
            tests.append(DiagnosticTest(
                test_name="virtual_environment",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={"in_venv": in_venv}
            ))
            
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="virtual_environment",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Failed to check virtual environment",
                error=str(e)
            ))
        
        return tests
    
    async def _run_configuration_tests(self) -> List[DiagnosticTest]:
        """Run configuration tests."""
        tests = []
        
        # AWS configuration test
        start_time = time.time()
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ProfileNotFound
            
            # Use default credentials (IAM role) instead of hardcoded profile
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials:
                status = "pass"
                message = "AWS credentials configured (IAM role or profile)"
                details = {
                    "credentials_source": "IAM role or default profile",
                    "region": session.region_name or "us-east-1"
                }
            else:
                status = "fail"
                message = "AWS credentials not found"
                details = None
            
            tests.append(DiagnosticTest(
                test_name="aws_configuration",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details
            ))
            
        except (NoCredentialsError, ProfileNotFound) as e:
            tests.append(DiagnosticTest(
                test_name="aws_configuration",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="AWS configuration error",
                error=str(e)
            ))
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="aws_configuration",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Failed to check AWS configuration",
                error=str(e)
            ))
        
        return tests
    
    async def _run_agent_tests(self) -> List[DiagnosticTest]:
        """Run agent connectivity tests."""
        tests = []
        
        for agent_name, agent in self.agents.items():
            tests.append(await self._test_agent_health(agent_name, agent))
        
        return tests
    
    async def _test_agent_health(self, agent_name: str, agent: BaseAgent) -> DiagnosticTest:
        """Test individual agent health."""
        start_time = time.time()
        
        try:
            if hasattr(agent, 'health_check'):
                health_result = await agent.health_check()
                
                if health_result.is_healthy:
                    status = "pass"
                    message = f"Agent {agent_name} is healthy"
                    details = {
                        "response_time": health_result.response_time,
                        "last_check": health_result.last_check
                    }
                else:
                    status = "fail"
                    message = f"Agent {agent_name} is unhealthy: {health_result.error_message}"
                    details = {"error_message": health_result.error_message}
            else:
                status = "warning"
                message = f"Agent {agent_name} has no health check method"
                details = None
            
            return DiagnosticTest(
                test_name=f"agent_health_{agent_name}",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details
            )
            
        except Exception as e:
            return DiagnosticTest(
                test_name=f"agent_health_{agent_name}",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed for {agent_name}",
                error=str(e)
            )
    
    async def _test_agent_initialization(self, agent_name: str, agent: BaseAgent) -> DiagnosticTest:
        """Test agent initialization."""
        start_time = time.time()
        
        try:
            if hasattr(agent, 'initialize'):
                await agent.initialize()
                status = "pass"
                message = f"Agent {agent_name} initialized successfully"
            else:
                status = "skip"
                message = f"Agent {agent_name} has no initialization method"
            
            return DiagnosticTest(
                test_name=f"agent_init_{agent_name}",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message
            )
            
        except Exception as e:
            return DiagnosticTest(
                test_name=f"agent_init_{agent_name}",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Initialization failed for {agent_name}",
                error=str(e)
            )
    
    async def _test_agent_processing(self, agent_name: str, agent: BaseAgent) -> DiagnosticTest:
        """Test agent processing capability."""
        start_time = time.time()
        
        try:
            # Use a simple test input
            test_input = ProductInput(
                product_name="iPhone 13",
                language_hint=LanguageHint.ENGLISH
            )
            
            result = await agent.process(test_input)
            
            if result:
                status = "pass"
                message = f"Agent {agent_name} processed test input successfully"
                details = {"result_type": type(result).__name__}
            else:
                status = "warning"
                message = f"Agent {agent_name} returned empty result"
                details = None
            
            return DiagnosticTest(
                test_name=f"agent_processing_{agent_name}",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details
            )
            
        except Exception as e:
            return DiagnosticTest(
                test_name=f"agent_processing_{agent_name}",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Processing test failed for {agent_name}",
                error=str(e)
            )
    
    async def _test_agent_performance(self, agent_name: str, agent: BaseAgent) -> DiagnosticTest:
        """Test agent performance."""
        start_time = time.time()
        
        try:
            # Run multiple test inputs to measure performance
            test_inputs = [
                ProductInput(product_name=sample, language_hint=LanguageHint.AUTO)
                for sample in self.test_samples[:3]  # Use first 3 samples
            ]
            
            durations = []
            
            for test_input in test_inputs:
                input_start = time.time()
                await agent.process(test_input)
                durations.append(time.time() - input_start)
            
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            # Performance thresholds (configurable)
            if avg_duration < 1.0:
                status = "pass"
                message = f"Agent {agent_name} performance is good (avg: {avg_duration:.3f}s)"
            elif avg_duration < 5.0:
                status = "warning"
                message = f"Agent {agent_name} performance is acceptable (avg: {avg_duration:.3f}s)"
            else:
                status = "fail"
                message = f"Agent {agent_name} performance is poor (avg: {avg_duration:.3f}s)"
            
            return DiagnosticTest(
                test_name=f"agent_performance_{agent_name}",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "avg_duration_s": avg_duration,
                    "max_duration_s": max_duration,
                    "min_duration_s": min_duration,
                    "samples_tested": len(test_inputs)
                }
            )
            
        except Exception as e:
            return DiagnosticTest(
                test_name=f"agent_performance_{agent_name}",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Performance test failed for {agent_name}",
                error=str(e)
            )
    
    async def _run_external_service_tests(self) -> List[DiagnosticTest]:
        """Run external service tests."""
        tests = []
        
        # AWS Bedrock test
        tests.append(await self._test_aws_bedrock())
        
        # Milvus test (if configured)
        tests.append(await self._test_milvus())
        
        return tests
    
    async def _test_aws_bedrock(self) -> DiagnosticTest:
        """Test AWS Bedrock connectivity."""
        start_time = time.time()
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            client = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            # Try to list models
            response = client.list_foundation_models()
            
            status = "pass"
            message = "AWS Bedrock is accessible"
            details = {
                "models_available": len(response.get('modelSummaries', [])),
                "region": "us-east-1"
            }
            
            return DiagnosticTest(
                test_name="aws_bedrock",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details
            )
            
        except NoCredentialsError:
            return DiagnosticTest(
                test_name="aws_bedrock",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="AWS Bedrock: No credentials configured",
                error="NoCredentialsError"
            )
        except ClientError as e:
            return DiagnosticTest(
                test_name="aws_bedrock",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="AWS Bedrock: Client error",
                error=str(e)
            )
        except Exception as e:
            return DiagnosticTest(
                test_name="aws_bedrock",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="AWS Bedrock: Connection failed",
                error=str(e)
            )
    
    async def _test_milvus(self) -> DiagnosticTest:
        """Test Milvus connectivity."""
        start_time = time.time()
        
        try:
            from pymilvus import connections, utility
            
            # Try to connect to Milvus
            connections.connect(host="localhost", port=19530)
            
            # Check server version
            version = utility.get_server_version()
            
            status = "pass"
            message = f"Milvus is accessible (version: {version})"
            details = {
                "version": version,
                "host": "localhost",
                "port": 19530
            }
            
            return DiagnosticTest(
                test_name="milvus",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details=details
            )
            
        except ImportError:
            return DiagnosticTest(
                test_name="milvus",
                status="skip",
                duration_ms=(time.time() - start_time) * 1000,
                message="Milvus client not installed",
                error="ImportError: pymilvus not available"
            )
        except Exception as e:
            return DiagnosticTest(
                test_name="milvus",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Milvus connection failed",
                error=str(e)
            )
    
    async def _run_performance_tests(self) -> List[DiagnosticTest]:
        """Run system performance tests."""
        tests = []
        
        # Memory usage test
        start_time = time.time()
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 70:
                status = "pass"
                message = f"Memory usage is normal ({memory_percent:.1f}%)"
            elif memory_percent < 85:
                status = "warning"
                message = f"Memory usage is high ({memory_percent:.1f}%)"
            else:
                status = "fail"
                message = f"Memory usage is critical ({memory_percent:.1f}%)"
            
            tests.append(DiagnosticTest(
                test_name="memory_usage",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "percent": memory_percent,
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3)
                }
            ))
            
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="memory_usage",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Failed to check memory usage",
                error=str(e)
            ))
        
        return tests
    
    async def _run_resource_tests(self) -> List[DiagnosticTest]:
        """Run resource availability tests."""
        tests = []
        
        # Disk space test
        start_time = time.time()
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent < 80:
                status = "pass"
                message = f"Disk space is sufficient ({disk_percent:.1f}% used)"
            elif disk_percent < 90:
                status = "warning"
                message = f"Disk space is low ({disk_percent:.1f}% used)"
            else:
                status = "fail"
                message = f"Disk space is critical ({disk_percent:.1f}% used)"
            
            tests.append(DiagnosticTest(
                test_name="disk_space",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "percent": disk_percent,
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3)
                }
            ))
            
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="disk_space",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Failed to check disk space",
                error=str(e)
            ))
        
        return tests
    
    async def _run_inference_tests(self) -> List[DiagnosticTest]:
        """Run inference accuracy tests."""
        tests = []
        
        if not self.agents:
            tests.append(DiagnosticTest(
                test_name="inference_accuracy",
                status="skip",
                duration_ms=0,
                message="No agents registered for testing"
            ))
            return tests
        
        # Test with sample inputs
        start_time = time.time()
        try:
            successful_inferences = 0
            total_inferences = 0
            
            for sample in self.test_samples[:3]:  # Test first 3 samples
                test_input = ProductInput(
                    product_name=sample,
                    language_hint=LanguageHint.AUTO
                )
                
                for agent_name, agent in self.agents.items():
                    try:
                        result = await agent.process(test_input)
                        if result:
                            successful_inferences += 1
                        total_inferences += 1
                    except Exception:
                        total_inferences += 1
            
            if total_inferences == 0:
                status = "skip"
                message = "No inference tests performed"
            else:
                success_rate = successful_inferences / total_inferences
                
                if success_rate >= 0.8:
                    status = "pass"
                    message = f"Inference success rate is good ({success_rate:.1%})"
                elif success_rate >= 0.5:
                    status = "warning"
                    message = f"Inference success rate is acceptable ({success_rate:.1%})"
                else:
                    status = "fail"
                    message = f"Inference success rate is poor ({success_rate:.1%})"
            
            tests.append(DiagnosticTest(
                test_name="inference_accuracy",
                status=status,
                duration_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "successful_inferences": successful_inferences,
                    "total_inferences": total_inferences,
                    "success_rate": successful_inferences / total_inferences if total_inferences > 0 else 0
                }
            ))
            
        except Exception as e:
            tests.append(DiagnosticTest(
                test_name="inference_accuracy",
                status="fail",
                duration_ms=(time.time() - start_time) * 1000,
                message="Inference accuracy test failed",
                error=str(e)
            ))
        
        return tests
    
    def _calculate_overall_status(self, tests: List[DiagnosticTest]) -> str:
        """Calculate overall diagnostic status."""
        if not tests:
            return "unknown"
        
        fail_count = sum(1 for test in tests if test.status == "fail")
        warning_count = sum(1 for test in tests if test.status == "warning")
        pass_count = sum(1 for test in tests if test.status == "pass")
        
        if fail_count > 0:
            return "fail"
        elif warning_count > pass_count:
            return "warning"
        elif pass_count > 0:
            return "pass"
        else:
            return "unknown"
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        try:
            import platform
            import sys
            
            return {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "prefix": sys.prefix
                },
                "environment": {
                    "virtual_env": hasattr(sys, 'real_prefix') or (
                        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
                    )
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_mb": memory.total / (1024**2),
                    "available_mb": memory.available / (1024**2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": (disk.used / disk.total) * 100
                },
                "process": {
                    "memory_mb": process_memory.rss / (1024**2),
                    "cpu_percent": process.cpu_percent()
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_inference_results(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze inference results for patterns and issues."""
        analysis = {
            "successful_agents": [],
            "failed_agents": [],
            "performance_summary": {},
            "recommendations": []
        }
        
        total_duration = 0
        successful_count = 0
        
        for agent_name, result in agent_results.items():
            if result["status"] == "success":
                analysis["successful_agents"].append(agent_name)
                successful_count += 1
                total_duration += result["duration_ms"]
            else:
                analysis["failed_agents"].append({
                    "agent": agent_name,
                    "error": result.get("error", "Unknown error")
                })
        
        if successful_count > 0:
            analysis["performance_summary"]["avg_duration_ms"] = total_duration / successful_count
            analysis["performance_summary"]["success_rate"] = successful_count / len(agent_results)
        
        # Generate recommendations
        if len(analysis["failed_agents"]) > 0:
            analysis["recommendations"].append("Some agents are failing - check logs for details")
        
        if analysis["performance_summary"].get("avg_duration_ms", 0) > 5000:
            analysis["recommendations"].append("Average response time is high - consider optimization")
        
        return analysis