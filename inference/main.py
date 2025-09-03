"""
Main entry point for the multilingual product inference system.

This module provides the main application entry point and CLI interface
for running the inference system, following PEP 8 standards.
"""

import asyncio
import logging
import sys
from typing import Optional
import click

from .config.settings import get_config, setup_logging
from .models.data_models import ProductInput, LanguageHint
from .agents.orchestrator_agent import create_orchestrator_agent
from .monitoring.integration import create_monitoring_integration


logger = logging.getLogger(__name__)


async def initialize_system() -> None:
    """Initialize the inference system with configuration and logging."""
    try:
        # Load configuration
        config = get_config()
        
        # Setup logging
        setup_logging(config)
        
        logger.info("Multilingual Product Inference System starting...")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"AWS Profile: {config.aws.profile_name}")
        logger.info(f"AWS Region: {config.aws.region}")
        
        # Validate AWS configuration
        try:
            import boto3
            session = boto3.Session(profile_name=config.aws.profile_name)
            credentials = session.get_credentials()
            if credentials:
                logger.info("AWS credentials validated successfully")
            else:
                logger.warning("AWS credentials not found")
        except Exception as e:
            logger.warning(f"AWS validation failed: {str(e)}")
        
        logger.info("System initialization completed")
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise


async def health_check() -> dict:
    """
    Perform system health check.
    
    Returns:
        Dictionary with health status information
    """
    try:
        config = get_config()
        
        health_status = {
            "status": "healthy",
            "environment": config.environment.value,
            "aws_profile": config.aws.profile_name,
            "aws_region": config.aws.region,
            "components": {
                "config": "loaded",
                "logging": "configured"
            }
        }
        
        # Check AWS connectivity
        try:
            import boto3
            session = boto3.Session(profile_name=config.aws.profile_name)
            sts = session.client('sts', region_name=config.aws.region)
            identity = sts.get_caller_identity()
            health_status["components"]["aws"] = "connected"
            health_status["aws_account"] = identity.get("Account", "unknown")
        except Exception as e:
            health_status["components"]["aws"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def process_product_sample(product_name: str, language_hint: Optional[str] = None) -> dict:
    """
    Process a sample product name for testing.
    
    Args:
        product_name: Product name to process
        language_hint: Optional language hint
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Convert language hint
        lang_hint = LanguageHint.AUTO
        if language_hint:
            try:
                lang_hint = LanguageHint(language_hint)
            except ValueError:
                logger.warning(f"Invalid language hint '{language_hint}', using AUTO")
        
        # Create product input
        product_input = ProductInput(
            product_name=product_name,
            language_hint=lang_hint
        )
        
        # Create orchestrator agent for processing
        orchestrator = create_orchestrator_agent()
        
        try:
            await orchestrator.initialize()
            
            # Check if we have any agents registered
            if not orchestrator.agents:
                result = {
                    "input": {
                        "product_name": product_input.product_name,
                        "language_hint": product_input.language_hint.value
                    },
                    "status": "ready",
                    "message": "Orchestrator agent created but no agents were registered successfully",
                    "orchestrator_status": "initialized",
                    "registered_agents": 0,
                    "next_steps": [
                        "Check agent dependencies (spaCy, sentence-transformers, boto3)",
                        "Verify AWS credentials for LLM agent",
                        "Check Milvus database connectivity for RAG agent",
                        "Review agent configuration in settings"
                    ]
                }
            else:
                # Process with orchestrator if agents are available
                inference_result = await orchestrator.orchestrate_inference(product_input)
                result = {
                    "input": {
                        "product_name": product_input.product_name,
                        "language_hint": product_input.language_hint.value
                    },
                    "status": "completed",
                    "inference_result": {
                        "best_prediction": inference_result.best_prediction,
                        "best_confidence": inference_result.best_confidence,
                        "best_method": inference_result.best_method,
                        "processing_time": inference_result.total_processing_time
                    },
                    "orchestrator_status": "active",
                    "registered_agents": len(orchestrator.agents),
                    "available_agents": list(orchestrator.agents.keys())
                }
            
        finally:
            await orchestrator.cleanup()
        
        logger.info(f"Processed product: {product_name}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process product '{product_name}': {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """Multilingual Product Inference System CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
async def init() -> None:
    """Initialize the inference system."""
    try:
        await initialize_system()
        click.echo("✅ System initialized successfully")
    except Exception as e:
        click.echo(f"❌ Initialization failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
async def health() -> None:
    """Check system health status."""
    try:
        status = await health_check()
        
        if status["status"] == "healthy":
            click.echo("✅ System is healthy")
        elif status["status"] == "degraded":
            click.echo("⚠️  System is degraded")
        else:
            click.echo("❌ System is unhealthy")
        
        # Print detailed status
        click.echo(f"\nEnvironment: {status.get('environment', 'unknown')}")
        click.echo(f"AWS Profile: {status.get('aws_profile', 'unknown')}")
        click.echo(f"AWS Region: {status.get('aws_region', 'unknown')}")
        
        if "aws_account" in status:
            click.echo(f"AWS Account: {status['aws_account']}")
        
        click.echo("\nComponents:")
        for component, component_status in status.get("components", {}).items():
            click.echo(f"  {component}: {component_status}")
        
        if "error" in status:
            click.echo(f"\nError: {status['error']}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Health check failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('product_name')
@click.option('--language', '-l', help='Language hint (en, th, mixed, auto)')
async def process(product_name: str, language: Optional[str]) -> None:
    """Process a product name for brand inference."""
    try:
        await initialize_system()
        
        result = await process_product_sample(product_name, language)
        
        click.echo(f"\n📦 Processing: {product_name}")
        click.echo(f"Language hint: {result['input']['language_hint']}")
        click.echo(f"Status: {result['status']}")
        
        if result['status'] == 'processed':
            click.echo(f"Message: {result['message']}")
            click.echo("\nNext steps:")
            for step in result['next_steps']:
                click.echo(f"  • {step}")
        elif result['status'] == 'completed':
            click.echo(f"\n🎯 Prediction: {result['inference_result']['best_prediction']}")
            click.echo(f"Confidence: {result['inference_result']['best_confidence']:.3f}")
            click.echo(f"Method: {result['inference_result']['best_method']}")
            click.echo(f"Processing time: {result['inference_result']['processing_time']:.3f}s")
            click.echo(f"Available agents: {', '.join(result.get('available_agents', []))}")
        elif result['status'] == 'error':
            click.echo(f"Error: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Processing failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8080, help='Port to bind to')
@click.option('--dev', is_flag=True, help='Run in development mode')
@click.option('--enable-monitoring', is_flag=True, help='Enable monitoring endpoints')
async def serve(host: str, port: int, dev: bool, enable_monitoring: bool) -> None:
    """Start the inference API server."""
    try:
        await initialize_system()
        
        click.echo(f"🚀 Starting server on {host}:{port}")
        if dev:
            click.echo("🔧 Development mode enabled")
        if enable_monitoring:
            click.echo("📊 Monitoring endpoints enabled")
        
        # Initialize monitoring if enabled
        monitoring = None
        if enable_monitoring:
            try:
                monitoring = create_monitoring_integration(
                    log_level="DEBUG" if dev else "INFO",
                    enable_cloudwatch=not dev,  # Disable CloudWatch in dev mode
                    enable_endpoints=True
                )
                await monitoring.initialize()
                click.echo("✅ Monitoring system initialized")
            except Exception as e:
                click.echo(f"⚠️  Monitoring initialization failed: {str(e)}")
        
        # For now, just show that the server would start
        # In a real implementation, this would start an HTTP server
        click.echo("📡 Server ready (mock implementation)")
        click.echo("Available endpoints:")
        click.echo("  GET  /health - Health check")
        click.echo("  POST /infer - Product inference")
        
        if enable_monitoring:
            click.echo("  GET  /monitoring/health - System health")
            click.echo("  GET  /monitoring/diagnostics - System diagnostics")
            click.echo("  GET  /monitoring/metrics - Metrics buffer")
            click.echo("  POST /monitoring/cloudwatch/dashboard - Create dashboard")
        
        click.echo("\nPress Ctrl+C to stop")
        
        # Simulate server running
        try:
            # Start monitoring loop if enabled
            if monitoring:
                monitoring_task = asyncio.create_task(
                    monitoring.start_monitoring_loop(interval_seconds=30)
                )
            
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            click.echo("\n🛑 Stopping server...")
            
            # Cleanup monitoring
            if monitoring:
                await monitoring.cleanup()
                click.echo("✅ Monitoring cleanup completed")
            
            click.echo("🛑 Server stopped")
            
    except Exception as e:
        click.echo(f"❌ Server failed to start: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--agent', help='Specific agent to check (optional)')
async def monitor_health(agent: Optional[str]) -> None:
    """Check system or agent health status."""
    try:
        await initialize_system()
        
        # Initialize monitoring
        monitoring = create_monitoring_integration(
            enable_cloudwatch=False,  # Don't need CloudWatch for health check
            enable_endpoints=False
        )
        await monitoring.initialize()
        
        if agent:
            click.echo(f"🔍 Checking health for agent: {agent}")
            # Would check specific agent if registered
            click.echo(f"Agent '{agent}' health check not implemented yet")
        else:
            click.echo("🔍 Checking system health...")
            health = await monitoring.run_system_health_check()
            
            status = health.get("overall_status", "unknown")
            if status == "healthy":
                click.echo("✅ System is healthy")
            elif status == "degraded":
                click.echo("⚠️  System is degraded")
            else:
                click.echo("❌ System is unhealthy")
            
            # Show component status
            components = health.get("components", {})
            if components:
                click.echo("\nComponent Status:")
                for comp_name, comp_health in components.items():
                    status_icon = "✅" if comp_health.get("status") == "healthy" else "❌"
                    click.echo(f"  {status_icon} {comp_name}: {comp_health.get('status', 'unknown')}")
        
        await monitoring.cleanup()
        
    except Exception as e:
        click.echo(f"❌ Health check failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--full', is_flag=True, help='Run full diagnostics')
@click.option('--agent', help='Run diagnostics for specific agent')
async def diagnose(full: bool, agent: Optional[str]) -> None:
    """Run system diagnostics."""
    try:
        await initialize_system()
        
        # Initialize monitoring
        monitoring = create_monitoring_integration(
            enable_cloudwatch=False,
            enable_endpoints=False
        )
        await monitoring.initialize()
        
        if agent:
            click.echo(f"🔍 Running diagnostics for agent: {agent}")
            # Would run agent-specific diagnostics
            click.echo(f"Agent '{agent}' diagnostics not implemented yet")
        elif full:
            click.echo("🔍 Running full system diagnostics...")
            diagnostics = await monitoring.run_system_diagnostics()
            
            status = diagnostics.get("overall_status", "unknown")
            tests = diagnostics.get("tests", [])
            
            if status == "pass":
                click.echo("✅ All diagnostics passed")
            elif status == "warning":
                click.echo("⚠️  Some diagnostics have warnings")
            else:
                click.echo("❌ Some diagnostics failed")
            
            # Show test results
            if tests:
                click.echo(f"\nTest Results ({len(tests)} tests):")
                for test in tests:
                    test_name = test.get("test_name", "unknown")
                    test_status = test.get("status", "unknown")
                    test_message = test.get("message", "")
                    
                    if test_status == "pass":
                        icon = "✅"
                    elif test_status == "warning":
                        icon = "⚠️ "
                    elif test_status == "fail":
                        icon = "❌"
                    else:
                        icon = "❓"
                    
                    click.echo(f"  {icon} {test_name}: {test_message}")
        else:
            click.echo("🔍 Running basic diagnostics...")
            click.echo("Use --full for comprehensive diagnostics")
        
        await monitoring.cleanup()
        
    except Exception as e:
        click.echo(f"❌ Diagnostics failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--dashboard-name', default='multilingual-inference', help='Dashboard name')
async def setup_cloudwatch(dashboard_name: str) -> None:
    """Setup CloudWatch dashboard and alarms."""
    try:
        await initialize_system()
        
        click.echo("🔧 Setting up CloudWatch monitoring...")
        
        # Initialize monitoring with CloudWatch enabled
        monitoring = create_monitoring_integration(
            enable_cloudwatch=True,
            enable_endpoints=False
        )
        await monitoring.initialize()
        
        # Setup CloudWatch resources
        results = await monitoring.setup_cloudwatch_resources()
        
        if results.get("dashboard"):
            click.echo(f"✅ Created CloudWatch dashboard: {dashboard_name}")
        else:
            click.echo("❌ Failed to create CloudWatch dashboard")
        
        if results.get("alarms"):
            alarms_count = results.get("alarms_count", 0)
            click.echo(f"✅ Created {alarms_count} CloudWatch alarms")
        else:
            click.echo("❌ Failed to create CloudWatch alarms")
        
        if "error" in results:
            click.echo(f"⚠️  Setup completed with errors: {results['error']}")
        
        await monitoring.cleanup()
        
    except Exception as e:
        click.echo(f"❌ CloudWatch setup failed: {str(e)}", err=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the application."""
    # Handle async commands
    import inspect
    
    def async_command(f):
        """Decorator to handle async click commands."""
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to async commands
    for command_name in ['init', 'health', 'process', 'serve', 'monitor_health', 'diagnose', 'setup_cloudwatch']:
        command = cli.commands.get(command_name)
        if command and inspect.iscoroutinefunction(command.callback):
            command.callback = async_command(command.callback)
    
    cli()


if __name__ == '__main__':
    main()