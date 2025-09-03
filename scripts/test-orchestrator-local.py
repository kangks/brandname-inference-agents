#!/usr/bin/env python3
"""
Local test script for orchestrator with default agents.

This script tests the orchestrator agent with automatically registered default agents
in a local environment to verify functionality before deployment.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.config.settings import get_config, setup_logging
from inference.models.data_models import ProductInput, LanguageHint
from inference.agents.orchestrator_agent import create_orchestrator_agent


async def test_orchestrator_initialization():
    """Test orchestrator initialization with default agents."""
    print("🔧 Testing orchestrator initialization...")
    
    try:
        # Create and initialize orchestrator
        orchestrator = create_orchestrator_agent()
        await orchestrator.initialize()
        
        print(f"✅ Orchestrator initialized successfully")
        print(f"📊 Registered agents: {len(orchestrator.agents)}")
        
        if orchestrator.agents:
            print(f"🤖 Available agents: {list(orchestrator.agents.keys())}")
        else:
            print("⚠️  No agents were registered")
        
        return orchestrator
        
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {str(e)}")
        return None


async def test_agent_health(orchestrator):
    """Test health of registered agents."""
    print("\n🏥 Testing agent health...")
    
    if not orchestrator or not orchestrator.agents:
        print("⚠️  No agents to test")
        return False
    
    try:
        health_results = await orchestrator.get_agent_health()
        
        all_healthy = True
        for agent_name, health in health_results.items():
            status = "✅ Healthy" if health.is_healthy else "❌ Unhealthy"
            print(f"  {agent_name}: {status}")
            
            if not health.is_healthy:
                all_healthy = False
                if health.error_message:
                    print(f"    Error: {health.error_message}")
        
        return all_healthy
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


async def test_inference_samples(orchestrator):
    """Test inference with sample product names."""
    print("\n🧪 Testing inference with sample data...")
    
    if not orchestrator:
        print("⚠️  No orchestrator available")
        return False
    
    # Test cases
    test_cases = [
        ("Samsung Galaxy S23", LanguageHint.EN),
        ("iPhone 15 Pro Max", LanguageHint.EN),
        ("Sony WH-1000XM4 หูฟัง", LanguageHint.MIXED),
        ("ยาสีฟัน Colgate Total", LanguageHint.MIXED),
        ("Nintendo Switch OLED", LanguageHint.EN),
    ]
    
    success_count = 0
    
    for product_name, language_hint in test_cases:
        print(f"\n📦 Testing: {product_name}")
        
        try:
            # Create product input
            product_input = ProductInput(
                product_name=product_name,
                language_hint=language_hint
            )
            
            # Run inference
            start_time = time.time()
            
            if orchestrator.agents:
                # Use orchestrator if agents are available
                result = await orchestrator.orchestrate_inference(product_input)
                
                print(f"  🎯 Prediction: {result.best_prediction}")
                print(f"  📊 Confidence: {result.best_confidence:.3f}")
                print(f"  🔧 Method: {result.best_method}")
                print(f"  ⏱️  Time: {result.total_processing_time:.3f}s")
                
                if result.best_prediction != "Unknown":
                    success_count += 1
                    print("  ✅ Success")
                else:
                    print("  ⚠️  No prediction")
            else:
                print("  ⚠️  No agents available for inference")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print(f"\n📈 Inference results: {success_count}/{len(test_cases)} successful")
    return success_count > 0


async def test_orchestrator_cleanup(orchestrator):
    """Test orchestrator cleanup."""
    print("\n🧹 Testing orchestrator cleanup...")
    
    if orchestrator:
        try:
            await orchestrator.cleanup()
            print("✅ Cleanup completed successfully")
            return True
        except Exception as e:
            print(f"❌ Cleanup failed: {str(e)}")
            return False
    
    return True


async def main():
    """Main test function."""
    print("🚀 Starting orchestrator with default agents local test")
    print("=" * 60)
    
    # Setup configuration and logging
    try:
        config = get_config()
        setup_logging(config)
        print(f"✅ Configuration loaded (Environment: {config.environment.value})")
    except Exception as e:
        print(f"❌ Configuration failed: {str(e)}")
        return False
    
    # Test orchestrator initialization
    orchestrator = await test_orchestrator_initialization()
    
    # Test agent health
    health_ok = await test_agent_health(orchestrator)
    
    # Test inference
    inference_ok = await test_inference_samples(orchestrator)
    
    # Cleanup
    cleanup_ok = await test_orchestrator_cleanup(orchestrator)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print(f"Orchestrator initialization: {'✅ PASSED' if orchestrator else '❌ FAILED'}")
    print(f"Agent health checks:        {'✅ PASSED' if health_ok else '❌ FAILED'}")
    print(f"Inference tests:            {'✅ PASSED' if inference_ok else '❌ FAILED'}")
    print(f"Cleanup:                    {'✅ PASSED' if cleanup_ok else '❌ FAILED'}")
    
    # Overall result
    all_passed = orchestrator and health_ok and inference_ok and cleanup_ok
    
    if all_passed:
        print("\n🎉 All tests passed! Orchestrator with default agents is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        
        if not orchestrator:
            print("\n💡 Troubleshooting tips:")
            print("  - Check if required dependencies are installed (spaCy, sentence-transformers, boto3)")
            print("  - Verify AWS credentials are configured")
            print("  - Check if Milvus database is accessible")
        
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)