#!/usr/bin/env python3
"""
Test the complete inference system with Strands v1.7.1.

This script tests the full inference pipeline to ensure everything works correctly
after the v1.7.1 update.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_orchestrator_inference():
    """Test the complete orchestrator inference pipeline."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        from inference.models.data_models import ProductInput, LanguageHint
        
        logger.info("🧪 Testing complete orchestrator inference pipeline")
        
        # Create orchestrator
        config = {
            "confidence_threshold": 0.6,
            "max_parallel_agents": 4
        }
        orchestrator = StrandsMultiAgentOrchestrator(config)
        
        # Test products
        test_products = [
            "Samsung Galaxy S24 Ultra 256GB",
            "iPhone 15 Pro Max",
            "Sony WH-1000XM5 Headphones",
            "Nike Air Max 270",
            "Toyota Camry 2024"
        ]
        
        results = []
        
        for product in test_products:
            logger.info(f"🔍 Testing product: {product}")
            
            # Test multiagent inference
            result = await orchestrator.orchestrate_multiagent_inference(
                product, 
                coordination_method="swarm"
            )
            
            results.append({
                "product": product,
                "brand": result.get("best_prediction", "Unknown"),
                "confidence": result.get("best_confidence", 0.0),
                "method": result.get("best_method", "unknown"),
                "time": result.get("orchestration_time", 0.0)
            })
            
            logger.info(f"   ✅ Brand: {result.get('best_prediction', 'Unknown')} "
                       f"(confidence: {result.get('best_confidence', 0.0):.3f})")
        
        # Summary
        logger.info("\n📊 Inference Results Summary:")
        for result in results:
            logger.info(f"   {result['product']}: {result['brand']} "
                       f"({result['confidence']:.3f}) - {result['time']:.2f}s")
        
        # Calculate success rate
        successful = sum(1 for r in results if r['brand'] != 'Unknown')
        success_rate = successful / len(results) * 100
        
        logger.info(f"\n📈 Success Rate: {successful}/{len(results)} ({success_rate:.1f}%)")
        
        return success_rate > 50  # At least 50% success rate
        
    except Exception as e:
        logger.error(f"❌ Orchestrator inference test failed: {e}")
        return False

async def test_compatibility_layer():
    """Test the compatibility layer with existing code."""
    try:
        from inference.agents.orchestrator_agent import StrandsOrchestratorAgent
        from inference.models.data_models import ProductInput, LanguageHint
        
        logger.info("🧪 Testing compatibility layer")
        
        # Create compatibility orchestrator
        orchestrator = StrandsOrchestratorAgent()
        await orchestrator.initialize()
        
        # Test with ProductInput
        product_input = ProductInput(
            product_name="Samsung Galaxy S24 Ultra",
            language_hint=LanguageHint.AUTO
        )
        
        result = await orchestrator.process(product_input)
        
        logger.info(f"✅ Compatibility test result: {result.get('brand_predictions', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Compatibility layer test failed: {e}")
        return False

async def test_agent_status():
    """Test agent status and monitoring."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        logger.info("🧪 Testing agent status and monitoring")
        
        orchestrator = StrandsMultiAgentOrchestrator()
        
        # Create some agents
        orchestrator.create_ner_agent()
        orchestrator.create_rag_agent()
        orchestrator.create_llm_agent()
        orchestrator.create_hybrid_agent()
        
        # Get status
        status = orchestrator.get_agent_status()
        
        logger.info(f"✅ Agent status:")
        logger.info(f"   - Orchestrator type: {status.get('orchestrator_type')}")
        logger.info(f"   - Total agents: {status.get('total_agents')}")
        logger.info(f"   - Multiagent tools: {status.get('multiagent_tools')}")
        logger.info(f"   - Coordination methods: {status.get('coordination_methods')}")
        
        return status.get('total_agents', 0) > 0
        
    except Exception as e:
        logger.error(f"❌ Agent status test failed: {e}")
        return False

async def main():
    """Run all inference system tests."""
    logger.info("🚀 Starting complete inference system tests with Strands v1.7.1")
    
    tests = [
        ("Orchestrator Inference", test_orchestrator_inference),
        ("Compatibility Layer", test_compatibility_layer),
        ("Agent Status", test_agent_status),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All inference system tests passed! System is ready for production.")
        logger.info("\n📋 System Status:")
        logger.info("   ✅ Strands Agents v1.7.1 integration working")
        logger.info("   ✅ Multi-agent orchestration operational")
        logger.info("   ✅ Brand extraction functioning")
        logger.info("   ✅ Compatibility layer working")
        logger.info("   ✅ Monitoring and status reporting active")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))