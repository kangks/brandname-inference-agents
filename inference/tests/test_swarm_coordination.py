#!/usr/bin/env python3
"""
Test Swarm coordination specifically.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_swarm_coordination():
    """Test the Swarm coordination functionality."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        logger.info("🧪 Testing Swarm coordination")
        
        orchestrator = StrandsMultiAgentOrchestrator()
        
        # Test swarm coordination
        result = await orchestrator.orchestrate_multiagent_inference(
            "Samsung Galaxy S24 Ultra",
            coordination_method="swarm"
        )
        
        logger.info(f"✅ Swarm coordination result:")
        logger.info(f"   - Method: {result.get('coordination_method')}")
        logger.info(f"   - Best prediction: {result.get('best_prediction')}")
        logger.info(f"   - Confidence: {result.get('best_confidence')}")
        logger.info(f"   - Success: {result.get('success', True)}")
        
        if 'error' in result:
            logger.error(f"❌ Error: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Swarm coordination test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_swarm_coordination())
    if result:
        logger.info("🎉 Swarm coordination test passed!")
    else:
        logger.error("💥 Swarm coordination test failed!")