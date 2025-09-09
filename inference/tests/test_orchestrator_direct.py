#!/usr/bin/env python3
"""
Direct test of orchestrator agent to debug the issue.
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from inference.models.data_models import ProductInput, LanguageHint
from inference.agents.orchestrator_agent import create_orchestrator_agent

async def test_orchestrator():
    """Test the orchestrator directly."""
    print("🔬 Testing Orchestrator Agent Directly")
    print("=" * 50)
    
    try:
        # Create orchestrator
        print("1. Creating orchestrator...")
        orchestrator = create_orchestrator_agent()
        
        # Initialize orchestrator
        print("2. Initializing orchestrator...")
        await orchestrator.initialize()
        
        print(f"3. Orchestrator initialized with {len(orchestrator.agents)} agents")
        print(f"   Registered agents: {list(orchestrator.agents.keys())}")
        
        # Check agent availability
        print("4. Checking agent availability...")
        for agent_name in orchestrator.agents.keys():
            available = orchestrator._is_agent_available(agent_name)
            print(f"   - {agent_name}: {'Available' if available else 'Not Available'}")
            
            # Check circuit breaker state
            cb_state = orchestrator.circuit_breaker_states.get(agent_name, {})
            print(f"     Circuit breaker state: {cb_state.get('state', 'unknown')}")
        
        # Test inference
        print("5. Testing inference...")
        product_input = ProductInput(
            product_name="iPhone 15 Pro Max",
            language_hint=LanguageHint.ENGLISH
        )
        
        try:
            result = await orchestrator.process(product_input)
            print("✅ Inference successful!")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
        
        # Cleanup
        print("6. Cleaning up...")
        await orchestrator.cleanup()
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator())