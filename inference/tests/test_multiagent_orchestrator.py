#!/usr/bin/env python3
"""
Test script demonstrating the refactored Strands multiagent orchestrator.

This script showcases the new multiagent capabilities including:
- Specialized agent creation (NER, RAG, LLM, Hybrid)
- Multiple coordination methods (swarm, graph, workflow)
- Strands multiagent tools integration
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.agents.orchestrator_agent import (
    StrandsMultiAgentOrchestrator,
    StrandsOrchestratorAgent,
    create_multiagent_orchestrator_example
)
from inference.models.data_models import ProductInput, LanguageHint


async def test_multiagent_orchestrator():
    """Test the new multiagent orchestrator capabilities."""
    
    print("🚀 Testing Strands MultiAgent Orchestrator")
    print("=" * 50)
    
    # Create orchestrator instance
    config = {
        "confidence_threshold": 0.6,
        "max_parallel_agents": 4
    }
    
    orchestrator = StrandsMultiAgentOrchestrator(config)
    
    # Test product inputs
    test_products = [
        "Samsung Galaxy S24 Ultra 256GB",
        "iPhone 15 Pro Max 512GB",
        "Sony WH-1000XM5 Wireless Headphones",
        "MacBook Pro M3 14-inch",
        "Nintendo Switch OLED"
    ]
    
    print(f"📱 Testing with {len(test_products)} products")
    print()
    
    # Test different coordination methods
    coordination_methods = ["swarm", "graph", "workflow"]
    
    for i, product_name in enumerate(test_products, 1):
        print(f"Test {i}: {product_name}")
        print("-" * 40)
        
        # Test each coordination method
        for method in coordination_methods:
            try:
                print(f"  🔄 Testing {method} coordination...")
                
                result = await orchestrator.orchestrate_multiagent_inference(
                    product_name, 
                    coordination_method=method
                )
                
                print(f"    ✅ {method.upper()}: {result['best_prediction']} "
                      f"(confidence: {result['best_confidence']:.3f}, "
                      f"time: {result['orchestration_time']:.3f}s)")
                
            except Exception as e:
                print(f"    ❌ {method.upper()}: Error - {str(e)}")
        
        print()
    
    # Test agent status
    print("🔍 Agent Status:")
    print("-" * 20)
    status = orchestrator.get_agent_status()
    
    print(f"Orchestrator Type: {status['orchestrator_type']}")
    print(f"Total Specialized Agents: {status['total_agents']}")
    print(f"Status: {status['status']}")
    
    print("\nSpecialized Agents:")
    for agent_id, agent_info in status['specialized_agents'].items():
        print(f"  - {agent_id}: {agent_info['type']} ({agent_info['model']})")
    
    print("\nMultiagent Tools Available:")
    for tool_name, available in status['multiagent_tools'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {tool_name}")
    
    print(f"\nSupported Coordination Methods: {', '.join(status['coordination_methods'])}")


async def test_compatibility_layer():
    """Test the compatibility layer for existing code."""
    
    print("\n🔄 Testing Compatibility Layer")
    print("=" * 30)
    
    # Create orchestrator using compatibility class
    orchestrator = StrandsOrchestratorAgent()
    
    # Initialize (compatibility method)
    await orchestrator.initialize()
    
    # Test with ProductInput (existing interface)
    product_input = ProductInput(
        product_name="Apple AirPods Pro 2nd Generation",
        language_hint=LanguageHint.AUTO
    )
    
    try:
        result = await orchestrator.process(product_input)
        
        print(f"✅ Compatibility test passed!")
        print(f"   Product: {result['product_name']}")
        print(f"   Agent Used: {result['agent_used']}")
        print(f"   Coordination: {result.get('coordination_method', 'N/A')}")
        print(f"   Processing Time: {result['processing_time_ms']}ms")
        
        if result['brand_predictions']:
            pred = result['brand_predictions'][0]
            print(f"   Prediction: {pred['brand']} (confidence: {pred['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {str(e)}")


async def test_example_usage():
    """Test the example usage patterns."""
    
    print("\n📚 Testing Example Usage Patterns")
    print("=" * 35)
    
    try:
        orchestrator, examples = create_multiagent_orchestrator_example()
        
        print("✅ Example orchestrator created successfully")
        print(f"   Available examples: {list(examples.keys())}")
        
        # Test one example
        if "swarm_example" in examples:
            print("\n🐝 Testing swarm example...")
            try:
                result = await examples["swarm_example"]()
                print(f"   ✅ Swarm example completed: {result.get('best_prediction', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Swarm example failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ Example usage test failed: {str(e)}")


async def main():
    """Main test function."""
    
    print("🧪 Strands MultiAgent Orchestrator Test Suite")
    print("=" * 50)
    print()
    
    try:
        # Test core multiagent functionality
        await test_multiagent_orchestrator()
        
        # Test compatibility layer
        await test_compatibility_layer()
        
        # Test example usage
        await test_example_usage()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())