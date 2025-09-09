#!/usr/bin/env python3
"""
Local test for orchestrator agent to verify agent_results functionality.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.models.data_models import ProductInput, LanguageHint
from inference.agents.orchestrator_agent import StrandsOrchestratorAgent


async def test_orchestrator_agent_results():
    """Test that orchestrator returns agent_results properly."""
    
    print("🧪 Testing Orchestrator Agent Results Locally")
    print("=" * 50)
    
    try:
        # Create orchestrator instance
        orchestrator = StrandsOrchestratorAgent()
        
        # Initialize orchestrator
        print("🔧 Initializing orchestrator...")
        await orchestrator.initialize()
        
        print(f"✅ Orchestrator initialized with {len(orchestrator.agents)} agents")
        print(f"   Available agents: {list(orchestrator.agents.keys())}")
        
        # Test with a sample product
        test_product = ProductInput(
            product_name="iPhone 15 Pro Max",
            language_hint=LanguageHint.ENGLISH
        )
        
        print(f"\n🚀 Testing inference with: {test_product.product_name}")
        
        # Process through orchestrator
        result = await orchestrator.process(test_product)
        
        print(f"\n📊 Results:")
        print(f"   Product: {result.get('product_name', 'N/A')}")
        print(f"   Agent Used: {result.get('agent_used', 'N/A')}")
        print(f"   Processing Time: {result.get('processing_time_ms', 0)}ms")
        
        # Check brand predictions
        predictions = result.get('brand_predictions', [])
        if predictions:
            best_pred = predictions[0]
            print(f"   Best Prediction: {best_pred.get('brand', 'Unknown')}")
            print(f"   Confidence: {best_pred.get('confidence', 0.0):.3f}")
            print(f"   Method: {best_pred.get('method', 'unknown')}")
        
        # Check orchestrator agents
        orchestrator_agents = result.get('orchestrator_agents', [])
        print(f"   Orchestrator Agents: {orchestrator_agents}")
        
        # Check agent results - this is what we're testing
        agent_results = result.get('agent_results', {})
        print(f"\n🤖 Agent Results:")
        if agent_results:
            for agent_name, agent_detail in agent_results.items():
                if isinstance(agent_detail, dict):
                    success = agent_detail.get('success', False)
                    prediction = agent_detail.get('prediction', 'N/A')
                    confidence = agent_detail.get('confidence', 0.0)
                    processing_time = agent_detail.get('processing_time', 0.0)
                    error = agent_detail.get('error')
                    
                    if success:
                        print(f"   - {agent_name.upper()}: ✅ {prediction} (conf: {confidence:.3f}, time: {processing_time:.3f}s)")
                        
                        # Show additional details for specific agents
                        if agent_name == "ner" and "entities_count" in agent_detail:
                            print(f"     └─ Entities found: {agent_detail['entities_count']}")
                        elif agent_name == "rag" and "similar_products_count" in agent_detail:
                            print(f"     └─ Similar products: {agent_detail['similar_products_count']}")
                        elif agent_name in ["llm", "simple"] and "reasoning" in agent_detail:
                            print(f"     └─ Reasoning: {agent_detail['reasoning']}")
                        elif agent_name == "hybrid":
                            ner_contrib = agent_detail.get('ner_contribution', 0.0)
                            rag_contrib = agent_detail.get('rag_contribution', 0.0)
                            llm_contrib = agent_detail.get('llm_contribution', 0.0)
                            print(f"     └─ Contributions: NER={ner_contrib:.2f}, RAG={rag_contrib:.2f}, LLM={llm_contrib:.2f}")
                    else:
                        print(f"   - {agent_name.upper()}: ❌ Failed - {error}")
                else:
                    # Legacy format (boolean)
                    status = "✅ Success" if agent_detail else "❌ Failed"
                    print(f"   - {agent_name.upper()}: {status}")
        else:
            print("   ❌ No agent results found!")
            return False
        
        # Check entities
        entities = result.get('entities', [])
        if entities:
            print(f"\n🏷️  Entities ({len(entities)}):")
            for entity in entities[:3]:  # Show first 3
                print(f"   - {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')}, {entity.get('confidence', 0.0):.3f})")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Agent results properly populated: {len(agent_results)} agents")
        
        # Cleanup
        await orchestrator.cleanup()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_orchestrator_agent_results()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)