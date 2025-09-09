#!/usr/bin/env python3
"""
Orchestrator Coordination Test Script

This script demonstrates how the orchestrator_agent.py coordinates multiple
sub-agents and aggregates their results to provide the best prediction.

Usage:
    python test_orchestrator_coordination.py
    python test_orchestrator_coordination.py --product "iPhone 15 Pro"
"""

import asyncio
import argparse
import sys
import time
import json
from pathlib import Path

# Add the inference directory to the path
sys.path.insert(0, str(Path(__file__).parent / "inference"))

try:
    from inference.agents.orchestrator_agent import StrandsOrchestratorAgent
    from inference.models.data_models import ProductInput, LanguageHint
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


async def test_orchestrator_coordination(product_name: str = "Samsung Galaxy S24 Ultra"):
    """
    Test how the orchestrator coordinates multiple agents.
    
    Args:
        product_name: Product to test with
    """
    print("🎯 Orchestrator Coordination Test")
    print("=" * 50)
    print(f"Testing Product: {product_name}")
    print()
    
    orchestrator = None
    
    try:
        # Step 1: Initialize Orchestrator
        print("🚀 Step 1: Initializing Orchestrator...")
        orchestrator = StrandsOrchestratorAgent()
        await orchestrator.initialize()
        
        registered_agents = list(orchestrator.agents.keys())
        print(f"✅ Orchestrator initialized with {len(registered_agents)} agents:")
        for agent_name in registered_agents:
            agent = orchestrator.agents[agent_name]
            status = "✅ Ready" if agent.is_initialized else "❌ Failed"
            print(f"   - {agent_name.upper()}: {status}")
        print()
        
        # Step 2: Show Agent Health
        print("🏥 Step 2: Checking Agent Health...")
        health_results = await orchestrator.get_agent_health()
        
        healthy_agents = []
        for agent_name, health in health_results.items():
            status = "✅ Healthy" if health.is_healthy else "❌ Unhealthy"
            print(f"   {agent_name.upper()}: {status}")
            if health.is_healthy:
                healthy_agents.append(agent_name)
            elif health.error_message:
                print(f"      Error: {health.error_message}")
        
        print(f"\n📊 Available Agents: {len(healthy_agents)}/{len(registered_agents)}")
        print()
        
        # Step 3: Execute Orchestrated Inference
        print("🧪 Step 3: Executing Orchestrated Inference...")
        
        input_data = ProductInput(
            product_name=product_name,
            language_hint=LanguageHint("en")
        )
        
        start_time = time.time()
        
        # This is the key call - orchestrator coordinates all sub-agents
        inference_result = await orchestrator.orchestrate_inference(input_data)
        
        total_time = time.time() - start_time
        
        print(f"✅ Orchestration completed in {total_time:.3f}s")
        print()
        
        # Step 4: Show Orchestrator Decision
        print("🎯 Step 4: Orchestrator Decision Process")
        print("-" * 40)
        print(f"🏆 BEST PREDICTION: {inference_result.best_prediction}")
        print(f"🎯 CONFIDENCE: {inference_result.best_confidence:.3f}")
        print(f"🔧 METHOD USED: {inference_result.best_method}")
        print(f"⏱️  TOTAL TIME: {inference_result.total_processing_time:.3f}s")
        print()
        
        # Step 5: Show Individual Agent Contributions
        print("📋 Step 5: Individual Agent Contributions")
        print("-" * 45)
        
        agent_results = []
        
        # NER Agent Results
        if inference_result.ner_result:
            ner = inference_result.ner_result
            entities = [e for e in ner.entities if e.entity_type.value == "BRAND"]
            
            print(f"🏷️  NER AGENT:")
            print(f"   Entities Found: {len(entities)}")
            if entities:
                best_entity = max(entities, key=lambda x: x.confidence)
                print(f"   Best Entity: {best_entity.text} (confidence: {best_entity.confidence:.3f})")
                agent_results.append(("NER", best_entity.text, best_entity.confidence))
            else:
                print(f"   No brand entities found")
            print(f"   Processing Time: {ner.processing_time:.3f}s")
            print()
        else:
            print(f"🏷️  NER AGENT: No results")
            print()
        
        # RAG Agent Results
        if inference_result.rag_result:
            rag = inference_result.rag_result
            print(f"🔍 RAG AGENT:")
            print(f"   Predicted Brand: {rag.predicted_brand}")
            print(f"   Confidence: {rag.confidence:.3f}")
            print(f"   Processing Time: {rag.processing_time:.3f}s")
            agent_results.append(("RAG", rag.predicted_brand, rag.confidence))
            print()
        else:
            print(f"🔍 RAG AGENT: No results")
            print()
        
        # LLM Agent Results
        if inference_result.llm_result:
            llm = inference_result.llm_result
            print(f"🤖 LLM AGENT:")
            print(f"   Predicted Brand: {llm.predicted_brand}")
            print(f"   Confidence: {llm.confidence:.3f}")
            reasoning = llm.reasoning[:50] + "..." if len(llm.reasoning) > 50 else llm.reasoning
            print(f"   Reasoning: {reasoning}")
            print(f"   Processing Time: {llm.processing_time:.3f}s")
            agent_results.append(("LLM", llm.predicted_brand, llm.confidence))
            print()
        else:
            print(f"🤖 LLM AGENT: No results")
            print()
        
        # Hybrid Agent Results
        if inference_result.hybrid_result:
            hybrid = inference_result.hybrid_result
            print(f"🔄 HYBRID AGENT:")
            print(f"   Predicted Brand: {hybrid.predicted_brand}")
            print(f"   Confidence: {hybrid.confidence:.3f}")
            print(f"   Processing Time: {hybrid.processing_time:.3f}s")
            agent_results.append(("HYBRID", hybrid.predicted_brand, hybrid.confidence))
            print()
        else:
            print(f"🔄 HYBRID AGENT: No results")
            print()
        
        # Step 6: Show Orchestrator Logic
        print("🧠 Step 6: Orchestrator Decision Logic")
        print("-" * 40)
        
        if agent_results:
            print("Agent Predictions Comparison:")
            for agent_name, prediction, confidence in agent_results:
                selected = "👑" if agent_name.lower() == inference_result.best_method else "  "
                print(f"   {selected} {agent_name}: {prediction} ({confidence:.3f})")
            
            print(f"\n🎯 Orchestrator selected '{inference_result.best_method.upper()}' agent")
            print(f"   Reason: Highest confidence score ({inference_result.best_confidence:.3f})")
        else:
            print("⚠️  No agent predictions available")
        
        print()
        
        # Step 7: Summary
        print("📊 Step 7: Coordination Summary")
        print("-" * 35)
        print(f"✅ Agents Executed: {len([r for r in [inference_result.ner_result, inference_result.rag_result, inference_result.llm_result, inference_result.hybrid_result] if r is not None])}")
        print(f"🎯 Predictions Made: {len(agent_results)}")
        print(f"🏆 Best Method: {inference_result.best_method}")
        print(f"⏱️  Total Coordination Time: {inference_result.total_processing_time:.3f}s")
        print(f"🎉 Final Result: {inference_result.best_prediction} ({inference_result.best_confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during orchestrator test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if orchestrator:
            try:
                await orchestrator.cleanup()
                print("\n🧹 Cleanup completed")
            except Exception as e:
                print(f"\n⚠️  Cleanup warning: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Orchestrator Coordination")
    parser.add_argument("--product", "-p", default="Samsung Galaxy S24 Ultra", 
                       help="Product name to test")
    
    args = parser.parse_args()
    
    try:
        success = await test_orchestrator_coordination(args.product)
        
        if success:
            print("\n🎉 Orchestrator coordination test completed successfully!")
            return 0
        else:
            print("\n❌ Orchestrator coordination test failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)