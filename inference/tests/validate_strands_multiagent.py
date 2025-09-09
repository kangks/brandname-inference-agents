#!/usr/bin/env python3
"""
Validation script for Strands multiagent orchestrator.

This script validates that the imports and multiagent tools work correctly
when strands-agents and strands-agents-tools are properly installed.
"""

import sys
import os
import asyncio

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate_strands_imports():
    """Validate that Strands imports work correctly."""
    print("🔍 Validating Strands imports...")
    
    try:
        from strands import Agent, tool
        print("✅ strands core imports successful")
        
        # Test Agent creation
        test_agent = Agent(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            system_prompt="Test agent"
        )
        print("✅ Agent creation successful")
        
    except ImportError as e:
        print(f"❌ strands core import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False
    
    try:
        from strands_tools import agent_graph, swarm, workflow, journal
        print("✅ strands_tools multiagent imports successful")
        
        # Check if tools are available
        tools_status = {
            "agent_graph": agent_graph is not None,
            "swarm": swarm is not None,
            "workflow": workflow is not None,
            "journal": journal is not None
        }
        
        for tool_name, available in tools_status.items():
            status = "✅" if available else "❌"
            print(f"   {status} {tool_name}: {available}")
        
        return all(tools_status.values())
        
    except ImportError as e:
        print(f"❌ strands_tools import failed: {e}")
        return False


def validate_orchestrator_imports():
    """Validate that the orchestrator imports work correctly."""
    print("\n🔍 Validating orchestrator imports...")
    
    try:
        from inference.agents.orchestrator_agent import (
            StrandsMultiAgentOrchestrator,
            StrandsOrchestratorAgent,
            create_multiagent_orchestrator_example
        )
        print("✅ Orchestrator imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Orchestrator import failed: {e}")
        return False


async def validate_multiagent_functionality():
    """Validate multiagent functionality."""
    print("\n🔍 Validating multiagent functionality...")
    
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        # Create orchestrator
        config = {
            "confidence_threshold": 0.6,
            "max_parallel_agents": 4
        }
        orchestrator = StrandsMultiAgentOrchestrator(config)
        print("✅ Orchestrator created successfully")
        
        # Check agent status
        status = orchestrator.get_agent_status()
        print(f"✅ Agent status: {status['orchestrator_type']}")
        
        # Test agent creation
        ner_id = orchestrator.create_ner_agent()
        rag_id = orchestrator.create_rag_agent()
        llm_id = orchestrator.create_llm_agent()
        hybrid_id = orchestrator.create_hybrid_agent()
        
        print(f"✅ Created {len(orchestrator.specialized_agents)} specialized agents")
        
        # Test coordination methods
        test_product = "Samsung Galaxy S24 Ultra"
        coordination_methods = ["swarm", "graph", "workflow"]
        
        for method in coordination_methods:
            try:
                result = await orchestrator.orchestrate_multiagent_inference(
                    test_product, 
                    coordination_method=method
                )
                
                print(f"✅ {method.upper()} coordination: {result['best_prediction']} "
                      f"(confidence: {result['best_confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ {method.upper()} coordination failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiagent functionality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def validate_compatibility():
    """Validate backward compatibility."""
    print("\n🔍 Validating backward compatibility...")
    
    try:
        from inference.agents.orchestrator_agent import StrandsOrchestratorAgent
        from inference.models.data_models import ProductInput, LanguageHint
        
        # Test compatibility layer
        orchestrator = StrandsOrchestratorAgent()
        await orchestrator.initialize()
        print("✅ Compatibility orchestrator initialized")
        
        # Test with existing interface
        product_input = ProductInput(
            product_name="iPhone 15 Pro Max",
            language_hint=LanguageHint.AUTO
        )
        
        result = await orchestrator.process(product_input)
        print("✅ Compatibility processing successful")
        print(f"   Agent used: {result['agent_used']}")
        print(f"   Coordination: {result.get('coordination_method', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_documentation_examples():
    """Validate examples from documentation."""
    print("\n🔍 Validating documentation examples...")
    
    try:
        from inference.agents.orchestrator_agent import create_multiagent_orchestrator_example
        
        orchestrator, examples = create_multiagent_orchestrator_example()
        print("✅ Example orchestrator created")
        print(f"   Available examples: {list(examples.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Documentation examples validation failed: {e}")
        return False


async def main():
    """Main validation function."""
    print("🧪 Strands MultiAgent Orchestrator Validation")
    print("=" * 50)
    
    results = []
    
    # Validate imports
    results.append(validate_strands_imports())
    results.append(validate_orchestrator_imports())
    
    # Validate functionality
    results.append(await validate_multiagent_functionality())
    results.append(await validate_compatibility())
    results.append(validate_documentation_examples())
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 20)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All validations passed! The Strands multiagent orchestrator is working correctly.")
        return True
    else:
        print("❌ Some validations failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)