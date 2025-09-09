#!/usr/bin/env python3
"""
Comprehensive Orchestrator Test Script

This script directly tests the orchestrator_agent.py to show how it coordinates
all registered sub-agents and returns results from each agent.

Usage:
    python test_orchestrator_all_agents.py
    python test_orchestrator_all_agents.py --product "Samsung Galaxy S24"
    python test_orchestrator_all_agents.py --verbose
"""

import asyncio
import argparse
import sys
import time
import json
from typing import Dict, Any, List
from pathlib import Path

# Add the inference directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from inference.agents.orchestrator_agent import StrandsOrchestratorAgent
    from inference.models.data_models import ProductInput, LanguageHint
    from inference.config.settings import get_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class OrchestratorTester:
    """Test class for comprehensive orchestrator testing."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the tester."""
        self.verbose = verbose
        self.orchestrator = None
        
        # Test products for comprehensive testing
        self.test_products = [
            {"name": "Samsung Galaxy S24 Ultra", "lang": "en", "expected": "Samsung"},
            {"name": "iPhone 15 Pro Max", "lang": "en", "expected": "Apple"},
            {"name": "Nike Air Jordan 1", "lang": "en", "expected": "Nike"},
            {"name": "Sony PlayStation 5", "lang": "en", "expected": "Sony"},
            {"name": "Microsoft Surface Pro", "lang": "en", "expected": "Microsoft"},
            {"name": "Google Pixel 8", "lang": "en", "expected": "Google"},
            {"name": "Toyota Camry 2024", "lang": "en", "expected": "Toyota"},
            {"name": "โค้ก เซโร่", "lang": "th", "expected": "Coca-Cola"},
        ]
    
    async def initialize_orchestrator(self) -> bool:
        """Initialize the orchestrator with all sub-agents."""
        try:
            print("🚀 Initializing Orchestrator Agent...")
            
            # Create orchestrator instance
            self.orchestrator = StrandsOrchestratorAgent()
            
            # Initialize orchestrator (this will register and initialize all sub-agents)
            await self.orchestrator.initialize()
            
            # Get registered agents
            registered_agents = list(self.orchestrator.agents.keys())
            
            print(f"✅ Orchestrator initialized successfully!")
            print(f"📋 Registered agents: {', '.join(registered_agents)}")
            
            if self.verbose:
                await self._show_agent_details()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    async def _show_agent_details(self) -> None:
        """Show detailed information about registered agents."""
        print("\n📊 Agent Details:")
        print("-" * 50)
        
        for agent_name, agent in self.orchestrator.agents.items():
            print(f"🔧 {agent_name.upper()} Agent:")
            print(f"   Type: {type(agent).__name__}")
            print(f"   Initialized: {agent.is_initialized}")
            print(f"   Timeout: {self.orchestrator.agent_timeouts.get(agent_name, 'default')}s")
            
            # Check circuit breaker state
            cb_state = self.orchestrator.circuit_breaker_states.get(agent_name, {})
            state = cb_state.get("state", "unknown")
            failures = cb_state.get("failure_count", 0)
            print(f"   Circuit Breaker: {state} (failures: {failures})")
            print()
    
    async def test_single_product(self, product_name: str, language: str = "en") -> Dict[str, Any]:
        """Test orchestrator with a single product and show all agent results."""
        try:
            print(f"\n🧪 Testing Product: {product_name}")
            print("=" * 60)
            
            # Create input data
            input_data = ProductInput(
                product_name=product_name,
                language_hint=LanguageHint(language)
            )
            
            start_time = time.time()
            
            # Call orchestrator to get results from all agents
            result = await self.orchestrator.orchestrate_inference(input_data)
            
            total_time = time.time() - start_time
            
            # Display orchestrator summary
            print(f"🎯 Orchestrator Result:")
            print(f"   Best Prediction: {result.best_prediction}")
            print(f"   Best Confidence: {result.best_confidence:.3f}")
            print(f"   Best Method: {result.best_method}")
            print(f"   Total Time: {total_time:.3f}s")
            
            # Display individual agent results
            print(f"\n📋 Individual Agent Results:")
            print("-" * 40)
            
            # Show NER results
            if result.ner_result:
                print(f"🏷️  NER Agent:")
                print(f"   Entities: {len(result.ner_result.entities)}")
                for entity in result.ner_result.entities[:3]:  # Show first 3
                    print(f"     - {entity.text} ({entity.entity_type.value}, {entity.confidence:.3f})")
                print(f"   Processing Time: {result.ner_result.processing_time:.3f}s")
            else:
                print(f"🏷️  NER Agent: No results")
            
            # Show RAG results
            if result.rag_result:
                print(f"🔍 RAG Agent:")
                print(f"   Predicted Brand: {result.rag_result.predicted_brand}")
                print(f"   Confidence: {result.rag_result.confidence:.3f}")
                print(f"   Similar Products: {len(getattr(result.rag_result, 'similar_products', []))}")
                print(f"   Processing Time: {result.rag_result.processing_time:.3f}s")
            else:
                print(f"🔍 RAG Agent: No results")
            
            # Show LLM results
            if result.llm_result:
                print(f"🤖 LLM Agent:")
                print(f"   Predicted Brand: {result.llm_result.predicted_brand}")
                print(f"   Confidence: {result.llm_result.confidence:.3f}")
                reasoning = result.llm_result.reasoning[:60] + "..." if len(result.llm_result.reasoning) > 60 else result.llm_result.reasoning
                print(f"   Reasoning: {reasoning}")
                print(f"   Processing Time: {result.llm_result.processing_time:.3f}s")
            else:
                print(f"🤖 LLM Agent: No results")
            
            # Show Hybrid results
            if result.hybrid_result:
                print(f"🔄 Hybrid Agent:")
                print(f"   Predicted Brand: {result.hybrid_result.predicted_brand}")
                print(f"   Confidence: {result.hybrid_result.confidence:.3f}")
                if hasattr(result.hybrid_result, 'contributions'):
                    contrib = result.hybrid_result.contributions
                    print(f"   Contributions: NER={contrib.get('ner', 0):.2f}, RAG={contrib.get('rag', 0):.2f}, LLM={contrib.get('llm', 0):.2f}")
                print(f"   Processing Time: {result.hybrid_result.processing_time:.3f}s")
            else:
                print(f"🔄 Hybrid Agent: No results")
            
            # Show Simple agent results (if available)
            # Note: Simple agent results might be in llm_result or separate field
            
            return {
                "product_name": product_name,
                "orchestrator_result": {
                    "best_prediction": result.best_prediction,
                    "best_confidence": result.best_confidence,
                    "best_method": result.best_method,
                    "total_time": total_time
                },
                "agent_results": {
                    "ner": result.ner_result is not None,
                    "rag": result.rag_result is not None,
                    "llm": result.llm_result is not None,
                    "hybrid": result.hybrid_result is not None
                },
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error testing product '{product_name}': {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {
                "product_name": product_name,
                "success": False,
                "error": str(e)
            }
    
    async def test_all_products(self) -> Dict[str, Any]:
        """Test orchestrator with all test products."""
        print(f"\n🚀 Testing All Products with Orchestrator")
        print("=" * 70)
        
        results = []
        successful_tests = 0
        
        for i, product in enumerate(self.test_products, 1):
            print(f"\n[{i}/{len(self.test_products)}] Testing: {product['name']}")
            
            result = await self.test_single_product(product['name'], product['lang'])
            results.append(result)
            
            if result['success']:
                successful_tests += 1
                orchestrator_result = result['orchestrator_result']
                expected = product['expected'].lower()
                actual = orchestrator_result['best_prediction'].lower()
                
                if expected in actual or actual in expected:
                    print(f"   ✅ Correct prediction: {orchestrator_result['best_prediction']}")
                else:
                    print(f"   ⚠️  Different prediction: {orchestrator_result['best_prediction']} (expected: {product['expected']})")
            else:
                print(f"   ❌ Test failed")
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Summary
        print(f"\n📊 Test Summary:")
        print("=" * 50)
        print(f"Total Tests: {len(self.test_products)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {len(self.test_products) - successful_tests}")
        print(f"Success Rate: {(successful_tests / len(self.test_products)) * 100:.1f}%")
        
        # Agent availability summary
        agent_availability = {}
        for result in results:
            if result['success']:
                for agent, available in result['agent_results'].items():
                    if agent not in agent_availability:
                        agent_availability[agent] = 0
                    if available:
                        agent_availability[agent] += 1
        
        print(f"\n🔧 Agent Availability:")
        for agent, count in agent_availability.items():
            percentage = (count / successful_tests) * 100 if successful_tests > 0 else 0
            print(f"   {agent.upper()}: {count}/{successful_tests} ({percentage:.1f}%)")
        
        return {
            "total_tests": len(self.test_products),
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / len(self.test_products)) * 100,
            "agent_availability": agent_availability,
            "results": results
        }
    
    async def test_agent_health(self) -> None:
        """Test health status of all registered agents."""
        try:
            print(f"\n🏥 Agent Health Check")
            print("=" * 40)
            
            health_results = await self.orchestrator.get_agent_health()
            
            for agent_name, health in health_results.items():
                status = "✅ Healthy" if health.is_healthy else "❌ Unhealthy"
                print(f"{agent_name.upper()}: {status}")
                
                if not health.is_healthy and health.error_message:
                    print(f"   Error: {health.error_message}")
                
                if health.response_time:
                    print(f"   Response Time: {health.response_time:.3f}s")
                
                print(f"   Last Check: {time.ctime(health.last_check)}")
                print()
        
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up orchestrator and agents."""
        if self.orchestrator:
            try:
                await self.orchestrator.cleanup()
                print("🧹 Orchestrator cleanup completed")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Orchestrator Agent with All Sub-Agents")
    parser.add_argument("--product", "-p", help="Single product to test")
    parser.add_argument("--language", "-l", default="en", help="Language hint (en/th)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--health", action="store_true", help="Run health check only")
    
    args = parser.parse_args()
    
    # Create tester
    tester = OrchestratorTester(verbose=args.verbose)
    
    try:
        # Initialize orchestrator
        if not await tester.initialize_orchestrator():
            print("❌ Failed to initialize orchestrator")
            return 1
        
        # Run health check if requested
        if args.health:
            await tester.test_agent_health()
            return 0
        
        # Test single product or all products
        if args.product:
            result = await tester.test_single_product(args.product, args.language)
            if result['success']:
                print(f"\n✅ Test completed successfully!")
            else:
                print(f"\n❌ Test failed!")
                return 1
        else:
            summary = await tester.test_all_products()
            
            if summary['success_rate'] >= 80:
                print(f"\n🎉 All tests completed with {summary['success_rate']:.1f}% success rate!")
            else:
                print(f"\n⚠️  Tests completed with {summary['success_rate']:.1f}% success rate")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Always cleanup
        await tester.cleanup()


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)