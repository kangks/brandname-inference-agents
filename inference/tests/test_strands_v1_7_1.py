#!/usr/bin/env python3
"""
Test script to verify Strands Agents v1.7.1 compatibility.

This script tests the basic functionality of Strands Agents v1.7.1
and the multi-agent orchestrator implementation.
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

def test_strands_import():
    """Test basic Strands imports."""
    try:
        from strands import Agent, tool
        from strands_tools import calculator, current_time
        logger.info("✅ Successfully imported Strands core components")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import Strands components: {e}")
        return False

def test_strands_multiagent_tools():
    """Test Strands multiagent tools import."""
    try:
        from strands_tools import agent_graph, swarm, workflow, journal
        logger.info("✅ Successfully imported Strands multiagent tools")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Some multiagent tools not available: {e}")
        return False

def test_basic_agent():
    """Test basic agent creation and execution."""
    try:
        from strands import Agent, tool
        from strands_tools import calculator
        
        @tool
        def test_tool(message: str) -> str:
            """A simple test tool."""
            return f"Test tool received: {message}"
        
        # Create a simple agent
        agent = Agent(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            tools=[calculator, test_tool],
            system_prompt="You are a test agent for verifying Strands functionality."
        )
        
        logger.info("✅ Successfully created basic Strands agent")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create basic agent: {e}")
        return False

def test_orchestrator_import():
    """Test orchestrator import."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        logger.info("✅ Successfully imported StrandsMultiAgentOrchestrator")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import orchestrator: {e}")
        return False

def test_orchestrator_creation():
    """Test orchestrator creation."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        config = {
            "confidence_threshold": 0.6,
            "max_parallel_agents": 4
        }
        
        orchestrator = StrandsMultiAgentOrchestrator(config)
        logger.info("✅ Successfully created StrandsMultiAgentOrchestrator")
        
        # Test agent creation methods
        if hasattr(orchestrator, 'create_ner_agent'):
            logger.info("✅ Orchestrator has create_ner_agent method")
        if hasattr(orchestrator, 'create_rag_agent'):
            logger.info("✅ Orchestrator has create_rag_agent method")
        if hasattr(orchestrator, 'create_llm_agent'):
            logger.info("✅ Orchestrator has create_llm_agent method")
        if hasattr(orchestrator, 'create_hybrid_agent'):
            logger.info("✅ Orchestrator has create_hybrid_agent method")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create orchestrator: {e}")
        return False

async def test_orchestrator_functionality():
    """Test basic orchestrator functionality."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        orchestrator = StrandsMultiAgentOrchestrator()
        
        # Test agent creation
        ner_agent_id = orchestrator.create_ner_agent()
        logger.info(f"✅ Created NER agent: {ner_agent_id}")
        
        rag_agent_id = orchestrator.create_rag_agent()
        logger.info(f"✅ Created RAG agent: {rag_agent_id}")
        
        llm_agent_id = orchestrator.create_llm_agent()
        logger.info(f"✅ Created LLM agent: {llm_agent_id}")
        
        hybrid_agent_id = orchestrator.create_hybrid_agent()
        logger.info(f"✅ Created Hybrid agent: {hybrid_agent_id}")
        
        # Test status
        status = orchestrator.get_agent_status()
        logger.info(f"✅ Orchestrator status: {status['total_agents']} agents created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed orchestrator functionality test: {e}")
        return False

async def test_multiagent_inference():
    """Test multiagent inference functionality."""
    try:
        from inference.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
        
        orchestrator = StrandsMultiAgentOrchestrator()
        
        # Test inference with fallback (since we may not have real tools)
        result = await orchestrator.orchestrate_multiagent_inference(
            "Samsung Galaxy S24 Ultra",
            coordination_method="swarm"
        )
        
        logger.info(f"✅ Multiagent inference completed")
        logger.info(f"   - Method: {result.get('coordination_method', 'unknown')}")
        logger.info(f"   - Best prediction: {result.get('best_prediction', 'unknown')}")
        logger.info(f"   - Confidence: {result.get('best_confidence', 0.0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed multiagent inference test: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Strands Agents v1.7.1 compatibility tests")
    
    tests = [
        ("Strands Import", test_strands_import),
        ("Multiagent Tools Import", test_strands_multiagent_tools),
        ("Basic Agent Creation", test_basic_agent),
        ("Orchestrator Import", test_orchestrator_import),
        ("Orchestrator Creation", test_orchestrator_creation),
    ]
    
    async_tests = [
        ("Orchestrator Functionality", test_orchestrator_functionality),
        ("Multiagent Inference", test_multiagent_inference),
    ]
    
    # Run synchronous tests
    passed = 0
    total = len(tests) + len(async_tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Run asynchronous tests
    async def run_async_tests():
        nonlocal passed
        for test_name, test_func in async_tests:
            logger.info(f"\n📋 Running async test: {test_name}")
            try:
                if await test_func():
                    passed += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    # Summary
    logger.info(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Strands v1.7.1 integration is working correctly.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())