#!/usr/bin/env python3
"""
Validation script for orchestrator deployment without standalone agent.
"""

import json
import requests
import sys
import time

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Health check successful")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Service: {health_data.get('service')}")
        print(f"   Orchestrator: {health_data.get('orchestrator')}")
        print(f"   Agents count: {health_data.get('agents_count')}")
        
        # Verify standalone_agent is NOT in the response
        if 'standalone_agent' in health_data:
            print(f"❌ ERROR: standalone_agent still present in health response")
            return False
        else:
            print(f"✅ Confirmed: standalone_agent removed from health response")
        
        return health_data.get('status') == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_inference_endpoint(base_url):
    """Test the inference endpoint."""
    print("\n🔍 Testing inference endpoint...")
    
    test_cases = [
        {
            "product_name": "iPhone 15 Pro Max",
            "language_hint": "en"
        },
        {
            "product_name": "Samsung Galaxy S24 Ultra",
            "language_hint": "en"
        },
        {
            "product_name": "ยาสีฟัน Colgate Total",
            "language_hint": "th"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['product_name']}")
        
        try:
            response = requests.post(
                f"{base_url}/infer",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check if we get a valid response
            if 'status' in result:
                print(f"   ✅ Response received")
                print(f"      Status: {result.get('status')}")
                
                # Check for agent_used field
                if 'agent_used' in result:
                    agent_used = result.get('agent_used')
                    print(f"      Agent used: {agent_used}")
                    
                    # Verify it's orchestrator, not standalone
                    if agent_used == 'orchestrator':
                        print(f"      ✅ Confirmed: Using orchestrator agent")
                    elif agent_used == 'standalone':
                        print(f"      ❌ ERROR: Still using standalone agent")
                        return False
                    else:
                        print(f"      ℹ️  Agent used: {agent_used}")
                
                # Check orchestrator status
                if 'orchestrator_status' in result:
                    print(f"      Orchestrator status: {result.get('orchestrator_status')}")
                
                # Check available agents
                if 'available_agents' in result:
                    agents = result.get('available_agents', [])
                    print(f"      Available agents: {', '.join(agents) if agents else 'None'}")
                
                # Check registered agents count
                if 'registered_agents' in result:
                    count = result.get('registered_agents', 0)
                    print(f"      Registered agents: {count}")
                
            else:
                print(f"   ❌ Invalid response format")
                return False
                
        except Exception as e:
            print(f"   ❌ Inference test failed: {str(e)}")
            return False
    
    return True

def main():
    """Main validation function."""
    base_url = "http://production-alb-107602758.us-east-1.elb.amazonaws.com"
    
    print("🚀 Validating Orchestrator Deployment (Standalone Agent Removed)")
    print(f"   Target: {base_url}")
    print("=" * 60)
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    
    if not health_ok:
        print("\n❌ Health check failed - aborting validation")
        sys.exit(1)
    
    # Test inference endpoint
    inference_ok = test_inference_endpoint(base_url)
    
    print("\n" + "=" * 60)
    
    if health_ok and inference_ok:
        print("✅ VALIDATION SUCCESSFUL")
        print("   - Standalone agent successfully removed")
        print("   - Orchestrator agent is working")
        print("   - Health endpoint no longer shows standalone_agent")
        print("   - Inference endpoint is responding")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()