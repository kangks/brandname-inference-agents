#!/usr/bin/env python3
"""
Test script to get actual inference response from orchestrator.
"""

import json
import requests
import sys
import time

def test_inference_with_retry(base_url, max_retries=3):
    """Test inference with retry logic."""
    
    test_cases = [
        {
            "product_name": "Apple iPhone 15 Pro",
            "language_hint": "en"
        },
        {
            "product_name": "Nike Air Jordan",
            "language_hint": "en"
        },
        {
            "product_name": "Samsung Galaxy",
            "language_hint": "en"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['product_name']}")
        
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries}...")
                
                response = requests.post(
                    f"{base_url}/infer",
                    json=test_case,
                    headers={"Content-Type": "application/json"},
                    timeout=45  # Longer timeout
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Print the full response for analysis
                print(f"   📋 Full response:")
                print(json.dumps(result, indent=2))
                
                # Check if we got actual inference results
                if 'agent_used' in result:
                    print(f"\n   ✅ SUCCESS: Got inference response with agent_used = {result['agent_used']}")
                    return True
                elif 'status' in result and result['status'] == 'ready':
                    print(f"   ℹ️  Service ready but no inference performed")
                    print(f"      Available agents: {result.get('available_agents', [])}")
                    print(f"      Message: {result.get('message', 'No message')}")
                else:
                    print(f"   ⚠️  Unexpected response format")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    print(f"   ⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
    
    return False

def main():
    """Main test function."""
    base_url = "http://production-alb-107602758.us-east-1.elb.amazonaws.com"
    
    print("🔬 Testing Orchestrator Inference Response")
    print(f"   Target: {base_url}")
    print("=" * 50)
    
    # First check health
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        health_data = health_response.json()
        print(f"📊 Health Status:")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Agents: {health_data.get('agents_count')}")
        print(f"   Orchestrator: {health_data.get('orchestrator')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test inference
    success = test_inference_with_retry(base_url)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Successfully got inference response with agent_used field!")
    else:
        print("ℹ️  Orchestrator is running but agents may need configuration")
        print("   This is expected if dependencies are not fully available")

if __name__ == "__main__":
    main()