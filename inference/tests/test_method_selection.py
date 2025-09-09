#!/usr/bin/env python3
"""
Test script for method selection functionality.
"""

import requests
import json
import time

# Test endpoint
BASE_URL = "http://production-alb-107602758.us-east-1.elb.amazonaws.com"

def test_method_selection():
    """Test different inference methods."""
    
    test_cases = [
        {
            "name": "Default (orchestrator)",
            "payload": {
                "product_name": "Samsung Galaxy S24 Ultra",
                "language_hint": "en"
            }
        },
        {
            "name": "Simple method",
            "payload": {
                "product_name": "Samsung Galaxy S24 Ultra", 
                "language_hint": "en",
                "method": "simple"
            }
        },
        {
            "name": "RAG method",
            "payload": {
                "product_name": "Samsung Galaxy S24 Ultra",
                "language_hint": "en", 
                "method": "rag"
            }
        },
        {
            "name": "Hybrid method",
            "payload": {
                "product_name": "Samsung Galaxy S24 Ultra",
                "language_hint": "en",
                "method": "hybrid"
            }
        },
        {
            "name": "Invalid method",
            "payload": {
                "product_name": "Samsung Galaxy S24 Ultra",
                "language_hint": "en",
                "method": "invalid"
            }
        }
    ]
    
    print("🧪 Testing Method Selection Functionality")
    print("=" * 50)
    
    # Test root endpoint first
    print("\n📋 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data.get('service', 'Unknown')}")
            print(f"Version: {data.get('version', 'Unknown')}")
            if 'inference_methods' in data:
                print("✅ Method documentation found!")
                for method, desc in data['inference_methods'].items():
                    print(f"  - {method}: {desc}")
            else:
                print("❌ Method documentation not found (old version)")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test inference methods
    print("\n🔍 Testing inference methods...")
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            response = requests.post(
                f"{BASE_URL}/infer",
                headers={"Content-Type": "application/json"},
                json=test_case['payload'],
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success!")
                
                # Check for method-specific response format
                method = test_case['payload'].get('method', 'orchestrator')
                print(f"Method used: {data.get('method', data.get('agent_used', 'unknown'))}")
                
                if 'brand_predictions' in data:
                    for pred in data['brand_predictions']:
                        print(f"Brand: {pred.get('brand', 'Unknown')} (confidence: {pred.get('confidence', 0):.3f})")
                
                if method == 'simple' and 'reasoning' in data:
                    print(f"Reasoning: {data['reasoning']}")
                elif method == 'rag' and 'similar_products' in data:
                    print(f"Similar products: {len(data['similar_products'])}")
                elif method == 'hybrid' and 'contributions' in data:
                    print(f"Contributions: {data['contributions']}")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error message: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Response: {response.text}")
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(1)  # Rate limiting
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

if __name__ == "__main__":
    test_method_selection()