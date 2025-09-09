#!/usr/bin/env python3
"""
Demo script showing the method selection functionality.
"""

import requests
import json

BASE_URL = "http://production-alb-107602758.us-east-1.elb.amazonaws.com"

def demo_method_selection():
    """Demonstrate the method selection functionality."""
    
    print("🚀 Multilingual Product Inference - Method Selection Demo")
    print("=" * 60)
    
    # Show API documentation
    print("\n📋 API Documentation:")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data['service']}")
            print(f"Version: {data['version']}")
            
            print("\n🔧 Available Methods:")
            for method, desc in data.get('inference_methods', {}).items():
                print(f"  • {method}: {desc}")
            
            print("\n📝 Request Format:")
            req_format = data.get('request_format', {})
            for field, desc in req_format.items():
                print(f"  • {field}: {desc}")
    except Exception as e:
        print(f"❌ Error getting API docs: {e}")
    
    # Demo different methods
    print("\n" + "=" * 60)
    print("🧪 Method Demonstrations")
    print("=" * 60)
    
    test_product = "Samsung Galaxy S24 Ultra"
    
    demos = [
        {
            "name": "Default (Orchestrator)",
            "payload": {"product_name": test_product, "language_hint": "en"},
            "description": "Uses all available agents and selects the best result"
        },
        {
            "name": "Simple Method",
            "payload": {"product_name": test_product, "language_hint": "en", "method": "simple"},
            "description": "Basic pattern matching without external dependencies"
        },
        {
            "name": "RAG Method", 
            "payload": {"product_name": test_product, "language_hint": "en", "method": "rag"},
            "description": "Vector similarity search (requires training data)"
        },
        {
            "name": "Hybrid Method",
            "payload": {"product_name": test_product, "language_hint": "en", "method": "hybrid"},
            "description": "Sequential pipeline combining multiple approaches"
        },
        {
            "name": "Invalid Method",
            "payload": {"product_name": test_product, "language_hint": "en", "method": "invalid"},
            "description": "Shows error handling for invalid methods"
        }
    ]
    
    for demo in demos:
        print(f"\n🔍 {demo['name']}")
        print(f"📖 {demo['description']}")
        print(f"📤 Request: {json.dumps(demo['payload'])}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/infer",
                headers={"Content-Type": "application/json"},
                json=demo['payload'],
                timeout=30
            )
            
            print(f"📥 Response ({response.status_code}):")
            
            if response.status_code == 200:
                data = response.json()
                
                # Show key results
                if 'brand_predictions' in data:
                    for pred in data['brand_predictions']:
                        print(f"  ✅ Brand: {pred.get('brand', 'Unknown')} (confidence: {pred.get('confidence', 0):.3f})")
                
                # Show method-specific details
                method = data.get('method', 'unknown')
                if method == 'simple' and 'reasoning' in data:
                    print(f"  📝 Reasoning: {data['reasoning']}")
                elif method == 'rag':
                    similar_count = len(data.get('similar_products', []))
                    print(f"  🔗 Similar products found: {similar_count}")
                    print(f"  🤖 Embedding model: {data.get('embedding_model', 'Unknown')}")
                elif method == 'hybrid':
                    contrib = data.get('contributions', {})
                    print(f"  🔄 Pipeline contributions:")
                    print(f"    - NER: {contrib.get('ner', 0):.2f}")
                    print(f"    - RAG: {contrib.get('rag', 0):.2f}")
                    print(f"    - LLM: {contrib.get('llm', 0):.2f}")
                
                print(f"  ⏱️  Processing time: {data.get('processing_time_ms', 0)}ms")
                
            else:
                # Show error details
                try:
                    error_data = response.json()
                    print(f"  ❌ Error: {error_data.get('error', 'Unknown error')}")
                    if 'available_agents' in error_data:
                        print(f"  🔧 Available agents: {error_data['available_agents']}")
                except:
                    print(f"  ❌ Error: {response.text}")
                    
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    # Multilingual demo
    print(f"\n" + "=" * 60)
    print("🌍 Multilingual Support Demo")
    print("=" * 60)
    
    multilingual_tests = [
        {"product": "iPhone 15 Pro Max", "lang": "en", "desc": "English"},
        {"product": "โค้ก เซโร่", "lang": "th", "desc": "Thai"},
        {"product": "Samsung Galaxy S24 Ultra", "lang": "mixed", "desc": "Mixed language hint"}
    ]
    
    for test in multilingual_tests:
        print(f"\n🌐 {test['desc']} Test:")
        print(f"📱 Product: {test['product']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/infer",
                headers={"Content-Type": "application/json"},
                json={
                    "product_name": test['product'],
                    "language_hint": test['lang'],
                    "method": "simple"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('brand_predictions', [])
                if predictions:
                    pred = predictions[0]
                    print(f"  ✅ Detected brand: {pred.get('brand', 'Unknown')} (confidence: {pred.get('confidence', 0):.3f})")
                else:
                    print(f"  ❌ No brand detected")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\n💡 Key Features Demonstrated:")
    print("  • Method selection via 'method' parameter")
    print("  • Error handling for invalid methods")
    print("  • Different response formats per method")
    print("  • Multilingual support")
    print("  • Real-time inference with various confidence levels")
    print("\n🔗 Try it yourself:")
    print(f"  curl -X POST {BASE_URL}/infer \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"product_name\": \"Your Product\", \"method\": \"simple\"}'")

if __name__ == "__main__":
    demo_method_selection()