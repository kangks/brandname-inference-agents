#!/usr/bin/env python3
"""
Test script specifically for NER and LLM methods.
"""

import requests
import json

BASE_URL = "http://production-alb-107602758.us-east-1.elb.amazonaws.com"

def test_ner_llm_methods():
    """Test NER and LLM methods specifically."""
    
    test_products = [
        "Samsung Galaxy S24 Ultra",
        "iPhone 15 Pro Max", 
        "Nike Air Jordan 1",
        "Toyota Camry 2024",
        "โค้ก เซโร่"
    ]
    
    methods = ["ner", "llm"]
    
    print("🧪 Testing NER and LLM Methods")
    print("=" * 50)
    
    for product in test_products:
        print(f"\n📱 Product: {product}")
        print("-" * 30)
        
        for method in methods:
            try:
                response = requests.post(
                    f"{BASE_URL}/infer",
                    headers={"Content-Type": "application/json"},
                    json={
                        "product_name": product,
                        "language_hint": "auto",
                        "method": method
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get('brand_predictions', [])
                    
                    if predictions:
                        pred = predictions[0]
                        brand = pred.get('brand', 'Unknown')
                        confidence = pred.get('confidence', 0.0)
                        
                        print(f"  {method.upper()}: ✅ {brand} ({confidence:.3f})")
                        
                        # Show method-specific details
                        if method == "ner" and 'entities' in data:
                            entities = data['entities']
                            print(f"    Entities: {len(entities)}")
                            for entity in entities[:3]:  # Show first 3
                                print(f"      - {entity['text']} ({entity['type']}, {entity['confidence']:.3f})")
                        
                        elif method == "llm" and 'reasoning' in data:
                            reasoning = data['reasoning'][:100] + "..." if len(data['reasoning']) > 100 else data['reasoning']
                            print(f"    Reasoning: {reasoning}")
                    else:
                        print(f"  {method.upper()}: ❌ No predictions")
                        
                else:
                    print(f"  {method.upper()}: ❌ HTTP {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"    Error: {error_data.get('error', 'Unknown')}")
                    except:
                        pass
                        
            except Exception as e:
                print(f"  {method.upper()}: ❌ Error: {str(e)}")
    
    print(f"\n" + "=" * 50)
    print("✅ NER and LLM testing completed!")

if __name__ == "__main__":
    test_ner_llm_methods()