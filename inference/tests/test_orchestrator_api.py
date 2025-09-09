#!/usr/bin/env python3
"""
Orchestrator API Test Script

This script tests the orchestrator through the API endpoint to show how
it coordinates all sub-agents and returns aggregated results.

Usage:
    python test_orchestrator_api.py
    python test_orchestrator_api.py --product "iPhone 15"
    python test_orchestrator_api.py --endpoint http://localhost:8080
"""

import requests
import json
import argparse
import time
from typing import Dict, Any, List


class OrchestratorAPITester:
    """Test the orchestrator through API endpoints."""
    
    def __init__(self, base_url: str):
        """Initialize with API base URL."""
        self.base_url = base_url.rstrip('/')
        
        # Test products
        self.test_products = [
            {"name": "Samsung Galaxy S24 Ultra", "expected": "Samsung"},
            {"name": "iPhone 15 Pro Max", "expected": "Apple"},
            {"name": "Nike Air Jordan 1", "expected": "Nike"},
            {"name": "Sony PlayStation 5", "expected": "Sony"},
            {"name": "Microsoft Surface Pro", "expected": "Microsoft"},
            {"name": "Google Pixel 8", "expected": "Google"},
        ]
    
    def check_api_health(self) -> bool:
        """Check if the API is healthy and orchestrator is available."""
        try:
            print("🏥 Checking API Health...")
            
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ API Status: {data.get('status', 'unknown')}")
                print(f"🔧 Service: {data.get('service', 'unknown')}")
                print(f"🤖 Orchestrator: {data.get('orchestrator', 'unknown')}")
                print(f"👥 Agents Count: {data.get('agents_count', 0)}")
                
                # Check if orchestrator is available
                orchestrator_available = data.get('orchestrator') == 'available'
                agents_count = data.get('agents_count', 0)
                
                if orchestrator_available and agents_count > 0:
                    print(f"🎯 Orchestrator is ready with {agents_count} agents")
                    return True
                else:
                    print(f"⚠️  Orchestrator not fully ready")
                    return False
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_orchestrator_inference(self, product_name: str, language: str = "en") -> Dict[str, Any]:
        """
        Test orchestrator inference through API.
        
        Args:
            product_name: Product to test
            language: Language hint
            
        Returns:
            Test result dictionary
        """
        try:
            print(f"\n🧪 Testing Orchestrator with: {product_name}")
            print("-" * 50)
            
            # Prepare request payload
            payload = {
                "product_name": product_name,
                "language_hint": language,
                "method": "orchestrator"  # Explicitly request orchestrator
            }
            
            start_time = time.time()
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/infer",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract orchestrator information
                agent_used = data.get('agent_used', 'unknown')
                orchestrator_agents = data.get('orchestrator_agents', [])
                agent_results = data.get('agent_results', {})
                
                print(f"✅ Orchestrator Response:")
                print(f"   Agent Used: {agent_used}")
                print(f"   Request Time: {request_time:.3f}s")
                print(f"   Processing Time: {data.get('processing_time_ms', 0)}ms")
                
                # Show brand predictions
                predictions = data.get('brand_predictions', [])
                if predictions:
                    best_pred = predictions[0]
                    print(f"   Best Prediction: {best_pred.get('brand', 'Unknown')}")
                    print(f"   Confidence: {best_pred.get('confidence', 0.0):.3f}")
                    print(f"   Method: {best_pred.get('method', 'unknown')}")
                
                # Show orchestrator coordination info
                if orchestrator_agents:
                    print(f"\n🤖 Orchestrator Coordination:")
                    print(f"   Available Agents: {', '.join(orchestrator_agents)}")
                
                # Show agent results if available
                if agent_results:
                    print(f"   Agent Results:")
                    for agent_name, result in agent_results.items():
                        if isinstance(result, dict):
                            success = result.get('success', False)
                            prediction = result.get('prediction', 'N/A')
                            confidence = result.get('confidence', 0.0)
                            processing_time = result.get('processing_time', 0.0)
                            error = result.get('error')
                            
                            if success:
                                print(f"     - {agent_name.upper()}: ✅ {prediction} ({confidence:.3f}, {processing_time:.3f}s)")
                            else:
                                print(f"     - {agent_name.upper()}: ❌ Failed - {error}")
                        else:
                            # Legacy format (boolean)
                            if result:
                                print(f"     - {agent_name.upper()}: ✅ Success")
                            else:
                                print(f"     - {agent_name.upper()}: ❌ Failed")
                else:
                    print(f"   Agent Results: Not available")
                
                # Show entities if available
                entities = data.get('entities', [])
                if entities:
                    print(f"\n🏷️  Extracted Entities:")
                    for entity in entities[:3]:  # Show first 3
                        print(f"   - {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')}, {entity.get('confidence', 0.0):.3f})")
                
                return {
                    "success": True,
                    "product_name": product_name,
                    "agent_used": agent_used,
                    "orchestrator_agents": orchestrator_agents,
                    "predictions": predictions,
                    "entities": entities,
                    "request_time": request_time,
                    "processing_time": data.get('processing_time_ms', 0)
                }
                
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Unknown error')
                except:
                    error_msg = response.text
                
                print(f"❌ API Error ({response.status_code}): {error_msg}")
                
                return {
                    "success": False,
                    "product_name": product_name,
                    "error": error_msg,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            print(f"❌ Request Error: {e}")
            return {
                "success": False,
                "product_name": product_name,
                "error": str(e)
            }
    
    def test_all_products(self) -> Dict[str, Any]:
        """Test orchestrator with all test products."""
        print(f"\n🚀 Testing Orchestrator with Multiple Products")
        print("=" * 60)
        
        results = []
        successful_tests = 0
        total_request_time = 0
        total_processing_time = 0
        
        for i, product in enumerate(self.test_products, 1):
            print(f"\n[{i}/{len(self.test_products)}]", end=" ")
            
            result = self.test_orchestrator_inference(product['name'])
            results.append(result)
            
            if result['success']:
                successful_tests += 1
                total_request_time += result.get('request_time', 0)
                total_processing_time += result.get('processing_time', 0)
                
                # Check if prediction matches expected
                predictions = result.get('predictions', [])
                if predictions:
                    predicted_brand = predictions[0].get('brand', '').lower()
                    expected_brand = product['expected'].lower()
                    
                    if expected_brand in predicted_brand or predicted_brand in expected_brand:
                        print(f"   🎯 Correct: {predictions[0].get('brand', 'Unknown')}")
                    else:
                        print(f"   ⚠️  Different: {predictions[0].get('brand', 'Unknown')} (expected: {product['expected']})")
                else:
                    print(f"   ❌ No predictions")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between requests
            time.sleep(0.5)
        
        # Summary
        print(f"\n📊 Orchestrator Test Summary")
        print("=" * 40)
        print(f"Total Tests: {len(self.test_products)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {len(self.test_products) - successful_tests}")
        print(f"Success Rate: {(successful_tests / len(self.test_products)) * 100:.1f}%")
        
        if successful_tests > 0:
            avg_request_time = total_request_time / successful_tests
            avg_processing_time = total_processing_time / successful_tests
            print(f"Avg Request Time: {avg_request_time:.3f}s")
            print(f"Avg Processing Time: {avg_processing_time:.1f}ms")
        
        # Check orchestrator usage
        orchestrator_usage = sum(1 for r in results if r.get('success') and r.get('agent_used') == 'orchestrator')
        print(f"Orchestrator Usage: {orchestrator_usage}/{successful_tests} ({(orchestrator_usage/successful_tests)*100:.1f}%)" if successful_tests > 0 else "Orchestrator Usage: 0%")
        
        return {
            "total_tests": len(self.test_products),
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / len(self.test_products)) * 100,
            "orchestrator_usage": orchestrator_usage,
            "results": results
        }
    
    def show_orchestrator_info(self) -> None:
        """Show detailed orchestrator information."""
        try:
            print(f"\n📋 Orchestrator Information")
            print("-" * 35)
            
            # Get service info
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"Service: {data.get('service', 'Unknown')}")
                print(f"Version: {data.get('version', 'Unknown')}")
                print(f"Environment: {data.get('environment', 'Unknown')}")
                
                # Show available inference methods
                methods = data.get('inference_methods', {})
                if methods:
                    print(f"\nAvailable Methods:")
                    for method, info in methods.items():
                        status = "✅" if info.get('available', False) else "❌"
                        print(f"   {method}: {status} {info.get('description', '')}")
                
                # Show orchestrator-specific info
                orchestrator_info = methods.get('orchestrator', {})
                if orchestrator_info:
                    print(f"\nOrchestrator Details:")
                    print(f"   Available: {'✅' if orchestrator_info.get('available', False) else '❌'}")
                    print(f"   Description: {orchestrator_info.get('description', 'N/A')}")
                    
                    sub_agents = orchestrator_info.get('sub_agents', [])
                    if sub_agents:
                        print(f"   Sub-Agents: {', '.join(sub_agents)}")
            
        except Exception as e:
            print(f"❌ Failed to get orchestrator info: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Orchestrator through API")
    parser.add_argument("--endpoint", "-e", 
                       default="http://production-alb-107602758.us-east-1.elb.amazonaws.com",
                       help="API endpoint URL")
    parser.add_argument("--product", "-p", help="Single product to test")
    parser.add_argument("--info", action="store_true", help="Show orchestrator info only")
    
    args = parser.parse_args()
    
    # Create tester
    tester = OrchestratorAPITester(args.endpoint)
    
    print(f"🎯 Orchestrator API Test")
    print(f"Endpoint: {args.endpoint}")
    print("=" * 50)
    
    try:
        # Check API health first
        if not tester.check_api_health():
            print("❌ API health check failed - cannot proceed")
            return 1
        
        # Show orchestrator info if requested
        if args.info:
            tester.show_orchestrator_info()
            return 0
        
        # Test single product or all products
        if args.product:
            result = tester.test_orchestrator_inference(args.product)
            if result['success']:
                print(f"\n✅ Orchestrator test completed successfully!")
                return 0
            else:
                print(f"\n❌ Orchestrator test failed!")
                return 1
        else:
            summary = tester.test_all_products()
            
            if summary['success_rate'] >= 80:
                print(f"\n🎉 Orchestrator tests completed with {summary['success_rate']:.1f}% success rate!")
                return 0
            else:
                print(f"\n⚠️  Orchestrator tests completed with {summary['success_rate']:.1f}% success rate")
                return 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)