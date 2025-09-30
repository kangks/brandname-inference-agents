# API Usage Guide

This guide provides comprehensive examples and documentation for using the multilingual product inference API with its advanced Strands-based multi-agent architecture.

## Base URL

```
Production: http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com
```

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Endpoints

### POST /infer

Main inference endpoint that accepts product names and returns brand predictions using advanced multi-agent coordination.

#### Request Format

```json
{
  "product_name": "Samsung Galaxy S24 Ultra โทรศัพท์มือถือ",
  "language_hint": "auto",
  "method": "orchestrator"
}
```

#### Parameters

- `product_name` (required): The product name to analyze
- `language_hint` (optional): Language hint for processing
  - `"en"`: English
  - `"th"`: Thai  
  - `"auto"`: Auto-detect (default)
- `method` (optional): Inference method to use
  - `"orchestrator"`: **Strands MultiAgent Orchestrator** - coordinates all agents (default, recommended)
  - `"finetuned"`: **Fine-tuned Nova Agent** - specialized Amazon Nova Pro model
  - `"simple"`: Pattern-based matching
  - `"ner"`: **Multilingual NER Agent** - enhanced entity recognition
  - `"rag"`: **Enhanced RAG Agent** - vector similarity search
  - `"llm"`: **Strands LLM Agent** - Amazon Nova Pro reasoning
  - `"hybrid"`: Combined NER → RAG → LLM pipeline

#### Response Format

**Orchestrator Response** (method: "orchestrator"):
```json
{
  "product_name": "Samsung Galaxy S24 Ultra โทรศัพท์มือถือ",
  "language": "auto",
  "method": "orchestrator",
  "agent_used": "orchestrator",
  "orchestrator_agents": ["finetuned", "ner", "rag", "llm", "hybrid"],
  "best_prediction": "Samsung",
  "best_confidence": 0.85,
  "best_method": "finetuned",
  "agent_results": {
    "finetuned_nova_agent_12345678": {
      "prediction": "Samsung",
      "confidence": 0.8,
      "method": "finetuned",
      "success": true
    },
    "ner_agent_87654321": {
      "prediction": "Samsung",
      "confidence": 0.85,
      "method": "ner",
      "success": true
    },
    "registry_rag": {
      "prediction": "Samsung",
      "confidence": 0.78,
      "method": "rag",
      "success": true
    }
  },
  "coordination_method": "enhanced",
  "strands_multiagent": true,
  "processing_time_ms": 1245,
  "timestamp": 1703123456.789
}
```

**Individual Agent Response** (method: "finetuned"):
```json
{
  "product_name": "Samsung Galaxy S24 Ultra โทรศัพท์มือถือ",
  "language": "auto",
  "method": "finetuned",
  "agent_used": "finetuned",
  "brand_predictions": [{
    "brand": "Samsung",
    "confidence": 0.8,
    "method": "finetuned"
  }],
  "model_info": {
    "deployment_arn": "arn:aws:bedrock:us-east-1:654654616949:custom-model-deployment/9o1i1v4ng8wy",
    "model_type": "fine-tuned-nova-pro",
    "specialization": "brand_extraction"
  },
  "processing_time_ms": 845,
  "timestamp": 1703123456.789
}
```

### GET /health

Health check endpoint to verify system status including Strands agent availability.

#### Response Format

```json
{
  "status": "healthy",
  "service": "multilingual-inference-orchestrator",
  "environment": "production",
  "orchestrator": "available",
  "strands_multiagent_available": true,
  "orchestrator_agents_count": 5,
  "individual_agents_count": 6,
  "available_methods": ["orchestrator", "finetuned", "simple", "ner", "rag", "llm", "hybrid"],
  "specialized_agents": {
    "finetuned_nova": "available",
    "strands_orchestrator": "available"
  },
  "custom_deployments": {
    "finetuned_nova_arn": "arn:aws:bedrock:us-east-1:654654616949:custom-model-deployment/9o1i1v4ng8wy"
  },
  "timestamp": 1703123456.789
}
```

### GET /

Service information endpoint with agent architecture details.

#### Response Format

```json
{
  "service": "Multilingual Product Inference API",
  "version": "2.0.0",
  "description": "Advanced AI-powered brand extraction using Strands multi-agent coordination",
  "architecture": "strands-multiagent",
  "available_methods": ["orchestrator", "finetuned", "simple", "ner", "rag", "llm", "hybrid"],
  "supported_languages": ["en", "th", "mixed"],
  "agent_types": {
    "orchestrator": "Strands MultiAgent Orchestrator with Swarm/Graph coordination",
    "finetuned": "Fine-tuned Amazon Nova Pro for brand extraction",
    "ner": "Multilingual NER with Thai-English support",
    "rag": "Enhanced RAG with vector similarity search",
    "llm": "Strands LLM Agent with specialized prompts",
    "hybrid": "Sequential pipeline with confidence weighting"
  },
  "coordination_methods": ["swarm", "graph", "enhanced"],
  "documentation": "/docs"
}
```

## Usage Examples

### Basic Usage - Orchestrator (Recommended)

```bash
# Use Strands MultiAgent Orchestrator for best results
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24 Ultra 256GB"
  }'
```

### Fine-tuned Agent Usage

```bash
# Use fine-tuned Nova agent for specialized brand extraction
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Dr.Althea Cream ด๊อกเตอร์อัลเทีย ครีมบำรุงผิวหน้า 50ml",
    "method": "finetuned"
  }'
```

### Method-Specific Usage

```bash
# Strands MultiAgent Orchestrator with enhanced coordination
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "CLEAR NOSE Moist Skin Barrier Moisturizing Gel 120ml เคลียร์โนส มอยส์เจอไรซิ่งเจล เฟเชียล.",
    "method": "orchestrator"
  }'

# Enhanced multilingual NER agent
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24 โทรศัพท์มือถือ",
    "method": "ner"
  }'

# Strands LLM agent with specialized prompts
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Eucerin Spotless Brightening Skin Tone Perfecting Body Lotion 250ml ยูเซอริน ผลิตภัณฑ์บำรุงผิวกาย",
    "method": "llm"
  }'
```

### Multilingual Examples

```bash
# Thai-English mixed content (auto-detection)
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24 โทรศัพท์มือถือ สมาร์ทโฟน",
    "language_hint": "auto",
    "method": "orchestrator"
  }'

# Complex multilingual product with fine-tuned agent
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "[โปรแรง]กันแดดเคลียร์โนส Clear Nose UV Sun Serum SPF50+PA++++ 80ml 1ชิ้น(CUV)",
    "language_hint": "auto",
    "method": "finetuned"
  }'
```

## Agent Comparison & Performance

### Latest Performance Results

Based on recent batch testing with real multilingual products:

| Method | Brand Accuracy | Confidence | Speed | Use Case |
|--------|----------------|------------|-------|----------|
| `orchestrator` | **Highest** | 0.7-0.85 | Slow (~1500ms) | **Production use** - best overall results |
| `finetuned` | **Excellent** | 0.7 | Fast (~800ms) | **Specific brands** - Clear, Dr.Althea, Eucerin |
| `simple` | Low | 0.5-0.7 | Fastest (~25ms) | Quick checks, fallback |
| `ner` | Generic | 0.9 | Fast (~150ms) | Entity extraction (often returns "name") |
| `rag` | Generic | 0.9 | Medium (~300ms) | Similarity matching (often returns "name") |
| `llm` | Generic | 0.85-1.0 | Slow (~1200ms) | Complex reasoning (often returns "name") |
| `hybrid` | Generic | 0.7 | Medium (~900ms) | Balanced approach (often returns "name") |

### Real Test Results

Recent batch comparison showing agent performance on multilingual products:

```
Product: "CLEAR NOSE Moist Skin Barrier Moisturizing Gel 120ml เคลียร์โนส มอยส์เจอไรซิ่งเจล เฟเชียล."
finetuned  → Clear     (0.7)  ✓ Correct specific brand
ner        → name      (0.9)  ✗ Generic classification
rag        → name      (0.9)  ✗ Generic classification
llm        → name      (0.85) ✗ Generic classification

Product: "Dr.Althea Cream ด๊อกเตอร์อัลเทีย ครีมบำรุงผิวหน้า 50ml"
finetuned  → Dr.Althea (0.7)  ✓ Correct specific brand
ner        → name      (0.9)  ✗ Generic classification
rag        → name      (0.9)  ✗ Generic classification
llm        → name      (1.0)  ✗ Generic classification

Product: "Eucerin Spotless Brightening Skin Tone Perfecting Body Lotion 250ml ยูเซอริน ผลิตภัณฑ์บำรุงผิวกาย"
finetuned  → Eucerin   (0.7)  ✓ Correct specific brand
ner        → name      (0.9)  ✗ Generic classification
rag        → name      (0.9)  ✗ Generic classification
llm        → name      (1.0)  ✗ Generic classification
```

### When to Use Each Method

#### Orchestrator (Recommended for Production)
- **Best for**: Production applications requiring highest accuracy
- **Architecture**: Strands MultiAgent coordination with Swarm/Graph patterns
- **Pros**: Uses all agents, intelligent result selection, handles edge cases
- **Cons**: Slower response time
- **Example**: E-commerce product categorization, content management systems

#### Fine-tuned (Best for Specific Brands)
- **Best for**: Applications needing specific brand identification
- **Architecture**: Custom Amazon Nova Pro deployment with domain-specific training
- **Pros**: Excellent accuracy for trained brands, faster than orchestrator
- **Cons**: Limited to trained brand patterns
- **Example**: Brand monitoring, product catalog validation

#### Simple
- **Best for**: Quick checks, high-volume processing, fallback scenarios
- **Pros**: Very fast, no external dependencies
- **Cons**: Limited accuracy, pattern-based only
- **Example**: Initial filtering, health checks

#### NER/RAG/LLM/Hybrid
- **Best for**: Specific use cases requiring entity extraction or similarity matching
- **Pros**: Good for structured processing, fast to medium speed
- **Cons**: Often returns generic classifications ("name") rather than specific brands
- **Example**: Entity extraction pipelines, similarity analysis

## Advanced Features

### Strands MultiAgent Coordination

The orchestrator uses advanced coordination patterns:

```bash
# Request with specific coordination method (if supported)
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24",
    "method": "orchestrator",
    "coordination_method": "swarm"
  }'
```

**Coordination Methods**:
- **Swarm**: Parallel execution using `Strands.multiagent.Swarm`
- **Graph**: Structured workflows using `Strands.multiagent.GraphBuilder`
- **Enhanced**: Priority-based selection with fine-tuned agent preference

### Custom Deployment Information

The fine-tuned agent uses a specialized deployment:

```json
{
  "model_info": {
    "deployment_arn": "arn:aws:bedrock:us-east-1:654654616949:custom-model-deployment/9o1i1v4ng8wy",
    "model_type": "fine-tuned-nova-pro",
    "specialization": "brand_extraction",
    "training_domain": "multilingual_product_titles",
    "supported_languages": ["en", "th", "mixed"]
  }
}
```

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "error": "Invalid request format",
  "message": "Missing required field: product_name",
  "timestamp": 1703123456.789
}
```

#### 422 Unprocessable Entity
```json
{
  "error": "Invalid method",
  "message": "Method 'invalid_method' not supported. Available methods: ['orchestrator', 'finetuned', 'simple', 'ner', 'rag', 'llm', 'hybrid']",
  "timestamp": 1703123456.789
}
```

#### 503 Service Unavailable - Agent Failure
```json
{
  "error": "Agent unavailable",
  "message": "Fine-tuned Nova agent initialization failed",
  "details": {
    "agent": "finetuned",
    "reason": "Custom deployment ARN access denied",
    "fallback_available": true
  },
  "timestamp": 1703123456.789
}
```

### Error Handling Best Practices

```python
import requests
import json
import time

def call_inference_api(product_name, method="orchestrator", max_retries=3):
    """
    Call inference API with comprehensive error handling for multi-agent system.
    """
    url = "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer"
    
    payload = {
        "product_name": product_name,
        "method": method,
        "language_hint": "auto"
    }
    
    # Method-specific timeouts
    timeouts = {
        "simple": 10,
        "ner": 15,
        "rag": 20,
        "llm": 35,
        "finetuned": 25,
        "hybrid": 30,
        "orchestrator": 40
    }
    
    timeout = timeouts.get(method, 35)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate result based on method
                if method == "orchestrator":
                    if "best_prediction" in result and result["best_prediction"] != "Unknown":
                        return result
                    elif "agent_results" in result:
                        # Check if any agent succeeded
                        successful_agents = [
                            r for r in result["agent_results"].values()
                            if r.get("success") and r.get("prediction") != "Unknown"
                        ]
                        if successful_agents:
                            return result
                
                elif method == "finetuned":
                    if "brand_predictions" in result and result["brand_predictions"]:
                        prediction = result["brand_predictions"][0]
                        if prediction.get("brand") != "Unknown":
                            return result
                
                # If no good results, try fallback method
                if attempt == 0 and method != "simple":
                    print(f"No good results from {method}, trying simple method")
                    payload["method"] = "simple"
                    timeout = timeouts["simple"]
                    continue
                
                return result
                
            elif response.status_code == 422:
                # Invalid method, don't retry
                print(f"Invalid method: {method}")
                return None
                
            elif response.status_code == 503:
                # Service unavailable, try fallback method
                error_data = response.json()
                if error_data.get("details", {}).get("fallback_available"):
                    print(f"Agent {method} unavailable, trying orchestrator")
                    payload["method"] = "orchestrator"
                    timeout = timeouts["orchestrator"]
                    continue
                
            elif response.status_code >= 500:
                # Server error, retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Server error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Server error after {max_retries} attempts")
                    return None
            else:
                print(f"Unexpected status code: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Request timeout (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    return None

# Usage examples
result = call_inference_api("Samsung Galaxy S24", "orchestrator")
if result and result.get("best_prediction"):
    print(f"Brand: {result['best_prediction']}")
    print(f"Confidence: {result['best_confidence']}")
    print(f"Method used: {result['best_method']}")

# Try fine-tuned agent first, fallback to orchestrator
result = call_inference_api("Dr.Althea Cream ด๊อกเตอร์อัลเทีย", "finetuned")
if result:
    if result.get("brand_predictions"):
        prediction = result["brand_predictions"][0]
        print(f"Brand: {prediction['brand']}")
        print(f"Confidence: {prediction['confidence']}")
```

## Rate Limiting & Performance

### Recommended Usage Patterns

- **Production**: Max 5 requests per second per client
- **Development**: Max 10 requests per second per client  
- **Burst**: Up to 25 requests in a 10-second window
- **Concurrent**: Max 3 concurrent requests per client

### Performance Optimization

```python
# Method selection for different scenarios
def select_optimal_method(use_case, performance_priority):
    """Select optimal method based on use case and performance requirements."""
    
    if performance_priority == "accuracy":
        return "orchestrator"  # Best accuracy, slower
    elif performance_priority == "speed":
        return "simple"        # Fastest, lower accuracy
    elif performance_priority == "balanced":
        if use_case == "brand_monitoring":
            return "finetuned"  # Good accuracy for known brands, faster
        else:
            return "hybrid"     # Balanced approach
    elif performance_priority == "specific_brands":
        return "finetuned"     # Best for trained brand patterns
    
    return "orchestrator"      # Default to best accuracy

# Batch processing optimization
def process_products_batch(products, batch_size=5):
    """Process products in batches with optimal method selection."""
    results = []
    
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]
        
        # Use different methods based on product characteristics
        for product in batch:
            if len(product) < 20:  # Short product names
                method = "simple"
            elif any(brand in product.lower() for brand in ["samsung", "apple", "sony"]):
                method = "finetuned"  # Known brands
            else:
                method = "orchestrator"  # Complex cases
            
            result = call_inference_api(product, method)
            results.append(result)
            
            # Small delay to respect rate limits
            time.sleep(0.2)
    
    return results
```

## Integration Examples

### Python Integration with Multi-Agent Support

```python
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class BrandPrediction:
    """Brand prediction result."""
    brand: str
    confidence: float
    method: str
    reliable: bool

class InferenceClient:
    """Client for multilingual product inference API with multi-agent support."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def infer_brand(
        self, 
        product_name: str, 
        method: str = "orchestrator",
        language_hint: str = "auto",
        prefer_specific_brands: bool = True
    ) -> Optional[BrandPrediction]:
        """
        Infer brand from product name with intelligent method selection.
        
        Args:
            product_name: Product name to analyze
            method: Inference method to use
            language_hint: Language hint for processing
            prefer_specific_brands: Whether to prefer specific brand names over generic classifications
            
        Returns:
            BrandPrediction object or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/infer",
                json={
                    "product_name": product_name,
                    "method": method,
                    "language_hint": language_hint
                },
                timeout=40 if method == "orchestrator" else 30
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_result(result, prefer_specific_brands)
            else:
                print(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def _parse_result(self, result: Dict[str, Any], prefer_specific_brands: bool) -> Optional[BrandPrediction]:
        """Parse API result into BrandPrediction object."""
        
        if result.get("method") == "orchestrator":
            # Orchestrator result - use best prediction
            best_brand = result.get("best_prediction")
            best_confidence = result.get("best_confidence", 0.0)
            best_method = result.get("best_method", "unknown")
            
            # If preferring specific brands, check for fine-tuned results
            if prefer_specific_brands and "agent_results" in result:
                finetuned_results = [
                    r for agent_id, r in result["agent_results"].items()
                    if "finetuned" in agent_id and r.get("success") and r.get("prediction") != "Unknown"
                ]
                
                if finetuned_results:
                    finetuned_result = finetuned_results[0]
                    return BrandPrediction(
                        brand=finetuned_result["prediction"],
                        confidence=finetuned_result["confidence"],
                        method="finetuned",
                        reliable=finetuned_result["confidence"] >= 0.6
                    )
            
            return BrandPrediction(
                brand=best_brand,
                confidence=best_confidence,
                method=best_method,
                reliable=best_confidence >= 0.7
            )
            
        else:
            # Individual agent result
            predictions = result.get("brand_predictions", [])
            if predictions:
                prediction = predictions[0]
                brand = prediction.get("brand", "Unknown")
                confidence = prediction.get("confidence", 0.0)
                
                # Filter out generic "name" classifications if preferring specific brands
                if prefer_specific_brands and brand.lower() in ["name", "unknown", "generic"]:
                    return BrandPrediction(
                        brand="Unknown",
                        confidence=0.0,
                        method=result.get("method", "unknown"),
                        reliable=False
                    )
                
                return BrandPrediction(
                    brand=brand,
                    confidence=confidence,
                    method=result.get("method", "unknown"),
                    reliable=confidence >= 0.7 and brand != "Unknown"
                )
        
        return None
    
    def infer_with_fallback(self, product_name: str) -> Optional[BrandPrediction]:
        """
        Infer brand with intelligent fallback strategy.
        
        1. Try fine-tuned agent first (fast, accurate for known brands)
        2. If no specific brand found, try orchestrator (comprehensive)
        3. If still no result, try simple method (fast fallback)
        """
        
        # Try fine-tuned agent first
        result = self.infer_brand(product_name, "finetuned", prefer_specific_brands=True)
        if result and result.reliable and result.brand != "Unknown":
            return result
        
        # Try orchestrator for comprehensive analysis
        result = self.infer_brand(product_name, "orchestrator", prefer_specific_brands=True)
        if result and result.reliable:
            return result
        
        # Final fallback to simple method
        result = self.infer_brand(product_name, "simple", prefer_specific_brands=False)
        return result
    
    def batch_infer(self, product_names: List[str]) -> List[Optional[BrandPrediction]]:
        """Process multiple products with rate limiting."""
        results = []
        
        for product_name in product_names:
            result = self.infer_with_fallback(product_name)
            results.append(result)
            
            # Rate limiting
            time.sleep(0.2)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Get detailed health information."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Usage Examples
client = InferenceClient("http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com")

# Check system health
health = client.health_check()
print(f"System status: {health.get('status')}")
print(f"Available methods: {health.get('available_methods', [])}")

# Single product inference with fallback
result = client.infer_with_fallback("Dr.Althea Cream ด๊อกเตอร์อัลเทีย ครีมบำรุงผิวหน้า 50ml")
if result:
    print(f"Brand: {result.brand}")
    print(f"Confidence: {result.confidence}")
    print(f"Method: {result.method}")
    print(f"Reliable: {result.reliable}")

# Batch processing
products = [
    "Samsung Galaxy S24 โทรศัพท์มือถือ",
    "CLEAR NOSE Moist Skin Barrier Moisturizing Gel 120ml เคลียร์โนส",
    "Eucerin Spotless Brightening Skin Tone Perfecting Body Lotion 250ml ยูเซอริน"
]

results = client.batch_infer(products)
for i, result in enumerate(results):
    if result:
        print(f"{products[i]} → {result.brand} ({result.confidence:.2f}, {result.method})")
```

This comprehensive API usage guide provides detailed documentation for integrating with the advanced multi-agent inference system, including the latest Strands-based architecture, fine-tuned Nova agent capabilities, and real-world performance characteristics.