# API Usage Guide

## Quick Reference

### Base URL
```
http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information and available methods |
| `/health` | GET | Health check and system status |
| `/infer` | POST | Main inference endpoint |

## Inference Methods

### Method Overview

| Method | Description | Use Case |
|--------|-------------|----------|
| `orchestrator` | Coordinates all agents, returns best result | Production use, highest accuracy |
| `simple` | Pattern-based matching | Fast, no dependencies |
| `ner` | Named Entity Recognition | Extract entities from text |
| `rag` | Retrieval Augmented Generation | Similarity-based matching |
| `llm` | Large Language Model | Complex reasoning |
| `hybrid` | Sequential NER → RAG → LLM pipeline | Balanced approach |

## API Examples

### 1. Orchestrator Method (Recommended)

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
    "language_hint": "en",
    "method": "orchestrator"
  }'
```

**Response:**
```json
{
  "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
  "language": "en",
  "method": "orchestrator",
  "agent_used": "orchestrator",
  "orchestrator_agents": ["ner", "rag", "llm", "hybrid"],
  "best_prediction": "Alectric",
  "best_confidence": 0.85,
  "best_method": "ner",
  "agent_results": {
    "ner": {"prediction": "Alectric", "confidence": 0.85},
    "rag": {"prediction": "Alectric", "confidence": 0.78},
    "llm": {"prediction": "Alectric", "confidence": 0.82}
  },
  "timestamp": 1703123456.789
}
```

### 2. NER Agent

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24 Ultra 256GB",
    "language_hint": "en",
    "method": "ner"
  }'
```

**Response:**
```json
{
  "product_name": "Samsung Galaxy S24 Ultra 256GB",
  "language": "en",
  "method": "ner",
  "agent_used": "ner",
  "brand_predictions": [{
    "brand": "Samsung",
    "confidence": 0.92,
    "method": "ner"
  }],
  "entities": [
    {
      "text": "Samsung",
      "type": "BRAND",
      "confidence": 0.92,
      "start": 0,
      "end": 7
    },
    {
      "text": "Galaxy S24 Ultra",
      "type": "PRODUCT",
      "confidence": 0.88,
      "start": 8,
      "end": 24
    }
  ],
  "processing_time_ms": 145,
  "timestamp": 1703123456.789
}
```

### 3. RAG Agent

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "iPhone 15 Pro Max 1TB",
    "language_hint": "en",
    "method": "rag"
  }'
```

**Response:**
```json
{
  "product_name": "iPhone 15 Pro Max 1TB",
  "language": "en",
  "method": "rag",
  "agent_used": "rag",
  "brand_predictions": [{
    "brand": "Apple",
    "confidence": 0.94,
    "method": "rag"
  }],
  "similar_products": [
    {
      "product_name": "iPhone 14 Pro Max",
      "brand": "Apple",
      "similarity_score": 0.95
    },
    {
      "product_name": "iPhone 15 Pro",
      "brand": "Apple", 
      "similarity_score": 0.92
    },
    {
      "product_name": "iPhone 13 Pro Max",
      "brand": "Apple",
      "similarity_score": 0.89
    }
  ],
  "processing_time_ms": 320,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "timestamp": 1703123456.789
}
```

### 4. LLM Agent

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "โทรศัพท์ Samsung Galaxy A54 5G",
    "language_hint": "th",
    "method": "llm"
  }'
```

**Response:**
```json
{
  "product_name": "โทรศัพท์ Samsung Galaxy A54 5G",
  "language": "th",
  "method": "llm",
  "agent_used": "llm",
  "brand_predictions": [{
    "brand": "Samsung",
    "confidence": 0.89,
    "method": "llm"
  }],
  "reasoning": "The product name contains 'Samsung' which is a well-known electronics brand. The text is in Thai but the brand name remains in English as is common for international brands.",
  "processing_time_ms": 1250,
  "timestamp": 1703123456.789
}
```

### 5. Hybrid Agent

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Nike Air Max 270 React รองเท้าผ้าใบ",
    "language_hint": "mixed",
    "method": "hybrid"
  }'
```

**Response:**
```json
{
  "product_name": "Nike Air Max 270 React รองเท้าผ้าใบ",
  "language": "mixed",
  "method": "hybrid",
  "agent_used": "hybrid",
  "brand_predictions": [{
    "brand": "Nike",
    "confidence": 0.91,
    "method": "hybrid"
  }],
  "pipeline_steps": [
    {"stage": "ner", "result": "Nike", "confidence": 0.88},
    {"stage": "rag", "result": "Nike", "confidence": 0.85},
    {"stage": "llm", "result": "Nike", "confidence": 0.90}
  ],
  "contributions": {
    "ner": 0.35,
    "rag": 0.30,
    "llm": 0.35
  },
  "processing_time_ms": 890,
  "timestamp": 1703123456.789
}
```

### 6. Simple Agent

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Sony WH-1000XM5 Headphones",
    "language_hint": "en",
    "method": "simple"
  }'
```

**Response:**
```json
{
  "product_name": "Sony WH-1000XM5 Headphones",
  "language": "en",
  "method": "simple",
  "agent_used": "simple",
  "brand_predictions": [{
    "brand": "Sony",
    "confidence": 0.80,
    "method": "simple"
  }],
  "reasoning": "Pattern match: First word 'Sony' matches known brand pattern",
  "processing_time_ms": 25,
  "timestamp": 1703123456.789
}
```

## Language Hints

| Value | Description | Use Case |
|-------|-------------|----------|
| `en` | English text | English product names |
| `th` | Thai text | Thai product names |
| `mixed` | Mixed languages | Thai + English product names |
| `auto` | Auto-detect | Unknown language mix |

## Health Check

**Request:**
```bash
curl -X GET "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "multilingual-inference-orchestrator",
  "environment": "production",
  "aws_region": "us-east-1",
  "orchestrator": "available",
  "orchestrator_agents_count": 4,
  "individual_agents_count": 5,
  "available_methods": ["orchestrator", "simple", "ner", "rag", "llm", "hybrid"],
  "agents_count": 5,
  "timestamp": 1703123456.789
}
```

## Service Information

**Request:**
```bash
curl -X GET "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/"
```

**Response:**
```json
{
  "service": "multilingual-inference-orchestrator",
  "version": "1.0.0",
  "endpoints": {
    "health": "GET /health",
    "inference": "POST /infer"
  },
  "inference_methods": {
    "orchestrator": "Use all available agents with best result selection",
    "simple": "Basic pattern matching without external dependencies",
    "rag": "Vector similarity search using sentence transformers",
    "hybrid": "Sequential pipeline combining NER, RAG, and LLM",
    "ner": "Named entity recognition for brand extraction",
    "llm": "Large language model inference"
  },
  "request_format": {
    "product_name": "string (required)",
    "language_hint": "string (optional: en, th, mixed, auto)",
    "method": "string (optional: orchestrator, simple, rag, hybrid, ner, llm)"
  }
}
```

## Error Responses

### Invalid Method

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Test Product",
    "method": "invalid_method"
  }'
```

**Response:**
```json
{
  "error": "Invalid method 'invalid_method'. Valid methods: orchestrator, simple, rag, hybrid, ner, llm",
  "status_code": 400,
  "timestamp": 1703123456.789
}
```

### Missing Product Name

**Request:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "language_hint": "en",
    "method": "ner"
  }'
```

**Response:**
```json
{
  "error": "Missing 'product_name' field",
  "status_code": 400,
  "timestamp": 1703123456.789
}
```

### Agent Not Available

**Response:**
```json
{
  "status": "error",
  "error": "Agent 'rag' not available. Available agents: ['simple', 'ner', 'llm']",
  "method": "rag",
  "available_agents": ["simple", "ner", "llm"],
  "timestamp": 1703123456.789
}
```

## Testing Script

Create a test script to try all methods:

```bash
#!/bin/bash

BASE_URL="http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com"
PRODUCT_NAME="Samsung Galaxy S24 Ultra"

echo "Testing all inference methods..."

for method in orchestrator simple ner rag llm hybrid; do
  echo "----------------------------------------"
  echo "Testing method: $method"
  echo "----------------------------------------"
  
  curl -X POST "$BASE_URL/infer" \
    -H "Content-Type: application/json" \
    -d "{
      \"product_name\": \"$PRODUCT_NAME\",
      \"language_hint\": \"en\",
      \"method\": \"$method\"
    }" | jq '.'
  
  echo ""
done
```

## Performance Comparison

| Method | Avg Response Time | Accuracy | Dependencies |
|--------|------------------|----------|--------------|
| `simple` | ~25ms | Medium | None |
| `ner` | ~150ms | High | spaCy |
| `rag` | ~300ms | High | Milvus, embeddings |
| `llm` | ~1200ms | Very High | AWS Bedrock |
| `hybrid` | ~900ms | Very High | All above |
| `orchestrator` | ~1500ms | Highest | All above |

## Best Practices

1. **Use `orchestrator` for production** - highest accuracy
2. **Use `simple` for fast responses** - when speed is critical
3. **Use specific agents for debugging** - to understand individual performance
4. **Include `language_hint`** - improves accuracy for multilingual content
5. **Handle errors gracefully** - check response status and available agents
6. **Monitor response times** - different methods have different performance characteristics

## Rate Limits and Quotas

- No explicit rate limits currently configured
- AWS Bedrock has service quotas for LLM calls
- Monitor CloudWatch metrics for usage patterns
- Consider implementing client-side rate limiting for high-volume usage