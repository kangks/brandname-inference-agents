# Architecture and Deployment Guide

## Overview

The multilingual product inference system is designed with a flexible architecture that supports both **monolithic** and **microservices** deployment patterns. This document clarifies the architecture, deployment options, and how to interact with the system.

## Architecture Patterns

### 1. Advanced Multi-Agent Architecture (Current Implementation)

The system implements a sophisticated multi-agent architecture using Strands Agents SDK with independent agent coordination:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECS Container                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                HTTP Server (server.py)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Strands MultiAgent Orchestrator                │   │ ← Advanced Coordination
│  │                                                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │   │
│  │  │ Strands     │ │ Strands     │ │ Fine-tuned  │      │   │
│  │  │ NER Agent   │ │ RAG Agent   │ │ Nova Agent  │      │   │ ← Independent Agents
│  │  │ (Nova Pro)  │ │ (Nova Pro)  │ │ (Custom ARN)│      │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘      │   │
│  │                                                         │   │
│  │  ┌─────────────┐ ┌─────────────┐                      │   │
│  │  │ Strands     │ │ Hybrid      │                      │   │
│  │  │ LLM Agent   │ │ Agent       │                      │   │
│  │  │ (Nova Pro)  │ │ (Combined)  │                      │   │
│  │  └─────────────┘ └─────────────┘                      │   │
│  │                                                         │   │
│  │  Coordination Methods:                                  │   │
│  │  • Swarm (Parallel execution)                          │   │
│  │  • Graph (Structured workflows)                        │   │
│  │  • Enhanced (Priority-based selection)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Legacy Agent Registry                      │   │ ← Fallback System
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │   │
│  │  │ NER │ │ RAG │ │ LLM │ │Hybr.│ │Simp.│              │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Independent Agent Execution**: Each Strands agent runs with its own model instance and processing pipeline
- **Advanced Coordination**: Uses Strands multiagent tools (Swarm, GraphBuilder, workflow) for sophisticated orchestration
- **Fine-tuned Specialization**: Custom Amazon Nova Pro deployment for domain-specific brand extraction
- **Fallback Architecture**: Legacy agent registry provides backup functionality
- **Dynamic Agent Creation**: Orchestrator creates specialized agents on-demand with unique configurations
- **Parallel Processing**: Swarm coordination enables concurrent agent execution for improved performance

### 2. Microservices Deployment (Infrastructure Available)

The system also supports microservices deployment where each agent runs as a separate ECS service:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Orchestrator   │    │   NER Service   │    │   RAG Service   │
│    Service      │    │                 │    │                 │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Application     │
                    │ Load Balancer   │
                    └─────────────────┘
```

**Key Characteristics:**
- Each agent runs as a separate ECS service
- Network communication between services
- Independent scaling and deployment
- Service discovery via ALB or service mesh

## ECS Service Configurations

### Available ECS Services

The infrastructure includes separate ECS service definitions for:

1. **orchestrator-service.json** - Main orchestrator service
2. **ner-service.json** - Named Entity Recognition service
3. **rag-service.json** - Retrieval Augmented Generation service
4. **llm-service.json** - Large Language Model service
5. **hybrid-service.json** - Hybrid approach service
6. **milvus-service.json** - Vector database service

### Current Deployment Status

**Currently deployed:** Only the orchestrator service is deployed, which contains all agents within the same container.

**Infrastructure available but not deployed:** Individual agent services are configured but not currently running as separate services.

## API Usage and Method Selection

### Endpoint Structure

```
POST http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer
```

### Request Format

```json
{
  "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
  "language_hint": "en",
  "method": "orchestrator"
}
```

### Method Parameter Options

| Method | Description | Implementation | Execution Mode |
|--------|-------------|----------------|----------------|
| `orchestrator` | **Strands MultiAgent Orchestrator** - Coordinates all agents using advanced multiagent patterns | Strands Agents SDK with Swarm/Graph coordination | Advanced multi-agent coordination |
| `finetuned` | **Fine-tuned Nova Agent** - Specialized Amazon Nova Pro model for brand extraction | Custom deployment ARN with domain-specific training | Direct Strands agent call |
| `simple` | Basic pattern matching without external dependencies | Legacy regex-based implementation | Direct agent call |
| `ner` | **Multilingual NER Agent** - spaCy-based entity recognition with Thai-English support | SpacyNERAgent with enhanced multilingual processing | Direct agent call |
| `rag` | **Enhanced RAG Agent** - Vector similarity search using SentenceTransformers and Milvus | Milvus vector database with multilingual embeddings | Direct agent call |
| `llm` | **Strands LLM Agent** - Amazon Nova Pro with specialized prompts | Strands Agent with multilingual prompt engineering | Direct Strands agent call |
| `hybrid` | Sequential pipeline combining NER, RAG, and LLM with confidence weighting | Combined approach with intelligent result aggregation | Direct agent call |

### Agent Implementation Details

#### Fine-tuned Nova Agent
- **Model ARN**: `arn:aws:bedrock:us-east-1:654654616949:custom-model-deployment/9o1i1v4ng8wy`
- **Training Domain**: Brand extraction from product titles
- **System Prompt**: Optimized for concise brand name extraction
- **Languages**: English, Thai, mixed-language transliterations
- **Confidence Range**: 0.7+ for clear brand identifications

#### Strands MultiAgent Orchestrator
- **Coordination Methods**:
  - `swarm`: Parallel execution using `Strands.multiagent.Swarm`
  - `graph`: Structured workflows using `Strands.multiagent.GraphBuilder`  
  - `enhanced`: Custom priority-based agent selection
- **Agent Management**: Dynamic creation of specialized Strands agents
- **Fallback Strategy**: Automatic fallback to legacy registry agents
- **Result Aggregation**: Intelligent selection based on confidence and method performance

### How Method Selection Works

1. **Orchestrator Method (`method: "orchestrator"`):**
   ```python
   # server.py - _handle_orchestrator_inference_sync()
   result = await self.orchestrator.process(product_input)
   ```
   - Coordinates multiple agents within the same memory space
   - Uses Strands multiagent capabilities (swarm, graph, workflow)
   - Returns aggregated results from all available agents

2. **Individual Agent Methods (`method: "ner"`, `method: "rag"`, etc.):**
   ```python
   # server.py - _handle_specific_agent_inference_sync()
   agent = self.individual_agents[method]
   result = await agent.process(product_input)
   ```
   - Calls specific agent directly
   - No network communication - all in-memory
   - Returns results from that specific agent only

## Deployment Scenarios

### Scenario 1: Current Monolithic Deployment

**What's deployed:**
- Single ECS service: `multilingual-inference-orchestrator`
- All agents run within the same container
- Load balancer routes to orchestrator service only

**How it works:**
```bash
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
    "language_hint": "en",
    "method": "orchestrator"
  }'
```

**Execution flow:**
1. ALB routes request to orchestrator service
2. Orchestrator service processes request in-memory
3. If `method: "orchestrator"` → coordinates all agents internally
4. If `method: "ner"` → calls NER agent directly (same container)
5. Returns unified response

### Scenario 2: Microservices Deployment (Not Currently Active)

**What would be deployed:**
- Multiple ECS services: orchestrator, ner, rag, llm, hybrid
- Each agent runs in separate containers
- Service mesh or ALB routing between services

**How it would work:**
```bash
# Route to specific agent service
curl -X POST "http://ner-service-alb.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
    "language_hint": "en"
  }'
```

## Calling Individual Agents

### Current Method (In-Memory)

To call a specific agent in the current deployment:

```bash
# Call NER agent specifically
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24",
    "language_hint": "en",
    "method": "ner"
  }'

# Call RAG agent specifically  
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24",
    "language_hint": "en", 
    "method": "rag"
  }'

# Call LLM agent specifically
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24",
    "language_hint": "en",
    "method": "llm"
  }'
```

### Future Method (Microservices)

If individual services were deployed, you would call them directly:

```bash
# Direct service calls (hypothetical)
curl -X POST "http://ner-service-alb.us-east-1.elb.amazonaws.com/infer" ...
curl -X POST "http://rag-service-alb.us-east-1.elb.amazonaws.com/infer" ...
curl -X POST "http://llm-service-alb.us-east-1.elb.amazonaws.com/infer" ...
```

## Response Formats

### Orchestrator Response

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

### Individual Agent Response

```json
{
  "product_name": "Alectric Smart Slide Fan Remote พัดลมสไลด์ 16 นิ้ว รุ่น RF2",
  "language": "en",
  "method": "ner",
  "agent_used": "ner",
  "brand_predictions": [{
    "brand": "Alectric",
    "confidence": 0.85,
    "method": "ner"
  }],
  "entities": [
    {
      "text": "Alectric",
      "type": "BRAND",
      "confidence": 0.85,
      "start": 0,
      "end": 8
    }
  ],
  "processing_time_ms": 245,
  "timestamp": 1703123456.789
}
```

## Health Check and Status

### Check System Status

```bash
curl -X GET "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/health"
```

Response:
```json
{
  "status": "healthy",
  "service": "multilingual-inference-orchestrator",
  "environment": "production",
  "orchestrator": "available",
  "orchestrator_agents_count": 4,
  "individual_agents_count": 5,
  "available_methods": ["orchestrator", "simple", "ner", "rag", "llm", "hybrid"],
  "timestamp": 1703123456.789
}
```

### Check Available Methods

```bash
curl -X GET "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/"
```

## Deployment Commands

### Deploy Orchestrator Service (Current)

```bash
# Deploy the monolithic orchestrator service
./scripts/deploy-orchestrator-simple.sh
```

### Deploy Individual Services (Future)

```bash
# Deploy individual agent services
./scripts/deploy-ner-service.sh
./scripts/deploy-rag-service.sh  
./scripts/deploy-llm-service.sh
./scripts/deploy-hybrid-service.sh
```

## Configuration Management

### Environment Variables

Key environment variables for deployment:

```bash
# Application settings
INFERENCE_ENV=production
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8080

# AWS settings  
AWS_REGION=us-east-1
AWS_PROFILE=default

# Agent settings
USE_MOCK_SERVICES=false
ENABLE_NER_AGENT=true
ENABLE_RAG_AGENT=true
ENABLE_LLM_AGENT=true
ENABLE_HYBRID_AGENT=true
```

### ECS Task Definition Configuration

Each service can be configured via its task definition:

```json
{
  "family": "multilingual-inference-orchestrator",
  "containerDefinitions": [{
    "name": "orchestrator-agent",
    "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/multilingual-inference:latest",
    "portMappings": [{"containerPort": 8080}],
    "environment": [
      {"name": "INFERENCE_ENV", "value": "production"},
      {"name": "LOG_LEVEL", "value": "INFO"}
    ]
  }]
}
```

## Monitoring and Troubleshooting

### Check Agent Status

```bash
# Check which agents are available
curl -X GET "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/health" | jq '.available_methods'
```

### Test Individual Agents

```bash
# Test each agent individually
for method in simple ner rag llm hybrid orchestrator; do
  echo "Testing $method agent..."
  curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
    -H "Content-Type: application/json" \
    -d "{\"product_name\": \"Samsung Galaxy S24\", \"method\": \"$method\"}" | jq '.method, .brand_predictions[0].brand'
done
```

### Common Issues

1. **Agent Not Available**: Check health endpoint to see which agents are initialized
2. **Low Confidence**: Try different methods or check input data quality
3. **Timeout**: Increase timeout settings or check resource allocation
4. **Memory Issues**: Monitor container memory usage and adjust limits

## Migration Path

### From Monolithic to Microservices

1. **Phase 1**: Current state - all agents in orchestrator service
2. **Phase 2**: Deploy individual services alongside orchestrator
3. **Phase 3**: Update orchestrator to call individual services via HTTP
4. **Phase 4**: Remove agents from orchestrator container
5. **Phase 5**: Pure microservices architecture

### Rollback Strategy

- Keep orchestrator service with embedded agents as fallback
- Use feature flags to switch between embedded and service calls
- Monitor performance and reliability during transition

## Summary

**Current Architecture**: Monolithic deployment with all agents in the same container
**Agent Communication**: In-memory method calls, no network communication
**Method Selection**: Via `method` parameter in API request
**Individual Agent Access**: Through the same endpoint with different `method` values
**Future Architecture**: Microservices deployment with separate ECS services per agent

The system is designed to be flexible and can operate in both deployment modes depending on requirements for scalability, isolation, and operational complexity.