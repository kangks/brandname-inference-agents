# Inference Architecture FAQ

## Your Questions Answered

### 1. Why are there different ECS services for agents? Do they run as separate servers in ECS service?

**Answer:** The infrastructure supports both deployment patterns, but currently only the monolithic pattern is deployed.

**Current State (Deployed):**
- Only the **orchestrator service** is running in ECS
- All agents (NER, RAG, LLM, Hybrid) run **within the same container/memory space**
- No separate ECS services for individual agents are currently deployed

**Infrastructure Available (Not Deployed):**
- Separate ECS service definitions exist for each agent:
  - `ner-service.json`
  - `rag-service.json` 
  - `llm-service.json`
  - `hybrid-service.json`
  - `orchestrator-service.json`
- These could be deployed as separate microservices if needed

**Why separate service definitions exist:**
- **Flexibility**: Support both monolithic and microservices deployment
- **Future scaling**: Individual agents can be scaled independently
- **Isolation**: Separate services provide better fault isolation
- **Resource optimization**: Different agents have different resource requirements

### 2. When I invoke the endpoint, does the orchestrator call agents via network to ECS services? Or within the same memory?

**Answer:** The orchestrator calls agents **within the same memory space** - no network calls.

**Current Architecture:**
```
┌─────────────────────────────────────────┐
│           ECS Container                 │
│  ┌─────────────────────────────────┐   │
│  │        HTTP Server              │   │ ← Your curl request hits here
│  │         (server.py)             │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │      Orchestrator Agent         │   │ ← Orchestrator coordinates here
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐ │   │
│  │  │ NER │ │ RAG │ │ LLM │ │Hyb.│ │   │ ← All agents in same memory
│  │  └─────┘ └─────┘ └─────┘ └────┘ │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Execution Flow:**
1. Your curl request → ALB → ECS Container → HTTP Server
2. HTTP Server → Orchestrator Agent (in-memory method call)
3. Orchestrator Agent → Individual Agents (in-memory method calls)
4. Results aggregated and returned

**Code Evidence:**
```python
# server.py - _handle_orchestrator_inference_sync()
result = await self.orchestrator.process(product_input)  # In-memory call

# orchestrator_agent.py - coordinates agents internally
for agent_id, agent in self.specialized_agents.items():
    response = agent(prompt)  # Direct method call, no HTTP
```

### 3. How do I call respective agents ECS service? Is it the "method" parameter?

**Answer:** Yes, you use the `method` parameter to call specific agents, but they're still called within the same container.

**Available Methods:**
```bash
# Call orchestrator (coordinates all agents)
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'

# Call NER agent directly
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "ner"}'

# Call RAG agent directly  
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "rag"}'

# Call LLM agent directly
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "llm"}'

# Call Hybrid agent directly
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "hybrid"}'

# Call Simple agent directly
curl -X POST "http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "simple"}'
```

**How Method Selection Works:**
```python
# server.py - _handle_inference()
method = request_data.get('method', 'orchestrator')  # Default to orchestrator

if method == 'orchestrator':
    result = self._handle_orchestrator_inference_sync(product_input)
else:
    result = self._handle_specific_agent_inference_sync(product_input, method)
```

**Current Reality:**
- All methods hit the **same ECS service** (orchestrator service)
- All agents run in the **same container**
- The `method` parameter determines which agent code path to execute
- **No separate ECS services** are currently deployed for individual agents

**Future Possibility:**
If individual agent services were deployed, you could call them directly:
```bash
# Hypothetical direct service calls (not currently available)
curl -X POST "http://ner-service-alb.us-east-1.elb.amazonaws.com/infer" ...
curl -X POST "http://rag-service-alb.us-east-1.elb.amazonaws.com/infer" ...
```

## Summary

| Question | Answer |
|----------|--------|
| **Separate ECS services?** | Infrastructure exists but not deployed. Only orchestrator service is running. |
| **Network calls between agents?** | No. All agents run in same memory space with direct method calls. |
| **How to call specific agents?** | Use `method` parameter in API request to same endpoint. |

## Current vs Future Architecture

### Current (Deployed)
- **Single ECS service**: `multilingual-inference-orchestrator`
- **Single container**: All agents in same memory
- **Single endpoint**: All methods via same URL with `method` parameter
- **Communication**: In-memory method calls

### Future (Infrastructure Ready)
- **Multiple ECS services**: One per agent type
- **Multiple containers**: Each agent in separate container
- **Multiple endpoints**: Direct URLs per service
- **Communication**: HTTP calls between services

The system is designed to support both patterns, giving you flexibility to choose based on your scaling and operational requirements.

## Testing All Methods

You can test all available methods with this script:

```bash
#!/bin/bash
BASE_URL="http://production-inference-alb-2106753314.us-east-1.elb.amazonaws.com"

for method in orchestrator simple ner rag llm hybrid; do
  echo "Testing $method..."
  curl -X POST "$BASE_URL/infer" \
    -H "Content-Type: application/json" \
    -d "{\"product_name\": \"Samsung Galaxy S24\", \"method\": \"$method\"}" \
    | jq '.method, .brand_predictions[0].brand // .best_prediction'
done
```

This will show you how each method processes the same input and returns results in slightly different formats.