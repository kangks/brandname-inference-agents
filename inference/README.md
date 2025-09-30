# Multilingual Product Inference System

A sophisticated AI-powered system for extracting brand names from multilingual product descriptions using advanced multi-agent coordination with Strands Agents SDK.

## 🚀 Quick Start

### Option 1: One-Command Deployment
```bash
./infrastructure/scripts/deploy-monolithic.sh
```

### Option 2: Step-by-Step Deployment
```bash
./infrastructure/scripts/step1_deploy-cloudformation.sh  # Deploy AWS infrastructure
./infrastructure/scripts/step2_build-and-push-images.sh  # Build and push Docker image
./infrastructure/scripts/step3_deploy-ecs.sh            # Deploy ECS service
```

### Local Development
```bash
pip install -r requirements.txt
python -m inference.server
```

## 🏗️ Agentic Architecture

**Current Implementation**: Advanced multi-agent system using Strands Agents SDK with independent agent coordination

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECS Container                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                HTTP Server (server.py)                 │   │ ← API Requests
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Strands MultiAgent Orchestrator                │   │ ← Coordinates via Swarm/Graph
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │   │
│  │  │ Strands     │ │ Strands     │ │ Fine-tuned  │      │   │
│  │  │ NER Agent   │ │ RAG Agent   │ │ Nova Agent  │      │   │ ← Independent Strands Agents
│  │  └─────────────┘ └─────────────┘ └─────────────┘      │   │
│  │  ┌─────────────┐ ┌─────────────┐                      │   │
│  │  │ Strands     │ │ Hybrid      │                      │   │
│  │  │ LLM Agent   │ │ Agent       │                      │   │
│  │  └─────────────┘ └─────────────┘                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Legacy Agent Registry                      │   │ ← Fallback agents
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │   │
│  │  │ NER │ │ RAG │ │ LLM │ │Hybr.│ │Simp.│              │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Independence & Coordination

Each agent runs independently with its own:
- **Strands Agent Instance**: Individual AI agent with specialized system prompts
- **Processing Pipeline**: Independent inference logic and confidence scoring  
- **Resource Management**: Separate model loading and cleanup procedures
- **Error Handling**: Isolated failure modes without affecting other agents

**Coordination Methods**:
- **Swarm Coordination**: Parallel execution using `Strands.multiagent.Swarm`
- **Graph Coordination**: Structured workflows using `Strands.multiagent.GraphBuilder`
- **Enhanced Coordination**: Custom orchestration with priority-based agent selection

## 🎯 API Usage

### Single Endpoint, Multiple Methods

```bash
# Base URL (replace with your ALB DNS)
BASE_URL="http://your-alb-dns.us-east-1.elb.amazonaws.com"

# Orchestrator (coordinates all agents) - Recommended
curl -X POST "$BASE_URL/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'

# Individual agents
curl -X POST "$BASE_URL/infer" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "ner"}'     # Named Entity Recognition
curl -X POST "$BASE_URL/infer" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "rag"}'     # Vector similarity search
curl -X POST "$BASE_URL/infer" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "llm"}'     # Large Language Model
curl -X POST "$BASE_URL/infer" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "hybrid"}'  # Combined pipeline
curl -X POST "$BASE_URL/infer" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "simple"}'  # Pattern matching
```

### Health Check
```bash
curl "$BASE_URL/health"  # Check system status
curl "$BASE_URL/"        # Service information
```

## 🧠 Available Agents

| Method | Description | Implementation | Use Case | Response Time |
|--------|-------------|----------------|----------|---------------|
| `orchestrator` | **Strands MultiAgent Orchestrator** - Coordinates all agents using Swarm/Graph | Strands Agents SDK with multiagent coordination | **Production use** - highest accuracy | ~1500ms |
| `finetuned` | **Fine-tuned Nova Agent** - Specialized Amazon Nova Pro model for brand extraction | Custom deployment ARN with optimized prompts | **Highest accuracy** - domain-specific training | ~800ms |
| `simple` | Pattern-based matching with regex | Legacy implementation | Fast responses, no dependencies | ~25ms |
| `ner` | **Multilingual NER Agent** - spaCy-based entity recognition | SpacyNERAgent with Thai-English support | Extract entities from multilingual text | ~150ms |
| `rag` | **Enhanced RAG Agent** - Vector similarity with SentenceTransformers | Milvus + sentence-transformers | Similarity-based brand matching | ~300ms |
| `llm` | **Strands LLM Agent** - Amazon Nova Pro reasoning | Strands Agent with specialized prompts | Complex multilingual reasoning | ~1200ms |
| `hybrid` | Sequential NER → RAG → LLM pipeline | Combined approach with confidence weighting | Balanced accuracy and speed | ~900ms |

### Agent Implementations

#### Fine-tuned Nova Agent (Recommended)
- **Model**: Custom Amazon Nova Pro deployment (`arn:aws:bedrock:us-east-1:654654616949:custom-model-deployment/9o1i1v4ng8wy`)
- **Specialization**: Fine-tuned specifically for brand extraction from product titles
- **Languages**: English, Thai, mixed-language support
- **Confidence**: Typically 0.7+ for clear brand names

#### Strands MultiAgent Orchestrator
- **Coordination**: Uses `Strands.multiagent.Swarm` for parallel agent execution
- **Agent Management**: Dynamic creation and coordination of specialized agents
- **Fallback**: Automatic fallback to legacy agents if Strands agents fail
- **Result Aggregation**: Intelligent selection of best prediction across all agents

## 📁 Project Structure

```
inference/
├── src/                    # Source code
│   ├── agents/            # All agent implementations
│   ├── config/            # Configuration management
│   ├── models/            # Data models and schemas
│   └── monitoring/        # Health checks and monitoring
├── infrastructure/        # AWS deployment
│   ├── scripts/          # Deployment scripts
│   ├── ecs/              # ECS configurations
│   └── cloudformation/   # Infrastructure templates
├── tests/                # Test suite
├── docs/                 # Detailed documentation
├── Dockerfile            # Container definition
└── README.md            # This file
```

## 📚 Documentation

### Quick References
- **[API Usage Examples](docs/API_USAGE_GUIDE.md)** - Complete API reference with examples
- **[Architecture FAQ](../INFERENCE_ARCHITECTURE_FAQ.md)** - Common questions answered
- **[Deployment Guide](infrastructure/README.md)** - Infrastructure setup

### Detailed Guides
- **[Architecture Deep Dive](docs/ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md)** - Complete system architecture
- **[Infrastructure Guide](infrastructure/README.md)** - AWS deployment details

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src

# Test by category
python -m pytest tests/unit/           # Unit tests for individual agents
python -m pytest tests/integration/   # Integration tests for agent coordination  
python -m pytest tests/end_to_end/    # End-to-end system tests

# Test specific agents
python -m pytest tests/unit/test_orchestrator_agent.py  # Strands orchestrator
python -m pytest tests/unit/test_ner_agent.py          # NER agent
python -m pytest tests/unit/test_rag_agent.py          # RAG agent
python -m pytest tests/unit/test_llm_agent.py          # LLM agent

# Test multiagent coordination
python -m pytest tests/integration/test_swarm_coordination.py
python -m pytest tests/integration/test_orchestrator_coordination.py

# AWS and custom deployment tests
python -m pytest tests/end_to_end/test_custom_deployment.py
python -m pytest tests/end_to_end/test_aws_environment.py

# Performance and batch testing
./scripts/batch_compare.sh              # Batch comparison across methods
./scripts/test_brand_inference_endpoints.sh  # API endpoint validation
```

### Test Results Example

Recent batch comparison results showing agent performance:

```bash
% bash ./scripts/batch_compare.sh
=======================================================================
🔎 Querying for: CLEAR NOSE Moist Skin Barrier Moisturizing Gel 120ml เคลียร์โนส มอยส์เจอไรซิ่งเจล เฟเชียล.
=======================================================================
Method     Prediction  Confidence
finetuned  Clear       0.7
ner        name        0.9
rag        name        0.9
llm        name        0.85
hybrid     name        0.7000000000000001

=======================================================================
🔎 Querying for: Dr.Althea Cream ด๊อกเตอร์อัลเทีย ครีมบำรุงผิวหน้า 50ml (345 Relief/147 Barrier)
=======================================================================
Method     Prediction  Confidence
finetuned  Dr.Althea   0.7
ner        name        0.9
rag        name        0.9
llm        name        1.0
hybrid     name        0.7000000000000001

=======================================================================
🔎 Querying for: Eucerin Spotless Brightening Skin Tone Perfecting Body Lotion 250ml ยูเซอริน ผลิตภัณฑ์บำรุงผิวกาย
=======================================================================
Method     Prediction  Confidence
finetuned  Eucerin     0.7
ner        name        0.9
rag        name        0.9
llm        name        1.0
hybrid     name        0.7000000000000001
```

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Individual agent functionality
- **Integration Tests** (`@pytest.mark.integration`): Agent coordination and communication
- **End-to-End Tests** (`@pytest.mark.e2e`): Full system workflows
- **AWS Tests** (`@pytest.mark.aws`): AWS service integration
- **Multilingual Tests** (`@pytest.mark.multilingual`): Thai-English mixed content
- **Performance Tests** (`@pytest.mark.performance`): Load and latency testing

## 🔧 Configuration

Key environment variables:
```bash
INFERENCE_ENV=production     # Environment
LOG_LEVEL=INFO              # Logging level
AWS_REGION=us-east-1        # AWS region
USE_MOCK_SERVICES=false     # Enable mock services for testing
```

## 🚨 Troubleshooting

### Common Issues
1. **Service won't start**: Check CloudWatch logs at `/ecs/multilingual-inference-orchestrator`
2. **High latency**: Monitor CPU/memory usage, consider scaling
3. **Agent failures**: Check individual agent logs and dependencies

### Health Monitoring
```bash
# Check service status
aws ecs describe-services --cluster multilingual-inference-cluster --services multilingual-inference-orchestrator

# View logs
aws logs tail /ecs/multilingual-inference-orchestrator --since 1h
```

## 🎯 Next Steps

1. **Deploy**: Use `./infrastructure/scripts/deploy-monolithic.sh`
2. **Test**: Try different methods with your product data
3. **Monitor**: Check CloudWatch dashboards for performance
4. **Scale**: Adjust auto-scaling policies based on usage

## 📞 Support

- Check [Architecture FAQ](docs/INFERENCE_ARCHITECTURE_FAQ.md) for common questions
- Review CloudWatch logs for detailed error information
- Consult [API Usage Guide](docs/API_USAGE_GUIDE.md) for implementation examples