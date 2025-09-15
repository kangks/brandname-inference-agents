# Multilingual Product Inference System

A sophisticated AI-powered system for extracting brand names from multilingual product descriptions using advanced multi-agent coordination.

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

## 🏗️ Architecture

**Current Deployment**: Monolithic architecture with all agents in a single container

```
┌─────────────────────────────────────────┐
│           ECS Container                 │
│  ┌─────────────────────────────────┐   │
│  │        HTTP Server              │   │ ← API Requests
│  │         (server.py)             │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │      Orchestrator Agent         │   │ ← Coordinates all agents
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐ │   │
│  │  │ NER │ │ RAG │ │ LLM │ │Hyb.│ │   │ ← All agents in same memory
│  │  └─────┘ └─────┘ └─────┘ └────┘ │   │
│  │  ┌─────┐                       │   │
│  │  │Simp.│                       │   │
│  │  └─────┘                       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

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

| Method | Description | Use Case | Response Time |
|--------|-------------|----------|---------------|
| `orchestrator` | Coordinates all agents, returns best result | **Production use** - highest accuracy | ~1500ms |
| `simple` | Pattern-based matching | Fast responses, no dependencies | ~25ms |
| `ner` | Named Entity Recognition | Extract entities from text | ~150ms |
| `rag` | Vector similarity search | Similarity-based matching | ~300ms |
| `llm` | Large Language Model reasoning | Complex reasoning tasks | ~1200ms |
| `hybrid` | Sequential NER → RAG → LLM pipeline | Balanced accuracy and speed | ~900ms |

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

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_orchestrator_*.py

# End-to-end validation
python tests/final_validation.py

# Test all API methods
./scripts/test_brand_inference_endpoints.sh
```

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

- Check [Architecture FAQ](../INFERENCE_ARCHITECTURE_FAQ.md) for common questions
- Review CloudWatch logs for detailed error information
- Consult [API Usage Guide](docs/API_USAGE_GUIDE.md) for implementation examples