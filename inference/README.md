# Inference System

This directory contains all inference-related code, infrastructure, documentation, and tests for the multilingual product inference system.

## Directory Structure

```
inference/
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── orchestrator_agent.py  # Multi-agent orchestrator
│   ├── simple_agent.py    # Pattern-based inference
│   ├── ner_agent.py       # Named Entity Recognition
│   ├── llm_agent.py       # Large Language Model
│   ├── rag_agent.py       # Retrieval Augmented Generation
│   ├── hybrid_agent.py    # Hybrid approach
│   └── registry.py        # Agent registry
├── config/                # Configuration management
│   ├── settings.py        # Application settings
│   ├── model_registry.py  # Model configurations
│   └── validators.py      # Configuration validators
├── models/                # Data models and schemas
│   └── data_models.py     # Pydantic models
├── monitoring/            # Monitoring and observability
│   ├── health_checker.py  # Health check endpoints
│   ├── logger.py          # Structured logging
│   ├── diagnostics.py     # System diagnostics
│   └── cloudwatch_integration.py  # AWS CloudWatch
├── infrastructure/        # Infrastructure as Code
│   ├── cloudformation/    # CloudFormation templates
│   ├── docker/           # Docker configurations
│   ├── ecs/              # ECS task definitions
│   ├── milvus/           # Vector database setup
│   ├── monitoring/       # Monitoring infrastructure
│   ├── scripts/          # Deployment scripts
│   └── storage/          # Storage configurations
├── tests/                # Test suite
│   ├── test_*.py         # Unit and integration tests
│   ├── final_*.py        # End-to-end validation
│   └── validate_*.py     # Deployment validation
├── scripts/              # Utility scripts
│   ├── deploy_*.sh       # Deployment scripts
│   └── test_*.sh         # Testing scripts
├── docs/                 # Documentation
│   └── inference.md      # System documentation
├── diagrams/             # Architecture diagrams
│   └── *.png            # Generated diagrams
├── demo_method_selection.py  # Method selection demo
├── main.py              # Application entry point
├── server.py            # FastAPI server
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   cd inference
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python -m inference.server
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

4. **Deploy to AWS:**
   ```bash
   ./scripts/deploy-orchestrator-simple.sh
   ```

## Key Components

### Agents
- **Orchestrator Agent**: Coordinates multiple inference methods
- **Simple Agent**: Pattern-based brand name extraction
- **NER Agent**: Named Entity Recognition using spaCy
- **LLM Agent**: Large Language Model inference
- **RAG Agent**: Retrieval Augmented Generation
- **Hybrid Agent**: Combines multiple approaches

### Infrastructure
- **ECS Fargate**: Containerized deployment
- **Milvus**: Vector database for embeddings
- **CloudWatch**: Monitoring and logging
- **Application Load Balancer**: Traffic distribution

### API Endpoints
- `POST /infer` - Main inference endpoint
- `GET /health` - Health check
- `GET /methods` - Available inference methods
- `GET /agents` - Active agents status

## Configuration

Environment variables are managed through the `config/settings.py` module. Key settings include:

- `INFERENCE_ENV`: Environment (local/dev/prod)
- `LOG_LEVEL`: Logging verbosity
- `USE_MOCK_SERVICES`: Enable mock services for testing
- `AWS_REGION`: AWS region for services

## Testing

The test suite includes:
- Unit tests for individual components
- Integration tests for agent coordination
- End-to-end validation scripts
- Performance benchmarks

Run specific test categories:
```bash
# Unit tests
python -m pytest tests/test_*.py

# Integration tests
python -m pytest tests/test_orchestrator_*.py

# Validation tests
python tests/final_validation.py
```

## Deployment

See `infrastructure/README.md` for detailed deployment instructions.

Quick deployment:
```bash
# Deploy to ECS
./scripts/deploy-orchestrator-simple.sh

# Validate deployment
python tests/final_validation.py
```