# Multilingual Product Inference System

A proof-of-concept agentic workflow system that processes product names in English, Thai, or mixed languages using multiple AI inference mechanisms (NER, RAG, LLM, and hybrid approaches).

## 🏗️ Architecture Overview

The system consists of two main components:

- **Training Stack** (`training/` folder) - Handles data preprocessing, model training, and knowledge base preparation
- **Inference Stack** (`inference/` folder) - Handles real-time product name inference using trained models

### Agent Architecture

- **Orchestrator Agent**: Coordinates parallel execution of inference agents
- **NER Agent**: Named Entity Recognition for brand/product identification  
- **RAG Agent**: Retrieval-Augmented Generation using vector similarity
- **LLM Agent**: Direct language model inference using fine-tuned Nova Pro
- **Hybrid Agent**: Sequential combination of multiple approaches

## 🚀 Quick Start

### Prerequisites

- Python 3.13
- AWS CLI configured with `ml-sandbox` profile
- Access to AWS us-east-1 region

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd multilingual-product-inference
   make setup
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Verify installation:**
   ```bash
   make check-aws
   ```

## 🐳 Local Docker Testing

Before deploying to ECS, test the inference container locally:

### Quick Test
```bash
# Validate Docker environment
make docker-validate

# Run simple container test
make docker-test-simple
```

### Comprehensive Testing
```bash
# Run full test suite with mock services
make docker-test-compose
```

### Test Results
- ✅ **Pass**: Container is ready for ECS deployment
- ❌ **Fail**: Review logs and fix issues before deploying

See [Docker Testing Workflow](DOCKER_TESTING_WORKFLOW.md) for detailed instructions.
   make validate-env
   ```

### Basic Usage

1. **Initialize system:**
   ```bash
   python -m inference.main init
   ```

2. **Check system health:**
   ```bash
   python -m inference.main health
   ```

3. **Process a product name:**
   ```bash
   python -m inference.main process "iPhone 15 Pro Max" --language en
   python -m inference.main process "ยาสีฟัน Wonder smile toothpaste kid" --language mixed
   ```

4. **Start development server:**
   ```bash
   python -m inference.main serve --dev
   ```

## 📁 Project Structure

```
multilingual-product-inference/
├── inference/                    # Inference stack
│   ├── agents/                  # Agent implementations
│   │   ├── __init__.py
│   │   └── base_agent.py       # Base agent interfaces
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # System configuration
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   └── data_models.py      # Core data structures
│   ├── __init__.py
│   └── main.py                 # Main entry point
├── training/                    # Training stack
│   ├── preprocessing/          # Data preprocessing
│   ├── pipelines/             # Training pipelines
│   └── __init__.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_config.py         # Configuration tests
│   └── test_data_models.py    # Data model tests
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── Makefile                  # Development commands
├── Dockerfile                # ARM64 container image
├── .env.example             # Environment template
└── README.md               # This file
```

## 🛠️ Development

### Code Quality Standards

This project follows **PEP 8** coding standards with the following tools:

- **Black**: Code formatting (88 character line length)
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking
- **Pytest**: Unit and integration testing

### Development Commands

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make check

# Run tests
make test
make test-unit
make test-integration

# Build package
make build

# Clean artifacts
make clean
```

### Environment Setup

The system requires:

- **Python 3.13** in a virtual environment
- **AWS profile**: `ml-sandbox` for all AWS interactions
- **AWS region**: `us-east-1` for all resources
- **Platform**: ARM64 architecture for containerized services

### Configuration

Configuration is managed through:

1. **Environment variables** (see `.env.example`)
2. **Configuration files** (JSON format)
3. **Default values** in `SystemConfig`

Key configuration areas:

- **Model settings**: NER, embedding, and LLM model configurations
- **AWS settings**: Profile, region, and service configurations  
- **Agent settings**: Timeouts, concurrency, and confidence thresholds
- **Infrastructure**: Milvus, ECS, and monitoring settings

## 🧪 Testing

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/ -m "unit"

# Integration tests only  
pytest tests/ -m "integration"

# With coverage
pytest tests/ --cov=inference --cov=training --cov-report=html
```

### Test Structure

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Mock services**: Local testing without AWS dependencies

## 🐳 Docker Deployment

### Building ARM64 Image

```bash
make docker-build
```

### Running Container

```bash
make docker-run
```

### ECS Deployment

The system is designed for AWS ECS Fargate deployment with:

- **ARM64 platform** for cost efficiency
- **Auto-scaling** based on demand
- **Health checks** and monitoring
- **CloudWatch** logging

## 📊 Monitoring and Debugging

### Health Checks

```bash
# System health
python -m inference.main health

# Component status
curl http://localhost:8080/health
```

### Logging

- **Structured logging** with configurable levels
- **CloudWatch integration** for production
- **Local file logging** for development

### Debugging

- **Diagnostic endpoints** for troubleshooting
- **Agent health monitoring** 
- **Performance metrics** and timing

## 🔧 Configuration Management

### Model Switching

```python
from inference.config.settings import config_manager, ModelType

# Register custom model
config_manager.register_model(
    ModelType.NER, 
    "custom_ner", 
    {"model_name": "custom_model", "confidence_threshold": 0.8}
)

# Switch to custom model
config_manager.switch_model(ModelType.NER, "custom_ner")
```

### Environment Configuration

```bash
# Development
export INFERENCE_ENV=development

# Production  
export INFERENCE_ENV=production
export LOG_LEVEL=WARNING
export AGENT_TIMEOUT=60
```

## 📚 Implementation Reference

This implementation references patterns from `brand_extraction_ner_rag_llm_rev2.ipynb`:

- **Data structures**: Training data format and processing patterns
- **Model integration**: spaCy NER, sentence transformers, AWS Bedrock
- **Vector search**: Milvus collection setup and query patterns
- **Fine-tuning**: Conversation format and AWS Bedrock job submission

## 🚧 Current Status

**✅ Completed (Task 1):**
- Project structure and organization
- Core data models and interfaces  
- Configuration management system
- Development environment setup
- PEP 8 compliance and tooling
- Basic CLI and health checks

**🔄 Next Steps:**
- Task 2.1: Implement NER Agent
- Task 2.2: Implement RAG Agent  
- Task 2.3: Implement LLM Agent
- Task 2.4: Implement Hybrid Agent
- Task 3: Implement Orchestrator Agent

## 📚 Documentation

### Comprehensive Guides

- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and debugging steps including ARM64 platform considerations
- **[Model Tuning Guide](docs/MODEL_TUNING.md)** - Detailed procedures for improving inference accuracy with ml-sandbox AWS profile usage
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Complete AWS infrastructure setup and management in us-east-1 region
- **[Local Testing Guide](docs/LOCAL_TESTING.md)** - Mock service setup and testing procedures for Python 3.13 .venv environment

### Quick Reference

- **Setup**: Python 3.13 + .venv virtual environment + PEP 8 compliance
- **AWS**: ml-sandbox profile + us-east-1 region + ARM64 platform
- **Testing**: Comprehensive mock services for local development
- **Deployment**: ECS Fargate + Milvus + CloudFormation templates

## 🔧 Troubleshooting

For common issues and debugging steps, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) which covers:

- Environment setup issues (Python 3.13, .venv, dependencies)
- AWS configuration problems (ml-sandbox profile, us-east-1 region)
- ARM64 platform issues (Docker builds, ECS deployment)
- Agent-specific issues (NER, RAG, LLM, Hybrid agents)
- Performance and deployment issues
- Network connectivity problems
- Comprehensive debugging tools and commands

## 🎯 Model Accuracy Tuning

For detailed procedures to improve inference accuracy, see the [Model Tuning Guide](docs/MODEL_TUNING.md) which covers:

- NER model optimization (spaCy custom models, transformer-based NER)
- RAG system tuning (embedding models, vector database optimization)
- LLM fine-tuning (Nova Pro via AWS Bedrock, conversation format data)
- Hybrid agent optimization (pipeline configuration, weight optimization)
- Confidence score calibration and performance trade-offs
- Comprehensive evaluation and validation frameworks

## 🚀 Deployment

For complete AWS deployment instructions, see the [Deployment Guide](docs/DEPLOYMENT.md) which covers:

- Prerequisites and AWS account setup (ml-sandbox profile, us-east-1 region)
- Infrastructure deployment (CloudFormation, ECS, ARM64 platform)
- Service deployment (all inference agents, auto-scaling)
- Milvus vector database deployment and initialization
- Monitoring and logging setup (CloudWatch, health checks)
- Post-deployment validation and troubleshooting

## 🧪 Local Development and Testing

For comprehensive local testing setup, see the [Local Testing Guide](docs/LOCAL_TESTING.md) which covers:

- Python 3.13 .venv environment setup and configuration
- Mock services for AWS, Milvus, spaCy, and Sentence Transformers
- Local development server setup and interactive testing
- Unit, integration, performance, and accuracy testing
- End-to-end testing and debugging procedures
- Continuous integration and automated test execution

## 🤝 Contributing

1. **Environment**: Use Python 3.13 in .venv virtual environment
2. **Code Quality**: Follow PEP 8 coding standards (use black, flake8, mypy)
3. **Testing**: Add comprehensive tests using mock services for local development
4. **Documentation**: Update relevant documentation files
5. **AWS**: Use ml-sandbox profile and us-east-1 region for all AWS operations
6. **Platform**: Ensure ARM64 compatibility for containerized services

### Development Workflow

```bash
# Setup development environment
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Code quality checks
black .
flake8 .
mypy inference/ training/

# Run tests
python tests/run_local_tests.py --full

# AWS operations (use ml-sandbox profile)
export AWS_PROFILE=ml-sandbox
export AWS_REGION=us-east-1
```

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:

1. **Check Documentation**: Review the comprehensive guides above
2. **Troubleshooting**: Start with the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
3. **Local Testing**: Use the [Local Testing Guide](docs/LOCAL_TESTING.md) for development issues
4. **AWS Issues**: Refer to the [Deployment Guide](docs/DEPLOYMENT.md) for infrastructure problems
5. **Model Issues**: Use the [Model Tuning Guide](docs/MODEL_TUNING.md) for accuracy improvements
6. **GitHub Issues**: Create a new issue with detailed information including:
   - Environment details (Python 3.13, .venv, OS, ARM64/x86_64)
   - Configuration (AWS profile, region, environment variables)
   - Error messages and logs
   - Steps to reproduce
   - Expected vs actual behavior