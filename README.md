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

## 🚀 Production Inference Testing

The inference system is deployed and accessible via production endpoint:

### Quick Test
```bash
# Test the production inference endpoint
curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}'
```

### Multilingual Testing
```bash
# Test Thai product name
curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "ยาสีฟัน Colgate Total", "language_hint": "th"}'
```

### Expected Response
The orchestrator coordinates multiple agents (NER, RAG, LLM, Hybrid) and returns:
```json
{
  "input": {"product_name": "iPhone 15 Pro Max", "language_hint": "en"},
  "best_prediction": "Apple",
  "best_confidence": 0.95,
  "agent_results": {...},
  "orchestration_time": 0.245
}
```

See [Inference Testing Guide](INFERENCE_TESTING_GUIDE.md) for comprehensive testing examples.

### Production Usage

1. **Test inference endpoint:**
   ```bash
   curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
     -H "Content-Type: application/json" \
     -d '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}'
   ```

2. **Check system health:**
   ```bash
   curl http://production-alb-107602758.us-east-1.elb.amazonaws.com/health
   ```

3. **Process multilingual products:**
   ```bash
   # English product
   curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
     -H "Content-Type: application/json" \
     -d '{"product_name": "Samsung Galaxy S24", "language_hint": "en"}'
   
   # Thai product  
   curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
     -H "Content-Type: application/json" \
     -d '{"product_name": "ยาสีฟัน Wonder smile toothpaste kid", "language_hint": "th"}'
   ```

4. **Local development server:**
   ```bash
   python -m inference.main serve --dev
   ```

## 📁 Project Structure

```
multilingual-product-inference/
├── inference/                    # ✅ Inference stack (DEPLOYED)
│   ├── agents/                  # Multi-agent system implementations
│   │   ├── ner_agent.py        # Named Entity Recognition agent
│   │   ├── rag_agent.py        # Retrieval-Augmented Generation agent
│   │   ├── llm_agent.py        # Large Language Model agent
│   │   ├── hybrid_agent.py     # Hybrid ensemble agent
│   │   └── orchestrator_agent.py # Orchestrator coordination agent
│   ├── config/                 # Configuration management
│   ├── models/                 # Data models and schemas
│   ├── infrastructure/         # AWS deployment scripts
│   └── main.py                 # FastAPI application entry point
├── training/                    # 🔄 Training stack (NEXT PHASE)
│   ├── data/                   # Training data processing
│   ├── preprocessing/          # Data cleaning and preparation
│   ├── pipelines/             # Model training pipelines
│   ├── evaluation/            # Model validation and testing
│   └── __init__.py
├── training_dataset.txt        # 📊 Multilingual product dataset (Thai/English)
├── tests/                      # Comprehensive test suite
├── examples/                   # Usage examples and testing scripts
├── Dockerfile                  # ARM64 production container
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

### Training Dataset

The `training_dataset.txt` contains **multilingual e-commerce product data** with:
- **Product names** in Thai, English, and mixed languages
- **Brand labels** for supervised learning
- **Category information** for context
- **45+ product entries** covering diverse brands and categories

Example entry:
```json
{
  "brand": "Wonder Smile(วันเดอร์สไมล์)", 
  "product_name": "ยาสีฟัน Wonder smile toothpaste kid วันเดอร์สไมล์ ยาสีฟัน ยาสีฟันเด็ก",
  "category": "กลุ่มผลิตภัณฑ์เพื่อสุขภาพ"
}
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

## 🎓 Training System (Next Phase)

### Training Data Overview

The system includes a **multilingual product dataset** (`training_dataset.txt`) with:
- **45+ product entries** in Thai, English, and mixed languages
- **Brand extraction targets** for supervised learning
- **E-commerce categories** including electronics, fashion, beauty, and more
- **Real-world complexity** with mixed scripts and transliterations

### Training Pipeline Components

#### 1. **Data Preprocessing**
```bash
# Process multilingual training data
python -m training.preprocessing.clean_data --input training_dataset.txt
python -m training.preprocessing.extract_features --language mixed
```

#### 2. **Model Training**
- **NER Model**: Train spaCy custom model for brand entity recognition
- **Embedding Model**: Fine-tune sentence transformers for multilingual similarity
- **LLM Fine-tuning**: Train Nova Pro model via AWS Bedrock for brand reasoning
- **Vector Database**: Build Milvus collection from training embeddings

#### 3. **Training Commands** (To be implemented)
```bash
# Train NER model
python -m training.pipelines.train_ner --data training_dataset.txt --output models/ner_model

# Fine-tune embeddings
python -m training.pipelines.train_embeddings --data training_dataset.txt --model sentence-transformers

# Submit LLM fine-tuning job
python -m training.pipelines.train_llm --data training_dataset.txt --model nova-pro --aws-profile ml-sandbox

# Build knowledge base
python -m training.pipelines.build_knowledge_base --data training_dataset.txt --milvus-host production
```

### Training Data Examples

**Thai Product with English Brand:**
```json
{"brand": "BENQ(เบ็นคิว)", "product_name": "BenQ GW2785TC 27นิ้ว FHD Eye Care Monitor"}
```

**Mixed Language Fashion:**
```json
{"brand": "zanzea", "product_name": "Esolo ZANZEA ชุดสไตล์เกาหลีของผู้หญิงทางการแขนบาน"}
```

**Thai Brand with Mixed Description:**
```json
{"brand": "Wonder Smile(วันเดอร์สไมล์)", "product_name": "ยาสีฟัน Wonder smile toothpaste kid"}
```

## 🧪 Testing

### Inference Testing (Production Ready)

```bash
# Test production endpoint
curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}'

# Test with training data examples
curl -X POST http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "ยาสีฟัน Wonder smile toothpaste kid", "language_hint": "th"}'
```

### Development Testing

```bash
# All tests
make test

# Unit tests only
pytest tests/ -m "unit"

# Integration tests only  
pytest tests/ -m "integration"

# Training pipeline tests (to be implemented)
pytest tests/training/ -v
```

### Test Structure

- **Inference tests**: Test deployed production system
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Training tests**: Validate training pipeline components
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

**✅ Completed - Inference System:**
- ✅ **Project Structure**: Complete organization with inference and training stacks
- ✅ **Core Infrastructure**: Data models, configuration management, PEP 8 compliance
- ✅ **Agent Implementation**: NER, RAG, LLM, Hybrid, and Orchestrator agents fully implemented
- ✅ **Docker Deployment**: ARM64 containers built and pushed to ECR
- ✅ **AWS Infrastructure**: ECS cluster, services, load balancer, and Milvus vector database
- ✅ **Orchestrator Service**: Running and coordinating multiple inference agents
- ✅ **API Endpoint**: Production inference endpoint accessible at `http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer`

**🔧 Current Issue:**
- Load balancer health check configuration (final deployment step)

**🔄 Next Phase - Training System:**
- **Training Data Processing**: Process multilingual product dataset (Thai/English mixed)
- **Model Training Pipeline**: Implement NER model training, embedding fine-tuning, and LLM fine-tuning
- **Knowledge Base Creation**: Build vector database from training data for RAG system
- **Model Evaluation**: Implement accuracy testing and validation frameworks
- **Training Infrastructure**: Set up training jobs on AWS Bedrock and SageMaker

## 📚 Documentation

### System Status Guides

- **[Final Status Summary](FINAL_STATUS_SUMMARY.md)** - Complete deployment status and current system capabilities
- **[Inference Testing Guide](INFERENCE_TESTING_GUIDE.md)** - Production endpoint testing with examples and expected responses
- **[ECS Deployment Success Summary](ECS_DEPLOYMENT_SUCCESS_SUMMARY.md)** - Infrastructure deployment details and service status

### Implementation Guides

- **[Strands Multiagent Orchestrator](STRANDS_MULTIAGENT_ORCHESTRATOR_README.md)** - Agent coordination and orchestration patterns
- **[Working Inference Examples](WORKING_INFERENCE_EXAMPLES.md)** - Successful inference patterns and agent results
- **[Docker Testing Workflow](DOCKER_TESTING_WORKFLOW.md)** - Container testing and validation procedures

### Training Documentation (Next Phase)

- **Training Dataset**: `training_dataset.txt` - 45+ multilingual product entries for model training
- **Model Training Guide** (To be created) - NER, embedding, and LLM fine-tuning procedures
- **Knowledge Base Setup** (To be created) - Vector database creation from training data

### Quick Reference

- **Production Endpoint**: `http://production-alb-107602758.us-east-1.elb.amazonaws.com/infer`
- **Infrastructure**: ECS Fargate + Milvus + Application Load Balancer
- **Agent Architecture**: Orchestrator + NER + RAG + LLM + Hybrid agents
- **Training Data**: Multilingual Thai/English product dataset ready for processing

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

### Current System Status

**✅ Production Ready**: Inference system is deployed and accessible
**🔧 Minor Issue**: Load balancer health check configuration (system functional)
**🔄 Next Phase**: Training system implementation

### For Issues and Questions:

1. **Inference Testing**: Use the [Inference Testing Guide](INFERENCE_TESTING_GUIDE.md) for endpoint testing
2. **System Status**: Check [Final Status Summary](FINAL_STATUS_SUMMARY.md) for current capabilities
3. **Deployment Issues**: Review [ECS Deployment Success Summary](ECS_DEPLOYMENT_SUCCESS_SUMMARY.md)
4. **Training Questions**: Refer to `training_dataset.txt` and training documentation (to be created)
5. **GitHub Issues**: Create a new issue with detailed information including:
   - **Inference Issues**: Include endpoint URL, request payload, and response
   - **Training Issues**: Include dataset details and training pipeline step
   - **Infrastructure Issues**: Include AWS service logs and configuration
   - Environment details (Python 3.13, AWS profile ml-sandbox, region us-east-1)

### Quick Debugging

```bash
# Test production inference
curl http://production-alb-107602758.us-east-1.elb.amazonaws.com/health

# Check AWS services
aws ecs describe-services --cluster multilingual-inference-cluster --profile ml-sandbox --region us-east-1

# View training data
head -5 training_dataset.txt
```