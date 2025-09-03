# Local Testing Guide

This guide provides comprehensive instructions for setting up and running local tests for the Multilingual Product Inference System using mock services in a Python 3.13 .venv environment.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Mock Services Overview](#mock-services-overview)
3. [Local Development Server](#local-development-server)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Accuracy Testing](#accuracy-testing)
8. [End-to-End Testing](#end-to-end-testing)
9. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
10. [Continuous Integration](#continuous-integration)

## Environment Setup

### 1. Python 3.13 Virtual Environment Setup

```bash
# Ensure Python 3.13 is installed
python3.13 --version

# Create virtual environment
python3.13 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Verify activation
which python  # Should point to .venv/bin/python
python --version  # Should show Python 3.13.x
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# Install development and testing dependencies
pip install -r requirements-dev.txt

# Or install specific testing packages
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install black flake8 mypy isort
pip install httpx aioresponses fakeredis
```

### 3. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env.local

# Edit local configuration
export INFERENCE_ENV=local
export LOG_LEVEL=DEBUG
export USE_MOCK_SERVICES=true
export MOCK_AWS_SERVICES=true
export MOCK_MILVUS=true
export MOCK_SPACY=true
```

Edit `.env.local`:
```bash
# Local Testing Configuration
INFERENCE_ENV=local
LOG_LEVEL=DEBUG

# Mock Service Configuration
USE_MOCK_SERVICES=true
MOCK_AWS_SERVICES=true
MOCK_MILVUS=true
MOCK_SPACY=true
MOCK_SENTENCE_TRANSFORMERS=true

# Local Service Ports
ORCHESTRATOR_PORT=8080
NER_AGENT_PORT=8081
RAG_AGENT_PORT=8082
LLM_AGENT_PORT=8083
HYBRID_AGENT_PORT=8084

# Test Data Paths
TEST_DATA_DIR=tests/data
MOCK_MODELS_DIR=tests/mocks/models
TRAINING_DATA_PATH=tests/data/training_dataset.txt

# Performance Testing
MAX_CONCURRENT_REQUESTS=10
TEST_TIMEOUT=30
BENCHMARK_ITERATIONS=100
```

### 4. Verify Installation

```bash
# Run installation verification
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')

# Test imports
try:
    import pytest
    import asyncio
    import aiohttp
    print('✓ Testing dependencies installed')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')

try:
    from inference.config.settings import config
    print('✓ Inference package importable')
except ImportError as e:
    print(f'✗ Inference package issue: {e}')
"
```

## Mock Services Overview

The system provides comprehensive mock services to enable local testing without external dependencies.

### Available Mock Services

1. **Mock AWS Services** (`tests/mocks/mock_aws_services.py`)
   - Bedrock LLM inference
   - S3 operations
   - CloudWatch logging

2. **Mock Milvus Service** (`tests/mocks/mock_milvus_service.py`)
   - Vector database operations
   - Collection management
   - Similarity search

3. **Mock spaCy Service** (`tests/mocks/mock_spacy_service.py`)
   - NER model loading
   - Entity extraction
   - Multilingual processing

4. **Mock Sentence Transformers** (`tests/mocks/mock_sentence_transformers.py`)
   - Embedding generation
   - Model loading
   - Batch processing

### Mock Service Configuration

```python
# Configure mock services in tests/conftest.py
import pytest
from tests.mocks.mock_aws_services import MockAWSServices
from tests.mocks.mock_milvus_service import MockMilvusService
from tests.mocks.mock_spacy_service import MockSpacyService
from tests.mocks.mock_sentence_transformers import MockSentenceTransformers

@pytest.fixture(scope="session")
def mock_services():
    """Set up all mock services for testing."""
    services = {
        'aws': MockAWSServices(),
        'milvus': MockMilvusService(),
        'spacy': MockSpacyService(),
        'sentence_transformers': MockSentenceTransformers()
    }
    
    # Start mock services
    for service in services.values():
        service.start()
    
    yield services
    
    # Cleanup
    for service in services.values():
        service.stop()
```

## Local Development Server

### 1. Start Local Development Server

```bash
# Start with mock services
python inference/main.py --dev --use-mocks

# Or use the development script
python tests/examples/demo_local_testing.py --start-server

# Start individual agents for debugging
python -m inference.agents.ner_agent --port 8081 --mock
python -m inference.agents.rag_agent --port 8082 --mock
python -m inference.agents.llm_agent --port 8083 --mock
```

### 2. Test Local Server

```bash
# Check server health
curl http://localhost:8080/health

# Test inference endpoint
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "iPhone 15 Pro Max", "language": "en"}'

# Test individual agents
curl http://localhost:8081/extract -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "iPhone 15 Pro Max"}'

curl http://localhost:8082/retrieve -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "iPhone 15 Pro Max", "top_k": 5}'
```

### 3. Interactive Development

```python
# Start interactive Python session
python -i tests/examples/demo_local_testing.py

# Test agents interactively
>>> from inference.agents.ner_agent import NERAgent
>>> agent = NERAgent(use_mock=True)
>>> result = await agent.extract_entities("iPhone 15 Pro Max")
>>> print(result)

>>> from inference.agents.rag_agent import RAGAgent
>>> rag_agent = RAGAgent(use_mock=True)
>>> result = await rag_agent.retrieve_and_infer("Samsung Galaxy S24")
>>> print(result)
```

## Unit Testing

### 1. Run All Unit Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=inference --cov=training --cov-report=html

# Run specific test files
python -m pytest tests/test_ner_agent.py -v
python -m pytest tests/test_rag_agent.py -v
python -m pytest tests/test_llm_agent.py -v
python -m pytest tests/test_orchestrator_agent.py -v
```

### 2. Test Individual Components

```bash
# Test configuration system
python -m pytest tests/test_config.py -v
python -m pytest tests/test_config_validators.py -v
python -m pytest tests/test_model_registry.py -v

# Test data models
python -m pytest tests/test_data_models.py -v

# Test monitoring components
python -m pytest tests/test_monitoring.py -v
```

### 3. Test with Different Configurations

```bash
# Test with different mock configurations
MOCK_AWS_SERVICES=false python -m pytest tests/test_llm_agent.py -v
MOCK_MILVUS=false python -m pytest tests/test_rag_agent.py -v
USE_MOCK_SERVICES=false python -m pytest tests/integration/ -v
```

### 4. Parallel Testing

```bash
# Run tests in parallel
python -m pytest tests/ -n auto

# Run with specific number of workers
python -m pytest tests/ -n 4
```

## Integration Testing

### 1. Agent Integration Tests

```bash
# Run integration test suite
python tests/integration/workflow_tester.py

# Test agent communication
python -m pytest tests/integration/ -v -k "test_agent_communication"

# Test end-to-end workflows
python -m pytest tests/integration/ -v -k "test_end_to_end"
```

### 2. Mock Service Integration

```python
# Example integration test
import pytest
from tests.integration.workflow_tester import WorkflowTester

@pytest.mark.asyncio
async def test_full_inference_pipeline():
    """Test complete inference pipeline with mock services."""
    tester = WorkflowTester(use_mocks=True)
    
    # Test single inference
    result = await tester.test_single_inference("iPhone 15 Pro Max")
    assert result['success']
    assert 'ner_result' in result
    assert 'rag_result' in result
    assert 'llm_result' in result
    
    # Test batch inference
    batch_results = await tester.test_batch_inference([
        "Samsung Galaxy S24",
        "ยาสีฟัน Wonder smile",
        "MacBook Pro M3"
    ])
    assert len(batch_results) == 3
    assert all(r['success'] for r in batch_results)
```

### 3. Configuration Integration Tests

```bash
# Test configuration loading and validation
python -c "
from inference.config.settings import config
from inference.config.validators import validate_config

# Test configuration loading
print('Testing configuration loading...')
config.load_from_file('.env.local')
print('✓ Configuration loaded')

# Test validation
print('Testing configuration validation...')
validation_result = validate_config()
print(f'Validation result: {validation_result}')
"
```

## Performance Testing

### 1. Benchmark Individual Agents

```bash
# Run performance benchmarks
python tests/performance/benchmark_runner.py --agents all --iterations 100

# Benchmark specific agents
python tests/performance/benchmark_runner.py --agents ner --iterations 1000
python tests/performance/benchmark_runner.py --agents rag --iterations 500
python tests/performance/benchmark_runner.py --agents llm --iterations 100

# Benchmark with different configurations
python tests/performance/benchmark_runner.py --mock-services --iterations 1000
```

### 2. Load Testing

```python
# Example load test
import asyncio
import time
from tests.performance.performance_metrics import PerformanceMetrics

async def load_test():
    """Run load test with concurrent requests."""
    metrics = PerformanceMetrics()
    
    # Configure test parameters
    concurrent_users = 10
    test_duration = 60  # seconds
    
    # Run load test
    results = await metrics.run_load_test(
        endpoint="http://localhost:8080/infer",
        concurrent_users=concurrent_users,
        duration=test_duration,
        test_data="tests/data/load_test_queries.json"
    )
    
    print(f"Load test results: {results}")
    return results

# Run load test
asyncio.run(load_test())
```

### 3. Memory and CPU Profiling

```bash
# Profile memory usage
python -m memory_profiler tests/performance/memory_profile_test.py

# Profile CPU usage
python -m cProfile -o profile.stats tests/performance/cpu_profile_test.py

# Analyze profile results
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
"
```

### 4. Performance Monitoring

```python
# Monitor performance during testing
from tests.performance.performance_metrics import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Run your tests here
# ...

metrics = monitor.stop_monitoring()
print(f"Performance metrics: {metrics}")
```

## Accuracy Testing

### 1. Run Accuracy Validation

```bash
# Run accuracy tests with test dataset
python tests/accuracy/accuracy_validator.py \
  --test-data tests/data/test_queries.json \
  --ground-truth tests/data/ground_truth.json \
  --agents all \
  --output accuracy_report.json

# Test specific agents
python tests/accuracy/accuracy_validator.py \
  --agents ner \
  --test-data tests/data/ner_test_data.json \
  --output ner_accuracy.json
```

### 2. Test Data Management

```python
# Manage test datasets
from tests.accuracy.test_data_manager import TestDataManager

manager = TestDataManager()

# Load test data
test_data = manager.load_test_data("tests/data/test_queries.json")
ground_truth = manager.load_ground_truth("tests/data/ground_truth.json")

# Validate data consistency
validation_result = manager.validate_test_data(test_data, ground_truth)
print(f"Data validation: {validation_result}")

# Generate additional test cases
additional_cases = manager.generate_test_cases(
    base_cases=test_data,
    variations=['multilingual', 'misspellings', 'abbreviations']
)
```

### 3. Accuracy Metrics

```python
# Calculate detailed accuracy metrics
from tests.accuracy.accuracy_validator import AccuracyValidator

validator = AccuracyValidator()

# Test NER accuracy
ner_results = validator.test_ner_accuracy(
    test_cases="tests/data/ner_test_cases.json",
    use_mock=True
)

# Test RAG accuracy
rag_results = validator.test_rag_accuracy(
    test_cases="tests/data/rag_test_cases.json",
    use_mock=True
)

# Generate accuracy report
report = validator.generate_accuracy_report([ner_results, rag_results])
print(report)
```

## End-to-End Testing

### 1. Complete Workflow Testing

```bash
# Run end-to-end tests
python tests/run_local_tests.py --full-suite

# Test specific workflows
python tests/run_local_tests.py --workflow multilingual
python tests/run_local_tests.py --workflow performance
python tests/run_local_tests.py --workflow accuracy
```

### 2. Scenario-Based Testing

```python
# Test different scenarios
from tests.framework.test_framework import TestFramework

framework = TestFramework(use_mocks=True)

# Test multilingual scenarios
multilingual_results = framework.run_scenario('multilingual', {
    'test_cases': [
        'iPhone 15 Pro Max',
        'ยาสีฟัน Wonder smile toothpaste',
        'Samsung Galaxy S24 Ultra',
        'MacBook Pro M3 Max'
    ]
})

# Test edge cases
edge_case_results = framework.run_scenario('edge_cases', {
    'test_cases': [
        '',  # Empty string
        'a',  # Single character
        'x' * 1000,  # Very long string
        '123456789',  # Numbers only
        '!@#$%^&*()',  # Special characters only
    ]
})
```

### 3. Regression Testing

```bash
# Run regression tests
python tests/framework/test_framework.py --regression

# Compare with baseline results
python tests/framework/test_framework.py \
  --compare-baseline tests/data/baseline_results.json \
  --output regression_report.json
```

## Debugging and Troubleshooting

### 1. Debug Mode Testing

```bash
# Run tests in debug mode
python -m pytest tests/ -v -s --log-cli-level=DEBUG

# Debug specific test
python -m pytest tests/test_ner_agent.py::test_extract_entities -v -s --pdb

# Run with detailed output
python tests/run_local_tests.py --debug --verbose
```

### 2. Mock Service Debugging

```python
# Debug mock services
from tests.mocks.mock_aws_services import MockAWSServices

# Enable debug logging for mock services
import logging
logging.basicConfig(level=logging.DEBUG)

# Test mock service directly
mock_aws = MockAWSServices(debug=True)
mock_aws.start()

# Test Bedrock mock
response = mock_aws.bedrock_client.invoke_model(
    modelId="amazon.nova-pro-v1:0",
    body=json.dumps({"prompt": "Test prompt"})
)
print(f"Mock response: {response}")
```

### 3. Test Data Debugging

```bash
# Validate test data
python -c "
from tests.accuracy.test_data_manager import TestDataManager

manager = TestDataManager()
validation = manager.validate_all_test_data()
print(f'Test data validation: {validation}')

# Check for missing or corrupted files
missing_files = manager.check_missing_files()
if missing_files:
    print(f'Missing test files: {missing_files}')
else:
    print('All test files present')
"
```

### 4. Performance Debugging

```python
# Debug performance issues
from tests.performance.performance_metrics import PerformanceDebugger

debugger = PerformanceDebugger()

# Profile specific function
@debugger.profile
async def test_function():
    # Your test code here
    pass

# Run with profiling
await test_function()

# Get profiling results
results = debugger.get_results()
print(f"Performance profile: {results}")
```

## Continuous Integration

### 1. CI Test Configuration

```yaml
# .github/workflows/test.yml
name: Local Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.13
      uses: actions/setup-python@v4
      with:
        python-version: '3.13'
    
    - name: Create virtual environment
      run: |
        python -m venv .venv
        source .venv/bin/activate
        echo "VIRTUAL_ENV=$VIRTUAL_ENV" >> $GITHUB_ENV
        echo "$VIRTUAL_ENV/bin" >> $GITHUB_PATH
    
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        export USE_MOCK_SERVICES=true
        python -m pytest tests/ --cov=inference --cov=training
    
    - name: Run local integration tests
      run: |
        python tests/run_local_tests.py --ci-mode
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.13

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python -m pytest tests/ --maxfail=1
        language: system
        pass_filenames: false
        always_run: true
EOF
```

### 3. Automated Test Execution

```bash
# Create test automation script
cat > scripts/run_all_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "Setting up Python 3.13 environment..."
python3.13 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo "Running code quality checks..."
black --check .
flake8 .
isort --check-only .

echo "Running unit tests..."
python -m pytest tests/ --cov=inference --cov=training

echo "Running integration tests..."
python tests/integration/workflow_tester.py

echo "Running performance tests..."
python tests/performance/benchmark_runner.py --quick

echo "Running accuracy tests..."
python tests/accuracy/accuracy_validator.py --quick

echo "All tests completed successfully!"
EOF

chmod +x scripts/run_all_tests.sh
```

## Test Execution Examples

### Quick Test Suite

```bash
# Run quick test suite (< 5 minutes)
python tests/run_local_tests.py --quick

# This includes:
# - Basic unit tests
# - Mock service validation
# - Simple integration tests
# - Basic performance checks
```

### Full Test Suite

```bash
# Run comprehensive test suite (15-30 minutes)
python tests/run_local_tests.py --full

# This includes:
# - All unit tests
# - Complete integration tests
# - Performance benchmarks
# - Accuracy validation
# - End-to-end scenarios
```

### Custom Test Execution

```python
# Custom test execution
from tests.framework.test_framework import TestFramework

framework = TestFramework()

# Configure test execution
framework.configure({
    'use_mocks': True,
    'parallel_execution': True,
    'max_workers': 4,
    'timeout': 300,
    'retry_failed': True,
    'generate_report': True
})

# Run custom test suite
results = framework.run_tests([
    'unit_tests',
    'integration_tests',
    'performance_tests'
])

print(f"Test results: {results}")
```

This comprehensive local testing guide provides everything needed to set up and run thorough local testing of the Multilingual Product Inference System using Python 3.13 in a .venv environment with comprehensive mock services.