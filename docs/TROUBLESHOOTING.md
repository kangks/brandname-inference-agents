# Troubleshooting Guide

This guide covers common issues and debugging steps for the Multilingual Product Inference System, including ARM64 platform considerations.

## Table of Contents

1. [Environment Setup Issues](#environment-setup-issues)
2. [AWS Configuration Problems](#aws-configuration-problems)
3. [ARM64 Platform Issues](#arm64-platform-issues)
4. [Agent-Specific Issues](#agent-specific-issues)
5. [Performance Issues](#performance-issues)
6. [Deployment Issues](#deployment-issues)
7. [Model Loading Issues](#model-loading-issues)
8. [Network and Connectivity Issues](#network-and-connectivity-issues)
9. [Debugging Tools and Commands](#debugging-tools-and-commands)

## Environment Setup Issues

### Python 3.13 Virtual Environment Issues

**Problem**: Cannot create or activate Python 3.13 virtual environment
```bash
python3.13: command not found
```

**Solution**:
```bash
# On macOS with Homebrew
brew install python@3.13

# On Ubuntu/Debian
sudo apt update
sudo apt install python3.13 python3.13-venv

# Verify installation
python3.13 --version

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate
```

**Problem**: Virtual environment activation fails
```bash
source .venv/bin/activate
# No response or error
```

**Solution**:
```bash
# Check if .venv directory exists
ls -la .venv/

# Recreate if corrupted
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate

# Verify activation
which python
python --version  # Should show Python 3.13.x
```

### Dependency Installation Issues

**Problem**: Package installation fails with ARM64 compatibility issues
```bash
ERROR: Failed building wheel for [package]
```

**Solution**:
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with ARM64 compatibility
pip install --no-cache-dir -r requirements.txt

# For specific ARM64 issues
export ARCHFLAGS="-arch arm64"
pip install [problematic-package]
```

**Problem**: spaCy model download fails
```bash
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution**:
```bash
# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download th_core_news_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy working')"
```

## AWS Configuration Problems

### ml-sandbox Profile Issues

**Problem**: AWS profile not configured or invalid
```bash
The config profile (ml-sandbox) could not be found
```

**Solution**:
```bash
# Configure ml-sandbox profile
aws configure --profile ml-sandbox
# Enter Access Key ID, Secret Access Key, region (us-east-1), output format (json)

# Verify configuration
aws sts get-caller-identity --profile ml-sandbox
aws configure list --profile ml-sandbox
```

**Problem**: Insufficient permissions for ml-sandbox profile
```bash
AccessDenied: User is not authorized to perform operation
```

**Solution**:
```bash
# Check current permissions
aws iam get-user --profile ml-sandbox
aws iam list-attached-user-policies --user-name [username] --profile ml-sandbox

# Required permissions for ml-sandbox:
# - AmazonBedrockFullAccess
# - AmazonS3FullAccess
# - AmazonECS_FullAccess
# - CloudWatchFullAccess
# - IAMReadOnlyAccess
```

### Region Configuration Issues

**Problem**: Services not available in us-east-1
```bash
InvalidRegion: The region 'us-east-1' is not supported
```

**Solution**:
```bash
# Verify region configuration
aws configure get region --profile ml-sandbox

# Set correct region
aws configure set region us-east-1 --profile ml-sandbox

# Test region access
aws ec2 describe-regions --region us-east-1 --profile ml-sandbox
```

## ARM64 Platform Issues

### Docker Build Issues on ARM64

**Problem**: Docker build fails for ARM64 platform
```bash
ERROR: failed to solve: no match for platform in manifest
```

**Solution**:
```bash
# Build with explicit ARM64 platform
docker build --platform linux/arm64 -t multilingual-inference .

# Use buildx for multi-platform builds
docker buildx create --use
docker buildx build --platform linux/arm64 -t multilingual-inference .

# Check available platforms
docker buildx ls
```

**Problem**: Base image not available for ARM64
```bash
ERROR: python:3.13-slim: no matching manifest for linux/arm64
```

**Solution**:
```dockerfile
# Use ARM64-compatible base image
FROM --platform=linux/arm64 python:3.13-slim

# Or use multi-arch image
FROM python:3.13-slim
```

### ECS ARM64 Deployment Issues

**Problem**: ECS task fails to start on ARM64
```bash
CannotPullContainerError: pull image manifest has been retried
```

**Solution**:
```json
// In ECS task definition
{
  "runtimePlatform": {
    "cpuArchitecture": "ARM64",
    "operatingSystemFamily": "LINUX"
  },
  "requiresCompatibilities": ["FARGATE"]
}
```

**Problem**: ARM64 performance issues
```bash
Task running slowly or timing out
```

**Solution**:
```bash
# Increase CPU and memory allocation
# ARM64 Graviton processors may need different resource allocation

# In task definition:
"cpu": "1024",      # Increase from 512
"memory": "2048"    # Increase from 1024
```

## Agent-Specific Issues

### NER Agent Issues

**Problem**: spaCy model loading fails
```bash
OSError: [E050] Can't find model
```

**Solution**:
```bash
# Check model installation
python -c "import spacy; print(spacy.util.find_model('en_core_web_sm'))"

# Reinstall model
pip uninstall spacy
pip install spacy
python -m spacy download en_core_web_sm

# Verify model path in configuration
python -c "from inference.config.settings import config; print(config.ner_model_path)"
```

**Problem**: Low NER confidence scores
```bash
All NER predictions below confidence threshold
```

**Solution**:
```python
# Adjust confidence threshold in configuration
from inference.config.settings import config
config.ner_confidence_threshold = 0.5  # Lower from 0.8

# Check training data quality
python training/preprocessing/ner_data_validator.py

# Retrain with more data
python training/pipelines/train_ner_pipeline.py
```

### RAG Agent Issues

**Problem**: Milvus connection fails
```bash
MilvusException: <RpcError of RPC that terminated with: status = UNAVAILABLE>
```

**Solution**:
```bash
# Check Milvus service status
curl http://localhost:19530/health

# Restart Milvus service
docker restart milvus-standalone

# Check Milvus logs
docker logs milvus-standalone

# Verify network connectivity
telnet localhost 19530
```

**Problem**: Vector embeddings not found
```bash
Collection 'product_embeddings' does not exist
```

**Solution**:
```bash
# Create Milvus collection
python training/preprocessing/milvus_collection_manager.py create

# Populate with embeddings
python training/pipelines/train_rag_pipeline.py

# Verify collection
python -c "from pymilvus import connections, Collection; connections.connect(); print(Collection('product_embeddings').num_entities)"
```

### LLM Agent Issues

**Problem**: AWS Bedrock access denied
```bash
AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```

**Solution**:
```bash
# Check Bedrock permissions
aws bedrock list-foundation-models --profile ml-sandbox --region us-east-1

# Verify model access
aws bedrock get-foundation-model --model-identifier amazon.nova-pro-v1:0 --profile ml-sandbox --region us-east-1

# Request model access in AWS Console if needed
```

**Problem**: Nova Pro model not available
```bash
ValidationException: The provided model identifier is invalid
```

**Solution**:
```bash
# List available models
aws bedrock list-foundation-models --profile ml-sandbox --region us-east-1

# Check model ID in configuration
python -c "from inference.config.settings import config; print(config.bedrock_model_id)"

# Update model ID if needed
export BEDROCK_MODEL_ID="amazon.nova-pro-v1:0"
```

## Performance Issues

### High Latency Issues

**Problem**: Inference requests taking too long
```bash
Request timeout after 30 seconds
```

**Solution**:
```bash
# Increase timeout settings
export AGENT_TIMEOUT=60
export REQUEST_TIMEOUT=120

# Check agent performance
python tests/performance/benchmark_runner.py

# Optimize model loading
# - Use model caching
# - Reduce model size
# - Use faster inference backends
```

**Problem**: Memory usage too high
```bash
MemoryError: Unable to allocate memory
```

**Solution**:
```bash
# Monitor memory usage
python tests/performance/performance_metrics.py

# Reduce batch sizes
export MAX_BATCH_SIZE=16  # Reduce from 32

# Increase container memory
# In ECS task definition: "memory": "4096"

# Use memory-efficient models
export USE_QUANTIZED_MODELS=true
```

### Throughput Issues

**Problem**: Low concurrent request handling
```bash
Service unavailable under load
```

**Solution**:
```bash
# Increase worker processes
export WORKER_PROCESSES=4

# Enable auto-scaling
# Configure ECS auto-scaling policies

# Use connection pooling
export CONNECTION_POOL_SIZE=20

# Monitor with CloudWatch
aws logs tail /aws/ecs/multilingual-inference --follow --profile ml-sandbox
```

## Deployment Issues

### CloudFormation Stack Issues

**Problem**: Stack creation fails
```bash
CREATE_FAILED: Resource creation cancelled
```

**Solution**:
```bash
# Check CloudFormation events
aws cloudformation describe-stack-events --stack-name multilingual-inference-stack --profile ml-sandbox --region us-east-1

# Validate template
aws cloudformation validate-template --template-body file://infrastructure/cloudformation/main-stack.yaml --profile ml-sandbox

# Check resource limits
aws service-quotas get-service-quota --service-code ecs --quota-code L-34B43A08 --profile ml-sandbox --region us-east-1
```

**Problem**: ECS service deployment fails
```bash
Service failed to reach steady state
```

**Solution**:
```bash
# Check ECS service events
aws ecs describe-services --cluster multilingual-inference --services orchestrator-service --profile ml-sandbox --region us-east-1

# Check task definition
aws ecs describe-task-definition --task-definition orchestrator-agent --profile ml-sandbox --region us-east-1

# Check container logs
aws logs get-log-events --log-group-name /ecs/orchestrator-agent --log-stream-name [stream-name] --profile ml-sandbox --region us-east-1
```

### Container Registry Issues

**Problem**: Image push to ECR fails
```bash
denied: requested access to the resource is denied
```

**Solution**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 --profile ml-sandbox | docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

# Create repository if needed
aws ecr create-repository --repository-name multilingual-inference --region us-east-1 --profile ml-sandbox

# Tag and push image
docker tag multilingual-inference:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/multilingual-inference:latest
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/multilingual-inference:latest
```

## Model Loading Issues

### Model File Not Found

**Problem**: Model files missing or corrupted
```bash
FileNotFoundError: Model file not found at path
```

**Solution**:
```bash
# Check model registry
python -c "from inference.config.model_registry import ModelRegistry; registry = ModelRegistry(); print(registry.list_models())"

# Download missing models
python training/pipelines/model_manager.py download --model-type ner
python training/pipelines/model_manager.py download --model-type embedding

# Verify model integrity
python training/pipelines/model_manager.py validate --model-type all
```

### Model Version Conflicts

**Problem**: Model version mismatch
```bash
VersionError: Model version 2.1 not compatible with agent version 1.0
```

**Solution**:
```bash
# Check model versions
python -c "from inference.config.model_registry import ModelRegistry; registry = ModelRegistry(); print(registry.get_model_info('ner'))"

# Update to compatible version
python training/pipelines/model_manager.py update --model-type ner --version 1.0

# Rebuild models if needed
python training/pipelines/train_ner_pipeline.py --force-rebuild
```

## Network and Connectivity Issues

### Milvus Connection Issues

**Problem**: Cannot connect to Milvus
```bash
ConnectionError: Failed to connect to Milvus server
```

**Solution**:
```bash
# Check Milvus status
docker ps | grep milvus
curl http://localhost:19530/health

# Restart Milvus
docker-compose -f infrastructure/milvus/milvus.yaml restart

# Check network configuration
docker network ls
docker network inspect [network-name]

# Test connectivity
telnet localhost 19530
```

### AWS Service Connectivity

**Problem**: Cannot reach AWS services
```bash
EndpointConnectionError: Could not connect to the endpoint URL
```

**Solution**:
```bash
# Check internet connectivity
ping aws.amazon.com

# Verify AWS endpoints
aws sts get-caller-identity --profile ml-sandbox --region us-east-1

# Check VPC configuration (if using VPC)
aws ec2 describe-vpcs --profile ml-sandbox --region us-east-1

# Test specific service endpoints
curl https://bedrock-runtime.us-east-1.amazonaws.com/
```

## Debugging Tools and Commands

### System Health Checks

```bash
# Overall system health
python inference/monitoring/health_checker.py

# Individual agent health
curl http://localhost:8080/agents/ner/health
curl http://localhost:8080/agents/rag/health
curl http://localhost:8080/agents/llm/health

# Infrastructure health
python infrastructure/monitoring/milvus-health-check.py
```

### Diagnostic Commands

```bash
# Run comprehensive diagnostics
python inference/monitoring/diagnostics.py --full

# Check configuration
python -c "from inference.config.settings import config; config.validate()"

# Test AWS connectivity
aws sts get-caller-identity --profile ml-sandbox
aws bedrock list-foundation-models --profile ml-sandbox --region us-east-1

# Test model loading
python -c "from inference.agents.ner_agent import NERAgent; agent = NERAgent(); print('NER agent loaded')"
```

### Log Analysis

```bash
# View application logs
tail -f logs/inference.log

# Filter by log level
grep "ERROR" logs/inference.log | tail -20
grep "WARNING" logs/inference.log | tail -20

# View CloudWatch logs (production)
aws logs tail /aws/ecs/multilingual-inference --follow --profile ml-sandbox --region us-east-1

# Search logs for specific errors
aws logs filter-log-events --log-group-name /aws/ecs/multilingual-inference --filter-pattern "ERROR" --profile ml-sandbox --region us-east-1
```

### Performance Monitoring

```bash
# Monitor resource usage
python tests/performance/performance_metrics.py --monitor

# Benchmark inference speed
python tests/performance/benchmark_runner.py --agents all

# Memory profiling
python -m memory_profiler inference/main.py

# CPU profiling
python -m cProfile -o profile.stats inference/main.py
```

### Configuration Debugging

```bash
# Validate configuration
python -c "from inference.config.validators import validate_config; validate_config()"

# Check environment variables
env | grep -E "(AWS_|INFERENCE_|MODEL_)"

# Test model registry
python -c "from inference.config.model_registry import ModelRegistry; registry = ModelRegistry(); print(registry.validate_all())"
```

## Getting Help

If you're still experiencing issues after trying these solutions:

1. **Check the logs**: Always start with application and system logs
2. **Verify configuration**: Ensure all environment variables and configuration files are correct
3. **Test components individually**: Isolate the problem to specific components
4. **Check AWS status**: Verify AWS service status at https://status.aws.amazon.com/
5. **Review recent changes**: Consider what changed since the system last worked
6. **Create minimal reproduction**: Try to reproduce the issue with minimal configuration

### Reporting Issues

When reporting issues, include:

- **Environment details**: Python version, OS, ARM64/x86_64 architecture
- **Configuration**: Relevant configuration files and environment variables
- **Error messages**: Complete error messages and stack traces
- **Steps to reproduce**: Exact steps that lead to the issue
- **Expected vs actual behavior**: What you expected vs what happened
- **Logs**: Relevant log entries around the time of the issue

### Emergency Procedures

For production issues:

1. **Check service health**: `curl http://[load-balancer]/health`
2. **Scale up if needed**: Increase ECS service desired count
3. **Check CloudWatch alarms**: Review any triggered alarms
4. **Rollback if necessary**: Revert to previous working deployment
5. **Enable debug logging**: Temporarily increase log verbosity
6. **Contact support**: Escalate to appropriate support channels