# Multilingual Product Inference System - Customer Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Multi-Agent Architecture](#multi-agent-architecture)
3. [Implementation Details](#implementation-details)
4. [AWS Deployment Architecture](#aws-deployment-architecture)
5. [Deployment Prerequisites](#deployment-prerequisites)
6. [Step-by-Step Deployment Guide](#step-by-step-deployment-guide)
7. [Post-Deployment Validation](#post-deployment-validation)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

## System Overview

The Multilingual Product Inference System is a sophisticated AI-powered solution designed to extract brand names from product descriptions in multiple languages (English, Thai, and mixed-language content). The system leverages a multi-agent architecture built on the Strands Agents SDK, deployed on AWS using containerized microservices.

### Key Features
- **Multi-Agent Orchestration**: Advanced coordination using Strands Agents SDK v1.7.1
- **Multilingual Support**: Handles English, Thai, and mixed-language product descriptions
- **Multiple Inference Methods**: NER, RAG, LLM, and Hybrid approaches
- **Scalable Architecture**: AWS ECS Fargate with auto-scaling capabilities
- **Production-Ready**: Comprehensive monitoring, logging, and error handling

### Architecture Diagram

![Inference Architecture](./generated-diagrams/inference_architecture_diagram.png)

**Component Flow Annotations:**
- ❶ **API Requests**: Users send product inference requests via REST API
- ❷ **Load Balancing**: ALB routes requests to available orchestrator instances
- ❸-❻ **Agent Coordination**: Orchestrator coordinates with specialized agents
- ❼ **LLM Inference**: LLM agent uses Amazon Bedrock for advanced reasoning
- ❽ **Vector Search**: RAG agent performs similarity search in Milvus
- ❾ **Pipeline Processing**: Hybrid agent combines multiple approaches
- ❿ **Persistent Storage**: Milvus data stored on EFS for durability
- ⓫ **Model Artifacts**: Pre-trained models and embeddings stored in S3
- ⓬ **Monitoring**: All services send metrics and logs to CloudWatch
- ⓭ **Tracing**: Distributed tracing via AWS X-Ray
- ⓮ **Container Images**: Services pull Docker images from ECR

## Multi-Agent Architecture

### Strands Agents SDK Integration

The system is built on **Strands Agents SDK v1.7.1**, providing advanced multi-agent capabilities:

```python
from strands import Agent, tool
from strands.multiagent import Swarm, GraphBuilder
```

### Core Components

#### 1. Orchestrator Agent (`StrandsMultiAgentOrchestrator`)

The central coordinator that manages all specialized agents using Strands multi-agent tools:

**Key Features:**
- **Model**: Claude 3.7 Sonnet via Amazon Bedrock (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
- **Multi-Agent Tools**: Integrates `agent_graph`, `swarm`, `workflow`, and `journal`
- **Dynamic Agent Creation**: Runtime creation of specialized agents
- **Coordination Strategies**: Multiple coordination methods for different use cases

**Coordination Methods:**
1. **Swarm Coordination**: Parallel execution of multiple agents
2. **Agent Graph**: Structured workflows with dependency management
3. **Workflow**: Sequential processing pipelines
4. **Enhanced Fallback**: Robust fallback when tools are unavailable

#### 2. Specialized Agents

##### NER Agent (Named Entity Recognition)
- **Purpose**: Extract brand entities from product text
- **Technologies**: spaCy, Transformers, multilingual models
- **Capabilities**:
  - Multilingual entity recognition (English/Thai)
  - Confidence scoring for each extraction
  - Position tracking for entities
  - Fallback pattern matching

##### RAG Agent (Retrieval-Augmented Generation)
- **Purpose**: Vector similarity search for brand inference
- **Technologies**: Sentence Transformers, Milvus vector database
- **Capabilities**:
  - Semantic similarity matching
  - Multilingual embedding generation
  - Product catalog knowledge retrieval
  - Fuzzy matching for variations

##### LLM Agent (Large Language Model)
- **Purpose**: Advanced reasoning for brand extraction
- **Technologies**: Amazon Bedrock (Claude 3.7 Sonnet)
- **Capabilities**:
  - Contextual understanding
  - Complex reasoning patterns
  - Multilingual comprehension
  - Confidence assessment with reasoning

##### Hybrid Agent
- **Purpose**: Combine multiple approaches for optimal results
- **Technologies**: Pipeline orchestration of NER, RAG, and LLM
- **Capabilities**:
  - Weighted result combination
  - Cross-validation between methods
  - Ensemble decision making
  - Performance optimization

### Agent Coordination Patterns

#### Swarm Coordination
```python
def _coordinate_with_swarm(self, product_name: str) -> Dict[str, Any]:
    """Parallel execution of multiple agents for speed."""
    swarm_instance = Swarm(
        nodes=list(self.specialized_agents.values()),
        max_handoffs=5,
        max_iterations=10,
        execution_timeout=60.0
    )
    # Execute parallel inference across all agents
```

#### Agent Graph Coordination
```python
def _coordinate_with_agent_graph(self, product_name: str) -> Dict[str, Any]:
    """Structured workflows with dependencies."""
    graph_builder = GraphBuilder()
    # Define dependencies: hybrid depends on ner, rag, llm
    # Execute in dependency order
```

#### Workflow Coordination
```python
def _coordinate_with_workflow(self, product_name: str) -> Dict[str, Any]:
    """Multi-stage processing pipeline."""
    workflow_config = {
        "steps": [
            {"name": "parallel_extraction", "agents": ["ner", "rag", "llm"]},
            {"name": "hybrid_synthesis", "agents": ["hybrid"]},
            {"name": "final_aggregation", "agents": ["orchestrator"]}
        ]
    }
```

## Implementation Details

### Directory Structure
```
inference/
├── __init__.py
├── main.py                 # CLI entry point
├── server.py              # HTTP server
├── agents/                # Agent implementations
│   ├── orchestrator_agent.py    # Strands multi-agent orchestrator
│   ├── base_agent.py           # Abstract base classes
│   ├── ner_agent.py           # NER implementations
│   ├── rag_agent.py           # RAG implementations
│   ├── llm_agent.py           # LLM implementations
│   ├── hybrid_agent.py        # Hybrid implementations
│   ├── simple_agent.py        # Fallback agent
│   └── registry.py            # Agent registry
├── config/                # Configuration management
├── models/                # Data models and schemas
├── monitoring/            # Monitoring and logging
└── strands-multi-agents.md    # Detailed architecture docs
```

### Key Implementation Files

#### Orchestrator Agent (`orchestrator_agent.py`)
- **Class**: `StrandsMultiAgentOrchestrator(Agent)`
- **Tools**: Dynamic agent creation, coordination, result aggregation
- **Methods**: 
  - `create_ner_agent()`, `create_rag_agent()`, `create_llm_agent()`, `create_hybrid_agent()`
  - `coordinate_inference()`, `aggregate_results()`
  - `orchestrate_multiagent_inference()`

#### HTTP Server (`server.py`)
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /` - Service information
  - `POST /infer` - Product inference
- **Features**: Async processing, error handling, monitoring integration

#### Agent Registry (`registry.py`)
- **Purpose**: Centralized agent management
- **Features**: Default agent initialization, configuration management, cleanup

### Configuration Management

The system uses a hierarchical configuration system:

```python
# Agent Configurations
agent_configs = {
    "ner": {
        "model_name": "en_core_web_sm",
        "confidence_threshold": 0.5,
        "thai_text_threshold": 0.3
    },
    "rag": {
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "milvus_uri": "./milvus_rag.db",
        "similarity_threshold": 0.7
    },
    "llm": {
        "model_id": "amazon.nova-pro-v1:0",
        "aws_region": "us-east-1",
        "temperature": 0.1
    },
    "hybrid": {
        "ner_weight": 0.3,
        "rag_weight": 0.4,
        "llm_weight": 0.3
    }
}
```

## AWS Deployment Architecture

### Infrastructure Components

#### 1. Networking (VPC)
- **VPC**: Custom VPC with public and private subnets
- **Subnets**: 2 public, 2 private subnets across AZs
- **NAT Gateways**: For private subnet internet access
- **Security Groups**: Layered security for ALB, ECS, and EFS

#### 2. Compute (ECS Fargate)
- **ECS Cluster**: `multilingual-inference-cluster`
- **Services**: 
  - Orchestrator service (main API)
  - Individual agent services (NER, RAG, LLM, Hybrid)
  - Milvus vector database service
- **Platform**: ARM64 for cost optimization
- **Auto-scaling**: Based on CPU/memory utilization

#### 3. Load Balancing
- **Application Load Balancer**: Routes traffic to services
- **Target Groups**: Health check enabled
- **SSL/TLS**: Certificate management via ACM

#### 4. Storage
- **EFS**: Persistent storage for Milvus data
- **S3**: Model artifacts, embeddings, backups
- **ECR**: Container image registry

#### 5. AI/ML Services
- **Amazon Bedrock**: LLM inference (Claude 3.7 Sonnet)
- **Milvus**: Vector database for RAG operations

#### 6. Monitoring & Observability
- **CloudWatch**: Metrics, logs, alarms
- **X-Ray**: Distributed tracing
- **CloudWatch Dashboards**: System monitoring

### Security Architecture

#### IAM Roles and Policies
- **ECS Task Role**: Bedrock access, CloudWatch logging
- **ECS Execution Role**: ECR image pulling, CloudWatch logs
- **Auto-scaling Role**: ECS service scaling

#### Network Security
- **Security Groups**: 
  - ALB: Ports 80, 443 from internet
  - ECS: Ports 8080-8084 from ALB only
  - EFS: Port 2049 from ECS only
- **Private Subnets**: ECS tasks in private subnets
- **NAT Gateways**: Controlled internet access

## Deployment Prerequisites

### 1. AWS Account Setup
- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- AWS profile: `ml-sandbox` (or update scripts)
- Region: `us-east-1` (or update scripts)

### 2. Required AWS Services
- Amazon ECS
- Amazon ECR
- Amazon EFS
- Amazon S3
- Amazon Bedrock (Claude 3.7 Sonnet access)
- Application Load Balancer
- CloudWatch
- AWS X-Ray

### 3. Local Development Environment
- Docker (for building images)
- AWS CLI v2
- Python 3.13+
- Git

### 4. Bedrock Model Access
Ensure access to required Bedrock models:
```bash
aws bedrock list-foundation-models --region us-east-1 --query 'modelSummaries[?contains(modelId, `claude-3-7-sonnet`)]'
```

### 5. Resource Limits
Verify AWS service limits:
- ECS tasks per service: 10+
- ECR repositories: 10+
- EFS file systems: 5+
- ALB target groups: 10+

## Step-by-Step Deployment Guide

### Overview of Deployment Scripts

The deployment process consists of 5 sequential scripts located in `infrastructure/scripts/`:

1. **step1_deploy-cloudformation.sh** - Deploy AWS infrastructure
2. **step2_build-and-push-images.sh** - Build and push Docker images
3. **step3_deploy-ecs.sh** - Deploy ECS services
4. **step4_deploy-milvus.sh** - Deploy Milvus vector database
5. **step5_setup-milvus-storage.sh** - Setup EFS storage for Milvus

### Step 1: Deploy CloudFormation Infrastructure

**Script**: `infrastructure/scripts/step1_deploy-cloudformation.sh`

**Purpose**: Creates the complete AWS infrastructure including VPC, subnets, security groups, ALB, ECS cluster, and monitoring resources.

**Dependencies Required**:
- `infrastructure/cloudformation/main-stack.yaml`
- `infrastructure/cloudformation/ecs-stack.yaml`
- `infrastructure/cloudformation/alb-stack.yaml`
- `infrastructure/cloudformation/minimal-storage-stack.yaml`
- `infrastructure/cloudformation/monitoring-stack.yaml`

**What it does**:
1. Validates all CloudFormation templates
2. Creates S3 bucket for template storage
3. Uploads nested stack templates to S3
4. Deploys main CloudFormation stack
5. Creates VPC, subnets, security groups
6. Sets up ECS cluster and ALB
7. Configures monitoring resources

**Execution**:
```bash
# Dry run (recommended first)
./infrastructure/scripts/step1_deploy-cloudformation.sh --dry-run

# Full deployment
./infrastructure/scripts/step1_deploy-cloudformation.sh
```

**Expected Output**:
- CloudFormation stack: `multilingual-inference`
- VPC with public/private subnets
- ECS cluster: `multilingual-inference-cluster`
- Application Load Balancer
- Security groups and IAM roles

### Step 2: Build and Push Docker Images

**Script**: `infrastructure/scripts/step2_build-and-push-images.sh`

**Purpose**: Builds Docker images for all services and pushes them to Amazon ECR.

**Dependencies Required**:
- `Dockerfile` (main application dockerfile)
- `requirements.txt`
- `inference/` directory with all agent code
- Docker daemon running locally

**What it does**:
1. Creates ECR repositories for each service
2. Builds ARM64 Docker images using main Dockerfile
3. Tags images for different services (orchestrator, ner, rag, llm, hybrid)
4. Pushes all images to ECR

**Execution**:
```bash
./infrastructure/scripts/step2_build-and-push-images.sh
```

**Images Created**:
- `multilingual-inference-orchestrator:latest`
- `multilingual-inference-ner:latest`
- `multilingual-inference-rag:latest`
- `multilingual-inference-llm:latest`
- `multilingual-inference-hybrid:latest`

### Step 3: Deploy ECS Services

**Script**: `infrastructure/scripts/step3_deploy-ecs.sh`

**Purpose**: Deploys ECS Fargate services for all inference agents.

**Dependencies Required**:
- `infrastructure/ecs/task-definitions/` directory with task definitions:
  - `orchestrator-task-def.json`
  - `ner-task-def.json`
  - `rag-task-def.json`
  - `llm-task-def.json`
  - `hybrid-task-def.json`
- `infrastructure/ecs/services/` directory with service definitions
- ECR images from Step 2

**What it does**:
1. Creates CloudWatch log groups
2. Registers ECS task definitions
3. Creates ECS services with auto-scaling
4. Configures networking and security groups
5. Sets up health checks and monitoring

**Execution**:
```bash
./infrastructure/scripts/step3_deploy-ecs.sh
```

**Services Created**:
- `multilingual-inference-orchestrator` (main API service)
- `multilingual-inference-ner`
- `multilingual-inference-rag`
- `multilingual-inference-llm`
- `multilingual-inference-hybrid`

### Step 4: Deploy Milvus Vector Database

**Script**: `infrastructure/scripts/step4_deploy-milvus.sh`

**Purpose**: Deploys Milvus vector database service for RAG operations.

**Dependencies Required**:
- `infrastructure/docker/Dockerfile.milvus`
- `infrastructure/ecs/task-definitions/milvus-task-def.json`
- `infrastructure/ecs/services/milvus-service.json`
- EFS storage from Step 5 (can be run in parallel)

**What it does**:
1. Builds Milvus Docker image for ARM64
2. Pushes image to ECR
3. Registers Milvus task definition
4. Creates Milvus ECS service
5. Configures persistent storage mounting

**Execution**:
```bash
./infrastructure/scripts/step4_deploy-milvus.sh
```

**Service Created**:
- `multilingual-inference-milvus` (vector database)

### Step 5: Setup Milvus Storage

**Script**: `infrastructure/scripts/step5_setup-milvus-storage.sh`

**Purpose**: Creates EFS file system for Milvus persistent storage.

**Dependencies Required**:
- VPC and subnets from Step 1
- Security groups configured for EFS access

**What it does**:
1. Creates EFS file system with encryption
2. Creates mount targets in all subnets
3. Sets up access points for data and logs
4. Configures backup policy
5. Provides EFS ID for task definition updates

**Execution**:
```bash
./infrastructure/scripts/step5_setup-milvus-storage.sh
```

**Resources Created**:
- EFS file system for Milvus data
- Mount targets in all subnets
- Access points for organized storage
- Backup policy for data protection

### Deployment Verification

After each step, verify the deployment:

**Step 1 Verification**:
```bash
aws cloudformation describe-stacks --stack-name multilingual-inference --profile ml-sandbox
```

**Step 2 Verification**:
```bash
aws ecr describe-repositories --profile ml-sandbox --region us-east-1
```

**Step 3 Verification**:
```bash
aws ecs list-services --cluster multilingual-inference-cluster --profile ml-sandbox
```

**Step 4 & 5 Verification**:
```bash
aws ecs describe-services --cluster multilingual-inference-cluster --services multilingual-inference-milvus --profile ml-sandbox
```

## Post-Deployment Validation

### 1. Service Health Checks

Check all services are running:
```bash
aws ecs describe-services \
  --cluster multilingual-inference-cluster \
  --services multilingual-inference-orchestrator \
  --profile ml-sandbox \
  --query 'services[0].{Name:serviceName,Status:status,Running:runningCount,Desired:desiredCount}'
```

### 2. Load Balancer Health

Get ALB DNS name:
```bash
aws elbv2 describe-load-balancers \
  --profile ml-sandbox \
  --query 'LoadBalancers[?contains(LoadBalancerName, `multilingual`)].DNSName' \
  --output text
```

Test health endpoint:
```bash
curl http://<ALB-DNS-NAME>/health
```

### 3. API Functionality Test

Test inference endpoint:
```bash
curl -X POST http://<ALB-DNS-NAME>/infer \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Samsung Galaxy S24 Ultra 256GB",
    "language_hint": "en",
    "method": "orchestrator"
  }'
```

Expected response:
```json
{
  "product_name": "Samsung Galaxy S24 Ultra 256GB",
  "language": "en",
  "method": "orchestrator",
  "brand_predictions": [{
    "brand": "Samsung",
    "confidence": 0.95,
    "method": "orchestrator"
  }],
  "processing_time_ms": 1250,
  "orchestrator_agents": ["ner_agent_abc123", "rag_agent_def456", "llm_agent_ghi789"],
  "timestamp": 1703123456.789
}
```

### 4. Monitoring Validation

Check CloudWatch logs:
```bash
aws logs describe-log-groups \
  --log-group-name-prefix "/ecs/multilingual-inference" \
  --profile ml-sandbox
```

View recent logs:
```bash
aws logs tail /ecs/multilingual-inference-orchestrator \
  --since 1h \
  --profile ml-sandbox
```

## Monitoring and Maintenance

### CloudWatch Dashboards

The deployment creates comprehensive monitoring dashboards:

1. **Service Health Dashboard**
   - ECS service status
   - Task health and count
   - Load balancer metrics

2. **Performance Dashboard**
   - Response times
   - Throughput (requests/minute)
   - Error rates

3. **Resource Utilization Dashboard**
   - CPU and memory usage
   - Network I/O
   - Storage utilization

### Key Metrics to Monitor

#### Service Metrics
- **Service Health**: Running task count vs desired count
- **Response Time**: P50, P95, P99 latencies
- **Error Rate**: 4xx and 5xx error percentages
- **Throughput**: Requests per minute/hour

#### Infrastructure Metrics
- **CPU Utilization**: Target <70% average
- **Memory Utilization**: Target <80% average
- **Network I/O**: Monitor for bottlenecks
- **EFS Performance**: IOPS and throughput

#### Business Metrics
- **Inference Accuracy**: Confidence scores distribution
- **Agent Performance**: Individual agent success rates
- **Language Distribution**: Request language breakdown

### Alerting Configuration

Critical alerts configured:
- Service down (0 running tasks)
- High error rate (>5% for 5 minutes)
- High latency (P95 >5 seconds)
- Resource exhaustion (CPU >90%, Memory >95%)

### Maintenance Tasks

#### Daily
- Review CloudWatch dashboards
- Check service health status
- Monitor error logs

#### Weekly
- Review performance trends
- Check auto-scaling effectiveness
- Update security patches if needed

#### Monthly
- Review and optimize resource allocation
- Update dependencies and models
- Performance testing and optimization

### Scaling Configuration

Auto-scaling policies configured for:

**Target Tracking Scaling**:
- CPU utilization target: 70%
- Memory utilization target: 80%
- Scale out cooldown: 300 seconds
- Scale in cooldown: 300 seconds

**Step Scaling** (for rapid changes):
- Scale out: +2 tasks when CPU >85%
- Scale in: -1 task when CPU <50%

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Won't Start

**Symptoms**: ECS service shows 0 running tasks

**Diagnosis**:
```bash
aws ecs describe-services --cluster multilingual-inference-cluster --services multilingual-inference-orchestrator --profile ml-sandbox
aws ecs describe-tasks --cluster multilingual-inference-cluster --tasks <task-arn> --profile ml-sandbox
```

**Common Causes**:
- **Image pull errors**: Check ECR permissions and image existence
- **Resource constraints**: Insufficient CPU/memory allocation
- **Network issues**: Security group or subnet configuration
- **Environment variables**: Missing or incorrect configuration

**Solutions**:
- Verify ECR image exists and is accessible
- Check task definition resource allocation
- Validate security group rules
- Review CloudWatch logs for specific errors

#### 2. High Latency

**Symptoms**: API responses taking >5 seconds

**Diagnosis**:
```bash
# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=<alb-name> \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Average \
  --profile ml-sandbox
```

**Common Causes**:
- **Bedrock throttling**: Rate limits exceeded
- **Milvus performance**: Vector search bottlenecks
- **Resource constraints**: CPU/memory saturation
- **Cold starts**: New task initialization

**Solutions**:
- Implement request queuing and retry logic
- Optimize Milvus queries and indexing
- Scale up resources or task count
- Use warm pools or pre-scaling

#### 3. Bedrock Access Issues

**Symptoms**: LLM agent failures, 403 errors

**Diagnosis**:
```bash
# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1 --profile ml-sandbox
aws bedrock get-model-invocation-logging-configuration --profile ml-sandbox
```

**Common Causes**:
- **Model access**: Claude 3.7 Sonnet not enabled
- **IAM permissions**: Insufficient Bedrock permissions
- **Region mismatch**: Model not available in region
- **Rate limits**: Exceeded Bedrock quotas

**Solutions**:
- Request model access through AWS console
- Update IAM policies for Bedrock access
- Verify model availability in deployment region
- Implement exponential backoff and retry

#### 4. Milvus Connection Issues

**Symptoms**: RAG agent failures, vector search errors

**Diagnosis**:
```bash
# Check Milvus service status
aws ecs describe-services --cluster multilingual-inference-cluster --services multilingual-inference-milvus --profile ml-sandbox

# Check EFS mount status
aws efs describe-file-systems --profile ml-sandbox
```

**Common Causes**:
- **Service down**: Milvus container not running
- **Storage issues**: EFS mount failures
- **Network connectivity**: Security group restrictions
- **Data corruption**: Milvus database issues

**Solutions**:
- Restart Milvus service
- Verify EFS mount targets and access points
- Check security group rules for port 19530
- Restore from backup or reinitialize database

#### 5. Memory Issues

**Symptoms**: Tasks being killed, out of memory errors

**Diagnosis**:
```bash
# Check memory utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name MemoryUtilization \
  --dimensions Name=ServiceName,Value=multilingual-inference-orchestrator Name=ClusterName,Value=multilingual-inference-cluster \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Maximum \
  --profile ml-sandbox
```

**Solutions**:
- Increase memory allocation in task definitions
- Optimize model loading and caching
- Implement memory-efficient processing
- Use memory profiling tools

### Log Analysis

#### Key Log Locations
- **Application logs**: `/ecs/multilingual-inference-*`
- **System logs**: ECS agent logs
- **Load balancer logs**: S3 bucket (if enabled)

#### Important Log Patterns
```bash
# Error patterns to search for
aws logs filter-log-events \
  --log-group-name "/ecs/multilingual-inference-orchestrator" \
  --filter-pattern "ERROR" \
  --start-time 1703123456000 \
  --profile ml-sandbox

# Performance patterns
aws logs filter-log-events \
  --log-group-name "/ecs/multilingual-inference-orchestrator" \
  --filter-pattern "processing_time" \
  --profile ml-sandbox
```

### Performance Optimization

#### 1. Model Optimization
- Use model quantization for faster inference
- Implement model caching strategies
- Optimize embedding dimensions

#### 2. Infrastructure Optimization
- Use ARM64 instances for cost efficiency
- Implement connection pooling
- Optimize container resource allocation

#### 3. Application Optimization
- Implement async processing where possible
- Use batch processing for multiple requests
- Optimize agent coordination strategies

---

## Summary

This comprehensive deployment guide provides everything needed to successfully deploy and maintain the Multilingual Product Inference System on AWS. The system leverages cutting-edge AI technologies with a robust, scalable architecture designed for production workloads.

**Key Benefits**:
- **Advanced AI**: Strands Agents SDK with multi-agent coordination
- **Scalable**: Auto-scaling ECS Fargate deployment
- **Reliable**: Comprehensive monitoring and error handling
- **Secure**: AWS best practices for security and compliance
- **Cost-Effective**: ARM64 platform optimization

For additional support or customization requirements, refer to the detailed implementation documentation in `inference/strands-multi-agents.md` or contact the development team.