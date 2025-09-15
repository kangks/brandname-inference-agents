# Infrastructure Deployment Guide

AWS infrastructure for the multilingual product inference system using ARM64 ECS Fargate.

## Architecture

**Monolithic Deployment**: Single ECS service containing all agents
- **ECS Fargate**: ARM64 container with orchestrator + all agents
- **Application Load Balancer**: Request routing and health checks  
- **Milvus** (optional): Vector database for RAG agent
- **CloudWatch**: Monitoring and logging

## Prerequisites

- AWS CLI configured with `ml-sandbox` profile (or update scripts)
- Docker with ARM64 support
- Access to `us-east-1` region (or update scripts)

## Directory Structure

```
infrastructure/
├── scripts/               # Deployment scripts
│   ├── deploy-monolithic.sh             # 🚀 One-command deployment
│   ├── step1_deploy-cloudformation.sh   # Deploy AWS infrastructure
│   ├── step2_build-and-push-images.sh   # Build orchestrator image
│   ├── step3_deploy-ecs.sh              # Deploy ECS service
│   ├── step4_deploy-milvus.sh           # Deploy vector database (optional)
│   └── step5_setup-milvus-storage.sh    # Setup EFS storage (optional)
├── ecs/                   # ECS configurations
│   ├── task-definitions/  # Task definitions
│   └── services/         # Service configurations
├── cloudformation/        # Infrastructure templates
├── docker/               # Docker configurations
└── README.md
```

## Quick Deployment

### Option 1: One-Command Deployment (Recommended)
```bash
./scripts/deploy-monolithic.sh
```
This script automatically:
- Deploys infrastructure
- Builds and pushes Docker image
- Deploys ECS service
- **Validates deployment with API tests**
- Provides ready-to-use API endpoints

### Option 2: Step-by-Step Deployment
```bash
./scripts/step1_deploy-cloudformation.sh  # Deploy AWS infrastructure
./scripts/step2_build-and-push-images.sh  # Build and push orchestrator image
./scripts/step3_deploy-ecs.sh            # Deploy ECS service

# Optional: Deploy Milvus for RAG agent
./scripts/step5_setup-milvus-storage.sh   # Setup EFS storage
./scripts/step4_deploy-milvus.sh          # Deploy Milvus service
```

### Option 3: Test Existing Deployment
```bash
./scripts/test-deployment.sh              # Test all inference methods
```

## What Gets Deployed

### Core Infrastructure
- **VPC**: Public/private subnets, security groups
- **ECS Cluster**: ARM64 Fargate cluster
- **Application Load Balancer**: Request routing
- **CloudWatch**: Logging and monitoring

### Main Service
- **Orchestrator Service**: Single container with all agents
  - NER, RAG, LLM, Hybrid, Simple agents
  - Auto-scaling (2-10 instances)
  - Health checks enabled

### Optional Components
- **Milvus Service**: Vector database for RAG agent
- **EFS Storage**: Persistent storage for Milvus

## Configuration

### Key Settings
- **AWS Profile**: `ml-sandbox` (update in scripts if different)
- **AWS Region**: `us-east-1` (update in scripts if different)
- **Platform**: ARM64 for cost optimization
- **Auto-scaling**: 2-10 instances based on CPU/Memory

### Resource Allocation
| Service | CPU | Memory | Contains |
|---------|-----|--------|----------|
| Orchestrator | 1024 | 2048 MB | All agents (NER, RAG, LLM, Hybrid, Simple) |
| Milvus (optional) | 2048 | 4096 MB | Vector database for RAG agent |

## Monitoring

### CloudWatch Logs
- `/ecs/multilingual-inference-orchestrator` - All agent logs
- `/ecs/multilingual-inference-milvus` - Vector database logs (if deployed)

### Health Checks
```bash
# Get ALB DNS name
aws elbv2 describe-load-balancers --query 'LoadBalancers[?contains(LoadBalancerName, `multilingual`)].DNSName' --output text

# Test health endpoint
curl http://<ALB-DNS>/health
```

## API Usage

After deployment, use the ALB DNS name for API calls:

```bash
# Get ALB DNS
ALB_DNS=$(aws elbv2 describe-load-balancers --query 'LoadBalancers[?contains(LoadBalancerName, `multilingual`)].DNSName' --output text)

# Test API
curl -X POST "http://$ALB_DNS/infer" \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'
```

Available methods: `orchestrator`, `ner`, `rag`, `llm`, `hybrid`, `simple`

## Cleanup Old Services

If you previously deployed individual agent services, clean them up:

```bash
./scripts/cleanup-old-services.sh
```

This removes old NER, RAG, LLM, and Hybrid services, leaving only the orchestrator service.

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check service status
aws ecs describe-services --cluster multilingual-inference-cluster --services multilingual-inference-orchestrator

# Check logs
aws logs tail /ecs/multilingual-inference-orchestrator --since 1h
```

**API not responding:**
```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>

# Test health endpoint
curl http://<ALB-DNS>/health
```

**Multiple services running:**
```bash
# List all services
aws ecs list-services --cluster multilingual-inference-cluster

# Clean up old individual agent services
./scripts/cleanup-old-services.sh
```

## Cost Optimization

- **ARM64 Platform**: 30-40% cost savings vs x86_64
- **Auto-scaling**: Scales 2-10 instances based on demand
- **Fargate**: Pay only for resources used

## Support

For detailed information:
- **[Main README](../README.md)** - Getting started guide
- **[API Usage Guide](../docs/API_USAGE_GUIDE.md)** - Complete API reference
- **[Architecture FAQ](../../INFERENCE_ARCHITECTURE_FAQ.md)** - Common questions