# Multilingual Product Inference Infrastructure

This directory contains the complete AWS infrastructure setup for the multilingual product inference system using ARM64 ECS Fargate deployment.

## Architecture Overview

The infrastructure is designed for ARM64 platform deployment on AWS using:
- **ECS Fargate** for containerized agent deployment
- **Application Load Balancer** for request routing and health checks
- **Milvus on ECS** for vector database storage
- **CloudWatch** for logging and monitoring
- **EFS** for persistent storage
- **S3** for model artifacts and training data
- **API Gateway** for external API access

## Prerequisites

- AWS CLI configured with `ml-sandbox` profile
- Docker with ARM64 support
- Python 3.13 in `.venv` virtual environment
- Access to `us-east-1` region

## Directory Structure

```
infrastructure/
├── cloudformation/          # CloudFormation templates
│   ├── main-stack.yaml     # Main infrastructure stack
│   ├── ecs-stack.yaml      # ECS cluster and services
│   ├── alb-stack.yaml      # Load balancer and API Gateway
│   ├── storage-stack.yaml  # EFS and S3 storage
│   └── monitoring-stack.yaml # CloudWatch monitoring
├── docker/                 # Dockerfiles for each agent
│   ├── Dockerfile.orchestrator
│   ├── Dockerfile.ner
│   ├── Dockerfile.rag
│   ├── Dockerfile.llm
│   ├── Dockerfile.hybrid
│   └── Dockerfile.milvus
├── ecs/                    # ECS configurations
│   ├── task-definitions/   # ECS task definitions
│   ├── services/          # ECS service configurations
│   └── autoscaling/       # Auto-scaling policies
├── milvus/                # Milvus configuration
│   └── milvus.yaml        # Milvus server configuration
├── monitoring/            # Monitoring scripts
│   └── milvus-health-check.py
├── scripts/               # Deployment scripts
│   ├── deploy-cloudformation.sh
│   ├── build-and-push-images.sh
│   ├── deploy-ecs.sh
│   ├── setup-milvus-storage.sh
│   └── deploy-milvus.sh
├── storage/               # Storage configurations
│   └── efs-milvus.json
└── README.md
```

## Deployment Guide

### Step 1: Deploy CloudFormation Infrastructure

Deploy the complete AWS infrastructure stack:

```bash
# Validate and deploy infrastructure (dry run first)
./infrastructure/scripts/deploy-cloudformation.sh --dry-run

# Deploy infrastructure
./infrastructure/scripts/deploy-cloudformation.sh
```

This creates:
- VPC with public/private subnets
- ECS cluster with ARM64 support
- Application Load Balancer
- Security groups and IAM roles
- CloudWatch logging and monitoring
- S3 buckets for storage

### Step 2: Build and Push Docker Images

Build ARM64 Docker images and push to ECR:

```bash
./infrastructure/scripts/build-and-push-images.sh
```

This builds and pushes:
- Orchestrator agent image
- NER agent image
- RAG agent image
- LLM agent image
- Hybrid agent image

### Step 3: Setup Milvus Storage

Create EFS storage for Milvus persistence:

```bash
./infrastructure/scripts/setup-milvus-storage.sh
```

This creates:
- EFS file system with encryption
- Mount targets in private subnets
- Access points for data and logs
- Backup policies

### Step 4: Deploy Milvus Vector Database

Deploy Milvus on ECS with ARM64 platform:

```bash
./infrastructure/scripts/deploy-milvus.sh
```

This deploys:
- Milvus standalone container
- etcd for metadata storage
- MinIO for object storage
- Health monitoring

### Step 5: Deploy Agent Services

Deploy all inference agent services:

```bash
./infrastructure/scripts/deploy-ecs.sh
```

This creates:
- ECS services for all agents
- Auto-scaling policies
- Service discovery
- Load balancer target groups

## Configuration

### Environment Variables

The system uses the following environment variables:

```bash
AWS_DEFAULT_REGION=us-east-1
AWS_PROFILE=ml-sandbox
MILVUS_HOST=milvus.multilingual-inference.local
MILVUS_PORT=19530
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
LOG_LEVEL=INFO
```

### Auto-Scaling Configuration

Auto-scaling is configured for all services:

- **Orchestrator**: 2-10 instances, scales on CPU/Memory
- **NER Agent**: 1-5 instances, scales on CPU
- **RAG Agent**: 1-5 instances, scales on CPU
- **LLM Agent**: 1-3 instances, scales on CPU (conservative)
- **Hybrid Agent**: 1-4 instances, scales on CPU
- **Milvus**: 1 instance (stateful service)

### Resource Allocation

ARM64 ECS task resource allocation:

| Service | CPU | Memory | Platform |
|---------|-----|--------|----------|
| Orchestrator | 1024 | 2048 MB | ARM64 |
| NER Agent | 512 | 1024 MB | ARM64 |
| RAG Agent | 512 | 1024 MB | ARM64 |
| LLM Agent | 512 | 1024 MB | ARM64 |
| Hybrid Agent | 1024 | 2048 MB | ARM64 |
| Milvus | 2048 | 4096 MB | ARM64 |

## Monitoring and Logging

### CloudWatch Dashboard

Access the monitoring dashboard:
```
https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=multilingual-inference-multilingual-inference
```

### Log Groups

All services log to CloudWatch:
- `/ecs/multilingual-inference-orchestrator`
- `/ecs/multilingual-inference-ner`
- `/ecs/multilingual-inference-rag`
- `/ecs/multilingual-inference-llm`
- `/ecs/multilingual-inference-hybrid`
- `/ecs/multilingual-inference-milvus`

### Health Monitoring

Milvus health monitoring script:
```bash
python infrastructure/monitoring/milvus-health-check.py
```

### Alerts

CloudWatch alarms monitor:
- High CPU/Memory utilization
- Milvus connection health
- Load balancer response times
- Error rates

## API Endpoints

### Application Load Balancer

Direct access via ALB:
```
http://<alb-dns-name>/          # Orchestrator (default)
http://<alb-dns-name>/ner/      # NER Agent
http://<alb-dns-name>/rag/      # RAG Agent
http://<alb-dns-name>/llm/      # LLM Agent
http://<alb-dns-name>/hybrid/   # Hybrid Agent
```

### API Gateway

External API access:
```
https://<api-id>.execute-api.us-east-1.amazonaws.com/production/inference
```

### Service Discovery

Internal service communication:
```
orchestrator.multilingual-inference.local:8080
ner.multilingual-inference.local:8081
rag.multilingual-inference.local:8082
llm.multilingual-inference.local:8083
hybrid.multilingual-inference.local:8084
milvus.multilingual-inference.local:19530
```

## Troubleshooting

### Common Issues

1. **ECS Task Startup Failures**
   ```bash
   # Check task logs
   aws ecs describe-tasks --cluster multilingual-inference-cluster --tasks <task-arn>
   
   # Check CloudWatch logs
   aws logs get-log-events --log-group-name /ecs/multilingual-inference-<service>
   ```

2. **Milvus Connection Issues**
   ```bash
   # Check Milvus health
   curl http://milvus.multilingual-inference.local:9091/healthz
   
   # Run health monitoring
   python infrastructure/monitoring/milvus-health-check.py
   ```

3. **Load Balancer Health Check Failures**
   ```bash
   # Check target group health
   aws elbv2 describe-target-health --target-group-arn <target-group-arn>
   ```

### Debugging Commands

```bash
# List ECS services
aws ecs list-services --cluster multilingual-inference-cluster

# Check service status
aws ecs describe-services --cluster multilingual-inference-cluster --services <service-name>

# View CloudWatch metrics
aws cloudwatch get-metric-statistics --namespace AWS/ECS --metric-name CPUUtilization

# Check EFS mount targets
aws efs describe-mount-targets --file-system-id <efs-id>
```

## Security

### IAM Roles

- **ECS Task Execution Role**: ECR, CloudWatch Logs access
- **ECS Task Role**: Bedrock, S3, EFS access
- **Auto Scaling Role**: ECS service scaling permissions

### Security Groups

- **ALB Security Group**: HTTP/HTTPS from internet
- **ECS Security Group**: Agent ports from ALB only
- **EFS Security Group**: NFS from ECS tasks only

### Encryption

- **EFS**: Encrypted at rest and in transit
- **S3**: Server-side encryption (AES-256)
- **CloudWatch Logs**: Encrypted with AWS managed keys

## Cost Optimization

### ARM64 Benefits

- **30-40% cost savings** compared to x86_64
- **Better price-performance ratio** for ML workloads
- **Native ARM64 support** for Python and ML libraries

### Auto-Scaling

- **Fargate Spot**: 70% cost savings for non-critical workloads
- **Scheduled Scaling**: Scale down during off-hours
- **Target Tracking**: Efficient resource utilization

### Storage Optimization

- **S3 Lifecycle Policies**: Automatic transition to cheaper storage classes
- **EFS Provisioned Throughput**: Pay only for required performance
- **Log Retention**: 30-day retention to control costs

## Maintenance

### Regular Tasks

1. **Update Docker Images**: Rebuild and deploy updated agent images
2. **Monitor Costs**: Review AWS Cost Explorer monthly
3. **Security Updates**: Update base images and dependencies
4. **Performance Tuning**: Adjust auto-scaling policies based on usage

### Backup and Recovery

- **EFS Automatic Backups**: Enabled with 35-day retention
- **S3 Versioning**: Enabled for model artifacts
- **CloudFormation**: Infrastructure as code for disaster recovery

## Support

For issues and questions:
1. Check CloudWatch logs and metrics
2. Review ECS service events
3. Run health monitoring scripts
4. Consult AWS documentation for service-specific issues