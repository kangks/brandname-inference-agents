# AWS Deployment Guide

This guide provides comprehensive instructions for deploying the Multilingual Product Inference System on AWS infrastructure in the us-east-1 region using the ml-sandbox profile.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Overview](#infrastructure-overview)
3. [Pre-deployment Setup](#pre-deployment-setup)
4. [Core Infrastructure Deployment](#core-infrastructure-deployment)
5. [Service Deployment](#service-deployment)
6. [Milvus Vector Database Deployment](#milvus-vector-database-deployment)
7. [Monitoring and Logging Setup](#monitoring-and-logging-setup)
8. [Post-deployment Validation](#post-deployment-validation)
9. [Scaling and Optimization](#scaling-and-optimization)
10. [Maintenance and Updates](#maintenance-and-updates)
11. [Troubleshooting Deployment Issues](#troubleshooting-deployment-issues)

## Prerequisites

### AWS Account Setup

- AWS account with appropriate permissions
- AWS CLI v2 installed and configured
- `ml-sandbox` profile configured with required permissions
- Access to `us-east-1` region

### Required AWS Permissions

The `ml-sandbox` profile must have the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:*",
        "ecr:*",
        "ec2:*",
        "iam:*",
        "logs:*",
        "cloudformation:*",
        "s3:*",
        "bedrock:*",
        "elasticloadbalancing:*",
        "application-autoscaling:*",
        "cloudwatch:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Local Environment

- Python 3.13 in `.venv` virtual environment
- Docker installed and running
- Git repository cloned locally

### Verification Commands

```bash
# Verify AWS CLI configuration
aws sts get-caller-identity --profile ml-sandbox --region us-east-1

# Verify Docker
docker --version
docker info

# Verify Python environment
python --version  # Should show Python 3.13.x
which python      # Should point to .venv/bin/python
```

## Infrastructure Overview

The deployment consists of the following AWS components:

### Core Infrastructure
- **VPC**: Custom VPC with public/private subnets
- **ECS Cluster**: Fargate cluster for containerized services
- **Application Load Balancer**: Traffic distribution and health checks
- **ECR Repositories**: Container image storage
- **CloudWatch**: Logging and monitoring

### Services
- **Orchestrator Agent**: Main coordination service
- **NER Agent**: Named Entity Recognition service
- **RAG Agent**: Retrieval-Augmented Generation service
- **LLM Agent**: Language Model inference service
- **Hybrid Agent**: Sequential processing service
- **Milvus**: Vector database service

### Storage and Data
- **S3 Buckets**: Model artifacts and training data
- **EFS**: Persistent storage for Milvus
- **Parameter Store**: Configuration management

## Pre-deployment Setup

### 1. Environment Configuration

```bash
# Set up environment variables
export AWS_PROFILE=ml-sandbox
export AWS_REGION=us-east-1
export DEPLOYMENT_ENV=production
export STACK_NAME=multilingual-inference

# Create deployment configuration
cp .env.example .env.production
```

Edit `.env.production`:
```bash
# AWS Configuration
AWS_PROFILE=ml-sandbox
AWS_REGION=us-east-1

# Deployment Configuration
DEPLOYMENT_ENV=production
STACK_NAME=multilingual-inference
CLUSTER_NAME=multilingual-inference-cluster

# Service Configuration
ORCHESTRATOR_CPU=1024
ORCHESTRATOR_MEMORY=2048
AGENT_CPU=512
AGENT_MEMORY=1024

# Milvus Configuration
MILVUS_CPU=1024
MILVUS_MEMORY=2048
MILVUS_STORAGE_SIZE=100

# Monitoring Configuration
LOG_RETENTION_DAYS=30
ENABLE_DETAILED_MONITORING=true
```

### 2. Build and Push Container Images

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 --profile ml-sandbox | \
  docker login --username AWS --password-stdin $(aws sts get-caller-identity --profile ml-sandbox --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Build and push all images
./infrastructure/scripts/build-and-push-images.sh
```

The script will build and push:
- Orchestrator agent image
- NER agent image
- RAG agent image
- LLM agent image
- Hybrid agent image
- Milvus image (ARM64 compatible)

### 3. Create S3 Buckets

```bash
# Create S3 buckets for model storage
aws s3 mb s3://ml-sandbox-multilingual-models --profile ml-sandbox --region us-east-1
aws s3 mb s3://ml-sandbox-multilingual-training --profile ml-sandbox --region us-east-1
aws s3 mb s3://ml-sandbox-multilingual-logs --profile ml-sandbox --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning --bucket ml-sandbox-multilingual-models --versioning-configuration Status=Enabled --profile ml-sandbox --region us-east-1
```

## Core Infrastructure Deployment

### 1. Deploy Main Infrastructure Stack

```bash
# Deploy the main CloudFormation stack
./infrastructure/scripts/deploy-cloudformation.sh

# Or deploy manually
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/main-stack.yaml \
  --stack-name multilingual-inference-main \
  --parameter-overrides \
    Environment=production \
    ClusterName=multilingual-inference-cluster \
  --capabilities CAPABILITY_IAM \
  --profile ml-sandbox \
  --region us-east-1
```

### 2. Deploy Storage Stack

```bash
# Deploy EFS and storage resources
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/storage-stack.yaml \
  --stack-name multilingual-inference-storage \
  --parameter-overrides \
    Environment=production \
    MilvusStorageSize=100 \
  --profile ml-sandbox \
  --region us-east-1
```

### 3. Deploy ALB Stack

```bash
# Deploy Application Load Balancer
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/alb-stack.yaml \
  --stack-name multilingual-inference-alb \
  --parameter-overrides \
    Environment=production \
    VpcId=$(aws cloudformation describe-stacks --stack-name multilingual-inference-main --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' --output text --profile ml-sandbox --region us-east-1) \
  --profile ml-sandbox \
  --region us-east-1
```

### 4. Deploy ECS Stack

```bash
# Deploy ECS cluster and base configuration
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/ecs-stack.yaml \
  --stack-name multilingual-inference-ecs \
  --parameter-overrides \
    Environment=production \
    ClusterName=multilingual-inference-cluster \
  --capabilities CAPABILITY_IAM \
  --profile ml-sandbox \
  --region us-east-1
```

### 5. Verify Infrastructure Deployment

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name multilingual-inference-main --profile ml-sandbox --region us-east-1
aws cloudformation describe-stacks --stack-name multilingual-inference-storage --profile ml-sandbox --region us-east-1
aws cloudformation describe-stacks --stack-name multilingual-inference-alb --profile ml-sandbox --region us-east-1
aws cloudformation describe-stacks --stack-name multilingual-inference-ecs --profile ml-sandbox --region us-east-1

# Get important outputs
aws cloudformation describe-stacks --stack-name multilingual-inference-main --query 'Stacks[0].Outputs' --profile ml-sandbox --region us-east-1
```

## Service Deployment

### 1. Deploy Orchestrator Service

```bash
# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/orchestrator-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

# Create service
aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/orchestrator-service.json \
  --profile ml-sandbox \
  --region us-east-1
```

### 2. Deploy Individual Agent Services

```bash
# Deploy NER Agent
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/ner-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/ner-service.json \
  --profile ml-sandbox \
  --region us-east-1

# Deploy RAG Agent
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/rag-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/rag-service.json \
  --profile ml-sandbox \
  --region us-east-1

# Deploy LLM Agent
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/llm-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/llm-service.json \
  --profile ml-sandbox \
  --region us-east-1

# Deploy Hybrid Agent
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/hybrid-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/hybrid-service.json \
  --profile ml-sandbox \
  --region us-east-1
```

### 3. Automated Service Deployment

```bash
# Use the automated deployment script
./infrastructure/scripts/deploy-ecs.sh

# This script will:
# 1. Register all task definitions
# 2. Create all services
# 3. Configure auto-scaling
# 4. Set up health checks
```

### 4. Configure Auto-scaling

```bash
# Apply auto-scaling policies
aws application-autoscaling put-scaling-policy \
  --cli-input-json file://infrastructure/ecs/autoscaling/orchestrator-autoscaling.json \
  --profile ml-sandbox \
  --region us-east-1

aws application-autoscaling put-scaling-policy \
  --cli-input-json file://infrastructure/ecs/autoscaling/agent-autoscaling.json \
  --profile ml-sandbox \
  --region us-east-1
```

## Milvus Vector Database Deployment

### 1. Deploy Milvus Infrastructure

```bash
# Deploy Milvus-specific storage
./infrastructure/scripts/setup-milvus-storage.sh

# Deploy Milvus service
./infrastructure/scripts/deploy-milvus.sh
```

### 2. Manual Milvus Deployment

```bash
# Register Milvus task definition
aws ecs register-task-definition \
  --cli-input-json file://infrastructure/ecs/task-definitions/milvus-task-def.json \
  --profile ml-sandbox \
  --region us-east-1

# Create Milvus service
aws ecs create-service \
  --cli-input-json file://infrastructure/ecs/services/milvus-service.json \
  --profile ml-sandbox \
  --region us-east-1
```

### 3. Initialize Milvus Collections

```bash
# Wait for Milvus to be ready
sleep 60

# Initialize collections
python training/preprocessing/milvus_collection_manager.py \
  --create-collections \
  --milvus-host $(aws elbv2 describe-load-balancers --names multilingual-inference-alb --query 'LoadBalancers[0].DNSName' --output text --profile ml-sandbox --region us-east-1) \
  --aws-profile ml-sandbox
```

### 4. Populate Vector Database

```bash
# Upload initial embeddings
python training/pipelines/train_rag_pipeline.py \
  --populate-milvus \
  --embeddings-file models/initial_embeddings.json \
  --milvus-host $(aws elbv2 describe-load-balancers --names multilingual-inference-alb --query 'LoadBalancers[0].DNSName' --output text --profile ml-sandbox --region us-east-1) \
  --aws-profile ml-sandbox
```

## Monitoring and Logging Setup

### 1. Deploy Monitoring Stack

```bash
# Deploy CloudWatch dashboards and alarms
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/monitoring-stack.yaml \
  --stack-name multilingual-inference-monitoring \
  --parameter-overrides \
    Environment=production \
    ClusterName=multilingual-inference-cluster \
  --profile ml-sandbox \
  --region us-east-1
```

### 2. Configure Log Groups

```bash
# Create log groups for each service
aws logs create-log-group --log-group-name /ecs/orchestrator-agent --profile ml-sandbox --region us-east-1
aws logs create-log-group --log-group-name /ecs/ner-agent --profile ml-sandbox --region us-east-1
aws logs create-log-group --log-group-name /ecs/rag-agent --profile ml-sandbox --region us-east-1
aws logs create-log-group --log-group-name /ecs/llm-agent --profile ml-sandbox --region us-east-1
aws logs create-log-group --log-group-name /ecs/hybrid-agent --profile ml-sandbox --region us-east-1
aws logs create-log-group --log-group-name /ecs/milvus --profile ml-sandbox --region us-east-1

# Set retention policy
aws logs put-retention-policy --log-group-name /ecs/orchestrator-agent --retention-in-days 30 --profile ml-sandbox --region us-east-1
```

### 3. Set up Health Monitoring

```bash
# Deploy health check monitoring
python infrastructure/monitoring/milvus-health-check.py --deploy --aws-profile ml-sandbox --region us-east-1
```

## Post-deployment Validation

### 1. Service Health Checks

```bash
# Get load balancer DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers --names multilingual-inference-alb --query 'LoadBalancers[0].DNSName' --output text --profile ml-sandbox --region us-east-1)

# Check overall system health
curl http://$ALB_DNS/health

# Check individual service health
curl http://$ALB_DNS/agents/orchestrator/health
curl http://$ALB_DNS/agents/ner/health
curl http://$ALB_DNS/agents/rag/health
curl http://$ALB_DNS/agents/llm/health
curl http://$ALB_DNS/agents/hybrid/health
```

### 2. Functional Testing

```bash
# Test inference endpoint
curl -X POST http://$ALB_DNS/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "iPhone 15 Pro Max", "language": "en"}'

# Test multilingual inference
curl -X POST http://$ALB_DNS/infer \
  -H "Content-Type: application/json" \
  -d '{"product_name": "ยาสีฟัน Wonder smile toothpaste", "language": "mixed"}'
```

### 3. Performance Testing

```bash
# Run load testing
python tests/performance/benchmark_runner.py \
  --endpoint http://$ALB_DNS \
  --concurrent-users 10 \
  --duration 300 \
  --test-data tests/data/load_test_queries.json
```

### 4. Monitoring Validation

```bash
# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=orchestrator-service Name=ClusterName,Value=multilingual-inference-cluster \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average \
  --profile ml-sandbox \
  --region us-east-1
```

## Scaling and Optimization

### 1. Auto-scaling Configuration

```bash
# Update auto-scaling targets
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/multilingual-inference-cluster/orchestrator-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10 \
  --profile ml-sandbox \
  --region us-east-1

# Configure scaling policies
aws application-autoscaling put-scaling-policy \
  --policy-name orchestrator-scale-up \
  --service-namespace ecs \
  --resource-id service/multilingual-inference-cluster/orchestrator-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://infrastructure/ecs/autoscaling/target-tracking-policy.json \
  --profile ml-sandbox \
  --region us-east-1
```

### 2. Performance Optimization

```bash
# Optimize task definitions for better performance
# Update CPU and memory allocations based on monitoring data

# Update orchestrator service
aws ecs update-service \
  --cluster multilingual-inference-cluster \
  --service orchestrator-service \
  --task-definition orchestrator-agent:2 \
  --profile ml-sandbox \
  --region us-east-1
```

### 3. Cost Optimization

```bash
# Use Spot instances for non-critical workloads
# Configure capacity providers for cost optimization

aws ecs create-capacity-provider \
  --name multilingual-inference-spot \
  --auto-scaling-group-provider file://infrastructure/ecs/capacity-providers/spot-provider.json \
  --profile ml-sandbox \
  --region us-east-1
```

## Maintenance and Updates

### 1. Rolling Updates

```bash
# Update service with new task definition
aws ecs update-service \
  --cluster multilingual-inference-cluster \
  --service orchestrator-service \
  --task-definition orchestrator-agent:3 \
  --deployment-configuration maximumPercent=200,minimumHealthyPercent=50 \
  --profile ml-sandbox \
  --region us-east-1
```

### 2. Blue-Green Deployment

```bash
# Create new service version
aws ecs create-service \
  --cluster multilingual-inference-cluster \
  --service-name orchestrator-service-v2 \
  --task-definition orchestrator-agent:3 \
  --desired-count 2 \
  --load-balancers file://infrastructure/ecs/services/orchestrator-service-v2.json \
  --profile ml-sandbox \
  --region us-east-1

# Switch traffic gradually
# Update ALB target group weights
```

### 3. Backup and Recovery

```bash
# Backup Milvus data
aws efs create-backup \
  --file-system-id $(aws efs describe-file-systems --query 'FileSystems[?Name==`multilingual-milvus-efs`].FileSystemId' --output text --profile ml-sandbox --region us-east-1) \
  --profile ml-sandbox \
  --region us-east-1

# Backup model artifacts
aws s3 sync s3://ml-sandbox-multilingual-models s3://ml-sandbox-multilingual-models-backup --profile ml-sandbox --region us-east-1
```

## Troubleshooting Deployment Issues

### Common Issues and Solutions

#### 1. Service Won't Start

```bash
# Check service events
aws ecs describe-services \
  --cluster multilingual-inference-cluster \
  --services orchestrator-service \
  --profile ml-sandbox \
  --region us-east-1

# Check task logs
aws logs get-log-events \
  --log-group-name /ecs/orchestrator-agent \
  --log-stream-name [stream-name] \
  --profile ml-sandbox \
  --region us-east-1
```

#### 2. Health Check Failures

```bash
# Check target group health
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups --names orchestrator-tg --query 'TargetGroups[0].TargetGroupArn' --output text --profile ml-sandbox --region us-east-1) \
  --profile ml-sandbox \
  --region us-east-1

# Update health check configuration
aws elbv2 modify-target-group \
  --target-group-arn [target-group-arn] \
  --health-check-interval-seconds 60 \
  --health-check-timeout-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 5 \
  --profile ml-sandbox \
  --region us-east-1
```

#### 3. Resource Constraints

```bash
# Check cluster capacity
aws ecs describe-clusters \
  --clusters multilingual-inference-cluster \
  --include STATISTICS \
  --profile ml-sandbox \
  --region us-east-1

# Scale cluster if needed
aws ecs update-cluster-settings \
  --cluster multilingual-inference-cluster \
  --settings name=containerInsights,value=enabled \
  --profile ml-sandbox \
  --region us-east-1
```

### Emergency Procedures

#### 1. Service Rollback

```bash
# Rollback to previous task definition
aws ecs update-service \
  --cluster multilingual-inference-cluster \
  --service orchestrator-service \
  --task-definition orchestrator-agent:1 \
  --force-new-deployment \
  --profile ml-sandbox \
  --region us-east-1
```

#### 2. Scale Down for Issues

```bash
# Temporarily scale down problematic service
aws ecs update-service \
  --cluster multilingual-inference-cluster \
  --service orchestrator-service \
  --desired-count 0 \
  --profile ml-sandbox \
  --region us-east-1
```

#### 3. Emergency Maintenance Mode

```bash
# Put system in maintenance mode
# Update ALB to return maintenance page
aws elbv2 modify-rule \
  --rule-arn [maintenance-rule-arn] \
  --actions Type=fixed-response,FixedResponseConfig='{StatusCode=503,ContentType=text/plain,MessageBody=System under maintenance}' \
  --profile ml-sandbox \
  --region us-east-1
```

## Deployment Checklist

### Pre-deployment
- [ ] AWS credentials configured for ml-sandbox profile
- [ ] us-east-1 region access verified
- [ ] Python 3.13 environment set up
- [ ] Docker images built and pushed to ECR
- [ ] S3 buckets created
- [ ] Configuration files updated

### Infrastructure Deployment
- [ ] Main CloudFormation stack deployed
- [ ] Storage stack deployed
- [ ] ALB stack deployed
- [ ] ECS stack deployed
- [ ] All stacks in CREATE_COMPLETE status

### Service Deployment
- [ ] All task definitions registered
- [ ] All services created and running
- [ ] Auto-scaling policies configured
- [ ] Health checks passing

### Milvus Deployment
- [ ] Milvus service deployed
- [ ] Collections created
- [ ] Initial data populated
- [ ] Health checks passing

### Monitoring Setup
- [ ] CloudWatch dashboards created
- [ ] Log groups configured
- [ ] Alarms set up
- [ ] Health monitoring active

### Validation
- [ ] All health endpoints responding
- [ ] Functional tests passing
- [ ] Performance tests completed
- [ ] Monitoring data flowing

This comprehensive deployment guide ensures a successful and maintainable deployment of the Multilingual Product Inference System on AWS using the specified ml-sandbox profile and us-east-1 region constraints.