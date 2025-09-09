#!/bin/bash

# Simple deployment script for orchestrator with agents
set -e

STACK_NAME="multilingual-inference"
AWS_REGION="us-east-1"
AWS_PROFILE="ml-sandbox"

echo "🚀 Deploying orchestrator with agents..."

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
echo "AWS Account: $AWS_ACCOUNT_ID"

# Get stack outputs
ECS_CLUSTER_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
    --output text)

LOAD_BALANCER_DNS=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
    --output text)

echo "ECS Cluster: $ECS_CLUSTER_NAME"
echo "Load Balancer: $LOAD_BALANCER_DNS"

# Create ECR repository if it doesn't exist
ECR_REPOSITORY_NAME="multilingual-inference-orchestrator"
ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"

aws ecr describe-repositories \
    --repository-names "$ECR_REPOSITORY_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" &> /dev/null || \
aws ecr create-repository \
    --repository-name "$ECR_REPOSITORY_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION"

echo "ECR Repository: $ECR_REPOSITORY_URI"

# Login to ECR
aws ecr get-login-password \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" | \
docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"

# Build and push image
echo "🔨 Building Docker image..."
docker build --platform linux/arm64 -t "$ECR_REPOSITORY_URI:latest" .

echo "📤 Pushing Docker image..."
docker push "$ECR_REPOSITORY_URI:latest"

echo "✅ Deployment completed!"
echo "Load Balancer DNS: $LOAD_BALANCER_DNS"
echo "Health Check: http://$LOAD_BALANCER_DNS/health"
echo "Inference Endpoint: http://$LOAD_BALANCER_DNS/infer"