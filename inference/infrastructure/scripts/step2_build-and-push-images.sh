#!/bin/bash

# Build and push Docker images for multilingual inference agents
# Uses ml-sandbox AWS profile and us-east-1 region with ARM64 platform

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "Building and pushing Docker images for multilingual inference system..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"
echo "ECR Registry: $ECR_REGISTRY"

# Function to create ECR repository if it doesn't exist
create_ecr_repo() {
    local repo_name=$1
    
    echo "Creating ECR repository: $repo_name"
    aws ecr create-repository \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --repository-name $repo_name \
        --image-scanning-configuration scanOnPush=true \
        --tags Key=Environment,Value=production Key=Platform,Value=ARM64 \
        2>/dev/null || echo "Repository $repo_name already exists"
}

# Function to build and push Docker image
build_and_push() {
    local agent_name=$1
    local dockerfile_path=$2
    local repo_name="multilingual-inference-$agent_name"
    
    echo "Building and pushing $agent_name agent..."
    
    # Create ECR repository
    create_ecr_repo $repo_name
    
    # Build Docker image for ARM64 platform
    docker build \
        --platform linux/arm64 \
        -f $dockerfile_path \
        -t $repo_name:latest \
        -t $ECR_REGISTRY/$repo_name:latest \
        .
    
    # Push to ECR
    docker push $ECR_REGISTRY/$repo_name:latest
    
    echo "Successfully pushed $ECR_REGISTRY/$repo_name:latest"
}

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY

# Build and push all agent images using the main Dockerfile
echo "Building and pushing agent images..."
build_and_push "orchestrator" "Dockerfile"
build_and_push "ner" "Dockerfile"
build_and_push "rag" "Dockerfile"
build_and_push "llm" "Dockerfile"
build_and_push "hybrid" "Dockerfile"

echo "All images built and pushed successfully!"
echo "Images available at:"
echo "  - $ECR_REGISTRY/multilingual-inference-orchestrator:latest"
echo "  - $ECR_REGISTRY/multilingual-inference-ner:latest"
echo "  - $ECR_REGISTRY/multilingual-inference-rag:latest"
echo "  - $ECR_REGISTRY/multilingual-inference-llm:latest"
echo "  - $ECR_REGISTRY/multilingual-inference-hybrid:latest"