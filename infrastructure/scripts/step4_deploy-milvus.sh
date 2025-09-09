#!/bin/bash

# Deploy Milvus vector database on ECS with ARM64 platform
# Uses ml-sandbox AWS profile and us-east-1 region

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
CLUSTER_NAME="multilingual-inference-cluster"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "Deploying Milvus vector database on ECS..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"

# Function to build and push Milvus Docker image
build_and_push_milvus() {
    echo "Building and pushing Milvus Docker image..."
    
    # Create ECR repository for Milvus
    aws ecr create-repository \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --repository-name multilingual-inference-milvus \
        --image-scanning-configuration scanOnPush=true \
        --tags Key=Environment,Value=production Key=Platform,Value=ARM64 \
        2>/dev/null || echo "Repository multilingual-inference-milvus already exists"
    
    # Login to ECR
    aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Build Docker image for ARM64 platform
    docker build \
        --platform linux/arm64 \
        -f infrastructure/docker/Dockerfile.milvus \
        -t multilingual-inference-milvus:latest \
        -t $ECR_REGISTRY/multilingual-inference-milvus:latest \
        .
    
    # Push to ECR
    docker push $ECR_REGISTRY/multilingual-inference-milvus:latest
    
    echo "Successfully pushed $ECR_REGISTRY/multilingual-inference-milvus:latest"
}

# Function to register Milvus task definition
register_milvus_task_definition() {
    echo "Registering Milvus task definition..."
    
    # Replace ACCOUNT_ID placeholder in task definition
    sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" infrastructure/ecs/task-definitions/milvus-task-def.json > /tmp/milvus-task-def.json
    
    aws ecs register-task-definition \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cli-input-json file:///tmp/milvus-task-def.json
    
    rm /tmp/milvus-task-def.json
    echo "Milvus task definition registered successfully"
}

# Function to create Milvus service
create_milvus_service() {
    echo "Creating Milvus service..."
    
    # Check if service exists
    if aws ecs describe-services \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services multilingual-inference-milvus \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q multilingual-inference-milvus; then
        
        echo "Milvus service exists, updating..."
        aws ecs update-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service multilingual-inference-milvus \
            --task-definition multilingual-inference-milvus
    else
        echo "Creating new Milvus service..."
        
        # Get VPC and subnet information
        VPC_ID=$(aws ec2 describe-vpcs \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=is-default,Values=true" \
            --query 'Vpcs[0].VpcId' \
            --output text)
        
        SUBNET1=$(aws ec2 describe-subnets \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$VPC_ID" \
            --query 'Subnets[0].SubnetId' \
            --output text)
        
        SUBNET2=$(aws ec2 describe-subnets \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$VPC_ID" \
            --query 'Subnets[1].SubnetId' \
            --output text)
        
        SG_ID=$(aws ec2 describe-security-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-names default \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
        
        # Replace placeholders in service definition
        sed "s/ACCOUNT_ID/$ACCOUNT_ID/g; s/subnet-XXXXXXXXX/$SUBNET1/g; s/subnet-YYYYYYYYY/$SUBNET2/g; s/sg-XXXXXXXXX/$SG_ID/g" infrastructure/ecs/services/milvus-service.json > /tmp/milvus-service.json
        
        aws ecs create-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cli-input-json file:///tmp/milvus-service.json
        
        rm /tmp/milvus-service.json
    fi
    
    echo "Milvus service created/updated successfully"
}

# Function to create CloudWatch log groups
create_log_groups() {
    echo "Creating CloudWatch log groups..."
    
    aws logs create-log-group \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --log-group-name "/ecs/multilingual-inference-milvus" \
        --tags Environment=production,Service=milvus,Platform=ARM64 \
        2>/dev/null || echo "Log group already exists"
    
    # Set retention policy
    aws logs put-retention-policy \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --log-group-name "/ecs/multilingual-inference-milvus" \
        --retention-in-days 30
    
    echo "CloudWatch log groups created"
}

# Function to wait for service to be stable
wait_for_service() {
    echo "Waiting for Milvus service to be stable..."
    
    aws ecs wait services-stable \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services multilingual-inference-milvus
    
    echo "Milvus service is stable and running"
}

# Main execution
echo "Starting Milvus deployment..."

# Create CloudWatch log groups
create_log_groups

# Build and push Milvus image
build_and_push_milvus

# Register task definition
register_milvus_task_definition

# Create service
create_milvus_service

# Wait for service to be stable
wait_for_service

echo "Milvus deployment completed successfully!"
echo "Cluster: $CLUSTER_NAME"
echo "Service: multilingual-inference-milvus"
echo "Platform: ARM64"
echo ""
echo "Milvus is now running and accessible at:"
echo "- Internal endpoint: milvus.multilingual-inference.local:19530"
echo "- Metrics endpoint: milvus.multilingual-inference.local:9091"