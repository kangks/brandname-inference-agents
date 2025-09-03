#!/bin/bash

# Deploy ECS Fargate infrastructure for multilingual inference system
# Uses ml-sandbox AWS profile and us-east-1 region with ARM64 platform

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
CLUSTER_NAME="multilingual-inference-cluster"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

echo "Deploying ECS infrastructure for multilingual inference system..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"

# Function to register task definition
register_task_definition() {
    local task_def_file=$1
    local task_def_name=$2
    
    echo "Registering task definition: $task_def_name"
    
    # Replace ACCOUNT_ID placeholder in task definition
    sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" $task_def_file > /tmp/${task_def_name}-task-def.json
    
    aws ecs register-task-definition \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cli-input-json file:///tmp/${task_def_name}-task-def.json
    
    rm /tmp/${task_def_name}-task-def.json
}

# Function to create or update service
create_or_update_service() {
    local service_file=$1
    local service_name=$2
    
    echo "Creating/updating service: $service_name"
    
    # Check if service exists
    if aws ecs describe-services \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services $service_name \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q $service_name; then
        
        echo "Service $service_name exists, updating..."
        aws ecs update-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service $service_name \
            --task-definition $service_name
    else
        echo "Creating new service: $service_name"
        # Replace placeholders in service definition
        sed "s/ACCOUNT_ID/$ACCOUNT_ID/g; s/subnet-XXXXXXXXX/subnet-$(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --query 'Subnets[0].SubnetId' --output text)/g; s/subnet-YYYYYYYYY/subnet-$(aws ec2 describe-subnets --profile $AWS_PROFILE --region $AWS_REGION --query 'Subnets[1].SubnetId' --output text)/g; s/sg-XXXXXXXXX/sg-$(aws ec2 describe-security-groups --profile $AWS_PROFILE --region $AWS_REGION --group-names default --query 'SecurityGroups[0].GroupId' --output text)/g" $service_file > /tmp/${service_name}-service.json
        
        aws ecs create-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cli-input-json file:///tmp/${service_name}-service.json
        
        rm /tmp/${service_name}-service.json
    fi
}

# Function to setup auto-scaling
setup_autoscaling() {
    local autoscaling_file=$1
    local resource_id=$2
    
    echo "Setting up auto-scaling for: $resource_id"
    
    # Register scalable target
    aws application-autoscaling register-scalable-target \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --service-namespace ecs \
        --resource-id $resource_id \
        --scalable-dimension ecs:service:DesiredCount \
        --min-capacity 1 \
        --max-capacity 10 \
        --role-arn "arn:aws:iam::$ACCOUNT_ID:role/application-autoscaling-ecs-service"
}

# Create ECS cluster if it doesn't exist
echo "Creating ECS cluster: $CLUSTER_NAME"
aws ecs create-cluster \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster-name $CLUSTER_NAME \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    --tags key=Environment,value=production key=Platform,value=ARM64 || echo "Cluster already exists"

# Register task definitions
echo "Registering task definitions..."
register_task_definition "infrastructure/ecs/task-definitions/orchestrator-task-def.json" "multilingual-inference-orchestrator"
register_task_definition "infrastructure/ecs/task-definitions/ner-task-def.json" "multilingual-inference-ner"
register_task_definition "infrastructure/ecs/task-definitions/rag-task-def.json" "multilingual-inference-rag"
register_task_definition "infrastructure/ecs/task-definitions/llm-task-def.json" "multilingual-inference-llm"
register_task_definition "infrastructure/ecs/task-definitions/hybrid-task-def.json" "multilingual-inference-hybrid"

# Create services
echo "Creating/updating services..."
create_or_update_service "infrastructure/ecs/services/orchestrator-service.json" "multilingual-inference-orchestrator"
create_or_update_service "infrastructure/ecs/services/ner-service.json" "multilingual-inference-ner"
create_or_update_service "infrastructure/ecs/services/rag-service.json" "multilingual-inference-rag"
create_or_update_service "infrastructure/ecs/services/llm-service.json" "multilingual-inference-llm"
create_or_update_service "infrastructure/ecs/services/hybrid-service.json" "multilingual-inference-hybrid"

# Setup auto-scaling
echo "Setting up auto-scaling..."
setup_autoscaling "infrastructure/ecs/autoscaling/orchestrator-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-orchestrator"
setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-ner"
setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-rag"
setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-llm"
setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-hybrid"

echo "ECS deployment completed successfully!"
echo "Cluster: $CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "Platform: ARM64"