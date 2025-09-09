#!/bin/bash

# Simple ECS Fargate deployment script for multilingual inference system
# Focuses on basic functionality without complex networking

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
CLUSTER_NAME="multilingual-inference-cluster"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

echo "🚀 Deploying ECS infrastructure (Simple Mode)"
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"

# Function to get VPC resources
get_vpc_resources() {
    echo "📡 Getting VPC resources..."
    
    # Get default VPC
    VPC_ID=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)
    
    if [ "$VPC_ID" = "None" ] || [ "$VPC_ID" = "null" ]; then
        echo "❌ No default VPC found. Creating one..."
        aws ec2 create-default-vpc --profile $AWS_PROFILE --region $AWS_REGION
        sleep 5
        
        VPC_ID=$(aws ec2 describe-vpcs \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=is-default,Values=true" \
            --query 'Vpcs[0].VpcId' \
            --output text)
    fi
    
    echo "✅ Using VPC: $VPC_ID"
    
    # Get subnets in different AZs
    SUBNETS=$(aws ec2 describe-subnets \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
        --query 'Subnets[].SubnetId' \
        --output text)
    
    SUBNET_ARRAY=($SUBNETS)
    
    if [ ${#SUBNET_ARRAY[@]} -lt 2 ]; then
        echo "❌ Need at least 2 subnets. Found: ${#SUBNET_ARRAY[@]}"
        echo "Available subnets: ${SUBNET_ARRAY[@]}"
        exit 1
    fi
    
    SUBNET_1=${SUBNET_ARRAY[0]}
    SUBNET_2=${SUBNET_ARRAY[1]}
    
    # Get default security group
    SECURITY_GROUP=$(aws ec2 describe-security-groups \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)
    
    echo "✅ Subnets: $SUBNET_1, $SUBNET_2"
    echo "✅ Security Group: $SECURITY_GROUP"
    
    # Validate format
    if [[ ! $SUBNET_1 =~ ^subnet-[0-9a-f]{8,17}$ ]]; then
        echo "❌ Invalid subnet format: $SUBNET_1"
        exit 1
    fi
    
    if [[ ! $SUBNET_2 =~ ^subnet-[0-9a-f]{8,17}$ ]]; then
        echo "❌ Invalid subnet format: $SUBNET_2"
        exit 1
    fi
}

# Function to create ECS cluster
create_cluster() {
    echo "🏗️  Creating ECS cluster: $CLUSTER_NAME"
    
    aws ecs create-cluster \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster-name $CLUSTER_NAME \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
        --tags key=Environment,value=production key=Platform,value=ARM64 \
        2>/dev/null || echo "ℹ️  Cluster already exists"
}

# Function to register task definition
register_task_definition() {
    local task_def_file=$1
    local task_def_name=$2
    
    echo "📋 Registering task definition: $task_def_name"
    
    if [ ! -f "$task_def_file" ]; then
        echo "❌ Task definition file not found: $task_def_file"
        return 1
    fi
    
    # Replace ACCOUNT_ID placeholder
    sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" "$task_def_file" > "/tmp/${task_def_name}-task-def.json"
    
    aws ecs register-task-definition \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cli-input-json "file:///tmp/${task_def_name}-task-def.json"
    
    rm "/tmp/${task_def_name}-task-def.json"
    echo "✅ Task definition registered: $task_def_name"
}

# Function to create service
create_service() {
    local service_name=$1
    
    echo "🚀 Creating service: $service_name"
    
    # Check if service exists
    if aws ecs describe-services \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services $service_name \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q "$service_name"; then
        
        echo "ℹ️  Service $service_name exists, updating..."
        aws ecs update-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service $service_name \
            --task-definition $service_name
    else
        echo "🆕 Creating new service: $service_name"
        
        # Create service using AWS CLI directly (simpler than JSON files)
        aws ecs create-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service-name $service_name \
            --task-definition $service_name \
            --desired-count 1 \
            --launch-type FARGATE \
            --platform-version LATEST \
            --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
            --enable-execute-command \
            --tags key=Environment,value=production key=Service,value=$service_name
    fi
    
    echo "✅ Service created/updated: $service_name"
}

# Main deployment process
main() {
    echo "🚀 Starting ECS deployment..."
    
    # Step 1: Get VPC resources
    get_vpc_resources
    
    # Step 2: Create cluster
    create_cluster
    
    # Step 3: Register task definitions
    echo "📋 Registering task definitions..."
    
    # Only deploy orchestrator for now (simpler)
    if [ -f "infrastructure/ecs/task-definitions/orchestrator-task-def.json" ]; then
        register_task_definition "infrastructure/ecs/task-definitions/orchestrator-task-def.json" "multilingual-inference-orchestrator"
    else
        echo "⚠️  Orchestrator task definition not found, skipping..."
    fi
    
    # Step 4: Create services
    echo "🚀 Creating services..."
    create_service "multilingual-inference-orchestrator"
    
    echo "🎉 ECS deployment completed successfully!"
    echo "📊 Deployment Summary:"
    echo "   - Cluster: $CLUSTER_NAME"
    echo "   - Region: $AWS_REGION"
    echo "   - VPC: $VPC_ID"
    echo "   - Subnets: $SUBNET_1, $SUBNET_2"
    echo "   - Security Group: $SECURITY_GROUP"
    echo ""
    echo "🔍 Check deployment status:"
    echo "   aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION --cluster $CLUSTER_NAME --services multilingual-inference-orchestrator"
}

# Run main function
main "$@"