#!/bin/bash

# Deploy ECS Fargate orchestrator service (monolithic architecture)
# Contains all agents in a single container for simplified deployment

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-ml-sandbox}"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="multilingual-inference-cluster"
USE_LOAD_BALANCER="${USE_LOAD_BALANCER:-false}"  # Set to true if ALB is deployed
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

echo "🚀 Deploying ECS orchestrator service (monolithic architecture)..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Use Load Balancer: $USE_LOAD_BALANCER"
echo ""
echo "ℹ️  This deploys a single service containing all agents:"
echo "   - NER, RAG, LLM, Hybrid, Simple agents in one container"
echo "   - All methods available via 'method' parameter"

# Function to get or create VPC resources
setup_vpc_resources() {
    echo "📡 Setting up VPC resources..."
    
    # Check if default VPC exists
    DEFAULT_VPC=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$DEFAULT_VPC" = "None" ] || [ "$DEFAULT_VPC" = "null" ]; then
        echo "❌ No default VPC found. Creating default VPC..."
        aws ec2 create-default-vpc \
            --profile $AWS_PROFILE \
            --region $AWS_REGION
        
        # Wait for VPC to be available
        sleep 10
        echo "✅ Default VPC created"
    else
        echo "✅ Using existing default VPC: $DEFAULT_VPC"
    fi
}

# Function to create CloudWatch log groups
create_log_groups() {
    echo "📊 Creating CloudWatch log group..."
    
    local log_group="/ecs/multilingual-inference-orchestrator"
    
    if ! aws logs describe-log-groups \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --log-group-name-prefix "$log_group" \
        --query 'logGroups[?logGroupName==`'$log_group'`]' \
        --output text | grep -q "$log_group"; then
        
        echo "Creating log group: $log_group"
        aws logs create-log-group \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --log-group-name "$log_group" \
            --tags Environment=production,Platform=ARM64,Service=orchestrator
    else
        echo "✅ Log group exists: $log_group"
    fi
}

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
    
    echo "🚀 Creating/updating service: $service_name"
    
    # Check if service exists
    if aws ecs describe-services \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services $service_name \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q $service_name; then
        
        echo "ℹ️  Service $service_name exists, updating..."
        aws ecs update-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service $service_name \
            --task-definition $service_name
        echo "✅ Service updated: $service_name"
    else
        echo "🆕 Creating new service: $service_name"
        
        # Get actual subnet and security group IDs
        echo "📡 Retrieving VPC resources..."
        
        # Get default VPC subnets (public subnets for Fargate with public IP)
        SUBNETS=$(aws ec2 describe-subnets \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=default-for-az,Values=true" \
            --query 'Subnets[].SubnetId' \
            --output text)
        
        # Convert to array
        SUBNET_ARRAY=($SUBNETS)
        
        if [ ${#SUBNET_ARRAY[@]} -lt 2 ]; then
            echo "❌ Error: Need at least 2 subnets for ECS Fargate deployment"
            echo "Available subnets: ${SUBNET_ARRAY[@]}"
            exit 1
        fi
        
        SUBNET_1=${SUBNET_ARRAY[0]}
        SUBNET_2=${SUBNET_ARRAY[1]}
        
        # Get default security group
        SECURITY_GROUP=$(aws ec2 describe-security-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --group-names default \
            --query 'SecurityGroups[0].GroupId' \
            --output text)
        
        # Validate subnet ID format
        if [[ ! $SUBNET_1 =~ ^subnet-[0-9a-f]{8,17}$ ]]; then
            echo "❌ Error: Invalid subnet ID format: $SUBNET_1"
            exit 1
        fi
        
        if [[ ! $SUBNET_2 =~ ^subnet-[0-9a-f]{8,17}$ ]]; then
            echo "❌ Error: Invalid subnet ID format: $SUBNET_2"
            exit 1
        fi
        
        echo "✅ Using subnets: $SUBNET_1, $SUBNET_2"
        echo "✅ Using security group: $SECURITY_GROUP"
        
        if [ "$USE_LOAD_BALANCER" = "true" ]; then
            # Use service file with load balancer configuration
            echo "🔗 Using load balancer configuration"
            sed "s/ACCOUNT_ID/$ACCOUNT_ID/g; s/subnet-XXXXXXXXX/$SUBNET_1/g; s/subnet-YYYYYYYYY/$SUBNET_2/g; s/sg-XXXXXXXXX/$SECURITY_GROUP/g" $service_file > /tmp/${service_name}-service.json
            
            aws ecs create-service \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --cli-input-json file:///tmp/${service_name}-service.json
            
            rm /tmp/${service_name}-service.json
        else
            # Create service without load balancer (simpler)
            echo "🚀 Creating service without load balancer"
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
        
        echo "✅ Service created: $service_name"
    fi
}

# Function to clean up old individual agent services
cleanup_old_services() {
    echo "🧹 Cleaning up old individual agent services..."
    
    local old_services=(
        "multilingual-inference-ner"
        "multilingual-inference-rag"
        "multilingual-inference-llm"
        "multilingual-inference-hybrid"
    )
    
    for service_name in "${old_services[@]}"; do
        if aws ecs describe-services \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --services $service_name \
            --query 'services[0].serviceName' \
            --output text 2>/dev/null | grep -q $service_name; then
            
            echo "🗑️  Found old service: $service_name - scaling down to 0"
            aws ecs update-service \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --cluster $CLUSTER_NAME \
                --service $service_name \
                --desired-count 0 \
                >/dev/null 2>&1
            
            echo "⏳ Waiting for service to scale down..."
            aws ecs wait services-stable \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --cluster $CLUSTER_NAME \
                --services $service_name
            
            echo "🗑️  Deleting old service: $service_name"
            aws ecs delete-service \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --cluster $CLUSTER_NAME \
                --service $service_name \
                >/dev/null 2>&1
            
            echo "✅ Cleaned up old service: $service_name"
        fi
    done
}

# Function to setup auto-scaling
setup_autoscaling() {
    local autoscaling_file=$1
    local resource_id=$2
    
    echo "📈 Setting up auto-scaling for: $resource_id"
    
    # Check if the autoscaling role exists
    if ! aws iam get-role \
        --profile $AWS_PROFILE \
        --role-name application-autoscaling-ecs-service \
        >/dev/null 2>&1; then
        
        echo "⚠️  Auto-scaling role not found, skipping auto-scaling setup for $resource_id"
        echo "ℹ️  To enable auto-scaling, create the role: application-autoscaling-ecs-service"
        return 0
    fi
    
    # Register scalable target
    if aws application-autoscaling register-scalable-target \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --service-namespace ecs \
        --resource-id $resource_id \
        --scalable-dimension ecs:service:DesiredCount \
        --min-capacity 1 \
        --max-capacity 10 \
        --role-arn "arn:aws:iam::$ACCOUNT_ID:role/application-autoscaling-ecs-service" \
        >/dev/null 2>&1; then
        echo "✅ Auto-scaling configured for: $resource_id"
    else
        echo "⚠️  Failed to configure auto-scaling for: $resource_id"
    fi
}

# Main deployment process
main() {
    echo "🚀 Starting ECS deployment process..."
    
    # Step 1: Setup VPC resources
    setup_vpc_resources
    
    # Step 2: Clean up old individual agent services
    cleanup_old_services
    
    # Step 3: Create CloudWatch log groups
    create_log_groups
    
    # Step 4: Create ECS cluster if it doesn't exist
    echo "🏗️  Creating ECS cluster: $CLUSTER_NAME"
    if aws ecs create-cluster \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster-name $CLUSTER_NAME \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
        --tags key=Environment,value=production key=Platform,value=ARM64 \
        >/dev/null 2>&1; then
        echo "✅ ECS cluster created: $CLUSTER_NAME"
    else
        echo "ℹ️  ECS cluster already exists: $CLUSTER_NAME"
    fi
    
    # Step 5: Register orchestrator task definition
    echo "📋 Registering orchestrator task definition..."
    
    local task_def_file="infrastructure/ecs/task-definitions/orchestrator-task-def.json"
    local task_def_name="multilingual-inference-orchestrator"
    
    if [ -f "$task_def_file" ]; then
        register_task_definition "$task_def_file" "$task_def_name"
        echo "✅ Orchestrator task definition registered"
    else
        echo "❌ Task definition file not found: $task_def_file"
        exit 1
    fi
    
    # Step 6: Create orchestrator service
    echo "🚀 Creating/updating orchestrator service..."
    
    local service_file="infrastructure/ecs/services/orchestrator-service.json"
    local service_name="multilingual-inference-orchestrator"
    
    if [ -f "$service_file" ]; then
        create_or_update_service "$service_file" "$service_name"
        echo "✅ Orchestrator service deployed"
    else
        echo "❌ Service definition file not found: $service_file"
        exit 1
    fi
    
    # Step 7: Setup auto-scaling for orchestrator
    echo "📈 Setting up auto-scaling for orchestrator..."
    setup_autoscaling "infrastructure/ecs/autoscaling/orchestrator-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-orchestrator"
    
    echo ""
    echo "🎉 ECS orchestrator deployment completed successfully!"
    echo "📊 Deployment Summary:"
    echo "   - Cluster: $CLUSTER_NAME"
    echo "   - Service: multilingual-inference-orchestrator (contains all agents)"
    echo "   - Region: $AWS_REGION"
    echo "   - Platform: ARM64"
    echo "   - Load Balancer: $USE_LOAD_BALANCER"
    echo ""
    echo "🧠 Available Agents (all in one container):"
    echo "   - NER, RAG, LLM, Hybrid, Simple agents"
    echo "   - Access via 'method' parameter in API requests"
    echo ""
    echo "🔍 Check deployment status:"
    echo "   aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION --cluster $CLUSTER_NAME --services multilingual-inference-orchestrator"
    echo ""
    echo "🧪 Test the API (replace <ALB-DNS> with your load balancer DNS):"
    echo "   curl -X POST \"http://<ALB-DNS>/infer\" -H \"Content-Type: application/json\" -d '{\"product_name\": \"Samsung Galaxy S24\", \"method\": \"orchestrator\"}'"
}

# Run main function
main "$@"