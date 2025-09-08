#!/bin/bash

# Deploy ECS Fargate infrastructure for multilingual inference system
# Uses ml-sandbox AWS profile and us-east-1 region with ARM64 platform

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-ml-sandbox}"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="multilingual-inference-cluster"
USE_LOAD_BALANCER="${USE_LOAD_BALANCER:-false}"  # Set to true if ALB is deployed
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

echo "🚀 Deploying ECS infrastructure for multilingual inference system..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Use Load Balancer: $USE_LOAD_BALANCER"

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
    echo "📊 Creating CloudWatch log groups..."
    
    local log_groups=(
        "/ecs/multilingual-inference-orchestrator"
        "/ecs/multilingual-inference-ner"
        "/ecs/multilingual-inference-rag"
        "/ecs/multilingual-inference-llm"
        "/ecs/multilingual-inference-hybrid"
    )
    
    for log_group in "${log_groups[@]}"; do
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
                --tags Environment=production,Platform=ARM64
        else
            echo "✅ Log group exists: $log_group"
        fi
    done
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
    
    # Step 2: Create CloudWatch log groups
    create_log_groups
    
    # Step 3: Create ECS cluster if it doesn't exist
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
    
    # Step 4: Register task definitions
    echo "📋 Registering task definitions..."
    
    local task_definitions=(
        "infrastructure/ecs/task-definitions/orchestrator-task-def.json:multilingual-inference-orchestrator"
        "infrastructure/ecs/task-definitions/ner-task-def.json:multilingual-inference-ner"
        "infrastructure/ecs/task-definitions/rag-task-def.json:multilingual-inference-rag"
        "infrastructure/ecs/task-definitions/llm-task-def.json:multilingual-inference-llm"
        "infrastructure/ecs/task-definitions/hybrid-task-def.json:multilingual-inference-hybrid"
    )
    
    for task_def in "${task_definitions[@]}"; do
        IFS=':' read -r file name <<< "$task_def"
        if [ -f "$file" ]; then
            register_task_definition "$file" "$name"
        else
            echo "⚠️  Task definition file not found: $file"
        fi
    done
    
    # Step 5: Create services
    echo "🚀 Creating/updating services..."
    
    local services=(
        "infrastructure/ecs/services/orchestrator-service.json:multilingual-inference-orchestrator"
        "infrastructure/ecs/services/ner-service.json:multilingual-inference-ner"
        "infrastructure/ecs/services/rag-service.json:multilingual-inference-rag"
        "infrastructure/ecs/services/llm-service.json:multilingual-inference-llm"
        "infrastructure/ecs/services/hybrid-service.json:multilingual-inference-hybrid"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r file name <<< "$service"
        if [ -f "$file" ]; then
            create_or_update_service "$file" "$name"
        else
            echo "⚠️  Service definition file not found: $file"
        fi
    done
    
    # Step 6: Setup auto-scaling (optional)
    echo "📈 Setting up auto-scaling..."
    setup_autoscaling "infrastructure/ecs/autoscaling/orchestrator-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-orchestrator"
    setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-ner"
    setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-rag"
    setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-llm"
    setup_autoscaling "infrastructure/ecs/autoscaling/agent-autoscaling.json" "service/$CLUSTER_NAME/multilingual-inference-hybrid"
    
    echo ""
    echo "🎉 ECS deployment completed successfully!"
    echo "📊 Deployment Summary:"
    echo "   - Cluster: $CLUSTER_NAME"
    echo "   - Region: $AWS_REGION"
    echo "   - Platform: ARM64"
    echo "   - Load Balancer: $USE_LOAD_BALANCER"
    echo ""
    echo "🔍 Check deployment status:"
    echo "   aws ecs list-services --profile $AWS_PROFILE --region $AWS_REGION --cluster $CLUSTER_NAME"
    echo "   aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION --cluster $CLUSTER_NAME --services multilingual-inference-orchestrator"
}

# Run main function
main "$@"