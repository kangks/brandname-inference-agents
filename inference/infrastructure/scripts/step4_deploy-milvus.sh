#!/bin/bash

# Deploy Milvus with configurable storage options
# Default: Local disk storage (ephemeral, faster)
# Optional: EFS storage (persistent, shared)

set -e

# Configuration
AWS_PROFILE="${AWS_PROFILE:-ml-sandbox}"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="multilingual-inference-cluster"
STORAGE_TYPE="${STORAGE_TYPE:-local}"  # Options: local (default), efs
MILVUS_MODE="${MILVUS_MODE:-standalone}"  # Options: standalone, distributed
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "🚀 Deploying Milvus with configurable storage..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Storage Type: $STORAGE_TYPE"
echo "Milvus Mode: $MILVUS_MODE"
echo ""

# Validate storage type
if [[ "$STORAGE_TYPE" != "local" && "$STORAGE_TYPE" != "efs" ]]; then
    echo "❌ Error: STORAGE_TYPE must be 'local' or 'efs'"
    echo "Usage examples:"
    echo "  STORAGE_TYPE=local $0    # Default: Local disk (ephemeral)"
    echo "  STORAGE_TYPE=efs $0      # EFS storage (persistent)"
    exit 1
fi

# Function to deploy EFS if needed
deploy_efs() {
    if [ "$STORAGE_TYPE" = "efs" ]; then
        echo "🗄️  Deploying EFS storage for Milvus..."
        
        # Check if EFS CloudFormation stack exists
        if aws cloudformation describe-stacks \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --stack-name multilingual-inference-efs \
            >/dev/null 2>&1; then
            echo "✅ EFS stack already exists"
        else
            echo "🆕 Creating EFS stack..."
            aws cloudformation create-stack \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --stack-name multilingual-inference-efs \
                --template-body file://infrastructure/cloudformation/efs-storage.yaml \
                --capabilities CAPABILITY_IAM \
                --tags Key=Environment,Value=production Key=Service,Value=milvus Key=StorageType,Value=efs
            
            echo "⏳ Waiting for EFS stack to complete..."
            aws cloudformation wait stack-create-complete \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --stack-name multilingual-inference-efs
            
            echo "✅ EFS stack created successfully"
        fi
    else
        echo "ℹ️  Using local disk storage - no EFS deployment needed"
    fi
}

# Function to build and push Milvus Docker image (only for EFS mode)
build_and_push_milvus() {
    if [ "$STORAGE_TYPE" = "efs" ] && [ "$MILVUS_MODE" = "distributed" ]; then
        echo "🐳 Building and pushing Milvus Docker image for distributed mode..."
        
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
        
        echo "✅ Successfully pushed $ECR_REGISTRY/multilingual-inference-milvus:latest"
    else
        echo "ℹ️  Using official Milvus image - no custom build needed"
    fi
}

# Function to register task definition based on storage type
register_task_definition() {
    echo "📋 Registering Milvus task definition for $STORAGE_TYPE storage..."
    
    local task_def_file
    local task_def_name
    
    if [ "$STORAGE_TYPE" = "local" ]; then
        task_def_file="infrastructure/ecs/task-definitions/milvus-local-storage-task-def.json"
        task_def_name="multilingual-inference-milvus-local"
    else
        if [ "$MILVUS_MODE" = "standalone" ]; then
            task_def_file="infrastructure/ecs/task-definitions/milvus-standalone-task-def.json"
            task_def_name="multilingual-inference-milvus-efs"
        else
            task_def_file="infrastructure/ecs/task-definitions/milvus-task-def.json"
            task_def_name="multilingual-inference-milvus-efs-distributed"
        fi
    fi
    
    if [ ! -f "$task_def_file" ]; then
        echo "❌ Task definition file not found: $task_def_file"
        exit 1
    fi
    
    echo "Using task definition: $task_def_file"
    
    # Replace ACCOUNT_ID placeholder in task definition
    sed "s/ACCOUNT_ID/$ACCOUNT_ID/g" $task_def_file > /tmp/${task_def_name}-task-def.json
    
    aws ecs register-task-definition \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cli-input-json file:///tmp/${task_def_name}-task-def.json
    
    rm /tmp/${task_def_name}-task-def.json
    echo "✅ Task definition registered: $task_def_name"
}

# Function to create or update Milvus service
create_or_update_service() {
    local service_name="multilingual-inference-milvus"
    local task_def_name
    
    if [ "$STORAGE_TYPE" = "local" ]; then
        task_def_name="multilingual-inference-milvus-local"
    else
        if [ "$MILVUS_MODE" = "standalone" ]; then
            task_def_name="multilingual-inference-milvus-efs"
        else
            task_def_name="multilingual-inference-milvus-efs-distributed"
        fi
    fi
    
    echo "🚀 Creating/updating Milvus service with $STORAGE_TYPE storage..."
    
    # Get VPC resources
    echo "📡 Retrieving VPC resources..."
    
    SUBNETS=$(aws ec2 describe-subnets \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=default-for-az,Values=true" \
        --query 'Subnets[].SubnetId' \
        --output text)
    
    SUBNET_ARRAY=($SUBNETS)
    
    if [ ${#SUBNET_ARRAY[@]} -lt 2 ]; then
        echo "❌ Error: Need at least 2 subnets for ECS Fargate deployment"
        exit 1
    fi
    
    SUBNET_1=${SUBNET_ARRAY[0]}
    SUBNET_2=${SUBNET_ARRAY[1]}
    
    SECURITY_GROUP=$(aws ec2 describe-security-groups \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --group-names default \
        --query 'SecurityGroups[0].GroupId' \
        --output text)
    
    echo "✅ Using subnets: $SUBNET_1, $SUBNET_2"
    echo "✅ Using security group: $SECURITY_GROUP"
    
    # Check if service exists
    if aws ecs describe-services \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services $service_name \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q $service_name; then
        
        echo "ℹ️  Service $service_name exists, updating to use $STORAGE_TYPE storage..."
        aws ecs update-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service $service_name \
            --task-definition $task_def_name
        echo "✅ Service updated: $service_name"
    else
        echo "🆕 Creating new Milvus service with $STORAGE_TYPE storage..."
        
        aws ecs create-service \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster $CLUSTER_NAME \
            --service-name $service_name \
            --task-definition $task_def_name \
            --desired-count 1 \
            --launch-type FARGATE \
            --platform-version LATEST \
            --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
            --enable-execute-command \
            --tags key=Environment,value=production key=Service,value=milvus key=StorageType,value=$STORAGE_TYPE
        
        echo "✅ Service created: $service_name"
    fi
}

# Function to create CloudWatch log groups
create_log_groups() {
    echo "📊 Creating CloudWatch log group..."
    
    local log_group="/ecs/multilingual-inference-milvus"
    
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
            --tags Environment=production,Platform=ARM64,Service=milvus,StorageType=$STORAGE_TYPE
    else
        echo "✅ Log group exists: $log_group"
    fi
    
    # Set retention policy
    aws logs put-retention-policy \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --log-group-name "$log_group" \
        --retention-in-days 30 \
        2>/dev/null || echo "Retention policy already set"
}

# Function to wait for service to be stable
wait_for_service() {
    echo "⏳ Waiting for Milvus service to be stable..."
    
    aws ecs wait services-stable \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --services multilingual-inference-milvus
    
    echo "✅ Milvus service is stable and running"
}

# Show usage if help requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Milvus Deployment Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Environment Variables:"
    echo "  STORAGE_TYPE    Storage type: 'local' (default) or 'efs'"
    echo "  MILVUS_MODE     Milvus mode: 'standalone' (default) or 'distributed'"
    echo "  AWS_PROFILE     AWS profile to use (default: ml-sandbox)"
    echo "  AWS_REGION      AWS region (default: us-east-1)"
    echo ""
    echo "Examples:"
    echo "  # Deploy with local storage (default, faster, ephemeral)"
    echo "  $0"
    echo "  STORAGE_TYPE=local $0"
    echo ""
    echo "  # Deploy with EFS storage (persistent, shared)"
    echo "  STORAGE_TYPE=efs $0"
    echo ""
    echo "  # Deploy distributed mode with EFS"
    echo "  STORAGE_TYPE=efs MILVUS_MODE=distributed $0"
    echo ""
    echo "Storage Types:"
    echo "  local - Container local disk (default)"
    echo "          + Faster performance"
    echo "          + Lower cost"
    echo "          - Data lost on container restart"
    echo "          - Not shared between containers"
    echo ""
    echo "  efs   - Amazon Elastic File System"
    echo "          + Persistent storage"
    echo "          + Shared between containers"
    echo "          + Suitable for production"
    echo "          - Slightly slower than local"
    echo "          - Additional EFS costs"
    exit 0
fi

# Main deployment process
main() {
    echo "🚀 Starting Milvus deployment with $STORAGE_TYPE storage..."
    
    # Step 1: Create CloudWatch log groups
    create_log_groups
    
    # Step 2: Deploy EFS if needed
    deploy_efs
    
    # Step 3: Create ECS cluster if it doesn't exist
    echo "🏗️  Ensuring ECS cluster exists: $CLUSTER_NAME"
    aws ecs create-cluster \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster-name $CLUSTER_NAME \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
        --tags key=Environment,value=production key=Platform,value=ARM64 \
        >/dev/null 2>&1 || echo "ℹ️  ECS cluster already exists: $CLUSTER_NAME"
    
    # Step 4: Build and push image if needed
    build_and_push_milvus
    
    # Step 5: Register task definition
    register_task_definition
    
    # Step 6: Create or update service
    create_or_update_service
    
    # Step 7: Wait for service to be stable
    wait_for_service
    
    echo ""
    echo "🎉 Milvus deployment completed successfully!"
    echo "📊 Deployment Summary:"
    echo "   - Cluster: $CLUSTER_NAME"
    echo "   - Service: multilingual-inference-milvus"
    echo "   - Storage Type: $STORAGE_TYPE"
    echo "   - Mode: $MILVUS_MODE"
    echo "   - Region: $AWS_REGION"
    echo "   - Platform: ARM64"
    echo ""
    
    if [ "$STORAGE_TYPE" = "local" ]; then
        echo "💾 Local Storage Configuration:"
        echo "   - Data stored in container's local disk"
        echo "   - Faster performance, ephemeral storage"
        echo "   - Data lost when container restarts"
        echo "   - Suitable for development and testing"
    else
        echo "🗄️  EFS Storage Configuration:"
        echo "   - Data stored in Amazon EFS"
        echo "   - Persistent storage across container restarts"
        echo "   - Shared storage for distributed deployments"
        echo "   - Suitable for production environments"
    fi
    
    echo ""
    echo "🔍 Check deployment status:"
    echo "   aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION --cluster $CLUSTER_NAME --services multilingual-inference-milvus"
    echo ""
    echo "🧪 Test Milvus connection:"
    echo "   # Get service public IP and test port 19530"
    echo ""
    echo "📚 Documentation:"
    echo "   - Storage Guide: ../storage/README.md"
    echo "   - Architecture Guide: ../docs/ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md"
}

# Run main function
main "$@"