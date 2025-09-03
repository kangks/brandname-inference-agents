#!/bin/bash

# Deploy orchestrator with default agents to AWS ECS
# This script deploys the updated orchestrator that automatically registers default agents

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STACK_NAME="${STACK_NAME:-multilingual-inference}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-ml-sandbox}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed"
        exit 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        log_error "AWS credentials not configured for profile: $AWS_PROFILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Get stack outputs
get_stack_outputs() {
    log_info "Getting CloudFormation stack outputs..."
    
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
    
    # Get stack outputs
    STACK_OUTPUTS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    if [ -z "$STACK_OUTPUTS" ] || [ "$STACK_OUTPUTS" = "null" ]; then
        log_error "Could not retrieve stack outputs for $STACK_NAME"
        exit 1
    fi
    
    # Extract required values
    ECS_CLUSTER_NAME=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="ECSClusterName") | .OutputValue')
    LOAD_BALANCER_DNS=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="LoadBalancerDNS") | .OutputValue')
    TARGET_GROUP_ARN=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="OrchestratorTargetGroupArn") | .OutputValue')
    PRIVATE_SUBNET_1_ID=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="PrivateSubnet1Id") | .OutputValue')
    PRIVATE_SUBNET_2_ID=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="PrivateSubnet2Id") | .OutputValue')
    ECS_SECURITY_GROUP_ID=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="ECSSecurityGroupId") | .OutputValue')
    
    log_success "Retrieved stack outputs"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
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
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true
    
    # Login to ECR
    aws ecr get-login-password \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REPOSITORY_URI"
    
    # Build image for ARM64
    cd "$PROJECT_ROOT"
    docker build \
        --platform linux/arm64 \
        -t "$ECR_REPOSITORY_NAME:latest" \
        -t "$ECR_REPOSITORY_URI:latest" \
        -f Dockerfile .
    
    # Push image
    docker push "$ECR_REPOSITORY_URI:latest"
    
    log_success "Docker image built and pushed: $ECR_REPOSITORY_URI:latest"
}

# Register task definition
register_task_definition() {
    log_info "Registering ECS task definition..."
    
    # Create task definition JSON
    TASK_DEF_JSON=$(cat << EOF
{
  "family": "multilingual-inference-orchestrator-with-agents",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "runtimePlatform": {
    "cpuArchitecture": "ARM64",
    "operatingSystemFamily": "LINUX"
  },
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${STACK_NAME}-ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${STACK_NAME}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "orchestrator-with-agents",
      "image": "${ECR_REPOSITORY_URI}:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp",
          "name": "http"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "${AWS_REGION}"
        },
        {
          "name": "HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "PORT",
          "value": "8080"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${STACK_NAME}-orchestrator",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8080/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF
)
    
    # Register task definition
    TASK_DEF_ARN=$(echo "$TASK_DEF_JSON" | aws ecs register-task-definition \
        --cli-input-json file:///dev/stdin \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    log_success "Task definition registered: $TASK_DEF_ARN"
}

# Deploy or update ECS service
deploy_service() {
    log_info "Deploying ECS service..."
    
    SERVICE_NAME="orchestrator-with-agents"
    
    # Check if service exists
    if aws ecs describe-services \
        --cluster "$ECS_CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'services[0].serviceName' \
        --output text 2>/dev/null | grep -q "$SERVICE_NAME"; then
        
        log_info "Updating existing service..."
        
        # Update service
        aws ecs update-service \
            --cluster "$ECS_CLUSTER_NAME" \
            --service "$SERVICE_NAME" \
            --task-definition "multilingual-inference-orchestrator-with-agents" \
            --profile "$AWS_PROFILE" \
            --region "$AWS_REGION" \
            --query 'service.serviceName' \
            --output text
        
        log_success "Service updated: $SERVICE_NAME"
    else
        log_info "Creating new service..."
        
        # Create service
        aws ecs create-service \
            --cluster "$ECS_CLUSTER_NAME" \
            --service-name "$SERVICE_NAME" \
            --task-definition "multilingual-inference-orchestrator-with-agents" \
            --desired-count 1 \
            --launch-type FARGATE \
            --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNET_1_ID,$PRIVATE_SUBNET_2_ID],securityGroups=[$ECS_SECURITY_GROUP_ID],assignPublicIp=DISABLED}" \
            --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=orchestrator-with-agents,containerPort=8080" \
            --profile "$AWS_PROFILE" \
            --region "$AWS_REGION" \
            --query 'service.serviceName' \
            --output text
        
        log_success "Service created: $SERVICE_NAME"
    fi
}

# Wait for deployment to complete
wait_for_deployment() {
    log_info "Waiting for deployment to complete..."
    
    SERVICE_NAME="orchestrator-with-agents"
    
    # Wait for service to be stable
    aws ecs wait services-stable \
        --cluster "$ECS_CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION"
    
    log_success "Deployment completed successfully"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Get service status
    SERVICE_STATUS=$(aws ecs describe-services \
        --cluster "$ECS_CLUSTER_NAME" \
        --services "orchestrator-with-agents" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'services[0].status' \
        --output text)
    
    if [ "$SERVICE_STATUS" = "ACTIVE" ]; then
        log_success "Service is active"
    else
        log_error "Service status: $SERVICE_STATUS"
        exit 1
    fi
    
    # Test health endpoint
    if [ -n "$LOAD_BALANCER_DNS" ]; then
        log_info "Testing health endpoint..."
        
        # Wait a bit for load balancer to be ready
        sleep 30
        
        if curl -f "http://$LOAD_BALANCER_DNS/health" &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed - service may still be starting up"
        fi
    fi
}

# Main deployment function
main() {
    log_info "Starting orchestrator with agents deployment..."
    log_info "Stack: $STACK_NAME"
    log_info "Region: $AWS_REGION"
    log_info "Profile: $AWS_PROFILE"
    log_info "Environment: $ENVIRONMENT"
    
    check_prerequisites
    get_stack_outputs
    build_and_push_image
    register_task_definition
    deploy_service
    wait_for_deployment
    validate_deployment
    
    log_success "Orchestrator with agents deployment completed successfully!"
    log_info "Load Balancer DNS: $LOAD_BALANCER_DNS"
    log_info "Health Check: http://$LOAD_BALANCER_DNS/health"
    log_info "Inference Endpoint: http://$LOAD_BALANCER_DNS/infer"
}

# Run main function
main "$@"
