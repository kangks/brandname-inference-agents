#!/bin/bash

# Deploy only the orchestrator service to ECS
# Uses ml-sandbox AWS profile and us-east-1 region

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
CLUSTER_NAME="multilingual-inference-cluster"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

# Get subnet and security group IDs from CloudFormation stack
PUBLIC_SUBNET_1=$(aws cloudformation describe-stacks --stack-name multilingual-inference --profile $AWS_PROFILE --region $AWS_REGION --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet1Id`].OutputValue' --output text)
PUBLIC_SUBNET_2=$(aws cloudformation describe-stacks --stack-name multilingual-inference --profile $AWS_PROFILE --region $AWS_REGION --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet2Id`].OutputValue' --output text)
ECS_SECURITY_GROUP=$(aws cloudformation describe-stacks --stack-name multilingual-inference --profile $AWS_PROFILE --region $AWS_REGION --query 'Stacks[0].Outputs[?OutputKey==`ECSSecurityGroupId`].OutputValue' --output text)
TARGET_GROUP_ARN=$(aws cloudformation describe-stacks --stack-name multilingual-inference --profile $AWS_PROFILE --region $AWS_REGION --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorTargetGroupArn`].OutputValue' --output text)

echo "Deploying orchestrator service to ECS..."
echo "Cluster: $CLUSTER_NAME"
echo "Subnets: $PUBLIC_SUBNET_1, $PUBLIC_SUBNET_2"
echo "Security Group: $ECS_SECURITY_GROUP"
echo "Target Group: $TARGET_GROUP_ARN"

# Check if service exists
if aws ecs describe-services \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --services multilingual-inference-orchestrator \
    --query 'services[0].serviceName' \
    --output text 2>/dev/null | grep -q multilingual-inference-orchestrator; then
    
    echo "Service exists, updating desired count to 1..."
    aws ecs update-service \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --cluster $CLUSTER_NAME \
        --service multilingual-inference-orchestrator \
        --desired-count 1
else
    echo "Service doesn't exist, creating..."
    # This shouldn't happen since the service was created in the previous step
fi

echo "Waiting for service to stabilize..."
aws ecs wait services-stable \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --services multilingual-inference-orchestrator

echo "Checking service status..."
aws ecs describe-services \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster $CLUSTER_NAME \
    --services multilingual-inference-orchestrator \
    --query 'services[0].[serviceName,status,runningCount,desiredCount,taskDefinition]' \
    --output table

echo "Orchestrator service deployment completed!"