#!/bin/bash

# Deploy complete CloudFormation infrastructure for multilingual inference system
# Uses ml-sandbox AWS profile and us-east-1 region with ARM64 platform

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
STACK_NAME="multilingual-inference"
ENVIRONMENT="production"
PLATFORM="ARM64"

echo "Deploying CloudFormation infrastructure for multilingual inference system..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Stack Name: $STACK_NAME"
echo "Environment: $ENVIRONMENT"
echo "Platform: $PLATFORM"

# Function to upload templates to S3
upload_templates() {
    local bucket_name="${STACK_NAME}-templates-$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)"
    
    echo "Creating S3 bucket for CloudFormation templates..."
    
    # Create S3 bucket for templates
    aws s3 mb s3://$bucket_name \
        --profile $AWS_PROFILE \
        --region $AWS_REGION || echo "Bucket already exists"
    
    # Upload nested stack templates
    echo "Uploading CloudFormation templates to S3..."
    aws s3 cp infrastructure/cloudformation/ecs-stack.yaml s3://$bucket_name/ \
        --profile $AWS_PROFILE \
        --region $AWS_REGION
    
    aws s3 cp infrastructure/cloudformation/alb-stack.yaml s3://$bucket_name/ \
        --profile $AWS_PROFILE \
        --region $AWS_REGION
    
    aws s3 cp infrastructure/cloudformation/storage-stack.yaml s3://$bucket_name/ \
        --profile $AWS_PROFILE \
        --region $AWS_REGION
    
    aws s3 cp infrastructure/cloudformation/monitoring-stack.yaml s3://$bucket_name/ \
        --profile $AWS_PROFILE \
        --region $AWS_REGION
    
    echo "Templates uploaded to S3 bucket: $bucket_name"
    echo $bucket_name
}

# Function to deploy main stack
deploy_main_stack() {
    local template_bucket=$1
    
    echo "Deploying main CloudFormation stack..."
    
    # Use the main stack template directly (already has correct S3 URLs)
    cp infrastructure/cloudformation/main-stack.yaml /tmp/main-stack-updated.yaml
    
    # Check if stack exists
    if aws cloudformation describe-stacks \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].StackName' \
        --output text 2>/dev/null | grep -q $STACK_NAME; then
        
        echo "Stack $STACK_NAME exists, updating..."
        aws cloudformation update-stack \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --stack-name $STACK_NAME \
            --template-body file:///tmp/main-stack-updated.yaml \
            --parameters ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
                        ParameterKey=Platform,ParameterValue=$PLATFORM \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
            --tags Key=Environment,Value=$ENVIRONMENT \
                   Key=Platform,Value=$PLATFORM \
                   Key=Project,Value=multilingual-inference
        
        echo "Waiting for stack update to complete..."
        aws cloudformation wait stack-update-complete \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --stack-name $STACK_NAME
    else
        echo "Creating new stack: $STACK_NAME"
        aws cloudformation create-stack \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --stack-name $STACK_NAME \
            --template-body file:///tmp/main-stack-updated.yaml \
            --parameters ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
                        ParameterKey=Platform,ParameterValue=$PLATFORM \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
            --tags Key=Environment,Value=$ENVIRONMENT \
                   Key=Platform,Value=$PLATFORM \
                   Key=Project,Value=multilingual-inference
        
        echo "Waiting for stack creation to complete..."
        aws cloudformation wait stack-create-complete \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --stack-name $STACK_NAME
    fi
    
    rm /tmp/main-stack-updated.yaml
    echo "Main stack deployment completed successfully"
}

# Function to get stack outputs
get_stack_outputs() {
    echo "Retrieving stack outputs..."
    
    aws cloudformation describe-stacks \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs' \
        --output table
}

# Function to validate templates
validate_templates() {
    echo "Validating CloudFormation templates..."
    
    local templates=(
        "infrastructure/cloudformation/main-stack.yaml"
        "infrastructure/cloudformation/ecs-stack.yaml"
        "infrastructure/cloudformation/alb-stack.yaml"
        "infrastructure/cloudformation/storage-stack.yaml"
        "infrastructure/cloudformation/monitoring-stack.yaml"
    )
    
    for template in "${templates[@]}"; do
        echo "Validating $template..."
        aws cloudformation validate-template \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --template-body file://$template > /dev/null
        echo "✓ $template is valid"
    done
    
    echo "All templates validated successfully"
}

# Function to create change set for review
create_change_set() {
    local template_bucket=$1
    local change_set_name="changeset-$(date +%Y%m%d-%H%M%S)"
    
    echo "Creating change set for review: $change_set_name"
    
    # Use the main stack template directly (already has correct S3 URLs)
    cp infrastructure/cloudformation/main-stack.yaml /tmp/main-stack-updated.yaml
    
    aws cloudformation create-change-set \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --stack-name $STACK_NAME \
        --change-set-name $change_set_name \
        --template-body file:///tmp/main-stack-updated.yaml \
        --parameters ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
                    ParameterKey=Platform,ParameterValue=$PLATFORM \
        --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
        --tags Key=Environment,Value=$ENVIRONMENT \
               Key=Platform,Value=$PLATFORM \
               Key=Project,Value=multilingual-inference
    
    echo "Change set created: $change_set_name"
    echo "Review the change set in AWS Console before executing"
    
    rm /tmp/main-stack-updated.yaml
}

# Main execution
echo "Starting CloudFormation deployment..."

# Validate templates first
validate_templates

# Upload templates to S3
TEMPLATE_BUCKET=$(upload_templates)

# Check if this is a dry run
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry run mode: Creating change set for review"
    create_change_set $TEMPLATE_BUCKET
    echo ""
    echo "To execute the change set, run:"
    echo "aws cloudformation execute-change-set --profile $AWS_PROFILE --region $AWS_REGION --change-set-name <changeset-name> --stack-name $STACK_NAME"
else
    # Deploy main stack
    deploy_main_stack $TEMPLATE_BUCKET
    
    # Get stack outputs
    get_stack_outputs
    
    echo ""
    echo "CloudFormation deployment completed successfully!"
    echo "Stack Name: $STACK_NAME"
    echo "Region: $AWS_REGION"
    echo "Platform: $PLATFORM"
    echo ""
    echo "Next steps:"
    echo "1. Build and push Docker images using: infrastructure/scripts/build-and-push-images.sh"
    echo "2. Deploy ECS services using: infrastructure/scripts/deploy-ecs.sh"
    echo "3. Setup Milvus storage using: infrastructure/scripts/setup-milvus-storage.sh"
    echo "4. Deploy Milvus service using: infrastructure/scripts/deploy-milvus.sh"
fi