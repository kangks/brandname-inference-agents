#!/bin/bash

# Deploy simplified CloudFormation stack for multilingual inference system
# Uses ml-sandbox AWS profile and us-east-1 region

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
STACK_NAME="multilingual-inference"
ENVIRONMENT="production"
PLATFORM="ARM64"

echo "Deploying simplified CloudFormation infrastructure..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Stack Name: $STACK_NAME"

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
        --template-body file://infrastructure/cloudformation/simple-main-stack.yaml \
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
        --template-body file://infrastructure/cloudformation/simple-main-stack.yaml \
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

echo "Getting stack outputs..."
aws cloudformation describe-stacks \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs' \
    --output table

echo "CloudFormation deployment completed successfully!"