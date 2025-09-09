#!/bin/bash

# VPC Diagnostic script for ECS deployment troubleshooting

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"

echo "🔍 VPC Diagnostic Report"
echo "======================="
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo ""

# Function to check AWS credentials
check_credentials() {
    echo "🔐 Checking AWS Credentials..."
    
    if ! ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text 2>/dev/null); then
        echo "❌ Failed to get AWS credentials with profile: $AWS_PROFILE"
        echo "💡 Try: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    echo "✅ AWS Account ID: $ACCOUNT_ID"
    echo ""
}

# Function to check VPCs
check_vpcs() {
    echo "🌐 Checking VPCs..."
    
    # List all VPCs
    VPCS=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'Vpcs[].[VpcId,IsDefault,State,CidrBlock]' \
        --output table)
    
    echo "$VPCS"
    echo ""
    
    # Check for default VPC
    DEFAULT_VPC=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$DEFAULT_VPC" = "None" ] || [ "$DEFAULT_VPC" = "null" ]; then
        echo "⚠️  No default VPC found"
        echo "💡 You can create one with: aws ec2 create-default-vpc --profile $AWS_PROFILE --region $AWS_REGION"
    else
        echo "✅ Default VPC: $DEFAULT_VPC"
    fi
    echo ""
}

# Function to check subnets
check_subnets() {
    echo "🏠 Checking Subnets..."
    
    # Get default VPC ID
    DEFAULT_VPC=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$DEFAULT_VPC" != "None" ] && [ "$DEFAULT_VPC" != "null" ]; then
        echo "📍 Subnets in default VPC ($DEFAULT_VPC):"
        
        SUBNETS=$(aws ec2 describe-subnets \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
            --query 'Subnets[].[SubnetId,AvailabilityZone,State,CidrBlock,DefaultForAz]' \
            --output table)
        
        echo "$SUBNETS"
        
        # Count subnets
        SUBNET_COUNT=$(aws ec2 describe-subnets \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
            --query 'length(Subnets)' \
            --output text)
        
        echo ""
        echo "📊 Total subnets: $SUBNET_COUNT"
        
        if [ "$SUBNET_COUNT" -lt 2 ]; then
            echo "⚠️  ECS Fargate requires at least 2 subnets in different AZs"
            echo "💡 Consider creating additional subnets or using a different VPC"
        else
            echo "✅ Sufficient subnets for ECS Fargate"
        fi
    else
        echo "❌ No default VPC found - cannot check subnets"
    fi
    echo ""
}

# Function to check security groups
check_security_groups() {
    echo "🛡️  Checking Security Groups..."
    
    DEFAULT_VPC=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$DEFAULT_VPC" != "None" ] && [ "$DEFAULT_VPC" != "null" ]; then
        echo "🔒 Security groups in default VPC ($DEFAULT_VPC):"
        
        SECURITY_GROUPS=$(aws ec2 describe-security-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
            --query 'SecurityGroups[].[GroupId,GroupName,Description]' \
            --output table)
        
        echo "$SECURITY_GROUPS"
        
        # Check default security group
        DEFAULT_SG=$(aws ec2 describe-security-groups \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=group-name,Values=default" \
            --query 'SecurityGroups[0].GroupId' \
            --output text 2>/dev/null || echo "None")
        
        if [ "$DEFAULT_SG" != "None" ] && [ "$DEFAULT_SG" != "null" ]; then
            echo "✅ Default security group: $DEFAULT_SG"
        else
            echo "❌ No default security group found"
        fi
    else
        echo "❌ No default VPC found - cannot check security groups"
    fi
    echo ""
}

# Function to check ECS prerequisites
check_ecs_prerequisites() {
    echo "🐳 Checking ECS Prerequisites..."
    
    # Check if ECS cluster exists
    CLUSTER_NAME="multilingual-inference-cluster"
    
    if aws ecs describe-clusters \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --clusters $CLUSTER_NAME \
        --query 'clusters[0].clusterName' \
        --output text 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        echo "✅ ECS Cluster exists: $CLUSTER_NAME"
        
        # Get cluster status
        CLUSTER_STATUS=$(aws ecs describe-clusters \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --clusters $CLUSTER_NAME \
            --query 'clusters[0].status' \
            --output text)
        
        echo "📊 Cluster status: $CLUSTER_STATUS"
    else
        echo "ℹ️  ECS Cluster does not exist: $CLUSTER_NAME"
        echo "💡 Will be created during deployment"
    fi
    echo ""
}

# Function to provide recommendations
provide_recommendations() {
    echo "💡 Recommendations"
    echo "=================="
    
    DEFAULT_VPC=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$DEFAULT_VPC" = "None" ] || [ "$DEFAULT_VPC" = "null" ]; then
        echo "🔧 Create default VPC:"
        echo "   aws ec2 create-default-vpc --profile $AWS_PROFILE --region $AWS_REGION"
        echo ""
    fi
    
    echo "🚀 Deploy ECS infrastructure:"
    echo "   ./infrastructure/scripts/deploy-ecs-simple.sh"
    echo ""
    
    echo "🔍 Monitor deployment:"
    echo "   aws ecs describe-services --profile $AWS_PROFILE --region $AWS_REGION --cluster multilingual-inference-cluster --services multilingual-inference-orchestrator"
    echo ""
    
    echo "📋 Check task definitions:"
    echo "   aws ecs list-task-definitions --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
}

# Main diagnostic function
main() {
    check_credentials
    check_vpcs
    check_subnets
    check_security_groups
    check_ecs_prerequisites
    provide_recommendations
    
    echo "🎉 VPC diagnostic completed!"
}

# Run main function
main "$@"