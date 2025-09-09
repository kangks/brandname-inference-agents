#!/bin/bash

# Setup EFS storage for Milvus vector database
# Uses ml-sandbox AWS profile and us-east-1 region

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

echo "Setting up EFS storage for Milvus vector database..."
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Account ID: $ACCOUNT_ID"

# Function to create EFS file system
create_efs_filesystem() {
    echo "Creating EFS file system for Milvus..."
    
    # Create EFS file system
    EFS_ID=$(aws efs create-file-system \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --creation-token "multilingual-inference-milvus-data-$(date +%s)" \
        --performance-mode generalPurpose \
        --throughput-mode bursting \
        --encrypted \
        --query 'FileSystemId' \
        --output text)
    
    # Add tags to the EFS file system
    aws efs tag-resource \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --resource-id $EFS_ID \
        --tags Key=Name,Value=multilingual-inference-milvus-storage Key=Environment,Value=production Key=Service,Value=milvus Key=Platform,Value=ARM64
    
    echo "Created EFS file system: $EFS_ID"
    
    # Wait for file system to be available
    echo "Waiting for EFS file system to be available..."
    aws efs wait file-system-available \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $EFS_ID
    
    echo "EFS file system is available: $EFS_ID"
    echo $EFS_ID
}

# Function to create mount targets
create_mount_targets() {
    local efs_id=$1
    
    echo "Creating EFS mount targets..."
    
    # Get VPC and subnets
    VPC_ID=$(aws ec2 describe-vpcs \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)
    
    # Get default security group
    SG_ID=$(aws ec2 describe-security-groups \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --group-names default \
        --query 'SecurityGroups[0].GroupId' \
        --output text)
    
    # Get subnets
    SUBNETS=$(aws ec2 describe-subnets \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[].SubnetId' \
        --output text)
    
    # Create mount targets for each subnet
    for subnet in $SUBNETS; do
        echo "Creating mount target in subnet: $subnet"
        aws efs create-mount-target \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --file-system-id $efs_id \
            --subnet-id $subnet \
            --security-groups $SG_ID || echo "Mount target already exists for subnet $subnet"
    done
}

# Function to create access points
create_access_points() {
    local efs_id=$1
    
    echo "Creating EFS access points..."
    
    # Create data access point
    DATA_AP_ID=$(aws efs create-access-point \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --posix-user Uid=1000,Gid=1000 \
        --root-directory Path=/milvus-data,CreationInfo='{OwnerUid=1000,OwnerGid=1000,Permissions=755}' \
        --tags Key=Name,Value=milvus-data-access-point Key=Purpose,Value=milvus-data-storage \
        --query 'AccessPointId' \
        --output text)
    
    echo "Created data access point: $DATA_AP_ID"
    
    # Create logs access point
    LOGS_AP_ID=$(aws efs create-access-point \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --posix-user Uid=1000,Gid=1000 \
        --root-directory Path=/milvus-logs,CreationInfo='{OwnerUid=1000,OwnerGid=1000,Permissions=755}' \
        --tags Key=Name,Value=milvus-logs-access-point Key=Purpose,Value=milvus-logs-storage \
        --query 'AccessPointId' \
        --output text)
    
    echo "Created logs access point: $LOGS_AP_ID"
    
    echo "Data Access Point ID: $DATA_AP_ID"
    echo "Logs Access Point ID: $LOGS_AP_ID"
}

# Function to setup backup policy
setup_backup_policy() {
    local efs_id=$1
    
    echo "Setting up EFS backup policy..."
    
    aws efs put-backup-policy \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --backup-policy Status=ENABLED
    
    echo "EFS backup policy enabled for: $efs_id"
}

# Main execution
echo "Starting EFS setup for Milvus..."

# Create EFS file system
EFS_ID=$(create_efs_filesystem)

# Create mount targets
create_mount_targets $EFS_ID

# Create access points
create_access_points $EFS_ID

# Setup backup policy
setup_backup_policy $EFS_ID

echo "EFS setup completed successfully!"
echo "EFS File System ID: $EFS_ID"
echo "Update your Milvus task definition with this EFS ID"
echo ""
echo "Next steps:"
echo "1. Update infrastructure/ecs/task-definitions/milvus-task-def.json with EFS ID: $EFS_ID"
echo "2. Update access point IDs in the task definition"
echo "3. Deploy Milvus service using deploy-milvus.sh"