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

# Function to get or create EFS file system
get_or_create_efs_filesystem() {
    echo "Checking for existing EFS file system for Milvus..." >&2
    
    # Check if EFS already exists
    local existing_efs=$(aws efs describe-file-systems \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'FileSystems[?Tags[?Key==`Name` && Value==`multilingual-inference-milvus-storage`] && LifeCycleState==`available`].FileSystemId' \
        --output text | awk '{print $1}')
    
    if [ ! -z "$existing_efs" ] && [ "$existing_efs" != "None" ]; then
        echo "Found existing EFS file system: $existing_efs" >&2
        echo $existing_efs
        return
    fi
    
    echo "Creating new EFS file system for Milvus..." >&2
    
    # Create EFS file system
    local efs_id=$(aws efs create-file-system \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --creation-token "multilingual-inference-milvus-data-$(date +%s)" \
        --performance-mode generalPurpose \
        --throughput-mode bursting \
        --encrypted \
        --query 'FileSystemId' \
        --output text)
    
    if [ -z "$efs_id" ] || [ "$efs_id" == "None" ]; then
        echo "Error: Failed to create EFS file system" >&2
        exit 1
    fi
    
    echo "Created EFS file system: $efs_id" >&2
    
    # Wait for file system to be available
    echo "Waiting for EFS file system to be available..." >&2
    aws efs wait file-system-available \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id
    
    # Add tags to the EFS file system
    aws efs tag-resource \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --resource-id $efs_id \
        --tags Key=Name,Value=multilingual-inference-milvus-storage Key=Environment,Value=production Key=Service,Value=milvus Key=Platform,Value=ARM64
    
    echo "EFS file system is available: $efs_id" >&2
    echo $efs_id
}

# Function to create mount targets
create_mount_targets() {
    local efs_id=$1
    
    if [ -z "$efs_id" ] || [ "$efs_id" == "None" ]; then
        echo "Error: Invalid EFS ID provided to create_mount_targets"
        exit 1
    fi
    
    echo "Creating EFS mount targets for file system: $efs_id"
    
    # Check existing mount targets
    EXISTING_TARGETS=$(aws efs describe-mount-targets \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --query 'MountTargets[].SubnetId' \
        --output text)
    
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
        if echo "$EXISTING_TARGETS" | grep -q "$subnet"; then
            echo "Mount target already exists for subnet: $subnet"
        else
            echo "Creating mount target in subnet: $subnet"
            aws efs create-mount-target \
                --profile $AWS_PROFILE \
                --region $AWS_REGION \
                --file-system-id $efs_id \
                --subnet-id $subnet \
                --security-groups $SG_ID || echo "Failed to create mount target for subnet $subnet"
        fi
    done
}

# Function to create access points
create_access_points() {
    local efs_id=$1
    
    if [ -z "$efs_id" ] || [ "$efs_id" == "None" ]; then
        echo "Error: Invalid EFS ID provided to create_access_points"
        exit 1
    fi
    
    echo "Creating EFS access points for file system: $efs_id"
    
    # Check for existing access points
    EXISTING_DATA_AP=$(aws efs describe-access-points \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --query 'AccessPoints[?Tags[?Key==`Name` && Value==`milvus-data-access-point`]].AccessPointId' \
        --output text)
    
    EXISTING_LOGS_AP=$(aws efs describe-access-points \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --query 'AccessPoints[?Tags[?Key==`Name` && Value==`milvus-logs-access-point`]].AccessPointId' \
        --output text)
    
    # Create data access point if it doesn't exist
    if [ -z "$EXISTING_DATA_AP" ] || [ "$EXISTING_DATA_AP" == "None" ]; then
        echo "Creating data access point..."
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
    else
        DATA_AP_ID=$EXISTING_DATA_AP
        echo "Using existing data access point: $DATA_AP_ID"
    fi
    
    # Create logs access point if it doesn't exist
    if [ -z "$EXISTING_LOGS_AP" ] || [ "$EXISTING_LOGS_AP" == "None" ]; then
        echo "Creating logs access point..."
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
    else
        LOGS_AP_ID=$EXISTING_LOGS_AP
        echo "Using existing logs access point: $LOGS_AP_ID"
    fi
    
    echo "Data Access Point ID: $DATA_AP_ID"
    echo "Logs Access Point ID: $LOGS_AP_ID"
}

# Function to setup backup policy
setup_backup_policy() {
    local efs_id=$1
    
    if [ -z "$efs_id" ] || [ "$efs_id" == "None" ]; then
        echo "Error: Invalid EFS ID provided to setup_backup_policy"
        exit 1
    fi
    
    echo "Setting up EFS backup policy for file system: $efs_id"
    
    aws efs put-backup-policy \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --file-system-id $efs_id \
        --backup-policy Status=ENABLED
    
    echo "EFS backup policy enabled for: $efs_id"
}

# Main execution
echo "Starting EFS setup for Milvus..."

# Get or create EFS file system
EFS_ID=$(get_or_create_efs_filesystem)

if [ -z "$EFS_ID" ] || [ "$EFS_ID" == "None" ]; then
    echo "Error: Could not determine EFS file system ID"
    exit 1
fi

echo "Using EFS file system: $EFS_ID"

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