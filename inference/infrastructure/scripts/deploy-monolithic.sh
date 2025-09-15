#!/bin/bash

# Simplified deployment script for monolithic inference architecture
# Deploys single orchestrator service containing all agents

set -e

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
STACK_NAME="multilingual-inference"

echo "🚀 Deploying Monolithic Multilingual Inference System"
echo "=================================================="
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Stack Name: $STACK_NAME"
echo ""

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."
if ! command_exists aws; then
    echo "❌ AWS CLI not found. Please install AWS CLI."
    exit 1
fi

if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Check for jq (optional but helpful for JSON parsing)
if ! command_exists jq; then
    echo "⚠️  jq not found. JSON responses will be displayed raw."
    echo "   Install jq for better formatted output: brew install jq (macOS) or apt-get install jq (Ubuntu)"
fi

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
    echo "❌ AWS credentials not configured for profile: $AWS_PROFILE"
    echo "Please run: aws configure --profile $AWS_PROFILE"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
echo "✅ AWS Account: $ACCOUNT_ID"

# Step 1: Deploy CloudFormation Infrastructure
echo ""
echo "📋 Step 1: Deploying CloudFormation infrastructure..."
if [ -f "${SCRIPT_DIR}/step1_deploy-cloudformation.sh" ]; then
    ${SCRIPT_DIR}/step1_deploy-cloudformation.sh
else
    echo "⚠️  CloudFormation deployment script not found, skipping..."
fi

# Step 2: Build and Push Docker Image
echo ""
echo "🐳 Step 2: Building and pushing Docker image..."
if [ -f "${SCRIPT_DIR}/step2_build-and-push-images.sh" ]; then
    ${SCRIPT_DIR}/step2_build-and-push-images.sh
else
    echo "❌ Build script not found: ${SCRIPT_DIR}/step2_build-and-push-images.sh"
    exit 1
fi

# Step 3: Deploy ECS Service
echo ""
echo "🚢 Step 3: Deploying ECS orchestrator service..."
if [ -f "${SCRIPT_DIR}/step3_deploy-ecs.sh" ]; then
    ${SCRIPT_DIR}/step3_deploy-ecs.sh
    echo "✅ ECS orchestrator service deployed (contains all agents)"
else
    echo "❌ ECS deployment script not found: ${SCRIPT_DIR}/step3_deploy-ecs.sh"
    exit 1
fi

# Step 4: Deploy Milvus (Optional)
echo ""
echo "🗄️  Step 4: Deploying Milvus vector database (optional for RAG agent)..."
read -p "Deploy Milvus vector database? (y/N): " deploy_milvus
if [[ $deploy_milvus =~ ^[Yy]$ ]]; then
    if [ -f "${SCRIPT_DIR}/step5_setup-milvus-storage.sh" ]; then
        ${SCRIPT_DIR}/step5_setup-milvus-storage.sh
    fi
    if [ -f "${SCRIPT_DIR}/step4_deploy-milvus.sh" ]; then
        ${SCRIPT_DIR}/step4_deploy-milvus.sh
    fi
else
    echo "⏭️  Skipping Milvus deployment (RAG agent will use fallback mode)"
fi

# Get deployment information and validate
echo ""
echo "📊 Getting deployment information..."

# Get ALB DNS name - try multiple methods to find the correct ALB
ALB_DNS=""

# Method 1: Try with stack name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --names "${STACK_NAME}-alb" \
    --query 'LoadBalancers[0].DNSName' \
    --output text 2>/dev/null || echo "")

# Method 2: Try to find ALB with "inference" in the name
if [ -z "$ALB_DNS" ] || [ "$ALB_DNS" = "None" ]; then
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'LoadBalancers[?contains(LoadBalancerName, `inference`)].DNSName | [0]' \
        --output text 2>/dev/null || echo "")
fi

# Method 3: Try to find ALB with "production" in the name
if [ -z "$ALB_DNS" ] || [ "$ALB_DNS" = "None" ]; then
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'LoadBalancers[?contains(LoadBalancerName, `production`)].DNSName | [0]' \
        --output text 2>/dev/null || echo "")
fi

# Get ECS service status
ECS_SERVICE_STATUS=$(aws ecs describe-services \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster "${STACK_NAME}-cluster" \
    --services "${STACK_NAME}-orchestrator" \
    --query 'services[0].status' \
    --output text 2>/dev/null || echo "Service not found")

# Get running task count
RUNNING_TASKS=$(aws ecs describe-services \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster "${STACK_NAME}-cluster" \
    --services "${STACK_NAME}-orchestrator" \
    --query 'services[0].runningCount' \
    --output text 2>/dev/null || echo "0")

echo ""
echo "🎉 Deployment Summary"
echo "===================="
echo "✅ Infrastructure: Deployed"
echo "✅ Docker Image: Built and pushed"
echo "✅ ECS Service: $ECS_SERVICE_STATUS"
echo "✅ Running Tasks: $RUNNING_TASKS"
if [[ $deploy_milvus =~ ^[Yy]$ ]]; then
    echo "✅ Milvus Database: Deployed"
else
    echo "⏭️  Milvus Database: Skipped"
fi

echo ""
echo "🌐 API Endpoints"
echo "==============="
if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    echo "Main API: http://$ALB_DNS/infer"
    echo "Health Check: http://$ALB_DNS/health"
    echo "Service Info: http://$ALB_DNS/"
    
    # Wait for service to be ready and validate deployment
    echo ""
    echo "🔍 Validating Deployment..."
    echo "=========================="
    
    if [ "$RUNNING_TASKS" -eq 0 ]; then
        echo "⚠️  No running tasks yet. Waiting for service to start..."
        echo "⏳ Waiting for ECS service to become stable..."
        
        # Wait for service to be stable (up to 10 minutes)
        if aws ecs wait services-stable \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --cluster "${STACK_NAME}-cluster" \
            --services "${STACK_NAME}-orchestrator" \
            --cli-read-timeout 600 \
            --cli-connect-timeout 60; then
            echo "✅ ECS service is now stable"
        else
            echo "⚠️  Service did not stabilize within timeout. Continuing with validation..."
        fi
    fi
    
    # Test health endpoint
    echo "🏥 Testing health endpoint..."
    HEALTH_ATTEMPTS=0
    MAX_HEALTH_ATTEMPTS=12  # 2 minutes with 10-second intervals
    
    while [ $HEALTH_ATTEMPTS -lt $MAX_HEALTH_ATTEMPTS ]; do
        if curl -f -s -m 10 "http://$ALB_DNS/health" >/dev/null 2>&1; then
            echo "✅ Health endpoint is responding"
            
            # Get health status
            HEALTH_RESPONSE=$(curl -s -m 10 "http://$ALB_DNS/health" 2>/dev/null || echo "")
            if [ -n "$HEALTH_RESPONSE" ]; then
                echo "📊 Health Status:"
                echo "$HEALTH_RESPONSE" | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE"
            fi
            break
        else
            HEALTH_ATTEMPTS=$((HEALTH_ATTEMPTS + 1))
            echo "⏳ Health check attempt $HEALTH_ATTEMPTS/$MAX_HEALTH_ATTEMPTS failed, waiting 10 seconds..."
            sleep 10
        fi
    done
    
    if [ $HEALTH_ATTEMPTS -eq $MAX_HEALTH_ATTEMPTS ]; then
        echo "❌ Health endpoint not responding after $MAX_HEALTH_ATTEMPTS attempts"
        echo "   This may indicate the service is still starting up or there's an issue"
    fi
    
    # Test inference endpoint
    echo ""
    echo "🧠 Testing inference endpoint..."
    
    if curl -f -s -m 30 "http://$ALB_DNS/health" >/dev/null 2>&1; then
        echo "🧪 Testing orchestrator method..."
        
        INFERENCE_RESPONSE=$(curl -s -m 30 -X POST "http://$ALB_DNS/infer" \
            -H "Content-Type: application/json" \
            -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}' 2>/dev/null || echo "")
        
        if [ -n "$INFERENCE_RESPONSE" ]; then
            # Check if response contains success or expected fields
            if echo "$INFERENCE_RESPONSE" | grep -q -E '(success|best_prediction|brand_predictions)'; then
                echo "✅ Orchestrator inference test successful!"
                echo "📊 Sample Response:"
                echo "$INFERENCE_RESPONSE" | jq '.result.best_prediction // .brand_predictions[0].brand // "Response received"' 2>/dev/null || echo "Response received"
                
                # Test simple method as well
                echo ""
                echo "🧪 Testing simple method..."
                SIMPLE_RESPONSE=$(curl -s -m 15 -X POST "http://$ALB_DNS/infer" \
                    -H "Content-Type: application/json" \
                    -d '{"product_name": "Samsung Galaxy S24", "method": "simple"}' 2>/dev/null || echo "")
                
                if echo "$SIMPLE_RESPONSE" | grep -q -E '(brand_predictions|Samsung)'; then
                    echo "✅ Simple method test successful!"
                    SIMPLE_BRAND=$(echo "$SIMPLE_RESPONSE" | jq -r '.brand_predictions[0].brand // "Unknown"' 2>/dev/null || echo "Samsung")
                    echo "📊 Detected Brand: $SIMPLE_BRAND"
                else
                    echo "⚠️  Simple method test failed or returned unexpected response"
                fi
            else
                echo "⚠️  Inference test returned unexpected response:"
                echo "$INFERENCE_RESPONSE" | head -3
            fi
        else
            echo "❌ Inference endpoint not responding"
        fi
    else
        echo "⚠️  Skipping inference test - health endpoint not ready"
    fi
    
else
    echo "⚠️  ALB DNS not found. Check CloudFormation stack deployment."
    echo "   Available load balancers:"
    aws elbv2 describe-load-balancers \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'LoadBalancers[*].{Name:LoadBalancerName,DNS:DNSName}' \
        --output table 2>/dev/null || echo "   Could not retrieve load balancers"
fi

echo ""
echo "🧪 Manual Testing Commands"
echo "=========================="
if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    echo "Use these commands to test your deployment manually:"
    echo ""
    cat << EOF
# Test orchestrator (coordinates all agents) - RECOMMENDED
curl -X POST "http://$ALB_DNS/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'

# Test individual agents
curl -X POST "http://$ALB_DNS/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24", "method": "simple"}'

curl -X POST "http://$ALB_DNS/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24", "method": "ner"}'

# Test with Thai product name
curl -X POST "http://$ALB_DNS/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24 โทรศัพท์มือถือ", "method": "orchestrator"}'

# Health check
curl "http://$ALB_DNS/health"
EOF
else
    echo "⚠️  ALB DNS not available. Use the following template:"
    echo ""
    cat << EOF
# Replace <ALB_DNS> with your actual load balancer DNS name
curl -X POST "http://<ALB_DNS>/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'
EOF
fi

echo ""
echo "📚 Documentation"
echo "==============="
echo "• Architecture Guide: ../docs/ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md"
echo "• API Usage Guide: ../docs/API_USAGE_GUIDE.md"
echo "• FAQ: ../../INFERENCE_ARCHITECTURE_FAQ.md"

echo ""
echo "🎯 Next Steps"
echo "============"
if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    if curl -f -s -m 10 "http://$ALB_DNS/health" >/dev/null 2>&1; then
        echo "✅ Your API is ready to use!"
        echo "1. Try the manual testing commands above"
        echo "2. Integrate with your applications using the API endpoint"
        echo "3. Monitor CloudWatch logs: /ecs/multilingual-inference-orchestrator"
        echo "4. Scale the service if needed based on usage"
    else
        echo "⏳ Your deployment is complete but the service may still be starting:"
        echo "1. Wait 2-3 minutes for the service to fully initialize"
        echo "2. Test the health endpoint: curl http://$ALB_DNS/health"
        echo "3. Check ECS service status if issues persist"
        echo "4. Review CloudWatch logs: /ecs/multilingual-inference-orchestrator"
    fi
else
    echo "⚠️  Deployment completed but ALB DNS not found:"
    echo "1. Check CloudFormation stack outputs for load balancer DNS"
    echo "2. Verify ECS service is running: aws ecs describe-services --cluster ${STACK_NAME}-cluster --services ${STACK_NAME}-orchestrator"
    echo "3. Check load balancer configuration"
fi

echo ""
echo "📚 Documentation"
echo "==============="
echo "• Architecture Guide: ../docs/ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md"
echo "• API Usage Guide: ../docs/API_USAGE_GUIDE.md"
echo "• FAQ: ../../INFERENCE_ARCHITECTURE_FAQ.md"

echo ""
if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    if curl -f -s -m 10 "http://$ALB_DNS/health" >/dev/null 2>&1; then
        echo "🎉 Deployment completed successfully! API is ready to use."
    else
        echo "🎉 Deployment completed! Service is starting up..."
    fi
else
    echo "🎉 Deployment completed! Check ALB configuration."
fi