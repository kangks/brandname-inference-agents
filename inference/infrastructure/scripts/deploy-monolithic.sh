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
if [ -f "./step1_deploy-cloudformation.sh" ]; then
    ./step1_deploy-cloudformation.sh
else
    echo "⚠️  CloudFormation deployment script not found, skipping..."
fi

# Step 2: Build and Push Docker Image
echo ""
echo "🐳 Step 2: Building and pushing Docker image..."
if [ -f "./step2_build-and-push-images.sh" ]; then
    ./step2_build-and-push-images.sh
else
    echo "❌ Build script not found: ./step2_build-and-push-images.sh"
    exit 1
fi

# Step 3: Deploy ECS Service
echo ""
echo "🚢 Step 3: Deploying ECS orchestrator service..."
if [ -f "./step3_deploy-ecs.sh" ]; then
    ./step3_deploy-ecs.sh
else
    echo "❌ ECS deployment script not found: ./step3_deploy-ecs.sh"
    exit 1
fi

# Step 4: Deploy Milvus (Optional)
echo ""
echo "🗄️  Step 4: Deploying Milvus vector database (optional for RAG agent)..."
read -p "Deploy Milvus vector database? (y/N): " deploy_milvus
if [[ $deploy_milvus =~ ^[Yy]$ ]]; then
    if [ -f "./step5_setup-milvus-storage.sh" ]; then
        ./step5_setup-milvus-storage.sh
    fi
    if [ -f "./step4_deploy-milvus.sh" ]; then
        ./step4_deploy-milvus.sh
    fi
else
    echo "⏭️  Skipping Milvus deployment (RAG agent will use fallback mode)"
fi

# Get deployment information
echo ""
echo "📊 Getting deployment information..."

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --names "${STACK_NAME}-alb" \
    --query 'LoadBalancers[0].DNSName' \
    --output text 2>/dev/null || echo "ALB not found")

# Get ECS service status
ECS_SERVICE_STATUS=$(aws ecs describe-services \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --cluster "${STACK_NAME}-cluster" \
    --services "${STACK_NAME}-orchestrator" \
    --query 'services[0].status' \
    --output text 2>/dev/null || echo "Service not found")

echo ""
echo "🎉 Deployment Summary"
echo "===================="
echo "✅ Infrastructure: Deployed"
echo "✅ Docker Image: Built and pushed"
echo "✅ ECS Service: $ECS_SERVICE_STATUS"
if [[ $deploy_milvus =~ ^[Yy]$ ]]; then
    echo "✅ Milvus Database: Deployed"
else
    echo "⏭️  Milvus Database: Skipped"
fi

echo ""
echo "🌐 API Endpoints"
echo "==============="
if [ "$ALB_DNS" != "ALB not found" ]; then
    echo "Main API: http://$ALB_DNS/infer"
    echo "Health Check: http://$ALB_DNS/health"
    echo "Service Info: http://$ALB_DNS/"
else
    echo "⚠️  ALB DNS not found. Check CloudFormation stack deployment."
fi

echo ""
echo "🧪 Test Your Deployment"
echo "======================"
echo "Test all inference methods:"
echo ""
if [ "$ALB_DNS" != "ALB not found" ]; then
    cat << EOF
# Test orchestrator (coordinates all agents)
curl -X POST "http://$ALB_DNS/infer" \\
  -H "Content-Type: application/json" \\
  -d '{"product_name": "Samsung Galaxy S24", "method": "orchestrator"}'

# Test individual agents
for method in simple ner rag llm hybrid; do
  echo "Testing \$method..."
  curl -X POST "http://$ALB_DNS/infer" \\
    -H "Content-Type: application/json" \\
    -d "{\"product_name\": \"Samsung Galaxy S24\", \"method\": \"\$method\"}"
done
EOF
else
    echo "⚠️  Get ALB DNS name from AWS Console and replace <ALB_DNS> in the examples above"
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
echo "1. Wait 2-3 minutes for ECS service to fully start"
echo "2. Test the health endpoint to verify deployment"
echo "3. Try different inference methods with your product data"
echo "4. Monitor CloudWatch logs for any issues"

echo ""
echo "✅ Deployment completed successfully!"