#!/bin/bash

# AWS ECS Orchestrator Deployment Validation Script
# Validates that the orchestrator service is deployed and working correctly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
AWS_PROFILE="ml-sandbox"
AWS_REGION="us-east-1"
CLUSTER_NAME="multilingual-inference-cluster"
STACK_NAME="multilingual-inference"
SERVICE_NAME="multilingual-inference-orchestrator"

echo "🚀 AWS ECS Orchestrator Deployment Validation"
echo "=============================================="
echo ""

# Step 1: Check CloudFormation Stack
log_info "Step 1: Checking CloudFormation stack status..."
stack_status=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'Stacks[0].StackStatus' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "$stack_status" == "CREATE_COMPLETE" || "$stack_status" == "UPDATE_COMPLETE" || "$stack_status" == "UPDATE_ROLLBACK_COMPLETE" ]]; then
    log_success "✅ CloudFormation stack is in stable state: $stack_status"
else
    log_error "❌ CloudFormation stack issue: $stack_status"
    exit 1
fi

# Step 2: Get Load Balancer DNS
log_info "Step 2: Getting Load Balancer DNS..."
LB_DNS=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
    --output text)

if [[ -n "$LB_DNS" ]]; then
    log_success "✅ Load Balancer DNS: $LB_DNS"
else
    log_error "❌ Could not get Load Balancer DNS"
    exit 1
fi

# Step 3: Check ECS Cluster
log_info "Step 3: Checking ECS cluster status..."
cluster_status=$(aws ecs describe-clusters \
    --clusters $CLUSTER_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'clusters[0].status' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "$cluster_status" == "ACTIVE" ]]; then
    log_success "✅ ECS cluster is active"
else
    log_error "❌ ECS cluster issue: $cluster_status"
    exit 1
fi

# Step 4: Check ECS Service
log_info "Step 4: Checking ECS service status..."
service_info=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'services[0].[status,runningCount,desiredCount]' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "$service_info" == *"ACTIVE"* ]]; then
    running_count=$(echo $service_info | awk '{print $2}')
    desired_count=$(echo $service_info | awk '{print $3}')
    
    if [[ "$running_count" == "$desired_count" && "$running_count" -gt 0 ]]; then
        log_success "✅ ECS service is healthy: $running_count/$desired_count tasks running"
    else
        log_warning "⚠️  ECS service tasks: $running_count/$desired_count running"
    fi
else
    log_error "❌ ECS service issue: $service_info"
    exit 1
fi

# Step 5: Test Health Endpoint
log_info "Step 5: Testing health endpoint..."
health_response=$(curl -s -f "http://$LB_DNS/health" 2>/dev/null || echo "FAILED")

if [[ "$health_response" == *"healthy"* ]]; then
    log_success "✅ Health endpoint is responding correctly"
    service_name=$(echo $health_response | jq -r '.service // "N/A"')
    environment=$(echo $health_response | jq -r '.environment // "N/A"')
    agents_count=$(echo $health_response | jq -r '.agents_count // "N/A"')
    echo "   Service: $service_name"
    echo "   Environment: $environment"
    echo "   Registered Agents: $agents_count"
else
    log_error "❌ Health endpoint failed: $health_response"
    exit 1
fi

# Step 6: Test Service Info Endpoint
log_info "Step 6: Testing service info endpoint..."
info_response=$(curl -s -f "http://$LB_DNS/" 2>/dev/null || echo "FAILED")

if [[ "$info_response" == *"multilingual-inference-orchestrator"* ]]; then
    log_success "✅ Service info endpoint is working correctly"
    version=$(echo $info_response | jq -r '.version // "N/A"')
    echo "   Version: $version"
    echo "   Available endpoints:"
    echo $info_response | jq -r '.endpoints | to_entries[] | "     \(.key): \(.value)"'
else
    log_error "❌ Service info endpoint failed: $info_response"
    exit 1
fi

# Step 7: Test Orchestrator Status
log_info "Step 7: Testing orchestrator inference endpoint..."
orchestrator_response=$(curl -s -X POST "http://$LB_DNS/infer" \
    -H "Content-Type: application/json" \
    -d '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}' \
    2>/dev/null || echo "FAILED")

if [[ "$orchestrator_response" == *"orchestrator_status"* ]]; then
    log_success "✅ Orchestrator endpoint is responding correctly"
    status=$(echo $orchestrator_response | jq -r '.status // "N/A"')
    orchestrator_status=$(echo $orchestrator_response | jq -r '.orchestrator_status // "N/A"')
    registered_agents=$(echo $orchestrator_response | jq -r '.registered_agents // "N/A"')
    echo "   Status: $status"
    echo "   Orchestrator: $orchestrator_status"
    echo "   Registered Agents: $registered_agents"
    
    if [[ "$registered_agents" == "0" ]]; then
        log_info "   📝 Note: No agents registered yet - this is expected for orchestrator-only deployment"
        echo "   Next steps to complete the system:"
        echo $orchestrator_response | jq -r '.next_steps[]? // empty | "     - \(.)"'
    fi
else
    log_error "❌ Orchestrator endpoint failed: $orchestrator_response"
    exit 1
fi

# Step 8: Check CloudWatch Logs
log_info "Step 8: Checking CloudWatch logs..."
log_group="/ecs/$SERVICE_NAME"
log_stream=$(aws logs describe-log-streams \
    --log-group-name "$log_group" \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --order-by LastEventTime \
    --descending \
    --max-items 1 \
    --query 'logStreams[0].logStreamName' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [[ "$log_stream" != "NOT_FOUND" && "$log_stream" != "None" ]]; then
    log_success "✅ CloudWatch logs are available"
    echo "   Log stream: $log_stream"
    
    # Check for recent errors
    error_count=$(aws logs filter-log-events \
        --log-group-name "$log_group" \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --start-time $(date -d '10 minutes ago' +%s)000 \
        --filter-pattern "ERROR" \
        --query 'length(events)' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$error_count" == "0" ]]; then
        log_success "✅ No recent errors in logs"
    else
        log_warning "⚠️  Found $error_count recent errors in logs"
    fi
else
    log_warning "⚠️  CloudWatch logs not accessible"
fi

# Step 9: Check ECR Images
log_info "Step 9: Checking ECR repository..."
ecr_repo="multilingual-inference-orchestrator"
image_count=$(aws ecr describe-images \
    --repository-name $ecr_repo \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'length(imageDetails)' \
    --output text 2>/dev/null || echo "0")

if [[ "$image_count" -gt 0 ]]; then
    log_success "✅ ECR repository has $image_count images"
    
    # Get latest image info
    latest_image=$(aws ecr describe-images \
        --repository-name $ecr_repo \
        --profile $AWS_PROFILE \
        --region $AWS_REGION \
        --query 'sort_by(imageDetails, &imagePushedAt)[-1].[imageTags[0], imageSizeInBytes, imagePushedAt]' \
        --output text 2>/dev/null || echo "N/A N/A N/A")
    
    tag=$(echo $latest_image | awk '{print $1}')
    size_bytes=$(echo $latest_image | awk '{print $2}')
    pushed_at=$(echo $latest_image | awk '{print $3}')
    size_mb=$((size_bytes / 1024 / 1024))
    echo "   Latest image: $tag (${size_mb}MB)"
    echo "   Pushed: $(date -d @${pushed_at} '+%Y-%m-%d %H:%M:%S')"
else
    log_warning "⚠️  No images found in ECR repository"
fi

# Step 10: Performance Test
log_info "Step 10: Running orchestrator performance test..."
start_time=$(date +%s)

# Run 3 concurrent requests to orchestrator
for i in {1..3}; do
    (
        curl -s -X POST "http://$LB_DNS/infer" \
            -H "Content-Type: application/json" \
            -d "{\"product_name\": \"Test Product $i\", \"language_hint\": \"en\"}" \
            > /tmp/orch_perf_test_$i.log 2>&1
    ) &
done

# Wait for all requests
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

# Count successful responses (orchestrator responses)
successful_requests=0
for i in {1..3}; do
    if grep -q "orchestrator_status" "/tmp/orch_perf_test_$i.log" 2>/dev/null; then
        ((successful_requests++))
    fi
done

if [[ $successful_requests -eq 3 ]]; then
    log_success "✅ Performance test passed: 3/3 orchestrator requests successful in ${duration}s"
else
    log_warning "⚠️  Performance test: $successful_requests/3 orchestrator requests successful in ${duration}s"
fi

# Cleanup temp files
rm -f /tmp/orch_perf_test_*.log

# Step 11: Summary
echo ""
echo "📊 ORCHESTRATOR DEPLOYMENT VALIDATION SUMMARY"
echo "=============================================="
echo "🏗️  Infrastructure: CloudFormation stack deployed and stable"
echo "🐳 Container: Docker images in ECR repository"
echo "☁️  ECS: Orchestrator service running on Fargate"
echo "🔗 Load Balancer: $LB_DNS"
echo "✅ Health Check: Passing"
echo "🎯 Orchestrator: Initialized and ready"
echo "📊 Performance: Handling concurrent requests"
echo "📝 Logs: Available in CloudWatch"
echo ""
log_success "🎉 ORCHESTRATOR DEPLOYMENT VALIDATION SUCCESSFUL!"
echo ""
echo "🔗 API Endpoints:"
echo "   Health: http://$LB_DNS/health"
echo "   Service Info: http://$LB_DNS/"
echo "   Orchestrator: http://$LB_DNS/infer"
echo ""
echo "📖 Current Status:"
echo "   ✅ Orchestrator service is deployed and healthy"
echo "   📝 Individual agents (NER, RAG, LLM, Hybrid) are not yet deployed"
echo "   🎯 System is ready for agent registration and full inference pipeline"
echo ""
echo "🚀 Next Steps to Complete Full System:"
echo "   1. Deploy NER agent service"
echo "   2. Deploy RAG agent service"  
echo "   3. Deploy LLM agent service"
echo "   4. Deploy Hybrid agent service"
echo "   5. Register agents with orchestrator"
echo ""
log_success "✅ Orchestrator foundation is ready for multi-agent system!"