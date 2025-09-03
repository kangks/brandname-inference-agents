#!/bin/bash

# Validate orchestrator with default agents deployment
# This script tests the orchestrator agent with automatically registered default agents

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STACK_NAME="${STACK_NAME:-multilingual-inference}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-ml-sandbox}"

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

# Get load balancer DNS
get_load_balancer_dns() {
    log_info "Getting load balancer DNS..."
    
    LOAD_BALANCER_DNS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
        --output text)
    
    if [ -z "$LOAD_BALANCER_DNS" ] || [ "$LOAD_BALANCER_DNS" = "None" ]; then
        log_error "Could not retrieve load balancer DNS"
        exit 1
    fi
    
    log_success "Load balancer DNS: $LOAD_BALANCER_DNS"
}

# Test health endpoint
test_health_endpoint() {
    log_info "Testing health endpoint..."
    
    HEALTH_URL="http://$LOAD_BALANCER_DNS/health"
    
    # Test with retries
    for i in {1..5}; do
        if curl -f -s "$HEALTH_URL" > /tmp/health_response.json; then
            log_success "Health endpoint is responding"
            
            # Parse health response
            STATUS=$(jq -r '.status' /tmp/health_response.json 2>/dev/null || echo "unknown")
            ORCHESTRATOR_STATUS=$(jq -r '.orchestrator // "not_available"' /tmp/health_response.json 2>/dev/null || echo "unknown")
            AGENTS_COUNT=$(jq -r '.agents_count // 0' /tmp/health_response.json 2>/dev/null || echo "0")
            
            log_info "Health Status: $STATUS"
            log_info "Orchestrator Status: $ORCHESTRATOR_STATUS"
            log_info "Agents Count: $AGENTS_COUNT"
            
            if [ "$STATUS" = "healthy" ]; then
                log_success "Service is healthy"
                return 0
            else
                log_warning "Service status: $STATUS"
            fi
            
            break
        else
            log_warning "Health check attempt $i failed, retrying in 10 seconds..."
            sleep 10
        fi
    done
    
    log_error "Health endpoint is not responding after 5 attempts"
    return 1
}

# Test inference endpoint with sample data
test_inference_endpoint() {
    log_info "Testing inference endpoint..."
    
    INFERENCE_URL="http://$LOAD_BALANCER_DNS/infer"
    
    # Test cases
    declare -a test_cases=(
        '{"product_name": "Samsung Galaxy S23", "language_hint": "en"}'
        '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}'
        '{"product_name": "Sony WH-1000XM4 หูฟัง", "language_hint": "mixed"}'
        '{"product_name": "ยาสีฟัน Colgate Total", "language_hint": "mixed"}'
        '{"product_name": "Nintendo Switch OLED", "language_hint": "en"}'
    )
    
    success_count=0
    total_tests=${#test_cases[@]}
    
    for i in "${!test_cases[@]}"; do
        test_case="${test_cases[$i]}"
        log_info "Running test case $((i+1))/$total_tests..."
        
        # Extract product name for logging
        product_name=$(echo "$test_case" | jq -r '.product_name')
        log_info "Testing: $product_name"
        
        # Make inference request
        if curl -f -s -X POST \
            -H "Content-Type: application/json" \
            -d "$test_case" \
            "$INFERENCE_URL" > "/tmp/inference_response_$i.json"; then
            
            # Parse response
            status=$(jq -r '.status // "unknown"' "/tmp/inference_response_$i.json" 2>/dev/null || echo "unknown")
            
            if [ "$status" = "completed" ]; then
                # Extract inference results
                best_prediction=$(jq -r '.inference_result.best_prediction // "unknown"' "/tmp/inference_response_$i.json" 2>/dev/null || echo "unknown")
                best_confidence=$(jq -r '.inference_result.best_confidence // 0' "/tmp/inference_response_$i.json" 2>/dev/null || echo "0")
                best_method=$(jq -r '.inference_result.best_method // "unknown"' "/tmp/inference_response_$i.json" 2>/dev/null || echo "unknown")
                processing_time=$(jq -r '.inference_result.processing_time // 0' "/tmp/inference_response_$i.json" 2>/dev/null || echo "0")
                
                log_success "✓ Prediction: $best_prediction (confidence: $best_confidence, method: $best_method, time: ${processing_time}s)"
                ((success_count++))
                
            elif [ "$status" = "ready" ]; then
                # Service is ready but no agents available
                registered_agents=$(jq -r '.registered_agents // 0' "/tmp/inference_response_$i.json" 2>/dev/null || echo "0")
                available_agents=$(jq -r '.available_agents // []' "/tmp/inference_response_$i.json" 2>/dev/null || echo "[]")
                
                log_warning "⚠ Service ready but no inference capability (agents: $registered_agents, available: $available_agents)"
                
            else
                log_error "✗ Inference failed with status: $status"
            fi
        else
            log_error "✗ Failed to make inference request"
        fi
        
        echo ""
    done
    
    log_info "Inference tests completed: $success_count/$total_tests successful"
    
    if [ $success_count -gt 0 ]; then
        log_success "At least some inference tests passed"
        return 0
    else
        log_error "All inference tests failed"
        return 1
    fi
}

# Test agent registration
test_agent_registration() {
    log_info "Testing agent registration..."
    
    # Make a simple inference request to check agent status
    INFERENCE_URL="http://$LOAD_BALANCER_DNS/infer"
    
    test_payload='{"product_name": "Test Product", "language_hint": "en"}'
    
    if curl -f -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "$INFERENCE_URL" > /tmp/agent_test_response.json; then
        
        # Check response for agent information
        status=$(jq -r '.status // "unknown"' /tmp/agent_test_response.json 2>/dev/null || echo "unknown")
        registered_agents=$(jq -r '.registered_agents // 0' /tmp/agent_test_response.json 2>/dev/null || echo "0")
        available_agents=$(jq -r '.available_agents // []' /tmp/agent_test_response.json 2>/dev/null || echo "[]")
        
        log_info "Status: $status"
        log_info "Registered Agents: $registered_agents"
        log_info "Available Agents: $available_agents"
        
        if [ "$registered_agents" -gt 0 ]; then
            log_success "Agents are registered and available"
            
            # Parse available agents
            if [ "$available_agents" != "[]" ] && [ "$available_agents" != "null" ]; then
                agent_list=$(echo "$available_agents" | jq -r '.[]' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
                log_info "Available agent types: $agent_list"
            fi
            
            return 0
        else
            log_warning "No agents are currently registered"
            
            # Check for next steps or error messages
            next_steps=$(jq -r '.next_steps // []' /tmp/agent_test_response.json 2>/dev/null || echo "[]")
            if [ "$next_steps" != "[]" ] && [ "$next_steps" != "null" ]; then
                log_info "Suggested next steps:"
                echo "$next_steps" | jq -r '.[]' 2>/dev/null | while read -r step; do
                    log_info "  - $step"
                done
            fi
            
            return 1
        fi
    else
        log_error "Failed to test agent registration"
        return 1
    fi
}

# Generate validation report
generate_report() {
    log_info "Generating validation report..."
    
    REPORT_FILE="$PROJECT_ROOT/ORCHESTRATOR_VALIDATION_REPORT.md"
    
    cat > "$REPORT_FILE" << EOF
# Orchestrator with Default Agents Validation Report

**Generated:** $(date)
**Stack:** $STACK_NAME
**Region:** $AWS_REGION
**Load Balancer:** $LOAD_BALANCER_DNS

## Validation Results

### Health Check
- **Endpoint:** http://$LOAD_BALANCER_DNS/health
- **Status:** $([ $health_test_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

### Agent Registration
- **Status:** $([ $agent_test_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

### Inference Tests
- **Status:** $([ $inference_test_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

## Test Details

### Sample Inference Results
EOF

    # Add inference results if available
    if [ -f "/tmp/inference_response_0.json" ]; then
        echo "" >> "$REPORT_FILE"
        echo "#### Samsung Galaxy S23" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "/tmp/inference_response_0.json" | jq '.' >> "$REPORT_FILE" 2>/dev/null || echo "Invalid JSON response" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## Recommendations

$(if [ $health_test_result -eq 0 ] && [ $agent_test_result -eq 0 ] && [ $inference_test_result -eq 0 ]; then
    echo "✅ **All tests passed!** The orchestrator with default agents is working correctly."
else
    echo "⚠️ **Some tests failed.** Please check the following:"
    [ $health_test_result -ne 0 ] && echo "- Health endpoint is not responding properly"
    [ $agent_test_result -ne 0 ] && echo "- Default agents are not registered correctly"
    [ $inference_test_result -ne 0 ] && echo "- Inference functionality is not working"
fi)

## Next Steps

1. Monitor the service using CloudWatch dashboards
2. Test with additional product names and languages
3. Verify agent performance and accuracy
4. Set up automated monitoring and alerting

---
*Report generated by validate-orchestrator-with-agents.sh*
EOF

    log_success "Validation report generated: $REPORT_FILE"
}

# Main validation function
main() {
    log_info "Starting orchestrator with default agents validation..."
    log_info "Stack: $STACK_NAME"
    log_info "Region: $AWS_REGION"
    log_info "Profile: $AWS_PROFILE"
    
    get_load_balancer_dns
    
    # Run tests
    health_test_result=1
    agent_test_result=1
    inference_test_result=1
    
    if test_health_endpoint; then
        health_test_result=0
    fi
    
    if test_agent_registration; then
        agent_test_result=0
    fi
    
    if test_inference_endpoint; then
        inference_test_result=0
    fi
    
    # Generate report
    generate_report
    
    # Summary
    echo ""
    log_info "=== VALIDATION SUMMARY ==="
    [ $health_test_result -eq 0 ] && log_success "✅ Health Check: PASSED" || log_error "❌ Health Check: FAILED"
    [ $agent_test_result -eq 0 ] && log_success "✅ Agent Registration: PASSED" || log_error "❌ Agent Registration: FAILED"
    [ $inference_test_result -eq 0 ] && log_success "✅ Inference Tests: PASSED" || log_error "❌ Inference Tests: FAILED"
    
    # Overall result
    if [ $health_test_result -eq 0 ] && [ $agent_test_result -eq 0 ] && [ $inference_test_result -eq 0 ]; then
        log_success "🎉 All validation tests passed!"
        exit 0
    else
        log_error "❌ Some validation tests failed. Check the report for details."
        exit 1
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/health_response.json
    rm -f /tmp/inference_response_*.json
    rm -f /tmp/agent_test_response.json
}

# Set up cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"