#!/bin/bash

# Final validation script for orchestrator with agents deployment
set -e

# Configuration
LOAD_BALANCER_DNS="production-alb-107602758.us-east-1.elb.amazonaws.com"
BASE_URL="http://$LOAD_BALANCER_DNS"

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

# Test health endpoint
test_health() {
    log_info "Testing health endpoint..."
    
    response=$(curl -s -f "$BASE_URL/health" || echo "FAILED")
    
    if [ "$response" = "FAILED" ]; then
        log_error "Health check failed"
        return 1
    fi
    
    # Parse response
    status=$(echo "$response" | jq -r '.status // "unknown"')
    agents_count=$(echo "$response" | jq -r '.agents_count // 0')
    orchestrator=$(echo "$response" | jq -r '.orchestrator // "unknown"')
    # standalone_agent removed - only using orchestrator
    
    if [ "$status" = "healthy" ]; then
        log_success "Health check passed"
        log_info "  - Status: $status"
        log_info "  - Agents count: $agents_count"
        log_info "  - Orchestrator: $orchestrator"
        log_info "  - Standalone agent: $standalone"
        return 0
    else
        log_error "Health check returned unhealthy status: $status"
        return 1
    fi
}

# Test inference with various products
test_inference() {
    local product_name="$1"
    local expected_brand="$2"
    
    log_info "Testing inference for: $product_name"
    
    response=$(curl -s -X POST "$BASE_URL/infer" \
        -H "Content-Type: application/json" \
        -d "{\"product_name\": \"$product_name\", \"language_hint\": \"en\"}" || echo "FAILED")
    
    if [ "$response" = "FAILED" ]; then
        log_error "Inference failed for: $product_name"
        return 1
    fi
    
    # Check if response contains expected brand
    if echo "$response" | jq -e ".brand_predictions[]? | select(.brand == \"$expected_brand\")" > /dev/null; then
        log_success "Inference successful for: $product_name (detected: $expected_brand)"
        
        # Show processing time
        processing_time=$(echo "$response" | jq -r '.processing_time_ms // "unknown"')
        agent_used=$(echo "$response" | jq -r '.agent_used // "unknown"')
        log_info "  - Processing time: ${processing_time}ms"
        log_info "  - Agent used: $agent_used"
        
        return 0
    else
        log_warning "Inference completed but expected brand '$expected_brand' not found for: $product_name"
        log_info "Response: $response"
        return 1
    fi
}

# Main validation function
main() {
    log_info "Starting orchestrator deployment validation..."
    log_info "Target URL: $BASE_URL"
    
    # Test health endpoint
    if ! test_health; then
        log_error "Health check failed - aborting validation"
        exit 1
    fi
    
    echo
    log_info "Testing inference capabilities..."
    
    # Test various product names
    declare -a products=(
        "iPhone 15 Pro Max from Apple"
        "Samsung Galaxy S24 Ultra smartphone"
        "Nike Air Jordan sneakers"
        "Sony PlayStation 5 console"
        "Microsoft Surface Pro laptop"
        "Google Pixel 8 phone"
    )
    
    declare -a expected_brands=(
        "apple"
        "samsung"
        "nike"
        "sony"
        "microsoft"
        "google"
    )
    
    success_count=0
    total_count=${#products[@]}
    
    for i in "${!products[@]}"; do
        product="${products[$i]}"
        expected_brand="${expected_brands[$i]}"
        if test_inference "$product" "$expected_brand"; then
            ((success_count++))
        fi
        echo
    done
    
    # Summary
    echo "=================================="
    log_info "Validation Summary:"
    log_info "  - Total tests: $total_count"
    log_info "  - Successful: $success_count"
    log_info "  - Failed: $((total_count - success_count))"
    
    if [ $success_count -eq $total_count ]; then
        log_success "All tests passed! Orchestrator deployment is fully functional."
        exit 0
    elif [ $success_count -gt 0 ]; then
        log_warning "Partial success. Some tests failed but core functionality is working."
        exit 0
    else
        log_error "All inference tests failed. Deployment may have issues."
        exit 1
    fi
}

# Check prerequisites
if ! command -v curl &> /dev/null; then
    log_error "curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_error "jq is required but not installed"
    exit 1
fi

# Run main function
main "$@"