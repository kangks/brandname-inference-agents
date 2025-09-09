#!/bin/bash

# Test script for Multilingual Brand Name Inference System
# Tests all deployed endpoints with sample product names

set -e

# Configuration
ALB_DNS="production-inference-alb-2106753314.us-east-1.elb.amazonaws.com"
BASE_URL="http://$ALB_DNS"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test data - sample product names for brand inference
declare -A TEST_PRODUCTS=(
    ["iPhone 15 Pro Max"]="Apple smartphone with advanced camera system"
    ["Samsung Galaxy S24 Ultra"]="Samsung flagship smartphone with S Pen"
    ["Nike Air Jordan 1"]="Classic basketball sneaker from Nike"
    ["Adidas Ultraboost 22"]="Running shoe with Boost technology"
    ["MacBook Pro M3"]="Apple laptop with M3 chip"
    ["Dell XPS 13"]="Dell premium ultrabook laptop"
    ["Coca-Cola Zero Sugar"]="Sugar-free cola beverage"
    ["Pepsi Max"]="Zero calorie cola drink"
    ["Toyota Camry Hybrid"]="Hybrid sedan from Toyota"
    ["Tesla Model 3"]="Electric sedan from Tesla"
)

# Function to test endpoint health
test_health() {
    local endpoint=$1
    local service_name=$2
    
    echo -e "${BLUE}Testing $service_name health...${NC}"
    
    response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$endpoint/health" || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ $service_name is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name health check failed (HTTP $response)${NC}"
        return 1
    fi
}

# Function to test brand inference
test_brand_inference() {
    local endpoint=$1
    local service_name=$2
    local product_name=$3
    local description=$4
    
    echo -e "${YELLOW}Testing $service_name with: '$product_name'${NC}"
    
    # Create JSON payload
    local payload=$(cat <<EOF
{
    "product_name": "$product_name",
    "description": "$description",
    "language": "en",
    "context": "e-commerce"
}
EOF
)
    
    # Make request
    response=$(curl -s -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "$payload" \
        -o /tmp/inference_response.json \
        "$endpoint/inference" || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ Success (HTTP $response)${NC}"
        echo -e "${BLUE}Response:${NC}"
        cat /tmp/inference_response.json | jq '.' 2>/dev/null || cat /tmp/inference_response.json
        echo ""
    else
        echo -e "${RED}❌ Failed (HTTP $response)${NC}"
        echo -e "${RED}Response:${NC}"
        cat /tmp/inference_response.json 2>/dev/null || echo "No response body"
        echo ""
    fi
    
    return 0
}

# Function to test Milvus directly
test_milvus() {
    echo -e "${BLUE}Testing Milvus vector database...${NC}"
    
    # Test Milvus health endpoint
    milvus_health=$(curl -s -w "%{http_code}" -o /tmp/milvus_health.json "http://milvus.multilingual-inference.local:9091/healthz" 2>/dev/null || echo "000")
    
    if [ "$milvus_health" = "200" ]; then
        echo -e "${GREEN}✅ Milvus is healthy${NC}"
    else
        echo -e "${RED}❌ Milvus health check failed (HTTP $milvus_health)${NC}"
        echo -e "${YELLOW}Note: Milvus might only be accessible from within the VPC${NC}"
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}🚀 Starting Multilingual Brand Name Inference System Tests${NC}"
    echo -e "${BLUE}Load Balancer: $ALB_DNS${NC}"
    echo ""
    
    # Test 1: Health checks for all services
    echo -e "${YELLOW}=== HEALTH CHECKS ===${NC}"
    test_health "$BASE_URL" "Orchestrator (Default)"
    test_health "$BASE_URL/ner" "NER Agent"
    test_health "$BASE_URL/rag" "RAG Agent" 
    test_health "$BASE_URL/llm" "LLM Agent"
    test_health "$BASE_URL/hybrid" "Hybrid Agent"
    echo ""
    
    # Test 2: Milvus vector database
    echo -e "${YELLOW}=== MILVUS DATABASE ===${NC}"
    test_milvus
    echo ""
    
    # Test 3: Brand inference with different agents
    echo -e "${YELLOW}=== BRAND INFERENCE TESTS ===${NC}"
    
    # Test with a few sample products
    local test_products=("iPhone 15 Pro Max" "Nike Air Jordan 1" "Tesla Model 3")
    
    for product in "${test_products[@]}"; do
        description="${TEST_PRODUCTS[$product]}"
        
        echo -e "${BLUE}--- Testing with: $product ---${NC}"
        
        # Test orchestrator (main endpoint)
        test_brand_inference "$BASE_URL" "Orchestrator" "$product" "$description"
        
        # Test NER agent
        test_brand_inference "$BASE_URL/ner" "NER Agent" "$product" "$description"
        
        # Test RAG agent  
        test_brand_inference "$BASE_URL/rag" "RAG Agent" "$product" "$description"
        
        # Test LLM agent
        test_brand_inference "$BASE_URL/llm" "LLM Agent" "$product" "$description"
        
        # Test Hybrid agent
        test_brand_inference "$BASE_URL/hybrid" "Hybrid Agent" "$product" "$description"
        
        echo -e "${BLUE}----------------------------------------${NC}"
        echo ""
    done
    
    echo -e "${GREEN}🎉 Testing completed!${NC}"
}

# Run tests
main "$@"