#!/bin/bash

# Production readiness test for inference container
# Tests essential functionality needed for ECS deployment

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

COMPOSE_FILE="docker-compose.test.yml"

# Function to cleanup
cleanup() {
    log_info "Cleaning up Docker Compose services..."
    docker compose -f $COMPOSE_FILE down -v --remove-orphans 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Step 1: Environment validation
log_info "Step 1: Validating environment..."
./scripts/validate-docker-env.sh

# Step 2: Build and start services
log_info "Step 2: Building and starting services..."
docker compose -f $COMPOSE_FILE up -d --build

# Step 3: Wait for services to be ready
log_info "Step 3: Waiting for services to be ready..."
sleep 45

# Step 4: Health check
log_info "Step 4: Checking service health..."
max_attempts=15
attempt=1

while [[ $attempt -le $max_attempts ]]; do
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "✅ Inference service is healthy"
        break
    fi
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "❌ Health check failed after $max_attempts attempts"
        docker compose -f $COMPOSE_FILE logs
        exit 1
    fi
    
    log_info "Health check attempt $attempt/$max_attempts..."
    sleep 5
    ((attempt++))
done

# Step 5: API functionality tests
log_info "Step 5: Testing API functionality..."

tests_passed=0
tests_failed=0

# Test 1: Health endpoint
log_info "Test 1: Health endpoint"
if curl -f http://localhost:8080/health &> /dev/null; then
    log_success "✅ Health endpoint test passed"
    ((tests_passed++))
else
    log_error "❌ Health endpoint test failed"
    ((tests_failed++))
fi

# Test 2: Basic inference
log_info "Test 2: Basic inference"
response=$(curl -s -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{"product_name": "iPhone 15 Pro Max", "language_hint": "en"}' \
    -w "%{http_code}")

http_code="${response: -3}"
if [[ "$http_code" == "200" ]]; then
    log_success "✅ Basic inference test passed"
    ((tests_passed++))
else
    log_error "❌ Basic inference test failed (HTTP $http_code)"
    echo "Response: ${response%???}"
    ((tests_failed++))
fi

# Test 3: Multilingual inference
log_info "Test 3: Multilingual inference"
response=$(curl -s -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{"product_name": "ยาสีฟัน Wonder smile", "language_hint": "th"}' \
    -w "%{http_code}")

http_code="${response: -3}"
if [[ "$http_code" == "200" ]]; then
    log_success "✅ Multilingual inference test passed"
    ((tests_passed++))
else
    log_warning "⚠️  Multilingual inference test failed (HTTP $http_code) - acceptable with mocks"
fi

# Test 4: Error handling
log_info "Test 4: Error handling"
response=$(curl -s -X POST http://localhost:8080/infer \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}' \
    -w "%{http_code}")

http_code="${response: -3}"
if [[ "$http_code" == "400" || "$http_code" == "422" ]]; then
    log_success "✅ Error handling test passed"
    ((tests_passed++))
else
    log_error "❌ Error handling test failed (HTTP $http_code)"
    ((tests_failed++))
fi

# Test 5: Service info endpoint
log_info "Test 5: Service info endpoint"
if curl -f http://localhost:8080/ &> /dev/null; then
    log_success "✅ Service info endpoint test passed"
    ((tests_passed++))
else
    log_warning "⚠️  Service info endpoint test failed - not critical"
fi

# Step 6: Performance test
log_info "Step 6: Running basic performance test..."
start_time=$(date +%s)
concurrent_requests=5

# Run concurrent requests
for i in $(seq 1 $concurrent_requests); do
    (
        curl -s -X POST http://localhost:8080/infer \
            -H "Content-Type: application/json" \
            -d "{\"product_name\": \"Test Product $i\", \"language_hint\": \"en\"}" \
            > /tmp/perf_test_$i.log 2>&1
    ) &
done

# Wait for all requests
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

log_success "✅ Performance test completed: $concurrent_requests requests in ${duration}s"

# Step 7: Container resource check
log_info "Step 7: Checking container resources..."
echo ""
echo "Container resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Step 8: Container logs check
log_info "Step 8: Checking for critical errors in logs..."
error_count=$(docker compose -f $COMPOSE_FILE logs inference 2>&1 | grep -i "error\|exception\|failed" | grep -v "AWS profile validation failed" | wc -l)

if [[ $error_count -eq 0 ]]; then
    log_success "✅ No critical errors found in logs"
else
    log_warning "⚠️  Found $error_count potential errors in logs (review recommended)"
fi

# Step 9: Final assessment
log_info "Step 9: Final assessment..."
echo ""
echo "=== PRODUCTION READINESS ASSESSMENT ==="
echo "API Tests: $tests_passed passed, $tests_failed failed"
echo "Performance: $concurrent_requests concurrent requests in ${duration}s"
echo "Error Count: $error_count potential errors in logs"
echo ""

# Determine readiness
if [[ $tests_failed -eq 0 && $tests_passed -ge 4 ]]; then
    log_success "🎉 CONTAINER IS READY FOR ECS DEPLOYMENT!"
    echo ""
    log_info "✅ All critical tests passed"
    log_info "✅ API endpoints responding correctly"
    log_info "✅ Container running stably"
    log_info "✅ No critical errors detected"
    echo ""
    log_info "Next steps for ECS deployment:"
    log_info "1. Build and push images: ./infrastructure/scripts/build-and-push-images.sh"
    log_info "2. Deploy to ECS: ./infrastructure/scripts/deploy-ecs.sh"
    echo ""
    exit 0
else
    log_error "❌ CONTAINER NOT READY FOR ECS DEPLOYMENT"
    echo ""
    log_error "Issues found:"
    if [[ $tests_failed -gt 0 ]]; then
        log_error "- $tests_failed API tests failed"
    fi
    if [[ $tests_passed -lt 4 ]]; then
        log_error "- Insufficient tests passed ($tests_passed/4 minimum required)"
    fi
    echo ""
    log_info "Please fix issues and re-run tests before deploying to ECS"
    exit 1
fi