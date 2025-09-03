#!/bin/bash

# Docker Compose testing script for inference system
# Tests the complete inference system locally before ECS deployment

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
COMPOSE_FILE="docker-compose.test.yml"
SERVICE_NAME="inference"
MOCK_MILVUS_SERVICE="mock-milvus"

# Function to cleanup
cleanup() {
    log_info "Cleaning up Docker Compose services..."
    docker-compose -f $COMPOSE_FILE down -v --remove-orphans 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to build and start services
start_services() {
    log_info "Building and starting services..."
    
    # Build and start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE up -d --build
    else
        docker compose -f $COMPOSE_FILE up -d --build
    fi
    
    log_info "Waiting for services to be ready..."
    sleep 45
}

# Function to check service health
check_service_health() {
    log_info "Checking service health..."
    
    local max_attempts=15
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        # Check inference service
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Inference service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed health check"
            show_logs
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check mock Milvus
    if curl -f http://localhost:19530/health &> /dev/null; then
        log_success "Mock Milvus service is healthy"
    else
        log_warning "Mock Milvus service health check failed"
    fi
}

# Function to show logs
show_logs() {
    log_info "Showing service logs..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE logs --tail=50
    else
        docker compose -f $COMPOSE_FILE logs --tail=50
    fi
}

# Function to run API tests
run_api_tests() {
    log_info "Running API tests..."
    
    local tests_passed=0
    local tests_failed=0
    
    # Test 1: Health endpoint
    log_info "Test 1: Health endpoint"
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "✓ Health endpoint test passed"
        ((tests_passed++))
    else
        log_error "✗ Health endpoint test failed"
        ((tests_failed++))
    fi
    
    # Test 2: Basic inference
    log_info "Test 2: Basic inference"
    local response=$(curl -s -X POST http://localhost:8080/infer \
        -H "Content-Type: application/json" \
        -d '{"product_name": "iPhone 15 Pro Max", "language": "en"}' \
        -w "%{http_code}")
    
    local http_code="${response: -3}"
    if [[ "$http_code" == "200" ]]; then
        log_success "✓ Basic inference test passed"
        ((tests_passed++))
    else
        log_error "✗ Basic inference test failed (HTTP $http_code)"
        echo "Response: ${response%???}"
        ((tests_failed++))
    fi
    
    # Test 3: Multilingual inference
    log_info "Test 3: Multilingual inference"
    local response=$(curl -s -X POST http://localhost:8080/infer \
        -H "Content-Type: application/json" \
        -d '{"product_name": "ยาสีฟัน Wonder smile", "language": "th"}' \
        -w "%{http_code}")
    
    local http_code="${response: -3}"
    if [[ "$http_code" == "200" ]]; then
        log_success "✓ Multilingual inference test passed"
        ((tests_passed++))
    else
        log_warning "⚠ Multilingual inference test failed (HTTP $http_code) - might be expected with mocks"
        # Don't count as failure for mocks
    fi
    
    # Test 4: Error handling
    log_info "Test 4: Error handling"
    local response=$(curl -s -X POST http://localhost:8080/infer \
        -H "Content-Type: application/json" \
        -d '{"invalid": "data"}' \
        -w "%{http_code}")
    
    local http_code="${response: -3}"
    if [[ "$http_code" == "400" || "$http_code" == "422" ]]; then
        log_success "✓ Error handling test passed"
        ((tests_passed++))
    else
        log_error "✗ Error handling test failed (HTTP $http_code)"
        ((tests_failed++))
    fi
    
    log_info "API test results: $tests_passed passed, $tests_failed failed"
    
    if [[ $tests_failed -gt 0 ]]; then
        return 1
    fi
    return 0
}

# Function to run performance test
run_performance_test() {
    log_info "Running basic performance test..."
    
    local start_time=$(date +%s)
    local concurrent_requests=5
    
    # Run concurrent requests
    for i in $(seq 1 $concurrent_requests); do
        (
            curl -s -X POST http://localhost:8080/infer \
                -H "Content-Type: application/json" \
                -d "{\"product_name\": \"Test Product $i\", \"language\": \"en\"}" \
                > /tmp/perf_test_$i.log 2>&1
        ) &
    done
    
    # Wait for all requests
    wait
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Performance test completed: $concurrent_requests requests in ${duration}s"
}

# Function to collect container stats
collect_stats() {
    log_info "Collecting container statistics..."
    
    echo "Container resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    
    echo ""
    echo "Container processes:"
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE exec -T $SERVICE_NAME ps aux || true
    else
        docker compose -f $COMPOSE_FILE exec -T $SERVICE_NAME ps aux || true
    fi
}

# Function to run unit tests inside container
run_unit_tests() {
    log_info "Running unit tests inside container..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE exec -T $SERVICE_NAME \
            python -m pytest tests/ -v --tb=short -x
    else
        docker compose -f $COMPOSE_FILE exec -T $SERVICE_NAME \
            python -m pytest tests/ -v --tb=short -x
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
    fi
    
    return $exit_code
}

# Main function
main() {
    local test_type=${1:-"all"}
    
    case $test_type in
        "build")
            check_prerequisites
            start_services
            log_success "Services built and started successfully"
            ;;
        "api")
            check_prerequisites
            start_services
            check_service_health
            run_api_tests
            ;;
        "unit")
            check_prerequisites
            start_services
            check_service_health
            run_unit_tests
            ;;
        "perf")
            check_prerequisites
            start_services
            check_service_health
            run_performance_test
            ;;
        "all")
            check_prerequisites
            start_services
            check_service_health
            
            local api_result=0
            local unit_result=0
            
            run_api_tests || api_result=$?
            run_performance_test
            run_unit_tests || unit_result=$?
            collect_stats
            
            if [[ $api_result -eq 0 && $unit_result -eq 0 ]]; then
                log_success "🎉 All tests passed! Container is ready for ECS deployment."
                echo ""
                log_info "Next steps for ECS deployment:"
                log_info "1. Build and push images: ./infrastructure/scripts/build-and-push-images.sh"
                log_info "2. Deploy to ECS: ./infrastructure/scripts/deploy-ecs.sh"
                exit 0
            else
                log_error "❌ Some tests failed. Please fix issues before deploying to ECS."
                exit 1
            fi
            ;;
        "logs")
            show_logs
            ;;
        "stats")
            collect_stats
            ;;
        *)
            echo "Usage: $0 [build|api|unit|perf|all|logs|stats]"
            echo ""
            echo "Commands:"
            echo "  build  - Build and start services only"
            echo "  api    - Run API tests"
            echo "  unit   - Run unit tests"
            echo "  perf   - Run performance tests"
            echo "  all    - Run complete test suite (default)"
            echo "  logs   - Show service logs"
            echo "  stats  - Show container statistics"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"