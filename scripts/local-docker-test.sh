#!/bin/bash

# Local Docker Testing Script for Multilingual Product Inference System
# Tests inference container locally before ECS deployment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.local.yml"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results"

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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_warning "No Python virtual environment detected. Activating .venv..."
        if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            source "$PROJECT_ROOT/.venv/bin/activate"
        else
            log_error "No .venv found. Please create virtual environment first."
            exit 1
        fi
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to create test results directory
setup_test_environment() {
    log_info "Setting up test environment..."
    
    mkdir -p "$TEST_RESULTS_DIR"
    mkdir -p "$TEST_RESULTS_DIR/logs"
    mkdir -p "$TEST_RESULTS_DIR/reports"
    
    # Create .env.local for testing
    cat > "$PROJECT_ROOT/.env.local" << EOF
# Local Docker Testing Configuration
INFERENCE_ENV=local
LOG_LEVEL=DEBUG

# Mock Service Configuration
USE_MOCK_SERVICES=true
MOCK_AWS_SERVICES=true
MOCK_MILVUS=true
MOCK_SPACY=true
MOCK_SENTENCE_TRANSFORMERS=true

# Local Service Ports
ORCHESTRATOR_PORT=8080
NER_AGENT_PORT=8081
RAG_AGENT_PORT=8082
LLM_AGENT_PORT=8083
HYBRID_AGENT_PORT=8084

# Test Configuration
TEST_DATA_DIR=tests/data
MOCK_MODELS_DIR=tests/mocks/models
TRAINING_DATA_PATH=tests/data/training_dataset.txt
EOF
    
    log_success "Test environment setup complete"
}

# Function to build Docker image locally
build_local_image() {
    log_info "Building local Docker image..."
    
    # Build the main inference image
    docker build \
        --platform linux/arm64 \
        -f Dockerfile \
        -t multilingual-inference:local \
        --build-arg BUILDPLATFORM=linux/arm64 \
        --build-arg TARGETPLATFORM=linux/arm64 \
        "$PROJECT_ROOT"
    
    log_success "Local Docker image built successfully"
}

# Function to create docker-compose file for local testing
create_docker_compose() {
    log_info "Creating Docker Compose configuration..."
    
    cat > "$DOCKER_COMPOSE_FILE" << 'EOF'
version: '3.8'

services:
  inference:
    image: multilingual-inference:local
    container_name: multilingual-inference-local
    platform: linux/arm64
    ports:
      - "8080:8080"
    environment:
      - INFERENCE_ENV=local
      - LOG_LEVEL=DEBUG
      - USE_MOCK_SERVICES=true
      - MOCK_AWS_SERVICES=true
      - MOCK_MILVUS=true
      - MOCK_SPACY=true
      - MOCK_SENTENCE_TRANSFORMERS=true
      - PYTHONPATH=/app
    volumes:
      - ./test-results/logs:/app/logs
      - ./.env.local:/app/.env.local
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - inference-network

  # Mock Milvus for testing (lightweight)
  mock-milvus:
    image: python:3.13-slim
    container_name: mock-milvus-local
    platform: linux/arm64
    ports:
      - "19530:19530"
    command: >
      sh -c "
        pip install fastapi uvicorn &&
        python -c \"
        from fastapi import FastAPI
        import uvicorn
        app = FastAPI()
        @app.get('/health')
        def health(): return {'status': 'ok'}
        @app.post('/collections')
        def create_collection(): return {'status': 'success'}
        @app.get('/collections')
        def list_collections(): return {'collections': []}
        uvicorn.run(app, host='0.0.0.0', port=19530)
        \"
      "
    networks:
      - inference-network

networks:
  inference-network:
    driver: bridge
EOF
    
    log_success "Docker Compose configuration created"
}

# Function to start local services
start_local_services() {
    log_info "Starting local services..."
    
    # Start services with docker-compose
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi
    
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check if services are healthy
    local max_attempts=12
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Inference service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to start properly"
            show_service_logs
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Function to show service logs
show_service_logs() {
    log_info "Showing service logs..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50
    fi
}

# Function to run basic API tests
run_api_tests() {
    log_info "Running API tests..."
    
    local test_passed=0
    local test_failed=0
    
    # Test 1: Health check
    log_info "Test 1: Health check"
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "✓ Health check passed"
        ((test_passed++))
    else
        log_error "✗ Health check failed"
        ((test_failed++))
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
        ((test_passed++))
    else
        log_error "✗ Basic inference test failed (HTTP $http_code)"
        ((test_failed++))
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
        ((test_passed++))
    else
        log_error "✗ Multilingual inference test failed (HTTP $http_code)"
        ((test_failed++))
    fi
    
    # Test 4: Batch inference
    log_info "Test 4: Batch inference"
    local response=$(curl -s -X POST http://localhost:8080/batch-infer \
        -H "Content-Type: application/json" \
        -d '{"products": [{"product_name": "Samsung Galaxy S24", "language": "en"}, {"product_name": "MacBook Pro M3", "language": "en"}]}' \
        -w "%{http_code}")
    
    local http_code="${response: -3}"
    if [[ "$http_code" == "200" ]]; then
        log_success "✓ Batch inference test passed"
        ((test_passed++))
    else
        log_error "✗ Batch inference test failed (HTTP $http_code)"
        ((test_failed++))
    fi
    
    # Save test results
    cat > "$TEST_RESULTS_DIR/reports/api_test_results.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "total_tests": $((test_passed + test_failed)),
    "passed": $test_passed,
    "failed": $test_failed,
    "success_rate": $(echo "scale=2; $test_passed * 100 / ($test_passed + $test_failed)" | bc -l)
}
EOF
    
    log_info "API test results: $test_passed passed, $test_failed failed"
    
    if [[ $test_failed -gt 0 ]]; then
        return 1
    fi
    return 0
}

# Function to run performance tests
run_performance_tests() {
    log_info "Running performance tests..."
    
    # Simple load test with curl
    log_info "Running load test (10 concurrent requests)..."
    
    local start_time=$(date +%s)
    
    # Run 10 concurrent requests
    for i in {1..10}; do
        (
            curl -s -X POST http://localhost:8080/infer \
                -H "Content-Type: application/json" \
                -d '{"product_name": "Test Product '$i'", "language": "en"}' \
                > "$TEST_RESULTS_DIR/logs/load_test_$i.log" 2>&1
        ) &
    done
    
    # Wait for all requests to complete
    wait
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Load test completed in ${duration}s"
    
    # Count successful responses
    local successful_requests=0
    for i in {1..10}; do
        if grep -q "success" "$TEST_RESULTS_DIR/logs/load_test_$i.log" 2>/dev/null; then
            ((successful_requests++))
        fi
    done
    
    # Save performance results
    cat > "$TEST_RESULTS_DIR/reports/performance_test_results.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "total_requests": 10,
    "successful_requests": $successful_requests,
    "duration_seconds": $duration,
    "requests_per_second": $(echo "scale=2; 10 / $duration" | bc -l)
}
EOF
    
    log_success "Performance test completed: $successful_requests/10 requests successful"
}

# Function to run Python unit tests in container
run_unit_tests() {
    log_info "Running unit tests in container..."
    
    # Run pytest inside the container
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T inference \
            python -m pytest tests/ -v --tb=short > "$TEST_RESULTS_DIR/reports/unit_test_results.log" 2>&1
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" exec -T inference \
            python -m pytest tests/ -v --tb=short > "$TEST_RESULTS_DIR/reports/unit_test_results.log" 2>&1
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        log_info "Check $TEST_RESULTS_DIR/reports/unit_test_results.log for details"
    fi
    
    return $exit_code
}

# Function to collect container metrics
collect_metrics() {
    log_info "Collecting container metrics..."
    
    # Get container stats
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" \
        > "$TEST_RESULTS_DIR/reports/container_metrics.txt"
    
    # Get container logs
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs > "$TEST_RESULTS_DIR/logs/container_logs.txt" 2>&1
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" logs > "$TEST_RESULTS_DIR/logs/container_logs.txt" 2>&1
    fi
    
    log_success "Metrics collected"
}

# Function to cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    else
        docker compose -f "$DOCKER_COMPOSE_FILE" down -v
    fi
    
    # Remove docker-compose file
    rm -f "$DOCKER_COMPOSE_FILE"
    
    log_success "Cleanup completed"
}

# Function to generate test report
generate_report() {
    log_info "Generating test report..."
    
    local report_file="$TEST_RESULTS_DIR/reports/local_docker_test_report.md"
    
    cat > "$report_file" << EOF
# Local Docker Test Report

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Test Summary

### API Tests
$(cat "$TEST_RESULTS_DIR/reports/api_test_results.json" 2>/dev/null || echo "No API test results found")

### Performance Tests
$(cat "$TEST_RESULTS_DIR/reports/performance_test_results.json" 2>/dev/null || echo "No performance test results found")

### Container Metrics
\`\`\`
$(cat "$TEST_RESULTS_DIR/reports/container_metrics.txt" 2>/dev/null || echo "No metrics collected")
\`\`\`

### Unit Test Results
\`\`\`
$(tail -20 "$TEST_RESULTS_DIR/reports/unit_test_results.log" 2>/dev/null || echo "No unit test results found")
\`\`\`

## Recommendations

- ✅ If all tests passed, the container is ready for ECS deployment
- ❌ If tests failed, review the logs and fix issues before deploying
- 📊 Monitor performance metrics to ensure adequate resource allocation

## Next Steps

1. If tests passed, proceed with ECS deployment:
   \`\`\`bash
   ./infrastructure/scripts/build-and-push-images.sh
   ./infrastructure/scripts/deploy-ecs.sh
   \`\`\`

2. If tests failed, review logs in:
   - \`$TEST_RESULTS_DIR/logs/\`
   - \`$TEST_RESULTS_DIR/reports/\`

EOF
    
    log_success "Test report generated: $report_file"
}

# Main execution function
main() {
    local command=${1:-"all"}
    
    case $command in
        "build")
            check_prerequisites
            setup_test_environment
            build_local_image
            ;;
        "test")
            check_prerequisites
            create_docker_compose
            start_local_services
            run_api_tests
            run_performance_tests
            collect_metrics
            cleanup
            ;;
        "unit-test")
            check_prerequisites
            create_docker_compose
            start_local_services
            run_unit_tests
            cleanup
            ;;
        "all")
            check_prerequisites
            setup_test_environment
            build_local_image
            create_docker_compose
            start_local_services
            
            local api_test_result=0
            local unit_test_result=0
            
            run_api_tests || api_test_result=$?
            run_performance_tests
            run_unit_tests || unit_test_result=$?
            collect_metrics
            
            cleanup
            generate_report
            
            if [[ $api_test_result -eq 0 && $unit_test_result -eq 0 ]]; then
                log_success "All tests passed! Container is ready for ECS deployment."
                exit 0
            else
                log_error "Some tests failed. Review the report before deploying."
                exit 1
            fi
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 [build|test|unit-test|all|cleanup]"
            echo ""
            echo "Commands:"
            echo "  build      - Build Docker image only"
            echo "  test       - Run API and performance tests only"
            echo "  unit-test  - Run unit tests only"
            echo "  all        - Run complete test suite (default)"
            echo "  cleanup    - Clean up containers and files"
            exit 1
            ;;
    esac
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"