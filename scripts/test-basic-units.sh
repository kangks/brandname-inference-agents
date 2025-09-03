#!/bin/bash

# Test basic unit tests in Docker container
# Runs only the core test files without problematic imports

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

COMPOSE_FILE="docker-compose.test.yml"

# Start services
log_info "Starting Docker Compose services..."
docker compose -f $COMPOSE_FILE up -d --build

# Wait for services
log_info "Waiting for services to be ready..."
sleep 30

# Check if inference service is healthy
if curl -f http://localhost:8080/health &> /dev/null; then
    log_success "Inference service is healthy"
else
    log_error "Inference service is not healthy"
    docker compose -f $COMPOSE_FILE logs
    docker compose -f $COMPOSE_FILE down -v
    exit 1
fi

# Run basic unit tests (only core test files)
log_info "Running basic unit tests..."
docker compose -f $COMPOSE_FILE exec -T inference \
    python -m pytest tests/test_config.py tests/test_data_models.py -v --tb=short

test_result=$?

# Cleanup
log_info "Cleaning up..."
docker compose -f $COMPOSE_FILE down -v

if [[ $test_result -eq 0 ]]; then
    log_success "Basic unit tests passed!"
    exit 0
else
    log_error "Basic unit tests failed!"
    exit 1
fi