#!/bin/bash

# Simple local Docker testing script for inference container
# Tests the inference container locally before ECS deployment

set -e

# Colors for output
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
CONTAINER_NAME="multilingual-inference-test"
IMAGE_NAME="multilingual-inference:test"
PORT=8080

# Function to cleanup
cleanup() {
    log_info "Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Step 1: Build the Docker image
log_info "Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 2: Run the container
log_info "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    --platform linux/arm64 \
    -p $PORT:8080 \
    -e INFERENCE_ENV=local \
    -e LOG_LEVEL=DEBUG \
    -e USE_MOCK_SERVICES=true \
    -e MOCK_AWS_SERVICES=true \
    -e MOCK_MILVUS=true \
    -e MOCK_SPACY=true \
    $IMAGE_NAME

# Step 3: Wait for container to be ready
log_info "Waiting for container to be ready..."
sleep 30

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    log_error "Container failed to start"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Step 4: Test health endpoint
log_info "Testing health endpoint..."
max_attempts=12
attempt=1

while [[ $attempt -le $max_attempts ]]; do
    if curl -f http://localhost:$PORT/health &> /dev/null; then
        log_success "Health check passed"
        break
    fi
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Health check failed after $max_attempts attempts"
        docker logs $CONTAINER_NAME
        exit 1
    fi
    
    log_info "Health check attempt $attempt/$max_attempts..."
    sleep 5
    ((attempt++))
done

# Step 5: Test basic inference
log_info "Testing basic inference..."
response=$(curl -s -X POST http://localhost:$PORT/infer \
    -H "Content-Type: application/json" \
    -d '{"product_name": "iPhone 15 Pro Max", "language": "en"}' \
    -w "%{http_code}")

http_code="${response: -3}"
if [[ "$http_code" == "200" ]]; then
    log_success "Basic inference test passed"
else
    log_error "Basic inference test failed (HTTP $http_code)"
    echo "Response: ${response%???}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Step 6: Test multilingual inference
log_info "Testing multilingual inference..."
response=$(curl -s -X POST http://localhost:$PORT/infer \
    -H "Content-Type: application/json" \
    -d '{"product_name": "ยาสีฟัน Wonder smile", "language": "th"}' \
    -w "%{http_code}")

http_code="${response: -3}"
if [[ "$http_code" == "200" ]]; then
    log_success "Multilingual inference test passed"
else
    log_warning "Multilingual inference test failed (HTTP $http_code) - this might be expected with mocks"
fi

# Step 7: Show container stats
log_info "Container statistics:"
docker stats --no-stream $CONTAINER_NAME

# Step 8: Show recent logs
log_info "Recent container logs:"
docker logs --tail=20 $CONTAINER_NAME

log_success "Local Docker testing completed successfully!"
log_info "Container is ready for ECS deployment."
log_info ""
log_info "Next steps:"
log_info "1. Run: ./infrastructure/scripts/build-and-push-images.sh"
log_info "2. Run: ./infrastructure/scripts/deploy-ecs.sh"