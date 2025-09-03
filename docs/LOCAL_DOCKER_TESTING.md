# Local Docker Testing Guide

This guide explains how to test the inference container locally before deploying to ECS.

## Quick Start

### 1. Simple Container Test
Test the inference container with a single command:

```bash
make docker-test-simple
```

This will:
- Build the Docker image
- Start the container
- Run basic health and API tests
- Show results and cleanup

### 2. Full Test Suite with Docker Compose
Run comprehensive tests with mock services:

```bash
make docker-test-compose
```

This will:
- Build and start inference + mock services
- Run API tests, unit tests, and performance tests
- Collect container statistics
- Generate a test report

## Available Testing Commands

### Quick Tests
```bash
# Simple container test (fastest)
make docker-test-simple

# API tests only
make docker-test-api

# Unit tests in container
make docker-test-unit
```

### Comprehensive Tests
```bash
# Full test suite with Docker Compose
make docker-test-compose

# Complete testing workflow with detailed reporting
make test-docker-full
```

### Development & Debugging
```bash
# Build services without running tests
make docker-test-build

# Show container logs
make docker-logs

# Show container resource usage
make docker-stats
```

## Test Workflow

### Step 1: Local Testing
Before deploying to ECS, always run local tests:

```bash
# Run the complete test suite
make docker-test-compose
```

### Step 2: Review Results
Check the test output for:
- ✅ All API tests passing
- ✅ Unit tests passing
- ✅ Container health checks passing
- ✅ Reasonable resource usage

### Step 3: Deploy to ECS (if tests pass)
```bash
# Build and push images to ECR
./infrastructure/scripts/build-and-push-images.sh

# Deploy to ECS
./infrastructure/scripts/deploy-ecs.sh
```

## Test Details

### What Gets Tested

1. **Container Build**: Dockerfile builds successfully
2. **Service Startup**: Container starts and becomes healthy
3. **API Endpoints**: 
   - Health check endpoint
   - Basic inference endpoint
   - Multilingual inference
   - Error handling
4. **Unit Tests**: All Python unit tests run inside container
5. **Performance**: Basic load testing with concurrent requests
6. **Resource Usage**: CPU and memory consumption

### Mock Services

The tests use lightweight mock services:
- **Mock AWS Services**: Simulates Bedrock, S3, CloudWatch
- **Mock Milvus**: Simple FastAPI service for vector operations
- **Mock spaCy**: Lightweight NER simulation
- **Mock Sentence Transformers**: Embedding generation simulation

### Test Environment

Tests run with these environment variables:
```bash
INFERENCE_ENV=local
LOG_LEVEL=DEBUG
USE_MOCK_SERVICES=true
MOCK_AWS_SERVICES=true
MOCK_MILVUS=true
MOCK_SPACY=true
```

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   ```bash
   # Check Docker is running
   docker info
   
   # Check platform support
   docker buildx ls
   ```

2. **Container Won't Start**
   ```bash
   # Check logs
   make docker-logs
   
   # Check container status
   docker ps -a
   ```

3. **Tests Fail**
   ```bash
   # Run individual test types
   make docker-test-api
   make docker-test-unit
   
   # Check detailed logs
   make docker-logs
   ```

4. **Port Conflicts**
   ```bash
   # Check what's using port 8080
   lsof -i :8080
   
   # Kill conflicting processes
   sudo kill -9 <PID>
   ```

### Debug Mode

For detailed debugging, run tests manually:

```bash
# Start services in background
./scripts/test-with-compose.sh build

# Check logs in real-time
docker-compose -f docker-compose.test.yml logs -f

# Run specific tests
./scripts/test-with-compose.sh api
./scripts/test-with-compose.sh unit
```

## Test Results

### Success Indicators
- ✅ All API tests pass (200 status codes)
- ✅ Unit tests pass (pytest exit code 0)
- ✅ Container uses reasonable resources (<1GB RAM, <50% CPU)
- ✅ No error logs in container output

### Failure Indicators
- ❌ API tests return 4xx/5xx status codes
- ❌ Unit tests fail (pytest exit code != 0)
- ❌ Container crashes or restarts
- ❌ High resource usage or memory leaks

## Next Steps

### If Tests Pass ✅
1. Proceed with ECS deployment
2. Monitor ECS deployment logs
3. Run post-deployment validation

### If Tests Fail ❌
1. Review container logs
2. Fix identified issues
3. Re-run local tests
4. Only deploy after all tests pass

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run Docker Tests
  run: |
    make docker-test-compose
    
- name: Check Test Results
  run: |
    if [ $? -eq 0 ]; then
      echo "✅ Docker tests passed - ready for deployment"
    else
      echo "❌ Docker tests failed - deployment blocked"
      exit 1
    fi
```

This ensures containers are tested before any ECS deployment.