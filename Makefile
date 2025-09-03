# Makefile for multilingual product inference system
# Supports Python 3.13 development environment with PEP 8 compliance

.PHONY: help install install-dev setup lint format type-check test test-unit test-integration clean build docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  setup         - Complete development environment setup"
	@echo "  lint          - Run flake8 linting"
	@echo "  format        - Format code with black"
	@echo "  type-check    - Run mypy type checking"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-local    - Run local development tests"
	@echo ""
	@echo "Docker Testing:"
	@echo "  docker-test-simple      - Simple Docker container test"
	@echo "  docker-test-compose     - Full Docker Compose test suite"
	@echo "  docker-test-api         - API tests with Docker"
	@echo "  docker-test-unit        - Unit tests in Docker"
	@echo "  docker-test-build       - Build Docker services"
	@echo "  docker-test-production  - Production readiness test (recommended)"
	@echo "  test-docker-full        - Complete Docker test workflow"
	@echo ""
	@echo "Docker Operations:"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  docker-logs   - Show container logs"
	@echo "  docker-stats  - Show container statistics"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         - Clean build artifacts"
	@echo "  build         - Build package"

# Python and pip commands
PYTHON := python3.13
PIP := $(PYTHON) -m pip
VENV := venv

# Check if virtual environment exists
check-venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating Python 3.13 virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi

# Install production dependencies
install: check-venv
	@echo "Installing production dependencies..."
	@. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	@. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

# Install development dependencies
install-dev: check-venv
	@echo "Installing development dependencies..."
	@. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	@. $(VENV)/bin/activate && $(PIP) install -e ".[dev]"

# Complete development environment setup
setup: install-dev
	@echo "Setting up development environment..."
	@. $(VENV)/bin/activate && $(PYTHON) -m spacy download en_core_web_sm
	@if [ ! -f ".env" ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "Please update .env file with your configuration"; \
	fi
	@echo "Development environment setup complete!"
	@echo "Activate with: source $(VENV)/bin/activate"

# Code formatting with black
format:
	@echo "Formatting code with black..."
	@. $(VENV)/bin/activate && black inference/ training/ tests/ --line-length 88

# Linting with flake8
lint:
	@echo "Running flake8 linting..."
	@. $(VENV)/bin/activate && flake8 inference/ training/ tests/ --max-line-length 88 --extend-ignore E203,W503

# Type checking with mypy
type-check:
	@echo "Running mypy type checking..."
	@. $(VENV)/bin/activate && mypy inference/ training/ --ignore-missing-imports

# Run all code quality checks
check: format lint type-check
	@echo "All code quality checks completed"

# Run all tests
test:
	@echo "Running all tests..."
	@. $(VENV)/bin/activate && pytest tests/ -v --cov=inference --cov=training --cov-report=html --cov-report=term

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	@. $(VENV)/bin/activate && pytest tests/ -v -m "unit" --cov=inference --cov=training

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	@. $(VENV)/bin/activate && pytest tests/ -v -m "integration"

# Run local development tests (no AWS dependencies)
test-local:
	@echo "Running local development tests..."
	@. $(VENV)/bin/activate && python tests/run_local_tests.py --test-type quick

# Run comprehensive local tests
test-local-full:
	@echo "Running comprehensive local tests..."
	@. $(VENV)/bin/activate && python tests/run_local_tests.py --test-type comprehensive

# Run accuracy validation tests
test-accuracy:
	@echo "Running accuracy validation tests..."
	@. $(VENV)/bin/activate && python tests/run_local_tests.py --test-type accuracy

# Run performance benchmark tests
test-performance:
	@echo "Running performance benchmark tests..."
	@. $(VENV)/bin/activate && python tests/run_local_tests.py --test-type performance

# Demo local testing framework
demo-testing:
	@echo "Running local testing framework demo..."
	@. $(VENV)/bin/activate && python tests/examples/demo_local_testing.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

# Build package
build: clean
	@echo "Building package..."
	@. $(VENV)/bin/activate && $(PYTHON) -m build

# Docker build for ARM64 platform
docker-build:
	@echo "Building Docker image for ARM64..."
	@docker build --platform linux/arm64 -t multilingual-inference:latest .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	@docker run --platform linux/arm64 -p 8080:8080 multilingual-inference:latest

# Local Docker testing targets
docker-test-simple:
	@echo "Running simple Docker container test..."
	@./scripts/test-inference-local.sh

docker-test-compose:
	@echo "Running Docker Compose test suite..."
	@./scripts/test-with-compose.sh all

docker-test-api:
	@echo "Running API tests with Docker Compose..."
	@./scripts/test-with-compose.sh api

docker-test-unit:
	@echo "Running unit tests in Docker container..."
	@./scripts/test-with-compose.sh unit

docker-test-build:
	@echo "Building Docker services..."
	@./scripts/test-with-compose.sh build

docker-logs:
	@echo "Showing Docker container logs..."
	@./scripts/test-with-compose.sh logs

docker-stats:
	@echo "Showing Docker container statistics..."
	@./scripts/test-with-compose.sh stats

# Validate Docker environment
docker-validate:
	@echo "Validating Docker environment..."
	@./scripts/validate-docker-env.sh

# Production readiness test
docker-test-production:
	@echo "Running production readiness test..."
	@./scripts/test-production-ready.sh

# Complete local testing workflow
test-docker-full:
	@echo "Running complete Docker test workflow..."
	@./scripts/local-docker-test.sh all

# Development server (if implemented)
dev-server:
	@echo "Starting development server..."
	@. $(VENV)/bin/activate && $(PYTHON) -m inference.main --dev

# Jupyter notebook server
jupyter:
	@echo "Starting Jupyter notebook server..."
	@. $(VENV)/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Install pre-commit hooks
pre-commit:
	@echo "Installing pre-commit hooks..."
	@. $(VENV)/bin/activate && pre-commit install

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	@. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	@. $(VENV)/bin/activate && $(PIP) install --upgrade -r requirements.txt

# Check AWS configuration
check-aws:
	@echo "Checking AWS ml-sandbox profile..."
	@aws --profile ml-sandbox sts get-caller-identity || echo "AWS ml-sandbox profile not configured"

# Validate environment
validate-env: check-aws
	@echo "Validating environment..."
	@. $(VENV)/bin/activate && $(PYTHON) -c "from inference.config.settings import get_config; print('Configuration loaded successfully')"

# Show project info
info:
	@echo "Project: Multilingual Product Inference System"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Platform: ARM64 optimized"
	@echo "AWS Profile: ml-sandbox"
	@echo "Region: us-east-1"
	@echo "Code Standards: PEP 8"