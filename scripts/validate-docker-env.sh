#!/bin/bash

# Docker Environment Validation Script
# Checks if the environment is ready for Docker testing

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

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    log_success "Docker is installed: $(docker --version)"
}

# Check Docker daemon
check_docker_daemon() {
    log_info "Checking Docker daemon..."
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        echo "Please start Docker daemon"
        return 1
    fi
    
    log_success "Docker daemon is running"
}

# Check Docker Compose
check_docker_compose() {
    log_info "Checking Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose is installed: $(docker-compose --version)"
    elif docker compose version &> /dev/null; then
        log_success "Docker Compose (plugin) is installed: $(docker compose version)"
    else
        log_error "Docker Compose is not installed"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        return 1
    fi
}

# Check platform support
check_platform_support() {
    log_info "Checking ARM64 platform support..."
    
    if docker buildx ls | grep -q "linux/arm64"; then
        log_success "ARM64 platform support is available"
    else
        log_warning "ARM64 platform support may not be available"
        echo "You may need to enable experimental features or install buildx"
    fi
}

# Check available ports
check_ports() {
    log_info "Checking required ports..."
    
    local ports=(8080 19530)
    local port_issues=0
    
    for port in "${ports[@]}"; do
        if lsof -i :$port &> /dev/null; then
            log_warning "Port $port is already in use"
            echo "Process using port $port:"
            lsof -i :$port
            ((port_issues++))
        else
            log_success "Port $port is available"
        fi
    done
    
    if [[ $port_issues -gt 0 ]]; then
        log_warning "Some ports are in use. Tests may fail or use alternative ports."
    fi
}

# Check disk space
check_disk_space() {
    log_info "Checking available disk space..."
    
    local available_gb=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [[ $available_gb -lt 2 ]]; then
        log_error "Insufficient disk space: ${available_gb}GB available (need at least 2GB)"
        return 1
    elif [[ $available_gb -lt 5 ]]; then
        log_warning "Low disk space: ${available_gb}GB available (recommended: 5GB+)"
    else
        log_success "Sufficient disk space: ${available_gb}GB available"
    fi
}

# Check memory
check_memory() {
    log_info "Checking available memory..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        local total_mb=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024)}')
    else
        # Linux
        local total_mb=$(free -m | awk 'NR==2{print $2}')
    fi
    
    local total_gb=$((total_mb / 1024))
    
    if [[ $total_gb -lt 4 ]]; then
        log_error "Insufficient memory: ${total_gb}GB total (need at least 4GB)"
        return 1
    elif [[ $total_gb -lt 8 ]]; then
        log_warning "Limited memory: ${total_gb}GB total (recommended: 8GB+)"
    else
        log_success "Sufficient memory: ${total_gb}GB total"
    fi
}

# Check Python virtual environment
check_python_env() {
    log_info "Checking Python environment..."
    
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_warning "No Python virtual environment detected"
        if [[ -f ".venv/bin/activate" ]]; then
            log_info "Found .venv directory. Activate with: source .venv/bin/activate"
        else
            log_warning "No .venv directory found. Create with: python3.13 -m venv .venv"
        fi
    else
        log_success "Python virtual environment is active: $VIRTUAL_ENV"
    fi
}

# Check required files
check_required_files() {
    log_info "Checking required files..."
    
    local required_files=(
        "Dockerfile"
        "docker-compose.test.yml"
        "scripts/test-with-compose.sh"
        "scripts/test-inference-local.sh"
        "requirements.txt"
    )
    
    local missing_files=0
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "Found: $file"
        else
            log_error "Missing: $file"
            ((missing_files++))
        fi
    done
    
    if [[ $missing_files -gt 0 ]]; then
        log_error "$missing_files required files are missing"
        return 1
    fi
}

# Main validation function
main() {
    log_info "Validating Docker environment for inference container testing..."
    echo ""
    
    local checks_failed=0
    
    check_docker || ((checks_failed++))
    check_docker_daemon || ((checks_failed++))
    check_docker_compose || ((checks_failed++))
    check_platform_support
    check_ports
    check_disk_space || ((checks_failed++))
    check_memory || ((checks_failed++))
    check_python_env
    check_required_files || ((checks_failed++))
    
    echo ""
    
    if [[ $checks_failed -eq 0 ]]; then
        log_success "🎉 Environment validation passed! Ready for Docker testing."
        echo ""
        log_info "Next steps:"
        log_info "1. Run simple test: make docker-test-simple"
        log_info "2. Run full test suite: make docker-test-compose"
        exit 0
    else
        log_error "❌ Environment validation failed ($checks_failed issues found)"
        echo ""
        log_info "Please fix the issues above before running Docker tests."
        exit 1
    fi
}

# Run validation
main "$@"