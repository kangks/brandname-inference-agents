"""
Configuration management package for multilingual product inference.

This package provides comprehensive configuration management, model registry,
and validation utilities following PEP 8 standards.
"""

from .settings import (
    SystemConfig,
    ConfigManager,
    Environment,
    ModelType,
    NERModelConfig,
    EmbeddingModelConfig,
    LLMModelConfig,
    MilvusConfig,
    AWSConfig,
    AgentConfig,
    get_config,
    setup_logging
)
from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelStatus,
    get_model_registry
)
from .validators import (
    ConfigValidator,
    ValidationError,
    validate_configuration,
    validate_environment_variables,
    check_model_compatibility
)

__all__ = [
    # Core configuration
    "SystemConfig",
    "ConfigManager",
    "Environment",
    "ModelType",
    
    # Model configurations
    "NERModelConfig",
    "EmbeddingModelConfig", 
    "LLMModelConfig",
    "MilvusConfig",
    "AWSConfig",
    "AgentConfig",
    
    # Model registry
    "ModelRegistry",
    "ModelMetadata",
    "ModelStatus",
    
    # Validation
    "ConfigValidator",
    "ValidationError",
    
    # Utility functions
    "get_config",
    "get_model_registry",
    "setup_logging",
    "validate_configuration",
    "validate_environment_variables",
    "check_model_compatibility"
]