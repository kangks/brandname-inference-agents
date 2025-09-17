"""
Configuration validation utilities.

This module provides comprehensive validation functions for configuration
settings, model parameters, and system requirements following PEP 8 standards.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import importlib.util

from .settings import SystemConfig, Environment


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Optional field name that caused the error
        """
        super().__init__(message)
        self.field = field
        self.message = message


class ConfigValidator:
    """Comprehensive configuration validator with detailed error reporting."""
    
    def __init__(self) -> None:
        """Initialize configuration validator."""
        self.logger = logging.getLogger(__name__)
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: SystemConfig) -> Tuple[bool, List[ValidationError], List[str]]:
        """
        Validate complete system configuration.
        
        Args:
            config: SystemConfig object to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate basic configuration
        self._validate_basic_config(config)
        
        # Validate environment-specific settings
        self._validate_environment_config(config)
        
        # Validate model configurations
        self._validate_model_configs(config)
        
        # Validate AWS configuration
        self._validate_aws_config(config)
        
        # Validate Milvus configuration
        self._validate_milvus_config(config)
        
        # Validate agent configuration
        self._validate_agent_config(config)
        
        # Check system dependencies
        self._validate_system_dependencies(config)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_basic_config(self, config: SystemConfig) -> None:
        """Validate basic configuration settings."""
        # Validate environment
        if not isinstance(config.environment, Environment):
            self.errors.append(
                ValidationError("Invalid environment type", "environment")
            )
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level not in valid_log_levels:
            self.errors.append(
                ValidationError(
                    f"Invalid log level '{config.log_level}'. Must be one of {valid_log_levels}",
                    "log_level"
                )
            )
    
    def _validate_environment_config(self, config: SystemConfig) -> None:
        """Validate environment-specific configuration."""
        if config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if config.log_level == "DEBUG":
                self.warnings.append(
                    "DEBUG log level not recommended for production environment"
                )
            
            if config.agent.timeout_seconds < 10:
                self.warnings.append(
                    "Very short timeout may cause issues in production"
                )
        
        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific validations
            if config.agent.max_concurrent_requests > 50:
                self.warnings.append(
                    "High concurrent request limit may overwhelm development resources"
                )
    
    def _validate_model_configs(self, config: SystemConfig) -> None:
        """Validate model configurations."""
        # Validate NER model
        if not config.ner_model.model_name:
            self.errors.append(
                ValidationError("NER model name cannot be empty", "ner_model.model_name")
            )
        
        if config.ner_model.model_path:
            model_path = Path(config.ner_model.model_path)
            if not model_path.exists() and not config.ner_model.model_name.startswith("en_"):
                self.warnings.append(
                    f"NER model path does not exist: {config.ner_model.model_path}"
                )
        
        # Validate embedding model
        if not config.embedding_model.model_name:
            self.errors.append(
                ValidationError("Embedding model name cannot be empty", "embedding_model.model_name")
            )
        
        if config.embedding_model.embedding_dimension <= 0:
            self.errors.append(
                ValidationError(
                    "Embedding dimension must be positive",
                    "embedding_model.embedding_dimension"
                )
            )
        
        # Validate LLM model
        if not config.llm_model.model_id:
            self.errors.append(
                ValidationError("LLM model ID cannot be empty", "llm_model.model_id")
            )
        
        if config.llm_model.temperature < 0 or config.llm_model.temperature > 2:
            self.errors.append(
                ValidationError(
                    "LLM temperature must be between 0 and 2",
                    "llm_model.temperature"
                )
            )
    
    def _validate_aws_config(self, config: SystemConfig) -> None:
        """Validate AWS configuration."""
        # AWS profile is optional - ECS deployments use IAM roles
        # No validation needed for profile_name
        
        # Validate AWS region
        valid_regions = [
            "us-east-1", "us-east-2", "us-west-1", "us-west-2",
            "eu-west-1", "eu-west-2", "eu-central-1",
            "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
        ]
        
        if config.aws.region not in valid_regions:
            self.warnings.append(
                f"AWS region '{config.aws.region}' may not support all required services"
            )
        
        # Check if AWS credentials are available
        try:
            import boto3
            session = boto3.Session(profile_name=config.aws.profile_name)
            credentials = session.get_credentials()
            if not credentials:
                self.errors.append(
                    ValidationError(
                        f"AWS credentials not found for profile '{config.aws.profile_name}'",
                        "aws.profile_name"
                    )
                )
        except Exception as e:
            self.warnings.append(f"Could not validate AWS credentials: {str(e)}")
    
    def _validate_milvus_config(self, config: SystemConfig) -> None:
        """Validate Milvus configuration."""
        # Validate host and port
        if not config.milvus.host:
            self.errors.append(
                ValidationError("Milvus host cannot be empty", "milvus.host")
            )
        
        if config.milvus.port <= 0 or config.milvus.port > 65535:
            self.errors.append(
                ValidationError(
                    "Milvus port must be between 1 and 65535",
                    "milvus.port"
                )
            )
        
        # Validate collection name
        if not config.milvus.collection_name:
            self.errors.append(
                ValidationError("Milvus collection name cannot be empty", "milvus.collection_name")
            )
        
        # Check if collection name is valid (alphanumeric and underscores only)
        if not config.milvus.collection_name.replace("_", "").isalnum():
            self.errors.append(
                ValidationError(
                    "Milvus collection name must contain only alphanumeric characters and underscores",
                    "milvus.collection_name"
                )
            )
    
    def _validate_agent_config(self, config: SystemConfig) -> None:
        """Validate agent configuration."""
        # Validate timeout
        if config.agent.timeout_seconds <= 0:
            self.errors.append(
                ValidationError("Agent timeout must be positive", "agent.timeout_seconds")
            )
        
        if config.agent.timeout_seconds > 300:  # 5 minutes
            self.warnings.append(
                "Very long timeout may cause poor user experience"
            )
        
        # Validate concurrent requests
        if config.agent.max_concurrent_requests <= 0:
            self.errors.append(
                ValidationError(
                    "Max concurrent requests must be positive",
                    "agent.max_concurrent_requests"
                )
            )
        
        if config.agent.max_concurrent_requests > 100:
            self.warnings.append(
                "High concurrent request limit may cause resource exhaustion"
            )
        
        # Validate confidence threshold
        if config.agent.confidence_threshold < 0 or config.agent.confidence_threshold > 1:
            self.errors.append(
                ValidationError(
                    "Confidence threshold must be between 0 and 1",
                    "agent.confidence_threshold"
                )
            )
    
    def _validate_system_dependencies(self, config: SystemConfig) -> None:
        """Validate system dependencies and requirements."""
        # Check Python version
        import sys
        if sys.version_info < (3, 9):
            self.errors.append(
                ValidationError("Python 3.9 or higher is required")
            )
        
        if sys.version_info < (3, 13):
            self.warnings.append(
                "Python 3.13 is recommended for optimal performance"
            )
        
        # Check required packages
        required_packages = [
            "boto3",
            "sentence-transformers",
            "spacy",
            "pymilvus"
        ]
        
        for package in required_packages:
            if not self._check_package_available(package):
                self.errors.append(
                    ValidationError(f"Required package '{package}' is not installed")
                )
        
        # Check spaCy models if using default NER
        if config.ner_model.model_name.startswith("en_"):
            if not self._check_spacy_model(config.ner_model.model_name):
                self.warnings.append(
                    f"spaCy model '{config.ner_model.model_name}' may not be installed"
                )
    
    def _check_package_available(self, package_name: str) -> bool:
        """Check if a Python package is available."""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except ImportError:
            return False
    
    def _check_spacy_model(self, model_name: str) -> bool:
        """Check if a spaCy model is available."""
        try:
            import spacy
            spacy.load(model_name)
            return True
        except (ImportError, OSError):
            return False


def validate_configuration(config: SystemConfig) -> Tuple[bool, List[str]]:
    """
    Validate configuration and return results.
    
    Args:
        config: SystemConfig object to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_config(config)
    
    # Combine errors and warnings into messages
    messages = []
    
    for error in errors:
        if error.field:
            messages.append(f"ERROR [{error.field}]: {error.message}")
        else:
            messages.append(f"ERROR: {error.message}")
    
    for warning in warnings:
        messages.append(f"WARNING: {warning}")
    
    return is_valid, messages


def validate_environment_variables() -> List[str]:
    """
    Validate environment variables for configuration.
    
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check critical environment variables
    critical_vars = {
        "AWS_REGION": "us-east-1"
    }
    # Note: AWS_PROFILE is optional - ECS uses IAM roles
    
    for var_name, expected_value in critical_vars.items():
        actual_value = os.getenv(var_name)
        if actual_value != expected_value:
            messages.append(
                f"WARNING: {var_name} is '{actual_value}', expected '{expected_value}'"
            )
    
    # Check optional but recommended variables
    recommended_vars = [
        "INFERENCE_ENV",
        "LOG_LEVEL",
        "MILVUS_HOST",
        "MILVUS_PORT"
    ]
    
    for var_name in recommended_vars:
        if not os.getenv(var_name):
            messages.append(f"INFO: {var_name} not set, using default value")
    
    return messages


def check_model_compatibility(config: SystemConfig) -> List[str]:
    """
    Check compatibility between different model configurations.
    
    Args:
        config: SystemConfig object to check
        
    Returns:
        List of compatibility messages
    """
    messages = []
    
    # Check embedding dimension compatibility
    embedding_dim = config.embedding_model.embedding_dimension
    
    # Common embedding dimensions
    common_dims = [128, 256, 384, 512, 768, 1024]
    if embedding_dim not in common_dims:
        messages.append(
            f"WARNING: Unusual embedding dimension {embedding_dim}. "
            f"Common dimensions are {common_dims}"
        )
    
    # Check NER and embedding model language compatibility
    if "multilingual" not in config.embedding_model.model_name.lower():
        if config.ner_model.model_name.startswith("en_"):
            messages.append(
                "INFO: Using English NER model with non-multilingual embeddings. "
                "Consider multilingual models for better Thai support."
            )
    
    # Check LLM region compatibility with AWS region
    if config.llm_model.region != config.aws.region:
        messages.append(
            f"WARNING: LLM region ({config.llm_model.region}) differs from "
            f"AWS region ({config.aws.region}). This may cause latency issues."
        )
    
    return messages