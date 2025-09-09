"""
Configuration management system for multilingual product inference.

This module provides environment-based configuration management with model switching
capabilities, following PEP 8 standards and supporting AWS ml-sandbox profile.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import logging
from pathlib import Path


class Environment(Enum):
    """Deployment environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelType(Enum):
    """Available model types for configuration."""
    
    NER = "ner"
    EMBEDDING = "embedding"
    LLM = "llm"


@dataclass
class NERModelConfig:
    """Configuration for NER models."""
    
    model_name: str = "en_core_web_sm"
    model_path: Optional[str] = None
    custom_entities: List[str] = field(default_factory=lambda: ["BRAND", "CATEGORY", "VARIANT"])
    confidence_threshold: float = 0.5
    max_entities: int = 10
    
    def __post_init__(self) -> None:
        """Validate NER model configuration."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.max_entities <= 0:
            raise ValueError("Max entities must be positive")


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_path: Optional[str] = None
    embedding_dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"
    
    def __post_init__(self) -> None:
        """Validate embedding model configuration."""
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class LLMModelConfig:
    """Configuration for LLM models."""
    
    model_id: str = "amazon.nova-pro-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 1000
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_seconds: int = 30
    
    def __post_init__(self) -> None:
        """Validate LLM model configuration."""
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class MilvusConfig:
    """Configuration for Milvus vector database."""
    
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "product_embeddings"
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024
    top_k: int = 5
    
    def __post_init__(self) -> None:
        """Validate Milvus configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")


@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    
    profile_name: str = "ml-sandbox"
    region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    bedrock_region: str = "us-east-1"
    
    def __post_init__(self) -> None:
        """Validate AWS configuration."""
        if not self.profile_name:
            raise ValueError("AWS profile name cannot be empty")
        if not self.region:
            raise ValueError("AWS region cannot be empty")


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    
    timeout_seconds: int = 30
    max_concurrent_requests: int = 10
    confidence_threshold: float = 0.5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate agent configuration."""
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts cannot be negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")


@dataclass
class SystemConfig:
    """Main system configuration."""
    
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    ner_model: NERModelConfig = field(default_factory=NERModelConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    llm_model: LLMModelConfig = field(default_factory=LLMModelConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    def __post_init__(self) -> None:
        """Validate system configuration."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")


class ConfigManager:
    """Configuration manager with environment-based loading and model switching."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self._config: Optional[SystemConfig] = None
        self._config_path = config_path
        self._model_registry: Dict[str, Dict[str, Any]] = {
            "ner": {},
            "embedding": {},
            "llm": {}
        }
        # Import here to avoid circular imports
        from .model_registry import get_model_registry
        self._registry = get_model_registry()
        
    def load_config(self) -> SystemConfig:
        """
        Load configuration from environment variables and files.
        
        Returns:
            SystemConfig object with loaded configuration
        """
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config = SystemConfig()
        
        # Override with environment variables
        config = self._load_from_environment(config)
        
        # Override with configuration file if provided
        if self._config_path and Path(self._config_path).exists():
            config = self._load_from_file(config, self._config_path)
        
        # Validate final configuration
        self._validate_config(config)
        
        self._config = config
        self.logger.info(f"Configuration loaded for environment: {config.environment.value}")
        
        return config
    
    def _load_from_environment(self, config: SystemConfig) -> SystemConfig:
        """Load configuration values from environment variables."""
        # Environment
        env_name = os.getenv("INFERENCE_ENV", "development")
        try:
            config.environment = Environment(env_name)
        except ValueError:
            self.logger.warning(f"Invalid environment '{env_name}', using development")
        
        # Logging
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        
        # NER Model
        config.ner_model.model_name = os.getenv("NER_MODEL_NAME", config.ner_model.model_name)
        config.ner_model.model_path = os.getenv("NER_MODEL_PATH")
        
        # Embedding Model
        config.embedding_model.model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", 
            config.embedding_model.model_name
        )
        config.embedding_model.device = os.getenv("EMBEDDING_DEVICE", config.embedding_model.device)
        
        # LLM Model
        config.llm_model.model_id = os.getenv("LLM_MODEL_ID", config.llm_model.model_id)
        config.llm_model.region = os.getenv("AWS_REGION", config.llm_model.region)
        
        # Milvus
        config.milvus.host = os.getenv("MILVUS_HOST", config.milvus.host)
        config.milvus.port = int(os.getenv("MILVUS_PORT", str(config.milvus.port)))
        config.milvus.collection_name = os.getenv("MILVUS_COLLECTION", config.milvus.collection_name)
        
        # AWS
        config.aws.profile_name = os.getenv("AWS_PROFILE", config.aws.profile_name)
        config.aws.region = os.getenv("AWS_REGION", config.aws.region)
        config.aws.s3_bucket = os.getenv("S3_BUCKET")
        
        # Agent settings
        if os.getenv("AGENT_TIMEOUT"):
            config.agent.timeout_seconds = int(os.getenv("AGENT_TIMEOUT"))
        if os.getenv("CONFIDENCE_THRESHOLD"):
            config.agent.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
        
        return config
    
    def _load_from_file(self, config: SystemConfig, file_path: str) -> SystemConfig:
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Update configuration with file values
            # This is a simplified implementation - in practice, you'd want
            # more sophisticated merging logic
            self.logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {str(e)}")
        
        return config
    
    def _validate_config(self, config: SystemConfig) -> None:
        """Validate configuration values."""
        # Validate AWS profile exists
        try:
            import boto3
            session = boto3.Session(profile_name=config.aws.profile_name)
            # Test if we can get credentials
            session.get_credentials()
        except Exception as e:
            self.logger.warning(f"AWS profile validation failed: {str(e)}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def register_model(self, model_type: ModelType, name: str, config: Dict[str, Any]) -> None:
        """
        Register a model configuration for switching using the model registry.
        
        Args:
            model_type: Type of model (NER, embedding, LLM)
            name: Model name identifier
            config: Model configuration dictionary
        """
        # Register with both old registry (for backward compatibility) and new registry
        self._model_registry[model_type.value][name] = config
        
        # Register with the new model registry
        if model_type == ModelType.NER:
            self._registry.register_ner_model(
                name=name,
                model_path=config.get("model_path", config.get("model_name", "")),
                description=config.get("description", f"Custom {name} NER model"),
                version=config.get("version", "1.0.0"),
                **{k: v for k, v in config.items() if k not in ["model_path", "model_name", "description", "version"]}
            )
        elif model_type == ModelType.EMBEDDING:
            self._registry.register_embedding_model(
                name=name,
                model_name=config.get("model_name", name),
                description=config.get("description", f"Custom {name} embedding model"),
                version=config.get("version", "1.0.0"),
                **{k: v for k, v in config.items() if k not in ["model_name", "description", "version"]}
            )
        elif model_type == ModelType.LLM:
            self._registry.register_llm_model(
                name=name,
                model_id=config.get("model_id", config.get("model_name", name)),
                description=config.get("description", f"Custom {name} LLM model"),
                version=config.get("version", "1.0.0"),
                **{k: v for k, v in config.items() if k not in ["model_id", "model_name", "description", "version"]}
            )
        
        self.logger.info(f"Registered {model_type.value} model: {name}")
    
    def switch_model(self, model_type: ModelType, name: str) -> None:
        """
        Switch to a different model configuration using the model registry.
        
        Args:
            model_type: Type of model to switch
            name: Name of the registered model
        """
        # Use model registry for switching
        self._registry.switch_model(model_type.value, name)
        
        # Get model configuration from registry
        model_config = self._registry.get_model_config(model_type.value, name)
        if not model_config:
            raise ValueError(f"Model configuration not found for '{name}'")
        
        config = self.get_config()
        
        # Update configuration based on model type
        if model_type == ModelType.NER:
            # Merge with existing config to preserve other settings
            current_config = asdict(config.ner_model)
            current_config.update(model_config)
            config.ner_model = NERModelConfig(**current_config)
        elif model_type == ModelType.EMBEDDING:
            current_config = asdict(config.embedding_model)
            current_config.update(model_config)
            config.embedding_model = EmbeddingModelConfig(**current_config)
        elif model_type == ModelType.LLM:
            current_config = asdict(config.llm_model)
            current_config.update(model_config)
            config.llm_model = LLMModelConfig(**current_config)
        
        self.logger.info(f"Switched {model_type.value} model to: {name}")
    
    def list_models(self, model_type: ModelType) -> List[str]:
        """
        List registered models for a given type using the model registry.
        
        Args:
            model_type: Type of model to list
            
        Returns:
            List of registered model names
        """
        return self._registry.list_models(model_type.value)
    
    def get_model_registry(self):
        """
        Get access to the underlying model registry.
        
        Returns:
            ModelRegistry instance for advanced model management
        """
        return self._registry


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def setup_logging(config: SystemConfig) -> None:
    """Setup logging configuration based on system config."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific logger levels
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)