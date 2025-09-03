"""
Model registry for managing and switching between different AI models.

This module provides a centralized registry for NER, embedding, and LLM models
with validation and switching capabilities, following PEP 8 standards.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .settings import (
    ModelType,
    NERModelConfig,
    EmbeddingModelConfig,
    LLMModelConfig,
    SystemConfig
)


class ModelStatus(Enum):
    """Model status enumeration."""
    
    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    
    name: str
    model_type: ModelType
    description: str
    version: str
    status: ModelStatus = ModelStatus.AVAILABLE
    file_path: Optional[str] = None
    download_url: Optional[str] = None
    size_mb: Optional[float] = None
    accuracy_metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Validate model metadata."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if not self.version:
            raise ValueError("Model version cannot be empty")


class ModelRegistry:
    """
    Centralized registry for managing AI models with validation and switching.
    
    Supports NER, embedding, and LLM models with metadata tracking,
    validation, and easy switching between configurations.
    """
    
    def __init__(self, registry_path: Optional[str] = None) -> None:
        """
        Initialize model registry.
        
        Args:
            registry_path: Optional path to persistent registry file
        """
        self.logger = logging.getLogger(__name__)
        self._registry_path = registry_path or "models/registry.json"
        self._models: Dict[str, Dict[str, ModelMetadata]] = {
            "ner": {},
            "embedding": {},
            "llm": {}
        }
        self._active_models: Dict[str, str] = {
            "ner": "",
            "embedding": "",
            "llm": ""
        }
        
        # Load existing registry if available
        self._load_registry()
        
        # Register default models
        self._register_default_models()
    
    def register_ner_model(
        self,
        name: str,
        model_path: str,
        description: str = "",
        version: str = "1.0.0",
        **kwargs: Any
    ) -> None:
        """
        Register a NER model in the registry.
        
        Args:
            name: Unique model name
            model_path: Path to model files or model identifier
            description: Model description
            version: Model version
            **kwargs: Additional model configuration parameters
        """
        # Create model configuration
        config = NERModelConfig(
            model_name=name,
            model_path=model_path,
            **{k: v for k, v in kwargs.items() if k in NERModelConfig.__dataclass_fields__}
        )
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            model_type=ModelType.NER,
            description=description,
            version=version,
            file_path=model_path,
            **{k: v for k, v in kwargs.items() if k in ModelMetadata.__dataclass_fields__}
        )
        
        # Validate model exists if it's a file path
        if model_path and Path(model_path).exists():
            metadata.status = ModelStatus.AVAILABLE
        elif model_path and not model_path.startswith(("http://", "https://", "en_")):
            # Only set to error if it's a local path that doesn't exist
            # Don't error for spaCy model names or URLs
            self.logger.warning(f"Model path does not exist: {model_path}")
            metadata.status = ModelStatus.ERROR
        else:
            # For spaCy models, URLs, or empty paths, assume available
            metadata.status = ModelStatus.AVAILABLE
        
        # Register model
        self._models["ner"][name] = metadata
        self.logger.info(f"Registered NER model: {name} (version {version})")
        
        # Save registry
        self._save_registry()
    
    def register_embedding_model(
        self,
        name: str,
        model_name: str,
        description: str = "",
        version: str = "1.0.0",
        **kwargs: Any
    ) -> None:
        """
        Register an embedding model in the registry.
        
        Args:
            name: Unique model name for registry
            model_name: Model identifier (e.g., HuggingFace model name)
            description: Model description
            version: Model version
            **kwargs: Additional model configuration parameters
        """
        # Create model configuration
        config = EmbeddingModelConfig(
            model_name=model_name,
            **{k: v for k, v in kwargs.items() if k in EmbeddingModelConfig.__dataclass_fields__}
        )
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            model_type=ModelType.EMBEDDING,
            description=description,
            version=version,
            **{k: v for k, v in kwargs.items() if k in ModelMetadata.__dataclass_fields__}
        )
        
        # Register model
        self._models["embedding"][name] = metadata
        self.logger.info(f"Registered embedding model: {name} (version {version})")
        
        # Save registry
        self._save_registry()
    
    def register_llm_model(
        self,
        name: str,
        model_id: str,
        description: str = "",
        version: str = "1.0.0",
        **kwargs: Any
    ) -> None:
        """
        Register an LLM model in the registry.
        
        Args:
            name: Unique model name for registry
            model_id: Model identifier (e.g., AWS Bedrock model ID)
            description: Model description
            version: Model version
            **kwargs: Additional model configuration parameters
        """
        # Create model configuration
        config = LLMModelConfig(
            model_id=model_id,
            **{k: v for k, v in kwargs.items() if k in LLMModelConfig.__dataclass_fields__}
        )
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            model_type=ModelType.LLM,
            description=description,
            version=version,
            **{k: v for k, v in kwargs.items() if k in ModelMetadata.__dataclass_fields__}
        )
        
        # Register model
        self._models["llm"][name] = metadata
        self.logger.info(f"Registered LLM model: {name} (version {version})")
        
        # Save registry
        self._save_registry()
    
    def switch_model(self, model_type: str, model_name: str) -> None:
        """
        Switch to a different model configuration.
        
        Args:
            model_type: Type of model ('ner', 'embedding', 'llm')
            model_name: Name of the registered model
            
        Raises:
            ValueError: If model type or name is invalid
        """
        # Validate model type
        if model_type not in self._models:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Validate model exists
        if model_name not in self._models[model_type]:
            available_models = list(self._models[model_type].keys())
            raise ValueError(
                f"Model '{model_name}' not found for type '{model_type}'. "
                f"Available models: {available_models}"
            )
        
        # Check model status
        metadata = self._models[model_type][model_name]
        if metadata.status == ModelStatus.ERROR:
            self.logger.warning(f"Switching to model '{model_name}' which is in error state")
        
        # Switch active model
        self._active_models[model_type] = model_name
        self.logger.info(f"Switched {model_type} model to: {model_name}")
        
        # Save registry
        self._save_registry()
    
    def get_active_model(self, model_type: str) -> Optional[str]:
        """
        Get the currently active model for a given type.
        
        Args:
            model_type: Type of model to query
            
        Returns:
            Name of active model or None if no model is active
        """
        return self._active_models.get(model_type)
    
    def get_model_metadata(self, model_type: str, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            ModelMetadata object or None if not found
        """
        return self._models.get(model_type, {}).get(model_name)
    
    def list_models(self, model_type: str) -> List[str]:
        """
        List all registered models for a given type.
        
        Args:
            model_type: Type of model to list
            
        Returns:
            List of model names
        """
        return list(self._models.get(model_type, {}).keys())
    
    def list_all_models(self) -> Dict[str, List[str]]:
        """
        List all registered models grouped by type.
        
        Returns:
            Dictionary mapping model types to lists of model names
        """
        return {
            model_type: list(models.keys())
            for model_type, models in self._models.items()
        }
    
    def get_model_config(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration for integration with ConfigManager.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary or None if not found
        """
        metadata = self.get_model_metadata(model_type, model_name)
        if not metadata:
            return None
        
        # Convert metadata to configuration format
        if model_type == "ner":
            return {
                "model_name": metadata.name,
                "model_path": metadata.file_path,
            }
        elif model_type == "embedding":
            return {
                "model_name": metadata.name,
                "model_path": metadata.file_path,
            }
        elif model_type == "llm":
            return {
                "model_id": metadata.name,
            }
        
        return None
    
    def update_model_status(self, model_type: str, model_name: str, status: ModelStatus) -> None:
        """
        Update the status of a registered model.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            status: New status
        """
        if model_type in self._models and model_name in self._models[model_type]:
            self._models[model_type][model_name].status = status
            self.logger.info(f"Updated {model_type} model '{model_name}' status to {status.value}")
            self._save_registry()
    
    def remove_model(self, model_type: str, model_name: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            True if model was removed, False if not found
        """
        if model_type in self._models and model_name in self._models[model_type]:
            del self._models[model_type][model_name]
            
            # Clear active model if it was the removed one
            if self._active_models.get(model_type) == model_name:
                self._active_models[model_type] = ""
            
            self.logger.info(f"Removed {model_type} model: {model_name}")
            self._save_registry()
            return True
        
        return False
    
    def _register_default_models(self) -> None:
        """Register default models for each type."""
        # Default NER model
        if "default" not in self._models["ner"]:
            self.register_ner_model(
                name="default",
                model_path="en_core_web_sm",
                description="Default spaCy English NER model",
                version="3.7.0",
                confidence_threshold=0.5,
                custom_entities=["BRAND", "CATEGORY", "VARIANT"]
            )
        
        # Default embedding model
        if "default" not in self._models["embedding"]:
            self.register_embedding_model(
                name="default",
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                description="Default multilingual sentence transformer",
                version="1.0.0",
                embedding_dimension=384,
                batch_size=32
            )
        
        # Default LLM model
        if "default" not in self._models["llm"]:
            self.register_llm_model(
                name="default",
                model_id="amazon.nova-pro-v1:0",
                description="Default AWS Nova Pro model",
                version="1.0.0",
                region="us-east-1",
                max_tokens=1000,
                temperature=0.1
            )
        
        # Set active models to default if not already set
        for model_type in ["ner", "embedding", "llm"]:
            if not self._active_models.get(model_type) and "default" in self._models[model_type]:
                self._active_models[model_type] = "default"
    
    def _load_registry(self) -> None:
        """Load registry from persistent storage."""
        registry_path = Path(self._registry_path)
        
        if not registry_path.exists():
            self.logger.info("No existing registry found, starting with empty registry")
            return
        
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load models
            for model_type, models in data.get("models", {}).items():
                for model_name, model_data in models.items():
                    metadata = ModelMetadata(
                        name=model_data["name"],
                        model_type=ModelType(model_data["model_type"]),
                        description=model_data.get("description", ""),
                        version=model_data.get("version", "1.0.0"),
                        status=ModelStatus(model_data.get("status", "available")),
                        file_path=model_data.get("file_path"),
                        download_url=model_data.get("download_url"),
                        size_mb=model_data.get("size_mb"),
                        accuracy_metrics=model_data.get("accuracy_metrics"),
                        tags=model_data.get("tags")
                    )
                    self._models[model_type][model_name] = metadata
            
            # Load active models
            self._active_models.update(data.get("active_models", {}))
            
            self.logger.info(f"Loaded registry from {registry_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry from {registry_path}: {str(e)}")
    
    def _save_registry(self) -> None:
        """Save registry to persistent storage."""
        registry_path = Path(self._registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert models to serializable format
            models_data = {}
            for model_type, models in self._models.items():
                models_data[model_type] = {}
                for model_name, metadata in models.items():
                    models_data[model_type][model_name] = asdict(metadata)
                    # Convert enum to string
                    models_data[model_type][model_name]["model_type"] = metadata.model_type.value
                    models_data[model_type][model_name]["status"] = metadata.status.value
            
            data = {
                "models": models_data,
                "active_models": self._active_models,
                "version": "1.0.0"
            }
            
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved registry to {registry_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save registry to {registry_path}: {str(e)}")


# Global model registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return model_registry