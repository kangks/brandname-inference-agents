# Configuration Management System

This module provides comprehensive configuration management for the multilingual product inference system, following PEP 8 standards and supporting Python 3.13 features.

## Features

- **Environment-based Configuration**: Automatic loading from environment variables and configuration files
- **Model Registry**: Centralized management of NER, embedding, and LLM models with metadata tracking
- **Configuration Validation**: Comprehensive validation with detailed error reporting
- **Model Switching**: Easy switching between different model configurations
- **AWS Integration**: Built-in support for AWS ml-sandbox profile and us-east-1 region
- **Type Safety**: Full type hints and dataclass validation

## Quick Start

```python
from inference.config import get_config, get_model_registry, setup_logging

# Load configuration
config = get_config()
setup_logging(config)

# Access configuration
print(f"Environment: {config.environment.value}")
print(f"NER Model: {config.ner_model.model_name}")
print(f"AWS Profile: {config.aws.profile_name}")

# Use model registry
registry = get_model_registry()
models = registry.list_models("ner")
print(f"Available NER models: {models}")
```

## Configuration Structure

### SystemConfig

The main configuration class containing all system settings:

```python
@dataclass
class SystemConfig:
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    ner_model: NERModelConfig = field(default_factory=NERModelConfig)
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    llm_model: LLMModelConfig = field(default_factory=LLMModelConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
```

### Model Configurations

#### NER Model Configuration
```python
@dataclass
class NERModelConfig:
    model_name: str = "en_core_web_sm"
    model_path: Optional[str] = None
    custom_entities: List[str] = field(default_factory=lambda: ["BRAND", "CATEGORY", "VARIANT"])
    confidence_threshold: float = 0.5
    max_entities: int = 10
```

#### Embedding Model Configuration
```python
@dataclass
class EmbeddingModelConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_path: Optional[str] = None
    embedding_dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"
```

#### LLM Model Configuration
```python
@dataclass
class LLMModelConfig:
    model_id: str = "amazon.nova-pro-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 1000
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_seconds: int = 30
```

## Environment Variables

The system automatically loads configuration from environment variables:

### Core Settings
- `INFERENCE_ENV`: Environment (development, testing, staging, production)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### AWS Settings
- `AWS_PROFILE`: AWS profile name (default: ml-sandbox)
- `AWS_REGION`: AWS region (default: us-east-1)
- `S3_BUCKET`: S3 bucket for model artifacts

### Model Settings
- `NER_MODEL_NAME`: NER model name
- `NER_MODEL_PATH`: Path to NER model files
- `EMBEDDING_MODEL_NAME`: Embedding model name
- `EMBEDDING_DEVICE`: Device for embedding model (cpu, cuda)
- `LLM_MODEL_ID`: LLM model identifier

### Infrastructure Settings
- `MILVUS_HOST`: Milvus database host
- `MILVUS_PORT`: Milvus database port
- `MILVUS_COLLECTION`: Milvus collection name

### Agent Settings
- `AGENT_TIMEOUT`: Agent timeout in seconds
- `CONFIDENCE_THRESHOLD`: Confidence threshold for predictions

## Model Registry

The model registry provides centralized management of AI models with metadata tracking and easy switching.

### Registering Models

```python
from inference.config import get_model_registry

registry = get_model_registry()

# Register NER model
registry.register_ner_model(
    name="custom_ner",
    model_path="/path/to/model",
    description="Custom NER model for Thai-English text",
    version="1.0.0",
    confidence_threshold=0.7
)

# Register embedding model
registry.register_embedding_model(
    name="thai_embedding",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    description="Thai-optimized embedding model",
    version="1.0.0",
    embedding_dimension=768
)

# Register LLM model
registry.register_llm_model(
    name="nova_lite",
    model_id="amazon.nova-lite-v1:0",
    description="Nova Lite for faster inference",
    version="1.0.0",
    temperature=0.2
)
```

### Switching Models

```python
# Switch to different models
registry.switch_model("ner", "custom_ner")
registry.switch_model("embedding", "thai_embedding")
registry.switch_model("llm", "nova_lite")

# Check active models
active_ner = registry.get_active_model("ner")
print(f"Active NER model: {active_ner}")
```

### Model Metadata

```python
# Get model metadata
metadata = registry.get_model_metadata("ner", "custom_ner")
print(f"Model: {metadata.name}")
print(f"Version: {metadata.version}")
print(f"Description: {metadata.description}")
print(f"Status: {metadata.status.value}")

# List all models
all_models = registry.list_all_models()
for model_type, models in all_models.items():
    print(f"{model_type}: {models}")
```

## Configuration Validation

The system provides comprehensive validation with detailed error reporting:

```python
from inference.config import validate_configuration, validate_environment_variables

# Validate configuration
config = get_config()
is_valid, messages = validate_configuration(config)

if not is_valid:
    for message in messages:
        print(message)

# Validate environment variables
env_messages = validate_environment_variables()
for message in env_messages:
    print(message)
```

### Validation Features

- **Type Validation**: Ensures all configuration values are of correct types
- **Range Validation**: Validates numeric ranges (e.g., confidence thresholds 0-1)
- **Dependency Checking**: Verifies required packages are installed
- **AWS Validation**: Checks AWS credentials and profile availability
- **Model Compatibility**: Validates model configurations work together
- **Environment-Specific Rules**: Different validation rules for different environments

## Error Handling

The configuration system provides detailed error handling:

```python
from inference.config import ValidationError

try:
    registry.switch_model("ner", "nonexistent_model")
except ValueError as e:
    print(f"Model switching error: {e}")

try:
    registry.register_ner_model("", "/invalid/path", "Invalid model")
except ValidationError as e:
    print(f"Validation error in field '{e.field}': {e.message}")
```

## Integration with ConfigManager

The ConfigManager integrates with the ModelRegistry for seamless model switching:

```python
from inference.config import ConfigManager, ModelType

manager = ConfigManager()

# Register model through ConfigManager
ner_config = {
    "model_name": "custom_ner",
    "confidence_threshold": 0.8
}
manager.register_model(ModelType.NER, "custom", ner_config)

# Switch model
manager.switch_model(ModelType.NER, "custom")

# Access updated configuration
config = manager.get_config()
print(f"Current NER model: {config.ner_model.model_name}")
```

## Development Environment Setup

### Python 3.13 Virtual Environment

```bash
# Create virtual environment
python3.13 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### AWS Configuration

```bash
# Configure AWS profile
aws configure --profile ml-sandbox

# Set environment variables
export AWS_PROFILE=ml-sandbox
export AWS_REGION=us-east-1
```

## Testing

The configuration system includes comprehensive tests:

```bash
# Run all configuration tests
python -m pytest tests/test_config.py tests/test_model_registry.py tests/test_config_validators.py -v

# Run specific test categories
python -m pytest tests/test_config.py -v  # Core configuration tests
python -m pytest tests/test_model_registry.py -v  # Model registry tests
python -m pytest tests/test_config_validators.py -v  # Validation tests
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end configuration workflows
- **Validation Tests**: Error handling and edge cases
- **Mock Tests**: AWS and external service mocking

## Best Practices

### Configuration Management

1. **Use Environment Variables**: Set configuration through environment variables for different deployment stages
2. **Validate Early**: Always validate configuration at startup
3. **Handle Errors Gracefully**: Provide meaningful error messages for configuration issues
4. **Document Changes**: Update configuration documentation when adding new settings

### Model Management

1. **Version Models**: Always specify model versions for reproducibility
2. **Test Model Switches**: Validate model compatibility before switching in production
3. **Monitor Model Status**: Check model status before using in inference
4. **Backup Configurations**: Save model registry to persistent storage

### Development

1. **Follow PEP 8**: All code follows PEP 8 coding standards
2. **Use Type Hints**: Comprehensive type annotations for better IDE support
3. **Write Tests**: Test all configuration changes and model operations
4. **Document APIs**: Provide clear docstrings for all public methods

## Troubleshooting

### Common Issues

#### Configuration Loading Errors
```python
# Check if configuration is valid
is_valid, messages = validate_configuration(config)
if not is_valid:
    for message in messages:
        print(f"Config issue: {message}")
```

#### Model Registry Issues
```python
# Check model status
metadata = registry.get_model_metadata("ner", "model_name")
if metadata and metadata.status == ModelStatus.ERROR:
    print(f"Model {metadata.name} is in error state")
```

#### AWS Credential Issues
```python
# Validate AWS configuration
env_messages = validate_environment_variables()
aws_messages = [msg for msg in env_messages if "AWS" in msg]
for message in aws_messages:
    print(f"AWS issue: {message}")
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
# export LOG_LEVEL=DEBUG
```

## API Reference

See the individual module documentation for detailed API reference:

- [`settings.py`](settings.py): Core configuration classes and management
- [`model_registry.py`](model_registry.py): Model registry and metadata management
- [`validators.py`](validators.py): Configuration validation utilities

## Examples

See [`examples/config_demo.py`](../../examples/config_demo.py) for a comprehensive demonstration of the configuration system features.