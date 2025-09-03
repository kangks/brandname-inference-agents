# Model Accuracy Tuning Guide

This guide provides detailed procedures for improving inference accuracy for each inference approach in the Multilingual Product Inference System using the ml-sandbox AWS profile.

## Table of Contents

1. [Overview](#overview)
2. [NER Model Tuning](#ner-model-tuning)
3. [RAG System Tuning](#rag-system-tuning)
4. [LLM Fine-tuning](#llm-fine-tuning)
5. [Hybrid Agent Optimization](#hybrid-agent-optimization)
6. [Confidence Score Calibration](#confidence-score-calibration)
7. [Performance vs Accuracy Trade-offs](#performance-vs-accuracy-trade-offs)
8. [Evaluation and Validation](#evaluation-and-validation)

## Overview

The system provides multiple inference approaches, each with different tuning strategies:

- **NER Agent**: Custom entity recognition models
- **RAG Agent**: Vector similarity and retrieval optimization
- **LLM Agent**: Fine-tuned Nova Pro models via AWS Bedrock
- **Hybrid Agent**: Sequential pipeline optimization

### Prerequisites

- AWS CLI configured with `ml-sandbox` profile
- Access to `us-east-1` region
- Python 3.13 environment with all dependencies
- Training datasets in the correct format

### General Tuning Workflow

1. **Baseline Evaluation**: Establish current performance metrics
2. **Data Analysis**: Identify failure patterns and edge cases
3. **Model Tuning**: Apply specific tuning techniques
4. **Validation**: Test improvements on held-out data
5. **Deployment**: Update production models

## NER Model Tuning

### Current NER Configuration

```python
# Check current NER model configuration
from inference.config.model_registry import ModelRegistry

registry = ModelRegistry()
ner_config = registry.get_model_config('ner')
print(f"Current NER model: {ner_config}")
```

### 1. Training Data Quality Improvement

**Analyze Current Training Data**:
```bash
# Validate training data quality
python training/preprocessing/ner_data_validator.py --input training_dataset.txt --output validation_report.json

# Check entity distribution
python -c "
from training.preprocessing.ner_data_preparation import analyze_entity_distribution
analyze_entity_distribution('training_dataset.txt')
"
```

**Improve Data Quality**:
```bash
# Clean and augment training data
python training/preprocessing/ner_data_preparation.py \
  --input training_dataset.txt \
  --output cleaned_training_data.txt \
  --augment-multilingual \
  --min-confidence 0.8

# Generate additional training examples
python training/preprocessing/ner_data_preparation.py \
  --generate-variations \
  --input cleaned_training_data.txt \
  --output augmented_training_data.txt \
  --variation-count 3
```

### 2. Model Architecture Tuning

**Switch to Transformer-based NER**:
```python
# Configure transformer-based NER model
from inference.config.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model('ner', 'xlm-roberta-ner', {
    'model_name': 'xlm-roberta-base',
    'num_labels': 4,  # O, B-BRAND, I-BRAND, B-CATEGORY, I-CATEGORY
    'learning_rate': 2e-5,
    'batch_size': 16,
    'epochs': 10,
    'confidence_threshold': 0.7
})

# Switch to new model
registry.switch_model('ner', 'xlm-roberta-ner')
```

**Train Custom NER Model**:
```bash
# Train with optimized parameters
python training/pipelines/train_ner_pipeline.py \
  --model-type transformer \
  --base-model xlm-roberta-base \
  --training-data augmented_training_data.txt \
  --validation-split 0.2 \
  --learning-rate 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --early-stopping \
  --aws-profile ml-sandbox \
  --region us-east-1
```

### 3. Hyperparameter Optimization

**Grid Search for Optimal Parameters**:
```python
# Run hyperparameter search
from training.pipelines.ner_trainer import NERHyperparameterSearch

search = NERHyperparameterSearch(
    training_data='augmented_training_data.txt',
    aws_profile='ml-sandbox',
    region='us-east-1'
)

best_params = search.grid_search({
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [8, 16, 32],
    'dropout': [0.1, 0.2, 0.3],
    'weight_decay': [0.01, 0.001, 0.0001]
})

print(f"Best parameters: {best_params}")
```

### 4. Multilingual Optimization

**Thai-English Mixed Text Handling**:
```bash
# Train multilingual tokenizer
python training/preprocessing/ner_data_preparation.py \
  --create-multilingual-tokenizer \
  --languages en,th \
  --output models/multilingual_tokenizer

# Train with multilingual focus
python training/pipelines/train_ner_pipeline.py \
  --multilingual \
  --tokenizer models/multilingual_tokenizer \
  --thai-english-mixed \
  --character-level-features
```

### 5. Evaluation and Validation

**Comprehensive NER Evaluation**:
```bash
# Run NER evaluation
python training/pipelines/ner_evaluator.py \
  --model-path models/ner_model \
  --test-data test_dataset.txt \
  --output evaluation_report.json \
  --detailed-analysis

# Analyze failure cases
python training/pipelines/ner_evaluator.py \
  --analyze-failures \
  --model-path models/ner_model \
  --failure-output ner_failures.json
```

## RAG System Tuning

### 1. Embedding Model Optimization

**Test Different Embedding Models**:
```python
# Compare embedding models
from training.preprocessing.rag_data_preparation import EmbeddingModelComparison

comparison = EmbeddingModelComparison(aws_profile='ml-sandbox')
results = comparison.compare_models([
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/LaBSE'
], test_queries=['iPhone 15 Pro Max', 'ยาสีฟัน Wonder smile'])

print(f"Best embedding model: {results['best_model']}")
```

**Fine-tune Embedding Model**:
```bash
# Fine-tune sentence transformer for product names
python training/pipelines/train_rag_pipeline.py \
  --fine-tune-embeddings \
  --base-model paraphrase-multilingual-MiniLM-L12-v2 \
  --training-pairs product_brand_pairs.json \
  --epochs 5 \
  --batch-size 32 \
  --aws-profile ml-sandbox \
  --region us-east-1
```

### 2. Vector Database Optimization

**Optimize Milvus Collection**:
```python
# Configure optimal Milvus settings
from training.preprocessing.milvus_collection_manager import MilvusOptimizer

optimizer = MilvusOptimizer()
optimizer.optimize_collection('product_embeddings', {
    'index_type': 'IVF_FLAT',
    'metric_type': 'COSINE',
    'nlist': 1024,
    'nprobe': 16,
    'ef': 64
})
```

**Improve Vector Quality**:
```bash
# Clean and deduplicate vectors
python training/preprocessing/rag_data_preparation.py \
  --clean-vectors \
  --input product_embeddings.json \
  --remove-duplicates \
  --similarity-threshold 0.95 \
  --output cleaned_embeddings.json

# Populate Milvus with cleaned data
python training/pipelines/train_rag_pipeline.py \
  --populate-milvus \
  --embeddings cleaned_embeddings.json \
  --collection-name product_embeddings_v2
```

### 3. Retrieval Strategy Tuning

**Optimize Search Parameters**:
```python
# Tune retrieval parameters
from inference.agents.rag_agent import RAGAgent

agent = RAGAgent()
agent.configure_search({
    'top_k': 10,  # Increase from 5
    'similarity_threshold': 0.7,  # Lower from 0.8
    'rerank_top_k': 5,
    'use_query_expansion': True,
    'expansion_terms': 3
})
```

**Implement Query Expansion**:
```bash
# Add query expansion for better retrieval
python training/preprocessing/rag_data_preparation.py \
  --create-query-expansion \
  --input product_names.txt \
  --output query_expansions.json \
  --method synonym_augmentation
```

### 4. Multilingual Retrieval Optimization

**Cross-lingual Retrieval**:
```python
# Configure cross-lingual search
from inference.agents.rag_agent import RAGAgent

agent = RAGAgent()
agent.configure_multilingual({
    'translate_query': True,
    'target_languages': ['en', 'th'],
    'translation_service': 'aws_translate',
    'combine_results': True,
    'language_weights': {'en': 0.6, 'th': 0.4}
})
```

### 5. RAG Evaluation

**Comprehensive RAG Evaluation**:
```bash
# Evaluate RAG performance
python tests/accuracy/accuracy_validator.py \
  --agent-type rag \
  --test-queries test_queries.json \
  --ground-truth ground_truth.json \
  --metrics recall@k,precision@k,mrr \
  --output rag_evaluation.json
```

## LLM Fine-tuning

### 1. Data Preparation for Fine-tuning

**Convert Training Data to Conversation Format**:
```bash
# Use the convert_to_conversation_format function from reference notebook
python training/preprocessing/llm_data_preparation.py \
  --input training_dataset.txt \
  --output conversation_format.jsonl \
  --format bedrock_nova \
  --include-reasoning \
  --aws-profile ml-sandbox
```

**Validate Conversation Format**:
```python
# Validate conversation format data
from training.preprocessing.llm_data_preparation import validate_conversation_format

validation_results = validate_conversation_format('conversation_format.jsonl')
print(f"Validation results: {validation_results}")

# Fix any format issues
if not validation_results['valid']:
    print("Fixing format issues...")
    # Apply fixes based on validation results
```

### 2. Upload to S3 for Bedrock Fine-tuning

**Upload Training Data to S3**:
```bash
# Upload conversation format data to S3
python training/preprocessing/llm_data_preparation.py \
  --upload-to-s3 \
  --input conversation_format.jsonl \
  --bucket ml-sandbox-bedrock-training \
  --key multilingual-product-inference/training-data/conversation_format.jsonl \
  --aws-profile ml-sandbox \
  --region us-east-1

# Verify upload
aws s3 ls s3://ml-sandbox-bedrock-training/multilingual-product-inference/training-data/ --profile ml-sandbox --region us-east-1
```

### 3. Submit Bedrock Fine-tuning Job

**Create Fine-tuning Job**:
```bash
# Submit Nova Pro fine-tuning job
python training/pipelines/train_nova_pipeline.py \
  --base-model amazon.nova-pro-v1:0 \
  --training-data s3://ml-sandbox-bedrock-training/multilingual-product-inference/training-data/conversation_format.jsonl \
  --job-name multilingual-product-inference-v1 \
  --hyperparameters learning_rate=0.0001,batch_size=4,epochs=3 \
  --aws-profile ml-sandbox \
  --region us-east-1
```

**Monitor Fine-tuning Progress**:
```bash
# Monitor job status
python training/pipelines/nova_fine_tuning.py \
  --monitor-job multilingual-product-inference-v1 \
  --aws-profile ml-sandbox \
  --region us-east-1

# Get job details
aws bedrock get-model-customization-job \
  --job-identifier multilingual-product-inference-v1 \
  --profile ml-sandbox \
  --region us-east-1
```

### 4. Hyperparameter Optimization for LLM

**Experiment with Different Hyperparameters**:
```python
# Run hyperparameter experiments
from training.pipelines.nova_fine_tuning import NovaHyperparameterSearch

search = NovaHyperparameterSearch(
    training_data_s3='s3://ml-sandbox-bedrock-training/multilingual-product-inference/training-data/',
    aws_profile='ml-sandbox',
    region='us-east-1'
)

experiments = [
    {'learning_rate': 0.0001, 'batch_size': 4, 'epochs': 3},
    {'learning_rate': 0.00005, 'batch_size': 8, 'epochs': 5},
    {'learning_rate': 0.0002, 'batch_size': 2, 'epochs': 2}
]

best_config = search.run_experiments(experiments)
print(f"Best configuration: {best_config}")
```

### 5. Deploy Fine-tuned Model

**Deploy Model for Inference**:
```bash
# Deploy fine-tuned model
python training/pipelines/nova_deployment.py \
  --model-id multilingual-product-inference-v1 \
  --endpoint-name multilingual-inference-endpoint \
  --instance-type ml.g4dn.xlarge \
  --aws-profile ml-sandbox \
  --region us-east-1

# Update agent configuration
python -c "
from inference.config.model_registry import ModelRegistry
registry = ModelRegistry()
registry.update_model_config('llm', {
    'model_id': 'multilingual-product-inference-v1',
    'endpoint_name': 'multilingual-inference-endpoint'
})
"
```

### 6. LLM Evaluation

**Evaluate Fine-tuned Model**:
```bash
# Test fine-tuned model performance
python tests/accuracy/accuracy_validator.py \
  --agent-type llm \
  --model-id multilingual-product-inference-v1 \
  --test-data test_queries.json \
  --metrics accuracy,f1,confidence_calibration \
  --output llm_evaluation.json \
  --aws-profile ml-sandbox
```

## Hybrid Agent Optimization

### 1. Pipeline Configuration

**Optimize Sequential Processing**:
```python
# Configure hybrid pipeline
from inference.agents.hybrid_agent import HybridAgent

agent = HybridAgent()
agent.configure_pipeline({
    'sequence': ['ner', 'rag', 'llm'],
    'ner_weight': 0.3,
    'rag_weight': 0.4,
    'llm_weight': 0.3,
    'confidence_threshold': 0.6,
    'early_stopping': True,
    'early_stopping_threshold': 0.9
})
```

### 2. Weight Optimization

**Optimize Agent Weights**:
```bash
# Run weight optimization
python training/pipelines/hybrid_optimization.py \
  --test-data test_queries.json \
  --ground-truth ground_truth.json \
  --optimization-method bayesian \
  --iterations 100 \
  --aws-profile ml-sandbox
```

### 3. Context Passing Optimization

**Improve Context Between Stages**:
```python
# Enhanced context passing
from inference.agents.hybrid_agent import HybridAgent

agent = HybridAgent()
agent.configure_context_passing({
    'ner_to_rag': {
        'pass_entities': True,
        'entity_boost': 1.2,
        'filter_low_confidence': True
    },
    'rag_to_llm': {
        'pass_similar_products': True,
        'max_context_products': 5,
        'include_similarity_scores': True
    }
})
```

## Confidence Score Calibration

### 1. Calibration Analysis

**Analyze Current Calibration**:
```bash
# Analyze confidence calibration
python tests/accuracy/accuracy_validator.py \
  --calibration-analysis \
  --test-data test_queries.json \
  --ground-truth ground_truth.json \
  --output calibration_analysis.json
```

### 2. Calibration Improvement

**Implement Platt Scaling**:
```python
# Calibrate confidence scores
from training.pipelines.confidence_calibration import PlattScaling

calibrator = PlattScaling()
calibrator.fit(validation_predictions, validation_ground_truth)

# Apply calibration to agents
from inference.config.model_registry import ModelRegistry
registry = ModelRegistry()
registry.update_calibration('ner', calibrator.ner_calibrator)
registry.update_calibration('rag', calibrator.rag_calibrator)
registry.update_calibration('llm', calibrator.llm_calibrator)
```

## Performance vs Accuracy Trade-offs

### 1. Model Size Optimization

**Use Quantized Models**:
```python
# Configure quantized models for better performance
from inference.config.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model('ner', 'ner-quantized', {
    'model_path': 'models/ner_model_quantized.bin',
    'quantization': 'int8',
    'performance_boost': 2.5,
    'accuracy_loss': 0.02
})
```

### 2. Caching Strategies

**Implement Result Caching**:
```python
# Configure intelligent caching
from inference.config.settings import config

config.update_caching({
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hour
    'cache_size': 10000,
    'cache_similar_queries': True,
    'similarity_threshold': 0.95
})
```

### 3. Batch Processing Optimization

**Optimize Batch Sizes**:
```bash
# Find optimal batch sizes
python tests/performance/benchmark_runner.py \
  --optimize-batch-size \
  --batch-sizes 1,4,8,16,32 \
  --measure latency,throughput,memory \
  --output batch_optimization.json
```

## Evaluation and Validation

### 1. Comprehensive Evaluation Framework

**Run Full Evaluation Suite**:
```bash
# Complete evaluation across all agents
python tests/accuracy/accuracy_validator.py \
  --full-evaluation \
  --test-data test_queries.json \
  --ground-truth ground_truth.json \
  --agents ner,rag,llm,hybrid \
  --metrics accuracy,precision,recall,f1,confidence_calibration \
  --output full_evaluation_report.json \
  --aws-profile ml-sandbox
```

### 2. A/B Testing Framework

**Set up A/B Testing**:
```python
# Configure A/B testing
from tests.framework.ab_testing import ABTestFramework

ab_test = ABTestFramework()
ab_test.setup_experiment({
    'name': 'ner_model_comparison',
    'control': 'spacy_ner_v1',
    'treatment': 'xlm_roberta_ner_v1',
    'traffic_split': 0.5,
    'success_metric': 'accuracy',
    'minimum_sample_size': 1000
})

# Run experiment
results = ab_test.run_experiment(duration_days=7)
print(f"A/B test results: {results}")
```

### 3. Continuous Monitoring

**Set up Performance Monitoring**:
```bash
# Configure continuous monitoring
python inference/monitoring/accuracy_monitor.py \
  --setup-monitoring \
  --cloudwatch-namespace MultilingualInference \
  --metrics accuracy,latency,confidence \
  --alert-thresholds accuracy:0.85,latency:2000 \
  --aws-profile ml-sandbox \
  --region us-east-1
```

### 4. Model Drift Detection

**Monitor Model Drift**:
```python
# Set up drift detection
from inference.monitoring.drift_detection import ModelDriftDetector

detector = ModelDriftDetector()
detector.configure({
    'baseline_data': 'baseline_predictions.json',
    'drift_threshold': 0.05,
    'monitoring_window': '7d',
    'alert_on_drift': True
})

# Check for drift
drift_report = detector.check_drift(current_predictions)
if drift_report['drift_detected']:
    print("Model drift detected! Consider retraining.")
```

## Best Practices Summary

### 1. Data Quality
- Ensure high-quality, diverse training data
- Regular data validation and cleaning
- Balanced representation of languages and product types

### 2. Iterative Improvement
- Start with baseline models
- Make incremental improvements
- Validate each change thoroughly

### 3. Monitoring and Alerting
- Continuous performance monitoring
- Automated drift detection
- Regular model evaluation

### 4. Documentation
- Document all tuning experiments
- Track model versions and configurations
- Maintain evaluation results history

### 5. AWS Best Practices
- Use ml-sandbox profile for all operations
- Deploy in us-east-1 region consistently
- Monitor AWS costs and usage
- Follow AWS security best practices

This comprehensive tuning guide provides the foundation for systematically improving model accuracy across all inference approaches while maintaining the required AWS configuration and regional constraints.