# Implementation Plan

**Reference Implementation**: Use `brand_extraction_ner_rag_llm_rev2.ipynb` as the primary reference for implementation patterns, data structures, and API usage examples throughout all tasks.

- [x] 1. Setup project structure and core interfaces
  - Create directory structure for inference/ and training/ folders with modular organization following PEP 8 standards
  - Define core data models and interfaces for all agent types (reference notebook's data structures) with proper type hints
  - Setup base configuration management system for model switching and environment settings
  - Configure development environment with Python 3.13 in .venv virtual environment, PEP 8 linting tools (black, flake8), and AWS ml-sandbox profile
  - _Requirements: 1.1, 7.1, 7.6, 9.1, 9.7_
  - _Reference: notebook sections 1-2 for project setup and imports_

- [x] 2. Implement individual inference agents
  - [x] 2.1 Create NER Agent implementation
    - Implement spaCy-based NER agent with custom entity recognition following PEP 8 coding standards (follow notebook's NER section patterns)
    - Add multilingual support for Thai-English mixed text processing with proper type annotations
    - Create confidence scoring mechanism for entity predictions with comprehensive docstrings
    - Write unit tests for NER agent functionality following PEP 8 test conventions
    - _Requirements: 1.1, 7.1, 8.4, 8.5, 9.7_
    - _Reference: notebook NER section for spaCy implementation patterns and entity extraction logic_

  - [x] 2.2 Implement RAG Agent with vector similarity search
    - Create sentence transformer embedding generation module following PEP 8 standards (use notebook's SentenceTransformer patterns)
    - Implement Milvus vector database query and retrieval logic with proper error handling (reference notebook's Milvus client usage)
    - Add similarity-based brand inference with confidence scoring and type hints
    - Write unit tests for RAG agent retrieval and inference following PEP 8 conventions
    - _Requirements: 1.2, 7.2, 8.3, 9.7_
    - _Reference: notebook RAG workflow section for Milvus search patterns and embedding generation_

  - [x] 2.3 Create LLM Agent using AWS Bedrock
    - Implement Nova Pro inference client using AWS Bedrock SDK with PEP 8 compliant code (reference notebook's boto3 usage patterns)
    - Create prompt engineering for multilingual brand extraction with proper documentation (use notebook's prompt examples)
    - Add confidence scoring based on LLM response analysis with type annotations
    - Write unit tests for LLM agent inference functionality using ml-sandbox AWS profile
    - _Requirements: 1.3, 7.3, 8.2, 9.2, 9.7_
    - _Reference: notebook LLM sections for AWS Bedrock integration and prompt engineering patterns_

  - [x] 2.4 Implement Hybrid Agent with sequential processing
    - Create sequential pipeline: NER → RAG → LLM processing following PEP 8 standards (follow notebook's hybrid approach patterns)
    - Implement result combination and confidence aggregation logic with comprehensive docstrings
    - Add enhanced context passing between pipeline stages with proper type hints
    - Write unit tests for hybrid agent workflow following PEP 8 test conventions
    - _Requirements: 1.2, 7.4, 8.6, 9.7_
    - _Reference: notebook hybrid sections for pipeline coordination and result combination strategies_

- [x] 3. Create orchestrator agent with strands-agents SDK
  - Implement orchestrator agent using strands-agents framework following PEP 8 coding standards
  - Add parallel execution coordination for multiple inference agents with proper type annotations
  - Create result aggregation and confidence-based ranking system with comprehensive documentation
  - Implement error handling and timeout management for agent coordination using AWS ml-sandbox profile
  - Write integration tests for orchestrator agent functionality following PEP 8 conventions
  - _Requirements: 1.4, 2.1, 2.2, 7.5, 9.2, 9.7_

- [x] 4. Setup AWS infrastructure components
  - [x] 4.1 Create ECS Fargate deployment configurations for ARM64 platform
    - Write Dockerfile for each agent type with ARM64 platform specification, Python 3.13, and health checks
    - Create ECS task definitions with ARM64 CPU architecture and service configurations for us-east-1 region
    - Implement auto-scaling policies for agent services using ml-sandbox AWS profile
    - _Requirements: 2.3, 6.1, 6.3, 9.2, 9.3, 9.6_

  - [x] 4.2 Deploy Milvus vector database on ECS with ARM64 platform
    - Create Milvus container deployment configuration for ARM64 architecture in us-east-1 region
    - Setup persistent storage and backup strategies using ml-sandbox AWS profile
    - Implement health monitoring and restart policies for ARM64 ECS tasks
    - _Requirements: 6.2, 6.6, 9.2, 9.3, 9.6_

  - [x] 4.3 Create CloudFormation infrastructure templates for ARM64 deployment
    - Write CloudFormation templates for complete AWS stack deployment in us-east-1 region with ARM64 ECS tasks
    - Include Application Load Balancer and API Gateway configuration using ml-sandbox AWS profile
    - Add CloudWatch logging and monitoring setup for ARM64 platform services
    - _Requirements: 6.3, 6.4, 9.2, 9.3, 9.6_

- [x] 5. Implement training data preprocessing pipeline
  - [x] 5.1 Create NER training data preparation scripts
    - Write PEP 8 compliant scripts to convert product title-brand pairs into NER training format (use notebook's training data structure)
    - Implement data augmentation for multilingual Thai-English mixed text with proper type annotations
    - Create validation scripts for NER training data quality following Python 3.13 best practices in .venv environment
    - _Requirements: 3.1, 8.1, 8.5, 9.1, 9.7_
    - _Reference: notebook training data section for data format and preprocessing patterns_

  - [x] 5.2 Implement RAG knowledge base preprocessing
    - Write scripts to generate embeddings for product titles using sentence transformers (reference notebook's embedding generation)
    - Create Milvus collection setup and data ingestion pipeline (use notebook's Milvus client patterns)
    - Implement data validation and duplicate detection for vector database
    - _Requirements: 3.3, 8.3, 8.4_
    - _Reference: notebook RAG preprocessing sections for embedding generation and Milvus collection setup_

  - [x] 5.3 Create LLM fine-tuning data preparation
    - Write PEP 8 compliant scripts to convert training dataset using convert_to_conversation_format() function from reference notebook
    - Implement S3 upload functionality for converted fine-tuning datasets using ml-sandbox AWS profile (reference notebook's AWS integration)
    - Create AWS Bedrock fine-tuning job submission scripts for us-east-1 region that reference S3 dataset location
    - Add data validation and quality checks for conversation format datasets with proper error handling
    - _Requirements: 3.4, 8.2, 8.1, 9.2, 9.3, 9.7_
    - _Reference: notebook fine-tuning sections for convert_to_conversation_format() implementation and AWS Bedrock job patterns_

- [x] 6. Implement training pipeline execution
  - [x] 6.1 Create NER model training scripts
    - Implement spaCy custom NER model training pipeline (reference notebook's NER training approach)
    - Add model evaluation and validation metrics
    - Create model artifact storage and versioning system
    - _Requirements: 3.2, 5.1_
    - _Reference: notebook NER training sections for spaCy model training patterns and evaluation metrics_

  - [x] 6.2 Implement RAG knowledge base population
    - Create automated pipeline to populate Milvus with product embeddings (use notebook's data ingestion patterns)
    - Add data validation and quality checks for vector database
    - Implement incremental updates for new product data
    - _Requirements: 3.3, 5.2_
    - _Reference: notebook RAG sections for Milvus data population and batch processing patterns_

  - [x] 6.3 Setup Nova Pro fine-tuning pipeline
    - Create AWS Bedrock fine-tuning job automation scripts using S3-stored conversation format datasets (reference notebook's AWS patterns)
    - Implement model evaluation and performance tracking for fine-tuned Nova Pro models
    - Add fine-tuned model deployment and endpoint management for inference
    - Create validation scripts to ensure conversation format data quality before fine-tuning
    - _Requirements: 3.4, 5.3_
    - _Reference: notebook fine-tuning sections for AWS Bedrock job management and model deployment patterns_

- [x] 7. Create local development and testing framework
  - Implement mock services for local agent testing without AWS dependencies (reference notebook's local testing approach)
  - Create comprehensive test suite for inference accuracy validation using notebook's test data patterns
  - Add performance benchmarking tools for latency and throughput testing
  - Write integration tests for complete end-to-end workflows
  - _Requirements: 4.4, 4.5_
  - _Reference: notebook testing sections for validation patterns and performance measurement approaches_

- [x] 8. Implement configuration and model management system
  - Create environment-based configuration management for different deployment stages following PEP 8 standards
  - Implement model registry for easy switching between NER, embedding, and LLM models with comprehensive type hints
  - Add configuration validation and error handling for invalid settings with proper documentation
  - Write tests for configuration management and model switching functionality using Python 3.13 features in .venv environment
  - _Requirements: 5.1, 5.2, 5.4, 9.1, 9.7_

- [x] 9. Add monitoring and debugging capabilities
  - Implement comprehensive logging for all agent operations and decisions
  - Create health check endpoints for each agent service
  - Add diagnostic endpoints for troubleshooting inference issues
  - Setup CloudWatch dashboards for system monitoring and alerting
  - _Requirements: 4.1, 6.4, 6.6_

- [x] 10. Create developer documentation and deployment guides
  - Write comprehensive README with Python 3.13 .venv setup, PEP 8 compliance, and usage instructions
  - Create troubleshooting guide for common issues and debugging steps including ARM64 platform considerations
  - Document model accuracy tuning procedures for each inference approach with ml-sandbox AWS profile usage
  - Write deployment guide for AWS infrastructure setup and management in us-east-1 region
  - Add local testing guide with mock service setup instructions for Python 3.13 .venv environment
  - _Requirements: 4.2, 4.3, 4.5, 9.1, 9.2, 9.3, 9.5, 9.6, 9.7_

- [x] 11. Implement end-to-end integration and validation
  - Create complete inference pipeline integration tests using notebook's test data and validation patterns
  - Implement accuracy validation using reference product title-brand datasets (use notebook's training data format)
  - Add performance validation for latency and throughput requirements
  - Write deployment validation scripts for AWS infrastructure health checks
  - _Requirements: 1.5, 2.4, 4.1_
  - _Reference: notebook validation sections for end-to-end testing patterns and accuracy measurement approaches_