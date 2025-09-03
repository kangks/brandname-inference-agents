# Requirements Document

## Introduction

This feature implements a proof-of-concept agentic workflow system that processes product names in English, Thai, or mixed languages using multiple AI inference mechanisms (NER, RAG, LLM, and hybrid approaches). The system uses the strands-agents SDK with an orchestrator agent that coordinates parallel inference agents, returning comprehensive results with confidence-based best inference selection. The solution includes modular training/preprocessing pipelines and inference infrastructure deployed on AWS using simple, debuggable architectures like ECS Fargate and Milvus for vector storage.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to submit a product name in any language (English, Thai, or mixed) and receive inference results from multiple AI mechanisms, so that I can get the most accurate product identification with confidence scores.

#### Acceptance Criteria

1. WHEN a product name is submitted THEN the system SHALL accept input in English, Thai, or mixed languages
2. WHEN processing begins THEN the orchestrator agent SHALL invoke NER, RAG, LLM, and hybrid inference agents in parallel
3. WHEN all inference mechanisms complete THEN the system SHALL return results from all mechanisms with individual confidence scores
4. WHEN confidence scores are calculated THEN the system SHALL identify and return the best inference result based on highest confidence
5. WHEN results are returned THEN the system SHALL include detailed inference data from each mechanism for analysis

### Requirement 2

**User Story:** As a system architect, I want the inference agents to be deployed on AWS with proper orchestration, so that the system can scale and handle concurrent requests efficiently.

#### Acceptance Criteria

1. WHEN deploying inference components THEN the system SHALL use strands-agents SDK for agent orchestration
2. WHEN orchestrator agent runs THEN it SHALL coordinate parallel execution of inference agents
3. WHEN hybrid agents execute THEN they SHALL support sequential steps (e.g., RAG+LLM pipeline)
4. WHEN deploying to AWS THEN all inference components SHALL be containerized and deployed using AWS services
5. WHEN agents communicate THEN the system SHALL handle inter-agent messaging and result aggregation

### Requirement 3

**User Story:** As a data scientist, I want modular training and preprocessing pipelines for each inference approach, so that I can independently prepare NER models, RAG knowledge bases, and fine-tune LLM models using AWS Bedrock.

#### Acceptance Criteria

1. WHEN preprocessing for NER THEN the system SHALL provide scripts to prepare training data and populate vector databases for entity recognition
2. WHEN training NER models THEN the system SHALL support custom entity training (BRAND, CATEGORY, VARIANT) using spaCy or transformer models
3. WHEN preparing RAG knowledge base THEN the system SHALL create brand name vectors using sentence transformers and populate Milvus vector database
4. WHEN fine-tuning LLM THEN the system SHALL convert product title-brand datasets to conversation format (using convert_to_conversation_format() approach), store in S3, and submit to AWS Bedrock for Nova Pro fine-tuning
5. WHEN managing training pipelines THEN each inference approach SHALL have separate, modular training scripts in the training/ folder
6. WHEN deploying training stack THEN it SHALL be independent from inference infrastructure for separate scaling and maintenance

### Requirement 4

**User Story:** As a developer, I want comprehensive documentation and debugging capabilities, so that I can troubleshoot issues, adjust accuracy, and modify inference approaches easily.

#### Acceptance Criteria

1. WHEN troubleshooting THEN the system SHALL provide detailed logging and debugging information
2. WHEN adjusting accuracy THEN developers SHALL have clear guidance on tuning each inference mechanism
3. WHEN adding new inference approaches THEN the system SHALL support pluggable architecture for easy extension
4. WHEN testing locally THEN developers SHALL have complete local development and testing capabilities
5. WHEN deploying to AWS THEN the system SHALL include deployment guides and testing procedures

### Requirement 5

**User Story:** As a developer, I want flexible model configuration, so that I can easily switch between different NER models, RAG embeddings, and LLM models for experimentation and optimization.

#### Acceptance Criteria

1. WHEN configuring NER THEN the system SHALL support multiple NER model backends through configuration
2. WHEN configuring RAG THEN the system SHALL support different embedding models and vector databases
3. WHEN configuring LLM THEN the system SHALL support multiple LLM providers and models
4. WHEN changing configurations THEN the system SHALL require minimal code changes
5. WHEN testing configurations THEN the system SHALL provide validation and performance comparison tools

### Requirement 6

**User Story:** As a DevOps engineer, I want simple and maintainable AWS infrastructure using least complicated deployment methods, so that I can easily deploy, debug, and modify the system components.

#### Acceptance Criteria

1. WHEN deploying inference agents THEN the system SHALL use ECS Fargate for containerized deployment with simple configuration
2. WHEN deploying vector storage THEN the system SHALL use Milvus on ECS for easy state management and debugging
3. WHEN deploying infrastructure THEN the system SHALL use CloudFormation or CDK for reproducible, simple stack deployments
4. WHEN monitoring THEN the system SHALL include basic CloudWatch logging and health checks for troubleshooting
5. WHEN maintaining THEN the system SHALL provide clear separation between inference/ and training/ folder structures
6. WHEN debugging THEN the system SHALL include diagnostic endpoints and easy log access for each agent component

### Requirement 7

**User Story:** As a developer, I want modular inference agent architecture with clear separation of concerns, so that I can easily add, modify, or replace individual inference mechanisms without affecting others.

#### Acceptance Criteria

1. WHEN implementing NER agent THEN it SHALL be a standalone module that processes product names and returns entity predictions with confidence scores
2. WHEN implementing RAG agent THEN it SHALL be a standalone module that retrieves similar products from Milvus and generates brand predictions
3. WHEN implementing LLM agent THEN it SHALL be a standalone module that uses fine-tuned Nova Pro via Bedrock for direct brand inference
4. WHEN implementing hybrid agents THEN they SHALL support sequential processing (e.g., NER→RAG→LLM pipeline) as separate workflow steps
5. WHEN orchestrating agents THEN the orchestrator SHALL coordinate parallel execution and aggregate results with confidence scoring
6. WHEN deploying agents THEN each SHALL be independently deployable and testable on AWS infrastructure

### Requirement 8

**User Story:** As a researcher, I want to reference existing implementations and easily adapt proven techniques, so that I can build upon established methods for multilingual brand extraction.

#### Acceptance Criteria

1. WHEN implementing LLM inference THEN the system SHALL reference and adapt techniques from brand_extraction_ner_rag_llm_rev2.ipynb
2. WHEN implementing Nova fine-tuning THEN the system SHALL use AWS Bedrock fine-tuning approaches demonstrated in the reference notebook
3. WHEN implementing RAG retrieval THEN the system SHALL support sentence transformers and Milvus vector search as shown in reference implementations
4. WHEN implementing NER THEN the system SHALL support spaCy custom entity recognition for multilingual product titles
5. WHEN processing multilingual input THEN the system SHALL handle Thai-English mixed text as demonstrated in reference examples
6. WHEN evaluating results THEN the system SHALL provide confidence scoring and comparison mechanisms for different inference approaches

### Requirement 9

**User Story:** As a developer, I want standardized development environment requirements and AWS configuration, so that I can ensure consistent setup across all development and deployment activities.

#### Acceptance Criteria

1. WHEN setting up development environment THEN the system SHALL require Python 3.13 in a .venv virtual environment for all development work
2. WHEN testing deployment functionality THEN the system SHALL use AWS profile "ml-sandbox" for all AWS service interactions
3. WHEN deploying to AWS THEN the system SHALL target us-east-1 region for all AWS resources and services
4. WHEN configuring AWS services THEN the system SHALL validate ml-sandbox profile access and us-east-1 region availability
5. WHEN documenting setup THEN all installation and deployment guides SHALL specify Python 3.13 .venv virtual environment and ml-sandbox AWS profile requirements
6. WHEN deploying to AWS THEN the system SHALL use ARM64 platform architecture for all containerized services and ECS tasks
7. WHEN writing code THEN all Python code SHALL follow PEP 8 coding standards for consistency and maintainability