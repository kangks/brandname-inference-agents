"""
Mock factories for creating test doubles and mock objects.

This module provides factories for creating consistent mock objects
for testing various components of the inference system.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import random

from inference.src.agents.base_agent import BaseAgent
from inference.src.agents.registry import AgentRegistry
from inference.src.models.data_models import (
    ProductInput,
    NERResult,
    RAGResult,
    LLMResult,
    HybridResult,
    EntityResult,
    EntityType,
    SimilarProduct,
    AgentHealth
)


class MockAgentFactory:
    """
    Factory for creating mock agents for testing.
    
    Provides consistent mock agents with configurable behavior
    for different testing scenarios.
    """
    
    @staticmethod
    def create_mock_base_agent(agent_name: str = "mock_agent",
                              config: Dict[str, Any] = None) -> Mock:
        """
        Create a mock base agent with standard interface.
        
        Args:
            agent_name: Name for the mock agent
            config: Configuration dictionary
            
        Returns:
            Mock BaseAgent instance
        """
        if config is None:
            config = {"confidence_threshold": 0.5}
        
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.agent_name = agent_name
        mock_agent.config = config
        mock_agent._is_initialized = False
        mock_agent.logger = Mock()
        
        # Setup async methods
        mock_agent.initialize = AsyncMock()
        mock_agent.process = AsyncMock()
        mock_agent.cleanup = AsyncMock()
        mock_agent.health_check = AsyncMock()
        
        # Configure initialization behavior
        async def mock_initialize():
            mock_agent._is_initialized = True
        
        mock_agent.initialize.side_effect = mock_initialize
        
        # Configure health check behavior
        async def mock_health_check():
            return AgentHealth(
                agent_name=agent_name,
                is_healthy=mock_agent._is_initialized,
                last_check=1234567890.0,
                error_message=None if mock_agent._is_initialized else "Not initialized",
                response_time=0.1 if mock_agent._is_initialized else None
            )
        
        mock_agent.health_check.side_effect = mock_health_check
        
        return mock_agent
    
    @staticmethod
    def create_mock_ner_agent(success_rate: float = 1.0,
                             confidence_range: tuple = (0.7, 0.9)) -> Mock:
        """
        Create a mock NER agent with configurable behavior.
        
        Args:
            success_rate: Probability of successful processing (0.0-1.0)
            confidence_range: Range for confidence values
            
        Returns:
            Mock NER agent
        """
        mock_agent = MockAgentFactory.create_mock_base_agent("ner")
        
        async def mock_process(input_data: ProductInput):
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Simulate failure based on success rate
            if random.random() > success_rate:
                return {
                    "agent_type": "ner",
                    "result": None,
                    "success": False,
                    "error": "Simulated NER processing failure"
                }
            
            # Create mock entities
            entities = [
                EntityResult(
                    entity_type=EntityType.BRAND,
                    text="MockBrand",
                    confidence=random.uniform(*confidence_range),
                    start_pos=0,
                    end_pos=9
                )
            ]
            
            result = NERResult(
                entities=entities,
                confidence=random.uniform(*confidence_range),
                processing_time=0.1,
                model_used="mock_ner_model"
            )
            
            return {
                "agent_type": "ner",
                "result": result,
                "success": True,
                "error": None
            }
        
        mock_agent.process.side_effect = mock_process
        return mock_agent
    
    @staticmethod
    def create_mock_rag_agent(success_rate: float = 1.0,
                             confidence_range: tuple = (0.6, 0.8)) -> Mock:
        """
        Create a mock RAG agent with configurable behavior.
        
        Args:
            success_rate: Probability of successful processing (0.0-1.0)
            confidence_range: Range for confidence values
            
        Returns:
            Mock RAG agent
        """
        mock_agent = MockAgentFactory.create_mock_base_agent("rag")
        
        async def mock_process(input_data: ProductInput):
            # Simulate processing time
            await asyncio.sleep(0.2)
            
            # Simulate failure based on success rate
            if random.random() > success_rate:
                return {
                    "agent_type": "rag",
                    "result": None,
                    "success": False,
                    "error": "Simulated RAG processing failure"
                }
            
            # Create mock similar products
            similar_products = [
                SimilarProduct(
                    product_name="Mock Similar Product",
                    brand="MockBrand",
                    category="MockCategory",
                    sub_category="MockSubCategory",
                    similarity_score=random.uniform(0.7, 0.9)
                )
            ]
            
            result = RAGResult(
                similar_products=similar_products,
                predicted_brand="MockBrand",
                confidence=random.uniform(*confidence_range),
                processing_time=0.2,
                embedding_model="mock_embedding_model"
            )
            
            return {
                "agent_type": "rag",
                "result": result,
                "success": True,
                "error": None
            }
        
        mock_agent.process.side_effect = mock_process
        return mock_agent
    
    @staticmethod
    def create_mock_llm_agent(success_rate: float = 1.0,
                             confidence_range: tuple = (0.8, 0.95)) -> Mock:
        """
        Create a mock LLM agent with configurable behavior.
        
        Args:
            success_rate: Probability of successful processing (0.0-1.0)
            confidence_range: Range for confidence values
            
        Returns:
            Mock LLM agent
        """
        mock_agent = MockAgentFactory.create_mock_base_agent("llm")
        
        async def mock_process(input_data: ProductInput):
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Simulate failure based on success rate
            if random.random() > success_rate:
                return {
                    "agent_type": "llm",
                    "result": None,
                    "success": False,
                    "error": "Simulated LLM processing failure"
                }
            
            result = LLMResult(
                predicted_brand="MockBrand",
                reasoning="Mock reasoning for brand prediction",
                confidence=random.uniform(*confidence_range),
                processing_time=0.5,
                model_id="mock_llm_model"
            )
            
            return {
                "agent_type": "llm",
                "result": result,
                "success": True,
                "error": None
            }
        
        mock_agent.process.side_effect = mock_process
        return mock_agent
    
    @staticmethod
    def create_mock_hybrid_agent(success_rate: float = 1.0,
                                confidence_range: tuple = (0.85, 0.95)) -> Mock:
        """
        Create a mock Hybrid agent with configurable behavior.
        
        Args:
            success_rate: Probability of successful processing (0.0-1.0)
            confidence_range: Range for confidence values
            
        Returns:
            Mock Hybrid agent
        """
        mock_agent = MockAgentFactory.create_mock_base_agent("hybrid")
        
        async def mock_process(input_data: ProductInput):
            # Simulate processing time
            await asyncio.sleep(0.8)
            
            # Simulate failure based on success rate
            if random.random() > success_rate:
                return {
                    "agent_type": "hybrid",
                    "result": None,
                    "success": False,
                    "error": "Simulated Hybrid processing failure"
                }
            
            stage_results = {
                "ner": {"brand": "MockBrand", "confidence": 0.8},
                "rag": {"brand": "MockBrand", "confidence": 0.7},
                "llm": {"brand": "MockBrand", "confidence": 0.9}
            }
            
            result = HybridResult(
                final_prediction="MockBrand",
                confidence=random.uniform(*confidence_range),
                processing_time=0.8,
                stage_results=stage_results,
                stages_used=["ner", "rag", "llm"]
            )
            
            return {
                "agent_type": "hybrid",
                "result": result,
                "success": True,
                "error": None
            }
        
        mock_agent.process.side_effect = mock_process
        return mock_agent


class MockRegistryFactory:
    """
    Factory for creating mock agent registries.
    
    Provides mock registries with configurable agent sets
    for different testing scenarios.
    """
    
    @staticmethod
    def create_mock_registry(agent_types: List[str] = None,
                           success_rates: Dict[str, float] = None) -> Mock:
        """
        Create a mock agent registry with specified agents.
        
        Args:
            agent_types: List of agent types to include
            success_rates: Success rates for each agent type
            
        Returns:
            Mock AgentRegistry instance
        """
        if agent_types is None:
            agent_types = ["ner", "rag", "llm", "hybrid"]
        
        if success_rates is None:
            success_rates = {agent_type: 1.0 for agent_type in agent_types}
        
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.registered_agents = {}
        
        # Create mock agents
        for agent_type in agent_types:
            success_rate = success_rates.get(agent_type, 1.0)
            
            if agent_type == "ner":
                agent = MockAgentFactory.create_mock_ner_agent(success_rate)
            elif agent_type == "rag":
                agent = MockAgentFactory.create_mock_rag_agent(success_rate)
            elif agent_type == "llm":
                agent = MockAgentFactory.create_mock_llm_agent(success_rate)
            elif agent_type == "hybrid":
                agent = MockAgentFactory.create_mock_hybrid_agent(success_rate)
            else:
                agent = MockAgentFactory.create_mock_base_agent(agent_type)
            
            mock_registry.registered_agents[agent_type] = agent
        
        # Setup registry methods
        mock_registry.get_agent = Mock(
            side_effect=lambda name: mock_registry.registered_agents.get(name)
        )
        mock_registry.get_all_agents = Mock(
            return_value=mock_registry.registered_agents.copy()
        )
        mock_registry.list_agent_names = Mock(
            return_value=list(mock_registry.registered_agents.keys())
        )
        mock_registry.register_default_agents = AsyncMock(
            return_value=mock_registry.registered_agents
        )
        mock_registry.cleanup_agents = AsyncMock()
        
        return mock_registry
    
    @staticmethod
    def create_failing_registry() -> Mock:
        """
        Create a mock registry that simulates initialization failures.
        
        Returns:
            Mock registry that fails during initialization
        """
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.registered_agents = {}
        
        # Setup failing methods
        mock_registry.register_default_agents = AsyncMock(
            side_effect=Exception("Simulated registry initialization failure")
        )
        mock_registry.get_agent = Mock(return_value=None)
        mock_registry.get_all_agents = Mock(return_value={})
        mock_registry.list_agent_names = Mock(return_value=[])
        
        return mock_registry


class MockServiceFactory:
    """
    Factory for creating mock external services.
    
    Provides mock services for AWS, Milvus, and other external dependencies.
    """
    
    @staticmethod
    def create_mock_aws_bedrock() -> Mock:
        """
        Create a mock AWS Bedrock client.
        
        Returns:
            Mock Bedrock client
        """
        mock_bedrock = Mock()
        
        # Mock invoke_model method
        async def mock_invoke_model(modelId, body, **kwargs):
            return {
                'body': Mock(read=Mock(return_value=b'{"completion": "Mock LLM response"}')),
                'contentType': 'application/json'
            }
        
        mock_bedrock.invoke_model = AsyncMock(side_effect=mock_invoke_model)
        
        # Mock invoke_model_with_response_stream method
        async def mock_invoke_stream(modelId, body, **kwargs):
            # Simulate streaming response
            for chunk in [b'{"completion": "Mock"} ', b'{"completion": "streaming"} ', b'{"completion": "response"}']:
                yield {'chunk': {'bytes': chunk}}
        
        mock_bedrock.invoke_model_with_response_stream = AsyncMock(
            side_effect=mock_invoke_stream
        )
        
        return mock_bedrock
    
    @staticmethod
    def create_mock_milvus_client() -> Mock:
        """
        Create a mock Milvus client.
        
        Returns:
            Mock Milvus client
        """
        mock_client = Mock()
        
        # Mock connection methods
        mock_client.connect = Mock()
        mock_client.disconnect = Mock()
        
        # Mock collection methods
        mock_client.create_collection = Mock()
        mock_client.drop_collection = Mock()
        mock_client.has_collection = Mock(return_value=True)
        
        # Mock data operations
        mock_client.insert = Mock(return_value={"insert_count": 1})
        
        # Mock search method
        def mock_search(collection_name, data, **kwargs):
            return [[{
                "id": 1,
                "distance": 0.8,
                "entity": {
                    "product_name": "Mock Product",
                    "brand": "MockBrand",
                    "category": "MockCategory"
                }
            }]]
        
        mock_client.search = Mock(side_effect=mock_search)
        
        # Mock query method
        mock_client.query = Mock(return_value=[{
            "id": 1,
            "product_name": "Mock Product",
            "brand": "MockBrand"
        }])
        
        return mock_client
    
    @staticmethod
    def create_mock_sentence_transformer() -> Mock:
        """
        Create a mock SentenceTransformer model.
        
        Returns:
            Mock SentenceTransformer instance
        """
        mock_model = Mock()
        
        # Mock encode method
        def mock_encode(sentences, **kwargs):
            # Return mock embeddings
            if isinstance(sentences, str):
                return [[0.1] * 384]  # Single embedding
            else:
                return [[0.1] * 384 for _ in sentences]  # Multiple embeddings
        
        mock_model.encode = Mock(side_effect=mock_encode)
        
        return mock_model
    
    @staticmethod
    def create_mock_spacy_model() -> Mock:
        """
        Create a mock spaCy NLP model.
        
        Returns:
            Mock spaCy model
        """
        mock_nlp = Mock()
        
        # Mock entity class
        mock_entity = Mock()
        mock_entity.text = "MockBrand"
        mock_entity.label_ = "ORG"
        mock_entity.start_char = 0
        mock_entity.end_char = 9
        
        # Mock doc class
        mock_doc = Mock()
        mock_doc.ents = [mock_entity]
        
        # Mock nlp call
        mock_nlp.return_value = mock_doc
        
        return mock_nlp


class MockPatchFactory:
    """
    Factory for creating context managers for patching dependencies.
    
    Provides convenient patch contexts for testing with mocked dependencies.
    """
    
    @staticmethod
    def patch_aws_services():
        """
        Create patch context for AWS services.
        
        Returns:
            Context manager that patches AWS services
        """
        return patch.multiple(
            'boto3.client',
            bedrock=MockServiceFactory.create_mock_aws_bedrock(),
            s3=Mock()
        )
    
    @staticmethod
    def patch_milvus_client():
        """
        Create patch context for Milvus client.
        
        Returns:
            Context manager that patches Milvus client
        """
        return patch(
            'pymilvus.MilvusClient',
            return_value=MockServiceFactory.create_mock_milvus_client()
        )
    
    @staticmethod
    def patch_sentence_transformer():
        """
        Create patch context for SentenceTransformer.
        
        Returns:
            Context manager that patches SentenceTransformer
        """
        return patch(
            'sentence_transformers.SentenceTransformer',
            return_value=MockServiceFactory.create_mock_sentence_transformer()
        )
    
    @staticmethod
    def patch_spacy_model():
        """
        Create patch context for spaCy model.
        
        Returns:
            Context manager that patches spaCy model loading
        """
        return patch(
            'spacy.load',
            return_value=MockServiceFactory.create_mock_spacy_model()
        )
    
    @staticmethod
    def patch_all_dependencies():
        """
        Create patch context for all major dependencies.
        
        Returns:
            Context manager that patches all dependencies
        """
        return patch.multiple(
            'inference.src.agents',
            # Add patches for all major dependencies
        )


class ErrorSimulatorFactory:
    """
    Factory for creating error simulators for testing error handling.
    
    Provides various error scenarios for comprehensive error testing.
    """
    
    @staticmethod
    def create_timeout_simulator(timeout_duration: float = 5.0) -> Callable:
        """
        Create a function that simulates timeout errors.
        
        Args:
            timeout_duration: Duration to wait before timing out
            
        Returns:
            Async function that times out
        """
        async def timeout_simulator(*args, **kwargs):
            await asyncio.sleep(timeout_duration)
            return "This should timeout"
        
        return timeout_simulator
    
    @staticmethod
    def create_connection_error_simulator() -> Callable:
        """
        Create a function that simulates connection errors.
        
        Returns:
            Function that raises ConnectionError
        """
        def connection_error_simulator(*args, **kwargs):
            raise ConnectionError("Simulated connection error")
        
        return connection_error_simulator
    
    @staticmethod
    def create_aws_error_simulator(error_code: str = "ValidationException") -> Callable:
        """
        Create a function that simulates AWS service errors.
        
        Args:
            error_code: AWS error code to simulate
            
        Returns:
            Function that raises AWS ClientError
        """
        def aws_error_simulator(*args, **kwargs):
            from botocore.exceptions import ClientError
            raise ClientError(
                error_response={
                    'Error': {
                        'Code': error_code,
                        'Message': f'Simulated AWS {error_code} error'
                    }
                },
                operation_name='TestOperation'
            )
        
        return aws_error_simulator
    
    @staticmethod
    def create_intermittent_failure_simulator(failure_rate: float = 0.3) -> Callable:
        """
        Create a function that fails intermittently.
        
        Args:
            failure_rate: Probability of failure (0.0-1.0)
            
        Returns:
            Function that fails based on failure rate
        """
        def intermittent_failure_simulator(*args, **kwargs):
            if random.random() < failure_rate:
                raise Exception("Simulated intermittent failure")
            return "Success"
        
        return intermittent_failure_simulator