"""
Agent Registry for managing default agent instances.

This module provides centralized registration and management of default agents
for the multilingual product inference system, following PEP 8 standards.
"""

import logging
import time
from typing import Dict, Any, Optional, Type, List
import asyncio

from ..config.settings import get_config
from .base_agent import BaseAgent, NERAgent, RAGAgent, LLMAgent, HybridAgent
from .ner_agent import SpacyNERAgent, MultilingualNERAgent
from .rag_agent import SentenceTransformerRAGAgent, EnhancedRAGAgent
from .llm_agent import BedrockLLMAgent, EnhancedBedrockLLMAgent, FinetunedNovaLLMAgent
from .hybrid_agent import SequentialHybridAgent, OptimizedHybridAgent
from .simple_agent import SimpleInferenceAgent
# Note: StrandsOrchestratorAgent import removed to avoid circular dependency


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing agent instances and configurations.
    
    Provides centralized registration, initialization, and management
    of default agents for the inference system.
    """
    
    def __init__(self) -> None:
        """Initialize agent registry."""
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.system_config = get_config()
        
        # Default agent configurations
        self._setup_default_configurations()
    
    def _setup_default_configurations(self) -> None:
        """Setup default configurations for all agent types."""
        
        # NER Agent Configuration
        self.agent_configs["ner"] = {
            "model_name": "en_core_web_sm",  # Use reliable English model
            "confidence_threshold": 0.5,
            "max_text_length": 1000,
            "use_transformer_fallback": False,
            "thai_text_threshold": 0.3,
            "mixed_language_boost": 0.15
        }
        
        # RAG Agent Configuration
        self.agent_configs["rag"] = {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": 384,
            "milvus_uri": "./milvus_rag.db",
            "collection_name": "product_brand",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "confidence_threshold": 0.5,
            "max_text_length": 512,
            "similarity_weight": 0.6,
            "frequency_weight": 0.3,
            "diversity_weight": 0.1,
            "use_fuzzy_matching": True,
            "fuzzy_threshold": 0.8,
            "multilingual_boost": 0.1
        }
        
        # LLM Agent Configuration
        self.agent_configs["llm"] = {
            "aws_profile": self.system_config.aws.profile_name,
            "aws_region": self.system_config.aws.region,
            "model_id": "us.amazon.nova-pro-v1:0",
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_p": 0.9,
            "confidence_threshold": 0.5,
            "max_text_length": 1000,
            "timeout_seconds": 30,
            "use_context_enhancement": True,
            "context_weight": 0.3,
            "enable_reasoning_analysis": True
        }
        
        # Fine-tuned Nova LLM Agent Configuration
        self.agent_configs["finetuned_nova_llm"] = {
            "aws_profile": self.system_config.aws.profile_name,
            "aws_region": self.system_config.aws.region,
            "model_id": "arn:aws:bedrock:us-east-1:654654616949:custom-model/amazon.nova-pro-v1:0:300k/e4oo8js4bjz5",
            "max_tokens": 500,  # Shorter responses expected from fine-tuned model
            "temperature": 0.05,  # Lower temperature for more focused responses
            "top_p": 0.9,
            "confidence_threshold": 0.6,  # Higher threshold due to specialization
            "max_text_length": 1000,
            "timeout_seconds": 30,
            "use_context_enhancement": False,  # Fine-tuned model may not need context enhancement
            "context_weight": 0.2,
            "enable_reasoning_analysis": False  # Fine-tuned model should give direct responses
        }
        
        # Hybrid Agent Configuration
        self.agent_configs["hybrid"] = {
            "enable_ner_stage": True,
            "enable_rag_stage": True,
            "enable_llm_stage": True,
            "ner_confidence_threshold": 0.8,
            "rag_confidence_threshold": 0.8,
            "ner_weight": 0.3,
            "rag_weight": 0.4,
            "llm_weight": 0.3,
            "use_early_termination": False,
            "use_context_enhancement": True,
            "max_pipeline_time": 60.0,
            "use_dynamic_thresholds": True,
            "use_stage_selection": True,
            "performance_mode": "balanced",  # fast, balanced, accurate
            "min_confidence_threshold": 0.6,
            "max_confidence_threshold": 0.9
        }
        
        # Simple Agent Configuration (fallback)
        self.agent_configs["simple"] = {
            "confidence_threshold": 0.6
        }
    
    async def register_default_agents(self) -> Dict[str, BaseAgent]:
        """
        Register and initialize all default agents.
        
        Returns:
            Dictionary of registered and initialized agents
        """
        logger.info("Registering default agents...")
        
        try:
            # Register NER agent
            await self._register_ner_agent()
            
            # Register RAG agent
            await self._register_rag_agent()
            
            # Register LLM agent
            await self._register_llm_agent()
            
            # Register Fine-tuned Nova LLM agent
            await self._register_finetuned_nova_agent()
            
            # Register Hybrid agent
            await self._register_hybrid_agent()
            
            # Register Simple agent as fallback
            await self._register_simple_agent()
            
            logger.info(f"Successfully registered {len(self.registered_agents)} default agents")
            return self.registered_agents.copy()
            
        except Exception as e:
            logger.error(f"Failed to register default agents: {str(e)}")
            raise
    
    async def _register_ner_agent(self) -> None:
        """Register and initialize NER agent."""
        try:
            logger.info("Registering NER agent...")
            
            # Try to create basic spaCy NER agent first (more reliable)
            try:
                logger.info("Attempting to initialize SpacyNERAgent...")
                ner_agent = SpacyNERAgent(self.agent_configs["ner"])
                await ner_agent.initialize()
                self.registered_agents["ner"] = ner_agent
                logger.info("✅ SpaCy NER agent registered successfully")
                return
            except Exception as e:
                logger.warning(f"❌ Failed to initialize SpaCy NER agent: {str(e)}")
                logger.warning(f"   Error type: {type(e).__name__}")
            
            # Try multilingual NER agent as fallback
            try:
                logger.info("Attempting to initialize MultilingualNERAgent...")
                ner_agent = MultilingualNERAgent(self.agent_configs["ner"])
                await ner_agent.initialize()
                self.registered_agents["ner"] = ner_agent
                logger.info("✅ Multilingual NER agent registered successfully")
                return
            except Exception as e:
                logger.warning(f"❌ Failed to initialize multilingual NER agent: {str(e)}")
                logger.warning(f"   Error type: {type(e).__name__}")
            
            # If spaCy is not available, create a mock NER agent
            logger.warning("⚠️  All real NER agents failed, falling back to mock NER agent")
            mock_ner_agent = MockNERAgent(self.agent_configs["ner"])
            await mock_ner_agent.initialize()
            self.registered_agents["ner"] = mock_ner_agent
            logger.info("✅ Mock NER agent registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to register any NER agent: {str(e)}")
            # Don't raise exception - system can work without NER
    
    async def _register_rag_agent(self) -> None:
        """Register and initialize RAG agent."""
        try:
            logger.info("Registering RAG agent...")
            
            # Create enhanced RAG agent
            rag_agent = EnhancedRAGAgent(self.agent_configs["rag"])
            
            # Initialize the agent
            await rag_agent.initialize()
            
            # Register the agent
            self.registered_agents["rag"] = rag_agent
            
            logger.info("RAG agent registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to register RAG agent: {str(e)}")
            # Don't raise exception - system can work without RAG
    
    async def _register_llm_agent(self) -> None:
        """Register and initialize LLM agent."""
        try:
            logger.info("Registering LLM agent...")
            
            # Try to create enhanced Bedrock LLM agent first
            try:
                llm_agent = EnhancedBedrockLLMAgent(self.agent_configs["llm"])
                await llm_agent.initialize()
                self.registered_agents["llm"] = llm_agent
                logger.info("Enhanced Bedrock LLM agent registered successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced Bedrock LLM agent: {str(e)}")
            
            # Fallback to basic Bedrock LLM agent
            try:
                llm_agent = BedrockLLMAgent(self.agent_configs["llm"])
                await llm_agent.initialize()
                self.registered_agents["llm"] = llm_agent
                logger.info("Basic Bedrock LLM agent registered successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize basic Bedrock LLM agent: {str(e)}")
            
            # If Bedrock is not available, create a mock LLM agent
            logger.info("Creating mock LLM agent as fallback...")
            mock_llm_agent = MockLLMAgent(self.agent_configs["llm"])
            await mock_llm_agent.initialize()
            self.registered_agents["llm"] = mock_llm_agent
            logger.info("Mock LLM agent registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to register any LLM agent: {str(e)}")
            # Don't raise exception - system can work without LLM
    
    async def _register_finetuned_nova_agent(self) -> None:
        """Register and initialize Fine-tuned Nova LLM agent."""
        try:
            logger.info("Registering Fine-tuned Nova LLM agent...")
            
            # Try to create fine-tuned Nova LLM agent
            try:
                finetuned_agent = FinetunedNovaLLMAgent(self.agent_configs["finetuned_nova_llm"])
                await finetuned_agent.initialize()
                self.registered_agents["finetuned_nova_llm"] = finetuned_agent
                logger.info("✅ Fine-tuned Nova LLM agent registered successfully")
                return
            except Exception as e:
                logger.warning(f"❌ Failed to initialize fine-tuned Nova LLM agent: {str(e)}")
                logger.warning(f"   Error type: {type(e).__name__}")
                logger.warning("   This may be due to:")
                logger.warning("   - Fine-tuned model ARN not accessible")
                logger.warning("   - AWS credentials insufficient for custom model access")
                logger.warning("   - Model not available in the specified region")
            
        except Exception as e:
            logger.warning(f"Failed to register fine-tuned Nova LLM agent: {str(e)}")
            # Don't raise exception - system can work without fine-tuned model
    
    async def _register_hybrid_agent(self) -> None:
        """Register and initialize Hybrid agent."""
        try:
            logger.info("Registering Hybrid agent...")
            
            # Create optimized hybrid agent
            hybrid_agent = OptimizedHybridAgent(self.agent_configs["hybrid"])
            
            # Initialize the agent
            await hybrid_agent.initialize()
            
            # Register the agent
            self.registered_agents["hybrid"] = hybrid_agent
            
            logger.info("Hybrid agent registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to register Hybrid agent: {str(e)}")
            # Don't raise exception - system can work without Hybrid
    
    async def _register_simple_agent(self) -> None:
        """Register and initialize Simple agent as fallback."""
        try:
            logger.info("Registering Simple agent...")
            
            # Create simple inference agent
            simple_agent = SimpleInferenceAgent(self.agent_configs["simple"])
            
            # Initialize the agent
            await simple_agent.initialize()
            
            # Register the agent
            self.registered_agents["simple"] = simple_agent
            
            logger.info("Simple agent registered successfully")
            
        except Exception as e:
            logger.warning(f"Failed to register Simple agent: {str(e)}")
            # Don't raise exception - this is just a fallback
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get a registered agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self.registered_agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """
        Get all registered agents.
        
        Returns:
            Dictionary of all registered agents
        """
        return self.registered_agents.copy()
    
    def list_agent_names(self) -> List[str]:
        """
        List names of all registered agents.
        
        Returns:
            List of agent names
        """
        return list(self.registered_agents.keys())
    
    async def cleanup_agents(self) -> None:
        """Clean up all registered agents."""
        logger.info("Cleaning up registered agents...")
        
        cleanup_tasks = []
        for agent_name, agent in self.registered_agents.items():
            try:
                cleanup_tasks.append(agent.cleanup())
            except Exception as e:
                logger.warning(f"Error scheduling cleanup for {agent_name}: {str(e)}")
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.registered_agents.clear()
        logger.info("Agent cleanup completed")
    
    def update_agent_config(self, agent_name: str, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific agent type.
        
        Args:
            agent_name: Name of the agent type
            config_updates: Configuration updates to apply
        """
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].update(config_updates)
            logger.info(f"Updated configuration for {agent_name} agent")
        else:
            logger.warning(f"Unknown agent type: {agent_name}")
    
    async def reinitialize_agent(self, agent_name: str) -> bool:
        """
        Reinitialize a specific agent with updated configuration.
        
        Args:
            agent_name: Name of the agent to reinitialize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean up existing agent if present
            if agent_name in self.registered_agents:
                await self.registered_agents[agent_name].cleanup()
                del self.registered_agents[agent_name]
            
            # Reinitialize based on agent type
            if agent_name == "ner":
                await self._register_ner_agent()
            elif agent_name == "rag":
                await self._register_rag_agent()
            elif agent_name == "llm":
                await self._register_llm_agent()
            elif agent_name == "finetuned_nova_llm":
                await self._register_finetuned_nova_agent()
            elif agent_name == "hybrid":
                await self._register_hybrid_agent()
            elif agent_name == "simple":
                await self._register_simple_agent()
            else:
                logger.error(f"Unknown agent type for reinitialization: {agent_name}")
                return False
            
            logger.info(f"Successfully reinitialized {agent_name} agent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reinitialize {agent_name} agent: {str(e)}")
            return False


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.
    
    Returns:
        Global AgentRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = AgentRegistry()
    
    return _global_registry


async def initialize_default_agents() -> Dict[str, BaseAgent]:
    """
    Initialize all default agents using the global registry.
    
    Returns:
        Dictionary of initialized agents
    """
    registry = get_agent_registry()
    return await registry.register_default_agents()


async def cleanup_default_agents() -> None:
    """Clean up all default agents using the global registry."""
    global _global_registry
    
    if _global_registry:
        await _global_registry.cleanup_agents()
        _global_registry = None


def get_default_agent(agent_name: str) -> Optional[BaseAgent]:
    """
    Get a default agent by name.
    
    Args:
        agent_name: Name of the agent to retrieve
        
    Returns:
        Agent instance or None if not found
    """
    registry = get_agent_registry()
    return registry.get_agent(agent_name)


def list_default_agents() -> List[str]:
    """
    List names of all registered default agents.
    
    Returns:
        List of agent names
    """
    registry = get_agent_registry()
    return registry.list_agent_names()


# Mock agents for fallback when dependencies are not available

class MockNERAgent(NERAgent):
    """Mock NER agent that provides basic pattern-based entity extraction."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("mock_ner", config)
        self.brand_patterns = [
            r'\b(Samsung|Apple|Sony|Nike|Adidas|Toyota|Honda|LG|Huawei|Xiaomi)\b',
            r'\b(Microsoft|Google|Amazon|Facebook|Tesla|BMW|Mercedes|Audi)\b',
            r'\b(Coca-Cola|Pepsi|McDonald|Starbucks|KFC|Pizza Hut)\b'
        ]
    
    async def initialize(self) -> None:
        self.set_initialized(True)
        self.logger.info("Mock NER agent initialized")
    
    async def process(self, input_data) -> Dict[str, Any]:
        from ..models.data_models import NERResult, EntityResult, EntityType
        
        start_time = time.time()
        try:
            entities = []
            text = input_data.product_name.lower()
            
            # Simple pattern matching for brands
            for pattern in self.brand_patterns:
                import re
                matches = re.finditer(pattern, input_data.product_name, re.IGNORECASE)
                for match in matches:
                    entities.append(EntityResult(
                        entity_type=EntityType.BRAND,
                        text=match.group(),
                        confidence=0.8,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
            
            result = NERResult(
                entities=entities,
                confidence=0.8 if entities else 0.0,
                processing_time=time.time() - start_time,
                model_used="mock_pattern_matcher"
            )
            
            return {
                "agent_type": "ner",
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "agent_type": "ner", 
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def extract_entities(self, product_name: str):
        # This method is called by the process method
        pass
    
    async def cleanup(self) -> None:
        self.set_initialized(False)


class MockLLMAgent(LLMAgent):
    """Mock LLM agent that provides rule-based brand inference."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__("mock_llm", config)
        self.brand_mapping = {
            'iphone': 'Apple', 'ipad': 'Apple', 'macbook': 'Apple', 'airpods': 'Apple',
            'galaxy': 'Samsung', 'samsung': 'Samsung',
            'pixel': 'Google', 'android': 'Google',
            'surface': 'Microsoft', 'xbox': 'Microsoft',
            'playstation': 'Sony', 'sony': 'Sony', 'bravia': 'Sony',
            'nike': 'Nike', 'jordan': 'Nike', 'air max': 'Nike',
            'adidas': 'Adidas', 'yeezy': 'Adidas',
            'toyota': 'Toyota', 'camry': 'Toyota', 'prius': 'Toyota',
            'honda': 'Honda', 'civic': 'Honda', 'accord': 'Honda',
            'coca-cola': 'Coca-Cola', 'coke': 'Coca-Cola', 'โค้ก': 'Coca-Cola',
            'pepsi': 'Pepsi', 'เป๊ปซี่': 'Pepsi'
        }
    
    async def initialize(self) -> None:
        self.set_initialized(True)
        self.logger.info("Mock LLM agent initialized")
    
    async def process(self, input_data) -> Dict[str, Any]:
        from ..models.data_models import LLMResult
        
        start_time = time.time()
        try:
            result = await self.infer_brand(input_data.product_name)
            
            return {
                "agent_type": "llm",
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "agent_type": "llm",
                "result": None,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def infer_brand(self, product_name: str, context: Optional[str] = None):
        from ..models.data_models import LLMResult
        
        start_time = time.time()
        text_lower = product_name.lower()
        
        # Find brand using keyword matching
        predicted_brand = "Unknown"
        confidence = 0.0
        reasoning = "No matching brand patterns found"
        
        for keyword, brand in self.brand_mapping.items():
            if keyword in text_lower:
                predicted_brand = brand
                confidence = 0.85
                reasoning = f"Detected '{keyword}' pattern indicating {brand} brand"
                break
        
        return LLMResult(
            predicted_brand=predicted_brand,
            reasoning=reasoning,
            confidence=confidence,
            processing_time=time.time() - start_time,
            model_id="mock_rule_based_llm"
        )
    
    async def cleanup(self) -> None:
        self.set_initialized(False)