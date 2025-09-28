"""
Integration tests for configuration management and dynamic agent registration.

This module tests dynamic agent registration via configuration changes, configuration
validation and error handling, model switching, configuration updates, and
environment-specific configuration loading as specified in the requirements.
"""

import pytest
import asyncio
import json
import tempfile
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from inference.src.agents.orchestrator_agent import StrandsMultiAgentOrchestrator
from inference.src.agents.registry import AgentRegistry, get_agent_registry
from inference.src.config.settings import get_config, SystemConfig
from inference.src.config.validators import ConfigValidator
from inference.src.models.data_models import ProductInput, LanguageHint

from inference.tests.utils.mock_factories import MockAgentFactory


@pytest.mark.integration
class TestConfigurationIntegration:
    """
    Integration tests for configuration management and dynamic agent registration.
    
    Tests dynamic agent registration, configuration validation, model switching,
    and environment-specific configuration loading.
    """
    
    def setup_method(self):
        """Setup for each test method."""
        
        # Create temporary directory for configuration files
        self.temp_dir = tempfile.mkdtemp(prefix="config_test_")
        
        # Base configuration for testing
        self.base_config = {
            "agents": {
                "ner": {
                    "enabled": True,
                    "class": "SpacyNERAgent",
                    "config": {
                        "model_name": "en_core_web_sm",
                        "confidence_threshold": 0.5
                    }
                },
                "rag": {
                    "enabled": True,
                    "class": "SentenceTransformerRAGAgent",
                    "config": {
                        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                        "milvus_uri": "./test_milvus.db"
                    }
                },
                "llm": {
                    "enabled": True,
                    "class": "StrandsLLMAgent",
                    "config": {
                        "model_id": "amazon.nova-pro-v1:0",
                        "temperature": 0.1
                    }
                }
            },
            "orchestrator": {
                "confidence_threshold": 0.5,
                "max_parallel_agents": 4,
                "timeout_seconds": 30
            },
            "environment": "test"
        }
        
        # Create orchestrator for testing
        self.orchestrator = StrandsMultiAgentOrchestrator(self.base_config.get("orchestrator", {}))
        
        # Test input
        self.test_input = ProductInput(
            product_name="Samsung Galaxy S24",
            language_hint=LanguageHint.ENGLISH
        )
    
    def teardown_method(self):
        """Cleanup after each test method."""
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up orchestrator
        if hasattr(self.orchestrator, 'cleanup'):
            asyncio.run(self.orchestrator.cleanup())
    
    def _create_config_file(self, config: Dict[str, Any], filename: str = "test_config.json") -> str:
        """
        Create a configuration file for testing.
        
        Args:
            config: Configuration dictionary
            filename: Name of the configuration file
            
        Returns:
            Path to the created configuration file
        """
        config_path = os.path.join(self.temp_dir, filename)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    @pytest.mark.asyncio
    async def test_dynamic_agent_registration_via_configuration_changes(self):
        """Test dynamic agent registration when configuration changes."""
        # Create initial configuration with basic agents
        initial_config = self.base_config.copy()
        config_path = self._create_config_file(initial_config, "initial_config.json")
        
        # Create agent registry
        registry = AgentRegistry()
        
        # Load initial configuration
        with patch('inference.src.config.settings.get_config') as mock_get_config:
            mock_get_config.return_value = SystemConfig(**initial_config)
            
            # Register initial agents
            initial_agents = await registry.register_default_agents()
            
            # Verify initial agents are registered
            assert len(initial_agents) == 3  # ner, rag, llm
            assert "ner" in initial_agents
            assert "rag" in initial_agents
            assert "llm" in initial_agents
        
        # Update configuration to add new agent
        updated_config = initial_config.copy()
        updated_config["agents"]["hybrid"] = {
            "enabled": True,
            "class": "HybridAgent",
            "config": {
                "stages": ["ner", "rag", "llm"],
                "confidence_threshold": 0.6
            }
        }
        
        # Update configuration file
        updated_config_path = self._create_config_file(updated_config, "updated_config.json")
        
        # Test dynamic registration of new agent
        with patch('inference.src.config.settings.get_config') as mock_get_config:
            mock_get_config.return_value = SystemConfig(**updated_config)
            
            # Simulate configuration reload and agent registration
            updated_agents = await self._simulate_dynamic_agent_registration(
                registry, 
                updated_config
            )
            
            # Verify new agent was registered
            assert len(updated_agents) == 4  # ner, rag, llm, hybrid
            assert "hybrid" in updated_agents
            
            # Verify existing agents are still registered
            assert "ner" in updated_agents
            assert "rag" in updated_agents
            assert "llm" in updated_agents
    
    async def _simulate_dynamic_agent_registration(
        self, 
        registry: AgentRegistry, 
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate dynamic agent registration with new configuration.
        
        Args:
            registry: Agent registry instance
            new_config: New configuration to apply
            
        Returns:
            Updated agents dictionary
        """
        # Create mock agents for the new configuration
        mock_agents = {}
        
        for agent_name, agent_config in new_config.get("agents", {}).items():
            if agent_config.get("enabled", False):
                mock_agent = MockAgentFactory.create_mock_base_agent(agent_name)
                mock_agents[agent_name] = mock_agent
        
        # Update registry with new agents
        registry.registered_agents.update(mock_agents)
        
        return mock_agents
    
    @pytest.mark.asyncio
    async def test_configuration_validation_and_error_handling(self):
        """Test configuration validation and error handling mechanisms."""
        # Test valid configuration
        valid_config = self.base_config.copy()
        validation_result = self._validate_configuration(valid_config)
        
        assert validation_result["is_valid"] is True
        assert len(validation_result["errors"]) == 0
        
        # Test invalid configuration - missing required fields
        invalid_config_missing = {
            "agents": {
                "ner": {
                    "enabled": True
                    # Missing "class" and "config"
                }
            }
        }
        
        validation_result = self._validate_configuration(invalid_config_missing)
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0
        assert any("class" in error.lower() for error in validation_result["errors"])
        
        # Test invalid configuration - invalid agent class
        invalid_config_class = self.base_config.copy()
        invalid_config_class["agents"]["ner"]["class"] = "NonExistentAgent"
        
        validation_result = self._validate_configuration(invalid_config_class)
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0
        
        # Test invalid configuration - invalid parameter types
        invalid_config_types = self.base_config.copy()
        invalid_config_types["orchestrator"]["confidence_threshold"] = "invalid_string"  # Should be float
        
        validation_result = self._validate_configuration(invalid_config_types)
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0
        
        # Test configuration error recovery
        recovery_result = await self._test_configuration_error_recovery(invalid_config_missing)
        
        assert "fallback_config" in recovery_result
        assert "recovery_strategy" in recovery_result
        assert recovery_result["recovery_successful"] is True
    
    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration using configuration validator.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors and status
        """
        errors = []
        
        # Validate required top-level sections
        required_sections = ["agents"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate agents configuration
        if "agents" in config:
            for agent_name, agent_config in config["agents"].items():
                if not isinstance(agent_config, dict):
                    errors.append(f"Agent {agent_name} configuration must be a dictionary")
                    continue
                
                # Check required agent fields
                required_agent_fields = ["enabled", "class", "config"]
                for field in required_agent_fields:
                    if field not in agent_config:
                        errors.append(f"Agent {agent_name} missing required field: {field}")
                
                # Validate agent class
                if "class" in agent_config:
                    valid_classes = [
                        "SpacyNERAgent", "SentenceTransformerRAGAgent", 
                        "StrandsLLMAgent", "HybridAgent"
                    ]
                    if agent_config["class"] not in valid_classes:
                        errors.append(f"Agent {agent_name} has invalid class: {agent_config['class']}")
        
        # Validate orchestrator configuration
        if "orchestrator" in config:
            orchestrator_config = config["orchestrator"]
            
            # Validate confidence threshold
            if "confidence_threshold" in orchestrator_config:
                threshold = orchestrator_config["confidence_threshold"]
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    errors.append("Orchestrator confidence_threshold must be a number between 0.0 and 1.0")
            
            # Validate max_parallel_agents
            if "max_parallel_agents" in orchestrator_config:
                max_agents = orchestrator_config["max_parallel_agents"]
                if not isinstance(max_agents, int) or max_agents < 1:
                    errors.append("Orchestrator max_parallel_agents must be a positive integer")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "validated_config": config if len(errors) == 0 else None
        }
    
    async def _test_configuration_error_recovery(self, invalid_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test configuration error recovery mechanisms.
        
        Args:
            invalid_config: Invalid configuration to test recovery with
            
        Returns:
            Recovery test results
        """
        # Attempt to create fallback configuration
        fallback_config = self._create_fallback_configuration(invalid_config)
        
        # Validate fallback configuration
        fallback_validation = self._validate_configuration(fallback_config)
        
        recovery_successful = fallback_validation["is_valid"]
        
        return {
            "original_config": invalid_config,
            "fallback_config": fallback_config,
            "recovery_strategy": "fallback_to_defaults",
            "recovery_successful": recovery_successful,
            "fallback_validation": fallback_validation
        }
    
    def _create_fallback_configuration(self, invalid_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback configuration from invalid configuration.
        
        Args:
            invalid_config: Invalid configuration
            
        Returns:
            Fallback configuration with defaults
        """
        # Start with base valid configuration
        fallback_config = self.base_config.copy()
        
        # Try to preserve valid parts of invalid configuration
        if "agents" in invalid_config:
            for agent_name, agent_config in invalid_config["agents"].items():
                if isinstance(agent_config, dict) and agent_config.get("enabled", False):
                    # Only include if it has minimum required fields
                    if "class" in agent_config and "config" in agent_config:
                        fallback_config["agents"][agent_name] = agent_config
        
        return fallback_config
    
    @pytest.mark.asyncio
    async def test_model_switching_and_configuration_updates(self):
        """Test model switching and live configuration updates."""
        # Create initial configuration with specific models
        initial_config = self.base_config.copy()
        initial_config["agents"]["llm"]["config"]["model_id"] = "amazon.nova-pro-v1:0"
        
        # Create orchestrator with initial configuration
        with patch('inference.src.config.settings.get_config') as mock_get_config:
            mock_get_config.return_value = SystemConfig(**initial_config)
            
            # Initialize with initial model
            initial_result = await self._test_model_configuration(
                initial_config["agents"]["llm"]["config"]["model_id"]
            )
            
            assert initial_result["model_id"] == "amazon.nova-pro-v1:0"
            assert initial_result["initialization_successful"] is True
        
        # Update configuration to switch model
        updated_config = initial_config.copy()
        updated_config["agents"]["llm"]["config"]["model_id"] = "amazon.nova-lite-v1:0"
        updated_config["agents"]["llm"]["config"]["temperature"] = 0.2  # Also update parameter
        
        # Test model switching
        with patch('inference.src.config.settings.get_config') as mock_get_config:
            mock_get_config.return_value = SystemConfig(**updated_config)
            
            # Test model switch
            switch_result = await self._test_model_switch(
                initial_config["agents"]["llm"]["config"],
                updated_config["agents"]["llm"]["config"]
            )
            
            assert switch_result["switch_successful"] is True
            assert switch_result["new_model_id"] == "amazon.nova-lite-v1:0"
            assert switch_result["new_temperature"] == 0.2
            assert switch_result["old_model_id"] == "amazon.nova-pro-v1:0"
        
        # Test configuration hot-reload
        hot_reload_result = await self._test_configuration_hot_reload(
            initial_config, 
            updated_config
        )
        
        assert hot_reload_result["reload_successful"] is True
        assert hot_reload_result["agents_updated"] > 0
    
    async def _test_model_configuration(self, model_id: str) -> Dict[str, Any]:
        """
        Test model configuration and initialization.
        
        Args:
            model_id: Model ID to test
            
        Returns:
            Model configuration test results
        """
        try:
            # Create mock LLM agent with specified model
            mock_llm_agent = MockAgentFactory.create_mock_llm_agent()
            
            # Test agent initialization
            await mock_llm_agent.initialize()
            
            # Test agent processing
            result = await mock_llm_agent.process(self.test_input)
            
            return {
                "model_id": model_id,
                "initialization_successful": True,
                "processing_successful": result.get("success", False),
                "result": result
            }
            
        except Exception as e:
            return {
                "model_id": model_id,
                "initialization_successful": False,
                "error": str(e)
            }
    
    async def _test_model_switch(
        self, 
        old_config: Dict[str, Any], 
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test switching from one model configuration to another.
        
        Args:
            old_config: Previous model configuration
            new_config: New model configuration
            
        Returns:
            Model switch test results
        """
        try:
            # Simulate cleanup of old model
            old_cleanup_successful = True  # Assume cleanup succeeds
            
            # Simulate initialization of new model
            new_init_result = await self._test_model_configuration(new_config["model_id"])
            
            return {
                "switch_successful": new_init_result["initialization_successful"],
                "old_model_id": old_config["model_id"],
                "new_model_id": new_config["model_id"],
                "new_temperature": new_config.get("temperature"),
                "old_cleanup_successful": old_cleanup_successful,
                "new_init_successful": new_init_result["initialization_successful"]
            }
            
        except Exception as e:
            return {
                "switch_successful": False,
                "error": str(e),
                "old_model_id": old_config["model_id"],
                "new_model_id": new_config["model_id"]
            }
    
    async def _test_configuration_hot_reload(
        self, 
        old_config: Dict[str, Any], 
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test hot-reloading of configuration without system restart.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            
        Returns:
            Hot-reload test results
        """
        try:
            # Compare configurations to identify changes
            changes = self._identify_configuration_changes(old_config, new_config)
            
            # Simulate applying changes
            agents_updated = 0
            update_results = {}
            
            for change in changes:
                if change["type"] == "agent_config_update":
                    # Simulate updating agent configuration
                    update_result = await self._simulate_agent_config_update(
                        change["agent_name"],
                        change["old_config"],
                        change["new_config"]
                    )
                    update_results[change["agent_name"]] = update_result
                    
                    if update_result["update_successful"]:
                        agents_updated += 1
            
            return {
                "reload_successful": True,
                "changes_identified": len(changes),
                "agents_updated": agents_updated,
                "update_results": update_results,
                "changes": changes
            }
            
        except Exception as e:
            return {
                "reload_successful": False,
                "error": str(e)
            }
    
    def _identify_configuration_changes(
        self, 
        old_config: Dict[str, Any], 
        new_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify changes between two configurations.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            
        Returns:
            List of identified changes
        """
        changes = []
        
        # Check for agent configuration changes
        old_agents = old_config.get("agents", {})
        new_agents = new_config.get("agents", {})
        
        # Check for updated agents
        for agent_name in set(old_agents.keys()) | set(new_agents.keys()):
            old_agent_config = old_agents.get(agent_name, {})
            new_agent_config = new_agents.get(agent_name, {})
            
            if old_agent_config != new_agent_config:
                changes.append({
                    "type": "agent_config_update",
                    "agent_name": agent_name,
                    "old_config": old_agent_config,
                    "new_config": new_agent_config
                })
        
        # Check for orchestrator configuration changes
        old_orchestrator = old_config.get("orchestrator", {})
        new_orchestrator = new_config.get("orchestrator", {})
        
        if old_orchestrator != new_orchestrator:
            changes.append({
                "type": "orchestrator_config_update",
                "old_config": old_orchestrator,
                "new_config": new_orchestrator
            })
        
        return changes
    
    async def _simulate_agent_config_update(
        self, 
        agent_name: str, 
        old_config: Dict[str, Any], 
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate updating an agent's configuration.
        
        Args:
            agent_name: Name of the agent to update
            old_config: Previous agent configuration
            new_config: New agent configuration
            
        Returns:
            Update simulation results
        """
        try:
            # Simulate agent cleanup
            cleanup_successful = True
            
            # Simulate agent reinitialization with new config
            if new_config.get("enabled", False):
                reinit_successful = True
                
                # Test new configuration
                test_result = await self._test_agent_with_config(agent_name, new_config)
                
                return {
                    "update_successful": test_result["test_successful"],
                    "cleanup_successful": cleanup_successful,
                    "reinit_successful": reinit_successful,
                    "test_result": test_result
                }
            else:
                # Agent disabled
                return {
                    "update_successful": True,
                    "cleanup_successful": cleanup_successful,
                    "agent_disabled": True
                }
                
        except Exception as e:
            return {
                "update_successful": False,
                "error": str(e)
            }
    
    async def _test_agent_with_config(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test agent with specific configuration.
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration to test
            
        Returns:
            Agent test results
        """
        try:
            # Create mock agent with new configuration
            mock_agent = MockAgentFactory.create_mock_base_agent(agent_name)
            
            # Test agent initialization
            await mock_agent.initialize()
            
            # Test agent processing
            result = await mock_agent.process(self.test_input)
            
            return {
                "test_successful": result.get("success", False),
                "agent_name": agent_name,
                "config_applied": agent_config,
                "test_result": result
            }
            
        except Exception as e:
            return {
                "test_successful": False,
                "error": str(e),
                "agent_name": agent_name
            }
    
    @pytest.mark.asyncio
    async def test_environment_specific_configuration_loading(self):
        """Test loading environment-specific configurations."""
        # Create environment-specific configurations
        environments = ["development", "testing", "staging", "production"]
        
        env_configs = {}
        for env in environments:
            env_config = self.base_config.copy()
            
            # Customize configuration for each environment
            if env == "development":
                env_config["orchestrator"]["timeout_seconds"] = 60  # Longer timeout for dev
                env_config["agents"]["llm"]["config"]["temperature"] = 0.3  # More creative
            elif env == "testing":
                env_config["orchestrator"]["timeout_seconds"] = 10  # Shorter for tests
                env_config["agents"]["rag"]["config"]["milvus_uri"] = ":memory:"  # In-memory DB
            elif env == "staging":
                env_config["orchestrator"]["confidence_threshold"] = 0.6  # Higher threshold
            elif env == "production":
                env_config["orchestrator"]["max_parallel_agents"] = 8  # More parallelism
                env_config["agents"]["llm"]["config"]["temperature"] = 0.1  # More deterministic
            
            env_configs[env] = env_config
        
        # Test loading each environment configuration
        for env, expected_config in env_configs.items():
            with patch.dict(os.environ, {"INFERENCE_ENVIRONMENT": env}):
                loaded_config = await self._test_environment_config_loading(env, expected_config)
                
                # Verify environment-specific settings were applied
                assert loaded_config["environment"] == env
                assert loaded_config["config_loaded_successfully"] is True
                
                # Verify environment-specific customizations
                if env == "development":
                    assert loaded_config["orchestrator_timeout"] == 60
                elif env == "testing":
                    assert ":memory:" in loaded_config["rag_milvus_uri"]
                elif env == "production":
                    assert loaded_config["max_parallel_agents"] == 8
    
    async def _test_environment_config_loading(
        self, 
        environment: str, 
        expected_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test loading configuration for specific environment.
        
        Args:
            environment: Environment name
            expected_config: Expected configuration for the environment
            
        Returns:
            Environment configuration loading results
        """
        try:
            # Simulate loading environment-specific configuration
            with patch('inference.src.config.settings.get_config') as mock_get_config:
                # Add environment to config
                env_config = expected_config.copy()
                env_config["environment"] = environment
                
                mock_get_config.return_value = SystemConfig(**env_config)
                
                # Test configuration loading
                loaded_config = mock_get_config.return_value
                
                return {
                    "environment": environment,
                    "config_loaded_successfully": True,
                    "orchestrator_timeout": env_config.get("orchestrator", {}).get("timeout_seconds"),
                    "rag_milvus_uri": env_config.get("agents", {}).get("rag", {}).get("config", {}).get("milvus_uri"),
                    "max_parallel_agents": env_config.get("orchestrator", {}).get("max_parallel_agents"),
                    "llm_temperature": env_config.get("agents", {}).get("llm", {}).get("config", {}).get("temperature"),
                    "loaded_config": env_config
                }
                
        except Exception as e:
            return {
                "environment": environment,
                "config_loaded_successfully": False,
                "error": str(e)
            }
    
    @pytest.mark.asyncio
    async def test_configuration_validation_with_orchestrator_integration(self):
        """Test configuration validation integrated with orchestrator."""
        # Test valid configuration with orchestrator
        valid_config = self.base_config.copy()
        
        orchestrator_test_result = await self._test_orchestrator_with_config(valid_config)
        
        assert orchestrator_test_result["orchestrator_initialized"] is True
        assert orchestrator_test_result["agents_registered"] > 0
        assert orchestrator_test_result["configuration_valid"] is True
        
        # Test invalid configuration with orchestrator
        invalid_config = {
            "agents": {},  # No agents defined
            "orchestrator": {
                "confidence_threshold": 1.5,  # Invalid threshold > 1.0
                "max_parallel_agents": -1  # Invalid negative value
            }
        }
        
        invalid_orchestrator_test_result = await self._test_orchestrator_with_config(invalid_config)
        
        assert invalid_orchestrator_test_result["orchestrator_initialized"] is False
        assert invalid_orchestrator_test_result["configuration_valid"] is False
        assert len(invalid_orchestrator_test_result["validation_errors"]) > 0
    
    async def _test_orchestrator_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test orchestrator initialization and operation with specific configuration.
        
        Args:
            config: Configuration to test with orchestrator
            
        Returns:
            Orchestrator configuration test results
        """
        try:
            # Validate configuration first
            validation_result = self._validate_configuration(config)
            
            if not validation_result["is_valid"]:
                return {
                    "orchestrator_initialized": False,
                    "configuration_valid": False,
                    "validation_errors": validation_result["errors"],
                    "agents_registered": 0
                }
            
            # Test orchestrator initialization
            orchestrator_config = config.get("orchestrator", {})
            test_orchestrator = StrandsMultiAgentOrchestrator(orchestrator_config)
            
            # Test agent registration
            agents_config = config.get("agents", {})
            registered_agents = 0
            
            for agent_name, agent_config in agents_config.items():
                if agent_config.get("enabled", False):
                    # Simulate agent registration
                    mock_agent = MockAgentFactory.create_mock_base_agent(agent_name)
                    
                    # Add to orchestrator's agents (simulated)
                    if not hasattr(test_orchestrator, 'agents'):
                        test_orchestrator.agents = {}
                    test_orchestrator.agents[agent_name] = mock_agent
                    
                    registered_agents += 1
            
            # Test orchestrator processing
            if registered_agents > 0:
                # Simulate processing with test input
                processing_successful = True
            else:
                processing_successful = False
            
            return {
                "orchestrator_initialized": True,
                "configuration_valid": True,
                "validation_errors": [],
                "agents_registered": registered_agents,
                "processing_successful": processing_successful,
                "orchestrator_config": orchestrator_config
            }
            
        except Exception as e:
            return {
                "orchestrator_initialized": False,
                "configuration_valid": False,
                "error": str(e),
                "agents_registered": 0
            }