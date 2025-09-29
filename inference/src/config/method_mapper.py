"""
Method-Agent Mapper for consistent API method-to-agent mapping.

This module provides dynamic mapping between API method names and agent implementations,
ensuring consistent terminology throughout the API interface.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from ..agents.registry import AgentRegistry, get_agent_registry
from ..agents.base_agent import BaseAgent


logger = logging.getLogger(__name__)


class MethodAgentMapper:
    """
    Maps API method names to agent implementations with dynamic availability checking.
    
    Provides consistent mapping between API method names and agent implementations,
    ensuring that method availability reflects actual agent availability.
    """
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None):
        """
        Initialize the method-agent mapper.
        
        Args:
            agent_registry: Optional agent registry instance. If None, uses global registry.
        """
        self.agent_registry = agent_registry or get_agent_registry()
        
        # Static method-to-agent mapping
        self._method_mappings = {
            'orchestrator': 'orchestrator_agent',  # Special case - handled separately
            'simple': 'simple',
            'rag': 'rag',
            'hybrid': 'hybrid',
            'ner': 'ner',
            'llm': 'llm',
            'finetuned_nova_llm': 'finetuned_nova_llm'
        }
        
        # Reverse mapping for agent-to-method lookup
        self._agent_to_method = {v: k for k, v in self._method_mappings.items()}
        
        logger.debug(f"MethodAgentMapper initialized with {len(self._method_mappings)} method mappings")
    
    def get_valid_methods(self, include_orchestrator: bool = True) -> List[str]:
        """
        Return list of valid method names based on actual agent availability.
        
        Args:
            include_orchestrator: Whether to include orchestrator method if available
            
        Returns:
            List of valid method names that have corresponding available agents
        """
        valid_methods = []
        
        # Check orchestrator availability separately
        if include_orchestrator:
            valid_methods.append('orchestrator')  # Always include orchestrator as it's handled specially
        
        # Check individual agent availability
        available_agents = self.agent_registry.list_agent_names()
        
        for method, agent_name in self._method_mappings.items():
            if method == 'orchestrator':
                continue  # Already handled above
            
            if agent_name in available_agents:
                valid_methods.append(method)
        
        logger.debug(f"Valid methods: {valid_methods} (from available agents: {available_agents})")
        return valid_methods
    
    def get_agent_for_method(self, method: str) -> Optional[BaseAgent]:
        """
        Get agent instance for given method name.
        
        Args:
            method: Method name to get agent for
            
        Returns:
            Agent instance or None if method is invalid or agent not available
        """
        if method not in self._method_mappings:
            logger.warning(f"Unknown method: {method}")
            return None
        
        if method == 'orchestrator':
            # Orchestrator is handled separately in the server
            logger.debug("Orchestrator method requested - handled separately")
            return None
        
        agent_name = self._method_mappings[method]
        agent = self.agent_registry.get_agent(agent_name)
        
        if agent is None:
            logger.warning(f"Agent '{agent_name}' not available for method '{method}'")
        else:
            logger.debug(f"Retrieved agent '{agent_name}' for method '{method}'")
        
        return agent
    
    def validate_method(self, method: str, include_orchestrator: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate method and return error message if invalid.
        
        Args:
            method: Method name to validate
            include_orchestrator: Whether orchestrator is considered valid
            
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not method:
            return False, "Method parameter is required"
        
        if method not in self._method_mappings:
            valid_methods = self.get_valid_methods(include_orchestrator)
            return False, f"Invalid method '{method}'. Valid methods: {', '.join(valid_methods)}"
        
        # Special handling for orchestrator
        if method == 'orchestrator':
            if include_orchestrator:
                return True, None
            else:
                valid_methods = self.get_valid_methods(include_orchestrator=False)
                return False, f"Orchestrator method not available. Valid methods: {', '.join(valid_methods)}"
        
        # Check if corresponding agent is available
        agent_name = self._method_mappings[method]
        available_agents = self.agent_registry.list_agent_names()
        
        if agent_name not in available_agents:
            valid_methods = self.get_valid_methods(include_orchestrator)
            return False, f"Method '{method}' not available (agent '{agent_name}' not initialized). Valid methods: {', '.join(valid_methods)}"
        
        return True, None
    
    def get_method_for_agent(self, agent_name: str) -> Optional[str]:
        """
        Get method name for given agent name.
        
        Args:
            agent_name: Agent name to get method for
            
        Returns:
            Method name or None if agent name is not mapped
        """
        return self._agent_to_method.get(agent_name)
    
    def get_all_method_mappings(self) -> Dict[str, str]:
        """
        Get all method-to-agent mappings.
        
        Returns:
            Dictionary of method names to agent names
        """
        return self._method_mappings.copy()
    
    def get_method_info(self, method: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a method.
        
        Args:
            method: Method name to get info for
            
        Returns:
            Dictionary with method information or None if method is invalid
        """
        if method not in self._method_mappings:
            return None
        
        agent_name = self._method_mappings[method]
        is_available = False
        agent_instance = None
        
        if method == 'orchestrator':
            # Orchestrator availability is handled separately
            is_available = True  # Assume available for info purposes
        else:
            agent_instance = self.agent_registry.get_agent(agent_name)
            is_available = agent_instance is not None
        
        return {
            'method': method,
            'agent_name': agent_name,
            'is_available': is_available,
            'agent_initialized': agent_instance.is_initialized() if agent_instance else False
        }
    
    def get_availability_summary(self) -> Dict[str, Any]:
        """
        Get summary of method availability.
        
        Returns:
            Dictionary with availability summary
        """
        all_methods = list(self._method_mappings.keys())
        valid_methods = self.get_valid_methods(include_orchestrator=True)
        unavailable_methods = [m for m in all_methods if m not in valid_methods]
        
        method_details = {}
        for method in all_methods:
            method_details[method] = self.get_method_info(method)
        
        return {
            'total_methods': len(all_methods),
            'available_methods': len(valid_methods),
            'unavailable_methods': len(unavailable_methods),
            'valid_methods': valid_methods,
            'unavailable_methods_list': unavailable_methods,
            'method_details': method_details
        }
    
    def refresh_agent_registry(self, new_registry: Optional[AgentRegistry] = None) -> None:
        """
        Refresh the agent registry reference.
        
        Args:
            new_registry: New agent registry instance. If None, uses global registry.
        """
        old_registry = self.agent_registry
        self.agent_registry = new_registry or get_agent_registry()
        
        if old_registry != self.agent_registry:
            logger.info("Agent registry refreshed in MethodAgentMapper")
        else:
            logger.debug("Agent registry refresh - same instance")