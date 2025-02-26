"""
Tool management for the AI Assistant.
"""

import importlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BaseTool:
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool class"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def can_handle(self, query: str) -> bool:
        """
        Check if this tool can handle the given query.
        
        Args:
            query: User query
            
        Returns:
            True if this tool can handle the query, False otherwise
        """
        return False
    
    def execute(self, query: str) -> str:
        """
        Execute the tool on the given query.
        
        Args:
            query: User query
            
        Returns:
            Tool response
        """
        raise NotImplementedError("Tool execution not implemented")


class CalculatorTool(BaseTool):
    """Simple calculator tool."""
    
    name: str = "calculator"
    description: str = "Performs basic arithmetic calculations"
    
    # Regex pattern for identifying math expressions
    MATH_PATTERN = r'(\d+\s*[\+\-\*\/\(\)\^\%]\s*\d+[\+\-\*\/\(\)\^\%\d\s]*)'
    
    def can_handle(self, query: str) -> bool:
        """Check if query contains a math expression."""
        # Check for explicit calculator requests
        calculator_keywords = [
            "calculate", "compute", "math", "arithmetic", "solve", 
            "what is", "what's", "whats", "how much is", "evaluate"
        ]
        
        for keyword in calculator_keywords:
            if keyword in query.lower():
                # Look for math expressions
                if re.search(self.MATH_PATTERN, query):
                    return True
                
                # Look for numbers and operators
                if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', query):
                    return True
        
        return False
    
    def execute(self, query: str) -> str:
        """Evaluate the math expression in the query."""
        try:
            # Extract math expression
            match = re.search(self.MATH_PATTERN, query)
            if not match:
                match = re.search(r'(\d+\s*[\+\-\*\/\^]\s*\d+)', query)
            
            if not match:
                return "No valid math expression found."
            
            expression = match.group(1).strip()
            
            # Replace ^ with ** for exponentiation
            expression = expression.replace("^", "**")
            
            # Safely evaluate the expression
            result = eval(expression, {"__builtins__": {}}, {})
            
            return f"The result of {expression} is {result}"
        
        except Exception as e:
            logger.error(f"Error in calculator tool: {str(e)}")
            return f"Error calculating result: {str(e)}"


class WeatherTool(BaseTool):
    """Weather information tool."""
    
    name: str = "weather"
    description: str = "Provides weather information for a location"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the weather tool."""
        super().__init__(config)
        self.api_key = config.get("WEATHER_API_KEY")
        self.enabled = config.get("ENABLE_WEATHER", "false").lower() in ("true", "1", "yes")
    
    def can_handle(self, query: str) -> bool:
        """Check if query is asking about weather."""
        if not self.enabled or not self.api_key:
            return False
        
        weather_keywords = [
            "weather", "temperature", "forecast", "rain", "sunny", 
            "cloudy", "humidity", "wind", "climate"
        ]
        
        query_lower = query.lower()
        
        # Check for weather keywords
        for keyword in weather_keywords:
            if keyword in query_lower:
                # Look for location indicators
                location_indicators = [
                    "in ", "at ", "for ", "of ", "around ", "near "
                ]
                
                for indicator in location_indicators:
                    if indicator in query_lower:
                        return True
        
        return False
    
    def execute(self, query: str) -> str:
        """Get weather information for the location in the query."""
        if not self.enabled:
            return "Weather tool is disabled."
        
        if not self.api_key:
            return "Weather API key not configured."
        
        try:
            # Extract location from query
            location_match = re.search(r'(?:in|at|for|of|around|near)\s+([a-zA-Z\s,]+)', query)
            
            if not location_match:
                return "Could not determine location from query."
            
            location = location_match.group(1).strip()
            
            # In a real implementation, we would call a weather API here
            # For now, return a placeholder
            return f"Weather information for {location} is not available in this demo version."
        
        except Exception as e:
            logger.error(f"Error in weather tool: {str(e)}")
            return f"Error getting weather information: {str(e)}"


class ToolManager:
    """
    Tool manager for the AI Assistant.
    
    This class manages the available tools and determines which tool
    to use for a given query.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tool manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tools: Dict[str, BaseTool] = {}
        
        # Initialize built-in tools
        self._init_tools()
        
        logger.info(f"Tool manager initialized with {len(self.tools)} tools")
    
    def _init_tools(self) -> None:
        """Initialize the available tools."""
        # Add calculator tool if enabled
        calculator_enabled = self.config.get("ENABLE_CALCULATOR", "true")
        if isinstance(calculator_enabled, bool):
            enable_calculator = calculator_enabled
        else:
            enable_calculator = calculator_enabled.lower() in ("true", "1", "yes")
            
        if enable_calculator:
            self.tools["calculator"] = CalculatorTool(self.config)
            logger.info("Calculator tool initialized")
        
        # Add weather tool if enabled
        weather_enabled = self.config.get("ENABLE_WEATHER", "false")
        if isinstance(weather_enabled, bool):
            enable_weather = weather_enabled
        else:
            enable_weather = weather_enabled.lower() in ("true", "1", "yes")
            
        if enable_weather:
            self.tools["weather"] = WeatherTool(self.config)
            logger.info("Weather tool initialized")
        
        # Load additional tools from plugins if available
        self._load_plugin_tools()
    
    def _load_plugin_tools(self) -> None:
        """Load tools from plugins directory."""
        try:
            # In a real implementation, we would dynamically load tools from plugins
            # For now, this is a placeholder
            pass
        except Exception as e:
            logger.error(f"Error loading plugin tools: {str(e)}")
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all available tools.
        
        Returns:
            List of available tools
        """
        return list(self.tools.values())
    
    def should_use_tool(self, query: str) -> Optional[str]:
        """
        Determine if a tool should be used for the given query.
        
        Args:
            query: User query
            
        Returns:
            Name of the tool to use, or None if no tool should be used
        """
        for name, tool in self.tools.items():
            if tool.can_handle(query):
                logger.info(f"Tool '{name}' can handle query: {query}")
                return name
        
        return None
    
    def use_tool(self, tool_name: str, query: str) -> Optional[str]:
        """
        Use the specified tool to handle the query.
        
        Args:
            tool_name: Name of the tool to use
            query: User query
            
        Returns:
            Tool response, or None if tool not found
        """
        tool = self.tools.get(tool_name)
        
        if not tool:
            logger.warning(f"Tool '{tool_name}' not found")
            return None
        
        try:
            logger.info(f"Using tool '{tool_name}' for query: {query}")
            response = tool.execute(query)
            logger.info(f"Tool '{tool_name}' response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error using tool '{tool_name}': {str(e)}")
            return f"Error using {tool_name} tool: {str(e)}"
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a new tool to the manager.
        
        Args:
            tool: Tool to add
        """
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the manager.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False otherwise
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Removed tool: {tool_name}")
            return True
        
        logger.warning(f"Tool '{tool_name}' not found, cannot remove")
        return False 