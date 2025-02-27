"""
Tool management for the AI Assistant.
"""

import importlib
import logging
import re
import os
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


class WebSearchTool(BaseTool):
    """Web search tool."""
    
    name: str = "web_search"
    description: str = "Searches the web for information"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the web search tool."""
        super().__init__(config)
        
        # Handle both string and boolean values for ENABLE_WEB_SEARCH
        enable_web_search = config.get("ENABLE_WEB_SEARCH", "false")
        if isinstance(enable_web_search, bool):
            self.enabled = enable_web_search
        else:
            self.enabled = enable_web_search.lower() in ("true", "1", "yes")
        
        if not self.enabled:
            logger.info("Web search tool is disabled")
            return
        
        try:
            # Import the web search module
            from src.tools.web_search import WebSearchToolkit, SearchProvider
            
            # Get the default search provider from config
            default_provider = config.get("DEFAULT_SEARCH_PROVIDER", "duckduckgo").lower()
            
            # Initialize the web search toolkit
            api_keys = {
                "serpapi": config.get("SERPAPI_API_KEY"),
                "google": config.get("GOOGLE_API_KEY"),
                "tavily": config.get("TAVILY_API_KEY")
            }
            
            # Filter out None values
            api_keys = {k: v for k, v in api_keys.items() if v}
            
            # Initialize with the default provider
            self.search_toolkit = WebSearchToolkit(
                providers=[default_provider],
                api_keys=api_keys
            )
            
            logger.info(f"Web search tool initialized with provider: {default_provider}")
        except Exception as e:
            logger.error(f"Error initializing web search tool: {str(e)}")
            self.enabled = False
    
    def can_handle(self, query: str) -> bool:
        """Check if query is asking for web search."""
        if not self.enabled:
            return False
        
        # Check for explicit search requests
        search_keywords = [
            "search", "look up", "find", "google", "web", "internet",
            "information about", "tell me about", "what is", "who is",
            "where is", "when did", "how to", "latest", "news about",
            "current", "recent"
        ]
        
        query_lower = query.lower()
        
        # Check for search keywords
        for keyword in search_keywords:
            if keyword in query_lower:
                return True
        
        # Check for question patterns that might benefit from web search
        question_patterns = [
            r'^what\s+is\s+',
            r'^who\s+is\s+',
            r'^where\s+is\s+',
            r'^when\s+did\s+',
            r'^how\s+to\s+',
            r'^why\s+did\s+',
            r'^can\s+you\s+find\s+'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def execute(self, query: str) -> str:
        """Search the web for the query."""
        if not self.enabled:
            return "Web search tool is disabled."
        
        try:
            # Extract the search query
            # For simplicity, we'll use the entire query
            search_query = query
            
            # Remove common prefixes to get a cleaner search query
            prefixes_to_remove = [
                "search for ", "look up ", "find ", "google ", 
                "search the web for ", "can you find ", "tell me about ",
                "what is ", "who is ", "where is ", "when did ", "how to "
            ]
            
            for prefix in prefixes_to_remove:
                if search_query.lower().startswith(prefix):
                    search_query = search_query[len(prefix):]
                    break
            
            # Perform the search
            result = self.search_toolkit.search(search_query)
            
            # Format the result
            if result:
                return f"Web search results for '{search_query}':\n\n{result}"
            else:
                return f"No results found for '{search_query}'."
        
        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}")
            return f"Error searching the web: {str(e)}"


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
        
        # Add web search tool if enabled
        web_search_enabled = self.config.get("ENABLE_WEB_SEARCH", "false")
        if isinstance(web_search_enabled, bool):
            enable_web_search = web_search_enabled
        else:
            enable_web_search = web_search_enabled.lower() in ("true", "1", "yes")
            
        if enable_web_search:
            self.tools["web_search"] = WebSearchTool(self.config)
            logger.info("Web search tool initialized")
        
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