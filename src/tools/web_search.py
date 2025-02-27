"""
Web Search Tool for AVA6.

This module provides web search capabilities using various search providers
through LangChain's tools integration.
"""

import os
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import logging

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools import Tool

logger = logging.getLogger(__name__)

class SearchProvider(str, Enum):
    """Supported search providers."""
    SERPAPI = "serpapi"
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"

class WebSearchTool:
    """Web search tool that provides access to different search providers."""

    def __init__(self, 
                 provider: Union[str, SearchProvider] = SearchProvider.DUCKDUCKGO,
                 api_key: Optional[str] = None):
        """
        Initialize the web search tool.
        
        Args:
            provider: The search provider to use
            api_key: API key for the provider (if needed)
        """
        self.provider = provider if isinstance(provider, SearchProvider) else SearchProvider(provider)
        self.api_key = api_key
        self._search_tool = self._initialize_search_tool()
        
    def _initialize_search_tool(self) -> Tool:
        """Initialize the appropriate search tool based on the provider."""
        if self.provider == SearchProvider.SERPAPI:
            api_key = self.api_key or os.getenv("SERPAPI_API_KEY")
            if not api_key:
                raise ValueError("SerpAPI requires an API key. Set SERPAPI_API_KEY environment variable or pass api_key.")
            search = SerpAPIWrapper(serpapi_api_key=api_key)
            return Tool(
                name="SerpAPI Search",
                description="Search the web using SerpAPI. Useful for when you need to answer questions about current events or the current state of the world.",
                func=search.run
            )
        
        elif self.provider == SearchProvider.GOOGLE:
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            cse_id = os.getenv("GOOGLE_CSE_ID")
            if not api_key or not cse_id:
                raise ValueError("Google Search requires API key and CSE ID. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
            search = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
            return Tool(
                name="Google Search",
                description="Search the web using Google Custom Search. Useful for when you need to answer questions about current events or the current state of the world.",
                func=search.run
            )
        
        elif self.provider == SearchProvider.TAVILY:
            api_key = self.api_key or os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("Tavily requires an API key. Set TAVILY_API_KEY environment variable or pass api_key.")
            search = TavilySearchAPIWrapper(tavily_api_key=api_key)
            return Tool(
                name="Tavily Search",
                description="Search the web using Tavily. Useful for when you need to answer questions about current events or the current state of the world.",
                func=search.run
            )
        
        elif self.provider == SearchProvider.DUCKDUCKGO:
            # DuckDuckGo doesn't require an API key
            search = DuckDuckGoSearchRun()
            return Tool(
                name="DuckDuckGo Search",
                description="Search the web using DuckDuckGo. Useful for when you need to answer questions about current events or the current state of the world.",
                func=search.run
            )
        
        else:
            raise ValueError(f"Unsupported search provider: {self.provider}")
    
    def search(self, query: str) -> str:
        """
        Perform a web search with the given query.
        
        Args:
            query: The search query
            
        Returns:
            Search results as a string
        """
        try:
            logger.info(f"Searching with {self.provider} for: {query}")
            result = self._search_tool.run(query)
            return result
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return f"Error performing search: {str(e)}"
    
    @property
    def tool(self) -> Tool:
        """Get the LangChain Tool instance for this search provider."""
        return self._search_tool


class WebSearchToolkit:
    """A toolkit that provides access to multiple search providers."""
    
    def __init__(self, 
                 providers: Optional[List[Union[str, SearchProvider]]] = None,
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the web search toolkit with multiple providers.
        
        Args:
            providers: List of search providers to initialize
            api_keys: Dictionary mapping provider names to API keys
        """
        self.api_keys = api_keys or {}
        
        # Default to DuckDuckGo if no providers specified
        self.providers = providers or [SearchProvider.DUCKDUCKGO]
        
        # Initialize search tools for each provider
        self.search_tools = {}
        for provider in self.providers:
            provider_name = provider if isinstance(provider, str) else provider.value
            api_key = self.api_keys.get(provider_name)
            try:
                self.search_tools[provider_name] = WebSearchTool(provider=provider, api_key=api_key)
                logger.info(f"Initialized {provider_name} search provider")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name} search provider: {str(e)}")
    
    def get_tools(self) -> List[Tool]:
        """Get all available search tools as LangChain Tool instances."""
        return [tool.tool for tool in self.search_tools.values()]
    
    def search(self, 
               query: str, 
               provider: Optional[Union[str, SearchProvider]] = None) -> str:
        """
        Perform a web search with the given query using the specified provider.
        
        Args:
            query: The search query
            provider: The specific provider to use (uses first available if None)
            
        Returns:
            Search results as a string
        """
        if provider:
            provider_name = provider if isinstance(provider, str) else provider.value
            if provider_name not in self.search_tools:
                available = list(self.search_tools.keys())
                raise ValueError(f"Provider {provider_name} not available. Available providers: {available}")
            return self.search_tools[provider_name].search(query)
        
        # Use first available provider if none specified
        if not self.search_tools:
            raise ValueError("No search providers available")
        
        first_provider = next(iter(self.search_tools.values()))
        return first_provider.search(query) 