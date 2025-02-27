"""
Test script for web search functionality.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.web_search import WebSearchTool, WebSearchToolkit, SearchProvider

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_web_search():
    """Test the web search functionality."""
    # Load environment variables
    load_dotenv()
    
    # Create a configuration dictionary
    config = {
        "ENABLE_WEB_SEARCH": "true",
        "DEFAULT_SEARCH_PROVIDER": "duckduckgo",
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
    }
    
    # Test individual search tool
    logger.info("Testing individual search tool with DuckDuckGo...")
    search_tool = WebSearchTool(provider="duckduckgo")
    
    # Test a simple search query
    query = "What is the capital of France?"
    logger.info(f"Searching for: {query}")
    result = search_tool.search(query)
    logger.info(f"Search result: {result[:500]}...")
    
    # Test the toolkit with multiple providers
    logger.info("\nTesting search toolkit with multiple providers...")
    try:
        # Try to initialize with DuckDuckGo (no API key needed)
        toolkit = WebSearchToolkit(providers=["duckduckgo"])
        
        # Test the same query
        logger.info(f"Searching for: {query}")
        result = toolkit.search(query)
        logger.info(f"Toolkit search result: {result[:500]}...")
        
        # Get available tools
        tools = toolkit.get_tools()
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        
    except Exception as e:
        logger.error(f"Error testing toolkit: {str(e)}")

if __name__ == "__main__":
    test_web_search() 