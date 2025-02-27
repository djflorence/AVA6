"""
Comprehensive test script for AVA6's web search capabilities.

This script tests the web search functionality with different types of queries:
1. Factual queries (e.g., "What is the capital of France?")
2. Current events queries (e.g., "What are the latest news about AI?")
3. How-to queries (e.g., "How to make pasta carbonara?")
4. Comparison queries (e.g., "Compare Python vs JavaScript")
5. Opinion-based queries (e.g., "What are the best programming languages to learn?")
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.web_search import WebSearchTool, WebSearchToolkit, SearchProvider

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSearchTester:
    """Class to test AVA6's web search capabilities."""
    
    def __init__(self):
        """Initialize the tester with configuration."""
        # Load environment variables
        load_dotenv()
        
        # Create a configuration dictionary
        self.config = {
            "ENABLE_WEB_SEARCH": "true",
            "DEFAULT_SEARCH_PROVIDER": "duckduckgo",
            "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        }
        
        # Initialize the search tool
        self.search_tool = None
        try:
            self.search_tool = WebSearchTool(config=self.config)
            logger.info("Web search tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web search tool: {str(e)}")
            raise
        
        # Create a directory for storing search results
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create a log file for this test session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(self.results_dir, f"web_search_test_{timestamp}.json")
        self.search_results = []
    
    def test_search_query(self, query, category):
        """
        Test a single search query.
        
        Args:
            query: The search query to test
            category: The category of the query (e.g., factual, current events)
            
        Returns:
            Dictionary with query, result, and metadata
        """
        logger.info(f"\nTesting {category} query: {query}")
        
        try:
            # Check if the tool can handle the query
            can_handle = self.search_tool.can_handle(query)
            logger.info(f"Can handle query: {can_handle}")
            
            # Execute the search
            start_time = time.time()
            result = self.search_tool.execute(query)
            end_time = time.time()
            
            # Log the result
            logger.info(f"Search result: {result[:200]}...")
            logger.info(f"Search time: {end_time - start_time:.2f} seconds")
            
            # Record the result
            search_result = {
                "query": query,
                "category": category,
                "can_handle": can_handle,
                "result": result,
                "search_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.search_results.append(search_result)
            self._save_results()
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error testing query '{query}': {str(e)}")
            
            # Record the error
            error_result = {
                "query": query,
                "category": category,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.search_results.append(error_result)
            self._save_results()
            
            return error_result
    
    def _save_results(self):
        """Save the search results to a JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.search_results, f, indent=2)
    
    def test_factual_queries(self):
        """Test factual queries."""
        logger.info("\n=== Testing Factual Queries ===")
        
        factual_queries = [
            "What is the capital of France?",
            "Who is the CEO of Microsoft?",
            "What is the population of Tokyo?",
            "When was the first iPhone released?",
            "What is the height of Mount Everest?",
            "Who wrote the book 'To Kill a Mockingbird'?",
            "What is the chemical formula for water?",
            "What is the speed of light?",
            "Who was the first person to walk on the moon?",
            "What is the largest planet in our solar system?"
        ]
        
        results = []
        for query in factual_queries:
            result = self.test_search_query(query, "factual")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_current_events_queries(self):
        """Test current events queries."""
        logger.info("\n=== Testing Current Events Queries ===")
        
        current_events_queries = [
            "What are the latest developments in AI?",
            "What are the recent news about climate change?",
            "What are the current trends in technology?",
            "What are the latest breakthroughs in medicine?",
            "What are the recent developments in space exploration?"
        ]
        
        results = []
        for query in current_events_queries:
            result = self.test_search_query(query, "current_events")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_how_to_queries(self):
        """Test how-to queries."""
        logger.info("\n=== Testing How-To Queries ===")
        
        how_to_queries = [
            "How to make pasta carbonara?",
            "How to learn Python programming?",
            "How to fix a leaky faucet?",
            "How to improve productivity?",
            "How to start a small business?"
        ]
        
        results = []
        for query in how_to_queries:
            result = self.test_search_query(query, "how_to")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_comparison_queries(self):
        """Test comparison queries."""
        logger.info("\n=== Testing Comparison Queries ===")
        
        comparison_queries = [
            "Compare Python vs JavaScript",
            "What's the difference between a virus and bacteria?",
            "Compare electric cars vs gas cars",
            "What's better: Windows or macOS?",
            "Compare React vs Angular for web development"
        ]
        
        results = []
        for query in comparison_queries:
            result = self.test_search_query(query, "comparison")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_opinion_based_queries(self):
        """Test opinion-based queries."""
        logger.info("\n=== Testing Opinion-Based Queries ===")
        
        opinion_queries = [
            "What are the best programming languages to learn?",
            "What are the most beautiful places to visit in Europe?",
            "What are the best books to read in 2023?",
            "What are the most promising technologies for the future?",
            "What are the best practices for remote work?"
        ]
        
        results = []
        for query in opinion_queries:
            result = self.test_search_query(query, "opinion")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_complex_queries(self):
        """Test complex queries."""
        logger.info("\n=== Testing Complex Queries ===")
        
        complex_queries = [
            "What are the implications of quantum computing on cryptography?",
            "How does climate change affect biodiversity in coral reefs?",
            "What are the ethical considerations of artificial intelligence in healthcare?",
            "How has remote work affected urban development and housing markets?",
            "What are the challenges of implementing renewable energy on a global scale?"
        ]
        
        results = []
        for query in complex_queries:
            result = self.test_search_query(query, "complex")
            results.append(result)
            time.sleep(1)  # Add a small delay to avoid rate limiting
        
        return results
    
    def test_all_providers(self, query="What is artificial intelligence?"):
        """
        Test all available search providers with the same query.
        
        Args:
            query: The query to test with all providers
        """
        logger.info("\n=== Testing All Search Providers ===")
        logger.info(f"Query: {query}")
        
        # List of providers to test
        providers = ["duckduckgo"]
        
        # Add other providers if API keys are available
        if os.getenv("SERPAPI_API_KEY"):
            providers.append("serpapi")
        
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
            providers.append("google")
        
        if os.getenv("TAVILY_API_KEY"):
            providers.append("tavily")
        
        results = []
        for provider in providers:
            logger.info(f"\nTesting provider: {provider}")
            
            try:
                # Initialize a toolkit with just this provider
                api_keys = {
                    "serpapi": os.getenv("SERPAPI_API_KEY"),
                    "google": os.getenv("GOOGLE_API_KEY"),
                    "tavily": os.getenv("TAVILY_API_KEY")
                }
                
                toolkit = WebSearchToolkit(
                    providers=[provider],
                    api_keys=api_keys
                )
                
                # Execute the search
                start_time = time.time()
                result = toolkit.search(query)
                end_time = time.time()
                
                # Log the result
                logger.info(f"Search result: {result[:200]}...")
                logger.info(f"Search time: {end_time - start_time:.2f} seconds")
                
                # Record the result
                provider_result = {
                    "query": query,
                    "provider": provider,
                    "result": result,
                    "search_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(provider_result)
                self.search_results.append(provider_result)
                self._save_results()
                
                time.sleep(1)  # Add a small delay to avoid rate limiting
                
            except Exception as e:
                logger.error(f"Error testing provider '{provider}': {str(e)}")
                
                # Record the error
                error_result = {
                    "query": query,
                    "provider": provider,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(error_result)
                self.search_results.append(error_result)
                self._save_results()
        
        return results
    
    def run_all_tests(self):
        """Run all search tests and report results."""
        logger.info("\n=== Starting All Web Search Tests ===")
        
        try:
            # Run all tests
            factual_results = self.test_factual_queries()
            current_events_results = self.test_current_events_queries()
            how_to_results = self.test_how_to_queries()
            comparison_results = self.test_comparison_queries()
            opinion_results = self.test_opinion_based_queries()
            complex_results = self.test_complex_queries()
            provider_results = self.test_all_providers()
            
            # Summarize results
            logger.info("\n=== Test Results Summary ===")
            
            # Count successful searches (non-error results)
            factual_success = sum(1 for r in factual_results if "error" not in r)
            current_events_success = sum(1 for r in current_events_results if "error" not in r)
            how_to_success = sum(1 for r in how_to_results if "error" not in r)
            comparison_success = sum(1 for r in comparison_results if "error" not in r)
            opinion_success = sum(1 for r in opinion_results if "error" not in r)
            complex_success = sum(1 for r in complex_results if "error" not in r)
            provider_success = sum(1 for r in provider_results if "error" not in r)
            
            # Calculate average search times
            factual_times = [r.get("search_time", 0) for r in factual_results if "search_time" in r]
            current_events_times = [r.get("search_time", 0) for r in current_events_results if "search_time" in r]
            how_to_times = [r.get("search_time", 0) for r in how_to_results if "search_time" in r]
            comparison_times = [r.get("search_time", 0) for r in comparison_results if "search_time" in r]
            opinion_times = [r.get("search_time", 0) for r in opinion_results if "search_time" in r]
            complex_times = [r.get("search_time", 0) for r in complex_results if "search_time" in r]
            provider_times = [r.get("search_time", 0) for r in provider_results if "search_time" in r]
            
            factual_avg_time = sum(factual_times) / len(factual_times) if factual_times else 0
            current_events_avg_time = sum(current_events_times) / len(current_events_times) if current_events_times else 0
            how_to_avg_time = sum(how_to_times) / len(how_to_times) if how_to_times else 0
            comparison_avg_time = sum(comparison_times) / len(comparison_times) if comparison_times else 0
            opinion_avg_time = sum(opinion_times) / len(opinion_times) if opinion_times else 0
            complex_avg_time = sum(complex_times) / len(complex_times) if complex_times else 0
            provider_avg_time = sum(provider_times) / len(provider_times) if provider_times else 0
            
            # Log summary for each category
            logger.info(f"\nFactual Queries:")
            logger.info(f"Success rate: {factual_success}/{len(factual_results)}")
            logger.info(f"Average search time: {factual_avg_time:.2f} seconds")
            
            logger.info(f"\nCurrent Events Queries:")
            logger.info(f"Success rate: {current_events_success}/{len(current_events_results)}")
            logger.info(f"Average search time: {current_events_avg_time:.2f} seconds")
            
            logger.info(f"\nHow-To Queries:")
            logger.info(f"Success rate: {how_to_success}/{len(how_to_results)}")
            logger.info(f"Average search time: {how_to_avg_time:.2f} seconds")
            
            logger.info(f"\nComparison Queries:")
            logger.info(f"Success rate: {comparison_success}/{len(comparison_results)}")
            logger.info(f"Average search time: {comparison_avg_time:.2f} seconds")
            
            logger.info(f"\nOpinion-Based Queries:")
            logger.info(f"Success rate: {opinion_success}/{len(opinion_results)}")
            logger.info(f"Average search time: {opinion_avg_time:.2f} seconds")
            
            logger.info(f"\nComplex Queries:")
            logger.info(f"Success rate: {complex_success}/{len(complex_results)}")
            logger.info(f"Average search time: {complex_avg_time:.2f} seconds")
            
            logger.info(f"\nProvider Tests:")
            logger.info(f"Success rate: {provider_success}/{len(provider_results)}")
            logger.info(f"Average search time: {provider_avg_time:.2f} seconds")
            
            # Overall summary
            total_queries = len(factual_results) + len(current_events_results) + len(how_to_results) + \
                           len(comparison_results) + len(opinion_results) + len(complex_results) + len(provider_results)
            total_success = factual_success + current_events_success + how_to_success + \
                           comparison_success + opinion_success + complex_success + provider_success
            
            all_times = factual_times + current_events_times + how_to_times + \
                       comparison_times + opinion_times + complex_times + provider_times
            overall_avg_time = sum(all_times) / len(all_times) if all_times else 0
            
            logger.info(f"\nOverall Results:")
            logger.info(f"Total queries: {total_queries}")
            logger.info(f"Success rate: {total_success}/{total_queries} ({total_success/total_queries*100:.2f}%)")
            logger.info(f"Average search time: {overall_avg_time:.2f} seconds")
            
            # Log the location of the results file
            logger.info(f"\nSearch results saved to: {self.results_file}")
            
            return {
                "factual_results": factual_results,
                "current_events_results": current_events_results,
                "how_to_results": how_to_results,
                "comparison_results": comparison_results,
                "opinion_results": opinion_results,
                "complex_results": complex_results,
                "provider_results": provider_results,
                "summary": {
                    "factual_success_rate": factual_success / len(factual_results) if factual_results else 0,
                    "current_events_success_rate": current_events_success / len(current_events_results) if current_events_results else 0,
                    "how_to_success_rate": how_to_success / len(how_to_results) if how_to_results else 0,
                    "comparison_success_rate": comparison_success / len(comparison_results) if comparison_results else 0,
                    "opinion_success_rate": opinion_success / len(opinion_results) if opinion_results else 0,
                    "complex_success_rate": complex_success / len(complex_results) if complex_results else 0,
                    "provider_success_rate": provider_success / len(provider_results) if provider_results else 0,
                    "overall_success_rate": total_success / total_queries if total_queries else 0,
                    "factual_avg_time": factual_avg_time,
                    "current_events_avg_time": current_events_avg_time,
                    "how_to_avg_time": how_to_avg_time,
                    "comparison_avg_time": comparison_avg_time,
                    "opinion_avg_time": opinion_avg_time,
                    "complex_avg_time": complex_avg_time,
                    "provider_avg_time": provider_avg_time,
                    "overall_avg_time": overall_avg_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            raise

def main():
    """Main function to run the tests."""
    logger.info("Starting web search capability test")
    
    try:
        # Create and run the tester
        tester = WebSearchTester()
        results = tester.run_all_tests()
        
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main() 