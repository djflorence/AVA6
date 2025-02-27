"""
Test script for evaluating AVA6's searching and information retrieval capabilities.

This script tests:
1. Web search functionality with different queries
2. Memory retrieval capabilities
3. Tool selection for different types of queries
4. Integration between search and memory
"""

import os
import sys
import logging
import time
import uuid
import json
from dotenv import load_dotenv
from datetime import datetime

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant.chat_assistant import ChatAssistant
from src.tools.web_search import WebSearchTool, WebSearchToolkit
from src.tools.tool_manager import ToolManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchCapabilityTester:
    """Class to test AVA6's search and information retrieval capabilities."""
    
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
            "DEFAULT_LLM_MODEL": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "MEMORY_TYPE": "simple",  # Use simple memory for testing
            "ENABLE_EMOTIONS": "false",
            "ENABLE_CALCULATOR": "true",
            "TEMPERATURE": 0.0,
            "MAX_TOKENS": 500,
        }
        
        # Generate a unique conversation ID for this test session
        self.conversation_id = str(uuid.uuid4())
        
        # Initialize the assistant
        self.assistant = None
        try:
            self.assistant = ChatAssistant(self.config)
            logger.info("Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {str(e)}")
            raise
    
    def test_web_search_queries(self):
        """Test various web search queries."""
        logger.info("\n=== Testing Web Search Queries ===")
        
        # List of test queries
        search_queries = [
            "What is the capital of France?",
            "Who is the current CEO of Microsoft?",
            "What are the latest developments in AI?",
            "What is the population of Tokyo?",
            "When was the first iPhone released?"
        ]
        
        results = []
        for query in search_queries:
            logger.info(f"\nTesting query: {query}")
            try:
                # Get response from assistant
                response = self.assistant.chat(query)
                logger.info(f"Response: {response[:200]}...")
                
                # Check if web search was used
                tool_used = self.assistant.tool_manager.should_use_tool(query)
                logger.info(f"Tool selected: {tool_used}")
                
                results.append({
                    "query": query,
                    "response": response,
                    "tool_used": tool_used
                })
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error testing query '{query}': {str(e)}")
        
        return results
    
    def test_memory_retrieval(self):
        """Test memory retrieval capabilities."""
        logger.info("\n=== Testing Memory Retrieval ===")
        
        # First, add some information to memory
        facts = [
            "My name is David",
            "I live in Boston",
            "I'm married to Marisa (Reece)",
            "I have a son named Owen",
            "I work as a software developer and digital artist"
        ]
        
        # Store facts in memory
        for fact in facts:
            logger.info(f"Storing fact: {fact}")
            response = self.assistant.chat(fact)
            logger.info(f"Response: {response[:100]}...")
            time.sleep(1)
        
        # Now test retrieval with related queries
        retrieval_queries = [
            "What is my name?",
            "Where do I live?",
            "Do I have any pets?",
            "What is my favorite color?",
            "What is my profession?"
        ]
        
        # Expected information in responses
        expected_info = [
            ["David", "name"],
            ["Boston", "live", "location"],
            ["Marisa", "Reece", "wife", "married"],
            ["Owen", "son"],
            ["software developer", "digital artist", "profession", "work"]
        ]
        
        results = []
        for i, query in enumerate(retrieval_queries):
            logger.info(f"\nTesting memory retrieval: {query}")
            try:
                response = self.assistant.chat(query)
                logger.info(f"Response: {response[:200]}...")
                
                # Check if the response contains the relevant information using semantic matching
                # For each expected info term, check if it or a related term is in the response
                response_lower = response.lower()
                contains_relevant_info = False
                
                for term in expected_info[i]:
                    if term.lower() in response_lower:
                        contains_relevant_info = True
                        logger.info(f"Found relevant term: {term}")
                        break
                
                # Additional semantic checks for specific query types
                if i == 0 and not contains_relevant_info:  # Name query
                    if "your name is" in response_lower or "you are" in response_lower:
                        contains_relevant_info = True
                        logger.info("Found semantic match for name query")
                
                if i == 1 and not contains_relevant_info:  # Location query
                    if "you live in" in response_lower or "you are from" in response_lower or "your location" in response_lower:
                        contains_relevant_info = True
                        logger.info("Found semantic match for location query")
                
                if i == 2 and not contains_relevant_info:  # Pet query
                    if "you have" in response_lower and ("pet" in response_lower or "dog" in response_lower):
                        contains_relevant_info = True
                        logger.info("Found semantic match for pet query")
                
                if i == 3 and not contains_relevant_info:  # Color query
                    if "your favorite" in response_lower and "color" in response_lower:
                        contains_relevant_info = True
                        logger.info("Found semantic match for color query")
                
                if i == 4 and not contains_relevant_info:  # Profession query
                    if "you work" in response_lower or "your job" in response_lower or "your profession" in response_lower:
                        contains_relevant_info = True
                        logger.info("Found semantic match for profession query")
                
                logger.info(f"Contains relevant information: {contains_relevant_info}")
                
                results.append({
                    "query": query,
                    "response": response,
                    "contains_relevant_info": contains_relevant_info,
                    "expected_terms": expected_info[i]
                })
                
                # Add a short delay between queries
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during memory retrieval test: {str(e)}")
                results.append({
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "contains_relevant_info": False,
                    "expected_terms": expected_info[i]
                })
        
        return results
    
    def test_combined_capabilities(self):
        """Test combined search and memory capabilities."""
        logger.info("\n=== Testing Combined Search and Memory Capabilities ===")
        
        # First, add some personal context
        self.assistant.chat("I'm planning a trip to Paris next month")
        time.sleep(1)
        self.assistant.chat("I'm interested in art and history")
        time.sleep(1)
        self.assistant.chat("I'll be staying for 7 days")
        time.sleep(1)
        
        # Now ask questions that require both memory and web search
        combined_queries = [
            "What are some must-see attractions in Paris for someone interested in art?",
            "What's the best time of year to visit Paris and will next month be good?",
            "Can you recommend some French phrases I should learn for my trip?",
            "What's the best way to get around Paris during my 7-day stay?",
            "What should I know about French cuisine for my visit to Paris?"
        ]
        
        # Expected personal context elements
        expected_personal_elements = [
            ["paris", "trip", "art", "next month"],
            ["paris", "trip", "next month"],
            ["paris", "trip", "french phrases"],
            ["paris", "7-day", "stay"],
            ["paris", "visit", "french cuisine"]
        ]
        
        results = []
        for i, query in enumerate(combined_queries):
            logger.info(f"\nTesting combined capabilities: {query}")
            try:
                response = self.assistant.chat(query)
                logger.info(f"Response: {response[:200]}...")
                
                # Check if the response references personal context elements
                response_lower = response.lower()
                personal_elements_found = [element for element in expected_personal_elements[i] 
                                          if element in response_lower]
                
                # Check if the response contains search information (more sophisticated check)
                contains_search_info = len(response) > 150 and any(
                    search_indicator in response_lower for search_indicator in 
                    ["according to", "popular", "recommended", "famous", "known for", "you can", "visitors"]
                )
                
                # Calculate personalization score (0-100%)
                personalization_score = (len(personal_elements_found) / len(expected_personal_elements[i])) * 100
                
                logger.info(f"Personal elements found: {personal_elements_found}")
                logger.info(f"Personalization score: {personalization_score:.1f}%")
                logger.info(f"Contains search information: {contains_search_info}")
                
                # Consider the test successful if personalization score > 50% and contains search info
                success = personalization_score >= 50 and contains_search_info
                
                results.append({
                    "query": query,
                    "response": response,
                    "personal_elements_found": personal_elements_found,
                    "personalization_score": personalization_score,
                    "contains_search_info": contains_search_info,
                    "success": success
                })
                
                # Add a small delay
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during combined capabilities test: {str(e)}")
                results.append({
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "personal_elements_found": [],
                    "personalization_score": 0,
                    "contains_search_info": False,
                    "success": False
                })
        
        # Calculate success rate
        successful_tests = sum(1 for result in results if result["success"])
        success_rate = (successful_tests / len(combined_queries)) * 100
        logger.info(f"\nCombined capabilities success rate: {success_rate:.1f}% ({successful_tests}/{len(combined_queries)})")
        
        return results
    
    def test_tool_selection(self):
        """Test tool selection for different types of queries."""
        logger.info("\n=== Testing Tool Selection ===")
        
        # Test queries for different tools
        tool_queries = [
            {"query": "What is 25 * 16?", "expected_tool": "calculator"},
            {"query": "Search for recent breakthroughs in quantum computing", "expected_tool": "web_search"},
            {"query": "Tell me about the history of Rome", "expected_tool": "web_search"},
            {"query": "What's 15% of 230?", "expected_tool": "calculator"},
            {"query": "Who won the last World Cup?", "expected_tool": "web_search"}
        ]
        
        results = []
        for test_case in tool_queries:
            query = test_case["query"]
            expected_tool = test_case["expected_tool"]
            
            logger.info(f"\nTesting tool selection: {query}")
            logger.info(f"Expected tool: {expected_tool}")
            
            try:
                # Check which tool is selected
                selected_tool = self.assistant.tool_manager.should_use_tool(query)
                logger.info(f"Selected tool: {selected_tool}")
                
                # Get response
                response = self.assistant.chat(query)
                logger.info(f"Response: {response[:200]}...")
                
                results.append({
                    "query": query,
                    "expected_tool": expected_tool,
                    "selected_tool": selected_tool,
                    "response": response,
                    "used_correct_tool": selected_tool == expected_tool
                })
                
                # Add a small delay
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error testing tool selection '{query}': {str(e)}")
        
        return results
    
    def run_all_tests(self):
        """Run all tests and return the results."""
        logger.info("\n=== Starting Search Capability Tests ===")
        
        # Run web search tests
        web_search_results = self.test_web_search_queries()
        
        # Run memory retrieval tests
        memory_retrieval_results = self.test_memory_retrieval()
        
        # Run combined capabilities tests
        combined_results = self.test_combined_capabilities()
        
        # Run tool selection tests
        tool_selection_results = self.test_tool_selection()
        
        # Calculate overall success rate
        total_queries = (
            len(web_search_results) + 
            len(memory_retrieval_results) + 
            len(combined_results) + 
            len(tool_selection_results)
        )
        
        # Web search success
        web_search_success = sum(1 for r in web_search_results if r["success"])
        
        # Memory retrieval success
        memory_success = sum(1 for r in memory_retrieval_results if r["contains_relevant_info"])
        
        # Combined capabilities success
        combined_success = sum(1 for r in combined_results if r["success"])
        
        # Tool selection success
        tool_success = sum(1 for r in tool_selection_results if r["used_correct_tool"])
        
        # Total success
        total_success = web_search_success + memory_success + combined_success + tool_success
        
        # Print summary
        logger.info("\n=== Test Results Summary ===")
        
        # Web search summary
        logger.info("\nWeb Search Tests:")
        logger.info(f"Queries tested: {len(web_search_results)}")
        logger.info(f"Successful searches: {web_search_success}/{len(web_search_results)}")
        logger.info(f"Success rate: {(web_search_success/len(web_search_results))*100:.1f}%")
        
        # Memory retrieval summary
        logger.info("\nMemory Retrieval Tests:")
        logger.info(f"Queries tested: {len(memory_retrieval_results)}")
        logger.info(f"Successful retrievals: {memory_success}/{len(memory_retrieval_results)}")
        logger.info(f"Success rate: {(memory_success/len(memory_retrieval_results))*100:.1f}%")
        
        # Combined capabilities summary
        logger.info("\nCombined Capabilities Tests:")
        logger.info(f"Queries tested: {len(combined_results)}")
        logger.info(f"Successful combined responses: {combined_success}/{len(combined_results)}")
        logger.info(f"Success rate: {(combined_success/len(combined_results))*100:.1f}%")
        
        # Tool selection summary
        logger.info("\nTool Selection Tests:")
        logger.info(f"Queries tested: {len(tool_selection_results)}")
        logger.info(f"Successful tool selections: {tool_success}/{len(tool_selection_results)}")
        logger.info(f"Success rate: {(tool_success/len(tool_selection_results))*100:.1f}%")
        
        # Overall summary
        logger.info("\nOverall Results:")
        logger.info(f"Total queries tested: {total_queries}")
        logger.info(f"Total successful responses: {total_success}/{total_queries}")
        logger.info(f"Overall success rate: {(total_success/total_queries)*100:.1f}%")
        
        # Save results to file
        results = {
            "timestamp": datetime.now().isoformat(),
            "web_search": {
                "queries": len(web_search_results),
                "success": web_search_success,
                "rate": (web_search_success/len(web_search_results))*100
            },
            "memory_retrieval": {
                "queries": len(memory_retrieval_results),
                "success": memory_success,
                "rate": (memory_success/len(memory_retrieval_results))*100
            },
            "combined_capabilities": {
                "queries": len(combined_results),
                "success": combined_success,
                "rate": (combined_success/len(combined_results))*100
            },
            "tool_selection": {
                "queries": len(tool_selection_results),
                "success": tool_success,
                "rate": (tool_success/len(tool_selection_results))*100
            },
            "overall": {
                "queries": total_queries,
                "success": total_success,
                "rate": (total_success/total_queries)*100
            }
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("src/logs", exist_ok=True)
        
        # Save results to JSON file
        results_file = f"src/logs/search_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        
        return results

def main():
    """Main function to run the tests."""
    logger.info("Starting search capabilities test")
    
    try:
        # Create and run the tester
        tester = SearchCapabilityTester()
        results = tester.run_all_tests()
        
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main() 