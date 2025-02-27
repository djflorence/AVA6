"""
Interactive chat test script for AVA6.

This script simulates a conversation with AVA6 to test its searching
and information retrieval capabilities in a more natural way.
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

from src.assistant.chat_assistant import ChatAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatInteractionTester:
    """Class to test AVA6's capabilities through interactive chat."""
    
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
            "TEMPERATURE": 0.7,  # Higher temperature for more varied responses
            "MAX_TOKENS": 500,
        }
        
        # Initialize the assistant
        self.assistant = None
        try:
            self.assistant = ChatAssistant(self.config)
            logger.info("Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {str(e)}")
            raise
        
        # Create a directory for storing conversation logs
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a log file for this conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"chat_test_{timestamp}.json")
        self.conversation_log = []
    
    def run_conversation(self, conversation_script):
        """
        Run a scripted conversation with the assistant.
        
        Args:
            conversation_script: List of dictionaries with user messages and expected info
        
        Returns:
            List of dictionaries with user messages, assistant responses, and evaluation
        """
        logger.info("Starting scripted conversation")
        
        results = []
        
        for i, script_item in enumerate(conversation_script):
            user_message = script_item["user_message"]
            expected_info = script_item.get("expected_info", [])
            expected_tool = script_item.get("expected_tool", None)
            
            logger.info(f"\n[Turn {i+1}] User: {user_message}")
            
            # Get response from assistant
            start_time = time.time()
            response = self.assistant.chat(user_message)
            end_time = time.time()
            
            logger.info(f"[Turn {i+1}] Assistant: {response}")
            
            # Check if expected information is in the response
            info_present = []
            for info in expected_info:
                is_present = info.lower() in response.lower()
                info_present.append(is_present)
                logger.info(f"Expected info '{info}': {'Present' if is_present else 'Not present'}")
            
            # Check if expected tool was used
            tool_used = self.assistant.tool_manager.should_use_tool(user_message)
            tool_correct = (tool_used == expected_tool) if expected_tool else True
            
            if expected_tool:
                logger.info(f"Expected tool: {expected_tool}, Tool used: {tool_used}")
            
            # Record the results
            result = {
                "turn": i + 1,
                "user_message": user_message,
                "assistant_response": response,
                "response_time": end_time - start_time,
                "expected_info": expected_info,
                "info_present": info_present,
                "info_accuracy": sum(info_present) / len(info_present) if expected_info else 1.0,
                "expected_tool": expected_tool,
                "tool_used": tool_used,
                "tool_correct": tool_correct
            }
            
            results.append(result)
            self.conversation_log.append(result)
            
            # Save the conversation log after each turn
            self._save_conversation_log()
            
            # Add a small delay between turns
            time.sleep(1)
        
        return results
    
    def _save_conversation_log(self):
        """Save the conversation log to a JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
    
    def run_search_test(self):
        """Run a test focused on search capabilities."""
        logger.info("\n=== Running Search Capability Test ===")
        
        search_conversation = [
            {
                "user_message": "Hello, I'd like to test your search capabilities.",
                "expected_info": ["search", "help", "information"],
                "expected_tool": None
            },
            {
                "user_message": "What is the capital of France?",
                "expected_info": ["Paris"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "Who is the current CEO of Microsoft?",
                "expected_info": ["Satya Nadella"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "Can you tell me about the latest developments in AI?",
                "expected_info": ["artificial intelligence", "recent", "development"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "What's 25 * 16?",
                "expected_info": ["400"],
                "expected_tool": "calculator"
            },
            {
                "user_message": "Now, can you find information about climate change?",
                "expected_info": ["climate", "global warming", "temperature"],
                "expected_tool": "web_search"
            }
        ]
        
        return self.run_conversation(search_conversation)
    
    def run_memory_test(self):
        """Run a test focused on memory capabilities."""
        logger.info("\n=== Running Memory Capability Test ===")
        
        memory_conversation = [
            {
                "user_message": "My name is David and I live in Boston.",
                "expected_info": ["nice", "meet", "hello"],
                "expected_tool": None
            },
            {
                "user_message": "I'm married to Marisa, who goes by Reece, and we have a son named Owen.",
                "expected_info": ["family", "wonderful"],
                "expected_tool": None
            },
            {
                "user_message": "I work as a software developer and digital artist.",
                "expected_info": ["interesting", "software", "creative"],
                "expected_tool": None
            },
            {
                "user_message": "What's my name?",
                "expected_info": ["David"],
                "expected_tool": None
            },
            {
                "user_message": "Where do I live?",
                "expected_info": ["Boston"],
                "expected_tool": None
            },
            {
                "user_message": "I have a dog named Max and a cat named Luna.",
                "expected_info": ["pets", "animals"],
                "expected_tool": None
            },
            {
                "user_message": "What are my pets' names?",
                "expected_info": ["Max", "Luna", "dog", "cat"],
                "expected_tool": None
            },
            {
                "user_message": "What do I do for work?",
                "expected_info": ["software engineer", "tech company"],
                "expected_tool": None
            },
            {
                "user_message": "Tell me about my family.",
                "expected_info": ["Marisa", "Reece", "Owen"],
                "expected_tool": None
            }
        ]
        
        return self.run_conversation(memory_conversation)
    
    def run_combined_test(self):
        """Run a test that combines memory and search capabilities."""
        logger.info("\n=== Running Combined Capabilities Test ===")
        
        combined_conversation = [
            {
                "user_message": "I'm planning a trip to Japan next month.",
                "expected_info": ["exciting", "travel", "Japan"],
                "expected_tool": None
            },
            {
                "user_message": "What are some must-visit places in Japan?",
                "expected_info": ["Tokyo", "Kyoto", "Mount Fuji"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "I'm particularly interested in Japanese cuisine.",
                "expected_info": ["food", "cuisine", "delicious"],
                "expected_tool": None
            },
            {
                "user_message": "What Japanese dishes should I try during my trip?",
                "expected_info": ["sushi", "ramen", "tempura"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "I'll be traveling with my family, including my children.",
                "expected_info": ["family", "children", "kid"],
                "expected_tool": None
            },
            {
                "user_message": "What are some family-friendly activities in Japan?",
                "expected_info": ["Disneyland", "museum", "park", "family"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "Can you remind me where I'm planning to travel?",
                "expected_info": ["Japan", "trip", "next month"],
                "expected_tool": None
            },
            {
                "user_message": "What's the weather like in Japan next month?",
                "expected_info": ["temperature", "season", "weather"],
                "expected_tool": "web_search"
            }
        ]
        
        return self.run_conversation(combined_conversation)
    
    def run_complex_query_test(self):
        """Run a test with complex queries that require both memory and search."""
        logger.info("\n=== Running Complex Query Test ===")
        
        complex_conversation = [
            {
                "user_message": "I'm a software developer working with Python and JavaScript.",
                "expected_info": ["programming", "developer", "coding"],
                "expected_tool": None
            },
            {
                "user_message": "What are the latest trends in web development?",
                "expected_info": ["framework", "technology", "web"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "I'm interested in machine learning. Can you recommend some resources for beginners?",
                "expected_info": ["course", "tutorial", "book", "learn"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "Based on my programming background, which ML framework would be easiest for me to learn?",
                "expected_info": ["Python", "TensorFlow", "PyTorch", "scikit-learn"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "I also enjoy hiking on weekends.",
                "expected_info": ["outdoor", "nature", "activity"],
                "expected_tool": None
            },
            {
                "user_message": "What are some good hiking trails that are good for someone who codes all week?",
                "expected_info": ["trail", "hike", "outdoor", "nature"],
                "expected_tool": "web_search"
            },
            {
                "user_message": "Can you summarize what you know about me so far?",
                "expected_info": ["software developer", "Python", "JavaScript", "hiking", "machine learning"],
                "expected_tool": None
            }
        ]
        
        return self.run_conversation(complex_conversation)
    
    def run_all_tests(self):
        """Run all conversation tests and report results."""
        logger.info("\n=== Starting All Conversation Tests ===")
        
        try:
            # Run all tests
            search_results = self.run_search_test()
            memory_results = self.run_memory_test()
            combined_results = self.run_combined_test()
            complex_results = self.run_complex_query_test()
            
            # Summarize results
            logger.info("\n=== Test Results Summary ===")
            
            # Search test summary
            search_info_accuracy = sum(r["info_accuracy"] for r in search_results) / len(search_results)
            search_tool_accuracy = sum(1 for r in search_results if r["tool_correct"]) / len(search_results)
            logger.info(f"\nSearch Test:")
            logger.info(f"Information accuracy: {search_info_accuracy:.2f}")
            logger.info(f"Tool selection accuracy: {search_tool_accuracy:.2f}")
            
            # Memory test summary
            memory_accuracy = sum(r["info_accuracy"] for r in memory_results) / len(memory_results)
            logger.info(f"\nMemory Test:")
            logger.info(f"Memory retrieval accuracy: {memory_accuracy:.2f}")
            
            # Combined test summary
            combined_info_accuracy = sum(r["info_accuracy"] for r in combined_results) / len(combined_results)
            combined_tool_accuracy = sum(1 for r in combined_results if r["tool_correct"]) / len(combined_results)
            logger.info(f"\nCombined Test:")
            logger.info(f"Information accuracy: {combined_info_accuracy:.2f}")
            logger.info(f"Tool selection accuracy: {combined_tool_accuracy:.2f}")
            
            # Complex query test summary
            complex_info_accuracy = sum(r["info_accuracy"] for r in complex_results) / len(complex_results)
            complex_tool_accuracy = sum(1 for r in complex_results if r["tool_correct"]) / len(complex_results)
            logger.info(f"\nComplex Query Test:")
            logger.info(f"Information accuracy: {complex_info_accuracy:.2f}")
            logger.info(f"Tool selection accuracy: {complex_tool_accuracy:.2f}")
            
            # Overall summary
            all_results = search_results + memory_results + combined_results + complex_results
            overall_info_accuracy = sum(r["info_accuracy"] for r in all_results) / len(all_results)
            overall_tool_accuracy = sum(1 for r in all_results if r["tool_correct"]) / len(all_results)
            
            logger.info(f"\nOverall Results:")
            logger.info(f"Information accuracy: {overall_info_accuracy:.2f}")
            logger.info(f"Tool selection accuracy: {overall_tool_accuracy:.2f}")
            logger.info(f"Total turns: {len(all_results)}")
            
            # Log the location of the conversation log
            logger.info(f"\nConversation log saved to: {self.log_file}")
            
            return {
                "search_results": search_results,
                "memory_results": memory_results,
                "combined_results": combined_results,
                "complex_results": complex_results,
                "summary": {
                    "search_info_accuracy": search_info_accuracy,
                    "search_tool_accuracy": search_tool_accuracy,
                    "memory_accuracy": memory_accuracy,
                    "combined_info_accuracy": combined_info_accuracy,
                    "combined_tool_accuracy": combined_tool_accuracy,
                    "complex_info_accuracy": complex_info_accuracy,
                    "complex_tool_accuracy": complex_tool_accuracy,
                    "overall_info_accuracy": overall_info_accuracy,
                    "overall_tool_accuracy": overall_tool_accuracy
                }
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            raise

def main():
    """Main function to run the tests."""
    logger.info("Starting interactive chat test")
    
    try:
        # Create and run the tester
        tester = ChatInteractionTester()
        results = tester.run_all_tests()
        
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main() 