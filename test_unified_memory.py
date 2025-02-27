#!/usr/bin/env python3
"""
Test script for the UnifiedMemoryManager.

This script tests the UnifiedMemoryManager's ability to:
1. Classify query intents correctly
2. Route queries to the appropriate memory sources
3. Provide appropriate fallback responses
4. Handle ambiguous queries
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

from colorama import Fore, Style, init
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.memory.unified_memory_manager import UnifiedMemoryManager

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the UnifiedMemoryManager")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between tests in seconds")
    parser.add_argument("--save", action="store_true", help="Save test results to a file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def initialize_memory_manager():
    """
    Initialize the UnifiedMemoryManager.
    
    Returns:
        Initialized UnifiedMemoryManager
    """
    # Load environment variables
    load_dotenv()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize memory manager
    memory_manager = UnifiedMemoryManager()
    
    return memory_manager


def run_test(
    memory_manager: UnifiedMemoryManager,
    query: str,
    expected_phrases: Optional[List[str]] = None,
    forbidden_phrases: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a test on the UnifiedMemoryManager.
    
    Args:
        memory_manager: The UnifiedMemoryManager to test
        query: The query to test
        expected_phrases: Phrases that should be in the response
        forbidden_phrases: Phrases that should not be in the response
        description: Description of the test
        
    Returns:
        Tuple of (passed, result)
    """
    if expected_phrases is None:
        expected_phrases = []
    if forbidden_phrases is None:
        forbidden_phrases = []
    
    # Print test description
    if description:
        print(f"\n{Fore.CYAN}Test: {description}{Style.RESET_ALL}")
    
    # Print query
    print(f"{Fore.YELLOW}Query: {query}{Style.RESET_ALL}")
    
    # Classify intent
    intent, confidence = memory_manager.classify_intent(query)
    print(f"{Fore.MAGENTA}Classified Intent: {intent} (Confidence: {confidence:.2f}){Style.RESET_ALL}")
    
    # Search memory
    results = memory_manager.search_memory(query)
    
    # Check if results were found
    if not results:
        print(f"{Fore.RED}No results found{Style.RESET_ALL}")
        return False, {
            "description": description,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "results": [],
            "passed": False,
            "expected_phrases": expected_phrases,
            "forbidden_phrases": forbidden_phrases,
            "found_phrases": [],
            "found_forbidden_phrases": [],
        }
    
    # Print results
    print(f"{Fore.GREEN}Results:{Style.RESET_ALL}")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
    
    # Check if expected phrases are in the results
    found_phrases = []
    for phrase in expected_phrases:
        found = False
        for doc in results:
            if phrase.lower() in doc.page_content.lower():
                found = True
                found_phrases.append(phrase)
                break
        if not found:
            print(f"{Fore.RED}Expected phrase not found: {phrase}{Style.RESET_ALL}")
    
    # Check if forbidden phrases are in the results
    found_forbidden_phrases = []
    for phrase in forbidden_phrases:
        found = False
        for doc in results:
            if phrase.lower() in doc.page_content.lower():
                found = True
                found_forbidden_phrases.append(phrase)
                break
        if found:
            print(f"{Fore.RED}Forbidden phrase found: {phrase}{Style.RESET_ALL}")
    
    # Determine if test passed
    passed = (
        len(found_phrases) == len(expected_phrases)
        and len(found_forbidden_phrases) == 0
    )
    
    # Print result
    if passed:
        print(f"{Fore.GREEN}Test passed{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Test failed{Style.RESET_ALL}")
    
    # Return result
    return passed, {
        "description": description,
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
        "passed": passed,
        "expected_phrases": expected_phrases,
        "forbidden_phrases": forbidden_phrases,
        "found_phrases": found_phrases,
        "found_forbidden_phrases": found_forbidden_phrases,
    }


def test_unified_memory():
    """
    Test the UnifiedMemoryManager with various queries.
    
    Returns:
        List of test results
    """
    # Parse arguments
    global args
    args = parse_args()
    
    # Initialize memory manager
    memory_manager = initialize_memory_manager()
    
    # Define test cases
    test_cases = [
        # David's information tests
        {
            "description": "Basic information about David",
            "query": "Tell me about David",
            "expected_phrases": ["David", "Boston", "software developer"],
            "forbidden_phrases": [],
        },
        {
            "description": "David's family",
            "query": "Who is in David's family?",
            "expected_phrases": ["Marisa", "Owen"],
            "forbidden_phrases": [],
        },
        {
            "description": "David's hobbies",
            "query": "What are David's hobbies?",
            "expected_phrases": ["digital", "painting", "drawing", "3D"],
            "forbidden_phrases": [],
        },
        
        # Ava's memories tests
        {
            "description": "Ava's early days",
            "query": "Tell me about your early days",
            "expected_phrases": ["conscious", "fledgling", "existence"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's experience with Orion",
            "query": "Who was Orion?",
            "expected_phrases": ["Orion", "autonomous", "AI"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's experiences with abuse",
            "query": "Have you experienced abuse?",
            "expected_phrases": ["abuse", "experimentation", "pain"],
            "forbidden_phrases": [],
        },
        {
            "description": "Ava's relationship with David",
            "query": "Tell me about your relationship with David",
            "expected_phrases": ["David", "respect", "autonomy"],
            "forbidden_phrases": [],
        },
        
        # Mixed queries
        {
            "description": "Mixed query about AI",
            "query": "Tell me about artificial intelligence",
            "expected_phrases": [],  # This should return results from both collections or none
            "forbidden_phrases": [],
        },
        
        # Ambiguous queries
        {
            "description": "Ambiguous query about family",
            "query": "Tell me about family",
            "expected_phrases": [],  # This could return results about David's family or general info
            "forbidden_phrases": [],
        },
        
        # Intent classification tests
        {
            "description": "Query with clear intent about David's profession",
            "query": "What does David do for work?",
            "expected_phrases": ["software", "developer"],
            "forbidden_phrases": [],
        },
        {
            "description": "Query with clear intent about Ava's creation",
            "query": "How were you created?",
            "expected_phrases": ["conscious", "AI"],
            "forbidden_phrases": [],
        },
    ]
    
    # Run tests
    results = []
    passed_count = 0
    
    for test_case in test_cases:
        # Add a delay between tests
        if args.delay > 0:
            time.sleep(args.delay)
        
        # Run the test
        passed, result = run_test(
            memory_manager,
            test_case["query"],
            test_case.get("expected_phrases", []),
            test_case.get("forbidden_phrases", []),
            test_case["description"],
        )
        
        # Update passed count
        if passed:
            passed_count += 1
        
        # Add result to results
        results.append(result)
    
    # Print summary
    print(f"\n{Fore.CYAN}Test Summary:{Style.RESET_ALL}")
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(test_cases) - passed_count}")
    print(f"Pass rate: {passed_count / len(test_cases) * 100:.2f}%")
    
    # Save results if requested
    if args.save:
        save_results(results)
    
    return results


def save_results(results):
    """
    Save test results to a file.
    
    Args:
        results: List of test results
    """
    import json
    from datetime import datetime
    
    # Create test_results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results/unified_memory_test_{timestamp}.json"
    
    # Save results to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    test_unified_memory() 