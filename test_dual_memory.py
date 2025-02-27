#!/usr/bin/env python3
"""
Test script for the DualMemoryManager.

This script tests the DualMemoryManager's ability to correctly route queries to the
appropriate collection (Ava's memories or David's information) and return relevant results.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import colorama
from colorama import Fore, Style
from dotenv import load_dotenv

from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings
from src.memory.dual_memory_manager import DualMemoryManager
from src.utils.logger import setup_logger

# Initialize colorama
colorama.init()

# Set up logging
setup_logger()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the DualMemoryManager")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save", action="store_true", help="Save test results to file")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between tests in seconds")
    return parser.parse_args()


def initialize_memory_manager():
    """Initialize the DualMemoryManager."""
    logger.info("Initializing DualMemoryManager")
    
    memory_manager = DualMemoryManager(
        chroma_dir="./src/data/chroma",
        window_size=10,
        conversation_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        batch_size=100,
        persist_interval=0.05,
    )
    
    logger.info("DualMemoryManager initialized")
    return memory_manager


def run_test(memory_manager, query, expected_phrases=None, forbidden_phrases=None, description=None):
    """
    Run a single test case.
    
    Args:
        memory_manager: The DualMemoryManager instance
        query: The query to test
        expected_phrases: List of phrases expected in the response
        forbidden_phrases: List of phrases that should not be in the response
        description: Description of the test
        
    Returns:
        Tuple of (passed, results)
    """
    if expected_phrases is None:
        expected_phrases = []
    if forbidden_phrases is None:
        forbidden_phrases = []
    
    logger.info(f"Running test: {description}")
    logger.info(f"Query: {query}")
    logger.info(f"Expected phrases: {expected_phrases}")
    
    # Search memory
    results = memory_manager.search_memory(query, k=5)
    
    # Check if results contain expected phrases
    passed = True
    missing_phrases = []
    forbidden_found = []
    
    # Combine all result content
    combined_content = " ".join([doc.page_content for doc in results])
    
    # Check for expected phrases
    for phrase in expected_phrases:
        if phrase.lower() not in combined_content.lower():
            passed = False
            missing_phrases.append(phrase)
    
    # Check for forbidden phrases
    for phrase in forbidden_phrases:
        if phrase.lower() in combined_content.lower():
            passed = False
            forbidden_found.append(phrase)
    
    # Print results
    if passed:
        print(f"{Fore.GREEN}✓ Test passed: {description}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ Test failed: {description}{Style.RESET_ALL}")
        if missing_phrases:
            print(f"{Fore.YELLOW}  Missing phrases: {missing_phrases}{Style.RESET_ALL}")
        if forbidden_found:
            print(f"{Fore.YELLOW}  Forbidden phrases found: {forbidden_found}{Style.RESET_ALL}")
    
    # Return results
    return passed, {
        "description": description,
        "query": query,
        "expected_phrases": expected_phrases,
        "forbidden_phrases": forbidden_phrases,
        "results": [doc.page_content for doc in results],
        "passed": passed,
        "missing_phrases": missing_phrases,
        "forbidden_found": forbidden_found,
    }


def test_dual_memory():
    """
    Test the DualMemoryManager with various queries.
    
    Returns:
        List of test results
    """
    # Parse arguments
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
    # Create results directory if it doesn't exist
    results_dir = Path("./test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"dual_memory_test_results_{timestamp}.json"
    
    # Save results
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    test_dual_memory() 