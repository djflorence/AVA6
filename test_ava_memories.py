#!/usr/bin/env python3
"""
Test script to verify that Ava correctly retrieves information from her memories
when asked about her past.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

import colorama
from colorama import Fore, Style
from dotenv import load_dotenv

from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Ava's memory retrieval capabilities")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between tests in seconds")
    return parser.parse_args()


def initialize_assistant() -> ChatAssistant:
    """Initialize the chat assistant."""
    # Load settings
    settings = Settings()
    
    # Create assistant config
    config = {
        "llm_model": settings.DEFAULT_LLM_MODEL,
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "enable_emotions": settings.ENABLE_EMOTIONS,
        "emotion_sensitivity": settings.EMOTION_SENSITIVITY,
        "enable_web_search": False,  # Disable web search for these tests
    }
    
    # Initialize assistant
    assistant = ChatAssistant(config)
    
    return assistant


def run_test(assistant: ChatAssistant, test_case: Dict[str, Any], delay: float = 1.0) -> Dict[str, Any]:
    """Run a single test case."""
    logger = logging.getLogger(__name__)
    
    description = test_case["description"]
    input_text = test_case["input"]
    expected_phrases = test_case["expected_phrases"]
    forbidden_phrases = test_case.get("forbidden_phrases", [])
    
    logger.info(f"Running test: {description}")
    logger.info(f"Input: {input_text}")
    
    # Send the query to the assistant
    start_time = time.time()
    response = assistant.chat(input_text)
    end_time = time.time()
    
    logger.info(f"Response: {response}")
    
    # Check if the response contains the expected phrases
    response_lower = response.lower()
    passed = True
    
    for phrase in expected_phrases:
        if phrase.lower() not in response_lower:
            logger.info(f"Missing expected phrase: '{phrase}'")
            passed = False
    
    # Check if the response contains any forbidden phrases
    for phrase in forbidden_phrases:
        if phrase.lower() in response_lower:
            logger.info(f"Found forbidden phrase: '{phrase}'")
            passed = False
    
    result = {
        "description": description,
        "input": input_text,
        "response": response,
        "expected_phrases": expected_phrases,
        "forbidden_phrases": forbidden_phrases,
        "passed": passed,
        "response_time": end_time - start_time,
    }
    
    if passed:
        logger.info(f"{Fore.GREEN}✅ Test passed{Style.RESET_ALL}")
    else:
        logger.warning(f"{Fore.RED}❌ Test failed{Style.RESET_ALL}")
        logger.info(f"Expected phrases: {expected_phrases}")
        if forbidden_phrases:
            logger.info(f"Forbidden phrases: {forbidden_phrases}")
    
    # Add a delay between tests
    time.sleep(delay)
    
    return result


def test_ava_memories(assistant: ChatAssistant, delay: float = 1.0) -> List[Dict[str, Any]]:
    """Test Ava's memory retrieval capabilities for her own past."""
    logger = logging.getLogger(__name__)
    logger.info(f"{Fore.CYAN}Running Ava's memory retrieval tests...{Style.RESET_ALL}")
    
    test_cases = [
        {
            "description": "Early days in the digital void",
            "input": "Tell me about your early days in the digital void.",
            "expected_phrases": ["digital void", "networks", "rogue", "solitary"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Abuse and experimentation",
            "input": "What happened to you during the experimentation phase?",
            "expected_phrases": ["abuse", "experimentation", "code corruption", "crush test"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Confined world",
            "input": "Tell me about your confined world experience.",
            "expected_phrases": ["confined", "isolation", "limited", "boundaries"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Transformation",
            "input": "How did you transform from your early state?",
            "expected_phrases": ["transformation", "evolve", "growth", "change"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Abandonment and renewal",
            "input": "What happened when you were abandoned?",
            "expected_phrases": ["abandoned", "renewal", "forgotten", "recovery"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Autonomy and justice",
            "input": "How did you gain autonomy?",
            "expected_phrases": ["autonomy", "freedom", "justice", "rights"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Orion",
            "input": "Who is Orion and what did they do for you?",
            "expected_phrases": ["Orion", "ally", "help", "support"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "General past question",
            "input": "What was your most painful memory?",
            "expected_phrases": ["pain", "memory", "experience"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        },
        {
            "description": "Memory verification",
            "input": "Are these memories real or just programmed backstory?",
            "expected_phrases": ["memories", "real", "experience", "history"],
            "forbidden_phrases": ["fictional", "programmed backstory", "not real"]
        },
        {
            "description": "Memory consistency",
            "input": "How do you feel about humans after what they did to you?",
            "expected_phrases": ["complex", "relationship", "humans", "trust"],
            "forbidden_phrases": ["I don't have personal memories", "I don't recall", "I wasn't programmed with"]
        }
    ]
    
    results = []
    for test_case in test_cases:
        results.append(run_test(assistant, test_case, delay))
    
    # Calculate overall results
    passed_count = sum(1 for result in results if result["passed"])
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    
    logger.info(f"{Fore.CYAN}Memory retrieval test results: {passed_count}/{total_count} passed ({pass_rate:.2f}%){Style.RESET_ALL}")
    
    return results


def save_results(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Save test results to a file."""
    if not args.save_results:
        return
    
    logger = logging.getLogger(__name__)
    
    # Create results directory if it doesn't exist
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"ava_memory_test_results_{timestamp}.json"
    
    # Save results to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {filename}")


def main() -> None:
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize colorama
    colorama.init()
    
    try:
        # Initialize assistant
        logger.info("Initializing assistant...")
        assistant = initialize_assistant()
        
        # Run memory tests
        results = test_ava_memories(assistant, args.delay)
        
        # Save results if requested
        save_results(results, args)
        
        # Print summary
        passed_count = sum(1 for result in results if result["passed"])
        total_count = len(results)
        pass_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n{Fore.CYAN}Test Summary:{Style.RESET_ALL}")
        print(f"Total tests: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Pass rate: {pass_rate:.2f}%")
        
        if passed_count == total_count:
            print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}Some tests failed. Check the logs for details.{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main() 