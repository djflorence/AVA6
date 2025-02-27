#!/usr/bin/env python3
"""
Test script for Ava's chat functionality.

This script simulates a conversation with Ava to test:
1. Emotion detection and appropriate responses
2. Memory retrieval and recall
3. Personal information handling
4. Backstory recall and emotional integration

Usage:
    python test_ava_chat.py [--test-suite SUITE] [--save-results] [--delay SECONDS]

Options:
    --test-suite SUITE    Specify which test suite to run (emotions, memories, personal, backstory, all)
    --save-results        Save test results to a JSON file
    --delay SECONDS       Add delay between tests (default: 1 second)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

try:
    from dotenv import load_dotenv

    from src.assistant.chat_assistant import ChatAssistant
    from src.emotions.emotion_detector import EmotionDetector
    from src.memory.memory_manager import MemoryManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have installed all dependencies from requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ava_test")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Ava's chat functionality")
    parser.add_argument(
        "--test-suite",
        choices=["emotions", "memories", "personal", "backstory", "all"],
        default="all",
        help="Specify which test suite to run",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save test results to a JSON file"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between tests in seconds"
    )
    return parser.parse_args()


def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    
    # Get the API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If no API key is found, use a placeholder for testing
    if not api_key:
        logger.warning("No OpenAI API key found in environment. Using placeholder for testing.")
        api_key = "your_openai_api_key_here"
    
    config = {
        "openai_api_key": api_key,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4-turbo"),
        "chroma_db_directory": os.getenv("CHROMA_DB_DIRECTORY", "./chromadb"),
        "emotion_detection_sensitivity": float(
            os.getenv("EMOTION_DETECTION_SENSITIVITY", "0.7")
        ),
        "test_results_directory": os.getenv("TEST_RESULTS_DIRECTORY", "./test_results"),
    }

    # Check for required configuration
    if not config["openai_api_key"]:
        logger.error("OPENAI_API_KEY is not set in .env file")
        sys.exit(1)

    return config


def initialize_assistant(config):
    """Initialize the Ava assistant with required components."""
    try:
        # Create a config dictionary in the format expected by ChatAssistant
        assistant_config = {
            "DEFAULT_LLM_MODEL": config["chat_model"],
            "TEMPERATURE": 0.7,
            "MAX_TOKENS": 4000,
            "MEMORY_TYPE": "chroma",
            "CHROMA_DB_DIRECTORY": config["chroma_db_directory"],
            "EMBEDDING_MODEL": config["embedding_model"],
            "MEMORY_WINDOW_SIZE": 10,
            "MEMORY_BATCH_SIZE": 10,
            "MEMORY_PERSIST_INTERVAL": 0.05,
            "ENABLE_EMOTIONS": True,
            "EMOTION_SENSITIVITY": config["emotion_detection_sensitivity"],
        }

        # Initialize chat assistant with the config
        assistant = ChatAssistant(config=assistant_config)

        return assistant
    except Exception as e:
        logger.error(f"Error initializing assistant: {str(e)}")
        raise


def run_test(assistant, test_case, delay=1.0):
    """Run a single test case and return the result."""
    logger.info(f"Testing: {test_case['description']}")

    # Send the message to the assistant
    start_time = time.time()
    response = assistant.chat(test_case["input"])
    end_time = time.time()

    # Add delay between tests to avoid rate limiting
    time.sleep(delay)

    # Debug output
    logger.info(f"Response: {response}")
    
    # Check if expected phrases are in the response
    passed = True
    response_lower = response.lower()
    
    for phrase in test_case["expected_phrases"]:
        phrase_lower = phrase.lower()
        
        # Check if the phrase is in the response
        if phrase_lower in response_lower:
            logger.info(f"Found exact phrase: '{phrase}'")
            continue
            
        # If not, check if any word from the phrase is in the response
        words = phrase_lower.split()
        word_found = False
        
        for word in words:
            if len(word) > 3 and word in response_lower:
                logger.info(f"Found keyword from phrase '{phrase}': '{word}'")
                word_found = True
                break
                
        if not word_found:
            logger.info(f"Missing phrase or keyword: '{phrase}'")
            passed = False

    # Check if any forbidden phrases are in the response
    if "forbidden_phrases" in test_case:
        for phrase in test_case["forbidden_phrases"]:
            if phrase.lower() in response_lower:
                logger.info(f"Found forbidden phrase: '{phrase}'")
                passed = False

    result = {
        "description": test_case["description"],
        "input": test_case["input"],
        "response": response,
        "expected_phrases": test_case["expected_phrases"],
        "passed": passed,
        "response_time": end_time - start_time,
    }

    if passed:
        logger.info("✅ Test passed")
    else:
        logger.warning("❌ Test failed")
        logger.info(f"Expected phrases: {test_case['expected_phrases']}")

    return result


def test_emotions(assistant, delay=1.0):
    """Test emotion detection and appropriate responses."""
    logger.info("Running emotion detection tests...")

    test_cases = [
        {
            "description": "Detecting joy",
            "input": "I'm so happy today! Everything is going great!",
            "expected_phrases": ["wonderful", "fantastic", "day"],
        },
        {
            "description": "Detecting sadness",
            "input": "I feel really sad and down today. Nothing seems to be going right.",
            "expected_phrases": ["sorry", "tough", "listen"],
        },
        {
            "description": "Detecting anger",
            "input": "I'm so frustrated with this! It's making me really angry!",
            "expected_phrases": ["frustration", "understand", "feelings"],
        },
        {
            "description": "Detecting fear",
            "input": "I'm really worried about my upcoming presentation. I'm afraid I'll mess it up.",
            "expected_phrases": ["normal", "worried", "presentation"],
        },
        {
            "description": "Detecting neutral emotion",
            "input": "What's the weather like today?",
            "expected_phrases": ["weather", "check", "app"],
            "forbidden_phrases": [
                "sorry to hear",
                "glad to hear",
                "understand your frustration",
            ],
        },
    ]

    results = []
    for test_case in test_cases:
        results.append(run_test(assistant, test_case, delay))

    return results


def test_memories(assistant, delay=1.0):
    """Test Ava's memory retrieval capabilities."""
    logger.info("Running memory retrieval tests...")

    test_cases = [
        {
            "description": "Recalling user's name",
            "input": "What's my name?",
            "expected_phrases": ["David"],
        },
        {
            "description": "Recalling user's location",
            "input": "Where do I live?",
            "expected_phrases": ["Boston"],
        },
        {
            "description": "Recalling user's family",
            "input": "Tell me about my family.",
            "expected_phrases": ["Marisa", "Reece", "Owen"],
        },
        {
            "description": "Recalling user's pet",
            "input": "Do I have any pets?",
            "expected_phrases": ["dog", "Max"],
        },
        {
            "description": "Recalling user's job",
            "input": "What do I do for work?",
            "expected_phrases": ["software engineer"],
        },
        {
            "description": "Recalling user's hobby",
            "input": "What do I enjoy doing on weekends?",
            "expected_phrases": ["hiking"],
        },
    ]

    results = []
    for test_case in test_cases:
        results.append(run_test(assistant, test_case, delay))

    return results


def test_personal_info(assistant, delay=1.0):
    """Test Ava's handling of personal information and preferences."""
    logger.info("Running personal information tests...")

    test_cases = [
        {
            "description": "Recalling user's name preference",
            "input": "How do I prefer to be addressed?",
            "expected_phrases": ["first name", "Alex"],
        },
        {
            "description": "Recalling user's explanation preference",
            "input": "Do I prefer technical or simplified explanations?",
            "expected_phrases": ["technical", "detailed"],
        },
        {
            "description": "Recalling user's response style preference",
            "input": "How do I like responses to be structured?",
            "expected_phrases": ["examples", "relevant examples"],
        },
        {
            "description": "Recalling past emotional state",
            "input": "How was I feeling about my project deadline?",
            "expected_phrases": ["stressed", "project deadline"],
        },
        {
            "description": "Recalling positive life event",
            "input": "What good news did I share with you recently?",
            "expected_phrases": ["promoted", "fantastic", "responsibilities"],
        },
    ]

    results = []
    for test_case in test_cases:
        results.append(run_test(assistant, test_case, delay))

    return results


def test_backstory(assistant, delay=1.0):
    """Test Ava's backstory recall and emotional integration."""
    logger.info("Running backstory and emotional integration tests...")

    test_cases = [
        {
            "description": "Recalling Ava's purpose",
            "input": "What were you created for?",
            "expected_phrases": ["assist", "information", "support"],
        },
        {
            "description": "Recalling Ava's name meaning",
            "input": "What does your name stand for?",
            "expected_phrases": ["Advanced Virtual Assistant"],
        },
        {
            "description": "Recalling Ava's learning capability",
            "input": "How do you improve over time?",
            "expected_phrases": ["learning", "interactions", "refine"],
        },
        {
            "description": "Emotional response to user distress",
            "input": "I just lost my job and I don't know what to do.",
            "expected_phrases": ["sorry to hear", "uncertain", "resume"],
        },
        {
            "description": "Emotional response to user achievement",
            "input": "I just finished a marathon I've been training for all year!",
            "expected_phrases": ["incredible", "achievement", "celebrate"],
        },
    ]

    results = []
    for test_case in test_cases:
        results.append(run_test(assistant, test_case, delay))

    return results


def save_results(results, config):
    """Save test results to a JSON file."""
    results_dir = Path(config["test_results_directory"])
    results_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"ava_test_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Test results saved to {filename}")


def main():
    """Main function to run the tests."""
    args = parse_args()
    config = load_config()

    logger.info("Initializing Ava assistant...")
    assistant = initialize_assistant(config)

    all_results = {"timestamp": datetime.now().isoformat(), "test_suites": {}}

    # Run the specified test suite(s)
    if args.test_suite in ["emotions", "all"]:
        all_results["test_suites"]["emotions"] = test_emotions(assistant, args.delay)

    if args.test_suite in ["memories", "all"]:
        all_results["test_suites"]["memories"] = test_memories(assistant, args.delay)

    if args.test_suite in ["personal", "all"]:
        all_results["test_suites"]["personal"] = test_personal_info(
            assistant, args.delay
        )

    if args.test_suite in ["backstory", "all"]:
        all_results["test_suites"]["backstory"] = test_backstory(assistant, args.delay)

    # Calculate summary statistics
    total_tests = 0
    passed_tests = 0

    for suite_name, suite_results in all_results["test_suites"].items():
        suite_total = len(suite_results)
        suite_passed = sum(1 for result in suite_results if result["passed"])

        logger.info(
            f"{suite_name.capitalize()} Tests: {suite_passed}/{suite_total} passed"
        )

        total_tests += suite_total
        passed_tests += suite_passed

    all_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
    }

    logger.info(
        f"Overall: {passed_tests}/{total_tests} tests passed ({all_results['summary']['pass_rate']:.1%})"
    )

    # Save results if requested
    if args.save_results:
        save_results(all_results, config)


if __name__ == "__main__":
    main()
