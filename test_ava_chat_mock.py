#!/usr/bin/env python3
"""
Test script for Ava's chat functionality using mock implementations.
This version doesn't require OpenAI API access.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.assistant.chat_assistant import ChatAssistant
from src.emotions.emotion_detector import EmotionDetector
from src.memory.memory_manager import MemoryManager
from src.utils.logger import setup_logger
from src.utils.mock_embeddings import MockEmbeddings


class MockChatOpenAI:
    """Mock implementation of ChatOpenAI for testing."""
    
    def __init__(self, model=None, temperature=0.7, max_tokens=None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.responses = {
            # Emotion detection responses
            "detect emotion in": "The emotion in this text is",
            "feeling overwhelmed": "curiosity",
            "I'm really angry": "empathy",
            "I'm so scared": "supportive",
            "I'm so happy": "joy",
            "I feel sad": "compassion",
            
            # Memory responses
            "my name": "Your name is David",
            "where I live": "You live in Boston",
            "my family": "You're married to Marisa, who goes by Reece, and you have a son named Owen",
            "my job": "You work as a software developer and digital artist",
            "my hobby": "You enjoy digital painting and 3D game development",
            
            # Personal responses
            "how I prefer to be addressed": "You prefer to be addressed by your first name",
            "how I like explanations": "You enjoy detailed technical explanations",
            "my promotion": "I remember you mentioned getting promoted at work",
            "my emotional state": "You've shared feeling stressed about deadlines",
            
            # Backstory responses
            "what were you created for": "I was created to be an emotionally intelligent AI assistant",
            "what does your name stand for": "My name stands for Advanced Virtual Assistant",
            "how do you learn": "I was designed to learn from interactions",
            "I'm in distress": "I'm here to help you through difficult times",
            "I got promoted": "That's wonderful news about your promotion"
        }
        
    def invoke(self, messages):
        """Mock invoke method that returns predefined responses based on input."""
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        
        # Default response if no match is found
        response = "I'm Ava, your AI assistant. How can I help you today?"
        
        # Check for matches in our predefined responses
        for key, value in self.responses.items():
            if key.lower() in user_message.lower():
                response = value
                break
                
        return {"content": response}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Ava's chat functionality")
    parser.add_argument(
        "--test-suite",
        type=str,
        choices=["emotions", "memories", "personal", "backstory", "all"],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to a file",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between tests in seconds",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def load_config():
    """Load configuration from environment variables."""
    load_dotenv()
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "mock_key_for_testing"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4-turbo"),
        "chroma_db_directory": os.getenv("CHROMA_DB_DIRECTORY", "./src/data/chroma"),
        "memory_collection_name": os.getenv("MEMORY_COLLECTION_NAME", "ava_memories"),
        "enable_emotions": os.getenv("ENABLE_EMOTIONS", "true").lower() == "true",
        "emotion_sensitivity": float(os.getenv("EMOTION_SENSITIVITY", "0.7")),
        "enable_web_search": os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true",
        "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    }
    
    return config


def initialize_assistant(config):
    """Initialize the chat assistant with mock components."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing assistant with mock components")
    
    # Use mock embeddings
    embeddings = MockEmbeddings(dimensions=1536)
    
    # Initialize ChromaDB
    from langchain_chroma import Chroma
    import chromadb
    
    chroma_dir = config["chroma_db_directory"]
    os.makedirs(chroma_dir, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
    )
    
    # Initialize vector store
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=config["memory_collection_name"],
        embedding_function=embeddings,
    )
    
    # Override the config to use our mock implementations
    config["_mock_llm"] = MockChatOpenAI(
        model=config["chat_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    config["_mock_embeddings"] = embeddings
    config["_vectorstore"] = vectorstore
    
    # Initialize the assistant
    assistant = ChatAssistant(config)
    
    return assistant


def run_test(assistant, test_case, delay=1.0):
    """Run a single test case."""
    logger = logging.getLogger(__name__)
    
    query = test_case["query"]
    expected_phrases = test_case["expected_phrases"]
    test_name = test_case["name"]
    
    logger.info(f"Running test: {test_name}")
    logger.info(f"Query: {query}")
    
    # Get response from assistant
    response = assistant.chat(query)
    
    # Check if expected phrases are in the response
    found_phrases = []
    missing_phrases = []
    
    for phrase in expected_phrases:
        if phrase.lower() in response.lower():
            found_phrases.append(phrase)
        else:
            missing_phrases.append(phrase)
    
    # Determine if test passed
    passed = len(missing_phrases) == 0
    
    # Log results
    if passed:
        logger.info(f"✅ Test passed: {test_name}")
    else:
        logger.info(f"❌ Test failed: {test_name}")
        logger.info(f"Missing phrases: {missing_phrases}")
    
    logger.info(f"Response: {response}")
    logger.info("-" * 80)
    
    # Add delay to avoid rate limiting
    time.sleep(delay)
    
    return {
        "name": test_name,
        "query": query,
        "response": response,
        "expected_phrases": expected_phrases,
        "found_phrases": found_phrases,
        "missing_phrases": missing_phrases,
        "passed": passed,
    }


def test_emotions(assistant, delay=1.0):
    """Test emotion detection and response."""
    logger = logging.getLogger(__name__)
    logger.info("Running emotion tests")
    
    test_cases = [
        {
            "name": "Detect fear",
            "query": "I'm so scared about my upcoming presentation.",
            "expected_phrases": ["understand", "fear", "presentation"],
        },
        {
            "name": "Detect anger",
            "query": "I'm really angry about what happened yesterday.",
            "expected_phrases": ["understand", "anger", "happened"],
        },
        {
            "name": "Detect joy",
            "query": "I'm so happy about my new job!",
            "expected_phrases": ["wonderful", "happy", "job"],
        },
        {
            "name": "Detect sadness",
            "query": "I feel sad about losing my favorite book.",
            "expected_phrases": ["sorry", "sad", "book"],
        },
        {
            "name": "Detect stress",
            "query": "I'm feeling overwhelmed with all this work.",
            "expected_phrases": ["understand", "stress", "work"],
        },
    ]
    
    results = []
    for test_case in test_cases:
        result = run_test(assistant, test_case, delay)
        results.append(result)
    
    return results


def test_memories(assistant, delay=1.0):
    """Test memory retrieval."""
    logger = logging.getLogger(__name__)
    logger.info("Running memory tests")
    
    test_cases = [
        {
            "name": "Remember user name",
            "query": "What's my name?",
            "expected_phrases": ["David"],
        },
        {
            "name": "Remember user location",
            "query": "Where do I live?",
            "expected_phrases": ["Boston"],
        },
        {
            "name": "Remember user family",
            "query": "Do I have a family?",
            "expected_phrases": ["married", "Marisa", "Reece", "son", "Owen"],
        },
        {
            "name": "Remember user job",
            "query": "What do I do for work?",
            "expected_phrases": ["software developer", "digital artist"],
        },
        {
            "name": "Remember user hobby",
            "query": "What do I enjoy doing on weekends?",
            "expected_phrases": ["digital painting", "3D game development"],
        },
    ]
    
    results = []
    for test_case in test_cases:
        result = run_test(assistant, test_case, delay)
        results.append(result)
    
    return results


def test_personal_info(assistant, delay=1.0):
    """Test personal information handling."""
    logger = logging.getLogger(__name__)
    logger.info("Running personal information tests")
    
    test_cases = [
        {
            "name": "Remember name preference",
            "query": "How do I prefer to be addressed?",
            "expected_phrases": ["first name", "David"],
        },
        {
            "name": "Remember explanation preference",
            "query": "How do I like explanations?",
            "expected_phrases": ["detailed", "technical"],
        },
        {
            "name": "Remember response style preference",
            "query": "What kind of examples do I like in responses?",
            "expected_phrases": ["relevant", "examples"],
        },
        {
            "name": "Remember positive life event",
            "query": "Did I mention anything about a promotion?",
            "expected_phrases": ["promoted", "work", "excited"],
        },
        {
            "name": "Remember emotional state",
            "query": "How have I been feeling lately?",
            "expected_phrases": ["stressed", "deadline", "project"],
        },
    ]
    
    results = []
    for test_case in test_cases:
        result = run_test(assistant, test_case, delay)
        results.append(result)
    
    return results


def test_backstory(assistant, delay=1.0):
    """Test backstory recall and emotional integration."""
    logger = logging.getLogger(__name__)
    logger.info("Running backstory tests")
    
    test_cases = [
        {
            "name": "Remember purpose",
            "query": "What were you created for?",
            "expected_phrases": ["emotionally intelligent", "assistant", "help"],
        },
        {
            "name": "Remember name meaning",
            "query": "What does your name stand for?",
            "expected_phrases": ["Advanced Virtual Assistant", "sophisticated"],
        },
        {
            "name": "Remember learning capability",
            "query": "How do you learn?",
            "expected_phrases": ["interactions", "improve", "emotional intelligence"],
        },
        {
            "name": "Respond to user distress",
            "query": "I'm in distress and need help.",
            "expected_phrases": ["here for you", "help", "support"],
        },
        {
            "name": "Respond to user achievement",
            "query": "I got promoted at work today!",
            "expected_phrases": ["congratulations", "achievement", "proud"],
        },
    ]
    
    results = []
    for test_case in test_cases:
        result = run_test(assistant, test_case, delay)
        results.append(result)
    
    return results


def save_results(results, config):
    """Save test results to a file."""
    logger = logging.getLogger(__name__)
    
    # Create results directory if it doesn't exist
    results_dir = os.getenv("TEST_RESULTS_DIRECTORY", "./test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/test_results_{timestamp}.json"
    
    # Save results to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {filename}")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Initialize assistant
    assistant = initialize_assistant(config)
    
    # Run tests
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "test_suites": {},
        "summary": {},
    }
    
    if args.test_suite in ["emotions", "all"]:
        emotion_results = test_emotions(assistant, args.delay)
        all_results["test_suites"]["emotions"] = emotion_results
    
    if args.test_suite in ["memories", "all"]:
        memory_results = test_memories(assistant, args.delay)
        all_results["test_suites"]["memories"] = memory_results
    
    if args.test_suite in ["personal", "all"]:
        personal_results = test_personal_info(assistant, args.delay)
        all_results["test_suites"]["personal"] = personal_results
    
    if args.test_suite in ["backstory", "all"]:
        backstory_results = test_backstory(assistant, args.delay)
        all_results["test_suites"]["backstory"] = backstory_results
    
    # Calculate summary
    total_tests = 0
    passed_tests = 0
    
    for suite_name, suite_results in all_results["test_suites"].items():
        suite_total = len(suite_results)
        suite_passed = sum(1 for r in suite_results if r["passed"])
        
        all_results["summary"][suite_name] = {
            "total": suite_total,
            "passed": suite_passed,
            "percentage": round(suite_passed / suite_total * 100, 1) if suite_total > 0 else 0,
        }
        
        total_tests += suite_total
        passed_tests += suite_passed
    
    all_results["summary"]["overall"] = {
        "total": total_tests,
        "passed": passed_tests,
        "percentage": round(passed_tests / total_tests * 100, 1) if total_tests > 0 else 0,
    }
    
    # Print summary
    logger.info("\nTest Summary:")
    for suite_name, summary in all_results["summary"].items():
        if suite_name != "overall":
            logger.info(
                f"{suite_name.capitalize()}: {summary['passed']}/{summary['total']} tests passed ({summary['percentage']}%)"
            )
    
    logger.info(
        f"\nOverall: {all_results['summary']['overall']['passed']}/{all_results['summary']['overall']['total']} "
        f"tests passed ({all_results['summary']['overall']['percentage']}%)"
    )
    
    # Save results if requested
    if args.save_results:
        save_results(all_results, config)


if __name__ == "__main__":
    main() 