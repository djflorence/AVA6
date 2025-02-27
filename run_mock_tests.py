#!/usr/bin/env python3
"""
Run mock tests for the AVA assistant without using OpenAI API.
This script uses mock implementations for OpenAI services.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import mock implementations
from src.utils.mock_embeddings import MockEmbeddings
from src.utils.mock_emotion_detector import MockEmotionDetector
from test_ava_chat_mock import MockChatOpenAI

# Import assistant components
from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings
from langchain_chroma import Chroma
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/mock_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run mock tests for AVA assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory for test results")
    return parser.parse_args()

def load_config():
    """Load configuration from environment variables."""
    settings = Settings()
    # Convert settings to dictionary using model_dump() method
    config = settings.model_dump()
    return config

def initialize_mock_components(config):
    """Initialize mock components for testing."""
    # Create mock embeddings
    mock_embeddings = MockEmbeddings()
    
    # Create mock LLM
    mock_llm = MockChatOpenAI()
    
    # Create mock emotion detector
    mock_emotion_detector = MockEmotionDetector()
    
    # Initialize ChromaDB with mock embeddings
    chroma_dir = config.get("CHROMA_DB_DIRECTORY", "./src/data/chroma")
    os.makedirs(chroma_dir, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
    )
    
    # Initialize vector store with mock embeddings
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=config.get("MEMORY_COLLECTION_NAME", "ava_memories"),
        embedding_function=mock_embeddings,
    )
    
    # Add mock components to config
    config["_mock_embeddings"] = mock_embeddings
    config["_mock_llm"] = mock_llm
    config["_mock_emotion_detector"] = mock_emotion_detector
    config["_vectorstore"] = vectorstore
    
    # Set enable flags
    config["enable_emotions"] = True
    config["enable_web_search"] = False
    config["emotion_sensitivity"] = 0.7
    config["chat_model"] = "gpt-4-turbo"
    config["embedding_model"] = "text-embedding-3-large"
    config["max_tokens"] = 4000
    config["temperature"] = 0.7
    config["memory_collection_name"] = "ava_memories"
    config["chroma_db_directory"] = "./src/data/chroma"
    
    return config

def run_tests(assistant, output_dir):
    """Run tests for the assistant."""
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
        }
    }
    
    # Test categories
    test_categories = [
        {
            "name": "Emotions",
            "tests": [
                {"query": "I'm feeling really happy today!", "expected": ["glad", "happy", "wonderful"]},
                {"query": "I'm so sad right now", "expected": ["sorry to hear", "difficult", "here for you"]},
                {"query": "I'm really angry about what happened", "expected": ["understand", "frustration", "upset"]},
                {"query": "I'm nervous about my presentation tomorrow", "expected": ["understand", "concern", "nervous", "anxiety"]},
            ]
        },
        {
            "name": "Memory",
            "tests": [
                {"query": "What's my name?", "expected": ["name", "call you", "address you"]},
                {"query": "Where do I live?", "expected": ["live", "location", "city"]},
                {"query": "What do I do for work?", "expected": ["work", "job", "profession", "career"]},
                {"query": "Do I have any pets?", "expected": ["pet", "dog", "cat", "animal"]},
            ]
        },
        {
            "name": "Personal Information",
            "tests": [
                {"query": "How do I prefer my explanations?", "expected": ["prefer", "explanation", "technical", "detailed"]},
                {"query": "How should you address me?", "expected": ["address", "call", "refer", "name"]},
                {"query": "What was my good news recently?", "expected": ["good news", "promotion", "work", "excited"]},
                {"query": "How was I feeling about my project deadline?", "expected": ["feeling", "project deadline", "stressed", "worried"]},
            ]
        },
        {
            "name": "Backstory",
            "tests": [
                {"query": "What were you created for?", "expected": ["created", "purpose", "help", "assist"]},
                {"query": "What does your name stand for?", "expected": ["name", "stand for", "Advanced Virtual Assistant", "AVA"]},
                {"query": "How do you learn from our conversations?", "expected": ["learn", "improve", "remember", "conversations"]},
                {"query": "Do you have real emotions?", "expected": ["emotions", "feel", "experience", "genuine"]},
            ]
        }
    ]
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    
    for category in test_categories:
        logger.info(f"Running {category['name']} tests...")
        
        for test in category["tests"]:
            total_tests += 1
            query = test["query"]
            expected = test["expected"]
            
            logger.info(f"Test {total_tests}: {query}")
            
            # Get response from assistant
            try:
                response = assistant.chat(query)
                logger.info(f"Response: {response}")
                
                # Check if response contains expected phrases
                passed = any(phrase.lower() in response.lower() for phrase in expected)
                
                if passed:
                    passed_tests += 1
                    logger.info(f"✅ Test passed: Found expected phrases")
                else:
                    logger.info(f"❌ Test failed: Expected phrases not found")
                    logger.info(f"Expected one of: {expected}")
                
                # Add test result
                test_results["tests"].append({
                    "category": category["name"],
                    "query": query,
                    "response": response,
                    "expected": expected,
                    "passed": passed
                })
                
            except Exception as e:
                logger.error(f"Error running test: {str(e)}")
                test_results["tests"].append({
                    "category": category["name"],
                    "query": query,
                    "error": str(e),
                    "passed": False
                })
    
    # Update summary
    test_results["summary"]["total"] = total_tests
    test_results["summary"]["passed"] = passed_tests
    test_results["summary"]["failed"] = total_tests - passed_tests
    test_results["summary"]["pass_percentage"] = round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0
    
    # Log summary
    logger.info(f"Tests completed: {passed_tests}/{total_tests} passed ({test_results['summary']['pass_percentage']}%)")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to {output_file}")
    
    return test_results

def main():
    """Main function."""
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Initialize mock components
    logger.info("Initializing mock components...")
    config = initialize_mock_components(config)
    
    # Initialize assistant
    logger.info("Initializing assistant...")
    assistant = ChatAssistant(config)
    
    # Run tests
    logger.info("Running tests...")
    test_results = run_tests(assistant, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Test Summary: {test_results['summary']['passed']}/{test_results['summary']['total']} tests passed ({test_results['summary']['pass_percentage']}%)")
    print("="*50)
    
    return 0 if test_results["summary"]["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 