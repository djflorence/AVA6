#!/usr/bin/env python3
"""
Prepare Ava for testing by loading memories and setting up the environment.

This script ensures that:
1. Ava's memories are properly loaded into ChromaDB
2. The environment is correctly set up
3. The emotion detection system is operational

Usage:
    python prepare_ava_test.py [--memory-file PATH] [--force-reload] [--debug]

Options:
    --memory-file PATH    Path to the memory file (default: ava_memory_manual.json)
    --force-reload        Force reload memories even if they already exist
    --debug               Enable debug mode for more detailed logging
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

try:
    from dotenv import load_dotenv

    from src.emotions.emotion_detector import EmotionDetector
    from src.memory.memory_manager import MemoryManager
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have installed all dependencies from requirements.txt")
    sys.exit(1)


# Configure logging
def setup_logging(debug=False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("ava_prepare")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Ava for testing")
    parser.add_argument(
        "--memory-file",
        type=str,
        default="ava_memory_manual.json",
        help="Path to the memory file",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload memories even if they already exist",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def check_environment():
    """Check if the environment is properly set up."""
    logger.info("Checking environment...")

    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "EMBEDDING_MODEL", "CHROMA_DB_DIRECTORY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.error("Please set these variables in your .env file")
        return False

    logger.info("Environment check passed")
    return True


def check_memories_exist(memory_manager):
    """Check if memories already exist in ChromaDB."""
    logger.info("Checking if memories exist...")

    try:
        # Try to get a count of memories
        if hasattr(memory_manager.vectorstore, "_collection"):
            collection = memory_manager.vectorstore._collection
            count = collection.count()

            if count > 0:
                logger.info(f"Found {count} existing memories in ChromaDB")
                return True
            else:
                logger.info("No memories found in ChromaDB")
                return False
        else:
            # Try to get count through the vectorstore API
            try:
                # Use similarity search with empty query to get a count
                results = memory_manager.vectorstore.similarity_search("", k=1)
                if results:
                    logger.info("Found existing memories in ChromaDB")
                    return True
                else:
                    logger.info("No memories found in ChromaDB")
                    return False
            except Exception as e:
                logger.warning(
                    f"Could not check memory count through vectorstore API: {e}"
                )
                return False
    except Exception as e:
        logger.error(f"Error checking memories: {e}")
        return False


def load_memories(memory_file, memory_manager):
    """Load memories from a JSON file into ChromaDB."""
    logger.info(f"Loading memories from {memory_file}...")

    try:
        # Check if file exists
        if not os.path.exists(memory_file):
            logger.error(f"Memory file not found: {memory_file}")
            return False

        # Load memories from file
        with open(memory_file, "r") as f:
            memories = json.load(f)

        logger.info(f"Loaded {len(memories)} memories from file")

        # Create documents for ChromaDB
        from langchain_core.documents import Document

        documents = []

        # Add memories to ChromaDB
        for memory in memories:
            # Extract content and metadata
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})

            # Add memory type to metadata if not present
            if "memory_type" not in metadata and "memory_type" in memory:
                metadata["memory_type"] = memory["memory_type"]

            # Create a document
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # Add documents to vectorstore
        if documents:
            memory_manager.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} memories to ChromaDB")

        logger.info("Memories loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading memories: {e}")
        return False


def test_emotion_detection():
    """Test the emotion detection system."""
    logger.info("Testing emotion detection...")

    try:
        # Initialize emotion detector
        sensitivity = float(os.getenv("EMOTION_DETECTION_SENSITIVITY", "0.7"))
        emotion_detector = EmotionDetector(sensitivity=sensitivity)

        # Test with sample texts
        test_texts = [
            "I'm so happy today!",
            "I feel really sad and down.",
            "I'm angry about what happened.",
            "I'm scared of the dark.",
        ]

        for text in test_texts:
            emotion = emotion_detector.detect_emotion(text)
            logger.info(f"Text: '{text}' -> Detected emotion: {emotion}")

        logger.info("Emotion detection test passed")
        return True
    except Exception as e:
        logger.error(f"Error testing emotion detection: {e}")
        return False


def main():
    """Main function to prepare Ava for testing."""
    args = parse_args()

    # Set up logging
    global logger
    logger = setup_logging(args.debug)

    # Load environment variables
    load_dotenv()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Initialize memory manager
    try:
        # Initialize embeddings
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize vector store
        chroma_dir = os.getenv("CHROMA_DB_DIRECTORY", "./chromadb")
        os.makedirs(chroma_dir, exist_ok=True)

        import chromadb
        from langchain_chroma import Chroma

        chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
        )

        # Initialize vector store with the client
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="ava_memories",
            embedding_function=embeddings,
        )

        # Initialize memory manager with the vectorstore
        memory_manager = MemoryManager(vectorstore=vectorstore)
    except Exception as e:
        logger.error(f"Error initializing memory manager: {e}")
        sys.exit(1)

    # Check if memories exist and load if needed
    memories_exist = check_memories_exist(memory_manager)

    if not memories_exist or args.force_reload:
        if args.force_reload and memories_exist:
            logger.info("Force reload requested, reloading memories...")

        if not load_memories(args.memory_file, memory_manager):
            logger.error("Failed to load memories")
            sys.exit(1)
    else:
        logger.info("Memories already exist, skipping load")
        logger.info("Use --force-reload to reload memories")

    # Test emotion detection
    if not test_emotion_detection():
        logger.warning("Emotion detection test failed")
        logger.warning("Some tests may not work correctly")

    logger.info("Ava is ready for testing!")


if __name__ == "__main__":
    main()
