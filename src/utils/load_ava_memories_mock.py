#!/usr/bin/env python3
"""
Script to load Ava's memories from a JSON file into the memory system using mock embeddings.
This version doesn't require OpenAI API access.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.memory.memory_manager import MemoryManager
from src.utils.logger import setup_logger
from src.utils.memory_loader import MemoryLoader
from src.utils.mock_embeddings import MockEmbeddings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load Ava's memories from a JSON file")
    parser.add_argument(
        "--file",
        type=str,
        default="ava_memory_manual.json",
        help="Path to the JSON file containing Ava's memories",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


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

    # Check if file exists
    memory_file = args.file
    if not os.path.exists(memory_file):
        logger.error(f"Memory file not found: {memory_file}")
        sys.exit(1)

    try:
        # Initialize mock embeddings instead of OpenAI embeddings
        logger.info("Using mock embeddings for testing (no API calls)")
        embeddings = MockEmbeddings(dimensions=1536)

        # Initialize vector store
        chroma_dir = os.getenv("CHROMA_DB_DIRECTORY", "./src/data/chroma")
        os.makedirs(chroma_dir, exist_ok=True)

        import chromadb

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

        # Initialize memory manager
        memory_manager = MemoryManager(vectorstore=vectorstore)

        # Initialize memory loader
        memory_loader = MemoryLoader(memory_manager=memory_manager)

        # Load memories
        logger.info(f"Loading memories from {memory_file}...")
        success = memory_loader.load_memories_from_file(memory_file)

        if success:
            logger.info("Successfully loaded Ava's memories")
            print("[SUCCESS] Successfully loaded Ava's memories into the system")
        else:
            logger.error("Failed to load Ava's memories")
            print("[ERROR] Failed to load Ava's memories")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error loading memories: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 