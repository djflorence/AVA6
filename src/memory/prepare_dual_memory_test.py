#!/usr/bin/env python3
"""
Script to prepare the test environment with both Ava's memories and David's information.

This script loads both Ava's memories and David's information into separate ChromaDB
collections, then initializes the DualMemoryManager for testing. This ensures that
both collections are properly set up before running tests.
"""

import logging
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

# Import the memory loading scripts
from src.memory.load_ava_memories import load_ava_memories, process_memories, store_memories
from src.memory.load_david_info import load_david_info, process_david_info, store_david_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def prepare_ava_memories():
    """Prepare Ava's memories collection."""
    logger.info("Preparing Ava's memories collection")
    
    # Load memories from file
    memories = load_ava_memories()
    
    # Process memories into documents
    documents = process_memories(memories)
    
    # Store documents in ChromaDB
    store_memories(documents)
    
    logger.info("Successfully prepared Ava's memories collection")


def prepare_david_info():
    """Prepare David's information collection."""
    logger.info("Preparing David's information collection")
    
    # Load information from file
    info = load_david_info()
    
    # Process information into documents
    documents = process_david_info(info)
    
    # Store documents in ChromaDB
    store_david_info(documents)
    
    logger.info("Successfully prepared David's information collection")


def main():
    """Main function to prepare both collections."""
    logger.info("Starting to prepare dual memory test environment")
    
    # Prepare Ava's memories
    prepare_ava_memories()
    
    # Prepare David's information
    prepare_david_info()
    
    logger.info("Successfully prepared dual memory test environment")
    logger.info("Both collections are ready for testing with DualMemoryManager")


if __name__ == "__main__":
    main() 