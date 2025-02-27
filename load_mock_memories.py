#!/usr/bin/env python3
"""
Load mock memories into the AVA assistant using mock implementations.
This script avoids using the OpenAI API by using mock embeddings.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import mock implementations
from src.utils.mock_embeddings import MockEmbeddings

# Import memory components
from src.memory.memory_loader import MemoryLoader
from src.memory.memory_manager import MemoryManager
from src.config.settings import Settings
from langchain_chroma import Chroma
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/memory_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load mock memories into AVA assistant")
    parser.add_argument("--file", type=str, default="ava_memory_manual.json", help="Memory file to load")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def load_config():
    """Load configuration from environment variables."""
    settings = Settings()
    # Convert settings to dictionary using model_dump() method
    config = settings.model_dump()
    return config

def initialize_mock_components(config):
    """Initialize mock components for memory loading."""
    # Create mock embeddings
    mock_embeddings = MockEmbeddings()
    
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
    
    return vectorstore

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
    vectorstore = initialize_mock_components(config)
    
    # Check if memory file exists
    memory_file = args.file
    if not os.path.exists(memory_file):
        logger.error(f"Memory file not found: {memory_file}")
        return 1
    
    logger.info(f"Loading memories from {memory_file}...")
    
    # Initialize memory manager
    conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_manager = MemoryManager(vectorstore=vectorstore)
    
    # Initialize memory loader
    memory_loader = MemoryLoader(memory_manager=memory_manager)
    
    # Load memories
    try:
        with open(memory_file, 'r') as f:
            memories = json.load(f)
        
        logger.info(f"Loaded {len(memories)} memories from file")
        
        # Process memories
        memory_loader.load_memories(memories)
        
        logger.info("Memories loaded successfully")
        return 0
    except Exception as e:
        logger.error(f"Error loading memories: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 