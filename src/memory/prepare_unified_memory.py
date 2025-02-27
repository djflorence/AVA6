#!/usr/bin/env python3
"""
Script to prepare the unified memory collection by migrating data from the existing
separate collections for Ava and David.

This script:
1. Loads data from the existing Ava and David collections
2. Combines them into a single unified collection with appropriate metadata
3. Initializes the UnifiedMemoryManager for testing
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.memory.load_ava_memories import load_ava_memories, process_memories
from src.memory.load_david_info import load_david_info, process_david_info
from src.memory.unified_memory_manager import UnifiedMemoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_ava_documents() -> List[Document]:
    """
    Get documents from Ava's memories.
    
    Returns:
        List of documents from Ava's memories
    """
    logger.info("Loading Ava's memories")
    
    # Load memories from file
    memories = load_ava_memories()
    
    # Process memories into documents
    documents = process_memories(memories)
    
    logger.info(f"Loaded {len(documents)} documents from Ava's memories")
    return documents


def get_david_documents() -> List[Document]:
    """
    Get documents from David's information.
    
    Returns:
        List of documents from David's information
    """
    logger.info("Loading David's information")
    
    # Load information from file
    info = load_david_info()
    
    # Process information into documents
    documents = process_david_info(info)
    
    logger.info(f"Loaded {len(documents)} documents from David's information")
    return documents


def prepare_unified_collection(
    ava_documents: List[Document],
    david_documents: List[Document],
    chroma_dir: str = "./src/data/chroma",
) -> None:
    """
    Prepare the unified collection by combining documents from Ava and David.
    
    Args:
        ava_documents: Documents from Ava's memories
        david_documents: Documents from David's information
        chroma_dir: Directory for ChromaDB
    """
    logger.info("Preparing unified collection")
    
    # Create ChromaDB directory if it doesn't exist
    os.makedirs(chroma_dir, exist_ok=True)
    
    # Initialize embedding function
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_function = OpenAIEmbeddings(model=embedding_model)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=False),
    )
    
    # Check if the collection already exists and delete it if it does
    try:
        chroma_client.delete_collection("unified_memories")
        logger.info("Deleted existing unified_memories collection")
    except Exception:
        logger.info("No existing unified_memories collection to delete")
    
    # Initialize unified vector store
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="unified_memories",
        embedding_function=embedding_function,
    )
    
    # Combine documents
    all_documents = ava_documents + david_documents
    logger.info(f"Combined {len(all_documents)} documents")
    
    # Add documents to the vector store
    vectorstore.add_documents(all_documents)
    logger.info("Added documents to unified collection")


def initialize_unified_memory_manager(chroma_dir: str = "./src/data/chroma") -> UnifiedMemoryManager:
    """
    Initialize the UnifiedMemoryManager.
    
    Args:
        chroma_dir: Directory for ChromaDB
        
    Returns:
        Initialized UnifiedMemoryManager
    """
    logger.info("Initializing UnifiedMemoryManager")
    
    # Initialize embedding function
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_function = OpenAIEmbeddings(model=embedding_model)
    
    # Initialize UnifiedMemoryManager
    memory_manager = UnifiedMemoryManager(
        embedding_function=embedding_function,
        chroma_dir=chroma_dir,
    )
    
    logger.info("UnifiedMemoryManager initialized")
    return memory_manager


def main():
    """Main function to prepare the unified memory collection."""
    logger.info("Starting to prepare unified memory collection")
    
    # Get documents from Ava's memories
    ava_documents = get_ava_documents()
    
    # Get documents from David's information
    david_documents = get_david_documents()
    
    # Prepare unified collection
    prepare_unified_collection(ava_documents, david_documents)
    
    # Initialize UnifiedMemoryManager
    memory_manager = initialize_unified_memory_manager()
    
    logger.info("Successfully prepared unified memory collection")
    logger.info("Unified memory collection is ready for testing with UnifiedMemoryManager")


if __name__ == "__main__":
    main() 