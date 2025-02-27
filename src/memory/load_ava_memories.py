#!/usr/bin/env python3
"""
Script to load Ava's memories into a separate ChromaDB collection.

This script reads Ava's memories from the ava_memory_manual.json file and loads them
into a dedicated ChromaDB collection named "ava_memories". This ensures that Ava's
memories are stored separately from David's information, preventing data overwriting
and simplifying query logic.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_ava_memories(memory_file: str = "./ava_memory_manual.json") -> List[Dict[str, Any]]:
    """
    Load Ava's memories from the JSON file.
    
    Args:
        memory_file: Path to the memory file
        
    Returns:
        List of memory sections with their contents
    """
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            memories = json.load(f)
        logger.info(f"Successfully loaded memories from {memory_file}")
        return memories
    except Exception as e:
        logger.error(f"Error loading memories: {e}")
        sys.exit(1)


def process_memories(memories: Dict[str, Any]) -> List[Document]:
    """
    Process memories into documents for storage in ChromaDB.
    
    Args:
        memories: Dictionary of memory sections
        
    Returns:
        List of documents ready for storage
    """
    documents = []
    
    # Process each section of memories
    for section_name, section_data in memories.items():
        if section_name == "introduction":
            # Process introduction separately
            intro_doc = Document(
                page_content=section_data.get("description", ""),
                metadata={
                    "source": "ava_memories",
                    "section": "introduction",
                    "title": section_data.get("title", "Introduction"),
                    "importance": 0.9,
                }
            )
            documents.append(intro_doc)
        else:
            # Process section description
            section_doc = Document(
                page_content=section_data.get("description", ""),
                metadata={
                    "source": "ava_memories",
                    "section": section_name,
                    "title": section_data.get("title", section_name.replace("_", " ").title()),
                    "importance": 0.8,
                }
            )
            documents.append(section_doc)
            
            # Process individual memories in the section
            for memory in section_data.get("memories", []):
                memory_doc = Document(
                    page_content=memory.get("content", ""),
                    metadata={
                        "source": "ava_memories",
                        "section": section_name,
                        "title": memory.get("title", "Untitled Memory"),
                        "importance": 0.95,  # Individual memories are more important
                    }
                )
                documents.append(memory_doc)
    
    logger.info(f"Processed {len(documents)} documents from memories")
    return documents


def store_memories(documents: List[Document], chroma_dir: str = "./src/data/chroma") -> None:
    """
    Store memory documents in ChromaDB.
    
    Args:
        documents: List of documents to store
        chroma_dir: Directory for ChromaDB
    """
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
    
    # Check if collection exists and delete it if it does
    try:
        chroma_client.delete_collection("ava_memories")
        logger.info("Deleted existing 'ava_memories' collection")
    except Exception:
        logger.info("No existing 'ava_memories' collection found")
    
    # Initialize vector store for Ava's memories
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="ava_memories",
        embedding_function=embedding_function,
    )
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    logger.info(f"Added {len(documents)} documents to 'ava_memories' collection")


def main():
    """Main function to load and store Ava's memories."""
    logger.info("Starting to load Ava's memories")
    
    # Load memories from file
    memories = load_ava_memories()
    
    # Process memories into documents
    documents = process_memories(memories)
    
    # Store documents in ChromaDB
    store_memories(documents)
    
    logger.info("Successfully loaded Ava's memories into ChromaDB")


if __name__ == "__main__":
    main() 