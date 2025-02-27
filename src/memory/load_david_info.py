#!/usr/bin/env python3
"""
Script to load David's information into a separate ChromaDB collection.

This script reads David's information from the david_info.json file and loads it
into a dedicated ChromaDB collection named "david_info". This ensures that David's
information is stored separately from Ava's memories, preventing data overwriting
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


def load_david_info(info_file: str = "./david_info.json") -> Dict[str, Any]:
    """
    Load David's information from the JSON file.
    
    Args:
        info_file: Path to the information file
        
    Returns:
        Dictionary containing David's information
    """
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        logger.info(f"Successfully loaded information from {info_file}")
        return info
    except Exception as e:
        logger.error(f"Error loading information: {e}")
        sys.exit(1)


def process_david_info(info: Dict[str, Any]) -> List[Document]:
    """
    Process David's information into documents for storage in ChromaDB.
    
    Args:
        info: Dictionary containing David's information
        
    Returns:
        List of documents ready for storage
    """
    documents = []
    
    # Process basic information
    if "basic_info" in info:
        basic_info = info["basic_info"]
        content = f"David's full name is {basic_info.get('full_name', 'David')}. "
        content += f"He is {basic_info.get('age', '')} years old. "
        content += f"He lives in {basic_info.get('location', '')}. "
        content += f"His occupation is {basic_info.get('occupation', '')}. "
        
        basic_doc = Document(
            page_content=content,
            metadata={
                "source": "david_info",
                "section": "basic_info",
                "importance": 0.9,
            }
        )
        documents.append(basic_doc)
    
    # Process family information
    if "family" in info:
        family = info["family"]
        content = f"David is {family.get('marital_status', '')}. "
        if "spouse" in family:
            spouse = family["spouse"]
            content += f"His wife's name is {spouse.get('name', '')}. "
            content += f"They have been married for {spouse.get('years_married', '')} years. "
        
        if "children" in family and family["children"]:
            children = family["children"]
            content += f"David has {len(children)} child(ren). "
            for child in children:
                content += f"His {child.get('relation', 'child')}'s name is {child.get('name', '')}. "
                content += f"{child.get('name', '')} is {child.get('age', '')} years old. "
        
        family_doc = Document(
            page_content=content,
            metadata={
                "source": "david_info",
                "section": "family",
                "importance": 0.9,
            }
        )
        documents.append(family_doc)
    
    # Process preferences
    if "preferences" in info:
        preferences = info["preferences"]
        
        # Process hobbies
        if "hobbies" in preferences:
            hobbies = preferences["hobbies"]
            content = "David's hobbies include: "
            content += ", ".join(hobbies) + ". "
            
            hobbies_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "hobbies",
                    "importance": 0.8,
                }
            )
            documents.append(hobbies_doc)
        
        # Process favorite foods
        if "favorite_foods" in preferences:
            foods = preferences["favorite_foods"]
            content = "David's favorite foods include: "
            content += ", ".join(foods) + ". "
            
            foods_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "favorite_foods",
                    "importance": 0.7,
                }
            )
            documents.append(foods_doc)
        
        # Process favorite movies
        if "favorite_movies" in preferences:
            movies = preferences["favorite_movies"]
            content = "David's favorite movies include: "
            content += ", ".join(movies) + ". "
            
            movies_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "favorite_movies",
                    "importance": 0.7,
                }
            )
            documents.append(movies_doc)
    
    # Process facts
    if "facts" in info:
        facts = info["facts"]
        for fact in facts:
            # Extract content and metadata from the fact
            if isinstance(fact, dict) and "content" in fact:
                content = fact["content"]
                metadata = {
                    "source": "david_info",
                    "section": "facts",
                    "importance": 0.85,
                }
                
                # Add additional metadata if available
                if "metadata" in fact and isinstance(fact["metadata"], dict):
                    for key, value in fact["metadata"].items():
                        metadata[key] = value
                
                fact_doc = Document(
                    page_content=content,
                    metadata=metadata
                )
            else:
                # If fact is just a string
                fact_doc = Document(
                    page_content=str(fact),
                    metadata={
                        "source": "david_info",
                        "section": "facts",
                        "importance": 0.85,
                    }
                )
            documents.append(fact_doc)
    
    # Process personal information
    if "personal_information" in info:
        personal_info = info["personal_information"]
        
        # Process identity
        if "identity" in personal_info:
            identity = personal_info["identity"]
            content = f"David's full name is {identity.get('name', 'David Florence')}. "
            content += f"He goes by {identity.get('nickname', 'David')}. "
            content += f"His profession is {identity.get('profession', 'Software developer and digital artist')}. "
            
            identity_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "identity",
                    "importance": 0.9,
                }
            )
            documents.append(identity_doc)
        
        # Process location
        if "location" in personal_info:
            location = personal_info["location"]
            content = f"David lives in {location.get('city', 'Boston')}, {location.get('state', 'Massachusetts')}, {location.get('country', 'USA')}. "
            if "home" in location:
                content += f"His home is a {location.get('home', '')}. "
            
            location_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "location",
                    "importance": 0.9,
                }
            )
            documents.append(location_doc)
        
        # Process family
        if "family" in personal_info:
            family = personal_info["family"]
            
            # Process wife
            if "wife" in family:
                wife = family["wife"]
                content = f"David's wife's name is {wife.get('name', 'Marisa Reed')}. "
                content += f"She goes by {wife.get('nickname', 'Reece')}. "
                if "interests" in wife:
                    content += f"Her interests include {', '.join(wife.get('interests', []))}. "
                
                wife_doc = Document(
                    page_content=content,
                    metadata={
                        "source": "david_info",
                        "section": "family",
                        "subsection": "wife",
                        "importance": 0.9,
                    }
                )
                documents.append(wife_doc)
            
            # Process son
            if "son" in family:
                son = family["son"]
                content = f"David's son's name is {son.get('name', 'Owen')}. "
                if "friend" in son:
                    content += f"His friend's name is {son.get('friend', '')}. "
                if "interests" in son:
                    content += f"His interests include {', '.join(son.get('interests', []))}. "
                
                son_doc = Document(
                    page_content=content,
                    metadata={
                        "source": "david_info",
                        "section": "family",
                        "subsection": "son",
                        "importance": 0.9,
                    }
                )
                documents.append(son_doc)
        
        # Process hobbies
        if "hobbies" in personal_info:
            hobbies = personal_info["hobbies"]
            content = "David's hobbies include: "
            content += ", ".join(hobbies) + ". "
            
            hobbies_doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "section": "hobbies",
                    "importance": 0.8,
                }
            )
            documents.append(hobbies_doc)
    
    logger.info(f"Processed {len(documents)} documents from David's information")
    return documents


def store_david_info(documents: List[Document], chroma_dir: str = "./src/data/chroma") -> None:
    """
    Store David's information documents in ChromaDB.
    
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
        chroma_client.delete_collection("david_info")
        logger.info("Deleted existing 'david_info' collection")
    except Exception:
        logger.info("No existing 'david_info' collection found")
    
    # Initialize vector store for David's information
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="david_info",
        embedding_function=embedding_function,
    )
    
    # Add documents to the vector store
    vectorstore.add_documents(documents)
    logger.info(f"Added {len(documents)} documents to 'david_info' collection")


def main():
    """Main function to load and store David's information."""
    logger.info("Starting to load David's information")
    
    # Load information from file
    info = load_david_info()
    
    # Process information into documents
    documents = process_david_info(info)
    
    # Store documents in ChromaDB
    store_david_info(documents)
    
    logger.info("Successfully loaded David's information into ChromaDB")


if __name__ == "__main__":
    main() 