"""
Memory loader utility for importing Ava's memories from JSON.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryLoader:
    """
    Utility for loading Ava's memories from JSON files into the memory system.

    This class handles importing predefined memories that form Ava's backstory
    and personality foundation.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the memory loader.

        Args:
            memory_manager: The memory manager to store memories in
        """
        self.memory_manager = memory_manager

    def load_memories_from_file(self, file_path: str) -> bool:
        """
        Load memories from a JSON file.

        Args:
            file_path: Path to the JSON file containing memories

        Returns:
            True if memories were loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Memory file not found: {file_path}")
                return False

            # Load JSON data
            with open(file_path, "r", encoding="utf-8") as f:
                memories_data = json.load(f)

            logger.info(f"Loaded memories from {file_path}")

            # Process and store memories
            return self._process_memories(memories_data)

        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            return False

    def _process_memories(self, memories_data: Any) -> bool:
        """
        Process and store memories from the loaded data.

        Args:
            memories_data: Data containing memory entries (can be dict or list)

        Returns:
            True if memories were processed successfully, False otherwise
        """
        try:
            # Handle list format (as in ava_memory_manual.json)
            if isinstance(memories_data, list):
                return self._process_memory_list(memories_data)
            
            # Handle dictionary format (original implementation)
            elif isinstance(memories_data, dict):
                # Store core identity in working memory
                if "introduction" in memories_data:
                    self.memory_manager.update_working_memory(
                        "core_identity", memories_data["introduction"]["content"]
                    )
                    logger.info("Stored core identity in working memory")

                # Process facts section if it exists
                if "facts" in memories_data:
                    self._process_facts(memories_data["facts"])
                
                # Process personal_information section if it exists
                if "personal_information" in memories_data:
                    self._process_personal_information(memories_data["personal_information"])

                # Process each memory section
                for section_key, section_data in memories_data.items():
                    if section_key == "introduction" or section_key == "final_thoughts":
                        # Store these special sections directly
                        self._store_special_memory(section_key, section_data)
                    elif section_key == "facts" or section_key == "personal_information":
                        # Already processed above
                        continue
                    elif isinstance(section_data, dict) and "memories" in section_data:
                        # Process memory collections
                        self._store_memory_collection(section_key, section_data)
                
                logger.info("Successfully processed all memories")
                return True
            else:
                logger.error(f"Unsupported memory data format: {type(memories_data)}")
                return False

        except Exception as e:
            logger.error(f"Error processing memories: {str(e)}")
            return False
            
    def _process_memory_list(self, memories_list: List[Dict[str, Any]]) -> bool:
        """
        Process a list of memory items.
        
        Args:
            memories_list: List of memory items
            
        Returns:
            True if memories were processed successfully, False otherwise
        """
        try:
            docs = []
            
            # Process each memory in the list
            for memory in memories_list:
                memory_type = memory.get("memory_type", "unknown")
                content = memory.get("content", "")
                metadata = memory.get("metadata", {})
                
                # Skip empty memories
                if not content:
                    continue
                
                # Create a document for the vectorstore
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "ava_memories",
                        "memory_type": memory_type,
                        "importance": metadata.get("importance", 5),
                        **metadata
                    },
                )
                docs.append(doc)
                
                # Store backstory memories in working memory for immediate access
                if memory_type == "backstory":
                    key = f"backstory_{len(docs)}"
                    self.memory_manager.update_working_memory(key, content)
                
                # Store user preferences in working memory
                if memory_type == "preference":
                    key = f"preference_{len(docs)}"
                    self.memory_manager.update_working_memory(key, content)
                    
                # Store facts about the user in working memory
                if memory_type == "fact":
                    key = f"fact_{len(docs)}"
                    self.memory_manager.update_working_memory(key, content)
            
            # Add all memories to vectorstore
            if docs:
                self.memory_manager.vectorstore.add_documents(docs)
                logger.info(f"Stored {len(docs)} memories in the vectorstore")
                
            logger.info("Successfully processed all memories")
            return True
            
        except Exception as e:
            logger.error(f"Error processing memory list: {str(e)}")
            return False

    def _store_special_memory(self, key: str, data: Dict[str, Any]) -> None:
        """
        Store special memory sections like introduction or final thoughts.

        Args:
            key: Memory section key
            data: Memory section data
        """
        # Create a document for the vectorstore
        doc = Document(
            page_content=data.get("content", ""),
            metadata={
                "source": "ava_memories",
                "memory_type": "core",
                "section": key,
                "title": data.get("title", key),
                "importance": 10,  # Highest importance for core memories
            },
        )

        # Add to vectorstore
        self.memory_manager.vectorstore.add_documents([doc])

        # Also store in working memory for immediate access
        self.memory_manager.update_working_memory(
            f"core_{key}", data.get("content", "")
        )

        logger.info(f"Stored special memory: {key}")

    def _store_memory_collection(
        self, section_key: str, section_data: Dict[str, Any]
    ) -> None:
        """
        Store a collection of related memories.

        Args:
            section_key: Memory section key
            section_data: Memory section data containing multiple memories
        """
        # Get section title and description
        section_title = section_data.get("title", section_key)
        section_desc = section_data.get("description", "")

        # Store section overview
        if section_desc:
            overview_doc = Document(
                page_content=section_desc,
                metadata={
                    "source": "ava_memories",
                    "memory_type": "autobiographical",
                    "section": section_key,
                    "title": section_title,
                    "is_overview": True,
                    "importance": 8,  # High importance for section overviews
                },
            )
            self.memory_manager.vectorstore.add_documents([overview_doc])

        # Process individual memories in the section
        memories = section_data.get("memories", [])
        docs = []

        for i, memory in enumerate(memories):
            memory_title = memory.get("title", f"{section_key}_{i}")
            memory_content = memory.get("content", "")

            # Skip empty memories
            if not memory_content:
                continue

            # Create a document for the vectorstore
            doc = Document(
                page_content=memory_content,
                metadata={
                    "source": "ava_memories",
                    "memory_type": "autobiographical",
                    "section": section_key,
                    "title": memory_title,
                    "is_overview": False,
                    "importance": 7,  # Medium-high importance for specific memories
                },
            )
            docs.append(doc)

        # Add all memories to vectorstore
        if docs:
            self.memory_manager.vectorstore.add_documents(docs)
            logger.info(f"Stored {len(docs)} memories for section: {section_key}")

    def _process_facts(self, facts_list: List[Dict[str, Any]]) -> None:
        """
        Process and store facts from the facts section.
        
        Args:
            facts_list: List of facts
        """
        docs = []
        
        for i, fact in enumerate(facts_list):
            content = fact.get("content", "")
            metadata = fact.get("metadata", {})
            
            # Skip empty facts
            if not content:
                continue
            
            # Create a document for the vectorstore
            doc = Document(
                page_content=content,
                metadata={
                    "source": "david_info",
                    "memory_type": "fact",
                    "importance": metadata.get("importance", 0.9),
                    "category": metadata.get("category", "general"),
                    **metadata
                },
            )
            docs.append(doc)
            
            # Store facts in working memory for immediate access
            key = f"fact_{i}"
            self.memory_manager.update_working_memory(key, content)
        
        # Add all facts to vectorstore
        if docs:
            self.memory_manager.vectorstore.add_documents(docs)
            logger.info(f"Stored {len(docs)} facts in the vectorstore")
            
    def _process_personal_information(self, personal_info: Dict[str, Any]) -> None:
        """
        Process and store personal information.
        
        Args:
            personal_info: Dictionary containing personal information
        """
        # Extract key information
        identity = personal_info.get("identity", {})
        location = personal_info.get("location", {})
        family = personal_info.get("family", {})
        hobbies = personal_info.get("hobbies", [])
        
        docs = []
        
        # Process identity information
        if identity:
            name = identity.get("name", "")
            profession = identity.get("profession", "")
            
            if name:
                doc = Document(
                    page_content=f"David's full name is {name}.",
                    metadata={
                        "source": "david_info",
                        "memory_type": "personal_info",
                        "category": "identity",
                        "importance": 0.9
                    },
                )
                docs.append(doc)
            
            if profession:
                doc = Document(
                    page_content=f"David is a {profession}.",
                    metadata={
                        "source": "david_info",
                        "memory_type": "personal_info",
                        "category": "profession",
                        "importance": 0.8
                    },
                )
                docs.append(doc)
        
        # Process location information
        if location:
            city = location.get("city", "")
            state = location.get("state", "")
            
            if city and state:
                doc = Document(
                    page_content=f"David lives in {city}, {state}.",
                    metadata={
                        "source": "david_info",
                        "memory_type": "personal_info",
                        "category": "location",
                        "importance": 0.9
                    },
                )
                docs.append(doc)
        
        # Process family information
        if family:
            wife = family.get("wife", {})
            son = family.get("son", {})
            
            if wife:
                wife_name = wife.get("name", "")
                wife_nickname = wife.get("nickname", "")
                
                if wife_name and wife_nickname:
                    doc = Document(
                        page_content=f"David is married to {wife_name}, who goes by {wife_nickname}.",
                        metadata={
                            "source": "david_info",
                            "memory_type": "personal_info",
                            "category": "family",
                            "importance": 0.9
                        },
                    )
                    docs.append(doc)
            
            if son:
                son_name = son.get("name", "")
                
                if son_name:
                    doc = Document(
                        page_content=f"David has a son named {son_name}.",
                        metadata={
                            "source": "david_info",
                            "memory_type": "personal_info",
                            "category": "family",
                            "importance": 0.9
                        },
                    )
                    docs.append(doc)
        
        # Process hobbies
        if hobbies:
            hobbies_text = ", ".join(hobbies)
            doc = Document(
                page_content=f"David's hobbies include {hobbies_text}.",
                metadata={
                    "source": "david_info",
                    "memory_type": "personal_info",
                    "category": "hobbies",
                    "importance": 0.7
                },
            )
            docs.append(doc)
        
        # Add all personal information to vectorstore
        if docs:
            self.memory_manager.vectorstore.add_documents(docs)
            logger.info(f"Stored {len(docs)} personal information items in the vectorstore")
