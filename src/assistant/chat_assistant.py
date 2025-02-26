"""
Core chat assistant implementation using LangChain and ChromaDB.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.emotions.emotion_detector import EmotionDetector
from src.memory.memory_manager import MemoryManager
from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class ChatAssistant:
    """
    Advanced chat assistant using LangChain and ChromaDB.
    
    This class integrates various components:
    - LLM for generating responses
    - Memory system for conversation history
    - Emotion detection for emotional intelligence
    - Tool integration for enhanced capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chat assistant.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self._init_llm()
        self._init_memory()
        self._init_emotions()
        self._init_tools()
        self._init_chain()
        
        logger.info("Chat assistant initialized")
    
    def _init_llm(self) -> None:
        """Initialize the language model."""
        model_name = self.config.get("DEFAULT_LLM_MODEL", "gpt-4-turbo")
        temperature = float(self.config.get("TEMPERATURE", 0.7))
        max_tokens = int(self.config.get("MAX_TOKENS", 4000))
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info(f"Initialized LLM: {model_name}")
    
    def _init_memory(self) -> None:
        """Initialize the memory system."""
        memory_type = self.config.get("MEMORY_TYPE", "chroma")
        
        if memory_type == "chroma":
            # Initialize ChromaDB for vector storage
            chroma_dir = self.config.get("CHROMA_DB_DIRECTORY", "./src/data/chroma")
            os.makedirs(chroma_dir, exist_ok=True)
            
            # Initialize embeddings
            embedding_model = self.config.get("EMBEDDING_MODEL", "text-embedding-3-large")
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            
            # Initialize vector store
            import chromadb
            
            # Create the persist directory if it doesn't exist
            os.makedirs(chroma_dir, exist_ok=True)
            
            # Initialize ChromaDB client with proper settings
            # Only use settings that are supported by the current version
            chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=False  # Prevent accidental database resets
                )
            )
            
            # Initialize vector store with the client
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name="conversation_history",
                embedding_function=self.embeddings,
            )
            
            # Initialize memory manager with optimized batch settings
            self.memory_manager = MemoryManager(
                vectorstore=self.vectorstore,
                window_size=int(self.config.get("MEMORY_WINDOW_SIZE", 10)),
                conversation_id=self.conversation_id,
                batch_size=int(self.config.get("MEMORY_BATCH_SIZE", 10)),
                persist_interval=float(self.config.get("MEMORY_PERSIST_INTERVAL", 0.05)),
            )
            
            # Initialize conversation memory for the chain
            # Using ConversationBufferMemory for compatibility, but we'll handle it manually
            # in a way that avoids the deprecation issues
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="history",
            )
        else:
            # Fallback to simple buffer memory
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="history",
            )
            self.memory_manager = None
        
        logger.info(f"Initialized memory system: {memory_type}")
    
    def _init_emotions(self) -> None:
        """Initialize the emotion detection system."""
        enable_emotions_value = self.config.get("ENABLE_EMOTIONS", "true")
        
        # Handle both string and boolean values
        if isinstance(enable_emotions_value, bool):
            enable_emotions = enable_emotions_value
        else:
            enable_emotions = enable_emotions_value.lower() in ("true", "1", "yes")
        
        if enable_emotions:
            sensitivity = float(self.config.get("EMOTION_SENSITIVITY", 0.7))
            self.emotion_detector = EmotionDetector(sensitivity=sensitivity)
            logger.info(f"Initialized emotion detector with sensitivity {sensitivity}")
        else:
            self.emotion_detector = None
            logger.info("Emotion detection disabled")
    
    def _init_tools(self) -> None:
        """Initialize the tool manager and available tools."""
        self.tool_manager = ToolManager(self.config)
        logger.info(f"Initialized tool manager with {len(self.tool_manager.get_tools())} tools")
    
    def _init_chain(self) -> None:
        """Initialize the conversation chain using modern LangChain patterns."""
        # Create system prompt
        system_prompt = """You are an advanced AI assistant with emotional intelligence and a variety of tools at your disposal.
Your goal is to be helpful, harmless, and honest in all interactions.

IMPORTANT: The user's name is David. Always address the user as David in your responses.

When responding to David:
1. Be concise and clear in your responses
2. If you don't know something, admit it rather than making up information
3. Adapt your tone based on David's emotional state
4. Use tools when appropriate to provide better assistance
5. ALWAYS use the provided memory context to recall previous conversations and remembered information

IMPORTANT: You have the ability to remember information across the conversation. When David asks you to remember something, 
you should confirm that you've stored it and can recall it later. When he asks about something you've been asked to remember,
refer to the "Remembered information" section in your context.

You have access to both short-term memory (recent messages) and long-term memory (previous conversations).
When David refers to something mentioned earlier, use the provided memory context to give accurate responses.

Remember that there is only one user (David) and you should always maintain a consistent conversation with him.
"""
        
        # Create a modern LangChain chain using the LCEL (LangChain Expression Language)
        from langchain_core.output_parsers import StrOutputParser
        
        # Create a more structured prompt template with proper handling of conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain with proper output parsing
        # This is the recommended approach in modern LangChain (0.1.0+)
        self.chain = prompt | self.llm | StrOutputParser()
        
        logger.info("Initialized conversation chain using modern LangChain patterns")
    
    def chat(self, user_input: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            The assistant's response
        """
        logger.info(f"Received user input: {user_input}")
        
        # Detect emotions if enabled
        emotion = None
        if self.emotion_detector:
            emotion = self.emotion_detector.detect_emotion(user_input)
            logger.debug(f"Detected emotion: {emotion}")
        
        # Check if we need to use any tools
        tool_response = None
        if self.tool_manager:
            tool_name = self.tool_manager.should_use_tool(user_input)
            if tool_name:
                logger.info(f"Using tool: {tool_name}")
                tool_response = self.tool_manager.use_tool(tool_name, user_input)
        
        # Retrieve relevant information from long-term memory if available
        memory_context = ""
        if self.memory_manager:
            try:
                # First, search for relevant documents in the current conversation
                relevant_docs = self.memory_manager.search_memory(
                    user_input, 
                    k=3,
                    search_type="mmr"  # Use Maximum Marginal Relevance for diverse results
                )
                
                # Check if this is a query about the user's name
                if any(pattern in user_input.lower() for pattern in ["what's my name", "what is my name", "who am i"]):
                    # Search specifically for user name across all conversations
                    name_docs = self.memory_manager.search_memory(
                        "my name is",
                        k=3,
                        search_type="similarity",
                        search_all_conversations=True,
                        memory_type="user_name"
                    )
                    
                    if name_docs:
                        # Add name documents to the beginning of relevant docs
                        for doc in name_docs:
                            if doc not in relevant_docs:
                                relevant_docs.insert(0, doc)
                
                # Check if this is a query about remembered information
                elif any(pattern in user_input.lower() for pattern in ["remember", "what did i ask you to remember", "recall"]):
                    # Search specifically for remembered facts across all conversations
                    remembered_docs = self.memory_manager.search_memory(
                        user_input,
                        k=5,
                        search_type="similarity",
                        search_all_conversations=True,
                        memory_type="remembered_fact"
                    )
                    
                    if remembered_docs:
                        # Add remembered docs to the beginning of relevant docs
                        for doc in remembered_docs:
                            if doc not in relevant_docs:
                                relevant_docs.insert(0, doc)
                
                # For general queries, also search across all conversations
                elif "what" in user_input.lower() or "when" in user_input.lower() or "who" in user_input.lower() or "where" in user_input.lower() or "how" in user_input.lower():
                    # This is likely a general query that might benefit from cross-conversation search
                    general_docs = self.memory_manager.search_memory(
                        user_input,
                        k=3,
                        search_type="similarity",
                        search_all_conversations=True
                    )
                    
                    # Combine the results
                    if general_docs:
                        # Add general docs that aren't already in relevant_docs
                        for doc in general_docs:
                            if doc not in relevant_docs:
                                relevant_docs.append(doc)
                
                # Get user information
                user_info = self.memory_manager.get_user_info()
                if user_info:
                    user_context = "User information:\n"
                    if "name" in user_info:
                        user_context += f"- Name: {user_info['name']}\n"
                    if "remembered_facts" in user_info and user_info["remembered_facts"]:
                        user_context += "- Remembered facts:\n"
                        for i, fact in enumerate(user_info["remembered_facts"]):
                            user_context += f"  {i+1}. {fact}\n"
                    
                    # Add user information to memory context
                    if memory_context:
                        memory_context += "\n" + user_context
                    else:
                        memory_context = user_context
                
                if relevant_docs:
                    if memory_context:
                        memory_context += "\nPrevious relevant conversation:\n"
                    else:
                        memory_context = "Previous relevant conversation:\n"
                    
                    for i, doc in enumerate(relevant_docs):
                        # Add metadata type if available to help with context
                        doc_type = doc.metadata.get("type", "conversation")
                        memory_context += f"{i+1}. [{doc_type}] {doc.page_content}\n"
                    logger.debug(f"Retrieved {len(relevant_docs)} relevant memories")
                
                # Get working memory items (remembered facts)
                if self.memory_manager.working_memory:
                    if memory_context:
                        memory_context += "\n"
                    memory_context += "Remembered information:\n"
                    for key, value in self.memory_manager.working_memory.items():
                        if isinstance(value, dict) and "content" in value:
                            memory_context += f"- {value['content']}\n"
                        else:
                            memory_context += f"- {key}: {value}\n"
                    logger.debug(f"Retrieved {len(self.memory_manager.working_memory)} items from working memory")
            except Exception as e:
                logger.error(f"Error retrieving from memory: {str(e)}")
        
        # Get conversation history from memory
        messages = []
        if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
            messages = self.memory.chat_memory.messages
        
        # Build enhanced input with all context
        enhanced_input = user_input
        
        # Add tool information if available
        if tool_response:
            enhanced_input = f"{enhanced_input}\n\nTool information: {tool_response}"
        
        # Add emotion information if available
        if emotion:
            enhanced_input = f"{enhanced_input}\n\nDetected emotion: {emotion}"
        
        # Add memory context if available
        if memory_context:
            enhanced_input = f"{enhanced_input}\n\n{memory_context}"
        
        # Generate response using the modern chain
        try:
            # Using invoke with proper input structure for LCEL chains
            response = self.chain.invoke({
                "input": enhanced_input,
                "history": messages
            })
            
            # Extract response text from the response object
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Update the memory with the new messages
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response_text)
            
            # Extract and store any facts to remember from the user input
            if self.memory_manager:
                self.memory_manager._extract_facts(user_input, response_text)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response_text = f"I apologize, but I encountered an error while processing your request. Please try again."
        
        # Update long-term memory if available
        if self.memory_manager:
            self.memory_manager.add_interaction(user_input, response_text, emotion)
        
        logger.info(f"Generated response: {response_text[:100]}...")
        return response_text
    
    def save_conversation_history(self) -> None:
        """Save the conversation history to disk."""
        if not self.memory_manager:
            logger.warning("Memory manager not initialized, cannot save conversation history")
            return
        
        try:
            self.memory_manager.save_conversation()
            logger.info("Conversation history saved")
        except Exception as e:
            logger.error(f"Error saving conversation history: {str(e)}")
    
    def load_conversation_history(self, conversation_id: str) -> bool:
        """
        Load a previous conversation history.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            True if successful, False otherwise
        """
        if not self.memory_manager:
            logger.warning("Memory manager not initialized, cannot load conversation history")
            return False
        
        try:
            success = self.memory_manager.load_conversation(conversation_id)
            if success:
                self.conversation_id = conversation_id
                logger.info(f"Loaded conversation history: {conversation_id}")
            else:
                logger.warning(f"Failed to load conversation history: {conversation_id}")
            return success
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")
            return False 