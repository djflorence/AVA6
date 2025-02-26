# Architecture

This document provides an overview of AVA6's architecture, explaining how the different components work together.

## System Overview

AVA6 is built with a modular architecture that separates concerns and allows for easy extension and customization. The main components are:

1. **Chat Assistant**: The central component that orchestrates the interaction between the user and the system
2. **Memory System**: Manages the storage and retrieval of conversation history and user preferences
3. **Language Models**: Handles the generation of responses using large language models
4. **Utilities**: Provides common functionality used across the system

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│                  User                       │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              Chat Assistant                 │
└───────┬─────────────┬─────────────┬─────────┘
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ LLM Service  │ │  Memory  │ │    Utils     │
└──────────────┘ └────┬─────┘ └──────────────┘
                      │
                      ▼
                ┌──────────┐
                │ ChromaDB │
                └──────────┘
```

## Component Details

### Chat Assistant (`src/chat_assistant.py`)

The Chat Assistant is the main entry point for the application. It:

- Processes user input
- Manages conversation context
- Coordinates between the memory system and language models
- Formats and returns responses

```python
class ChatAssistant:
    def __init__(self, config=None):
        # Initialize components
        self.memory_manager = MemoryManager(...)
        self.llm = LLMService(...)
        
    def chat(self, user_input, conversation_id=None):
        # Process user input and generate response
        memories = self.memory_manager.get_relevant_memories(user_input)
        response = self.llm.generate_response(user_input, memories)
        self.memory_manager.add_memory(user_input, response)
        return response
```

### Memory System (`src/memory/`)

The Memory System handles the storage and retrieval of information. It consists of:

- **MemoryManager** (`memory_manager.py`): Coordinates memory operations
- **VectorStore** (ChromaDB): Stores and retrieves vector embeddings

```python
class MemoryManager:
    def __init__(self, embedding_function, persist_directory):
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        
    def add_memory(self, text, metadata=None):
        # Store a memory
        self.vectorstore.add_texts([text], metadatas=[metadata])
        
    def get_relevant_memories(self, query, k=5):
        # Retrieve relevant memories
        return self.vectorstore.similarity_search(query, k=k)
        
    def persist(self):
        # Save memories to disk
        self._persist_vectorstore()
```

### LLM Service (`src/models/llm_service.py`)

The LLM Service handles the interaction with language models:

- Configures the language model
- Formats prompts with appropriate context
- Handles token limits and other constraints
- Processes model responses

```python
class LLMService:
    def __init__(self, model_name, temperature=0.7, max_tokens=1000):
        # Initialize language model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    def generate_response(self, user_input, memories=None):
        # Create prompt with context
        prompt = self._create_prompt(user_input, memories)
        
        # Generate response
        return self.llm.predict(prompt)
```

### Utilities (`src/utils/`)

The Utilities module provides common functionality used across the system:

- **Config** (`config.py`): Handles configuration loading and validation
- **Logging** (`logging.py`): Provides logging functionality
- **Helpers** (`helpers.py`): Contains utility functions

## Data Flow

1. User sends a message to the Chat Assistant
2. Chat Assistant queries the Memory Manager for relevant memories
3. Memory Manager retrieves memories from ChromaDB
4. Chat Assistant formats a prompt with the user message and memories
5. LLM Service generates a response using the prompt
6. Chat Assistant stores the interaction in the Memory Manager
7. Memory Manager persists the memory to disk
8. Chat Assistant returns the response to the user

## Extension Points

AVA6 is designed to be extensible. Here are the main extension points:

1. **Custom Memory Types**: Add new memory types by extending the Memory Manager
2. **Alternative LLMs**: Support different language models by implementing the LLM interface
3. **Custom Embeddings**: Use different embedding models by providing a custom embedding function
4. **Additional Tools**: Add new capabilities by implementing tool interfaces

## Configuration

The system is configured through:

- Environment variables
- Configuration files
- Command-line arguments

See the [Configuration Guide](../user-guide/configuration.md) for details.

## Performance Considerations

- **Memory Optimization**: The HNSW configuration in ChromaDB is optimized for performance
- **Batch Operations**: Memory operations are batched when possible
- **Caching**: Frequently used data is cached to reduce database queries
- **Asynchronous Operations**: Long-running operations are performed asynchronously when possible 