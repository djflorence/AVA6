# Memory System

This document explains the memory system architecture in AVA6 and how it uses ChromaDB for efficient vector storage and retrieval.

## Overview

AVA6's memory system is designed to provide the assistant with both short-term and long-term memory capabilities. It uses ChromaDB as the vector database backend for storing and retrieving conversation history and user preferences.

## Architecture

The memory system consists of several components:

1. **MemoryManager**: The central component that coordinates memory operations
2. **VectorStore**: ChromaDB implementation for storing and retrieving embeddings
3. **Embeddings**: Converts text to vector representations
4. **Memory Types**: Different memory categories (conversation, user preferences, etc.)

## HNSW Configuration

AVA6 uses Hierarchical Navigable Small World (HNSW) graphs for efficient similarity search in ChromaDB. The configuration has been optimized for performance with the following parameters:

```python
hnsw_config = {
    "M": 16,               # Number of bi-directional links created for each new element
    "ef_construction": 100, # Controls index quality and build time
    "ef": 50               # Controls query time/accuracy trade-off
}
```

These parameters provide a good balance between search speed and accuracy.

## Memory Persistence

The memory system ensures that data is properly persisted to disk using the following approach:

1. Checking for the appropriate persistence method based on the ChromaDB client type
2. Calling the appropriate persistence method (`persist()` or `_persist()`)
3. Handling exceptions gracefully to prevent critical failures

## Memory Types

### Conversation Memory

Stores the history of conversations with users, including:
- User messages
- Assistant responses
- Timestamps
- Conversation IDs

### User Preference Memory

Stores information about user preferences, such as:
- Name
- Favorite color
- Important dates
- Custom codes or identifiers

## Usage Example

Here's how the memory system is used in the application:

```python
# Initialize the memory manager
memory_manager = MemoryManager(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./data/memory"
)

# Store a memory
memory_manager.add_memory(
    text="User's favorite color is blue",
    metadata={"type": "preference", "user_id": "user123"}
)

# Retrieve relevant memories
relevant_memories = memory_manager.get_relevant_memories(
    "What color does the user like?",
    k=3
)
```

## Optimization Tips

1. **Batch Operations**: When adding multiple memories, use batch operations to reduce overhead
2. **Regular Persistence**: Call `memory_manager.persist()` at appropriate intervals to ensure data is saved
3. **Memory Cleanup**: Implement a strategy to archive or remove old, less relevant memories
4. **Index Parameters**: Adjust HNSW parameters based on your specific use case and dataset size

## Troubleshooting

### Common Issues

1. **Slow Queries**: If queries are slow, try increasing the `ef` parameter
2. **High Memory Usage**: If memory usage is too high, consider reducing the `M` parameter
3. **Persistence Failures**: Ensure the persistence directory exists and has write permissions 