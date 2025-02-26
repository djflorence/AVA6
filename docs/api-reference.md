# API Reference

This document provides a reference for the AVA6 API, detailing the classes and methods available for programmatic use.

## Chat Assistant

The `ChatAssistant` class is the main entry point for interacting with AVA6.

### `ChatAssistant`

```python
from src.chat_assistant import ChatAssistant

assistant = ChatAssistant(config=None)
```

#### Parameters

- `config` (dict, optional): Configuration dictionary. If not provided, configuration is loaded from environment variables or default values.

#### Methods

##### `chat`

```python
response = assistant.chat(user_input, conversation_id=None)
```

Process a user message and generate a response.

**Parameters:**
- `user_input` (str): The user's message
- `conversation_id` (str, optional): Identifier for the conversation. If not provided, a new ID is generated.

**Returns:**
- `str`: The assistant's response

##### `reset`

```python
assistant.reset()
```

Reset the conversation context.

**Returns:**
- None

## Memory Manager

The `MemoryManager` class handles the storage and retrieval of memories.

### `MemoryManager`

```python
from src.memory.memory_manager import MemoryManager
from langchain.embeddings import OpenAIEmbeddings

memory_manager = MemoryManager(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./data/memory"
)
```

#### Parameters

- `embedding_function`: Function to convert text to vector embeddings
- `persist_directory` (str): Directory to store memory data
- `collection_name` (str, optional): Name of the ChromaDB collection. Default: "memories"

#### Methods

##### `add_memory`

```python
memory_manager.add_memory(
    text="User's favorite color is blue",
    metadata={"type": "preference", "user_id": "user123"}
)
```

Add a memory to the vector store.

**Parameters:**
- `text` (str): The text content of the memory
- `metadata` (dict, optional): Additional metadata for the memory

**Returns:**
- None

##### `get_relevant_memories`

```python
memories = memory_manager.get_relevant_memories(
    query="What color does the user like?",
    k=3
)
```

Retrieve memories relevant to a query.

**Parameters:**
- `query` (str): The query to search for
- `k` (int, optional): Number of memories to retrieve. Default: 5

**Returns:**
- `list`: List of Document objects containing the memory text and metadata

##### `persist`

```python
memory_manager.persist()
```

Save memories to disk.

**Returns:**
- None

## LLM Service

The `LLMService` class handles interactions with language models.

### `LLMService`

```python
from src.models.llm_service import LLMService

llm_service = LLMService(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
```

#### Parameters

- `model_name` (str): Name of the language model to use
- `temperature` (float, optional): Temperature for response generation. Default: 0.7
- `max_tokens` (int, optional): Maximum tokens for response. Default: 1000

#### Methods

##### `generate_response`

```python
response = llm_service.generate_response(
    user_input="Hello, who are you?",
    memories=relevant_memories
)
```

Generate a response to a user message.

**Parameters:**
- `user_input` (str): The user's message
- `memories` (list, optional): List of relevant memories

**Returns:**
- `str`: The generated response

## Configuration

The `Config` class handles loading and validation of configuration.

### `Config`

```python
from src.utils.config import Config

config = Config.from_env()
```

#### Class Methods

##### `from_env`

```python
config = Config.from_env()
```

Load configuration from environment variables.

**Returns:**
- `Config`: Configuration object

##### `from_file`

```python
config = Config.from_file("config.yaml")
```

Load configuration from a file.

**Parameters:**
- `file_path` (str): Path to the configuration file

**Returns:**
- `Config`: Configuration object

## Utility Functions

### Logging

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("This is an informational message")
logger.error("This is an error message")
```

### Helpers

```python
from src.utils.helpers import generate_timestamp, generate_id

timestamp = generate_timestamp()
conversation_id = generate_id()
```

## Example Usage

### Basic Conversation

```python
from src.chat_assistant import ChatAssistant

# Initialize the assistant
assistant = ChatAssistant()

# Start a conversation
response = assistant.chat("Hello, who are you?")
print(response)

# Continue the conversation
response = assistant.chat("What can you help me with?")
print(response)
```

### Custom Configuration

```python
from src.chat_assistant import ChatAssistant

# Custom configuration
config = {
    "model": {
        "name": "gpt-3.5-turbo",
        "temperature": 0.8,
        "max_tokens": 2000
    },
    "memory": {
        "persist_directory": "./custom_memory",
        "k": 10
    }
}

# Initialize with custom configuration
assistant = ChatAssistant(config=config)

# Use the assistant
response = assistant.chat("Hello, tell me about yourself.")
print(response)
```

### Working with Memory Directly

```python
from src.memory.memory_manager import MemoryManager
from langchain.embeddings import OpenAIEmbeddings

# Initialize memory manager
memory_manager = MemoryManager(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./data/memory"
)

# Add memories
memory_manager.add_memory(
    text="User's name is Alice",
    metadata={"type": "user_info", "attribute": "name"}
)

memory_manager.add_memory(
    text="User's favorite color is blue",
    metadata={"type": "preference", "attribute": "color"}
)

# Retrieve memories
memories = memory_manager.get_relevant_memories("What is the user's name?")
for memory in memories:
    print(memory.page_content, memory.metadata)

# Save to disk
memory_manager.persist()
``` 