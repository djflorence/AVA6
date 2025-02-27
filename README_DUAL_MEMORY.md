# Dual Memory System for Ava

This document describes the implementation of a dual memory system for Ava, which maintains separate collections for Ava's memories and David's information.

## Problem

The original memory system used a single collection for both Ava's memories and David's information, which led to several issues:

1. **Data Overwriting**: When loading both Ava's memories and David's information into the same collection, they could potentially overwrite each other.
2. **Query Complexity**: The memory manager had to use complex logic to determine whether a query was about David or Ava.
3. **Inefficient Searches**: All queries searched the entire collection, even when only a specific subset of memories was relevant.

## Solution

The dual memory system addresses these issues by:

1. **Separate Collections**: Maintaining separate ChromaDB collections for Ava's memories and David's information.
2. **Intelligent Routing**: Routing queries to the appropriate collection based on their content.
3. **Fallback Responses**: Providing fallback responses when no relevant memories are found.

## Implementation

The implementation consists of several components:

### 1. DualMemoryManager

The `DualMemoryManager` class manages separate collections for Ava and David. It:

- Initializes separate vector stores for Ava and David
- Routes queries to the appropriate collection based on their content
- Provides fallback responses for queries about Ava's past when no memories are found

### 2. Memory Loading Scripts

Two scripts are provided to load memories into the separate collections:

- `load_ava_memories.py`: Loads Ava's memories from `ava_memory_manual.json` into the "ava_memories" collection
- `load_david_info.py`: Loads David's information from `david_info.json` into the "david_info" collection

### 3. DualChatAssistant

The `DualChatAssistant` class uses the `DualMemoryManager` to:

- Initialize the dual memory system
- Search for relevant memories when processing user queries
- Add interactions to both memory collections

## Usage

### Preparing the Environment

Before using the dual memory system, you need to prepare the environment by loading both collections:

```bash
# Load Ava's memories and David's information
python src/memory/prepare_dual_memory_test.py
```

### Testing the Dual Memory Manager

You can test the dual memory manager directly:

```bash
# Test the dual memory manager
python test_dual_memory.py
```

### Testing the Dual Chat Assistant

You can test the dual chat assistant:

```bash
# Test the dual chat assistant
python test_dual_chat.py
```

## Benefits

The dual memory system provides several benefits:

1. **Improved Accuracy**: By routing queries to the appropriate collection, the system provides more accurate responses.
2. **Reduced Complexity**: The query logic is simplified by separating the collections.
3. **Better Performance**: Searches are more efficient because they only search the relevant collection.
4. **Fallback Responses**: The system provides reasonable responses even when no memories are found.

## Future Improvements

Potential future improvements include:

1. **More Collections**: Adding more specialized collections for different types of memories.
2. **Better Query Routing**: Improving the logic for determining which collection to search.
3. **Enhanced Fallback Responses**: Making fallback responses more contextually relevant.
4. **Memory Prioritization**: Prioritizing certain memories based on their importance or relevance.

## Files

- `src/memory/dual_memory_manager.py`: The main dual memory manager class
- `src/memory/load_ava_memories.py`: Script to load Ava's memories
- `src/memory/load_david_info.py`: Script to load David's information
- `src/memory/prepare_dual_memory_test.py`: Script to prepare the test environment
- `src/assistant/dual_chat_assistant.py`: Chat assistant that uses the dual memory manager
- `test_dual_memory.py`: Test script for the dual memory manager
- `test_dual_chat.py`: Test script for the dual chat assistant 