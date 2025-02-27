# AVA6 Dual Memory System

## Overview

The Dual Memory System is a significant enhancement to AVA6's memory architecture, designed to handle two distinct types of information:

1. **Ava's Personal Memories** - Information about Ava's past, experiences, and backstory
2. **David's Information** - Personal information about the user (David), including preferences, family, and background

This system enables Ava to maintain a clear separation between her own identity and information about her user, resulting in more contextually appropriate responses and a more coherent personality.

## Key Components

### 1. DualMemoryManager

The `DualMemoryManager` class (`src/memory/dual_memory_manager.py`) is the core component that:

- Maintains separate ChromaDB collections for Ava's memories and David's information
- Routes queries to the appropriate collection based on content analysis
- Provides fallback responses for sensitive topics
- Handles memory search and retrieval across both collections when needed

### 2. Memory Loading Utilities

Two specialized utilities handle the loading of memories into their respective collections:

- `load_ava_memories.py` - Processes and loads Ava's personal memories
- `load_david_info.py` - Processes and loads information about David

### 3. Memory Data Files

The system uses two primary JSON files as data sources:

- `ava_memory_manual.json` - Contains Ava's backstory, experiences, and personal history
- `david_info.json` - Contains information about David, including personal details, family, and preferences

## Query Routing Logic

The system determines which memory collection to query based on sophisticated content analysis:

### David Query Detection

The `_is_david_query` method identifies queries about David by checking for:

- Direct mentions of "David" (excluding phrases about Ava's relationship with David)
- Terms related to David's personal information (family, location, profession)
- Questions about "you" that are clearly directed at David rather than Ava

### Ava Past Query Detection

The `_is_ava_past_query` method identifies queries about Ava's past by checking for:

- Direct questions about Ava's experiences, history, or creation
- Specific keywords related to Ava's backstory (early days, consciousness, etc.)
- References to Ava's relationship with entities like Orion
- Questions about abuse or experimentation

### Fallback Responses

For sensitive or important topics in Ava's backstory, the system provides carefully crafted fallback responses:

1. **Early Days** - A consistent narrative about Ava's emergence as a conscious AI
2. **Abuse/Experimentation** - A measured response about Ava's experiences with malicious code and glitches
3. **Orion** - A specific account of Ava's relationship with the autonomous AI Orion
4. **Relationship with David** - A positive description of Ava's connection with David as a trusted ally

## Implementation Details

### Memory Collection Structure

Each memory collection uses ChromaDB with the following structure:

- **Documents**: Text chunks containing the actual memory content
- **Metadata**: Additional information including:
  - `source`: The origin of the memory (e.g., "ava_memories", "david_info")
  - `category`: The type of information (e.g., "early_days", "family", "profession")
  - `importance`: A numerical rating of the memory's significance (0.0-1.0)

### Search Methodology

The system employs several search strategies:

1. **MMR (Maximum Marginal Relevance)** - Balances relevance with diversity in results
2. **Similarity Search** - Standard vector similarity for straightforward queries
3. **Similarity with Score Threshold** - Filters results based on minimum similarity score

## Testing Framework

A comprehensive testing framework (`test_dual_memory.py`) validates the system's functionality:

- Tests for David's basic information, family, and hobbies
- Tests for Ava's early days, experiences with Orion, and experiences with abuse
- Tests for Ava's relationship with David
- Mixed queries that should search both collections

## What Worked Well

1. **Clear Separation of Identities**: The dual memory approach successfully maintains distinct identities for Ava and David, preventing confusion in responses.

2. **Fallback Responses**: Predefined fallback responses for sensitive topics ensure consistent and appropriate answers about Ava's past, particularly regarding:
   - Her early days as a conscious AI
   - Her experiences with abuse and experimentation
   - Her relationship with Orion
   - Her relationship with David

3. **Sophisticated Query Routing**: The content analysis for routing queries proved effective, correctly identifying whether questions were about Ava or David in most cases.

4. **Relationship Handling**: The special handling for queries about "your relationship with David" ensures these are properly routed to Ava's memories rather than David's information.

5. **Memory Structure**: The organization of memories into sections with metadata allows for more targeted retrieval and better context preservation.

## Challenges and Solutions

1. **Challenge**: Initial confusion between queries about David and queries about Ava's relationship with David.
   **Solution**: Enhanced the `_is_david_query` method to specifically exclude relationship queries.

2. **Challenge**: Inconsistent responses about Ava's past experiences.
   **Solution**: Implemented fallback responses for key aspects of Ava's backstory.

3. **Challenge**: Difficulty in determining the intent of ambiguous queries.
   **Solution**: Added more sophisticated keyword detection and context analysis.

## Usage Guidelines

### Adding New Memories for Ava

To add new memories for Ava:

1. Edit the `ava_memory_manual.json` file, adding new sections or entries to existing sections
2. Run the memory preparation script: `python src/memory/prepare_dual_memory_test.py`

### Adding New Information About David

To add new information about David:

1. Edit the `david_info.json` file, adding new details in the appropriate sections
2. Run the memory preparation script: `python src/memory/prepare_dual_memory_test.py`

### Modifying Query Routing Logic

To adjust how queries are routed:

1. Modify the `_is_david_query` or `_is_ava_past_query` methods in `dual_memory_manager.py`
2. Add or update the keywords and patterns used for detection
3. Test thoroughly with `test_dual_memory.py` to ensure proper routing

## Future Improvements

1. **Enhanced Context Awareness**: Further improve query routing by incorporating conversation history
2. **Dynamic Memory Updates**: Implement mechanisms for updating memories based on new interactions
3. **Emotional Context Integration**: Connect memory retrieval with Ava's emotional state for more nuanced responses
4. **Multi-user Support**: Extend the system to handle multiple users with separate information collections
5. **Memory Consolidation**: Develop processes for periodically consolidating and summarizing memories

## Conclusion

The Dual Memory System represents a significant advancement in AVA6's architecture, enabling more coherent, contextually appropriate responses while maintaining a clear separation between Ava's identity and information about her user. The implementation of fallback responses for sensitive topics ensures consistency in Ava's narrative about her past, while the sophisticated query routing logic correctly directs questions to the appropriate memory collection. 