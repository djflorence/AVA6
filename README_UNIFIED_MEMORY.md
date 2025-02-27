# Unified Memory System for AVA6

## Overview

The Unified Memory System is an enhancement to AVA6's memory architecture that streamlines query processing through intent classification and unified memory storage. This document outlines the system's design, implementation, and usage.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
4. [Intent Classification](#intent-classification)
5. [Memory Storage](#memory-storage)
6. [Query Processing](#query-processing)
7. [Performance Comparison](#performance-comparison)
8. [Usage Guide](#usage-guide)
9. [Future Improvements](#future-improvements)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Unified Memory System addresses limitations in the previous dual memory architecture by:

- Implementing automatic intent classification for queries
- Consolidating memory storage while preserving source distinctions
- Streamlining the query routing process
- Providing more accurate and contextually relevant responses

This approach reduces the complexity of memory management while improving response quality and processing efficiency.

## System Architecture

The Unified Memory System consists of three main components:

1. **Intent Classifier**: Determines the intent behind user queries using a zero-shot classification model
2. **Unified Memory Storage**: A single ChromaDB collection with metadata to distinguish between memory sources
3. **Query Router**: Routes queries to the appropriate memory segments based on classified intent

```
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐
│  User Query │────▶│ Intent Classifier│────▶│ Memory Filter     │
└─────────────┘     └─────────────────┘     │ Generation        │
                                            └─────────┬─────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐
│  Response   │◀────│ Result Processing│◀────│ Unified ChromaDB  │
└─────────────┘     └─────────────────┘     │ Collection        │
                                            └───────────────────┘
```

## Key Components

### UnifiedMemoryManager

The `UnifiedMemoryManager` class serves as the central component of the system, providing:

- Initialization of the intent classification model
- Management of the unified ChromaDB collection
- Methods for adding and retrieving memories
- Fallback mechanisms for handling edge cases

### Intent Classification Pipeline

The system uses a pre-trained zero-shot classification model (`facebook/bart-large-mnli`) to determine the intent behind user queries, with a fallback to rule-based classification when needed.

### Memory Preparation Script

The `prepare_unified_memory.py` script migrates existing memories from separate collections into the unified collection, preserving source information through metadata.

## Intent Classification

The system classifies queries into the following intent categories:

- `david_personal`: Queries about David's personal information
- `david_family`: Queries about David's family
- `david_work`: Queries about David's professional life
- `ava_past`: Queries about Ava's past experiences
- `ava_capabilities`: Queries about what Ava can do
- `ava_relationship_with_david`: Queries about Ava's relationship with David
- `general`: General queries that don't fit into other categories

Classification is performed using:

1. **Zero-shot classification**: The primary method, using a pre-trained model
2. **Rule-based classification**: A fallback method using keyword matching

## Memory Storage

All memories are stored in a single ChromaDB collection with metadata fields:

- `source`: Identifies the memory source (e.g., "ava_memories", "david_information")
- `section`: Categorizes the memory within its source
- `importance`: A numerical rating of the memory's importance
- Additional metadata specific to the memory type

## Query Processing

The query processing flow:

1. Receive user query
2. Classify the intent
3. Generate appropriate filters based on the intent
4. Search the unified memory collection with the filters
5. Check for fallback responses if needed
6. Process and return the results

## Performance Comparison

The `compare_memory_managers.py` script provides a comprehensive comparison between the original Dual Memory System and the new Unified Memory System, measuring:

- **Accuracy**: Correct routing of queries to appropriate memory sources
- **Response Quality**: Presence of expected information in responses
- **Processing Time**: Time taken to process queries and return results

### Running the Comparison

```bash
python compare_memory_managers.py --save --plot
```

Options:
- `--save`: Save test results to JSON files
- `--plot`: Generate performance comparison plots
- `--delay`: Set delay between tests (default: 0.5s)
- `--debug`: Enable debug logging

### Comparison Metrics

The comparison evaluates:

- **Pass Rate**: Percentage of tests that return expected information
- **Average Processing Time**: Mean time to process queries
- **Phrase Coverage**: Percentage of expected phrases found in responses
- **Per-Query Performance**: Detailed breakdown of performance for each test query

## Usage Guide

### Initializing the Unified Memory Manager

```python
from src.memory.unified_memory_manager import UnifiedMemoryManager

# Initialize the manager
memory_manager = UnifiedMemoryManager(
    chroma_dir="path/to/chroma",
    use_intent_classification=True
)
```

### Preparing the Unified Memory Collection

```bash
python src/memory/prepare_unified_memory.py
```

### Searching Memory

```python
# Search memory with automatic intent classification
results = memory_manager.search_memory("Tell me about David's family")

# Process results
for doc in results:
    print(doc.page_content)
    print(f"Source: {doc.metadata['source']}")
```

### Testing the System

```bash
python test_unified_memory.py --save
```

## Future Improvements

1. **Enhanced Intent Classification**: Train a custom model on domain-specific data
2. **Dynamic Memory Weighting**: Adjust memory importance based on usage patterns
3. **Multi-hop Reasoning**: Enable complex queries that require information from multiple memory sources
4. **Temporal Context**: Incorporate time-awareness in memory retrieval
5. **Emotion-aware Responses**: Adjust responses based on emotional context

## Troubleshooting

### Common Issues

1. **Intent Classification Failures**
   - Check if the model is properly loaded
   - Verify that the query is clear and well-formed
   - Consider adding more rules to the rule-based fallback

2. **Missing Results**
   - Ensure the unified collection is properly prepared
   - Check if the filters are too restrictive
   - Verify that the expected information exists in the collection

3. **Slow Performance**
   - Consider optimizing the ChromaDB collection
   - Reduce the complexity of filters
   - Adjust the number of results returned

### Logging

The system uses Python's logging module for troubleshooting:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

---

## Conclusion

The Unified Memory System represents a significant advancement in AVA6's memory architecture, providing more accurate, efficient, and contextually relevant responses through intelligent intent classification and unified memory storage. By consolidating memory while preserving source distinctions, the system maintains the benefits of specialized memory handling while reducing complexity and improving performance. 