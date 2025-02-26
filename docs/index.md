# AVA6: Advanced AI Assistant

Welcome to the documentation for AVA6, an advanced AI assistant built with LangChain and ChromaDB.

## Overview

AVA6 is a powerful AI assistant that combines the capabilities of large language models with a sophisticated memory system to provide a more personalized and context-aware experience. It leverages LangChain for orchestrating the AI components and ChromaDB for efficient vector storage and retrieval.

## Key Features

- **Long-term Memory**: Remembers past conversations and user preferences across sessions
- **Context-aware Responses**: Generates responses that take into account the full conversation history
- **Optimized Vector Storage**: Uses ChromaDB with HNSW configuration for fast similarity searches
- **Modular Architecture**: Easy to extend and customize for different use cases

## Getting Started

To get started with AVA6, check out the [Getting Started](getting-started.md) guide or dive into the [Installation](user-guide/installation.md) instructions.

## Project Structure

```
AVA6/
├── src/                    # Source code
│   ├── memory/             # Memory management components
│   ├── models/             # Model configurations
│   ├── utils/              # Utility functions
│   └── main.py             # Entry point
├── tests/                  # Test suite
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD pipelines
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](developer-guide/contributing.md) for more information on how to get involved. 