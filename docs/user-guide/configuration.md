# Configuration Guide

This guide explains how to configure AVA6 to suit your needs.

## Configuration Methods

AVA6 can be configured through:

1. Environment variables
2. Configuration files
3. Command-line arguments

## Environment Variables

The simplest way to configure AVA6 is through environment variables. Copy the `.env.example` file to `.env` and edit it:

```bash
cp .env.example .env
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `MEMORY_PERSIST_DIR` | Directory to store memory data | `./data/memory` |

### Optional Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MODEL_NAME` | The LLM model to use | `gpt-4` | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | The embedding model to use | `text-embedding-ada-002` | `text-embedding-3-small` |
| `MAX_TOKENS` | Maximum tokens for LLM responses | `1000` | `2000` |
| `TEMPERATURE` | Temperature for LLM responses | `0.7` | `0.9` |
| `MEMORY_K` | Number of memories to retrieve | `5` | `10` |

## Configuration File

For more advanced configuration, you can use a YAML configuration file:

```bash
python -m src.main --config config.yaml
```

Example `config.yaml`:

```yaml
model:
  name: gpt-4
  temperature: 0.7
  max_tokens: 1000
  
memory:
  persist_directory: ./data/memory
  embedding_model: text-embedding-ada-002
  k: 5
  
  hnsw_config:
    M: 16
    ef_construction: 100
    ef: 50
```

## Command-line Arguments

You can also configure AVA6 using command-line arguments:

```bash
python -m src.main --model gpt-4 --temperature 0.7 --memory-dir ./data/memory
```

Run with `--help` to see all available options:

```bash
python -m src.main --help
```

## Advanced Configuration

### Memory System Configuration

The memory system can be fine-tuned with the following parameters:

```yaml
memory:
  # Basic settings
  persist_directory: ./data/memory
  embedding_model: text-embedding-ada-002
  k: 5
  
  # HNSW configuration for ChromaDB
  hnsw_config:
    M: 16                # Number of bi-directional links
    ef_construction: 100 # Index quality vs. build time
    ef: 50               # Query time/accuracy trade-off
    
  # Memory types to enable
  types:
    conversation: true
    user_preferences: true
    factual: true
```

### Model Configuration

Configure the language model behavior:

```yaml
model:
  name: gpt-4
  temperature: 0.7
  max_tokens: 1000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
  system_message: "You are AVA6, an advanced AI assistant..."
```

## Configuration Examples

### Minimal Configuration

```yaml
model:
  name: gpt-3.5-turbo
  
memory:
  persist_directory: ./data/memory
```

### Production Configuration

```yaml
model:
  name: gpt-4
  temperature: 0.7
  max_tokens: 2000
  
memory:
  persist_directory: /var/lib/ava6/memory
  embedding_model: text-embedding-3-large
  k: 10
  
  hnsw_config:
    M: 64
    ef_construction: 200
    ef: 100
    
logging:
  level: INFO
  file: /var/log/ava6/ava6.log
```

## Troubleshooting

If you encounter issues with your configuration:

1. Check that all required environment variables are set
2. Verify the YAML syntax in your configuration file
3. Ensure the memory persistence directory exists and is writable
4. Check the logs for specific error messages 