# Configuration Guide

This guide explains how to configure the Advanced AI Assistant to suit your needs.

## Environment Variables

The assistant can be configured through environment variables. You can set these in a `.env` file in the root directory of the project.

### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | None |
| `DEFAULT_LLM_MODEL` | Default LLM model to use | `gpt-4-turbo` |
| `ANTHROPIC_MODEL` | Anthropic model to use | `claude-3-opus-20240229` |
| `EMBEDDING_MODEL` | Embedding model to use | `text-embedding-3-large` |
| `TEMPERATURE` | Temperature for response generation | `0.7` |
| `MAX_TOKENS` | Maximum tokens for response generation | `4000` |

### Memory Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_TYPE` | Type of memory to use (`chroma`, `simple`) | `chroma` |
| `CHROMA_DB_DIRECTORY` | Directory for ChromaDB storage | `./src/data/chroma` |
| `MEMORY_WINDOW_SIZE` | Number of recent messages to keep in short-term memory | `10` |

### Emotion Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_EMOTIONS` | Whether to enable emotion detection | `true` |
| `EMOTION_SENSITIVITY` | Sensitivity for emotion detection (0.0 to 1.0) | `0.7` |

### Tool Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_WEB_SEARCH` | Whether to enable web search | `false` |
| `ENABLE_CALCULATOR` | Whether to enable calculator | `true` |
| `ENABLE_WEATHER` | Whether to enable weather information | `false` |
| `WEATHER_API_KEY` | API key for weather service | None |

### Application Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO` |
| `ENABLE_TRACING` | Whether to enable tracing | `false` |

### Web UI Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WEB_UI_PORT` | Port for web UI | `8000` |
| `WEB_UI_HOST` | Host for web UI | `127.0.0.1` |
| `ENABLE_AUTH` | Whether to enable authentication for web UI | `false` |

## Configuration Profiles

You can create configuration profiles to easily switch between different configurations. Profiles are stored as JSON files in the `src/config/profiles` directory.

### Creating a Profile

Create a JSON file in the `src/config/profiles` directory with the name of your profile (e.g., `production.json`):

```json
{
  "DEFAULT_LLM_MODEL": "gpt-4-turbo",
  "MEMORY_TYPE": "chroma",
  "ENABLE_EMOTIONS": true,
  "ENABLE_WEB_SEARCH": true,
  "LOG_LEVEL": "INFO"
}
```

### Using a Profile

To use a profile, specify it when running the assistant:

```bash
python -m src.main --config production
```

## Command-Line Arguments

The assistant supports the following command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Configuration profile to use | `default` |
| `--model` | Override the LLM model specified in config | None |
| `--debug` | Enable debug mode | `false` |
| `--web` | Start the web UI | `false` |
| `--port` | Port for web UI (if enabled) | `8000` |

Example:

```bash
python -m src.main --config production --model gpt-4-turbo --web --port 8080
```

## Advanced Configuration

### Custom Tools

To add custom tools, create a new tool class that inherits from `BaseTool` in the `src/tools` directory. Then register it in the `ToolManager._init_tools` method.

Example:

```python
class MyCustomTool(BaseTool):
    name = "my_custom_tool"
    description = "My custom tool description"
    
    def can_handle(self, query: str) -> bool:
        # Determine if this tool can handle the query
        return "custom" in query.lower()
    
    def execute(self, query: str) -> str:
        # Execute the tool
        return "Custom tool response"
```

### Custom Embedding Models

To use a custom embedding model, set the `EMBEDDING_MODEL` environment variable to the name of your model. You can also modify the `EmbeddingProvider` class in `src/embeddings/embedding_utils.py` to support additional embedding providers. 