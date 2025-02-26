# Getting Started with AVA6

This guide will help you get up and running with AVA6 quickly.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10 or higher
- pip (Python package manager)
- An OpenAI API key or other supported LLM provider API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/djflorence/AVA6.git
   cd AVA6
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   
4. Edit the `.env` file with your API keys and configuration settings.

## Quick Start

To start the assistant, run:

```bash
python -m src.main
```

This will launch the assistant in interactive mode, allowing you to chat with it directly.

## Basic Usage

Here are some examples of how to interact with AVA6:

```
You: Hello, who are you?
AVA6: I'm AVA6, an advanced AI assistant. How can I help you today?

You: Can you remember something for me?
AVA6: Of course! What would you like me to remember?

You: My favorite color is blue
AVA6: I'll remember that your favorite color is blue.
```

## Next Steps

- Learn more about [configuration options](user-guide/configuration.md)
- Explore the [memory system](developer-guide/memory-system.md)
- Check out the [API reference](api-reference.md) for programmatic usage 