# Ava - Emotionally Intelligent AI Assistant

Ava is an advanced AI assistant with emotional intelligence capabilities, designed to understand and respond to user emotions while maintaining a comprehensive memory of interactions.

## Features

- **Emotion Detection**: Recognizes user emotions from text input and responds appropriately
- **Memory Management**: Stores and retrieves information about users, conversations, and preferences
- **Personalized Responses**: Tailors responses based on user history and emotional context
- **Backstory Integration**: Incorporates a consistent backstory to provide depth and personality

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Hugging Face API key (optional, for local emotion detection)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ava.git
   cd ava
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your API keys and configuration.

### Running Ava

To start a conversation with Ava:

```bash
python src/main.py
```

## Development

### Setting Up the Development Environment

We provide a script to set up the development environment:

```bash
python setup_dev_environment.py
```

This script:
1. Installs all required dependencies
2. Sets up pre-commit hooks for code quality
3. Configures the development environment

### Code Quality

We maintain high code quality standards using several tools:

1. **Linting**: We use a combination of flake8, black, and isort to ensure code quality and consistency.

   ```bash
   # Check for linting issues
   python lint.py --check
   
   # Automatically fix linting issues
   python lint.py --fix
   ```

2. **Pre-commit Hooks**: We use pre-commit hooks to automatically check code quality before commits.

   ```bash
   # Install pre-commit hooks (done by setup_dev_environment.py)
   pre-commit install
   
   # Run pre-commit hooks manually
   pre-commit run --all-files
   ```

3. **Testing**: We have comprehensive tests to ensure functionality.

   ```bash
   # Run all tests
   python test_ava_chat.py
   
   # Run specific test suites
   python test_ava_chat.py --test-suite emotions
   ```

## Testing

Comprehensive testing tools are provided to ensure Ava's functionality:

1. Prepare the testing environment:
   ```bash
   python prepare_ava_test.py
   ```

2. Run the tests:
   ```bash
   # Windows
   run_ava_tests.bat
   
   # Linux/Mac
   ./run_ava_tests.sh
   ```

For detailed testing instructions, see [README_TESTING.md](README_TESTING.md).

## Project Structure

```
ava/
├── src/                    # Source code
│   ├── assistant/          # Core assistant functionality
│   ├── emotions/           # Emotion detection components
│   ├── memory/             # Memory management system
│   ├── utils/              # Utility functions
│   └── main.py             # Entry point
├── tests/                  # Test files
├── .env.example            # Example environment variables
├── prepare_ava_test.py     # Test preparation script
├── test_ava_chat.py        # Chat testing script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Memory System

Ava uses ChromaDB as a vector database to store and retrieve memories. Memories are categorized into:

- **Facts**: Information about the user
- **Interactions**: Previous conversations
- **Preferences**: User preferences and settings
- **Reflections**: Ava's thoughts and observations
- **Backstory**: Ava's personal history and characteristics

## Emotion Detection

Ava uses a combination of:

1. Hugging Face transformer models for local emotion detection
2. OpenAI's language models for nuanced emotion understanding

The emotion detection sensitivity can be configured in the `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Hugging Face for emotion detection models
- LangChain for the framework components 