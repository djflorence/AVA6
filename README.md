# Advanced AI Assistant with LangChain and ChromaDB

An advanced AI chat/assistant bot built with LangChain and ChromaDB, featuring enhanced memory, emotional intelligence, and a robust toolset.

## 🌟 Features

- **Advanced Memory System**: Long-term and short-term memory capabilities using ChromaDB for vector storage
- **Emotional Intelligence**: Emotion detection, tracking, and appropriate responses
- **Tool Integration**: Extensible tool framework for adding capabilities
- **Multi-Modal Support**: Ready for text, and expandable to other modalities
- **Conversation History**: Persistent conversation tracking with context management
- **Customizable Responses**: Tailored responses based on user preferences and history
- **Secure & Private**: Local deployment options with data privacy controls

## 📋 Project Structure

```
.
├── src/                    # Source code
│   ├── assistant/          # Core assistant functionality
│   ├── tools/              # Tool integrations
│   ├── memory/             # Memory management
│   ├── emotions/           # Emotional intelligence
│   ├── embeddings/         # Vector embeddings
│   ├── config/             # Configuration
│   ├── utils/              # Utility functions
│   ├── data/               # Data storage
│   └── tests/              # Unit and integration tests
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
├── .env.example            # Example environment variables
├── setup.py                # Package setup
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-assistant.git
   cd ai-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running the Assistant

```bash
python -m src.main
```

## 🔧 Configuration

The assistant can be configured through:
- Environment variables (see `.env.example`)
- Configuration files in `src/config/`
- Command-line arguments

## 🧠 Memory System

The assistant uses a multi-tiered memory system:
- **Short-term memory**: Recent conversation context
- **Working memory**: Active information for current tasks
- **Long-term memory**: Persistent knowledge stored in ChromaDB
- **Episodic memory**: Past interactions and their outcomes

## 🛠️ Tools

The assistant can be extended with various tools:
- Web search
- Document processing
- Calculator
- Weather information
- Calendar integration
- Custom API integrations

## 🔄 Development Workflow

1. Create a new branch for your feature
2. Implement and test your changes
3. Run the test suite: `pytest`
4. Format code: `black src/`
5. Submit a pull request

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory.
To build and view the documentation locally:

```bash
mkdocs serve
```

Then visit `http://localhost:8000` in your browser.

## 🧪 Testing

Run the test suite:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=src
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue on the GitHub repository. 