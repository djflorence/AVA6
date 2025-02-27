# AVA6 Search and Information Retrieval Test Suite

This directory contains test scripts for evaluating AVA6's search and information retrieval capabilities. These tests help assess how well the assistant can search for information, retrieve data from memory, and combine both capabilities to provide comprehensive responses.

## Available Test Scripts

### 1. `test_search_capabilities.py`

A comprehensive test script that evaluates AVA6's ability to search for information and retrieve data from memory. It tests:

- Web search functionality with different queries
- Memory retrieval capabilities
- Tool selection for different types of queries
- Integration between search and memory

### 2. `test_chat_interaction.py`

An interactive chat test script that simulates a conversation with AVA6 to test its searching and information retrieval capabilities in a more natural way. It includes:

- Search-focused conversations
- Memory-focused conversations
- Combined search and memory conversations
- Complex query conversations

### 3. `test_web_search_comprehensive.py`

A focused test script specifically for testing the web search functionality with different types of queries:

- Factual queries (e.g., "What is the capital of France?")
- Current events queries (e.g., "What are the latest news about AI?")
- How-to queries (e.g., "How to make pasta carbonara?")
- Comparison queries (e.g., "Compare Python vs JavaScript")
- Opinion-based queries (e.g., "What are the best programming languages to learn?")
- Complex queries that require deeper understanding

### 4. `test_web_search.py`

A basic test script for the web search functionality, focusing on the core functionality of the web search tool.

## Running the Tests

### Prerequisites

Before running the tests, make sure you have:

1. Set up your environment variables in the `.env` file, including:
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for the LLM
   - Search provider API keys as needed (`SERPAPI_API_KEY`, `GOOGLE_API_KEY`, etc.)

2. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Individual Tests

To run a specific test script, use the following command from the project root directory:

```bash
python -m src.tests.test_script_name
```

For example:

```bash
python -m src.tests.test_chat_interaction
```

### Test Results

All test scripts generate detailed logs and save results to the `src/logs` directory. The results include:

- Success rates for different types of queries
- Response times
- Detailed information about each query and response
- Summary statistics

## Customizing Tests

You can customize the tests by modifying the query lists in each test script. For example, to add more factual queries to the web search test, edit the `factual_queries` list in `test_web_search_comprehensive.py`.

## Interpreting Results

The test results provide insights into:

1. **Search Accuracy**: How well AVA6 can find relevant information for different types of queries.
2. **Memory Retrieval**: How effectively AVA6 can recall information from previous conversations.
3. **Tool Selection**: Whether AVA6 correctly chooses the appropriate tool for different queries.
4. **Response Quality**: The relevance and completeness of AVA6's responses.
5. **Performance**: Response times for different types of queries and search providers.

Use these insights to identify areas for improvement in AVA6's search and information retrieval capabilities. 