# Mock Testing for AVA Assistant

This document provides instructions on how to use the mock implementation for testing the AVA assistant without relying on the OpenAI API.

## Overview

The mock implementation includes:

1. **MockEmbeddings**: A mock implementation of OpenAI embeddings that generates random embeddings for testing.
2. **MockChatOpenAI**: A mock implementation of the OpenAI chat model that returns predefined responses.
3. **MockEmotionDetector**: A mock implementation of the emotion detector that uses keyword matching.
4. **Modified Memory Loader**: A version of the memory loader that works with mock embeddings.
5. **Test Scripts**: Scripts for running tests with the mock implementation.

## Setup

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. No API keys are required for the mock implementation.

## Loading Mock Memories

To load memories into the system using the mock implementation:

```bash
python load_mock_memories.py --file ava_memory_manual.json --debug
```

This will load the memories from the specified JSON file into the ChromaDB using mock embeddings.

## Running Mock Tests

To run tests using the mock implementation:

```bash
python run_mock_tests.py --debug
```

This will run a series of tests for emotions, memories, personal information, and backstory using the mock implementation.

The test results will be saved to the `test_results` directory.

## Mock Components

### MockEmbeddings

Located in `src/utils/mock_embeddings.py`, this class generates random embeddings for testing. It ensures that the same text always produces the same embedding by using a hash of the text to seed the random generator.

### MockEmotionDetector

Located in `src/utils/mock_emotion_detector.py`, this class detects emotions in text using keyword matching. It supports the following emotions:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Neutral

### MockChatOpenAI

Located in `test_ava_chat_mock.py`, this class simulates responses from the OpenAI API. It includes predefined responses for various queries related to emotions, memories, personal information, and backstory.

## Modifying the Mock Implementation

### Adding New Mock Responses

To add new mock responses to the `MockChatOpenAI` class, modify the `invoke` method in `test_ava_chat_mock.py`.

### Customizing Emotion Detection

To customize the emotion detection in the `MockEmotionDetector` class, modify the `emotion_keywords` dictionary in `src/utils/mock_emotion_detector.py`.

### Adding New Test Cases

To add new test cases to the `run_mock_tests.py` script, modify the `test_categories` list in the `run_tests` function.

## Troubleshooting

### ChromaDB Issues

If you encounter issues with ChromaDB, try deleting the ChromaDB directory and running the tests again:

```bash
rm -rf src/data/chroma
```

### Memory Loading Issues

If you encounter issues with memory loading, check the format of your memory JSON file. The memory loader supports both list and dictionary formats.

### Test Failures

If tests are failing, check the logs in the `logs` directory for more information. The logs include detailed information about the test execution, including the responses from the mock implementation.

## Contributing

Feel free to contribute to the mock implementation by adding new features or improving existing ones. Please follow the existing code style and add appropriate tests for new features. 