# Ava Testing Guide

This guide explains how to test Ava's emotions and memories functionality to ensure everything is working correctly.

## Prerequisites

Before running the tests, make sure you have:

1. Installed all required dependencies from `requirements.txt`
2. Set up your `.env` file with the necessary API keys and configuration
3. Downloaded the `ava_memory_manual.json` file containing Ava's memories

## Running Tests

### Windows

Use the provided batch file to run all tests:

```bash
run_ava_tests.bat
```

To run specific test suites:

```bash
run_ava_tests.bat --test-suite emotions
run_ava_tests.bat --test-suite memories
```

### Linux/Mac

First, make the shell script executable:

```bash
chmod +x run_ava_tests.sh
```

Then run the tests:

```bash
./run_ava_tests.sh
```

To run specific test suites:

```bash
./run_ava_tests.sh --test-suite emotions
./run_ava_tests.sh --test-suite memories
```

## Manual Testing

If you prefer to run the tests manually, follow these steps:

### Preparation

First, run the preparation script to ensure Ava's memories are properly loaded and the environment is correctly set up:

```bash
python prepare_ava_test.py
```

This script will:
- Check if your environment is properly configured
- Verify if Ava's memories are already loaded in ChromaDB
- Load memories if they don't exist or if you use the `--force-reload` flag
- Test the emotion detection system

#### Options

- `--memory-file PATH`: Specify a custom path to the memory file (default: `ava_memory_manual.json`)
- `--force-reload`: Force reload memories even if they already exist
- `--debug`: Enable debug mode for more detailed logging

### Running Tests Manually

After preparation, you can run the comprehensive test suite:

```bash
python test_ava_chat.py
```

This will run all test suites, including:
1. Emotion detection and response tests
2. Memory retrieval tests
3. Personal information handling tests
4. Backstory recall and emotional integration tests

#### Test Specific Functionality

You can test specific functionality using the `--test-suite` option:

```bash
# Test only emotion detection and response
python test_ava_chat.py --test-suite emotions

# Test only memory retrieval
python test_ava_chat.py --test-suite memories

# Test only personal information handling
python test_ava_chat.py --test-suite personal

# Test only backstory recall and emotional integration
python test_ava_chat.py --test-suite backstory
```

#### Save Test Results

To save the test results to a JSON file for later analysis:

```bash
python test_ava_chat.py --save-results
```

Results will be saved in the `test_results` directory with a timestamp in the filename.

## Troubleshooting

If you encounter issues during testing:

1. **Memory retrieval issues**: 
   - Run `python prepare_ava_test.py --force-reload` to reload Ava's memories
   - Check that your ChromaDB directory exists and has proper permissions

2. **Emotion detection issues**:
   - Ensure you have the required Hugging Face models installed
   - Check your OpenAI API key if using LLM-based emotion detection

3. **API rate limiting**:
   - If you encounter rate limiting errors, add delays between tests with `--delay 2` (seconds)
   - Consider using a different API key or reducing the number of tests

4. **Missing dependencies**:
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed

## Interpreting Results

The test script will output a summary of passed and failed tests for each test suite. For failed tests, check:

1. If the expected phrases were found in Ava's responses
2. If the emotion detection is working correctly
3. If personal information is being stored and retrieved properly

Remember that some tests may be subjective, especially for emotion detection and memory recall, so manual review of the responses is recommended.

## Extending Tests

To add more tests:

1. Add new test cases to the appropriate test function in `test_ava_chat.py`
2. Follow the existing format for test cases
3. Run the tests to verify your new test cases

## Contact

If you encounter persistent issues or have questions about the testing process, please contact the development team. 