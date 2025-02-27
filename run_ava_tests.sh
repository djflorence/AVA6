#!/bin/bash

echo "Ava Testing Suite"
echo "================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "Step 1: Preparing Ava for testing..."
python3 prepare_ava_test.py "$@"
if [ $? -ne 0 ]; then
    echo
    echo "Error: Preparation failed"
    echo "Please check the error messages above"
    exit 1
fi

echo
echo "Step 2: Running Ava tests..."
echo

# Parse arguments
TEST_SUITE="all"
SAVE_RESULTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        --save-results)
            SAVE_RESULTS="--save-results"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run tests
if [ "$TEST_SUITE" == "all" ]; then
    python3 test_ava_chat.py $SAVE_RESULTS
else
    python3 test_ava_chat.py --test-suite "$TEST_SUITE" $SAVE_RESULTS
fi

if [ $? -ne 0 ]; then
    echo
    echo "Error: Tests failed"
    echo "Please check the error messages above"
    exit 1
fi

echo
echo "All tests completed successfully!"
echo
echo "To view detailed results, check the test_results directory if you used --save-results"
echo 