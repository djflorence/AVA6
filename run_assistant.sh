#!/bin/bash
# Run the AI Assistant with a specific profile

# Check if profile is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_assistant.sh [profile] [--web]"
    echo "Example: ./run_assistant.sh development --web"
    exit 1
fi

# Set profile
PROFILE=$1

# Check for web flag
WEB_FLAG=""
if [ "$2" = "--web" ]; then
    WEB_FLAG="--web"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Run the assistant
echo "Running assistant with profile: $PROFILE"
python -m src.main --config $PROFILE $WEB_FLAG

# Deactivate virtual environment
deactivate 