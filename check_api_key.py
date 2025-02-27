import os
import sys
from dotenv import load_dotenv, find_dotenv

# Print Python version
print(f"Python version: {sys.version}")

# Find .env file
dotenv_path = find_dotenv()
print(f"Found .env file at: {dotenv_path}")

# Load environment variables from .env file
load_result = load_dotenv(dotenv_path)
print(f"Load result: {load_result}")

# Get all environment variables
print("\nAll environment variables:")
for key, value in os.environ.items():
    if "API" in key:
        print(f"{key}: {value[:10]}..." if value and len(value) > 10 else f"{key}: {value}")

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
print("\nOPENAI_API_KEY from env:", api_key) 