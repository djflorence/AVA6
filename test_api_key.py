import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {api_key[:10]}...{api_key[-5:]}")

# Set the API key for OpenAI
openai.api_key = api_key

try:
    # Test the API key with a simple request
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
    )
    print("API Key is valid. Response received.")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}") 