import os
import openai

# Directly set the API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Print the API key
api_key = os.environ.get("OPENAI_API_KEY")
print(f"API Key: {api_key}")

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