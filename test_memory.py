#!/usr/bin/env python3
"""
Test script for memory improvements in the AI Assistant.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings

def main():
    """Run the memory test."""
    print("\n🧪 Testing AI Assistant Memory Improvements\n")
    
    # Load configuration
    settings = Settings()
    config = settings.dict()
    
    # Create data directories if they don't exist
    os.makedirs("src/data/conversations", exist_ok=True)
    os.makedirs(config["CHROMA_DB_DIRECTORY"], exist_ok=True)
    
    # Initialize the assistant
    print("Initializing assistant...")
    assistant = ChatAssistant(config)
    conversation_id = assistant.conversation_id
    print(f"Conversation ID: {conversation_id}")
    
    # Test 1: Basic memory storage
    print("\n📝 Test 1: Basic memory storage")
    print("User: My name is David")
    response = assistant.chat("My name is David")
    print(f"Assistant: {response}")
    
    print("\nUser: Remember the number 12345")
    response = assistant.chat("Remember the number 12345")
    print(f"Assistant: {response}")
    
    # Save conversation
    assistant.save_conversation_history()
    print(f"Saved conversation: {conversation_id}")
    
    # Test 2: Create a new conversation and test cross-conversation memory
    print("\n🔄 Test 2: Cross-conversation memory retrieval")
    print("Creating new conversation...")
    new_assistant = ChatAssistant(config)
    new_conversation_id = new_assistant.conversation_id
    print(f"New conversation ID: {new_conversation_id}")
    
    print("\nUser: What's my name?")
    response = new_assistant.chat("What's my name?")
    print(f"Assistant: {response}")
    
    print("\nUser: What number did I ask you to remember?")
    response = new_assistant.chat("What number did I ask you to remember?")
    print(f"Assistant: {response}")
    
    # Save the second conversation
    new_assistant.save_conversation_history()
    print(f"Saved conversation: {new_conversation_id}")
    
    print("\n✅ Memory test completed")

if __name__ == "__main__":
    main() 