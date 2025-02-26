#!/usr/bin/env python3
"""
Comprehensive test script for memory improvements in the AI Assistant.
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
    """Run the comprehensive memory test."""
    print("\n🧪 Comprehensive Memory System Test\n")
    
    # Load configuration
    settings = Settings()
    config = settings.dict()
    
    # Create data directories if they don't exist
    os.makedirs("src/data/conversations", exist_ok=True)
    os.makedirs(config["CHROMA_DB_DIRECTORY"], exist_ok=True)
    
    # Test 1: Basic conversation and memory storage
    print("\n📝 Test 1: Basic conversation and memory storage")
    print("Initializing first assistant...")
    assistant1 = ChatAssistant(config)
    conversation_id1 = assistant1.conversation_id
    print(f"Conversation ID: {conversation_id1}")
    
    # Store personal information
    print("\nUser: My name is Alice")
    response = assistant1.chat("My name is Alice")
    print(f"Assistant: {response}")
    
    # Store a fact to remember
    print("\nUser: Remember that my favorite color is blue")
    response = assistant1.chat("Remember that my favorite color is blue")
    print(f"Assistant: {response}")
    
    # Store a number
    print("\nUser: Please remember the code 9876")
    response = assistant1.chat("Please remember the code 9876")
    print(f"Assistant: {response}")
    
    # Ask about stored information in the same conversation
    print("\nUser: What's my name?")
    response = assistant1.chat("What's my name?")
    print(f"Assistant: {response}")
    
    print("\nUser: What's my favorite color?")
    response = assistant1.chat("What's my favorite color?")
    print(f"Assistant: {response}")
    
    # Save conversation
    assistant1.save_conversation_history()
    print(f"Saved conversation: {conversation_id1}")
    
    # Test 2: New conversation with the same user
    print("\n🔄 Test 2: New conversation with the same user")
    print("Initializing second assistant...")
    assistant2 = ChatAssistant(config)
    conversation_id2 = assistant2.conversation_id
    print(f"Conversation ID: {conversation_id2}")
    
    # Ask about stored information from previous conversation
    print("\nUser: What's my name?")
    response = assistant2.chat("What's my name?")
    print(f"Assistant: {response}")
    
    print("\nUser: What's my favorite color?")
    response = assistant2.chat("What's my favorite color?")
    print(f"Assistant: {response}")
    
    print("\nUser: What code did I ask you to remember?")
    response = assistant2.chat("What code did I ask you to remember?")
    print(f"Assistant: {response}")
    
    # Store new information
    print("\nUser: Remember that I have a meeting tomorrow at 2pm")
    response = assistant2.chat("Remember that I have a meeting tomorrow at 2pm")
    print(f"Assistant: {response}")
    
    # Save conversation
    assistant2.save_conversation_history()
    print(f"Saved conversation: {conversation_id2}")
    
    # Test 3: Third conversation with combined knowledge
    print("\n🔄 Test 3: Third conversation with combined knowledge")
    print("Initializing third assistant...")
    assistant3 = ChatAssistant(config)
    conversation_id3 = assistant3.conversation_id
    print(f"Conversation ID: {conversation_id3}")
    
    # Ask about all stored information
    print("\nUser: What do you know about me?")
    response = assistant3.chat("What do you know about me?")
    print(f"Assistant: {response}")
    
    print("\nUser: When is my meeting?")
    response = assistant3.chat("When is my meeting?")
    print(f"Assistant: {response}")
    
    # Test memory with a complex query
    print("\nUser: What was the code I asked you to remember and what's my favorite color?")
    response = assistant3.chat("What was the code I asked you to remember and what's my favorite color?")
    print(f"Assistant: {response}")
    
    # Save conversation
    assistant3.save_conversation_history()
    print(f"Saved conversation: {conversation_id3}")
    
    print("\n✅ Comprehensive memory test completed")

if __name__ == "__main__":
    main() 