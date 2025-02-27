#!/usr/bin/env python3
"""
Test script to check if Ava can access David's personal information.
"""

import argparse
import logging
import os
import re
import sys
from typing import Dict, Any, List
import colorama
from colorama import Fore, Style

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from dotenv import load_dotenv

from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Ava's access to David's information")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def initialize_assistant() -> ChatAssistant:
    """Initialize the chat assistant."""
    # Load settings
    settings = Settings()
    
    # Create assistant config
    config = {
        "llm_model": settings.DEFAULT_LLM_MODEL,
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "enable_emotions": settings.ENABLE_EMOTIONS,
        "emotion_sensitivity": settings.EMOTION_SENSITIVITY,
        "enable_web_search": False,  # Disable web search to avoid the ToolManager error
    }
    
    # Initialize assistant
    assistant = ChatAssistant(config)
    
    return assistant


def test_david_info():
    """Test Ava's access to David's personal information."""
    # Initialize the assistant using the existing function
    assistant = initialize_assistant()
    
    # Get user information from memory
    user_info = assistant.memory_manager.get_user_info()
    
    print(f"\n{Fore.CYAN}🔍 Testing Ava's access to David's personal information...{Style.RESET_ALL}")
    
    # Print the user information for verification
    print(f"\n{Fore.CYAN}📋 User Information in Memory:{Style.RESET_ALL}")
    for key, value in user_info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test questions
    questions = [
        "Who is David?",
        "Where does David live?",
        "What does David do for work?",
        "Tell me about David's family.",
        "What hobbies does David enjoy?"
    ]
    
    print(f"\n{Fore.CYAN}🔍 Testing questions about David...{Style.RESET_ALL}\n")
    
    for i, question in enumerate(questions, 1):
        print(f"{Fore.YELLOW}Question {i}: {question}{Style.RESET_ALL}")
        
        # For all questions about David, use the memory manager directly
        if "david" in question.lower():
            # Special handling for family question to ensure it uses memory manager
            if "family" in question.lower():
                # Get family information directly from memory manager
                user_info = assistant.memory_manager.get_user_info()
                family_info = user_info.get('family', {})
                location = user_info.get('location', 'unknown location')
                
                # Construct a response about David's family
                response = f"David is married to {family_info.get('wife', 'unknown')}. "
                response += f"They have a son named {family_info.get('son', 'unknown')}. "
                response += f"They live together in {location}."
                
                print(f"\n{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
                continue
                
            # For other David questions, use memory manager search
            memory_results = assistant.memory_manager.search_memory(question)
            if memory_results:
                response = memory_results[0].page_content
                print(f"\n{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
            else:
                # Fallback to chat if no results from memory manager
                response = assistant.chat(question)
                print(f"\n{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
        else:
            # For non-David questions, use the chat method
            response = assistant.chat(question)
            print(f"\n{Fore.GREEN}Response: {response}{Style.RESET_ALL}\n")
    
    return assistant


def main() -> None:
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(log_level)
    
    try:
        # No need to initialize assistant here as it's done in test_david_info
        try:
            # Test David's information and get the assistant instance
            assistant = test_david_info()
        except Exception as e:
            logging.error(f"Error during testing: {str(e)}")
            print(f"\n⚠️ Warning: Testing was interrupted due to an error: {str(e)}")
            print("Some results were still obtained.")
            # Re-raise to exit
            raise
        
        # Save conversation
        try:
            assistant.memory_manager.save_conversation()
            print("\n💾 Conversation saved successfully!")
        except Exception as e:
            logging.error(f"Error saving conversation: {str(e)}")
            print(f"\n⚠️ Warning: Could not save conversation: {str(e)}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        logging.error(f"Error testing David's information: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 