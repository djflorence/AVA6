#!/usr/bin/env python3
"""
Main entry point for the Advanced AI Assistant.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from src.assistant.chat_assistant import ChatAssistant
from src.config.settings import Settings
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced AI Assistant with LangChain and ChromaDB"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="default", 
        help="Configuration profile to use"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Override the LLM model specified in config"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Start the web UI"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port for web UI (if enabled)"
    )
    return parser.parse_args()


def load_config(profile: str = "default") -> Dict[str, Any]:
    """
    Load configuration from .env file and specified profile.
    
    Args:
        profile: Configuration profile name
        
    Returns:
        Dict containing merged configuration
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Create settings object
    settings = Settings()
    
    # Override with profile-specific settings if needed
    if profile != "default":
        try:
            profile_settings = settings.load_profile(profile)
            settings.update(profile_settings)
        except FileNotFoundError:
            logging.warning(f"Profile '{profile}' not found, using default settings")
    
    return settings.dict()


def main() -> None:
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config.get("LOG_LEVEL", "INFO")
    setup_logger(log_level)
    
    # Override model if specified
    if args.model:
        config["DEFAULT_LLM_MODEL"] = args.model
    
    # Initialize the assistant
    assistant = ChatAssistant(config)
    
    # Start web UI if requested
    if args.web:
        from src.utils.web_ui import start_web_ui
        start_web_ui(assistant, port=args.port)
        return
    
    # Otherwise, start CLI chat loop
    try:
        print("\n🤖 Advanced AI Assistant initialized. Type 'exit' or 'quit' to end the session.\n")
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋")
                break
                
            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}")
            
    except KeyboardInterrupt:
        print("\nSession terminated by user. Goodbye! 👋")
    except Exception as e:
        logging.error(f"Error in chat loop: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Cleanup
        assistant.memory_manager.save_conversation()


if __name__ == "__main__":
    main() 