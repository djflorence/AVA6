#!/usr/bin/env python3
"""
Demo script for the emotion-based response system.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from dotenv import load_dotenv

from src.emotions.emotion_handler import EmotionHandler
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo the emotion-based response system")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["pattern", "llm", "hf"],
        default="pattern",
        help="Emotion detection method to use",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="Sensitivity threshold for emotion detection (0.0-1.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def print_colored(text: str, color: str = None) -> None:
    """Print colored text."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    
    if color and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)


def get_emotion_color(emotion: str) -> str:
    """Get color for an emotion."""
    emotion_colors = {
        "joy": "green",
        "sadness": "blue",
        "anger": "red",
        "fear": "yellow",
        "surprise": "magenta",
        "neutral": "white",
    }
    return emotion_colors.get(emotion, "white")


def main() -> None:
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize emotion handler
        use_llm = args.detector == "llm"
        use_hf = args.detector == "hf"
        
        emotion_handler = EmotionHandler(
            use_llm=use_llm,
            use_hf=use_hf,
            sensitivity=args.sensitivity,
            enable_responses=True
        )
        
        logger.info(f"Initialized emotion handler (Method: {args.detector}, Sensitivity: {args.sensitivity})")
        
        # Print welcome message
        print_colored("\n=== Emotion-Based Response System Demo ===", "cyan")
        print_colored("Type a message to see the detected emotion and response template.", "cyan")
        print_colored("Type 'exit' or 'quit' to end the demo.\n", "cyan")
        
        # Main interaction loop
        while True:
            # Get user input
            user_input = input("> ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print_colored("Goodbye!", "cyan")
                break
                
            # Process the input
            emotion, guidance = emotion_handler.process_input(user_input)
            
            # Print the results
            emotion_color = get_emotion_color(emotion)
            print_colored(f"\nDetected emotion: {emotion}", emotion_color)
            
            if guidance:
                print_colored("\nSystem prompt:", "yellow")
                print(guidance["system_prompt"])
                
                print_colored("\nResponse template:", "green")
                print(guidance["response_template"])
                
                print_colored("\nExample response:", "blue")
                print(guidance["response_template"].replace("[User's concern/question]", user_input))
            else:
                print_colored("No response guidance available.", "red")
                
            print()  # Empty line for readability

    except KeyboardInterrupt:
        print_colored("\nDemo interrupted. Goodbye!", "cyan")
    except Exception as e:
        logger.error(f"Error in emotion response demo: {str(e)}")
        print_colored(f"\n[ERROR] {str(e)}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()