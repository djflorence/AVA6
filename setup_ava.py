#!/usr/bin/env python3
"""
Setup script for Ava's memories and emotional system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Set up Ava's memories and emotional system"
    )
    parser.add_argument(
        "--memory-file", 
        type=str, 
        default="ava_memory_manual.json", 
        help="Path to the JSON file containing Ava's memories"
    )
    parser.add_argument(
        "--skip-dependencies", 
        action="store_true", 
        help="Skip installing dependencies"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    return parser.parse_args()


def install_dependencies() -> bool:
    """Install required dependencies."""
    try:
        import subprocess
        
        print("Installing required dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("[SUCCESS] Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error installing dependencies: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return False


def load_memories(memory_file: str) -> bool:
    """Load Ava's memories."""
    try:
        import subprocess
        
        print(f"Loading Ava's memories from {memory_file}...")
        result = subprocess.run(
            [sys.executable, "-m", "src.utils.load_ava_memories", "--file", memory_file],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if "Successfully loaded Ava's memories" in result.stdout:
            print("[SUCCESS] Memories loaded successfully")
            return True
        else:
            print(f"[ERROR] Error loading memories: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error loading memories: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return False


def test_emotion_detection() -> bool:
    """Test the emotion detection system."""
    try:
        import subprocess
        
        print("Testing emotion detection system...")
        
        # Test with different emotions
        test_texts = [
            "I am so happy today!",
            "I feel sad and depressed.",
            "I'm really angry about what happened.",
            "I'm scared of what might happen next.",
            "Wow, that's surprising!"
        ]
        
        for text in test_texts:
            print(f"\nTesting with: \"{text}\"")
            result = subprocess.run(
                [sys.executable, "-m", "src.utils.test_emotion_detection", "--text", text, "--method", "hf"],
                check=True,
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
        
        print("[SUCCESS] Emotion detection system tested successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error testing emotion detection: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return False


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
    
    # Check if memory file exists
    memory_file = args.memory_file
    if not os.path.exists(memory_file):
        logger.error(f"Memory file not found: {memory_file}")
        print(f"[ERROR] Memory file not found: {memory_file}")
        sys.exit(1)
    
    # Install dependencies if not skipped
    if not args.skip_dependencies:
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Load memories
    if not load_memories(memory_file):
        logger.error("Failed to load memories")
        sys.exit(1)
    
    # Test emotion detection
    if not test_emotion_detection():
        logger.error("Failed to test emotion detection")
        sys.exit(1)
    
    print("\n[SUCCESS] Ava's memories and emotional system have been set up successfully!")
    print("\nYou can now run the assistant with:")
    print("  python -m src.main")
    print("\nOr with the web UI:")
    print("  python -m src.main --web")


if __name__ == "__main__":
    main() 