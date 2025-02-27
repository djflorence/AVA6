#!/usr/bin/env python3
"""
Test script for the emotion detection system.
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from src.emotions.emotion_detector import EmotionDetector
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the emotion detection system")
    parser.add_argument("--text", type=str, help="Text to analyze for emotions")
    parser.add_argument(
        "--method",
        type=str,
        choices=["hf", "llm", "pattern", "all"],
        default="all",
        help="Emotion detection method to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


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

    # Check if text is provided
    if not args.text:
        print("Please provide text to analyze using the --text argument")
        sys.exit(1)

    try:
        # Initialize emotion detectors based on method
        if args.method == "hf" or args.method == "all":
            hf_detector = EmotionDetector(use_llm=False, use_hf=True)

        if args.method == "llm" or args.method == "all":
            llm_detector = EmotionDetector(use_llm=True, use_hf=False)

        if args.method == "pattern" or args.method == "all":
            pattern_detector = EmotionDetector(use_llm=False, use_hf=False)

        # Analyze text
        print(f'\nAnalyzing text: "{args.text}"\n')

        if args.method == "hf" or args.method == "all":
            hf_emotion = hf_detector.detect_emotion(args.text)
            print(f"Hugging Face model detected emotion: {hf_emotion}")

            # Get intensity for the detected emotion
            intensity = hf_detector.get_emotion_intensity(args.text, hf_emotion)
            print(f"Emotion intensity: {intensity:.2f}")

        if args.method == "llm" or args.method == "all":
            llm_emotion = llm_detector.detect_emotion(args.text)
            print(f"LLM detected emotion: {llm_emotion}")

        if args.method == "pattern" or args.method == "all":
            pattern_emotion = pattern_detector.detect_emotion(args.text)
            print(f"Pattern matching detected emotion: {pattern_emotion}")

    except Exception as e:
        logger.error(f"Error testing emotion detection: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
