#!/usr/bin/env python3
"""
Test script for evaluating the emotion detection system independently.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from dotenv import load_dotenv

from src.emotions.emotion_detector import EmotionDetector
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the emotion detection system")
    parser.add_argument(
        "--test-file",
        type=str,
        default="emotion_test_cases.json",
        help="Path to the JSON file containing test cases",
    )
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

    # Check if test file exists
    test_file = args.test_file
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)

    try:
        # Load test cases
        with open(test_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        logger.info(f"Loaded {len(test_cases)} test cases from {test_file}")

        # Initialize emotion detectors based on method
        detectors = {}
        
        if args.method == "hf" or args.method == "all":
            detectors["hf"] = EmotionDetector(use_llm=False, use_hf=True)

        if args.method == "llm" or args.method == "all":
            detectors["llm"] = EmotionDetector(use_llm=True, use_hf=False)

        if args.method == "pattern" or args.method == "all":
            detectors["pattern"] = EmotionDetector(use_llm=False, use_hf=False)

        # Run tests for each detector
        for detector_name, detector in detectors.items():
            logger.info(f"\nTesting {detector_name.upper()} detector:")
            correct = 0
            
            for i, case in enumerate(test_cases, 1):
                input_text = case["input"]
                expected_emotion = case["expected_emotion"]
                
                detected_emotion = detector.detect_emotion(input_text)
                
                if detected_emotion == expected_emotion:
                    correct += 1
                    logger.info(f"✅ PASS [{i}/{len(test_cases)}]: '{input_text}' -> {detected_emotion}")
                else:
                    logger.warning(f"❌ FAIL [{i}/{len(test_cases)}]: '{input_text}' -> {detected_emotion} (Expected: {expected_emotion})")
            
            accuracy = (correct / len(test_cases)) * 100
            logger.info(f"\n{detector_name.upper()} Detector Accuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})")

    except Exception as e:
        logger.error(f"Error testing emotion detection: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 