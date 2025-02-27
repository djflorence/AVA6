#!/usr/bin/env python3
"""
Test script for evaluating the emotion-based response system.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from dotenv import load_dotenv

from src.emotions.emotion_detector import EmotionDetector
from src.emotions.response_templates import get_response_template, get_system_prompt
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the emotion-based response system")
    parser.add_argument(
        "--test-file",
        type=str,
        default="emotion_test_cases.json",
        help="Path to the JSON file containing test cases",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["hf", "llm", "pattern"],
        default="pattern",
        help="Emotion detection method to use",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=None,
        help="Index of the response template to use (for consistent testing)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def generate_response(
    input_text: str, 
    detector: EmotionDetector, 
    template_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a response based on detected emotion.
    
    Args:
        input_text: User input text
        detector: Emotion detector instance
        template_index: Optional index for response template
        
    Returns:
        Dictionary with detected emotion and response
    """
    # Detect emotion
    emotion = detector.detect_emotion(input_text)
    
    # Get system prompt and response template
    system_prompt = get_system_prompt(emotion)
    response = get_response_template(emotion, template_index)
    
    return {
        "input": input_text,
        "detected_emotion": emotion,
        "system_prompt": system_prompt,
        "response": response
    }


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

        # Initialize emotion detector
        if args.detector == "hf":
            detector = EmotionDetector(use_llm=False, use_hf=True)
            logger.info("Using Hugging Face model for emotion detection")
        elif args.detector == "llm":
            detector = EmotionDetector(use_llm=True, use_hf=False)
            logger.info("Using LLM for emotion detection")
        else:
            detector = EmotionDetector(use_llm=False, use_hf=False)
            logger.info("Using pattern matching for emotion detection")

        # Run tests
        results = []
        for i, case in enumerate(test_cases, 1):
            input_text = case["input"]
            expected_emotion = case["expected_emotion"]
            
            logger.info(f"\nTest case {i}/{len(test_cases)}: '{input_text}'")
            logger.info(f"Expected emotion: {expected_emotion}")
            
            # Generate response
            start_time = time.time()
            result = generate_response(input_text, detector, args.template_index)
            end_time = time.time()
            
            # Add timing information
            result["processing_time"] = end_time - start_time
            result["expected_emotion"] = expected_emotion
            result["emotion_match"] = result["detected_emotion"] == expected_emotion
            
            # Log results
            logger.info(f"Detected emotion: {result['detected_emotion']}")
            logger.info(f"Emotion match: {'✅' if result['emotion_match'] else '❌'}")
            logger.info(f"Response: {result['response']}")
            logger.info(f"Processing time: {result['processing_time']:.4f} seconds")
            
            results.append(result)
        
        # Calculate statistics
        correct_emotions = sum(1 for r in results if r["emotion_match"])
        accuracy = (correct_emotions / len(results)) * 100
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        
        logger.info(f"\nTest Results Summary:")
        logger.info(f"Emotion Detection Accuracy: {accuracy:.2f}% ({correct_emotions}/{len(results)})")
        logger.info(f"Average Processing Time: {avg_time:.4f} seconds")
        
        # Save results to file
        output_file = f"emotion_response_test_results_{args.detector}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error testing emotion responses: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 