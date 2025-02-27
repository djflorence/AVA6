"""
Emotion handler module for integrating emotion detection with response generation.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from src.emotions.emotion_detector import EmotionDetector
from src.emotions.response_templates import get_response_template, get_system_prompt

logger = logging.getLogger(__name__)

class EmotionHandler:
    """
    Handles the integration of emotion detection and response generation.
    """
    
    def __init__(
        self, 
        use_llm: bool = False, 
        use_hf: bool = False,
        sensitivity: float = 0.7,
        enable_responses: bool = True
    ):
        """
        Initialize the emotion handler.
        
        Args:
            use_llm: Whether to use LLM for emotion detection
            use_hf: Whether to use Hugging Face for emotion detection
            sensitivity: Sensitivity threshold for emotion detection
            enable_responses: Whether to enable emotion-based responses
        """
        self.detector = EmotionDetector(
            use_llm=use_llm,
            use_hf=use_hf,
            sensitivity=sensitivity
        )
        self.enable_responses = enable_responses
        self.last_detected_emotion = "neutral"
        logger.info(f"Initialized EmotionHandler (LLM: {use_llm}, HF: {use_hf}, "
                   f"Sensitivity: {sensitivity}, Responses: {enable_responses})")
    
    def process_input(self, input_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process user input to detect emotion and prepare response guidance.
        
        Args:
            input_text: User input text
            
        Returns:
            Tuple of (detected_emotion, response_guidance)
        """
        # Detect emotion
        emotion = self.detector.detect_emotion(input_text)
        self.last_detected_emotion = emotion
        
        # Prepare response guidance if enabled
        response_guidance = None
        if self.enable_responses:
            system_prompt = get_system_prompt(emotion)
            response_template = get_response_template(emotion)
            
            response_guidance = {
                "emotion": emotion,
                "system_prompt": system_prompt,
                "response_template": response_template
            }
            
            logger.debug(f"Prepared response guidance for emotion: {emotion}")
        
        logger.info(f"Detected emotion: {emotion}")
        return emotion, response_guidance
    
    def get_last_emotion(self) -> str:
        """Get the last detected emotion."""
        return self.last_detected_emotion
    
    def toggle_responses(self, enable: bool) -> None:
        """Toggle emotion-based responses on/off."""
        self.enable_responses = enable
        logger.info(f"Emotion-based responses {'enabled' if enable else 'disabled'}")
    
    def update_sensitivity(self, sensitivity: float) -> None:
        """Update the sensitivity threshold for emotion detection."""
        self.detector.sensitivity = max(0.0, min(1.0, sensitivity))
        logger.info(f"Updated emotion detection sensitivity to {self.detector.sensitivity}") 