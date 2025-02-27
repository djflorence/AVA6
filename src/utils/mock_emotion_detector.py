"""
Mock implementation of the emotion detector for testing without OpenAI API.
"""

import logging
import random
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MockEmotionDetector:
    """
    Mock implementation of the emotion detector for testing.
    Generates predictable emotion analysis without using the OpenAI API.
    """
    
    def __init__(self, sensitivity: float = 0.7, llm: Any = None):
        """
        Initialize the mock emotion detector.
        
        Args:
            sensitivity: Sensitivity level for emotion detection (0.0 to 1.0)
            llm: Language model (not used in mock implementation)
        """
        self.sensitivity = sensitivity
        logger.info(f"Initialized mock emotion detector with sensitivity {sensitivity}")
        
        # Define emotion keywords for detection
        self.emotion_keywords = {
            "joy": ["happy", "glad", "excited", "wonderful", "great", "awesome", "amazing", "joy", "delighted", "pleased"],
            "sadness": ["sad", "unhappy", "depressed", "down", "upset", "miserable", "disappointed", "sorrow", "grief", "crying"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "outraged", "rage", "hate", "resent"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "nervous", "worried", "panic", "dread", "horror"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "unexpected", "wow", "incredible", "unbelievable", "startled"],
            "neutral": ["ok", "fine", "alright", "neutral", "normal", "average", "standard", "regular", "typical", "common"]
        }
    
    def detect_emotion(self, text: str) -> str:
        """
        Detect the primary emotion in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion
        """
        logger.debug(f"Detecting emotion in text: {text}")
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count emotion keywords in text
        emotion_counts = {}
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_counts[emotion] = count
        
        # Get emotion with highest count
        max_count = max(emotion_counts.values())
        
        # If no emotions detected or all have same count, return neutral
        if max_count == 0:
            logger.debug("No emotion keywords detected, returning neutral")
            return "neutral"
        
        # Get emotions with highest count
        max_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
        
        # If multiple emotions have same count, choose randomly
        if len(max_emotions) > 1:
            # Use hash of text to ensure consistent results for same text
            random.seed(hash(text))
            emotion = random.choice(max_emotions)
            logger.debug(f"Multiple emotions detected with same count, chose {emotion}")
            return emotion
        
        # Return emotion with highest count
        emotion = max_emotions[0]
        logger.debug(f"Detected emotion: {emotion}")
        return emotion
    
    def analyze_conversation(self, conversation: str) -> str:
        """
        Analyze the emotional context of a conversation.
        
        Args:
            conversation: Conversation text to analyze
            
        Returns:
            Emotional context analysis
        """
        logger.debug(f"Analyzing conversation: {conversation}")
        
        # Detect emotion in conversation
        emotion = self.detect_emotion(conversation)
        
        # Generate emotional context based on detected emotion
        if emotion == "joy":
            return "The user seems happy and enthusiastic. Respond with matching positive energy."
        elif emotion == "sadness":
            return "The user seems sad or disappointed. Show empathy and understanding in your response."
        elif emotion == "anger":
            return "The user seems frustrated or angry. Acknowledge their feelings and respond calmly."
        elif emotion == "fear":
            return "The user seems anxious or worried. Provide reassurance and support in your response."
        elif emotion == "surprise":
            return "The user seems surprised or shocked. Acknowledge the unexpected nature of the situation."
        else:
            return "The user's emotional state seems neutral. Maintain a balanced and informative tone."