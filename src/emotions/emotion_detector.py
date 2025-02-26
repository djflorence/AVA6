"""
Emotion detection for the AI Assistant.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Emotion detector for analyzing user messages.
    
    This class uses a language model to detect emotions in text.
    It can identify primary emotions and their intensity.
    """
    
    # Define primary emotions
    PRIMARY_EMOTIONS = [
        "joy", "sadness", "anger", "fear", "surprise", 
        "disgust", "trust", "anticipation", "neutral"
    ]
    
    # Define emotion keywords for basic pattern matching
    EMOTION_KEYWORDS = {
        "joy": ["happy", "joy", "delighted", "pleased", "glad", "excited", "content"],
        "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "grief"],
        "anger": ["angry", "mad", "furious", "outraged", "annoyed", "irritated", "frustrated"],
        "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
        "surprise": ["surprised", "shocked", "astonished", "amazed", "stunned", "unexpected"],
        "disgust": ["disgusted", "revolted", "repulsed", "appalled", "horrified"],
        "trust": ["trust", "believe", "faith", "confident", "assured", "reliance"],
        "anticipation": ["anticipate", "expect", "looking forward", "hopeful", "excited"],
    }
    
    def __init__(self, sensitivity: float = 0.7, use_llm: bool = True):
        """
        Initialize the emotion detector.
        
        Args:
            sensitivity: Sensitivity level (0.0 to 1.0)
            use_llm: Whether to use a language model for detection
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))  # Clamp between 0 and 1
        self.use_llm = use_llm
        self.llm = None
        
        if use_llm:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=100,
                )
                logger.info("Initialized LLM for emotion detection")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for emotion detection: {str(e)}")
                self.use_llm = False
        
        logger.info(f"Emotion detector initialized with sensitivity {sensitivity}")
    
    def detect_emotion(self, text: str) -> Optional[str]:
        """
        Detect the primary emotion in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion or None if no emotion detected
        """
        if not text:
            return None
        
        # Use LLM-based detection if available
        if self.use_llm and self.llm:
            return self._detect_emotion_llm(text)
        
        # Fallback to pattern-based detection
        return self._detect_emotion_pattern(text)
    
    def _detect_emotion_pattern(self, text: str) -> Optional[str]:
        """
        Detect emotion using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion or None if no emotion detected
        """
        text = text.lower()
        emotion_scores: Dict[str, int] = {emotion: 0 for emotion in self.PRIMARY_EMOTIONS}
        
        # Count emotion keywords
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                emotion_scores[emotion] += count
        
        # Apply sensitivity threshold
        threshold = 1.0 - self.sensitivity
        max_score = max(emotion_scores.values())
        
        if max_score == 0:
            return "neutral"
        
        # Get the emotion with the highest score
        max_emotions = [
            emotion for emotion, score in emotion_scores.items() 
            if score == max_score and score > threshold
        ]
        
        if not max_emotions:
            return "neutral"
        
        return max_emotions[0]
    
    def _detect_emotion_llm(self, text: str) -> str:
        """
        Detect emotion using a language model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion
        """
        try:
            prompt = f"""Analyze the following text and determine the primary emotion expressed. 
Choose only one emotion from this list: {', '.join(self.PRIMARY_EMOTIONS)}.
Respond with just the emotion name, nothing else.

Text: "{text}"

Emotion:"""
            
            response = self.llm.invoke(prompt)
            detected_emotion = response.content.strip().lower()
            
            # Validate that the response is one of our primary emotions
            if detected_emotion in self.PRIMARY_EMOTIONS:
                return detected_emotion
            
            # Try to map to a primary emotion if not exact match
            for emotion in self.PRIMARY_EMOTIONS:
                if emotion in detected_emotion:
                    return emotion
            
            # Default to neutral if no match
            logger.warning(f"LLM returned unrecognized emotion: {detected_emotion}")
            return "neutral"
            
        except Exception as e:
            logger.error(f"Error in LLM emotion detection: {str(e)}")
            # Fallback to pattern-based detection
            return self._detect_emotion_pattern(text)
    
    def analyze_emotional_change(
        self, 
        previous_emotions: List[str],
        current_emotion: str
    ) -> Dict[str, Any]:
        """
        Analyze how emotions have changed over time.
        
        Args:
            previous_emotions: List of previously detected emotions
            current_emotion: Currently detected emotion
            
        Returns:
            Analysis of emotional change
        """
        if not previous_emotions:
            return {"change": "initial", "trend": None, "current": current_emotion}
        
        # Check if emotion has changed
        if previous_emotions[-1] == current_emotion:
            change = "stable"
        else:
            change = "changed"
        
        # Analyze trend if we have enough data
        trend = None
        if len(previous_emotions) >= 3:
            # Check for consistent emotions
            if len(set(previous_emotions[-3:])) == 1:
                if current_emotion != previous_emotions[-1]:
                    trend = "breaking"
            
            # Check for oscillation
            elif len(set(previous_emotions[-3:])) == 2:
                if current_emotion in previous_emotions[-3:]:
                    trend = "oscillating"
            
            # Check for progression
            else:
                trend = "varied"
        
        return {
            "change": change,
            "trend": trend,
            "current": current_emotion,
            "previous": previous_emotions[-1] if previous_emotions else None,
        } 