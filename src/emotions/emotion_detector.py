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
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
        "neutral",
    ]

    # Define emotion keywords for basic pattern matching - expanded for better detection
    EMOTION_KEYWORDS = {
        "joy": [
            "happy", "joy", "delighted", "pleased", "glad", "excited", "content",
            "thrilled", "ecstatic", "cheerful", "elated", "jubilant", "overjoyed",
            "blissful", "wonderful", "fantastic", "great", "awesome", "amazing",
            "love", "enjoy", "celebrate", "congratulations", "proud", "achievement"
        ],
        "sadness": [
            "sad", "unhappy", "depressed", "down", "miserable", "upset", "grief",
            "heartbroken", "gloomy", "melancholy", "sorrowful", "disappointed",
            "disheartened", "devastated", "hurt", "painful", "regret", "sorry",
            "lonely", "hopeless", "despair", "crying", "tears", "miss", "lost"
        ],
        "anger": [
            "angry", "mad", "furious", "outraged", "annoyed", "irritated", "frustrated",
            "enraged", "hostile", "bitter", "resentful", "indignant", "irate",
            "livid", "infuriated", "exasperated", "disgusted", "hate", "dislike",
            "fed up", "tired of", "sick of", "upset", "offended", "insulted"
        ],
        "fear": [
            "afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous",
            "panicked", "alarmed", "horrified", "dread", "uneasy", "apprehensive",
            "concerned", "stressed", "overwhelmed", "intimidated", "threatened",
            "insecure", "vulnerable", "helpless", "uncertain", "doubt", "hesitant"
        ],
        "surprise": [
            "surprised", "shocked", "astonished", "amazed", "stunned", "unexpected",
            "startled", "dumbfounded", "bewildered", "flabbergasted", "speechless",
            "taken aback", "astounded", "awestruck", "wonder", "disbelief", "incredible",
            "unbelievable", "wow", "whoa", "oh my", "really", "seriously"
        ],
        "disgust": [
            "disgusted", "revolted", "repulsed", "appalled", "horrified", "nauseated",
            "sickened", "grossed out", "aversion", "distaste", "dislike", "contempt",
            "loathing", "abhorrence", "hate", "despise", "detest", "offensive", "foul"
        ],
        "trust": [
            "trust", "believe", "faith", "confident", "assured", "reliance", "dependable",
            "reliable", "honest", "loyal", "devoted", "dedicated", "committed", "secure",
            "safe", "certain", "convinced", "hopeful", "optimistic", "positive"
        ],
        "anticipation": [
            "anticipate", "expect", "looking forward", "hopeful", "excited", "eager",
            "await", "prepare", "ready", "enthusiastic", "keen", "interested", "curious",
            "intrigued", "fascinated", "motivated", "inspired", "determined", "ambitious"
        ],
    }

    # Mapping from HuggingFace model emotions to our primary emotions
    HF_EMOTION_MAPPING = {
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "fear": "fear",
        "love": "trust",  # Map love to trust
        "surprise": "surprise",
        "neutral": "neutral",
    }

    def __init__(
        self, sensitivity: float = 0.7, use_llm: bool = True, use_hf: bool = True
    ):
        """
        Initialize the emotion detector.

        Args:
            sensitivity: Sensitivity level (0.0 to 1.0)
            use_llm: Whether to use a language model for detection
            use_hf: Whether to use Hugging Face model for detection
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))  # Clamp between 0 and 1
        self.use_llm = use_llm
        self.use_hf = use_hf
        self.llm = None
        self.hf_model = None
        self.hf_tokenizer = None

        # Initialize LLM if enabled
        if use_llm:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=100,
                )
                logger.info("Initialized LLM for emotion detection")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM for emotion detection: {str(e)}"
                )
                self.use_llm = False

        # Initialize Hugging Face model if enabled
        if use_hf:
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                # Load model and tokenizer
                model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )

                # Get the labels
                self.hf_labels = self.hf_model.config.id2label

                logger.info("Initialized Hugging Face model for emotion detection")
            except Exception as e:
                logger.warning(f"Failed to initialize Hugging Face model: {str(e)}")
                self.use_hf = False

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

        # Use multiple detection methods and combine results for more accuracy
        detected_emotions = []
        
        # Use Hugging Face model if available
        if self.use_hf and self.hf_model and self.hf_tokenizer:
            hf_emotion = self._detect_emotion_hf(text)
            if hf_emotion:
                detected_emotions.append((hf_emotion, 0.7))  # Weight of 0.7
        
        # Use LLM-based detection if available
        if self.use_llm and self.llm:
            llm_emotion = self._detect_emotion_llm(text)
            if llm_emotion:
                detected_emotions.append((llm_emotion, 0.8))  # Weight of 0.8
        
        # Use pattern-based detection as fallback
        pattern_emotion = self._detect_emotion_pattern(text)
        if pattern_emotion:
            detected_emotions.append((pattern_emotion, 0.5))  # Weight of 0.5
            
        # If no emotions detected, return neutral
        if not detected_emotions:
            return "neutral"
            
        # If only one emotion detected, return it
        if len(detected_emotions) == 1:
            return detected_emotions[0][0]
            
        # Count weighted votes for each emotion
        emotion_votes = {}
        for emotion, weight in detected_emotions:
            if emotion in emotion_votes:
                emotion_votes[emotion] += weight
            else:
                emotion_votes[emotion] = weight
                
        # Return the emotion with the highest weighted vote
        return max(emotion_votes.items(), key=lambda x: x[1])[0]

    def _detect_emotion_pattern(self, text: str) -> Optional[str]:
        """
        Detect emotion using pattern matching.

        Args:
            text: Text to analyze

        Returns:
            Detected emotion or None if no emotion detected
        """
        text = text.lower()
        emotion_scores: Dict[str, int] = {
            emotion: 0 for emotion in self.PRIMARY_EMOTIONS
        }

        # Count emotion keywords
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
                emotion_scores[emotion] += count

        # Apply sensitivity threshold
        threshold = 1.0 - self.sensitivity
        max_score = max(emotion_scores.values())

        if max_score == 0:
            return "neutral"

        # Get the emotion with the highest score
        max_emotions = [
            emotion
            for emotion, score in emotion_scores.items()
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
            # Improved prompt with clear instructions and examples
            prompt = f"""Analyze the following text and determine the primary emotion expressed. 
Choose exactly one emotion from this list: {', '.join(self.PRIMARY_EMOTIONS)}.

Here are some examples:
- "I'm so happy today!" -> joy
- "I feel really sad." -> sadness
- "I'm angry about what happened." -> anger
- "I'm worried about my presentation." -> fear
- "Wow, I didn't expect that!" -> surprise
- "That's disgusting." -> disgust
- "I trust you to handle this." -> trust
- "I'm looking forward to the event." -> anticipation
- "What's the weather like today?" -> neutral

Text: "{text}"

Primary emotion (just one word):"""

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

    def _detect_emotion_hf(self, text: str) -> str:
        """
        Detect emotion using the Hugging Face model.

        Args:
            text: Text to analyze

        Returns:
            Detected emotion
        """
        try:
            import torch

            # Prepare the input
            inputs = self.hf_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            # Get the prediction
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                logits = outputs.logits

            # Get the predicted class
            predicted_class = torch.argmax(logits, dim=1).item()

            # Get the emotion label
            hf_emotion = self.hf_labels[predicted_class]

            # Map to our primary emotions
            if hf_emotion in self.HF_EMOTION_MAPPING:
                return self.HF_EMOTION_MAPPING[hf_emotion]

            # Default to neutral if no mapping
            logger.warning(
                f"Hugging Face model returned unmapped emotion: {hf_emotion}"
            )
            return "neutral"

        except Exception as e:
            logger.error(f"Error in Hugging Face emotion detection: {str(e)}")

            # Try LLM if available
            if self.use_llm and self.llm:
                return self._detect_emotion_llm(text)

            # Fallback to pattern-based detection
            return self._detect_emotion_pattern(text)

    def analyze_emotional_change(
        self, previous_emotions: List[str], current_emotion: str
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

    def get_emotion_intensity(self, text: str, emotion: str) -> float:
        """
        Get the intensity of a specific emotion in the text.

        Args:
            text: Text to analyze
            emotion: Emotion to check intensity for

        Returns:
            Intensity score (0.0 to 1.0)
        """
        if not text or not emotion or emotion not in self.PRIMARY_EMOTIONS:
            return 0.0

        # Use Hugging Face model if available
        if self.use_hf and self.hf_model and self.hf_tokenizer:
            try:
                import torch

                # Prepare the input
                inputs = self.hf_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )

                # Get the prediction
                with torch.no_grad():
                    outputs = self.hf_model(**inputs)
                    logits = outputs.logits

                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

                # Find the index of the target emotion
                for idx, label in self.hf_labels.items():
                    if (
                        label in self.HF_EMOTION_MAPPING
                        and self.HF_EMOTION_MAPPING[label] == emotion
                    ):
                        return probs[idx].item()

                return 0.0

            except Exception as e:
                logger.error(
                    f"Error getting emotion intensity from Hugging Face model: {str(e)}"
                )

        # Fallback to pattern-based intensity
        text = text.lower()
        count = 0

        # Count occurrences of emotion keywords
        if emotion in self.EMOTION_KEYWORDS:
            for keyword in self.EMOTION_KEYWORDS[emotion]:
                count += len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))

        # Normalize to 0-1 range
        return min(
            1.0, count / 5.0
        )  # Cap at 1.0, assume 5+ occurrences is maximum intensity

    def analyze_conversation(self, conversation: str) -> str:
        """
        Analyze the emotional context of a conversation.
        
        Args:
            conversation: Conversation text to analyze
            
        Returns:
            Emotional context analysis
        """
        logger.debug(f"Analyzing conversation for emotional context")
        
        # Detect emotion in conversation
        emotion = self.detect_emotion(conversation)
        
        # Get intensity of the emotion
        intensity = self.get_emotion_intensity(conversation, emotion)
        
        # Generate emotional context based on detected emotion and intensity
        if emotion == "joy":
            if intensity > 0.7:
                return "The user seems very happy and enthusiastic. Respond with matching positive energy."
            else:
                return "The user seems content. Maintain a positive and supportive tone."
        elif emotion == "sadness":
            if intensity > 0.7:
                return "The user seems deeply sad or disappointed. Show empathy and understanding in your response."
            else:
                return "The user seems somewhat down. Be supportive and gentle in your response."
        elif emotion == "anger":
            if intensity > 0.7:
                return "The user seems very frustrated or angry. Acknowledge their feelings and respond calmly and helpfully."
            else:
                return "The user seems mildly annoyed. Be patient and helpful in your response."
        elif emotion == "fear":
            if intensity > 0.7:
                return "The user seems very anxious or worried. Provide reassurance and support in your response."
            else:
                return "The user seems concerned. Offer helpful information and gentle reassurance."
        elif emotion == "surprise":
            if intensity > 0.7:
                return "The user seems very surprised or shocked. Acknowledge the unexpected nature of the situation."
            else:
                return "The user seems somewhat surprised. Acknowledge this in your response."
        elif emotion == "disgust":
            return "The user seems disgusted or repulsed. Be understanding but neutral in your response."
        elif emotion == "trust":
            return "The user seems to be expressing trust. Maintain this trust with a reliable and honest response."
        elif emotion == "anticipation":
            return "The user seems to be looking forward to something. Match their anticipation with an enthusiastic response."
        else:
            return "The user's emotional state seems neutral. Maintain a balanced and informative tone."
