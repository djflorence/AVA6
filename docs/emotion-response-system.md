# Emotion-Based Response System

This document provides an overview of AVA's emotion-based response system, which enables the assistant to detect user emotions and respond appropriately.

## Overview

The emotion-based response system consists of several components:

1. **Emotion Detector**: Detects emotions from user input using pattern matching, LLM-based analysis, or Hugging Face models.
2. **Response Templates**: Provides templates for responses based on detected emotions.
3. **Emotion Handler**: Integrates emotion detection with response generation.
4. **Chat Assistant Integration**: Incorporates emotional intelligence into the main assistant.

## Components

### Emotion Detector

The `EmotionDetector` class is responsible for detecting emotions from user input. It supports three detection methods:

- **Pattern Matching**: Uses keyword-based detection for fast, lightweight emotion detection.
- **LLM-based Analysis**: Uses OpenAI's models for more accurate but slower emotion detection.
- **Hugging Face Models**: Uses pre-trained emotion classification models for local processing.

The detector can identify six primary emotions:
- Joy
- Sadness
- Anger
- Fear
- Surprise
- Neutral

### Response Templates

The `response_templates.py` module provides templates for generating responses based on detected emotions. It includes:

- **Response Templates**: Pre-defined response templates for each emotion.
- **System Prompts**: System prompts that guide the assistant's response based on the user's emotional state.

### Emotion Handler

The `EmotionHandler` class integrates emotion detection with response generation. It:

- Processes user input to detect emotions
- Retrieves appropriate response templates and system prompts
- Provides guidance for generating emotionally intelligent responses

### Chat Assistant Integration

The main `ChatAssistant` class integrates the emotion-based response system by:

- Initializing the emotion handler
- Processing user input to detect emotions
- Using emotional context to guide response generation
- Storing emotional data in the memory system

## Usage

### Basic Usage

```python
from src.emotions.emotion_handler import EmotionHandler

# Initialize the emotion handler
emotion_handler = EmotionHandler(
    use_llm=False,  # Use LLM-based detection
    use_hf=False,   # Use Hugging Face models
    sensitivity=0.7  # Sensitivity threshold
)

# Process user input
emotion, guidance = emotion_handler.process_input("I'm feeling really happy today!")

# Use the detected emotion and guidance
print(f"Detected emotion: {emotion}")
if guidance:
    print(f"System prompt: {guidance['system_prompt']}")
    print(f"Response template: {guidance['response_template']}")
```

### Configuration Options

The emotion-based response system can be configured with several options:

- **Detection Method**: Choose between pattern matching, LLM-based analysis, or Hugging Face models.
- **Sensitivity**: Adjust the sensitivity threshold for emotion detection (0.0-1.0).
- **Enable/Disable**: Toggle emotion-based responses on or off.

## Testing

Two scripts are provided for testing the emotion-based response system:

1. **test_emotion_detector.py**: Tests the emotion detection system independently.
2. **test_emotion_responses.py**: Tests the complete emotion-based response system.

### Demo Script

A demo script is also provided to interactively test the emotion-based response system:

```bash
python demo_emotion_responses.py [--detector METHOD] [--sensitivity THRESHOLD]
```

Options:
- `--detector`: Emotion detection method (`pattern`, `llm`, or `hf`)
- `--sensitivity`: Sensitivity threshold (0.0-1.0)
- `--debug`: Enable debug mode

## Extending the System

### Adding New Emotions

To add new emotions to the system:

1. Update the `EMOTIONS` list in `emotion_detector.py`
2. Add patterns for the new emotion in `_init_patterns()`
3. Add response templates in `response_templates.py`

### Customizing Response Templates

To customize response templates:

1. Edit the `EMOTION_RESPONSE_TEMPLATES` dictionary in `response_templates.py`
2. Edit the `EMOTION_SYSTEM_PROMPTS` dictionary for system prompts

## Best Practices

- **Sensitivity Tuning**: Adjust the sensitivity threshold based on your use case. Higher values make the system more sensitive to emotional cues.
- **Method Selection**: Use pattern matching for speed, LLM-based analysis for accuracy, or Hugging Face models for privacy.
- **Template Customization**: Customize response templates to match your assistant's personality and tone.
- **Testing**: Regularly test the system with diverse inputs to ensure accurate emotion detection and appropriate responses. 