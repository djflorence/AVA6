"""
Tests for the chat assistant.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.assistant.chat_assistant import ChatAssistant


class TestChatAssistant(unittest.TestCase):
    """Tests for the ChatAssistant class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a minimal config for testing
        self.config = {
            "DEFAULT_LLM_MODEL": "gpt-3.5-turbo",
            "MEMORY_TYPE": "simple",  # Use simple memory to avoid ChromaDB dependency
            "ENABLE_EMOTIONS": "false",
            "ENABLE_CALCULATOR": "true",
            "ENABLE_WEATHER": "false",
            "TEMPERATURE": 0.0,
            "MAX_TOKENS": 100,
        }
    
    @patch("src.assistant.chat_assistant.ChatOpenAI")
    @patch("src.assistant.chat_assistant.ToolManager")
    def test_initialization(self, mock_tool_manager, mock_chat_openai):
        """Test that the assistant initializes correctly."""
        # Mock the LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the tool manager
        mock_tools = MagicMock()
        mock_tool_manager.return_value = mock_tools
        mock_tools.get_tools.return_value = []
        
        # Initialize the assistant
        assistant = ChatAssistant(self.config)
        
        # Check that the LLM was initialized
        mock_chat_openai.assert_called_once()
        
        # Check that the tool manager was initialized
        mock_tool_manager.assert_called_once()
        
        # Check that the assistant has the expected attributes
        self.assertEqual(assistant.config, self.config)
        self.assertIsNotNone(assistant.conversation_id)
        self.assertIsNotNone(assistant.llm)
        self.assertIsNotNone(assistant.memory)
        self.assertIsNone(assistant.emotion_detector)
        self.assertIsNotNone(assistant.tool_manager)
        self.assertIsNotNone(assistant.chain)
    
    @patch("src.assistant.chat_assistant.ChatOpenAI")
    @patch("src.assistant.chat_assistant.ToolManager")
    def test_chat(self, mock_tool_manager, mock_chat_openai):
        """Test the chat method."""
        # Mock the LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "This is a test response."
        mock_llm.invoke.return_value = mock_response
        
        # Mock the tool manager
        mock_tools = MagicMock()
        mock_tool_manager.return_value = mock_tools
        mock_tools.get_tools.return_value = []
        mock_tools.should_use_tool.return_value = None
        
        # Initialize the assistant
        assistant = ChatAssistant(self.config)
        
        # Test chat method
        response = assistant.chat("Hello, how are you?")
        
        # Check that the LLM was called
        mock_llm.invoke.assert_called_once()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a test response.")
        
        # Check that the memory was updated
        self.assertEqual(len(assistant.memory.chat_memory.messages), 2)
        self.assertEqual(assistant.memory.chat_memory.messages[0].content, "Hello, how are you?")
        self.assertEqual(assistant.memory.chat_memory.messages[1].content, "This is a test response.")
    
    @patch("src.assistant.chat_assistant.ChatOpenAI")
    @patch("src.assistant.chat_assistant.ToolManager")
    def test_chat_with_tool(self, mock_tool_manager, mock_chat_openai):
        """Test the chat method with tool usage."""
        # Mock the LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "The answer is 42."
        mock_llm.invoke.return_value = mock_response
        
        # Mock the tool manager
        mock_tools = MagicMock()
        mock_tool_manager.return_value = mock_tools
        mock_tools.get_tools.return_value = ["calculator"]
        mock_tools.should_use_tool.return_value = "calculator"
        mock_tools.use_tool.return_value = "The result of 6 * 7 is 42"
        
        # Initialize the assistant
        assistant = ChatAssistant(self.config)
        
        # Test chat method with a calculation query
        response = assistant.chat("What is 6 * 7?")
        
        # Check that the tool manager was called
        mock_tools.should_use_tool.assert_called_once_with("What is 6 * 7?")
        mock_tools.use_tool.assert_called_once_with("calculator", "What is 6 * 7?")
        
        # Check that the LLM was called with the enhanced input
        expected_input = "What is 6 * 7?\n\nTool information: The result of 6 * 7 is 42"
        self.assertIn(expected_input, str(mock_llm.invoke.call_args))
        
        # Check that the response is correct
        self.assertEqual(response, "The answer is 42.")


if __name__ == "__main__":
    unittest.main() 