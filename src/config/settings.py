"""
Settings management for the AI Assistant.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Settings for the AI Assistant.
    
    This class handles loading and managing configuration from environment
    variables and configuration files.
    """
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4-turbo")
    ANTHROPIC_MODEL: str = Field(default="claude-3-opus-20240229")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large")
    
    # Memory Configuration
    MEMORY_TYPE: str = Field(default="chroma")
    CHROMA_DB_DIRECTORY: str = Field(default="./src/data/chroma")
    MEMORY_WINDOW_SIZE: int = Field(default=10)
    
    # Emotion Settings
    ENABLE_EMOTIONS: bool = Field(default=True)
    EMOTION_SENSITIVITY: float = Field(default=0.7)
    
    # Tool Configuration
    ENABLE_WEB_SEARCH: bool = Field(default=False)
    ENABLE_CALCULATOR: bool = Field(default=True)
    ENABLE_WEATHER: bool = Field(default=False)
    WEATHER_API_KEY: Optional[str] = Field(default=None)
    
    # Application Settings
    LOG_LEVEL: str = Field(default="INFO")
    ENABLE_TRACING: bool = Field(default=False)
    MAX_TOKENS: int = Field(default=4000)
    TEMPERATURE: float = Field(default=0.7)
    
    # Web UI Settings
    WEB_UI_PORT: int = Field(default=8000)
    WEB_UI_HOST: str = Field(default="127.0.0.1")
    ENABLE_AUTH: bool = Field(default=False)
    
    def __init__(self, **data: Any):
        """
        Initialize settings from environment variables and provided data.
        
        Args:
            **data: Override settings
        """
        # First load from environment variables
        env_settings = {}
        for field_name in self.__annotations__:
            env_value = os.environ.get(field_name)
            if env_value is not None:
                # Convert environment variables to appropriate types
                field_type = self.__annotations__[field_name]
                
                # Handle Optional types
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                    # Extract the non-None type from Optional[T]
                    field_type = next((t for t in field_type.__args__ if t is not type(None)), str)
                
                if field_type is bool:
                    env_settings[field_name] = env_value.lower() in ("true", "1", "yes")
                elif field_type is int:
                    env_settings[field_name] = int(env_value)
                elif field_type is float:
                    env_settings[field_name] = float(env_value)
                else:
                    env_settings[field_name] = env_value
        
        # Then override with provided data
        env_settings.update(data)
        
        super().__init__(**env_settings)
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load settings from a profile configuration file.
        
        Args:
            profile_name: Name of the profile to load
            
        Returns:
            Dict containing profile settings
            
        Raises:
            FileNotFoundError: If profile file doesn't exist
        """
        config_dir = Path("src/config/profiles")
        profile_path = config_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_path}")
        
        with open(profile_path, "r") as f:
            return json.load(f)
    
    def update(self, settings: Dict[str, Any]) -> None:
        """
        Update settings with new values.
        
        Args:
            settings: New settings to apply
        """
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dict containing all settings
        """
        return super().dict() 