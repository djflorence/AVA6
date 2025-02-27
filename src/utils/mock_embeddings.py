"""
Mock embeddings for testing without OpenAI API.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MockEmbeddings:
    """
    Mock embeddings class that generates random embeddings for testing.
    
    This class mimics the interface of OpenAIEmbeddings but doesn't require
    API calls, making it suitable for testing when API quota is exceeded.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        """
        Initialize the mock embeddings.
        
        Args:
            model: Model name (ignored, just for compatibility)
            dimensions: Dimensions of the embeddings to generate
        """
        self.dimensions = dimensions
        self.model = model
        logger.info(f"Initialized MockEmbeddings with {dimensions} dimensions")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of mock embeddings (random vectors)
        """
        logger.debug(f"Generating mock embeddings for {len(texts)} documents")
        
        # Generate deterministic embeddings based on text content
        # This ensures the same text always gets the same embedding
        embeddings = []
        for text in texts:
            # Use hash of text to seed the random generator for consistency
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.normal(0, 1, self.dimensions).tolist()
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
            
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """
        Generate a mock embedding for a query.
        
        Args:
            text: Text to embed
            
        Returns:
            Mock embedding (random vector)
        """
        logger.debug(f"Generating mock embedding for query: {text[:50]}...")
        return self.embed_documents([text])[0] 