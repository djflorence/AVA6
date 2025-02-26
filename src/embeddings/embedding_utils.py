"""
Embedding utilities for the AI Assistant.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Provider for text embeddings.
    
    This class provides a unified interface for different embedding models.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the embedding provider.
        
        Args:
            provider: Embedding provider ("openai" or "sentence_transformers")
            model_name: Name of the model to use
            **kwargs: Additional arguments for the embedding model
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.model = None
        
        if self.provider == "openai":
            # Use OpenAI embeddings
            model_name = model_name or "text-embedding-3-large"
            self.model = OpenAIEmbeddings(model=model_name, **kwargs)
            logger.info(f"Initialized OpenAI embeddings with model: {model_name}")
        
        elif self.provider == "sentence_transformers":
            # Use sentence-transformers
            model_name = model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized sentence-transformers with model: {model_name}")
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        try:
            if self.provider == "openai":
                # OpenAI embeddings
                embeddings = self.model.embed_documents(texts)
                return embeddings
            
            elif self.provider == "sentence_transformers":
                # sentence-transformers embeddings
                embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
                return embeddings
            
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text:
            return []
        
        try:
            if self.provider == "openai":
                # OpenAI embeddings
                embedding = self.model.embed_query(text)
                return embedding
            
            elif self.provider == "sentence_transformers":
                # sentence-transformers embeddings
                embedding = self.model.encode(text, convert_to_tensor=False).tolist()
                return embedding
            
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    if not vec1 or not vec2:
        return 0.0
    
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")
    
    # Convert to numpy arrays
    np_vec1 = np.array(vec1)
    np_vec2 = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(np_vec1, np_vec2)
    norm_vec1 = np.linalg.norm(np_vec1)
    norm_vec2 = np.linalg.norm(np_vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)


def find_most_similar(
    query_embedding: List[float],
    embeddings: List[List[float]],
    texts: List[str],
    top_k: int = 5,
) -> List[Dict[str, Union[str, float]]]:
    """
    Find the most similar texts to a query.
    
    Args:
        query_embedding: Embedding of the query
        embeddings: List of embeddings to compare against
        texts: List of texts corresponding to the embeddings
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with text and similarity score
    """
    if not query_embedding or not embeddings or not texts:
        return []
    
    if len(embeddings) != len(texts):
        raise ValueError(f"Number of embeddings ({len(embeddings)}) doesn't match number of texts ({len(texts)})")
    
    # Calculate similarities
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create result list
    results = [
        {
            "text": texts[idx],
            "similarity": similarities[idx],
        }
        for idx in sorted_indices
    ]
    
    return results 