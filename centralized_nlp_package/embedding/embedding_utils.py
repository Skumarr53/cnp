# centralized_nlp_package/embedding/embedding_utils.py
from typing import List, Optional
import numpy as np
from gensim.models import Word2Vec

def average_token_embeddings(tokens: List[str], model: Word2Vec) -> Optional[np.ndarray]:
    """
    Generates an embedding for a list of tokens by averaging their vectors.

    Args:
        tokens (List[str]): List of tokens (e.g., unigrams or bigrams).
        model (Word2Vec): A trained Word2Vec model.

    Returns:
        Optional[np.ndarray]: The averaged embedding vector, or None if none of the tokens are in the model.

    Example:
        >>> from centralized_nlp_package.embedding import average_token_embeddings
        >>> from gensim.models import Word2Vec
        >>> model = Word2Vec([['king', 'queen', 'man']], vector_size=100, min_count=1, epochs=10)
        >>> tokens = ['king', 'queen', 'unknown_token']
        >>> embedding = average_token_embeddings(tokens, model)
        >>> print(embedding.shape)
        (100,)
    """
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not valid_vectors:
        return None
    return np.mean(np.stack(valid_vectors), axis=0)


