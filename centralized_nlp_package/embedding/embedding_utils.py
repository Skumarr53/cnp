# centralized_nlp_package/embedding/embedding_utils.py

from typing import List, Optional
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger



def average_token_embeddings(tokens: List[str], model: Word2Vec) -> Optional[np.ndarray]:
    """
    Generates an embedding for the given list of tokens by averaging their vectors.

    Args:
        tokens (List[str]): List of tokens (unigrams or bigrams).
        model (Word2Vec): Trained Word2Vec model.

    Returns:
        Optional[np.ndarray]: Averaged embedding vector or None if no tokens are in the model.
    """
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not valid_vectors:
        return None
    return np.mean(np.stack(valid_vectors), axis=0)

