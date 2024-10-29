# centralized_nlp_package/embedding/word2vec_model.py

from gensim.models import Word2Vec, Phrases
from typing import List, Dict, Any
from loguru import logger
from pathlib import Path



## TODO: Topic modelling 
def train_word2vec_model(
    sentences: List[List[str]],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 10,
    workers: int = 16,
    epochs: int = 15,
) -> Word2Vec:
    """
    Trains a Word2Vec model on the provided corpus.

    This function initializes and trains a Word2Vec model using either bigram or unigram configurations
    from the library's configuration. Additional Word2Vec parameters can be specified via keyword arguments.

    Args:
        sentences (List[List[str]]): Corpus of tokenized sentences.
        bigram (bool, optional): Whether to use bigram configurations. Defaults to False.
        **kwargs (Any): Additional parameters for Word2Vec training.

    Returns:
        Word2Vec: Trained Word2Vec model.

    Example:
        >>> from centralized_nlp_package.embedding import train_word2vec_model
        >>> sentences = [['hello', 'world'], ['machine', 'learning']]
        >>> model = train_word2vec_model(sentences, vector_size=100, window=5, min_count=1)
        >>> model.wv['hello']
        array([ 0.0123, -0.0456, ...,  0.0789], dtype=float32)
    """    
    logger.info("Starting Word2Vec model training.")
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs,)
    logger.info("Word2Vec model training completed.")
    return model

def save_word2vec_model(model: Word2Vec, path: str) -> None:
    """
    Saves the trained Word2Vec model to the specified file path.

    This function ensures that the directory for the specified path exists and then saves
    the Word2Vec model in Gensim's native format.

    Args:
        model (Word2Vec): Trained Word2Vec model.
        path (str): File path to save the model.

    Example:
        >>> from centralized_nlp_package.embedding import save_word2vec_model
        >>> from gensim.models import Word2Vec
        >>> from centralized_nlp_package.embedding.word2vec_model import save_word2vec_model
        >>> model = Word2Vec(sentences=[['hello', 'world']])
        >>> save_word2vec_model(model, 'models/word2vec.model')
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving Word2Vec model to {model_path}")
    model.save(str(model_path))
    logger.info("Word2Vec model saved successfully.")
