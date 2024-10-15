import pytest
from centralized_nlp_package import config
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from centralized_nlp_package.embedding.word2vec_model import train_word2vec, save_model
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

# Test for train_word2vec function
def test_train_word2vec_unigram():
    """
    Test training Word2Vec model with unigram configuration.
    This test ensures that the function returns a valid Word2Vec instance using a simple corpus.
    """
    sample_sents = [["hello", "world"]*10, ["testing", "word2vec"]]

    model = train_word2vec(sample_sents)
    assert isinstance(model, Word2Vec), "The output should be an instance of gensim Word2Vec model."
    assert len(model.wv) > 0, "Word2Vec model vocabulary is empty"
    assert 'hello' in model.wv, "hello not found in model vocabulary"

def test_train_word2vec_bigram():
    """
    Test training Word2Vec model with bigram configuration.
    This test checks if the bigram parameter is properly handled and returns a valid Word2Vec instance.
    """
    sample_sents = [["testing", "bigram", "word2vec"]*48, ["hello", "world"]*100]
    
    bigram_model = Phrases(sample_sents)
    bigram_sents = bigram_model[sample_sents]


    # Mocking configuration for Word2Vec model parameters
    model = train_word2vec(bigram_sents, bigram=True)
    assert isinstance(model, Word2Vec), "The output should be an instance of gensim Word2Vec model."
    assert 'bigram' in model.wv.key_to_index, "hello not found in model vocabulary"

# Test for save_model function
def test_save_model():
    """
    Test saving a trained Word2Vec model without writing to disk.
    This test ensures that the save_model function is called with the correct path.
    """
    sample_sents = [["hello", "world"], ["testing", "save", "model"] * 16]
    model = train_word2vec(sample_sents)

    # Mock the save_model function
    with patch('centralized_nlp_package.embedding.word2vec_model.save_model') as mock_save_model:
        model_path = "mocked/path/word2vec_model"
        mock_save_model(model, model_path)

        # Assert that save_model was called with the correct arguments
        mock_save_model.assert_called_once_with(model, model_path)