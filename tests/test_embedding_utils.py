import pytest
import numpy as np
from gensim.models import Word2Vec
from centralized_nlp_package.embedding.embedding_utils import average_token_embeddings

@pytest.fixture
def sample_model():
    """
    Fixture to create and return a sample Word2Vec model trained on a small corpus.
    This model will be used for testing the average_token_embeddings function.
    """
    # Sample corpus
    corpus = [
        ["hello", "world"],
        ["test", "embedding"],
        ["word2vec", "model"],
        ["average", "token", "embedding"]
    ]
    # Train Word2Vec model
    model = Word2Vec(sentences=corpus, vector_size=50, min_count=1, epochs=10)
    return model

def test_average_token_embeddings_all_tokens_present(sample_model):
    """
    Test the average_token_embeddings function when all tokens are present in the model.
    It should return the correct averaged embedding vector.
    """
    tokens = ["hello", "world"]
    result = average_token_embeddings(tokens, sample_model)
    assert result is not None, "Expected a numpy array, got None."
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."
    assert result.shape == (50,), f"Expected embedding size of 50, got {result.shape[0]}."

def test_average_token_embeddings_some_tokens_present(sample_model):
    """
    Test the average_token_embeddings function when some tokens are present in the model.
    It should average only the embeddings of the present tokens.
    """
    tokens = ["hello", "unknown_token"]
    result = average_token_embeddings(tokens, sample_model)
    assert result is not None, "Expected a numpy array, got None."
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."
    assert result.shape == (50,), f"Expected embedding size of 50, got {result.shape[0]}."
    # Verify that the result matches the embedding of 'hello'
    np.testing.assert_array_almost_equal(result, sample_model.wv["hello"], decimal=5,
                                         err_msg="Averaged embedding does not match 'hello' embedding.")

def test_average_token_embeddings_no_tokens_present(sample_model):
    """
    Test the average_token_embeddings function when none of the tokens are present in the model.
    It should return None.
    """
    tokens = ["unknown1", "unknown2"]
    result = average_token_embeddings(tokens, sample_model)
    assert result is None, "Expected None when no tokens are present in the model."

def test_average_token_embeddings_empty_token_list(sample_model):
    """
    Test the average_token_embeddings function with an empty token list.
    It should return None.
    """
    tokens = []
    result = average_token_embeddings(tokens, sample_model)
    assert result is None, "Expected None for empty token list."

def test_average_token_embeddings_duplicate_tokens(sample_model):
    """
    Test the average_token_embeddings function with duplicate tokens.
    It should correctly average the embeddings, accounting for duplicates.
    """
    tokens = ["hello", "hello", "world"]
    result = average_token_embeddings(tokens, sample_model)
    assert result is not None, "Expected a numpy array, got None."
    expected_average = np.mean(
        np.stack((sample_model.wv["hello"], sample_model.wv["hello"], sample_model.wv["world"])),
        axis=0
    )
    np.testing.assert_array_almost_equal(result, expected_average, decimal=5,
                                         err_msg="Averaged embedding does not match expected average.")

def test_average_token_embeddings_model_with_no_vocab():
    """
    Test the average_token_embeddings function when the Word2Vec model has no vocabulary.
    It should return None.
    """
    # Create an empty Word2Vec model
    model = Word2Vec(vector_size=50)
    tokens = ["hello", "world"]
    result = average_token_embeddings(tokens, model)
    assert result is None, "Expected None when the model has no vocabulary."

def test_average_token_embeddings_large_number_of_tokens(sample_model):
    """
    Test the average_token_embeddings function with a large number of tokens.
    It should efficiently compute the average without errors.
    """
    tokens = ["hello", "world", "test", "embedding", "word2vec", "model", "average", "token"]
    result = average_token_embeddings(tokens, sample_model)
    assert result is not None, "Expected a numpy array, got None."
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray."
    assert result.shape == (50,), f"Expected embedding size of 50, got {result.shape[0]}."
    # Manually compute expected average
    valid_tokens = [token for token in tokens if token in sample_model.wv]
    expected_average = np.mean(np.stack([sample_model.wv[token] for token in valid_tokens]), axis=0)
    np.testing.assert_array_almost_equal(result, expected_average, decimal=5,
                                         err_msg="Averaged embedding does not match expected average.")

def test_average_token_embeddings_non_string_tokens(sample_model):
    """
    Test the average_token_embeddings function with non-string tokens.
    It should handle them gracefully, ignoring non-string tokens.
    """
    tokens = ["hello", 123, None, "world"]
    # Modify the function to handle non-string tokens or expect it to raise an error
    # Assuming the function expects all tokens to be strings, non-string tokens are ignored
    result = average_token_embeddings(tokens, sample_model)
    assert result is not None, "Expected a numpy array, got None."
    valid_tokens = ["hello", "world"]
    expected_average = np.mean(np.stack([sample_model.wv[token] for token in valid_tokens]), axis=0)
    np.testing.assert_array_almost_equal(result, expected_average, decimal=5,
                                         err_msg="Averaged embedding does not match expected average with non-string tokens.")
