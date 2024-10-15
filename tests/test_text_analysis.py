# test_text_analysis.py

import pytest
from unittest.mock import patch, MagicMock
from centralized_nlp_package.text_processing.text_analysis import (
    load_word_set,
    load_syllable_counts,
    check_negation,
    calculate_polarity_score,
    polarity_score_per_section,
    polarity_score_per_sentence,
    is_complex,
    fog_analysis_per_section,
    fog_analysis_per_sentence,
    tone_count_with_negation_check,
    tone_count_with_negation_check_per_sentence,
    get_match_set,
    match_count,
    merge_counts,
    calculate_sentence_score,
    netscore,
    generate_match_count,
    generate_topic_statistics,
    generate_sentence_relevance_score
)
from centralized_nlp_package.utils.exception import FilesNotLoadedException
import os
import numpy as np
import pandas as pd
from collections import Counter

# Fixtures
@pytest.fixture
def mock_config():
    """
    Fixture to mock the config object used in the text_analysis module.
    """
    return MagicMock(
        lib_config=MagicMock(model_artifacts_path="/path/to/model_artifacts"),
        psycholinguistics=MagicMock(
            model_artifacts_path="/path/to/psycholinguistics",
            filecfg=MagicMock(
                positive_flnm="positive_words.txt",
                negative_flnm="negative_words.txt",
                vocab_pos_flnm="vocab_positive.txt",
                vocab_neg_flnm="vocab_negative.txt",
                syllable_flnm="syllable_counts.txt"
            )
        ),
        FILT_sections=["section1", "section2"]
    )

@pytest.fixture
def sample_word_set():
    """
    Fixture to provide a sample set of words.
    """
    return {"happy", "joyful", "sad", "terrible"}

@pytest.fixture
def sample_syllable_counts():
    """
    Fixture to provide a sample syllable counts dictionary.
    """
    return {
        "happy": 2,
        "joyful": 2,
        "sad": 1,
        "terrible": 3,
        "terribly": 3,
        "terriblying": 4
    }

@pytest.fixture
def sample_input_words():
    """
    Fixture to provide a sample list of input words.
    """
    return ["I", "am", "not", "happy", "with", "this", "terrible", "situation"]

@pytest.fixture
def sample_matches():
    """
    Fixture to provide a sample list of matched words/phrases.
    """
    return ["happy", "joyful", "sad", "terrible", "not happy", "very sad"]

@pytest.fixture
def sample_match_sets():
    """
    Fixture to provide a sample match sets dictionary.
    """
    return {
        "positive": {
            "unigrams": {"happy", "joyful"},
            "bigrams": {"not_happy"},
            "phrases": {"very sad"}
        },
        "negative": {
            "unigrams": {"sad", "terrible"},
            "bigrams": set(),
            "phrases": set()
        }
    }

# Test Cases

### 1. `load_word_set`

def test_load_word_set_success(mock_config):
    """
    Test load_word_set successfully loads a set of words from a file.
    """
    filename = "positive_words.txt"
    expected_word_set = {"happy", "joyful", "excited"}

    with patch("centralized_nlp_package.text_processing.text_analysis.load_list_from_txt", return_value=expected_word_set):
        with patch("os.path.join", return_value="/path/to/model_artifacts/positive_words.txt"):
            result = load_word_set(filename)
            assert result == expected_word_set, "The loaded word set does not match the expected set."

def test_load_word_set_file_not_found(mock_config):
    """
    Test load_word_set raises FilesNotLoadedException when the file is not found.
    """
    filename = "nonexistent_words.txt"

    with patch("centralized_nlp_package.text_processing.text_analysis.load_list_from_txt", side_effect=FileNotFoundError):
        with patch("os.path.join", return_value="/path/to/model_artifacts/nonexistent_words.txt"):
            with pytest.raises(FilesNotLoadedException) as exc_info:
                load_word_set(filename)
            assert exc_info.value.filename == filename, "Exception should contain the correct filename."

def test_load_word_set_other_exception(mock_config):
    """
    Test load_word_set raises a generic exception for unexpected errors.
    """
    filename = "positive_words.txt"

    with patch("centralized_nlp_package.text_processing.text_analysis.load_list_from_txt", side_effect=Exception("Unexpected Error")):
        with patch("os.path.join", return_value="/path/to/model_artifacts/positive_words.txt"):
            with pytest.raises(Exception) as exc_info:
                load_word_set(filename)
            assert "Unexpected Error" in str(exc_info.value), "Exception message should match the unexpected error."

### 2. `load_syllable_counts`

def test_load_syllable_counts_success(mock_config):
    """
    Test load_syllable_counts successfully loads syllable counts from a file.
    """
    filename = "syllable_counts.txt"
    expected_syllable_counts = {"happy": 2, "joyful": 2}

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts", return_value=expected_syllable_counts):
        with patch("os.path.join", return_value="/path/to/psycholinguistics/syllable_counts.txt"):
            result = load_syllable_counts(filename)
            assert result == expected_syllable_counts, "The loaded syllable counts do not match the expected counts."

def test_load_syllable_counts_file_not_found(mock_config):
    """
    Test load_syllable_counts raises FilesNotLoadedException when the file is not found.
    """
    filename = "nonexistent_syllable_counts.txt"

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts", side_effect=FileNotFoundError):
        with patch("os.path.join", return_value="/path/to/psycholinguistics/nonexistent_syllable_counts.txt"):
            with pytest.raises(FilesNotLoadedException) as exc_info:
                load_syllable_counts(filename)
            assert exc_info.value.filename == filename, "Exception should contain the correct filename."

def test_load_syllable_counts_other_exception(mock_config):
    """
    Test load_syllable_counts raises a generic exception for unexpected errors.
    """
    filename = "syllable_counts.txt"

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts", side_effect=Exception("Unexpected Error")):
        with patch("os.path.join", return_value="/path/to/psycholinguistics/syllable_counts.txt"):
            with pytest.raises(Exception) as exc_info:
                load_syllable_counts(filename)
            assert "Unexpected Error" in str(exc_info.value), "Exception message should match the unexpected error."

### 3. `check_negation`

def test_check_negation_with_negation_word():
    """
    Test check_negation returns True when a negation word is within the window.
    """
    input_words = ["I", "do", "not", "like", "this"]
    index = 3  # "like"
    negated_words = {"not", "never", "no"}

    with patch("centralized_nlp_package.text_processing.text_analysis.negated_words", negated_words):
        result = check_negation(input_words, index)
        assert result is True, "Expected negation to be detected."

def test_check_negation_without_negation_word():
    """
    Test check_negation returns False when no negation word is within the window.
    """
    input_words = ["I", "really", "like", "this"]
    index = 3  # "this"
    negated_words = {"not", "never", "no"}

    with patch("centralized_nlp_package.text_processing.text_analysis.negated_words", negated_words):
        result = check_negation(input_words, index)
        assert result is False, "Expected no negation to be detected."

def test_check_negation_edge_case_start_of_list():
    """
    Test check_negation when the index is at the start of the list.
    """
    input_words = ["not", "happy"]
    index = 1  # "happy"
    negated_words = {"not", "never", "no"}

    with patch("centralized_nlp_package.text_processing.text_analysis.negated_words", negated_words):
        result = check_negation(input_words, index)
        assert result is True, "Expected negation to be detected at the start of the list."

def test_check_negation_no_negated_words_defined():
    """
    Test check_negation returns False when negated_words is None.
    """
    input_words = ["I", "am", "happy"]
    index = 2  # "happy"

    with patch("centralized_nlp_package.text_processing.text_analysis.negated_words", None):
        result = check_negation(input_words, index)
        assert result is False, "Expected no negation to be detected when negated_words is None."

### 4. `calculate_polarity_score`

def test_calculate_polarity_score_all_positive():
    """
    Test calculate_polarity_score when all words are positive and no negations.
    """
    input_words = ["happy", "joyful"]
    positive_words = {"happy", "joyful"}
    negative_words = set()
    polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
        input_words, positive_words, negative_words
    )
    assert polarity_score == 1.0, "Polarity score should be 1.0 for all positive words."
    assert word_count == 2, "Word count should be 2."
    assert sum_negative == 0, "Sum of negatives should be 0."
    assert positive_count == 2, "Positive count should be 2."

def test_calculate_polarity_score_with_negations():
    """
    Test calculate_polarity_score with positive words preceded by negations.
    """
    input_words = ["not", "happy", "joyful"]
    positive_words = {"happy", "joyful"}
    negative_words = {"not"}
    with patch("centralized_nlp_package.text_processing.text_analysis.check_negation", return_value=True):
        polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
            input_words, positive_words, negative_words
        )
    assert polarity_score == (0 - 1) / 3, "Polarity score should account for negated positive word."
    assert word_count == 3, "Word count should be 3."
    assert sum_negative == 1, "Sum of negatives should be 1."
    assert positive_count == 0, "Positive count should be 0 due to negation."

def test_calculate_polarity_score_mixed():
    """
    Test calculate_polarity_score with mixed positive and negative words.
    """
    input_words = ["happy", "sad", "joyful", "terrible"]
    positive_words = {"happy", "joyful"}
    negative_words = {"sad", "terrible"}
    polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
        input_words, positive_words, negative_words
    )
    expected_polarity = (2 - 2) / 4
    assert polarity_score == expected_polarity, "Polarity score should reflect balanced positive and negative counts."
    assert word_count == 4, "Word count should be 4."
    assert sum_negative == 2, "Sum of negatives should be 2."
    assert positive_count == 2, "Positive count should be 2."

def test_calculate_polarity_score_no_words():
    """
    Test calculate_polarity_score with an empty input list.
    """
    input_words = []
    positive_words = {"happy"}
    negative_words = {"sad"}
    polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
        input_words, positive_words, negative_words
    )
    assert np.isnan(polarity_score), "Polarity score should be NaN when word count is 0."
    assert word_count == 0, "Word count should be 0."
    assert sum_negative == 0, "Sum of negatives should be 0."
    assert positive_count == 0, "Positive count should be 0."

### 5. `polarity_score_per_section`

def test_polarity_score_per_section_valid_input(mock_config):
    """
    Test polarity_score_per_section with valid text input.
    """
    text_list = ["I am happy", "This is terrible"]
    positive_word_set = {"happy", "joyful"}
    negative_word_set = {"sad", "terrible"}

    with patch("centralized_nlp_package.text_processing.text_analysis.load_word_set", side_effect=[positive_word_set, negative_word_set]):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text", return_value=("cleaned text", ["i", "am", "happy", "this", "is", "terrible"], 6)):
            with patch("centralized_nlp_package.text_processing.text_analysis.calculate_polarity_score", return_value=(0.0, 6, 1, 1, 0.0)):
                polarity_score, word_count, sum_negative, positive_count, legacy_score = polarity_score_per_section(text_list)
                assert polarity_score == 0.0, "Polarity score should be 0.0."
                assert word_count == 6, "Word count should be 6."
                assert sum_negative == 1, "Sum of negatives should be 1."
                assert positive_count == 1, "Positive count should be 1."
                assert legacy_score == 0.0, "Legacy score should be 0.0."

def test_polarity_score_per_section_insufficient_data(mock_config):
    """
    Test polarity_score_per_section with insufficient text input.
    """
    text_list = [""]

    with patch("centralized_nlp_package.text_processing.text_analysis.load_word_set"):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text", return_value=(None, [], 0)):
            polarity_score, word_count, sum_negative, positive_count, legacy_score = polarity_score_per_section(text_list)
            assert np.isnan(polarity_score), "Polarity score should be NaN for insufficient data."
            assert np.isnan(word_count), "Word count should be NaN for insufficient data."
            assert np.isnan(sum_negative), "Sum of negatives should be NaN for insufficient data."
            assert np.isnan(positive_count), "Positive count should be NaN for insufficient data."
            assert np.isnan(legacy_score), "Legacy score should be NaN for insufficient data."

### 6. `polarity_score_per_sentence`

def test_polarity_score_per_sentence_valid_input(mock_config):
    """
    Test polarity_score_per_sentence with valid list of sentences.
    """
    text_list = ["I am happy", "This is terrible"]
    positive_word_set = {"happy", "joyful"}
    negative_word_set = {"sad", "terrible"}
    input_words_list = [["i", "am", "happy"], ["this", "is", "terrible"]]
    word_count_list = [3, 3]

    with patch("centralized_nlp_package.text_processing.text_analysis.load_word_set", side_effect=[positive_word_set, negative_word_set]):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text_list", return_value=(None, input_words_list, word_count_list)):
            with patch("centralized_nlp_package.text_processing.text_analysis.calculate_polarity_score", side_effect=[(1.0, 3, 0, 1, 0.0), (-1.0, 3, 1, 0, 0.0)]):
                word_counts, positive_counts, negative_counts = polarity_score_per_sentence(text_list)
                assert word_counts == [3, 3], "Word counts should match the input sentences."
                assert positive_counts == [1, 0], "Positive counts should reflect the presence of positive words."
                assert negative_counts == [0, 1], "Negative counts should reflect the presence of negative words."

def test_polarity_score_per_sentence_insufficient_data(mock_config):
    """
    Test polarity_score_per_sentence with empty sentence list.
    """
    text_list = []

    with patch("centralized_nlp_package.text_processing.text_analysis.load_word_set"):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text_list", return_value=(None, [], 0)):
            word_counts, positive_counts, negative_counts = polarity_score_per_sentence(text_list)
            assert word_counts == [None], "Word counts should be None for insufficient data."
            assert positive_counts == [None], "Positive counts should be None for insufficient data."
            assert negative_counts == [None], "Negative counts should be None for insufficient data."

### 7. `is_complex`

def test_is_complex_true_due_to_suffix_rule(sample_syllable_counts):
    """
    Test is_complex returns True when a word meets suffix rules and root word has more than 2 syllables.
    """
    word = "terribly"
    result = is_complex(word, sample_syllable_counts)
    assert result is True, "Expected word to be complex based on suffix rules."

def test_is_complex_false_due_to_suffix_rule(sample_syllable_counts):
    """
    Test is_complex returns False when a word meets suffix rules but root word does not have more than 2 syllables.
    """
    word = "hoped"
    sample_syllable_counts["hope"] = 1
    result = is_complex(word, sample_syllable_counts)
    assert result is False, "Expected word to be not complex as root word does not meet syllable count."

def test_is_complex_true_due_to_syllable_count(sample_syllable_counts):
    """
    Test is_complex returns True when a word has more than 2 syllables.
    """
    word = "terrible"
    result = is_complex(word, sample_syllable_counts)
    assert result is True, "Expected word to be complex based on syllable count."

def test_is_complex_false_word_not_in_syllables(sample_syllable_counts):
    """
    Test is_complex returns False when the word is not in the syllables dictionary.
    """
    word = "unknownword"
    result = is_complex(word, sample_syllable_counts)
    assert result is False, "Expected word to be not complex as it is not in the syllables dictionary."

def test_is_complex_false_due_to_syllable_count(sample_syllable_counts):
    """
    Test is_complex returns False when a word has 2 or fewer syllables.
    """
    word = "happy"
    result = is_complex(word, sample_syllable_counts)
    assert result is False, "Expected word to be not complex as syllable count is not greater than 2."

### 8. `fog_analysis_per_section`

def test_fog_analysis_per_section_valid_input(mock_config, sample_syllable_counts):
    """
    Test fog_analysis_per_section with valid text input.
    """
    text_list = ["This is a simple sentence.", "This sentence is terribly complex."]
    syllable_counts = sample_syllable_counts

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts", return_value=syllable_counts):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text", return_value=("cleaned text", ["this", "is", "a", "simple", "sentence", "this", "sentence", "is", "terribly", "complex"], 10)):
            with patch("centralized_nlp_package.text_processing.text_analysis.is_complex", side_effect=[False, True]):
                fog_index, complex_word_count, average_words_per_sentence, total_word_count = fog_analysis_per_section(text_list)
                expected_fog_index = 0.4 * (5 + 100 * (1 / 10))
                assert fog_index == expected_fog_index, "Fog index should be calculated correctly."
                assert complex_word_count == 1, "Complex word count should be 1."
                assert average_words_per_sentence == 5.0, "Average words per sentence should be 5.0."
                assert total_word_count == 10, "Total word count should be 10."

def test_fog_analysis_per_section_insufficient_data(mock_config):
    """
    Test fog_analysis_per_section with insufficient text input.
    """
    text_list = [""]

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts"):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text", return_value=(None, [], 0)):
            fog_index, complex_word_count, average_words_per_sentence, total_word_count = fog_analysis_per_section(text_list)
            assert np.isnan(fog_index), "Fog index should be NaN for insufficient data."
            assert np.isnan(complex_word_count), "Complex word count should be NaN for insufficient data."
            assert np.isnan(average_words_per_sentence), "Average words per sentence should be NaN for insufficient data."
            assert np.isnan(total_word_count), "Total word count should be NaN for insufficient data."

### 9. `fog_analysis_per_sentence`

def test_fog_analysis_per_sentence_valid_input(mock_config, sample_syllable_counts):
    """
    Test fog_analysis_per_sentence with valid list of sentences.
    """
    text_list = ["This is simple.", "This is terribly complex."]
    syllable_counts = sample_syllable_counts
    input_words_list = [["this", "is", "simple"], ["this", "is", "terribly", "complex"]]
    word_count_list = [3, 4]
    average_words_per_sentence = 3.5

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts", return_value=syllable_counts):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text_list", return_value=(None, input_words_list, word_count_list)):
            with patch("centralized_nlp_package.text_processing.text_analysis.is_complex", side_effect=[False, True]):
                fog_index_list, complex_word_count_list, total_word_count_list = fog_analysis_per_sentence(text_list)
                expected_fog_indices = [
                    0.4 * (3.5 + 100 * (0 / 3)),
                    0.4 * (3.5 + 100 * (1 / 4))
                ]
                assert fog_index_list == expected_fog_indices, "Fog index list should be calculated correctly."
                assert complex_word_count_list == [0, 1], "Complex word counts should match the input."
                assert total_word_count_list == [3, 4], "Total word counts should match the input."

def test_fog_analysis_per_sentence_insufficient_data(mock_config):
    """
    Test fog_analysis_per_sentence with empty sentence list.
    """
    text_list = []

    with patch("centralized_nlp_package.text_processing.text_analysis.load_syllable_counts"):
        with patch("centralized_nlp_package.text_processing.text_analysis.preprocess_text_list", return_value=(None, [], 0)):
            fog_index_list, complex_word_count_list, total_word_count_list = fog_analysis_per_sentence(text_list)
            assert fog_index_list == [None], "Fog index list should be None for insufficient data."
            assert complex_word_count_list == [None], "Complex word count list should be None for insufficient data."
            assert total_word_count_list == [None], "Total word count list should be None for insufficient data."

### 10. `tone_count_with_negation_check`

def test_tone_count_with_negation_check_valid_input(mock_config):
    """
    Test tone_count_with_negation_check with valid text input.
    """
    text_list = ["I am happy", "This is terrible"]
    sentiment_metrics = (0.0, 4, 1, 1, 0.0)

    with patch("centralized_nlp_package.text_processing.text_analysis.polarity_score_per_section", return_value=sentiment_metrics):
        polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores = tone_count_with_negation_check(text_list)
        assert polarity_scores == [0.0], "Polarity scores should match the sentiment metrics."
        assert word_count == [4], "Word counts should match the sentiment metrics."
        assert negative_word_count_list == [1], "Negative word counts should match the sentiment metrics."
        assert positive_word_count_list == [1], "Positive word counts should match the sentiment metrics."
        assert legacy_scores == [0.0], "Legacy scores should match the sentiment metrics."

def test_tone_count_with_negation_check_empty_input(mock_config):
    """
    Test tone_count_with_negation_check with empty text input.
    """
    text_list = []

    with patch("centralized_nlp_package.text_processing.text_analysis.polarity_score_per_section", return_value=(np.nan, np.nan, np.nan, np.nan, np.nan)):
        polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores = tone_count_with_negation_check(text_list)
        assert polarity_scores == [np.nan], "Polarity scores should be NaN for empty input."
        assert word_count == [np.nan], "Word counts should be NaN for empty input."
        assert negative_word_count_list == [np.nan], "Negative word counts should be NaN for empty input."
        assert positive_word_count_list == [np.nan], "Positive word counts should be NaN for empty input."
        assert legacy_scores == [np.nan], "Legacy scores should be NaN for empty input."

### 11. `tone_count_with_negation_check_per_sentence`

def test_tone_count_with_negation_check_per_sentence_valid_input(mock_config):
    """
    Test tone_count_with_negation_check_per_sentence with valid list of sentences.
    """
    text_list = ["I am happy", "This is terrible"]
    sentiment_metrics = ([1, 0], [1, 0], [0, 1])

    with patch("centralized_nlp_package.text_processing.text_analysis.polarity_score_per_sentence", return_value=sentiment_metrics):
        word_count, positive_word_count_list_per_sentence, negative_word_count_list_per_sentence = tone_count_with_negation_check_per_sentence(text_list)
        assert word_count == [[1, 0]], "Word counts should match the sentiment metrics."
        assert positive_word_count_list_per_sentence == [[1, 0]], "Positive word counts should match the sentiment metrics."
        assert negative_word_count_list_per_sentence == [[0, 1]], "Negative word counts should match the sentiment metrics."

def test_tone_count_with_negation_check_per_sentence_empty_input(mock_config):
    """
    Test tone_count_with_negation_check_per_sentence with empty sentence list.
    """
    text_list = []

    with patch("centralized_nlp_package.text_processing.text_analysis.polarity_score_per_sentence", return_value=(None, None, None)):
        word_count, positive_word_count_list_per_sentence, negative_word_count_list_per_sentence = tone_count_with_negation_check_per_sentence(text_list)
        assert word_count == [None], "Word counts should be None for empty input."
        assert positive_word_count_list_per_sentence == [None], "Positive word counts should be None for empty input."
        assert negative_word_count_list_per_sentence == [None], "Negative word counts should be None for empty input."

### 12. `get_match_set`

def test_get_match_set_with_various_matches(sample_matches):
    """
    Test get_match_set generates correct unigrams, bigrams, and phrases.
    """
    expected_unigrams = {"happy", "joyful", "sad", "terrible"}
    expected_bigrams = {"not_happy", "very_sad"}
    expected_phrases = {"very sad"}

    with patch("centralized_nlp_package.text_processing.text_analysis.tokenize_matched_words", side_effect=[["not", "happy"], ["very", "sad"]]):
        result = get_match_set(sample_matches)
        assert result["original"] == set(sample_matches), "Original match set should match the input."
        assert result["unigrams"] == expected_unigrams, "Unigrams should match the expected set."
        assert result["bigrams"] == expected_bigrams, "Bigrams should match the expected set."
        assert result["phrases"] == expected_phrases, "Phrases should match the expected set."

### 13. `match_count`

def test_match_count_with_phrases(sample_match_sets):
    """
    Test match_count correctly counts unigrams, bigrams, and phrases with phrases enabled.
    """
    text = "I am not happy and very sad."
    match_sets = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.word_tokenizer", return_value=["i", "am", "not", "happy", "and", "very", "sad"]):
        with patch("centralized_nlp_package.text_processing.text_analysis.find_ngrams", return_value=[("i", "am"), ("am", "not"), ("not", "happy"), ("happy", "and"), ("and", "very"), ("very", "sad")]):
            result = match_count(text, match_sets, phrases=True)
            assert result["positive"]["uni"] == 1, "Unigram positive count should be 1."
            assert result["positive"]["bi"] == 1, "Bigram positive count should be 1."
            assert result["positive"]["phrase"] == 0, "Phrase positive count should be 0."
            assert result["negative"]["uni"] == 1, "Unigram negative count should be 1."
            assert result["negative"]["bi"] == 0, "Bigram negative count should be 0."
            assert result["negative"]["phrase"] == 0, "Phrase negative count should be 0."
            assert result["len"] == 7, "Unigram length should be 7."
            assert result["raw_len"] == 7, "Raw length should be 7."

def test_match_count_without_phrases(sample_match_sets):
    """
    Test match_count correctly counts unigrams and bigrams without phrases.
    """
    text = "I am not happy and very sad."
    match_sets = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.word_tokenizer", return_value=["i", "am", "not", "happy", "and", "very", "sad"]):
        with patch("centralized_nlp_package.text_processing.text_analysis.find_ngrams", return_value=[("i", "am"), ("am", "not"), ("not", "happy"), ("happy", "and"), ("and", "very"), ("very", "sad")]):
            result = match_count(text, match_sets, phrases=False)
            assert result["positive"]["uni"] == 1, "Unigram positive count should be 1."
            assert result["positive"]["bi"] == 1, "Bigram positive count should be 1."
            assert "phrase" not in result["positive"], "Phrase counts should not be present when phrases=False."
            assert result["negative"]["uni"] == 1, "Unigram negative count should be 1."
            assert result["negative"]["bi"] == 0, "Bigram negative count should be 0."
            assert "phrase" not in result["negative"], "Phrase counts should not be present when phrases=False."
            assert result["len"] == 7, "Unigram length should be 7."
            assert result["raw_len"] == 7, "Raw length should be 7."

### 14. `merge_counts`

def test_merge_counts_normal_case():
    """
    Test merge_counts correctly merges multiple count dictionaries.
    """
    counts = [
        {"happy": 2, "sad": 1},
        {"happy": 1, "terrible": 3},
        {"joyful": 4}
    ]
    expected_merged = {"happy": 3, "sad": 1, "terrible": 3, "joyful": 4}
    result = merge_counts(counts)
    assert result == expected_merged, "Merged counts should match the expected dictionary."

def test_merge_counts_empty_list():
    """
    Test merge_counts returns {"NO_MATCH": 1} when the input list is empty.
    """
    counts = []
    result = merge_counts(counts)
    assert result == {"NO_MATCH": 1}, "Should return {'NO_MATCH': 1} for empty input."

def test_merge_counts_error_handling():
    """
    Test merge_counts returns {"ERROR": 1} when an exception occurs.
    """
    counts = [{"happy": 2}]
    with patch("collections.Counter.__add__", side_effect=Exception("Merge Error")):
        result = merge_counts(counts)
        assert result == {"ERROR": 1}, "Should return {'ERROR': 1} when an exception occurs."

### 15. `calculate_sentence_score`

def test_calculate_sentence_score_weighted():
    """
    Test calculate_sentence_score with weighting enabled.
    """
    a = [1, 2, 0]
    b = [0.5, 1.5, 2.0]
    result = calculate_sentence_score(a, b, weight=True)
    expected_score = (1*0.5 + 2*1.5) / 2
    assert result == expected_score, "Weighted sentence score should be calculated correctly."

def test_calculate_sentence_score_unweighted():
    """
    Test calculate_sentence_score with weighting disabled.
    """
    a = [1, 2, 0]
    b = [0.5, 1.5, 2.0]
    result = calculate_sentence_score(a, b, weight=False)
    expected_score = (1*0.5 + 1*1.5) / 2
    assert result == expected_score, "Unweighted sentence score should be calculated correctly."

def test_calculate_sentence_score_length_mismatch():
    """
    Test calculate_sentence_score returns None when list lengths do not match.
    """
    a = [1, 2]
    b = [0.5]
    result = calculate_sentence_score(a, b, weight=True)
    assert result is None, "Should return None when list lengths do not match."

def test_calculate_sentence_score_no_relevant():
    """
    Test calculate_sentence_score returns None when there are no relevant items.
    """
    a = [0, 0, 0]
    b = [0.5, 1.5, 2.0]
    result = calculate_sentence_score(a, b, weight=True)
    assert result is None, "Should return None when there are no relevant items."

def test_calculate_sentence_score_empty_lists():
    """
    Test calculate_sentence_score returns None when input lists are empty.
    """
    a = []
    b = []
    result = calculate_sentence_score(a, b, weight=True)
    assert result is None, "Should return None for empty input lists."

### 16. `netscore`

def test_netscore_valid_input():
    """
    Test netscore correctly calculates the net score.
    """
    a = [1, -1, 2]
    b = [1, 0, 1]
    result = netscore(a, b)
    expected_score = (1*1 + 1*0 + 1*1)
    assert result == expected_score, "Net score should be calculated correctly."

def test_netscore_length_mismatch():
    """
    Test netscore returns None when list lengths do not match.
    """
    a = [1, 2]
    b = [1]
    result = netscore(a, b)
    assert result is None, "Should return None when list lengths do not match."

def test_netscore_no_relevant():
    """
    Test netscore returns None when there are no relevant items.
    """
    a = [0, 0, 0]
    b = [1, 0, 1]
    result = netscore(a, b)
    assert result is None, "Should return None when there are no relevant items."

def test_netscore_empty_lists():
    """
    Test netscore returns None when input lists are empty.
    """
    a = []
    b = []
    result = netscore(a, b)
    assert result is None, "Should return None for empty input lists."

### 17. `generate_match_count`

def test_generate_match_count_valid_input(mock_config, sample_match_sets):
    """
    Test generate_match_count adds match counts correctly to the DataFrame.
    """
    df = pd.DataFrame({
        "section1": ["I am happy", "This is sad"],
        "section2": ["Joyful day", "Terrible experience"]
    })
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.match_count", return_value={"positive": {"uni":1, "bi":1, "phrase":0}, "negative": {"uni":1, "bi":0, "phrase":0}, "len":7, "raw_len":7}):
        updated_df = generate_match_count(df, word_set_dict)
        assert "matches_section1" in updated_df.columns, "matches_section1 column should be added."
        assert "matches_section2" in updated_df.columns, "matches_section2 column should be added."
        assert updated_df["matches_section1"].iloc[0]["positive"]["uni"] == 1, "Match counts should be correctly assigned."
        assert updated_df["matches_section2"].iloc[1]["negative"]["uni"] == 1, "Match counts should be correctly assigned."

def test_generate_match_count_empty_df(mock_config, sample_match_sets):
    """
    Test generate_match_count handles empty DataFrame.
    """
    df = pd.DataFrame(columns=["section1", "section2"])
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.match_count", return_value={}):
        updated_df = generate_match_count(df, word_set_dict)
        for label in mock_config.FILT_sections:
            assert f"matches_{label}" in updated_df.columns, f"matches_{label} column should be added even for empty DataFrame."
        assert updated_df.empty, "DataFrame should remain empty."

### 18. `generate_topic_statistics`

def test_generate_topic_statistics_valid_input(mock_config, sample_match_sets):
    """
    Test generate_topic_statistics adds topic statistics correctly to the DataFrame.
    """
    df = pd.DataFrame({
        "matches_section1": [
            {"positive": {"uni":1, "bi":1, "phrase":0, "stats": {"happy":1, "not_happy":1}},
             "negative": {"uni":1, "bi":0, "phrase":0, "stats": {"sad":1}}}
        ]
    })
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.merge_counts", return_value={"happy":1, "not_happy":1, "sad":1}):
        updated_df = generate_topic_statistics(df, word_set_dict)
        assert "LEN_section1" in updated_df.columns, "LEN_section1 column should be added."
        assert "RAW_LEN_section1" in updated_df.columns, "RAW_LEN_section1 column should be added."
        assert "positive_TOTAL_section1" in updated_df.columns, "positive_TOTAL_section1 column should be added."
        assert "negative_TOTAL_section1" in updated_df.columns, "negative_TOTAL_section1 column should be added."
        assert "positive_STATS_section1" in updated_df.columns, "positive_STATS_section1 column should be added."
        assert "negative_STATS_section1" in updated_df.columns, "negative_STATS_section1 column should be added."
        assert "positive_STATS_LIST_section1" in updated_df.columns, "positive_STATS_LIST_section1 column should be added."
        assert "negative_STATS_LIST_section1" in updated_df.columns, "negative_STATS_LIST_section1 column should be added."
        assert "NUM_SENTS_section1" in updated_df.columns, "NUM_SENTS_section1 column should be added."

def test_generate_topic_statistics_empty_matches(mock_config, sample_match_sets):
    """
    Test generate_topic_statistics handles empty match counts.
    """
    df = pd.DataFrame({
        "matches_section1": [{}]
    })
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.merge_counts", return_value={"NO_MATCH": 1}):
        updated_df = generate_topic_statistics(df, word_set_dict)
        assert updated_df["positive_TOTAL_section1"].iloc[0] == 1, "Total counts should reflect NO_MATCH."
        assert updated_df["negative_TOTAL_section1"].iloc[0] == 1, "Total counts should reflect NO_MATCH."

### 19. `generate_sentence_relevance_score`

def test_generate_sentence_relevance_score_valid_input(mock_config, sample_match_sets):
    """
    Test generate_sentence_relevance_score adds relevance scores correctly to the DataFrame.
    """
    df = pd.DataFrame({
        "matches_section1": [
            {"positive_TOTAL_section1": [1, 0],
             "SENT_LABELS_section1": [1, 0],
             "positive_RELEVANCE_section1": 0.5,
             "positive_SENT_section1": 1.0,
             "positive_SENT_REL_section1": 0.5,
             "positive_SENT_WEIGHT_section1": 0.8,
             "positive_SENT_WEIGHT_REL_section1": 0.4,
             "positive_NET_SENT_section1": 1.0}
        ]
    })
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.calculate_sentence_score", return_value=0.8):
        with patch("centralized_nlp_package.text_processing.text_analysis.netscore", return_value=1.0):
            updated_df = generate_sentence_relevance_score(df, word_set_dict)
            assert "SENT_section1" in updated_df.columns, "SENT_section1 column should be added."
            assert "NET_SENT_section1" in updated_df.columns, "NET_SENT_section1 column should be added."
            assert "positive_RELEVANCE_section1" in updated_df.columns, "positive_RELEVANCE_section1 column should be added."
            assert "positive_SENT_rel_section1" in updated_df.columns, "positive_SENT_rel_section1 column should be added."
            assert "positive_SENT_weight_section1" in updated_df.columns, "positive_SENT_weight_section1 column should be added."
            assert "positive_SENT_weight_rel_section1" in updated_df.columns, "positive_SENT_weight_rel_section1 column should be added."

def test_generate_sentence_relevance_score_empty_matches(mock_config, sample_match_sets):
    """
    Test generate_sentence_relevance_score handles empty match counts.
    """
    df = pd.DataFrame({
        "matches_section1": [{}]
    })
    word_set_dict = sample_match_sets

    with patch("centralized_nlp_package.text_processing.text_analysis.calculate_sentence_score", return_value=None):
        with patch("centralized_nlp_package.text_processing.text_analysis.netscore", return_value=None):
            updated_df = generate_sentence_relevance_score(df, word_set_dict)
            assert "SENT_section1" in updated_df.columns, "SENT_section1 column should be added."
            assert np.isnan(updated_df["SENT_section1"].iloc[0]), "SENT_section1 should be NaN for empty matches."
            assert np.isnan(updated_df["NET_SENT_section1"].iloc[0]), "NET_SENT_section1 should be NaN for empty matches."

