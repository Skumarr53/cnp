# centralized_nlp_package/text_processing/text_analysis.py

import os
import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from collections import Counter
from loguru import logger
from centralized_nlp_package.text_processing.text_utils import (
    generate_ngrams,
    load_syllable_counts,
    load_set_from_txt, 
    combine_sentiment_scores
)
from centralized_nlp_package.text_processing.text_preprocessing import (
    preprocess_text,
    preprocess_text_list,
    tokenize_matched_words,
    tokenize_and_lemmatize_text,
)
from centralized_nlp_package import config
from centralized_nlp_package.utils import FilesNotLoadedException



def load_word_set(filename: str) -> set:
    """
    Loads a set of words from a specified file in the model artifacts path.

    Args:
        filename (str): Name of the file containing words.

    Returns:
        set: A set of words loaded from the file.

    Example:
        >>> from centralized_nlp_package.text_processing import load_word_set
        >>> word_set = load_word_set("positive_words.txt")
        >>> print(len(word_set))
        1500
    """
    file_path = os.path.join(config.lib_config.model_artifacts_path, filename)
    try:
        word_set = load_set_from_txt(file_path)
        logger.debug(f"Loaded word set from {file_path} with {len(word_set)} words.")
        return word_set
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(filename=filename)
    except Exception as e:
        logger.error(f"Error loading word set from {file_path}: {e}")
        raise


def check_negation(input_words: List[str], index: int, negation_words: set) -> bool:
    """
    Checks if a word at a given index is preceded by a negation within three words.

    Args:
        input_words (List[str]): List of tokenized words.
        index (int): Current word index.
        negation_words (set): Set of negation words to check against.

    Returns:
        bool: True if negation is found within the specified window, False otherwise.

    Example:
        >>> from centralized_nlp_package.text_processing import check_negation
        >>> words = ["I", "do", "not", "like", "this"]
        >>> negations = {"not", "never", "no"}
        >>> check_negation(words, 3, negations)
        True
    """
    negation_window = 3
    start = max(0, index - negation_window)
    for i in range(start, index):
        if input_words[i].lower() in negation_words:
            logger.debug(
                f"Negation found before word '{input_words[index]}' at position {i}."
            )
            return True
    return False

def calculate_polarity_score(
    input_words: List[str],
    positive_words: set,
    negative_words: set,
    negation_words: set,
) -> Tuple[float, int, int, int, float]:
    """
    Calculates the polarity score based on positive and negative word counts.

    Args:
        input_words (List[str]): List of tokenized words.
        positive_words (set): Set of positive words.
        negative_words (set): Set of negative words.
        negation_words (set): Set of negation words.

    Returns:
        Tuple[float, int, int, int, float]: 
            - Polarity score
            - Total word count
            - Sum of negatives
            - Count of positives
            - Legacy score

    Example:
        >>> from centralized_nlp_package.text_processing import calculate_polarity_score
        >>> words = ["I", "do", "not", "like", "this"]
        >>> pos = {"like", "love"}
        >>> neg = {"hate", "dislike"}
        >>> negations = {"not", "never"}
        >>> score = calculate_polarity_score(words, pos, neg, negations)
        >>> print(score)
        (0.2, 5, 1, 0, 0.0)
    """
    positive_count = 0
    negative_count = 0
    word_count = len(input_words)

    for i, word in enumerate(input_words):
        word_lower = word.lower()
        if word_lower in negative_words:
            negative_count += 1
            logger.debug(f"Negative word found: {word} at position {i}.")

        if word_lower in positive_words:
            if check_negation(input_words, i, negation_words):
                negative_count += 1
                logger.debug(f"Positive word '{word}' at position {i} negated.")
            else:
                positive_count += 1
                logger.debug(f"Positive word found: {word} at position {i}.")

    sum_negative = negative_count
    polarity_score = (
        (positive_count - sum_negative) / word_count if word_count > 0 else np.nan
    )
    legacy_score = combine_sentiment_scores(positive_count, sum_negative)
    logger.debug(
        f"Polarity Score: {polarity_score}, Word Count: {word_count}, "
        f"Sum Negative: {sum_negative}, Positive Count: {positive_count}, Legacy Score: {legacy_score}"
    )
    return polarity_score, word_count, sum_negative, positive_count, legacy_score

def polarity_score_per_section(
    text_list: List[str],
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[float]]:
    """
    Generates the polarity score for the input text to identify sentiment.

    Args:
        text_list (List[str]): List of texts to analyze.

    Returns:
        Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[float]]:
            - Polarity score
            - Word count
            - Sum of negatives
            - Count of positives
            - Legacy score

    Example:
        >>> from centralized_nlp_package.text_processing import polarity_score_per_section
        >>> texts = ["I love this product", "I do not like this service"]
        >>> score = polarity_score_per_section(texts)
        >>> print(score)
        (0.1, 8, 1, 1, 0.0)
    """
    positive_words = load_word_set(config.psycholinguistics.filecfg.positive_flnm)
    negative_words = load_word_set(config.psycholinguistics.filecfg.negative_flnm)
    negation_words = load_word_set(config.psycholinguistics.filecfg.negation_flnm)

    cleaned_text, input_words, word_count = preprocess_text(text_list)

    if cleaned_text and word_count > 1:
        polarity_score, word_count, sum_negative, positive_count, legacy_score = calculate_polarity_score(
            input_words, positive_words, negative_words, negation_words
        )
        return polarity_score, word_count, sum_negative, positive_count, legacy_score
    else:
        logger.warning("Insufficient data for polarity score calculation.")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def polarity_score_per_sentence(
    text_list: List[str],
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """
    Analyzes sentences to calculate polarity scores.

    Args:
        text_list (List[str]): List of sentences to analyze.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
            - Word counts per sentence
            - Positive word counts per sentence
            - Negative word counts per sentence

    Example:
        >>> from centralized_nlp_package.text_processing import polarity_score_per_sentence
        >>> sentences = ["I love this", "I do not like that"]
        >>> counts = polarity_score_per_sentence(sentences)
        >>> print(counts)
        ([3, 5], [1, 0], [0, 1])
    """
    positive_words = load_word_set(config.psycholinguistics.filecfg.vocab_pos_flnm)
    negative_words = load_word_set(config.psycholinguistics.filecfg.vocab_neg_flnm)
    negation_words = load_word_set(config.psycholinguistics.filecfg.negation_flnm)

    _, input_words_list, word_count_list = preprocess_text_list(text_list)

    if text_list and word_count_list:
        word_counts = []
        positive_counts = []
        negative_counts = []

        for input_words in input_words_list:
            polarity, wc, sum_neg, pos_count, _ = calculate_polarity_score(
                input_words, positive_words, negative_words, negation_words
            )
            word_counts.append(wc)
            positive_counts.append(pos_count)
            negative_counts.append(sum_neg)

        logger.info("Sentence-level polarity analysis completed.")
        return word_counts, positive_counts, negative_counts
    else:
        logger.warning("Insufficient data for sentence-level analysis.")
        return None, None, None


def is_complex(word: str, syllables: Dict[str, int]) -> bool:
    """
    Determines if a word is complex based on syllable count and suffix rules.

    Args:
        word (str): The word to evaluate.
        syllables (Dict[str, int]): Dictionary of syllable counts.

    Returns:
        bool: True if the word is complex, False otherwise.

    Example:
        >>> from centralized_nlp_package.text_processing import is_complex
        >>> syllable_dict = {"beautiful": 3, "cat": 1}
        >>> is_complex("beautiful", syllable_dict)
        True
        >>> is_complex("cat", syllable_dict)
        False
    """
    if word not in syllables:
        return False

    suffix_rules = {
        "es": [2, 1],
        "ing": [3],
        "ed": [2, 1],
    }

    for suffix, strip_lengths in suffix_rules.items():
        if word.endswith(suffix):
            for strip_length in strip_lengths:
                root = word[:-strip_length]
                if root in syllables and syllables[root] > 2:
                    return True

    if syllables.get(word, 0) > 2:
        return True

    return False

def fog_analysis_per_section(
    text_list: List[str],
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
    """
    Calculates the Fog Index for the input text to evaluate readability.

    Args:
        text_list (List[str]): List of texts to analyze.

    Returns:
        Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
            - Fog index
            - Complex word count
            - Average words per sentence
            - Total word count

    Example:
        >>> from centralized_nlp_package.text_processing import fog_analysis_per_section
        >>> texts = ["This is a simple sentence.", "This sentence is unnecessarily complex."]
        >>> fog = fog_analysis_per_section(texts)
        >>> print(fog)
        (12.0, 1, 5.0, 10)
    """
    syllables = load_syllable_counts(config.psycholinguistics.filecfg.syllable_flnm)

    raw_text = " ".join(text_list) if isinstance(text_list, list) else text_list
    total_word_count = len(raw_text.split())
    sentences = raw_text.split(". ")
    average_words_per_sentence = (
        np.mean([len(sentence.strip().split()) for sentence in sentences])
        if sentences
        else 0
    )

    cleaned_text, input_words, word_count = preprocess_text(text_list)

    if cleaned_text and word_count > 1:
        complex_word_count = sum(is_complex(word, syllables) for word in input_words)
        fog_index = 0.4 * (
            average_words_per_sentence + 100 * (complex_word_count / total_word_count)
        )
        logger.info(
            f"Fog Analysis - Fog Index: {fog_index}, Complex Words: {complex_word_count}, "
            f"Average Words/Sentence: {average_words_per_sentence}, Total Words: {total_word_count}"
        )
        return fog_index, complex_word_count, average_words_per_sentence, total_word_count
    else:
        logger.warning("Insufficient data for Fog Analysis.")
        return np.nan, np.nan, np.nan, np.nan


def fog_analysis_per_sentence(
    text_list: List[str],
) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
    """
    Calculates the Fog Index for each sentence in the input list.

    Args:
        text_list (List[str]): List of sentences to analyze.

    Returns:
        Tuple[Optional[List[float]], Optional[List[int]], Optional[List[int]]]:
            - Fog index list
            - Complex word count list
            - Total word count list

    Example:
        >>> from centralized_nlp_package.text_processing import fog_analysis_per_sentence
        >>> sentences = ["This is simple.", "This sentence is complex and unnecessarily verbose."]
        >>> fog_scores = fog_analysis_per_sentence(sentences)
        >>> print(fog_scores)
        ([7.2, 14.4], [0, 2], [3, 6])
    """
    syllables = load_syllable_counts(config.psycholinguistics.filecfg.syllable_flnm)

    word_count_list = [len(sentence.split()) for sentence in text_list]
    average_words_per_sentence = np.mean(word_count_list) if text_list else 0

    _, input_words_list, _ = preprocess_text_list(text_list)

    if text_list and word_count_list:
        fog_index_list = []
        complex_word_count_list = []
        total_word_count_list = word_count_list

        for input_words in input_words_list:
            complex_count = sum(is_complex(word, syllables) for word in input_words)
            word_count = len(input_words)
            fog_index = (
                0.4 * (average_words_per_sentence + 100 * (complex_count / word_count))
                if word_count > 0
                else np.nan
            )
            fog_index_list.append(fog_index)
            complex_word_count_list.append(complex_count)

        logger.info("Sentence-level Fog Analysis completed.")
        return fog_index_list, complex_word_count_list, total_word_count_list
    else:
        logger.warning("Insufficient data for sentence-level Fog Analysis.")
        return None, None, None


def tone_count_with_negation_check(
    text_list: List[str],
) -> Tuple[
    List[Optional[float]],
    List[Optional[int]],
    List[Optional[int]],
    List[Optional[int]],
    List[Optional[float]],
]:
    """
    Counts positive and negative words with negation checks. Accounts for simple negation only for positive words.
    Simple negation is considered as occurrences of negation words within three words preceding a positive word.

    Args:
        text_list (List[str]): List of texts to analyze.

    Returns:
        Tuple[
            List[Optional[float]],
            List[Optional[int]],
            List[Optional[int]],
            List[Optional[int]],
            List[Optional[float]],
        ]:
            - Polarity scores
            - Word counts
            - Negative word counts
            - Positive word counts
            - Legacy scores

    Example:
        >>> from centralized_nlp_package.text_processing import tone_count_with_negation_check
        >>> texts = ["I love this product", "I do not like this service"]
        >>> results = tone_count_with_negation_check(texts)
        >>> print(results)
        ([0.2, -0.2], [3, 5], [0, 1], [1, 0], [0.0, 0.0])
    """
    polarity_scores: List[Optional[float]] = []
    legacy_scores: List[Optional[float]] = []
    word_counts: List[Optional[int]] = []
    negative_word_counts: List[Optional[int]] = []
    positive_word_counts: List[Optional[int]] = []

    sentiment_metrics = polarity_score_per_section(text_list)

    polarity_scores.append(sentiment_metrics[0])
    word_counts.append(sentiment_metrics[1])
    negative_word_counts.append(sentiment_metrics[2])
    positive_word_counts.append(sentiment_metrics[3])
    legacy_scores.append(sentiment_metrics[4])

    return (
        polarity_scores,
        word_counts,
        negative_word_counts,
        positive_word_counts,
        legacy_scores,
    )


    
def tone_count_with_negation_check_per_sentence(
    text_list: List[str],
) -> Tuple[
    Optional[List[int]],
    Optional[List[int]],
    Optional[List[int]],
]:
    """
    Counts positive and negative words with negation checks for each sentence. 
    Accounts for simple negation only for positive words, defined as negation words within three words preceding a positive word.

    Args:
        text_list (List[str]): List of sentences to analyze.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
            - Word counts per sentence
            - Positive word counts per sentence
            - Negative word counts per sentence

    Example:
        >>> from centralized_nlp_package.text_processing import tone_count_with_negation_check_per_sentence
        >>> sentences = ["I love this", "I do not like that"]
        >>> counts = tone_count_with_negation_check_per_sentence(sentences)
        >>> print(counts)
        ([3, 5], [1, 0], [0, 1])
    """
    word_counts: Optional[List[int]] = []
    positive_word_counts_per_sentence: Optional[List[int]] = []
    negative_word_counts_per_sentence: Optional[List[int]] = []

    sentiment_metrics = polarity_score_per_sentence(text_list)
    word_counts.append(sentiment_metrics[0])
    positive_word_counts_per_sentence.append(sentiment_metrics[1])
    negative_word_counts_per_sentence.append(sentiment_metrics[2])

    return word_counts, positive_word_counts_per_sentence, negative_word_counts_per_sentence


## TODO: topic modelling
def get_match_set(matches: List[str]) -> Dict[str, set]:
    """
    Generates the match set including unigrams, bigrams, and phrases.

    Args:
        matches (List[str]): List of matched words or phrases.

    Returns:
        Dict[str, set]: Dictionary containing original matches, unigrams, bigrams, and phrases.

    Example:
        >>> from centralized_nlp_package.text_processing import get_match_set
        >>> matches = ["happy", "very happy", "extremely happy"]
        >>> match_set = get_match_set(matches)
        >>> print(match_set["unigrams"])
        {'happy'}
    """
    bigrams = set(
        [
            word.lower()
            for word in matches
            if len(word.split("_")) == 2
        ]
        + [
            word.lower().replace(" ", "_")
            for word in matches
            if len(word.split(" ")) == 2
        ]
        + [
            "_".join(tokenize_matched_words(word))
            for word in matches
            if len(word.split(" ")) == 2
        ]
    )

    unigrams = set(
        [
            tokenize_matched_words(match)[0]
            for match in matches
            if "_" not in match and len(match.split(" ")) == 1
        ]
        + [
            match.lower()
            for match in matches
            if "_" not in match and len(match.split(" ")) == 1
        ]
    )

    phrases = {phrase.lower() for phrase in matches if len(phrase.split(" ")) > 2}

    logger.debug(
        f"Generated match sets: {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(phrases)} phrases."
    )
    return {
        "original": set(matches),
        "unigrams": unigrams,
        "bigrams": bigrams,
        "phrases": phrases,
    }


def match_count(
    text: str,
    match_sets: Dict[str, set],
    phrases: bool = True,
) -> Dict[str, Any]:
    """
    Generates the count dictionary with matched counts of unigrams, bigrams, and phrases.

    Args:
        text (str): The text to analyze.
        match_sets (Dict[str, set]): Dictionary containing sets of unigrams, bigrams, and phrases.
        phrases (bool, optional): Whether to count phrases. Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary containing counts and statistics.

    Example:
        >>> from centralized_nlp_package.text_processing import match_count
        >>> text = "I am very happy and extremely joyful."
        >>> matches = {"unigrams": {"happy"}, "bigrams": {"very_happy"}, "phrases": {"extremely joyful"}}
        >>> counts = match_count(text, matches)
        >>> print(counts)
        {'original': {'happy', 'very_happy', 'extremely joyful'}, ...}
    """
    unigrams = tokenize_and_lemmatize_text(text)
    bigrams = ["_".join(g) for g in generate_ngrams(unigrams, 2)]

    # Initialize count dictionaries per label
    count_dict = {
        label: {match: 0 for match in match_set["unigrams"].union(match_set["bigrams"])}
        for label, match_set in match_sets.items()
    }

    unigram_count = {label: 0 for label in match_sets.keys()}
    bigram_count = {label: 0 for label in match_sets.keys()}

    if phrases:
        phrase_count = {label: 0 for label in match_sets.keys()}
    total_count = {label: 0 for label in match_sets.keys()}

    # Iterate over each label and its corresponding match sets
    for label, match_set in match_sets.items():
        # Count unigrams
        for word in unigrams:
            if word in match_set["unigrams"]:
                count_dict[label][word] += 1
                unigram_count[label] += 1

        # Count bigrams
        for bigram in bigrams:
            if bigram in match_set["bigrams"]:
                count_dict[label][bigram] += 1
                bigram_count[label] += 1

        # Count phrases as presence (1 if present, else 0)
        if phrases:
            for phrase in match_set.get("phrases", set()):
                if phrase.lower() in text.lower():
                    phrase_count[label] += 1
            # Total count per label
            total_count[label] = (
                unigram_count[label] + bigram_count[label] + phrase_count[label]
            )
        else:
            # Total count without phrases
            total_count[label] = unigram_count[label] + bigram_count[label]

    # Construct the return dictionary
    if phrases:
        ret = {
            label: {
                "uni": unigram_count[label],
                "bi": bigram_count[label],
                "phrase": phrase_count[label],
                "total": total_count[label],
                "stats": count_dict[label],
            }
            for label in match_sets.keys()
        }
    else:
        ret = {
            label: {
                "uni": unigram_count[label],
                "bi": bigram_count[label],
                "total": total_count[label],
                "stats": count_dict[label],
            }
            for label in match_sets.keys()
        }

    # Add additional metadata
    ret["len"] = len(unigrams)
    ret["raw_len"] = len(text.split(" "))
    ret["filt"] = text  # Optional: Remove if not necessary

    logger.debug(f"Match counts: {ret}")

    return ret


def merge_counts(counts: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Merges multiple count dictionaries into a single dictionary.

    Args:
        counts (List[Dict[str, int]]): List of count dictionaries.

    Returns:
        Dict[str, int]: Merged count dictionary.

    Example:
        >>> from centralized_nlp_package.text_processing import merge_counts
        >>> counts = [{"happy": 2}, {"happy": 3, "joyful": 1}]
        >>> merged = merge_counts(counts)
        >>> print(merged)
        {'happy': 5, 'joyful': 1}
    """
    try:
        merged = Counter()
        for count in counts:
            merged += Counter(count)
        if not merged:
            return {"NO_MATCH": 1}
        return dict(merged)
    except Exception as e:
        logger.error(f"Error merging counts: {e}")
        return {"ERROR": 1}


def calculate_sentence_score(
    indicators: List[int],
    weights: List[int],
    apply_weight: bool = True,
) -> Optional[float]:
    """
    Calculates the sentence score based on provided indicators and weights.

    Args:
        indicators (List[int]): List of relevant sentence indicators.
        weights (List[int]): List of weights or counts.
        apply_weight (bool, optional): Whether to apply weighting. Defaults to True.

    Returns:
        Optional[float]: Calculated sentence score or None.

    Example:
        >>> from centralized_nlp_package.text_processing import calculate_sentence_score
        >>> indicators = [1, 0, 1]
        >>> weights = [2, 3, 4]
        >>> score = calculate_sentence_score(indicators, weights)
        >>> print(score)
        3.0
    """
    length = len(indicators)
    if length == 0 or length != len(weights):
        logger.warning("Indicators and weights must be of the same non-zero length.")
        return None

    num_relevant = len([x for x in indicators if x > 0])
    if num_relevant == 0:
        logger.warning("No relevant indicators found.")
        return None

    if apply_weight:
        score = np.dot(indicators, weights) / num_relevant
    else:
        binary_indicators = [1 if x > 0 else 0 for x in indicators]
        score = np.dot(binary_indicators, weights) / num_relevant

    logger.debug(f"Calculated sentence score: {score}")
    return score

def netscore(
    counts: List[int],
    indicators: List[int],
) -> Optional[float]:
    """
    Calculates the net score based on provided counts and indicators.

    Args:
        counts (List[int]): List of counts.
        indicators (List[int]): List of indicators.

    Returns:
        Optional[float]: Calculated net score or None.

    Example:
        >>> from centralized_nlp_package.text_processing import netscore
        >>> counts = [1, 2, 0]
        >>> indicators = [1, 0, 1]
        >>> score = netscore(counts, indicators)
        >>> print(score)
        1.0
    """
    length = len(counts)
    if length == 0 or length != len(indicators):
        logger.warning("Counts and indicators must be of the same non-zero length.")
        return None

    num_relevant = len([x for x in counts if x > 0])
    if num_relevant == 0:
        logger.warning("No relevant counts found.")
        return None

    score = np.dot([1 if x > 0 else 0 for x in counts], indicators)
    logger.debug(f"Calculated net score: {score}")
    return score



def generate_match_count(
    df: pd.DataFrame,
    word_set_dict: Dict[str, Dict[str, set]],
) -> pd.DataFrame:
    """
    Generates the match count using the topics data.

    Args:
        df (pd.DataFrame): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.

    Returns:
        pd.DataFrame: Updated DataFrame with match counts.

    Example:
        >>> from centralized_nlp_package.text_processing import generate_match_count
        >>> import pandas as pd
        >>> data = {'section1': ["I love this product", "This is bad"]}
        >>> df = pd.DataFrame(data)
        >>> word_sets = {"section1": {"unigrams": {"love", "bad"}, "bigrams": set(), "phrases": set()}}
        >>> updated_df = generate_match_count(df, word_sets)
        >>> print(updated_df["matches_section1"])
        0    {'love': 1}
        1    {'bad': 1}
        Name: matches_section1, dtype: object
    """
    for label in config.lib_config.psycholinguistics.filt_sections:
        match_set = word_set_dict.get(label, {})
        df[f"matches_{label}"] = df[label].apply(
            lambda x: [match_count(sent, match_set, phrases=False) for sent in x], 
            meta=(f"matches_{label}", "object")
        )
    logger.info("Generated match counts for all sections.")
    return df


def generate_topic_statistics(
    df: pd.DataFrame,
    word_set_dict: Dict[str, Dict[str, set]],
) -> pd.DataFrame:
    """
    Generates new columns with topic total count statistics.

    Args:
        df (pd.DataFrame): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.

    Returns:
        pd.DataFrame: Updated DataFrame with topic statistics.

    Example:
        >>> from centralized_nlp_package.text_processing import generate_topic_statistics
        >>> import pandas as pd
        >>> data = {'matches_section1': [{"uni": 1, "bi": 0, "total": 1, "stats": {"love": 1}}, {"uni": 1, "bi": 0, "total": 1, "stats": {"bad": 1}}]}
        >>> df = pd.DataFrame(data)
        >>> word_sets = {"section1": {"unigrams": {"love", "bad"}, "bigrams": set(), "phrases": set()}}
        >>> updated_df = generate_topic_statistics(df, word_sets)
        >>> print(updated_df.columns)
        Index(['LEN_section1', 'RAW_LEN_section1', 'love_TOTAL_section1', 'bad_TOTAL_section1', 'love_STATS_section1', 'bad_STATS_section1', 'NUM_SENTS_section1'], dtype='object')
    """
    for label in config.lib_config.psycholinguistics.filt_sections:
        df[f"LEN_{label}"] = df[f"matches_{label}"].apply(
            lambda x: [calc['len'] for calc in x]
        )

        df[f"RAW_LEN_{label}"] = df[f"matches_{label}"].apply(
            lambda x: [calc['raw_len'] for calc in x]
        )

        for topic in word_set_dict.keys():
            df[f"{topic}_TOTAL_{label}"] = df[f"matches_{label}"].apply(
                lambda x: [calc[topic]['total'] for calc in x]
            )
            df[f"{topic}_STATS_{label}"] = df[f"matches_{label}"].apply(
                lambda x: merge_counts([calc[topic]['stats'] for calc in x])
            )
            df[f"{topic}_STATS_LIST_{label}"] = df[f"matches_{label}"].apply(
                lambda x: [dict(calc[topic]['stats']) for calc in x]
            )
        df[f"NUM_SENTS_{label}"] = df[f"LEN_{label}"].apply(lambda x: len(x))

        df.drop(columns=[f"matches_{label}"], inplace=True)
    logger.info("Generated topic statistics for all sections.")
    return df

def generate_sentence_relevance_score(
    df: pd.DataFrame,
    word_set_dict: Dict[str, Dict[str, set]],
) -> pd.DataFrame:
    """
    Generates relevance scores for each sentence.

    Args:
        df (pd.DataFrame): DataFrame to process.
        word_set_dict (Dict[str, Dict[str, set]]): Dictionary of word sets for each label.

    Returns:
        pd.DataFrame: Updated DataFrame with sentence relevance scores.

    Example:
        >>> from centralized_nlp_package.text_processing import generate_sentence_relevance_score
        >>> import pandas as pd
        >>> data = {'SENT_LABELS_section1': [[1, 0, 1], [0, 1, 0]], 'love_TOTAL_section1': [[1, 0, 1], [0, 1, 0]]}
        >>> df = pd.DataFrame(data)
        >>> word_sets = {"section1": {"unigrams": {"love"}, "bigrams": set(), "phrases": set()}}
        >>> updated_df = generate_sentence_relevance_score(df, word_sets)
        >>> print(updated_df["love_SENT_section1"])
        0    0.5
        1    0.0
        Name: love_SENT_section1, dtype: float64
    """
    for label in config.lib_config.psycholinguistics.filt_sections:
        df[f"SENT_{label}"] = df[f"SENT_LABELS_{label}"].apply(
            lambda x: float(np.sum(x) / len(x)) if len(x) > 0 else None
        )
        df[f"NET_SENT_{label}"] = df[f"SENT_LABELS_{label}"].apply(
            lambda x: np.sum(x) if len(x) > 0 else None
        )

        for topic in word_set_dict.keys():
            df[f"{topic}_RELEVANCE_{label}"] = df[f"{topic}_TOTAL_{label}"].apply(
                lambda x: len([a for a in x if a > 0]) / len(x) if len(x) > 0 else None
            )
            df[f"{topic}_SENT_{label}"] = df.apply(
                lambda row: calculate_sentence_score(
                    row[f"{topic}_TOTAL_{label}"],
                    row[f"SENT_LABELS_{label}"],
                    apply_weight=False
                ),
                axis=1
            )

            df[f"{topic}_SENT_REL_{label}"] = df.apply(
                lambda row: float(row[f"{topic}_RELEVANCE_{label}"] * row[f"{topic}_SENT_{label}"]) if row[f"{topic}_SENT_{label}"] else None,
                axis=1
            )

            df[f"{topic}_SENT_WEIGHT_{label}"] = df.apply(
                lambda row: calculate_sentence_score(
                    [sum(1 for val in stat.values() if val > 0) for stat in row[f"{topic}_STATS_LIST_{label}"]],
                    row[f"SENT_LABELS_{label}"],
                    apply_weight=True
                ),
                axis=1
            )
            df[f"{topic}_SENT_WEIGHT_REL_{label}"] = df.apply(
                lambda row: float(row[f"{topic}_RELEVANCE_{label}"] * row[f"{topic}_SENT_WEIGHT_{label}"]) if row[f"{topic}_SENT_WEIGHT_{label}"] else None,
                axis=1
            )

            df[f"{topic}_NET_SENT_{label}"] = df.apply(
                lambda row: netscore(row[f"{topic}_TOTAL_{label}"], row[f"SENT_LABELS_{label}"]),
                axis=1
            )
    logger.info("Generated sentence relevance scores for all sections.")
    return df