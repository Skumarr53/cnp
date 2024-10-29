
# centralized_nlp_package/text_processing/text_utils.py

import re
from typing import List, Tuple, Optional, Dict, Iterator, Union
from pathlib import Path

import spacy
import numpy as np
from loguru import logger

from centralized_nlp_package import config
from centralized_nlp_package.common_utils import load_content_from_txt
from centralized_nlp_package.utils import FilesNotLoadedException

# def check_datatype()
def validate_and_format_text(text_input: Optional[Union[str, List[str]]]) -> Optional[str]:
    """
    Validates and formats the input text by ensuring it's a non-empty string.
    If the input is a list of strings, joins them into a single string.

    Args:
        text_input (Optional[Union[str, List[str]]]): The input text or list of texts to validate.

    Returns:
        Optional[str]: Joined and stripped text if valid, else None.

    Example:
        >>> from centralized_nlp_package.text_processing import validate_and_format_text
        >>> validate_and_format_text("  Hello World!  ")
        'Hello World!'
        >>> validate_and_format_text(["Hello", "World"])
        'Hello World'
        >>> validate_and_format_text([])
        None
    """
    if isinstance(text_input, list):
        joined_text = ' '.join(text_input).strip()
    elif isinstance(text_input, str):
        joined_text = text_input.strip()
    else:
        joined_text = None

    if joined_text:
        logger.debug("Input text is valid and formatted.")
        return joined_text
    else:
        logger.warning("Input text is invalid or empty.")
        return None

## def generate_ngrams()
def generate_ngrams(input_list: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens.

    Args:
        input_list (List[str]): List of tokens.
        n (int): The number of tokens in each n-gram.

    Yields:
        Iterator[Tuple[str, ...]]: An iterator over n-grams as tuples.

    Example:
        >>> from centralized_nlp_package.text_processing import generate_ngrams
        >>> list(generate_ngrams(['I', 'love', 'coding'], 2))
        [('I', 'love'), ('love', 'coding')]
    """
    if n < 1:
        logger.warning("n must be at least 1.")
        return
    return zip(*[input_list[i:] for i in range(n)])


# def load_list_from_txt()
def load_set_from_txt(file_path: str, is_lower: bool = True) -> set:
    """
    Reads the content of a text file and returns it as a set of lines.

    Args:
        file_path (str): The path to the text file.
        is_lower (bool, optional): If True, converts the content to lowercase. Defaults to True.

    Returns:
        set: A set of lines from the text file.

    Raises:
        FilesNotLoadedException: If there is an error reading the file.

    Example:
        >>> from centralized_nlp_package.text_processing import load_set_from_txt
        >>> word_set = load_set_from_txt("data/positive_words.txt")
        >>> print(word_set)
        {'happy', 'joyful', 'delighted'}
    """
    try:
        content = load_content_from_txt(file_path)
        if is_lower:
            content = content.lower()
        words_set = set(filter(None, (line.strip() for line in content.split('\n'))))
        logger.debug(f"Loaded set from {file_path} with {len(words_set)} entries.")
        return words_set
    except Exception as e:
        logger.error(f"Error loading set from {file_path}: {e}")
        raise FilesNotLoadedException(f"Error loading set from {file_path}: {e}") from e


def expand_contractions(text):
    """    	
    Expands contractions in the input text based on a contraction dictionary.

    Args:
        text (str): The input text containing contractions.

    Returns:
        str: Text with expanded contractions.

    Example:
        >>> from centralized_nlp_package.text_processing import expand_contractions
        >>> contraction_map = {"can't": "cannot", "I'm": "I am"}
        >>> expand_contractions("I can't go.")
        'I cannot go.'
    
    """
    # TODO: add contraction words to config or txt file then load
    contraction_dict = None
    for word in text.split():
      if word.lower() in contraction_dict:
        text = text.replace(word, contraction_dict[word.lower()])
    return text

# def word_tokenizer()
def tokenize_text(text: str, spacy_tokenizer: spacy.Language) -> List[str]:
    """
    Tokenizes the text and performs lemmatization, excluding stop words.

    Args:
        text (str): The input text to tokenize.
        spacy_tokenizer (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        List[str]: A list of lemmatized and filtered words.

    Example:
        >>> from centralized_nlp_package.text_processing import tokenize_text
        >>> nlp = spacy.load("en_core_web_sm")
        >>> tokenize_text("I am loving the new features!", nlp)
        ['love', 'new', 'feature']
    """
    stop_words_path = Path(config.lib_config.paths.model_artifacts.path) / config.lib_config.filenames.stop_words_flnm
    try:
        stop_words_set = load_set_from_txt(str(stop_words_path), is_lower=True)
    except FilesNotLoadedException as e:
        logger.error(f"Stop words could not be loaded: {e}")
        raise e

    doc = spacy_tokenizer(text.lower())
    token_lemmatized = [token.lemma_ for token in doc]
    filtered_words = [word for word in token_lemmatized if word not in stop_words_set and word.isalpha()]
    logger.debug(f"Tokenized and filtered words. {len(filtered_words)} words remaining.")
    return filtered_words

# def combine_sent
def combine_sentiment_scores(positive_count: int, negative_count: int) -> float:
    """
    Combines two sentiment scores into a single score.

    Args:
        positive_count (int): Positive sentiment count.
        negative_count (int): Negative sentiment count.

    Returns:
        float: Combined sentiment score. Returns 0.0 if both counts are zero.

    Example:
        >>> from centralized_nlp_package.text_processing import combine_sentiment_scores
        >>> combine_sentiment_scores(5, 3)
        0.25
        >>> combine_sentiment_scores(0, 0)
        0.0
    """
    if (positive_count + negative_count) == 0:
        return 0.0
    combined_score = (positive_count - negative_count) / (positive_count + negative_count)
    logger.debug(f"Combined sentiment score: {combined_score}")
    return combined_score


def load_syllable_counts(file_path: str) -> Dict[str, int]:
    """
    Reads a file containing words and their syllable counts, and returns a dictionary.

    Args:
        file_path (str): The path to the text file.

    Returns:
        Dict[str, int]: A dictionary where keys are words and values are their syllable counts.

    Raises:
        FilesNotLoadedException: If the file is not found or has an invalid format.

    Example:
        >>> from centralized_nlp_package.text_processing import load_syllable_counts
        >>> syllables = load_syllable_counts("data/syllable_counts.txt")
        >>> syllables['beautiful']
        3
    """
    syllables: Dict[str, int] = {}
    try:
        with open(file_path, 'r', encoding="utf-8") as fs_pos_words:
            for line in fs_pos_words:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, count = parts
                    syllables[word.lower()] = int(count)
                else:
                    logger.warning(f"Ignoring invalid line in syllable count file: {line.strip()}")
        logger.debug(f"Loaded syllable counts from {file_path} with {len(syllables)} entries.")
        return syllables
    except FileNotFoundError as ex:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(f"File not found: {file_path}") from ex
    except ValueError as ve:
        logger.error(f"Value error in syllable count file: {ve}")
        raise FilesNotLoadedException(f"Invalid format in syllable count file: {ve}") from ve
