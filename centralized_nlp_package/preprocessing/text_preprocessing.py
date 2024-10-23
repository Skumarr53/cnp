# centralized_nlp_package/preprocessing/text_processing.py

import re
from typing import List, Tuple, Optional, Union, Dict
import spacy
from loguru import logger
from centralized_nlp_package import config
from centralized_nlp_package.text_processing.text_utils import (
    validate_and_format_text,
    expand_contractions,
    tokenize_text,
    load_set_from_txt,
)
from centralized_nlp_package.utils.exception import FilesNotLoadedException



def initialize_spacy() -> spacy.Language:
    """
    Initializes and configures the SpaCy model with custom settings.

    Returns:
        spacy.Language: Configured SpaCy model.

    Example:
        >>> nlp = initialize_spacy()
        >>> doc = nlp("This is a sample sentence.")
        >>> print([token.text for token in doc])
        ['This', 'is', 'a', 'sample', 'sentence', '.']
    """
    logger.info(f"Loading SpaCy model: {config.lib_config.preprocessing.spacy_model}")
    try:
        nlp = spacy.load(
            config.lib_config.preprocessing.spacy_model, disable=["parser"]
        )
        # Excluding financially relevant stopwords
        additional_stop_words = load_set_from_txt(
            str(
                config.lib_config.paths.model_artifacts.path
                / config.lib_config.filenames.additional_stop_words_flnm
            ),
            is_lower=True,
        )
        nlp.Defaults.stop_words -= additional_stop_words
        nlp.max_length = config.lib_config.preprocessing.max_length
        logger.info("SpaCy model initialized.")
        return nlp
    except FilesNotLoadedException as e:
        logger.error(f"Failed to load additional stop words: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error initializing SpaCy model: {e}")
        raise

def remove_unwanted_phrases_and_validate(sentence: str) -> Optional[str]:
    """
    Cleans the input sentence by removing unwanted phrases and validating its content.

    Args:
        sentence (str): The input sentence to process.

    Returns:
        Optional[str]: Cleaned sentence or None if it doesn't meet criteria.

    Example:
        >>> cleaned = remove_unwanted_phrases_and_validate("Hello! I love this product.")
        >>> print(cleaned)
        'I love this product.'
        >>> cleaned = remove_unwanted_phrases_and_validate("Hi! Love it.")
        >>> print(cleaned)
        None
    """
    logger.debug("Cleaning sentence.")
    # Remove specified phrases
    for phrase in config.lib_config.preprocessing.cleanup_phrases:
        sentence = sentence.replace(phrase, "")
    sentence = sentence.strip()

    # Check word count
    word_count = len(sentence.split())
    if word_count < config.lib_config.preprocessing.min_word_length:
        logger.debug("Sentence below minimum word length. Skipping.")
        return None

    # Remove greetings
    if any(
        greet in sentence.lower() for greet in config.lib_config.preprocessing.greeting_phrases
    ):
        logger.debug("Greeting phrase detected. Skipping.")
        return None

    logger.debug(f"Cleaned sentence: {sentence}")
    return sentence if sentence else None


def tokenize_and_lemmatize_text(doc: str, nlp: spacy.Language) -> List[str]:
    """
    Tokenizes and lemmatizes the document text, excluding stop words, punctuation, and numbers.

    Args:
        doc (str): The input text to tokenize.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        List[str]: List of lemmatized and filtered tokens.

    Example:
        >>> nlp = initialize_spacy()
        >>> tokens = tokenize_and_lemmatize_text("I am loving the new features!", nlp)
        >>> print(tokens)
        ['love', 'new', 'feature']
    """
    tokens = [
        token.lemma_.lower()
        for token in nlp(doc)
        if not token.is_stop and not token.is_punct and token.pos_ != "NUM"
    ]
    logger.debug(f"Tokenized document into {len(tokens)} tokens.")
    return tokens


def tokenize_matched_words(doc: str, nlp: spacy.Language) -> List[str]:
    """
    Tokenizes matched words by removing stop words, punctuation, and numbers. Prioritizes proper nouns and capitalized words.

    Args:
        doc (str): The input text to tokenize.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        List[str]: List of lemmatized and filtered tokens.

    Example:
        >>> nlp = initialize_spacy()
        >>> tokens = tokenize_matched_words("John loves New York!")
        >>> print(tokens)
        ['john', 'love', 'new', 'york']
    """
    ret = []
    for token in nlp(doc):
        if token.pos_ == "PROPN" or token.text[0].isupper():
            ret.append(token.text.lower())
            continue
        if not token.is_stop and not token.is_punct and token.pos_ != "NUM":
            ret.append(token.lemma_.lower())
    logger.debug(f"Tokenized matched words into {len(ret)} tokens.")
    return ret


def preprocess_text(
    text_input: Optional[Union[str, List[str]]], nlp: spacy.Language
) -> Tuple[Optional[str], List[str], int]:
    """
    Preprocesses the text by validating, cleaning, and tokenizing.

    Args:
        text_input (Optional[Union[str, List[str]]]): The input text or list of texts to preprocess.
        nlp (spacy.Language): Initialized SpaCy tokenizer.

    Returns:
        Tuple[Optional[str], List[str], int]: The preprocessed text, list of input words, and word count.

    Example:
        >>> nlp = initialize_spacy()
        >>> cleaned_text, tokens, count = preprocess_text("I can't believe it!", nlp)
        >>> print(cleaned_text)
        'i cannot believe it!'
        >>> print(tokens)
        ['cannot', 'believe']
        >>> print(count)
        2
    """
    text = validate_and_format_text(text_input)
    if text:
        cleaned_text = clean_text(text)
        input_words = tokenize_text(cleaned_text, nlp)
        word_count = len(input_words)
        logger.debug("Preprocessed single text input.")
        return cleaned_text, input_words, word_count
    else:
        logger.warning("Preprocessing failed due to invalid input.")
        return None, [], 0


def preprocess_text_list(
    text_list: List[str], nlp: spacy.Language
) -> Tuple[List[str], List[List[str]], List[int]]:
    """
    Preprocesses a list of texts by validating, cleaning, and tokenizing each.

    Args:
        text_list (List[str]): The list of texts to preprocess.
        nlp (spacy.Language): Initialized SpaCy model.

    Returns:
        Tuple[List[str], List[List[str]], List[int]]: Cleaned texts, list of token lists, and word counts.

    Example:
        >>> nlp = initialize_spacy()
        >>> texts = ["I love this product!", "Hi!"]
        >>> cleaned_texts, tokens_list, counts = preprocess_text_list(texts, nlp)
        >>> print(cleaned_texts)
        ['i love this product!', '']
        >>> print(tokens_list)
        [['love', 'product']]
        >>> print(counts)
        [3, 0]
    """
    logger.debug("Preprocessing list of texts.")
    final_text_list = []
    input_word_list = []
    word_count_list = []

    for text in text_list:
        cleaned = remove_unwanted_phrases_and_validate(text)
        if cleaned:
            tokens = tokenize_text(cleaned, nlp)
            final_text_list.append(cleaned)
            input_word_list.append(tokens)
            word_count_list.append(len(tokens))
        else:
            final_text_list.append("")
            input_word_list.append([])
            word_count_list.append(0)

    logger.debug(f"Preprocessed {len(text_list)} texts.")
    return final_text_list, input_word_list, word_count_list

def clean_text(text: str) -> str:
    """
    Cleans the input text by expanding contractions, removing unwanted characters, and normalizing spaces.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.

    Example:
        >>> clean_text("I can't go to the party.")
        "i cannot go to the party"
    """
    # Expand contractions
    text = expand_contractions(re.sub("’", "'", text))
    # Remove quotation marks
    text = text.replace('"', "")
    # Remove single-character words
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # Replace non-alphabetic characters with space
    text = re.sub("[^a-zA-Z]", " ", text)
    # Normalize multiple spaces to single space
    text = re.sub("\s+", " ", text)
    # Strip leading and trailing spaces
    text = text.strip().lower()
    logger.debug("Cleaned the text.")
    return text
