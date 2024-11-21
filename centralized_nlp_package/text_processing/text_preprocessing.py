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
)
from centralized_nlp_package.utils import FilesNotLoadedException



def initialize_spacy(
    model: str = "en_core_web_sm",
    max_length: int = 1000000000,
    exclude_stop_words: Optional[List[str]] = ["bottom", "top", "Bottom", "Top", "call"]
) -> spacy.Language:
    """
    Initializes and configures the SpaCy language model with custom settings.
    
    This function loads a specified SpaCy model, adjusts its maximum processing length, 
    and customizes its stop words by excluding a predefined or user-specified list of words.
    
    Args:
        model (str, optional): The name of the SpaCy model to load. Defaults to "en_core_web_sm".
            Example models include "en_core_web_sm", "en_core_web_md", "en_core_web_lg", etc.
        max_length (int, optional): The maximum number of characters the SpaCy model will process. 
            Setting this to a higher value allows processing of larger texts. Defaults to 1000000000.
        exclude_stop_words (Optional[List[str]], optional): A list of stop words to exclude from the model's 
            default stop words set. This allows for customization of what words are considered insignificant 
            during text processing. Defaults to ["bottom", "top", "Bottom", "Top", "call"].
    
    Returns:
        spacy.Language: The configured SpaCy language model ready for text processing.
    
    Raises:
        FilesNotLoadedException: If there is an error loading additional stop words.
        Exception: If there is a general error initializing the SpaCy model.
    
    Example:
        >>> from centralized_nlp_package.text_processing import initialize_spacy
        >>> nlp = initialize_spacy(
        ...     model="en_core_web_md",
        ...     max_length=2000000000,
        ...     exclude_stop_words=["example", "test"]
        ... )
        >>> doc = nlp("This is a sample sentence for testing the SpaCy model.")
        >>> print([token.text for token in doc])
        ['This', 'is', 'a', 'sample', 'sentence', 'for', 'testing', 'SpaCy', 'model', '.']
    """
    logger.info(f"Loading SpaCy model: {model}")
    try:
        nlp = spacy.load(
            model, disable=["parser"]
        )
        if exclude_stop_words:
            nlp.Defaults.stop_words -= set(exclude_stop_words)
            logger.debug(f"Excluded stop words: {exclude_stop_words}")
        nlp.max_length = max_length
        logger.info("SpaCy model initialized successfully.")
        return nlp
    except FilesNotLoadedException as e:
        logger.error(f"Failed to load additional stop words: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error initializing SpaCy model: {e}")
        raise

def remove_unwanted_phrases_and_validate(
    sentence: str,
    min_word_length: int = 5,
    cleanup_phrases: Optional[List[str]] = None,
    greeting_phrases: Optional[List[str]] = None
) -> Optional[str]:
    """
    Cleans the sentence by removing unwanted and greeting phrases, then validates its length.

    Args:
        sentence (str): The sentence to process.
        min_word_length (int, optional): Minimum word count required. Defaults to 5.
        cleanup_phrases (Optional[List[str]], optional): Phrases to remove. 
            Defaults to config values:
                ["Thank you", "thank you", "thanks", "Thanks", 
                 "earnings call", "earnings release", "earnings conference"]
        greeting_phrases (Optional[List[str]], optional): Greeting phrases to check. 
            Defaults to config values:
                ["good morning", "good afternoon", "good evening"]

    Returns:
        Optional[str]: Cleaned sentence or 'None' if invalid.

    Raises:
        TypeError: If 'sentence' is not a string.
        Exception: For unexpected processing errors.

    Example:
        >>> cleaned = remove_unwanted_phrases_and_validate(
        ...     sentence="Thank you! I love this product.",
        ...     min_word_length=3
        ... )
        >>> print(cleaned)
        'I love this product.'

        >>> cleaned = remove_unwanted_phrases_and_validate(
        ...     sentence="Good morning! Love it.",
        ...     min_word_length=3
        ... )
        >>> print(cleaned)
        None
    """
    logger.debug("Cleaning sentence.")
    
    # Assign default phrases from config if not provided
    if cleanup_phrases is None:
        cleanup_phrases = config.lib_config.text_processing.cleanup_phrases
        logger.debug(f"Using default cleanup phrases: {cleanup_phrases}")
    
    if greeting_phrases is None:
        greeting_phrases = config.lib_config.text_processing.greeting_phrases
        logger.debug(f"Using default greeting phrases: {greeting_phrases}")
    
    # Remove cleanup phrases
    for phrase in cleanup_phrases:
        sentence = sentence.replace(phrase, "")
    sentence = sentence.strip()
    
    # Validate word count
    word_count = len(sentence.split())
    if word_count < min_word_length:
        logger.debug("Sentence below minimum word length. Skipping.")
        return None
    
    # Check for greeting phrases
    if any(greet.lower() in sentence.lower() for greet in greeting_phrases):
        logger.debug("Greeting phrase detected. Skipping.")
        return None
    
    logger.debug(f"Cleaned sentence: {sentence}")
    return sentence if sentence else None


def tokenize_and_lemmatize_text(
    doc: str,
    nlp: spacy.Language,
    pos_exclude: Optional[List[str]] = None,
    ent_type_exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Tokenizes and lemmatizes the document text, excluding stop words, punctuation, numbers,
    and optionally excluding specific parts of speech and entity types.

    Args:
        doc (str): The input text to tokenize.
        nlp (spacy.Language): Initialized SpaCy model.
        pos_exclude (Optional[List[str]]): List of part-of-speech tags to exclude. Defaults to None.
            For available POS tags, refer to SpaCy's documentation:
            https://spacy.io/usage/linguistic-features#pos-tagging
        ent_type_exclude (Optional[List[str]]): List of named entity types to exclude. Defaults to None.
            For available entity types, refer to SpaCy's documentation:
            https://www.restack.io/p/entity-recognition-answer-spacy-entity-types-list-cat-ai


    Returns:
        List[str]: List of lemmatized and filtered tokens.

    Example:
        >>> nlp = spacy.load("en_core_web_sm")
        >>> tokens = tokenize_and_lemmatize_text("I am loving the new features!", nlp)
        >>> print(tokens)
        ['love', 'new', 'feature']
        
        >>> tokens = tokenize_and_lemmatize_text(
        ...     "Apple is looking at buying U.K. startup for $1 billion",
        ...     nlp,
        ...     pos_exclude=["VERB"],
        ...     ent_type_exclude=["ORG"]
        ... )
        >>> print(tokens)
        ['apple', 'look', 'buy', 'u.k.', 'startup', 'billion']
    """
    tokens = []
    for token in nlp(doc):
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if pos_exclude and token.pos_ in pos_exclude:
            continue
        if ent_type_exclude and token.ent_type_ in ent_type_exclude:
            continue
        tokens.append(token.lemma_.lower())

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
