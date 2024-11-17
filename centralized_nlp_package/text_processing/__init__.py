# centralized_nlp_package/preprocessing/__init__.py

from .text_preprocessing import (
    initialize_spacy,
    remove_unwanted_phrases_and_validate,
    tokenize_and_lemmatize_text,
    tokenize_matched_words,
    preprocess_text,
    preprocess_text_list,
    clean_text
)

from .text_analysis import (
    load_word_set,
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
    generate_sentence_relevance_score,
)
from .text_utils import (
    validate_and_format_text,
    generate_ngrams,
    load_set_from_txt,
    expand_contractions,
    tokenize_text,
    combine_sentiment_scores,
    load_syllable_counts,
)




__all__ = [
    'initialize_spacy',
    'remove_unwanted_phrases_and_validate',
    'tokenize_and_lemmatize_text',
    'tokenize_matched_words',
    'preprocess_text',
    'preprocess_text_list',
    'clean_text',

    # From text_analysis.py
    'load_word_set',
    'check_negation',
    'calculate_polarity_score',
    'polarity_score_per_section',
    'polarity_score_per_sentence',
    'is_complex',
    'fog_analysis_per_section',
    'fog_analysis_per_sentence',
    'tone_count_with_negation_check',
    'tone_count_with_negation_check_per_sentence',
    'get_match_set',
    'match_count',
    'merge_counts',
    'calculate_sentence_score',
    'netscore',
    'generate_match_count',
    'generate_topic_statistics',
    'generate_sentence_relevance_score',

    # From text_utils.py
    'validate_and_format_text',
    'generate_ngrams',
    'load_set_from_txt',
    'expand_contractions',
    'tokenize_text',
    'combine_sentiment_scores',
    'load_syllable_counts',
]
