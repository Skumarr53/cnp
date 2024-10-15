# centralized_nlp_package/preprocessing/ngram_utils.py

import numpy as np
from typing import List, Tuple, Iterator
from gensim.models import Word2Vec, Phrases
from centralized_nlp_package.preprocessing.text_preprocessing import tokenize_text

def find_ngrams(input_list: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    """
    Generates n-grams from a list of tokens.

    Args:
        input_list (List[str]): List of tokens.
        n (int): The number of tokens in each n-gram.

    Yields:
        Iterator[Tuple[str, ...]]: An iterator over n-grams as tuples.
    """
    return zip(*[input_list[i:] for i in range(n)])

