"""Public API indo_text_preprocessing."""

from __future__ import annotations

from .clean import (
    clean_all,
    lowercase,
    remove_emojis,
    remove_extra_whitespace,
    remove_mentions,
    remove_numbers,
    remove_punctuation,
    remove_urls,
)
from .slang import get_slang_dict, replace_slang
from .stopwords import get_stopwords, remove_stopwords
from .lemmatizer import lemmatize, lemmatize_batch
from .pipeline import IndoPreprocessor
from .stemmer import stem, stem_batch
from .vectorizer import CountVectorizer, TfidfVectorizer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # clean
    "clean_all",
    "lowercase",
    "remove_emojis",
    "remove_extra_whitespace",
    "remove_mentions",
    "remove_numbers",
    "remove_punctuation",
    "remove_urls",
    # stopwords
    "get_stopwords",
    "remove_stopwords",
    # slang
    "get_slang_dict",
    "replace_slang",
    # lemmatizer
    "lemmatize",
    "lemmatize_batch",
    # stemmer
    "stem",
    "stem_batch",
    # vectorizer
    "CountVectorizer",
    "TfidfVectorizer",
    # pipeline
    "IndoPreprocessor",
]