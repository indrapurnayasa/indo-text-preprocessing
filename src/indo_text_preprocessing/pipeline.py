"""Pipeline: rangkai semua langkah preprocessing."""

from __future__ import annotations

import numpy as np

from .clean import clean_all
from .lemmatizer import lemmatize
from .slang import replace_slang
from .stemmer import stem
from .stopwords import remove_stopwords
from .vectorizer import CountVectorizer, TfidfVectorizer


class IndoPreprocessor:
    """Pipeline configurable: clean → slang → stopwords → stem/lemma.

    Default: semua langkah aktif kecuali lemmatize (stem & lemma saling
    eksklusif — bila keduanya True, lemma menang).

    # ponytail: proses per dokumen serial. Multiprocessing bila corpus
    # > 100k docs dan profiling menunjukkan bottleneck.
    """

    def __init__(
        self,
        remove_url: bool = True,
        remove_mention: bool = True,
        remove_emoji: bool = True,
        replace_slang: bool = True,
        remove_stopwords: bool = True,
        stem: bool = True,
        lemmatize: bool = False,
        stopwords_kwargs: dict | None = None,
        slang_map: dict | None = None,
    ):
        self.remove_url = remove_url
        self.remove_mention = remove_mention
        self.remove_emoji = remove_emoji
        self.replace_slang = replace_slang
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.lemmatize = lemmatize
        self.stopwords_kwargs = stopwords_kwargs or {}
        self.slang_map = slang_map

    def preprocess(self, text: str) -> str:
        text = clean_all(
            text,
            remove_url=self.remove_url,
            remove_mention=self.remove_mention,
            remove_emoji=self.remove_emoji,
        )
        if self.replace_slang:
            text = replace_slang(text, mapping=self.slang_map)
        if self.remove_stopwords:
            text = remove_stopwords(text, **self.stopwords_kwargs)
        if self.lemmatize:
            return " ".join(lemmatize(w) for w in text.split())
        if self.stem:
            return " ".join(stem(w) for w in text.split())
        return text

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]

    def fit_transform(self, corpus: list[str], tfidf: bool = False) -> np.ndarray:
        cleaned = self.preprocess_batch(corpus)
        v = (TfidfVectorizer if tfidf else CountVectorizer)()
        return v.fit_transform(cleaned)