"""Vectorizer BoW + TF-IDF — from scratch, numpy.

Formula mengikuti sklearn:
- CountVectorizer: hitung frekuensi kata per dokumen (dense numpy).
- TfidfVectorizer: tf = count/len(doc); idf = ln((1+N)/(1+df)) + 1 (smooth_idf);
  hasil dinormalisasi L2.

# ponytail: dense matrix + tokenizer split-whitespace. Sparse (scipy) & custom
# tokenizer (ngram, regex) bila corpus > 100k docs atau butuh ngram.
"""

import math
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class CountVectorizer:
    """Bag-of-words: fit → vocab, transform → matrix count (dense)."""

    def __init__(self, max_features: int | None = None, min_df: int = 1):
        self.vocabulary_: dict[str, int] = {}
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, corpus: list[str]) -> "CountVectorizer":
        df: dict[str, int] = {}
        for doc in corpus:
            for tok in set(_tokenize(doc)):
                df[tok] = df.get(tok, 0) + 1
        items = [(t, c) for t, c in df.items() if c >= self.min_df]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_features:
            items = items[: self.max_features]
        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted(items))}
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        m = np.zeros((len(corpus), len(self.vocabulary_)), dtype=np.int64)
        for i, doc in enumerate(corpus):
            for tok in _tokenize(doc):
                j = self.vocabulary_.get(tok)
                if j is not None:
                    m[i, j] += 1
        return m

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)


class TfidfVectorizer(CountVectorizer):
    """TF-IDF: smooth_idf + L2 normalisasi (sklearn-compatible)."""

    def fit(self, corpus: list[str]) -> "TfidfVectorizer":
        super().fit(corpus)
        n_docs = len(corpus)
        df = np.zeros(len(self.vocabulary_))
        for doc in corpus:
            for tok in set(_tokenize(doc)):
                j = self.vocabulary_.get(tok)
                if j is not None:
                    df[j] += 1
        # smooth_idf: ln((1+N)/(1+df)) + 1
        self.idf_ = np.log((1 + n_docs) / (1 + df)) + 1
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        counts = super().transform(corpus).astype(np.float64)
        tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        m = tf * self.idf_
        # L2 normalisasi per dokumen
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norm, 1e-12)