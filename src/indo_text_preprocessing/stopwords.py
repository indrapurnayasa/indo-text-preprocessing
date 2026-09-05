"""Penghapusan stopwords Bahasa Indonesia."""

from __future__ import annotations

from ._data import load_words

_STOPWORDS = load_words("stopwords_id.txt")


def get_stopwords() -> frozenset:
    """Set stopwords bawaan."""
    return _STOPWORDS


def remove_stopwords(
    text: str,
    custom_words: list[str] | None = None,
    keep_words: list[str] | None = None,
) -> str:
    """Hapus stopwords dari teks.

    custom_words: stopwords tambahan milik pengguna.
    keep_words: kata yang dikecualikan dari penghapusan.
    """
    remove = _STOPWORDS
    if custom_words:
        remove = remove | frozenset(custom_words)
    if keep_words:
        remove = remove - frozenset(keep_words)
    return " ".join(w for w in text.split() if w not in remove)