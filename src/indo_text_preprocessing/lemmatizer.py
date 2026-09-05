"""Lemmatizer Bahasa Indonesia — rule-based di atas stemmer.

Lemma = bentuk kata yang ada di KBBI. Dalam NLP Bahasa Indonesia praktis,
lemmatizer ≈ stemmer + override kata tak beraturan yang konfix stripping
salah tangani secara semantik.

# ponytail: rule-based override saja, tanpa POS tagger. Upgrade ke POS-aware
# lemmatizer (spaCy id_core_news_sm) bila akurasi konteks jadi kebutuhan.
"""

from .stemmer import stem as _stem, stem_batch as _stem_batch
from importlib import resources
import json
from functools import lru_cache


@lru_cache(maxsize=None)
def _irregular() -> dict:
    text = resources.files("indo_text_preprocessing").joinpath("data/irregular_words.json").read_text(encoding="utf-8")
    return json.loads(text)


def lemmatize(word: str, custom_map: dict | None = None) -> str:
    """Lemma satu kata: irregular override dulu, sisanya stemmer."""
    word = word.lower().strip()
    m = _irregular()
    if custom_map:
        m = {**m, **custom_map}
    return m.get(word) or _stem(word)


def lemmatize_batch(words: list[str], custom_map: dict | None = None) -> list[str]:
    return [lemmatize(w, custom_map) for w in words]