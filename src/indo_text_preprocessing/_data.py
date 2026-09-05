"""Loader corpus internal package (dibundel di data/)."""

from importlib import resources
import json
from functools import lru_cache


@lru_cache(maxsize=None)
def load_words(filename: str) -> frozenset:
    """Muat satu kata per baris dari file teks corpus."""
    text = resources.files("indo_text_preprocessing").joinpath(f"data/{filename}").read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


@lru_cache(maxsize=None)
def load_json(filename: str) -> dict:
    """Muat mapping JSON dari corpus."""
    text = resources.files("indo_text_preprocessing").joinpath(f"data/{filename}").read_text(encoding="utf-8")
    return json.loads(text)