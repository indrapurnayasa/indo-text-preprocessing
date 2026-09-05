"""Penggantian kata slang/colloquial ke bentuk formal."""

from ._data import load_json

_SLANG = load_json("slang_id.json")


def get_slang_dict() -> dict:
    """Mapping slang bawaan (kata formal)."""
    return _SLANG


def replace_slang(text: str, mapping: dict | None = None) -> str:
    """Ganti slang dengan bentuk formal.

    mapping: dict tambahan milik pengguna, di-merge di atas corpus bawaan
    (mapping pengguna menang bila kunci bentrok).
    """
    m = {**_SLANG, **mapping} if mapping else _SLANG
    return " ".join(m.get(w, w) for w in text.split())