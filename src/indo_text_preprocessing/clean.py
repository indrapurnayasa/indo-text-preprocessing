"""Pembersihan teks: lowercase, URL, emoji, angka, tanda baca, whitespace."""

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[#@]\w+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "]+"
)


def lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Hapus mention/hashtag (@user, #tag) — umum di data media sosial."""
    return _MENTION_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_all(
    text: str,
    *,
    remove_url: bool = True,
    remove_mention: bool = True,
    remove_emoji: bool = True,
) -> str:
    """Jalankan seluruh langkah pembersihan sekaligus."""
    if remove_url:
        text = remove_urls(text)
    if remove_mention:
        text = remove_mentions(text)
    if remove_emoji:
        text = remove_emojis(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = lowercase(text)
    return remove_extra_whitespace(text)