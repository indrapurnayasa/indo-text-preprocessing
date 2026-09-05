"""Tests untuk clean.py."""

import pytest
from indo_text_preprocessing.clean import (
    clean_all,
    lowercase,
    remove_emojis,
    remove_extra_whitespace,
    remove_mentions,
    remove_numbers,
    remove_punctuation,
    remove_urls,
)


class TestLowercase:
    def test_basic(self):
        assert lowercase("HALO Dunia") == "halo dunia"

    def test_empty(self):
        assert lowercase("") == ""


class TestRemoveUrls:
    def test_http(self):
        assert remove_urls("cek https://example.com/x ya") == "cek   ya"

    def test_www(self):
        assert remove_urls("buka www.google.com").strip() == "buka"

    def test_no_url(self):
        assert remove_urls("tidak ada link") == "tidak ada link"


class TestRemoveMentions:
    def test_mention(self):
        assert remove_mentions("halo @budi apa kabar") == "halo   apa kabar"

    def test_hashtag(self):
        assert remove_mentions("seru #liburan dulu") == "seru   dulu"


class TestRemoveEmojis:
    def test_emoji(self):
        assert remove_emojis("bagus 😂🔥") == "bagus  "

    def test_no_emoji(self):
        assert remove_emojis("teks biasa") == "teks biasa"


class TestRemoveNumbers:
    def test_numbers(self):
        assert remove_numbers("tahun 2024 bulan 5") == "tahun   bulan  "


class TestRemovePunctuation:
    def test_basic(self):
        assert remove_punctuation("halo, dunia!") == "halo  dunia "

    def test_keep_alnum(self):
        assert remove_punctuation("kata-kata") == "kata kata"


class TestRemoveExtraWhitespace:
    def test_extra(self):
        assert remove_extra_whitespace("  halo   dunia  ") == "halo dunia"

    def test_tabs_newlines(self):
        assert remove_extra_whitespace("a\t\nb") == "a b"


class TestCleanAll:
    def test_full(self):
        text = "CEK https://x.co/1 @budi 😍 harga 15rb!!! mantap"
        assert clean_all(text) == "cek harga rb mantap"

    def test_flags(self):
        text = "cek https://x.co @budi 123"
        assert clean_all(text, remove_url=False, remove_mention=False) == "cek https x co budi"


@pytest.mark.parametrize(
    "fn,text,expected",
    [
        (lowercase, "", ""),
        (remove_numbers, "", ""),
        (remove_punctuation, "", ""),
        (remove_extra_whitespace, "", ""),
    ],
)
def test_empty_string(fn, text, expected):
    assert fn(text) == expected