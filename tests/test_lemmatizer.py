"""Tests untuk lemmatizer.py."""

from indo_text_preprocessing.lemmatizer import lemmatize, lemmatize_batch


class TestLemmatize:
    def test_falls_back_to_stem(self):
        assert lemmatize("memakan") == "makan"

    def test_root_passthrough(self):
        assert lemmatize("makan") == "makan"

    def test_case_insensitive(self):
        assert lemmatize("MAKAN") == "makan"

    def test_custom_map_override(self):
        assert lemmatize("jalanjalan", custom_map={"jalanjalan": "jalan"}) == "jalan"

    def test_custom_map_fallback(self):
        # kata tidak di custom_map → stemmer
        assert lemmatize("memasak", custom_map={"xxx": "yyy"}) == "masak"


def test_batch():
    assert lemmatize_batch(["memakan", "makan"]) == ["makan", "makan"]


def test_batch_custom():
    out = lemmatize_batch(["aaa", "memakan"], custom_map={"aaa": "bbb"})
    assert out == ["bbb", "makan"]