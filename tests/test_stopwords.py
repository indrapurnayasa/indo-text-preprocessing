"""Tests untuk stopwords.py."""

from indo_text_preprocessing.stopwords import get_stopwords, remove_stopwords


def test_stopwords_loaded():
    stops = get_stopwords()
    assert len(stops) > 500
    assert "adalah" in stops
    assert "yang" in stops


def test_remove_basic():
    assert remove_stopwords("aku tidak pergi ke pasar") == "pergi pasar"


def test_keep_sentence():
    out = remove_stopwords("saya sedang belajar bahasa indonesia")
    assert "belajar" in out


def test_custom_words():
    out = remove_stopwords("kucing hitam lucu banget", custom_words=["banget"])
    assert out == "kucing hitam lucu"


def test_keep_words():
    out = remove_stopwords("makan tidak pergi", keep_words=["tidak"])
    assert out == "makan tidak pergi"


def test_custom_and_keep():
    out = remove_stopwords(
        "aku banget tidak pergi",
        custom_words=["banget"],
        keep_words=["banget", "tidak"],
    )
    assert out == "banget tidak pergi"


def test_empty():
    assert remove_stopwords("") == ""