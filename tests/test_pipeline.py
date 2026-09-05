"""Tests untuk pipeline.py."""

import numpy as np

from indo_text_preprocessing.pipeline import IndoPreprocessor


class TestPreprocess:
    def test_full_default(self):
        p = IndoPreprocessor()
        out = p.preprocess("Saya SEDANG makan bakso di https://warung.id 😍 5menit lalu")
        assert out == "makan bakso menit"

    def test_no_stem(self):
        p = IndoPreprocessor(stem=False)
        out = p.preprocess("saya sedang makan")
        assert "makan" in out

    def test_lemmatize_overrides_stem(self):
        p = IndoPreprocessor(stem=True, lemmatize=True)
        assert p.preprocess("memakan") == "makan"

    def test_no_stopwords(self):
        p = IndoPreprocessor(remove_stopwords=False)
        out = p.preprocess("makan tidak pergi")
        assert "tidak" in out and "makan" in out

    def test_slang_map_custom(self):
        p = IndoPreprocessor(slang_map={"mangan": "makan"})
        assert p.preprocess("mangan bakso") == "makan bakso"

    def test_stopwords_kwargs(self):
        p = IndoPreprocessor(stopwords_kwargs={"keep_words": ["tidak"]})
        out = p.preprocess("aku tidak pergi")
        assert "tidak" in out


class TestBatch:
    def test_batch(self):
        p = IndoPreprocessor()
        out = p.preprocess_batch(["makan bakso", "minum es"])
        assert len(out) == 2
        assert out[0] == "makan bakso"

    def test_fit_transform_bow(self):
        p = IndoPreprocessor()
        m = p.fit_transform(["saya makan bakso", "dia minum es"])
        assert isinstance(m, np.ndarray)
        assert m.shape[0] == 2

    def test_fit_transform_tfidf(self):
        p = IndoPreprocessor()
        m = p.fit_transform(["saya makan bakso", "dia minum es"], tfidf=True)
        assert np.allclose(np.linalg.norm(m, axis=1), 1.0)