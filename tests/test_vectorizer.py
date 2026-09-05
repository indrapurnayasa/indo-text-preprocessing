"""Tests untuk vectorizer.py."""

import numpy as np

from indo_text_preprocessing.vectorizer import CountVectorizer, TfidfVectorizer

CORPUS = [
    "saya suka makan bakso",
    "saya suka minum es teh",
    "dia makan bakso juga",
]


class TestCountVectorizer:
    def test_vocab_built(self):
        v = CountVectorizer().fit(CORPUS)
        assert "saya" in v.vocabulary_
        assert "bakso" in v.vocabulary_
        assert "bandung" not in v.vocabulary_

    def test_counts(self):
        m = CountVectorizer().fit_transform(CORPUS)
        # dokumen 1: "saya suka makan bakso" → masing2 count 1
        i_saya = v.vocabulary_["saya"] if (v := CountVectorizer().fit(CORPUS)) else 0
        assert m[0, i_saya] == 1

    def test_repeat_word(self):
        m = CountVectorizer().fit_transform(["kucing kucing kucing"])
        assert m[0].sum() == 3

    def test_oov_ignored(self):
        v = CountVectorizer().fit(CORPUS)
        m = v.transform(["bandung juara"])
        assert m.sum() == 0

    def test_min_df(self):
        v = CountVectorizer(min_df=2).fit(CORPUS)
        # kata yang muncul di >= 2 dokumen: saya, suka, makan, bakso
        assert "saya" in v.vocabulary_
        assert "minum" not in v.vocabulary_

    def test_max_features(self):
        v = CountVectorizer(max_features=3).fit(CORPUS)
        assert len(v.vocabulary_) == 3


class TestTfidfVectorizer:
    def test_shape(self):
        m = TfidfVectorizer().fit_transform(CORPUS)
        assert m.shape == (3, len(CountVectorizer().fit(CORPUS).vocabulary_))

    def test_l2_normalized(self):
        m = TfidfVectorizer().fit_transform(CORPUS)
        norms = np.linalg.norm(m, axis=1)
        assert np.allclose(norms, 1.0)

    def test_common_word_lower_weight(self):
        m = TfidfVectorizer().fit_transform(CORPUS)
        v = TfidfVectorizer().fit(CORPUS)
        # 'saya' muncul di 2/3 dokumen → idf lebih kecil dari 'minum' (1/3)
        assert v.idf_[v.vocabulary_["saya"]] < v.idf_[v.vocabulary_["minum"]]

    def test_sklearn_compatible_values(self):
        # smooth_idf formula: ln((1+N)/(1+df)) + 1
        import math
        v = TfidfVectorizer().fit(CORPUS)
        n = len(CORPUS)
        # 'saya' df=2 → idf = ln(4/3)+1
        assert math.isclose(v.idf_[v.vocabulary_["saya"]], math.log(4 / 3) + 1)

    def test_empty_doc(self):
        m = TfidfVectorizer().fit_transform([CORPUS[0], ""])
        assert np.allclose(m[1], 0.0)  # dokumen kosong → baris nol, tanpa error