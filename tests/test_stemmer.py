"""Tests untuk stemmer.py — kata uji dari paper Nazief-Adriani & Sastrawi."""

from indo_text_preprocessing.stemmer import stem, stem_batch


class TestRootWords:
    def test_already_root(self):
        assert stem("makan") == "makan"

    def test_case_insensitive(self):
        assert stem("MAKAN") == "makan"

    def test_nonalpha_passthrough(self):
        assert stem("123") == "123"


class TestInflectionSuffixes:
    def test_ku(self):
        assert stem("bolaku") == "bola"

    def test_mu(self):
        assert stem("bajumu") == "baju"

    def test_nya(self):
        assert stem("rumahnya") == "rumah"

    def test_lah(self):
        assert stem("makanlah") == "makan"

    def test_kah(self):
        assert stem("bilangkah") == "bilang"


class TestDerivationalSuffixes:
    def test_an(self):
        assert stem("makanan") == "makan"

    def test_i(self):
        assert stem("sabani") == "saban"

    def test_suffix_then_prefix(self):
        assert stem("memakan") == "makan"


class TestPrefixes:
    def test_men_t(self):
        assert stem("menangkap") == "tangkap"

    def test_meng_k(self):
        assert stem("mengambil") == "ambil"

    def test_mem_p(self):
        assert stem("memasak") == "masak"

    def test_meny_s(self):
        assert stem("menyapu") == "sapu"

    def test_di(self):
        assert stem("dimakan") == "makan"

    def test_ter(self):
        assert stem("teratur") == "atur"

    def test_ber(self):
        assert stem("berlari") == "lari"

    def test_ke_an(self):
        assert stem("kemungkinan") == "mungkin"

    def test_memper(self):
        assert stem("memperbaiki") == "baik"


class TestFallback:
    def test_unknown_word(self):
        # kata tidak dikenal: tetap return string >= 3 huruf
        out = stem("zzzqqq")
        assert len(out) >= 3


def test_batch():
    assert stem_batch(["memasak", "makan", "dimakan"]) == ["masak", "makan", "makan"]