"""Stemmer Bahasa Indonesia — algoritma Nazief-Adriani (1996) dengan
modifikasi CS (Asian 2007), ECS, dan Improved ECS (Purnomo-Purwarianti 2011).

Alur (identik Sastrawi/PySastrawi):
1. Kata ada di kamus dasar -> return (kata pendek <= 3 huruf tidak diproses).
2. Confix stripping dengan precedence adjustment (Asian 2007):
   bila pola me-i / di-i / pe-i / ter-i / be-lah / be-an terdeteksi, prefix
   dicoba dulu sebelum suffix; bila gagal, restore dan lanjut suffix-first.
3. Suffix: particle (-lah -kah -tah -pun) -> possessive (-ku -mu -nya)
   -> derivational (-i -kan -an -is -isme -isasi).
4. Prefix: plain (di ke se) lalu disambiguator rules 1-40 (varian per rule,
   varian yang menghasilkan kata dasar di kamus menang), maksimal 3 pass.
5. ECS loop pengembalian akhiran: bila gagal, restore prefix dan coba
   kembalikan akhiran satu per satu ('kan' -> 'k' khusus).

Referensi:
- Nazief & Adriani, "Konsep Stemming Bahasa Indonesia" (1996).
- J. Asian, "Effective Techniques for Indonesian Text Retrieval" (2007).
- B. Purnomo, A. Purwarianti (2011) — ECS/Improved ECS.
- PySastrawi (har07, MIT) — referensi implementasi aturan disambiguator.
"""

from __future__ import annotations

import re

from ._data import load_words

_ROOT_WORDS = load_words("root_words.txt")

_MIN_WORD_LEN = 3

# pola confix: prefix dicoba SEBELUM suffix (Asian 2007, hal. 63)
_PRECEDENCE_RE = re.compile(
    r"^be(.*)lah$|^be(.*)an$|^me(.*)i$|^di(.*)i$|^pe(.*)i$|^ter(.*)i$"
)

# ---------------- disambiguator rules (port PySastrawi rules 1-40) ---------
# (regex, replacement, guard|None). Tiap rule bisa punya >1 varian;
# iterasi berhenti bila varian menghasilkan kata dasar di kamus.

_C = "[bcdfghjklmnpqrstvwxyz]"


def _guard_rule2_23(m: re.Match) -> bool:
    """Rule 2/23: P != 'er' (P = group 3). C == 'r' diperbolehkan
    (PySastrawi rule2/23 hanya cek group(3))."""
    return m.group(3).startswith("er")


def _guard_rule8(m: re.Match) -> bool:
    """Rule 8: C != 'r' AND P != 'er' (P = group 2)."""
    return m.group(1) == "r" or m.group(2).startswith("er")


def _guard_rule34(m: re.Match) -> bool:
    """Rule 34: P != 'er' (P = group 2). Kode PySastrawi hanya cek 'er'
    (C != {r|w|y|l|m|n} hanya di docstring, tidak diimplementasi)."""
    return m.group(2).startswith("er")


def _guard_c_r(m: re.Match) -> bool:
    """Rule 7/9: C != 'r' (group 1)."""
    return m.group(1) == "r"


_GROUPED_RULES: list[list[tuple[re.Pattern, str, object]], ...] = [
    # Rule 1a/1b: berV -> ber-V / ber-V (r return)
    [
        (re.compile(r"^ber([aiueo].*)$"), r"\1", None),
        (re.compile(r"^ber([aiueo].*)$"), r"r\1", None),
    ],
    # Rule 2: berCAP -> CAP (C != 'r', P != 'er')
    [
        (re.compile(rf"^ber({_C})([a-z])(.*)"), r"\1\2\3", _guard_rule2_23),
    ],
    # Rule 3: berCAerV -> CAerV (C != 'r')
    [
        (re.compile(rf"^ber({_C})([a-z])er([aiueo])(.*)$"), r"\1\2er\3\4", None),
    ],
    # Rule 4: belajar -> bel-ajar
    [
        (re.compile(r"^belajar$"), r"ajar", None),
    ],
    # Rule 5: beC1erC2 -> C1erC2 (C1 != 'r' & bukan khusus)
    [
        (re.compile(rf"^be({_C[:-1]}qstvwxyz])(er[bcdfghjklmnpqrstvwxyz])(.*)$"), r"\1\2\3", None),
    ],
    # Rule 6a/6b: terV -> ter-V (r return)
    [
        (re.compile(r"^ter([aiueo].*)$"), r"\1", None),
        (re.compile(r"^ter([aiueo].*)$"), r"r\1", None),
    ],
    # Rule 7: terCerV -> CerV (C != 'r')
    [
        (re.compile(rf"^ter({_C})er([aiueo].*)$"), r"\1er\2", _guard_c_r),
    ],
    # Rule 8: terCP -> CP (C != 'r', P != 'er')
    [
        (re.compile(rf"^ter({_C})(.*)$"), r"\1\2", _guard_rule8),
    ],
    # Rule 9: teC1erC2 -> C1erC2 (C1 != 'r')
    [
        (re.compile(rf"^te({_C})er({_C})(.*)$"), r"\1er\2\3", _guard_c_r),
    ],
    # Rule 10: me{l|r|w|y}V -> {l|r|w|y}V
    [
        (re.compile(r"^me([lrwy][aiueo].*)$"), r"\1", None),
    ],
    # Rule 11: mem{b|f|v} -> {b|f|v}...
    [
        (re.compile(r"^mem([bfv].*)$"), r"\1", None),
    ],
    # Rule 12: mempe -> mem-pe (lepas 'mem' saat ECS; di sini: mempe -> pe)
    [
        (re.compile(r"^mempe(.*)$"), r"pe\1", None),
    ],
    # Rule 13a/13b: mem{V} -> m{V} / p{V}
    [
        (re.compile(r"^mem([aiueo].*)$"), r"m\1", None),
        (re.compile(r"^mem([aiueo].*)$"), r"p\1", None),
    ],
    # Rule 14: men{c|d|j|s|t|z} -> {c|d|j|s|t|z}...
    [
        (re.compile(r"^men([cdjstz].*)$"), r"\1", None),
    ],
    # Rule 15a/15b: men{V} -> n{V} / t{V}
    [
        (re.compile(r"^men([aiueo].*)$"), r"n\1", None),
        (re.compile(r"^men([aiueo].*)$"), r"t\1", None),
    ],
    # Rule 16: meng{g|h|q|k} -> {g|h|q|k}...
    [
        (re.compile(r"^meng([ghqk].*)$"), r"\1", None),
    ],
    # Rule 17a/17b/17c/17d: meng{V} -> {V} / k{V} / mengeX -> X / ng{V}
    [
        (re.compile(r"^meng([aiueo].*)$"), r"\1", None),
        (re.compile(r"^meng([aiueo].*)$"), r"k\1", None),
        (re.compile(r"^menge(.*)$"), r"\1", None),
        (re.compile(r"^meng([aiueo].*)$"), r"ng\1", None),
    ],
    # Rule 18a/18b: meny{V} -> ny{V} / s{V}
    [
        (re.compile(r"^meny([aiueo].*)$"), r"ny\1", None),
        (re.compile(r"^meny([aiueo].*)$"), r"s\1", None),
    ],
    # Rule 19 (ECS): mempA -> pA where A != 'e' (utk memproteksi, memprerogatif)
    [
        (re.compile(r"^memp([abcdfghijklmopqrstuvwxyz].*)$"), r"p\1", None),
    ],
    # Rule 20: pe{w|y}V -> {w|y}V
    [
        (re.compile(r"^pe([wy][aiueo].*)$"), r"\1", None),
    ],
    # Rule 21a/21b: per{V} -> {V} / pe-r{V} -> r{V}
    [
        (re.compile(r"^per([aiueo].*)$"), r"\1", None),
        (re.compile(r"^pe(r[aiueo].*)$"), r"\1", None),
    ],
    # Rule 23: perCAP -> CAP (C != 'r', P != 'er')
    [
        (re.compile(rf"^per({_C})([a-z])(.*)$"), r"\1\2\3", _guard_rule2_23),
    ],
    # Rule 24: perCAerV -> CAerV (C != 'r')
    [
        (re.compile(rf"^per({_C})([a-z])er([aiueo])(.*)$"), r"\1\2er\3\4", None),
    ],
    # Rule 25: pem{b|f|v} -> {b|f|v}...
    [
        (re.compile(r"^pem([bfv].*)$"), r"\1", None),
    ],
    # Rule 26a/26b: pem{V} -> m{V} / p{V}
    [
        (re.compile(r"^pem([aiueo].*)$"), r"m\1", None),
        (re.compile(r"^pem([aiueo].*)$"), r"p\1", None),
    ],
    # Rule 27: pen{c|d|j|s|t|z} -> {c|d|j|s|t|z}...
    [
        (re.compile(r"^pen([cdjstz].*)$"), r"\1", None),
    ],
    # Rule 28a/28b: pen{V} -> n{V} / t{V}
    [
        (re.compile(r"^pen([aiueo].*)$"), r"n\1", None),
        (re.compile(r"^pen([aiueo].*)$"), r"t\1", None),
    ],
    # Rule 29: peng{C} -> {C}... (semua konsonan, bukan hanya ghqk)
    [
        (re.compile(r"^peng([bcdfghjklmnpqrstvwxyz].*)$"), r"\1", None),
    ],
    # Rule 30a/30b/30c: peng{V} -> {V} / k{V} / pengeX -> X
    [
        (re.compile(r"^peng([aiueo].*)$"), r"\1", None),
        (re.compile(r"^peng([aiueo].*)$"), r"k\1", None),
        (re.compile(r"^penge(.*)$"), r"\1", None),
    ],
    # Rule 31a/31b: peny{V} -> ny{V} / s{V}
    [
        (re.compile(r"^peny([aiueo].*)$"), r"ny\1", None),
        (re.compile(r"^peny([aiueo].*)$"), r"s\1", None),
    ],
    # Rule 32: pelV -> pel-V except pelajar -> ajar
    [
        (re.compile(r"^pelajar$"), r"ajar", None),
        (re.compile(r"^pel([aiueo].*)$"), r"l\1", None),
    ],
    # Rule 34: peCP -> CP (C != r|w|y|l|m|n, P != 'er')
    [
        (re.compile(rf"^pe({_C})(.*)$"), r"\1\2", _guard_rule34),
    ],
    # Rule 35: terC1erC2 -> C1erC2 (C1 dari {bcdfghjkpqstvxz}, tanpa l m n r w y)
    [
        (re.compile(rf"^ter([bcdfghjkpqstvxz])(er{_C})(.*)$"), r"\1\2\3", None),
    ],
    # Rule 36: peC1erC2 -> C1erC2 (C1 dari {bcdfghjkpqstvxz})
    [
        (re.compile(rf"^pe([bcdfghjkpqstvxz])(er{_C})(.*)$"), r"\1\2\3", None),
    ],
    # Rule 37a/37b: C(er|el|em|in)V...
    [
        (re.compile(rf"^({_C})(er[aiueo])(.*)$"), r"\1\2\3", None),
        (re.compile(rf"^({_C})er([aiueo])(.*)$"), r"\1\2\3", None),
    ],
    [
        (re.compile(rf"^({_C})(el[aiueo])(.*)$"), r"\1\2\3", None),
        (re.compile(rf"^({_C})el([aiueo])(.*)$"), r"\1\2\3", None),
    ],
    [
        (re.compile(rf"^({_C})(em[aiueo])(.*)$"), r"\1\2\3", None),
        (re.compile(rf"^({_C})em([aiueo])(.*)$"), r"\1\2\3", None),
    ],
    [
        (re.compile(rf"^({_C})(in[aiueo])(.*)$"), r"\1\2\3", None),
        (re.compile(rf"^({_C})in([aiueo])(.*)$"), r"\1\2\3", None),
    ],
    # Rule 41: kuA -> A
    [
        (re.compile(r"^ku(.*)$"), r"\1", None),
    ],
    # Rule 42: kauA -> A
    [
        (re.compile(r"^kau(.*)$"), r"\1", None),
    ],
]


def _is_root(word: str) -> bool:
    return word in _ROOT_WORDS


# ---------------- suffix removals (urutan Sastrawi) ------------------------

_PARTICLE_RE = re.compile(r"-*(lah|kah|tah|pun)$")
_POSSESSIVE_RE = re.compile(r"-*(ku|mu|nya)$")


def _remove_particle(word: str) -> str:
    """Cabut partikel (-lah -kah -tah -pun), termasuk '-' penghubung
    (semantik RemoveInflectionalParticle PySastrawi: r'-*(...)$')."""
    m = _PARTICLE_RE.search(word)
    if m and len(word) - len(m.group(0)) >= _MIN_WORD_LEN:
        return word[: len(word) - len(m.group(0))]
    return word


def _remove_possessive(word: str) -> str:
    """Cabut pronominal posesif (-ku -mu -nya) — r'-*(ku|mu|nya)$'."""
    m = _POSSESSIVE_RE.search(word)
    if m and len(word) - len(m.group(0)) >= _MIN_WORD_LEN:
        return word[: len(word) - len(m.group(0))]
    return word


_DS_RE = re.compile(r"(is|isme|isasi|i|kan|an)$")


def _remove_derivational_suffix(word: str) -> tuple[str, str] | None:
    """Return (kata_baru, suffix_yang_dicabut) atau None bila tidak ada.

    Satu regex seperti PySastrawi RemoveDerivationalSuffix — urutan
    alternasi 'is|isme|isasi|i|kan|an' penting: re.match mencoba varian
    paling kiri dulu sehingga 'kisasi' dicabut 'isasi' (bukan 'i').
    """
    m = _DS_RE.search(word)
    if m and len(word) > len(m.group(1)):
        return word[: len(word) - len(m.group(1))], m.group(1)
    return None


def _remove_suffixes(word: str) -> tuple[str, str | None, str]:
    """Partikel -> posesif -> derivasional. Stop bila hasil intermediate
    sudah kata dasar (semantik accept_visitors Sastrawi).

    Return (final_word, suffix_derivasional atau None, kata_setelah_pronominal)
    — ketiga untuk ECS: kata sebelum derivational strip dipakai sebagai
    subject pengembalian akhiran.
    """
    word = _remove_particle(word)
    if _is_root(word):
        return word, None, word
    word = _remove_possessive(word)
    if _is_root(word):
        return word, None, word
    ds = _remove_derivational_suffix(word)
    if ds is not None:
        return ds[0], ds[1], word
    return word, None, word


# ---------------- prefix removals ------------------------------------------

_PLAIN_PREFIX = re.compile(r"^(di|ke|se)")


def _apply_group(word: str, group: list[tuple[re.Pattern, str]]) -> str | None:
    """Terapkan satu grup disambiguator — semantik AbstractDisambiguatePrefixRule
    PySastrawi persis: result DITIMPA oleh tiap varian (termasuk None bila
    varian tidak match!); break hanya bila varian menghasilkan kata dasar.
    Karena itu varian None di akhir grup menghapus hasil varian sebelumnya
    (bug PySastrawi yang dipertahankan sebagai behavior)."""
    result = None
    for regex, repl, guard in group:
        m = regex.match(word)
        if not m:
            result = None
            continue
        cand = regex.sub(repl, word, count=1)
        if not cand:
            result = None
            continue
        if guard and guard(m):
            result = None
            continue
        result = cand
        if _is_root(result):
            break
    return result


def _guard_rule8(m: re.Match) -> bool:
    """Rule 8: C != 'r' AND P != 'er' (P = group 2)."""
    return m.group(1) == "r" or m.group(2).startswith("er")


def _guard_rule34(m: re.Match) -> bool:
    """Rule 34: P != 'er' (P = group 2). Kode PySastrawi hanya cek 'er'
    (C != {r|w|y|l|m|n} hanya di docstring, tidak diimplementasi)."""
    return m.group(2).startswith("er")


def _guard_c_r(m: re.Match) -> bool:
    """Rule 7/9: C != 'r' (group 1)."""
    return m.group(1) == "r"


def _remove_prefixes(word: str) -> str:
    """Maksimal 3 pass — semantik accept_prefix_visitors Sastrawi:
    tiap pass eksekusi visitor berurutan (RemovePlainPrefix di|ke|se dulu,
    lalu grup disambiguator rule1..40); cek kamus SETELAH TIAP visitor —
    bila current sudah kata dasar, proses berhenti (visitor berikutnya
    tidak dieksekusi)."""
    cur = word
    for _ in range(3):
        nxt = _remove_plain_prefix(cur)
        if nxt != cur:
            cur = nxt
            if _is_root(cur):
                return cur
            continue
        # plain tidak match: cek kamus sebelum coba disambiguator
        if _is_root(cur):
            return cur
        for group in _GROUPED_RULES:
            cand = _apply_group(cur, group)
            if cand is not None:
                cur = cand
                if _is_root(cur):
                    return cur
                break
        else:
            break
    return cur


def _remove_plain_prefix(word: str) -> str:
    m = _PLAIN_PREFIX.match(word)
    if m:
        return word[len(m.group(1)):]
    return word


# ---------------- ECS loop pengembalian akhiran ----------------------------
# Implementasi inline di stem(): restore kata sebelum suffix removal, coba
# 'kan' -> 'k', jalankan prefix removal pada hasil restore.


def stem(word: str) -> str:
    """Stem satu kata Bahasa Indonesia ke kata dasar.

    Mendukung kata plural/reduplikasi (buku-buku -> buku). Semantik
    Sastrawi (step 6 Context.execute): bila seluruh proses konfix
    stripping gagal menemukan kata dasar di kamus, kembalikan kata asli.
    """
    word = word.lower().strip()
    if _is_plural(word):
        return _stem_plural(word)
    return _stem_singular(word)


def _stem_singular(word: str) -> str:
    """Stem kata singular — jalur konfix stripping Nazief-Adriani/ECS.

    Tanpa isalpha guard — '-' bisa ikut tersangkut pada suffix
    possessive/partikel (semantik regex -*(...)$ PySastrawi).
    """
    if len(word) <= _MIN_WORD_LEN:
        return word
    if _is_root(word):
        return word

    # ---- jalur P: precedence adjustment (Asian 2007) — prefix dulu bila
    # pola me-i / di-i / pe-i / ter-i / be-lah / be-an terdeteksi ----
    if _PRECEDENCE_RE.match(word):
        cand = _remove_prefixes(word)
        if _is_root(cand):
            return cand
        cand2, _, _ = _remove_suffixes(cand)
        if _is_root(cand2):
            return cand2

    # ---- jalur A: suffix dulu, lalu prefix (precedence normal) ----
    w1, ds1, w_pron = _remove_suffixes(word)
    if _is_root(w1):
        return w1
    cand = _remove_prefixes(w1)
    if _is_root(cand):
        return cand

    # ---- jalur A2 / ECS loop pengembalian akhiran ----
    # Semantik PySastrawi loop_pengembalian_akhiran: iterasi SEMUA removal
    # (reversed); tiap iterasi restore kata sesuai removal ('kan' -> '+k',
    # lainnya -> subject removal itu) lalu jalankan remove_prefixes penuh.
    # Subject DS removal = kata setelah partikel+posesif (w_pron), bukan
    # kata asli; subject PP = kata sebelum pass prefix itu.
    if ds1 is not None:
        removals: list[tuple[str, str, str]] = [(w_pron, w1, ds1)]
        cur = w1
        for _ in range(3):
            nxt = _remove_plain_prefix(cur)
            if nxt == cur:
                nxt = None
                for group in _GROUPED_RULES:
                    cand = _apply_group(cur, group)
                    if cand is not None:
                        nxt = cand
                        break
                if nxt is None:
                    break
            removals.append((cur, nxt, "PP"))
            cur = nxt

        for subject, result, part in reversed(removals):
            current = result + "k" if part == "kan" else subject
            if not current:
                continue
            r = _remove_prefixes(current)
            if _is_root(r):
                return r

    # ---- jalur B: prefix langsung pada kata asli (suffix-first gagal) ----
    cand = _remove_prefixes(word)
    if _is_root(cand):
        return cand
    cand2, _, _ = _remove_suffixes(cand)
    if _is_root(cand2):
        return cand2

    # tidak ada kandidat di kamus -> kata asli (semantik Sastrawi)
    return word


def _is_plural(word: str) -> bool:
    """Sastrawi is_plural: ada '-' di kata (reduplikasi), ATAU pola
    X-ku/X-mu/X-nya dengan X juga mengandung '-' (malaikat-malaikat-nya)."""
    m = re.match(r"^(.*)-(ku|mu|nya|lah|kah|tah|pun)$", word)
    if m:
        return "-" in m.group(1)
    return "-" in word


def _stem_plural(plural: str) -> str:
    """Stem kata plural/reduplikasi — Asian J. (2007) hal. 76-77."""
    m = re.match(r"^(.*)-(.*)$", plural)
    if not m:
        return plural
    words = [m.group(1), m.group(2)]

    # malaikat-malaikat-nya -> malaikat malaikat-nya
    suffix = words[1]
    m2 = re.match(r"^(.*)-(.*)$", words[0])
    if suffix in ("ku", "mu", "nya", "lah", "kah", "tah", "pun") and m2:
        words[0] = m2.group(1)
        words[1] = m2.group(2) + "-" + suffix

    root1 = _stem_singular(words[0])
    root2 = _stem_singular(words[1])

    # meniru-nirukan -> tiru
    if not _is_root(words[1]) and root2 == words[1]:
        root2 = _stem_singular("me" + words[1])

    if root1 == root2:
        return root1
    return plural


def stem_batch(words: list[str]) -> list[str]:
    return [stem(w) for w in words]