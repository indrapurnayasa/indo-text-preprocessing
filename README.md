# indo-text-preprocessing

Preprocessing NLP Bahasa Indonesia — from scratch, satu dependency (`numpy`).

Clean, stopwords, slang replacement, stemming (Nazief-Adriani/ECS, kompatibel Sastrawi), lemmatization, vectorizer (BoW, TF-IDF), pipeline.

## Instalasi

```bash
pip install indo-text-preprocessing
```

Aturan Python: 3.9+

## Quick Start

```python
from indo_text_preprocessing import IndoPreprocessor, stem, clean_all

p = IndoPreprocessor()
p.preprocess("Saya gak jadi makan bakso di https://warung.id 😍 5rb aja!!!")
# → "makan bakso ribu"
```

## Fitur

### Cleaning
```python
from indo_text_preprocessing import clean_all, lowercase, remove_urls

clean_all("CEK https://x.co 😍 15rb!!!")  # → "cek rb"
```

### Stopwords
```python
from indo_text_preprocessing import remove_stopwords, get_stopwords

remove_stopwords("aku tidak pergi ke pasar")  # → "pergi pasar"
remove_stopwords(text, custom_words=["banget"], keep_words=["tidak"])
get_stopwords()  # frozenset 736 kata
```

### Slang
```python
from indo_text_preprocessing import replace_slang

replace_slang("gak tau emangnya kenapa")  # → "enggak tahu memangnya kenapa"
replace_slang(text, mapping={"gak": "tidak"})  # custom menang atas corpus
```

### Stemming (100% kompatibel PySastrawi)
```python
from indo_text_preprocessing import stem, stem_batch

stem("memperbaiki")   # → "baik"
stem("makananmu")     # → "makan"
stem("buku-buku")     # → "buku"
stem_batch(["dimakan", "menulis"])  # → ["makan", "tulis"]
```

### Lemmatization
```python
from indo_text_preprocessing import lemmatize

lemmatize("memakan")  # → "makan" (fallback ke stemmer)
lemmatize("kata Tak Baku", custom_map={"kata": "kata_baku"})
```

### Vectorizer
```python
from indo_text_preprocessing import CountVectorizer, TfidfVectorizer

corpus = ["saya suka makan bakso", "dia minum es teh"]
v = TfidfVectorizer()
m = v.fit_transform(corpus)   # numpy matrix, L2-normalized
m2 = v.transform(["saya makan es"])  # dokumen baru
```

### Pipeline (semua sekaligus)
```python
from indo_text_preprocessing import IndoPreprocessor

p = IndoPreprocessor(
    replace_slang=True,
    remove_stopwords=True,
    stem=True,          # lemmatize=True untuk lemma mode
)
p.preprocess("Saya gak jadi makan bakso 😍 5rb!!!")

p.preprocess_batch(["dokumen 1", "dokumen 2"])

# langsung ke vectorizer
X = p.fit_transform(corpus)           # BoW
X = p.fit_transform(corpus, tfidf=True)  # TF-IDF
```

Opsi pipeline: `remove_url, remove_mention, remove_emoji, replace_slang, remove_stopwords, stem, lemmatize, stopwords_kwargs, slang_map`.

## Benchmark

Stemmer: 91.629 kata/detik (~5× lebih cepat dari PySastrawi), 100% output match pada 14.000 sampel uji (sintetis + natural + reduplikasi).

## Sumber Data & Lisensi

| Corpus | Sumber | Lisensi |
|--------|--------|---------|
| 29.931 kata dasar | [PySastrawi](https://github.com/har07/PySastrawi) | MIT |
| 736 stopwords | [stopwords-iso](https://github.com/stopwords-iso/stopwords-id) + [masdevid/ID-Stopwords](https://github.com/masdevid/ID-Stopwords) | MIT |
| 4.534 slang | [fendiirfan/Kamus-Alay](https://github.com/fendiirfan/Kamus-Alay) + colloquial-indonesian-lexicon | MIT |

Detail kurasi: [src/indo_text_preprocessing/data/SOURCES.md](src/indo_text_preprocessing/data/SOURCES.md).

Algoritma stemming merujuk: Nazief & Adriani (1996), Asian (2007), Purnomo & Purwarianti (2011), implementasi referensi [PySastrawi](https://github.com/har07/PySastrawi).

## Development

```bash
git clone https://github.com/ngurahindrapurnayasa/indo-text-preprocessing
cd indo-text-preprocessing
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest
```

## License

MIT — lihat [LICENSE](LICENSE).