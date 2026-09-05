# Data Sources & Licenses

Corpus internal library ini diambil dan dikurasi dari sumber open source berikut:

| File | Sumber | License |
|------|--------|---------|
| `root_words.txt` | [azophy/id-wordlist](https://github.com/azophy/id-wordlist) (wordlist geovedi/ivan-lanin 2011) | CC-BY-4.0 |
| `stopwords_id.txt` | union [stopwords-iso/stopwords-id](https://github.com/stopwords-iso/stopwords-id) (MIT) + [masdevid/ID-Stopwords](https://github.com/masdevid/ID-Stopwords) | MIT / bebas |
| `slang_id.json` | merge [fendiirfan/Kamus-Alay](https://github.com/fendiirfan/Kamus-Alay) (MIT) + [nasalsabila/kamus-alay colloquial-indonesian-lexicon](https://github.com/nasalsabila/kamus-alay) | MIT / bebas |
| `irregular_words.json` | kurasi manual | — |

Kurasi yang dilakukan:
- filter kata non-alfabet, deduplikasi, sorting
- mapping slang identik (kata = bentuk formal) dibuang
- merge slang: colloquial lexicon diprioritaskan di atas kamus-alay