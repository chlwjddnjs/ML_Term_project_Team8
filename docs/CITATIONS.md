# Citations

The specification requires citing every source of code or data we used.
Add an entry **whenever** you adapt code from a blog, Kaggle, GitHub, etc.

## Datasets

1. de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). X-Wines: A Wine Dataset for
   Recommender Systems and Machine Learning. *Big Data and Cognitive Computing*, 7(1), 20.
   https://doi.org/10.3390/bdcc7010020 — **citation is mandatory** (dataset usage policy).
   The files in `data/sample/` are the X-Wines Test version (CC0 1.0 license).
2. Thoutt, Z. (2017). Wine Reviews [Dataset]. Kaggle.
   https://www.kaggle.com/datasets/zynicide/wine-reviews

## Methods

3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for
   Recommender Systems. *IEEE Computer*, 42(8), 30–37.
4. Matrix factorization hints listed in the term project specification
   (towardsdatascience.com articles, pypi `matrix-factorization`).

## Code sources

| File / function | Source (URL) | What was reused / changed |
|---|---|---|
| `adapter.normalize_text` | Python docs, unicodedata: https://docs.python.org/3/library/unicodedata.html | Standard NFKD + ASCII pattern for removing accents; stopword list written for wine names. |
| `adapter.match_items(method="fuzzy")` | Python docs, difflib: https://docs.python.org/3/library/difflib.html | `SequenceMatcher.ratio()` and `get_close_matches()` used as documented; winery-then-name blocking written by us. |
| `adapter.scale_points(method="quantile")` | Quantile mapping / histogram matching (pandas `rank(pct=True)` + `quantile`) | Hint given in the base code; implemented on top of pandas. |
| `cleaner.iqr_bounds`, `clean_expert` | IQR outlier rule (Tukey, 1977) | Applied on log(price) because prices are skewed. |

<!-- If AI tools were used to write code, state it here according to the course policy. -->
