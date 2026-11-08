# Citations

The specification requires citing every source of code or data we used.
Add an entry **whenever** you adapt code from a blog, Kaggle, GitHub, etc.

## Datasets

1. de Azambuja, R. X., Morais, A. J., & Filipe, V. (2023). X-Wines: A Wine Dataset for
   Recommender Systems and Machine Learning. *Big Data and Cognitive Computing*, 7(1), 20.
   https://doi.org/10.3390/bdcc7010020 — **citation is mandatory** (dataset usage policy).
   The files in `data/sample/` are the X-Wines Test version, and the main experiments use X-Wines Slim.
   See the [official dataset repository and citation policy](https://github.com/rogerioxavier/X-Wines).
2. Thoutt, Z. (2017). Wine Reviews [Dataset]. Kaggle.
   https://www.kaggle.com/datasets/zynicide/wine-reviews

## Methods

3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for
   Recommender Systems. *Computer*, 42(8), 30–37. [DOI](https://doi.org/10.1109/MC.2009.263).
4. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques.
   *ACM Transactions on Information Systems*, 20(4), 422–446. [DOI](https://doi.org/10.1145/582415.582418).
   Method reference for NDCG. The project computes a binary-relevance version in `recsys/eval/metrics.py`.
5. Matrix factorization references listed in the term project specification:
   [Recommender Systems in Python from Scratch](https://towardsdatascience.com/recommender-systems-in-python-from-scratch-643c8fc4f704),
   [Recommendation System: Matrix Factorization](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b),
   and [the `matrix-factorization` package](https://pypi.org/project/matrix-factorization/).
   The project runs its own MF implementation in `recsys/models/mf.py`.

## Code sources

| File / function | Source (URL) | What was reused / changed |
|---|---|---|
| `adapter.normalize_text` | Python docs, unicodedata: https://docs.python.org/3/library/unicodedata.html | Standard NFKD + ASCII pattern for removing accents; stopword list written for wine names. |
| `adapter.match_items(method="fuzzy")` | Python docs, difflib: https://docs.python.org/3/library/difflib.html | `SequenceMatcher.ratio()` and `get_close_matches()` used as documented; winery-then-name blocking written by us. |
| `adapter.scale_points(method="quantile")` | pandas documentation for [`rank`](https://pandas.pydata.org/docs/reference/api/pandas.Series.rank.html) and [`quantile`](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html) | Maps expert score percentiles to X-Wines rating quantiles. |
| `cleaner.iqr_bounds`, `clean_expert` | IQR outlier rule (Tukey, 1977) | Applied on log(price) because prices are skewed. |
| `experiments/compare_clean_matches.py` | pandas documentation for [`Series.nlargest`](https://pandas.pydata.org/docs/reference/api/pandas.Series.nlargest.html) | Selects the model and expert Top-K lists from the matched wine catalogue. |

## Tool assistance

OpenAI Codex assisted with parts of the content and hybrid implementation,
cross-validation and external evaluation experiments, documentation, and slide preparation.
