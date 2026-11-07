# Libraries, classes and functions not covered in class

The specification requires an explanation (with parameters) of every
module/class/function/method we used that was not taught in class.
**Each member adds the entries for the code they write.**
Delete rows that were actually covered in class.

## Base code (shared)

| Module | Function / class | Parameters used | Purpose in this project |
|---|---|---|---|
| `yaml` (PyYAML) | `yaml.safe_load(stream)` | `stream`: open config file | Reads `config.yaml` into a dict; `safe_load` only builds plain data types (no code execution). |
| `ast` | `ast.literal_eval(node_or_string)` | string such as `"['Merlot']"` | Parses list columns of X-Wines (`Grapes`, `Harmonize`) safely, unlike `eval`. |
| `argparse` | `ArgumentParser.add_argument(name, choices, nargs, default)` | `--data`, `--models` | Command-line options of `run.py`. |
| `abc` | `ABC`, `@abstractmethod` | – | Forces every model to implement `_fit` and `score` (common interface). |
| `numpy` | `np.random.default_rng(seed)` | `seed`: int | Reproducible random generator for the split and sampling. |
| `numpy` | `Generator.choice(a, size, replace)` | `replace=False`: no duplicates | Picks held-out ratings / evaluation users. |
| `numpy` | `np.argsort(a, kind="stable")` | `kind`: sort algorithm | Orders items by score for top-k recommendation; stable keeps ties in catalogue order. |
| `pandas` | `pd.cut(x, bins, right)` | `right=False`: intervals `[a, b)` | Groups users by number of training ratings (cold-start analysis). |
| `pandas` | `DataFrame.groupby(..., observed=True)` | `observed=True`: only existing categories | Averages metrics per user group. |
| `scipy.stats` | `spearmanr(a, b)` | two score arrays | Rank correlation between model scores and expert points. |
| `matplotlib` | `matplotlib.use("Agg")` | backend name | Draws figures directly to files (works without a display). |
| `pathlib` | `Path.glob(pattern)` | `"*wines*.csv"` | Finds dataset files even if the downloaded names differ. |

## Member 1 (김도원) – data & DatasetAdapter

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| `re` | `re.sub(pattern, repl, string)` | `r"[^a-z0-9]+"` → `" "`; trailing `"(...)"` → `""`; `r"\b(?:19\|20)\d{2}\b\|\bNV\b"` → `" "` | Replaces punctuation with spaces (`normalize_text`); removes the region in brackets and the vintage year from Wine Enthusiast titles (`parse_winemag_title`). |
| `unicodedata` | `unicodedata.normalize(form, s)` | `form="NFKD"` | Splits an accented letter into the letter + a separate accent mark ("è" → "e" + "̀") so the accent can be dropped (`normalize_text`). |
| built-in | `str.encode(encoding, errors)`, `bytes.decode(encoding)` | `"ascii"`, `errors="ignore"` | Drops the accent marks (and non-Latin letters) left by NFKD: "Fèvre" → "Fevre" (`normalize_text`). |
| `difflib` | `SequenceMatcher(isjunk, a, b).ratio()` | `isjunk=None`, `a`, `b`: normalized names | Similarity 0–1 = 2 × matching characters / total length; e.g. "djf vinhos" vs "dfj vinhos" = 0.9 (`_similarity`). |
| `difflib` | `get_close_matches(word, possibilities, n, cutoff)` | `n=3`, `cutoff=threshold` (0.85) | Finds up to 3 expert wineries whose name is similar to an X-Wines winery, most similar first (`match_items(method="fuzzy")`). |
| `pandas` | `Series.rank(pct)` | `pct=True` | Percentile rank (0–1) of each expert score; equal scores get the same average rank (`scale_points(method="quantile")`). |
| `pandas` | `Series.quantile(q)` | `q`: array of percentiles | X-Wines rating at each percentile, i.e. the user rating with the same rank as the expert score (`scale_points`). |
| `pandas` | `DataFrame.drop_duplicates(subset, keep)` | `subset=["user_id", "item_id"]`, `keep="last"` | Keeps only the latest rating when a user rated the same wine twice (`clean_ratings`). |
| `pandas` | `Series.clip(lower, upper)` | IQR limits on log(price): $4 / $163 | Caps price outliers instead of deleting the review (`clean_expert`). |
| `pandas` | `pd.to_numeric(arg, errors)` | `errors="coerce"` | Turns broken ABV values into NaN so they are treated as outliers (`clean_items`). |
| `pandas` | `Series.astype("string").str.strip()` | – | Removes stray spaces from body / acidity labels (`clean_items`). |
| `pandas` | `DataFrame.merge(right, on, how, indicator)` | `how="left"`, `indicator=True` | Finds the pairs found only by fuzzy matching (`_merge == "left_only"`) (`experiments/data_analysis.py`). |
| `numpy` | `np.log`, `np.exp` | – | IQR rule on the log scale for the skewed price, then back to dollars (`clean_expert`, `data_analysis.py`). |
| `matplotlib` | `Axes.set_yscale("log")`, `Axes.axhline` / `axvline(y/x, ls)` | `ls="--"` | Log axis for the long tail of ratings per wine; dashed lines for the ABV limit and the price IQR limits (`data_analysis.py`). |

## Member 2 (김한결) – Matrix Factorization

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| `numpy` | `Generator.normal(loc, scale, size)` | `loc=0`, `scale=0.1`, `size=(n_users, n_factors)` | Small random initial values for the user / item factor matrices P and Q. |
| `numpy` | `Generator.permutation(x)` | `x`: number of train ratings | New random order of the train ratings every epoch (the "stochastic" in SGD). |
| `numpy` | `ndarray.copy()` | – | Keeps p_u from before its update, so q_i is updated with the old value (simultaneous update). |
| `numpy` | `@` operator (`np.matmul`) | two vectors / matrix and vector | Dot product p_u · q_i; `Q[idx] @ P[u]` scores many wines for one user at once. |
| `numpy` | `np.full(shape, fill_value)` | `fill_value`: global mean | Starts every prediction at μ before adding the biases and the factor term. |
| `numpy` | `np.clip(a, a_min, a_max)` | `a_min=1`, `a_max=5` | Keeps logged predictions on the 1–5 rating scale (same as the evaluator). |
| `pandas` | `Series.fillna(value)`, `Series.astype(dtype)` | `value=-1`, `dtype=int` | Ids not seen in train become row number -1 (handled as unknown user / wine). |
| built-in | `dict.get(key, default)` | `default=-1` | Row number of a wine, or -1 if the wine is not in the catalogue. |
| `numpy` | `np.bincount(x, minlength)` | `x`: wine row of every train rating, `minlength`: number of wines | Number of train ratings per wine (`n_i`) for `bias_damping` in `mf.py`. |
| `numpy` | `np.divide(a, b, out, where)` | `a`: bias_damping, `b`: `n_i`, `where=n_i > 0` | `bias_damping / n_i` without dividing by zero for wines with no ratings (`mf.py`). |
| `collections` | `Counter.update(iterable)` | each user's top-k list | Counts how often each wine is recommended (`experiments/check_mf_top.py`). |
| `numpy` | `np.repeat(a, repeats)` | `repeats`: times each wine was recommended | Median train-rating count weighted by how often a wine is recommended (`check_mf_top.py`). |
| `pandas` | `Series.reindex(index, fill_value)` | `fill_value=0` | Per-wine rating counts that also list wines with no ratings (`check_mf_top.py`). |
| `time` | `time.perf_counter()` | – | Training time of each tuning candidate (`fit_seconds` in `experiments/tune_mf.py`). |
| `pandas` | `Series.idxmin()` | – | Epoch with the lowest validation RMSE, used as `n_epochs` (`tune_mf.py`). |
| `yaml` (PyYAML) | `yaml.safe_dump(data, sort_keys, allow_unicode)` | `sort_keys=False`: keep the order of config.yaml, `allow_unicode=True` | Writes `experiments/config_mf_<tag>.yaml`, a copy of config.yaml with the tuned MF setting (`tune_mf.py`). |
| `pathlib` | `Path.write_text(data, encoding)` | `encoding="utf-8"` | Saves that config file (`tune_mf.py`). |
| `sys` | `sys.path.insert(index, path)` | `index=0`, `path`: project root | Lets the scripts in `experiments/` import the `recsys` package (`tune_mf.py`, `check_mf_top.py`). |
| `pathlib` | `Path.resolve()`, `Path.parent`, `Path.relative_to(other)` | `other`: project root | Finds the project root from the script's own path; prints the config path relative to it (`tune_mf.py`). |
| `pandas` | `pd.set_option(name, value)` | `"display.width"`: 200, `"display.max_columns"`: None | Prints the wide result tables without cutting columns (both scripts). |
| `numpy` | `np.median(a)` | `a`: rating counts | Median train-rating count of the top-10 b_i wines / of the recommended wines (both scripts). |
| `pandas` | `DataFrame.to_csv(path, index, encoding)` | `index=False`, `encoding="utf-8-sig"` | Saves result tables; the BOM lets Excel show accented wine names correctly (both scripts). |
| `sys` | `sys.stdout.reconfigure(errors)` | `errors="replace"` | Wine names with accents never crash a console that cannot show them (`check_mf_top.py`). |
| `pandas` | `Series.str.slice(start, stop)` | `0, 32` | Shortens long wine names in the printed tables (`check_mf_top.py`). |
| `pandas` | `DataFrame.assign(**columns)` | `rec_count=`: times each wine was recommended | Adds a column to the per-wine table without changing the original (`check_mf_top.py`). |
| built-in | `dict.setdefault(key, default)` | `reg_bias` → `reg`, `bias_damping` → 0 | Fills MF options that config.yaml does not set (`tune_mf.py`). |
| built-in | `max(iterable, key)`, `min(iterable, key)` | `key`: function of a result row, e.g. `(P@10, -RMSE)` | Picks the best candidate by a rule (`tune_mf.py`). |
| built-in | `print(..., flush=True)` | `flush=True` | Shows each candidate's line right away during the 10-minute search (`tune_mf.py`). |
| `sys` | `sys.exit(message)` | error message | Stops with a message if the valid set is empty (`tune_mf.py`). |

## Member 3 (최정원) – content-based & hybrid

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| `sklearn.feature_extraction` | `DictVectorizer(dtype)`; `fit`, `transform`, `get_feature_names_out` | `dtype=np.float64` | Learns categorical/multi-label vocabularies on the catalogue; transforms new wines into the same sparse feature space, ignoring unseen labels. |
| `sklearn.preprocessing` | `normalize(X)` | default L2 row norm | Balances each attribute block and normalizes wine vectors for cosine similarity. |
| `scipy.sparse` | `csr_matrix`, `hstack(blocks, format)` | `format="csr"` | Stores and combines feature blocks without a large dense matrix. |
| `sklearn.feature_extraction.text` | `TfidfVectorizer(max_features, sublinear_tf)` | `max_features=5000`, `sublinear_tf=True` | Optional description features; skipped when descriptions are unavailable. |
| `pandas` | `to_numeric(arg, errors)` | `errors="coerce"` | Converts ABV to numbers; invalid/missing ABV contributes zero. |
| `numpy` | `linalg.norm`, `allclose` | `allclose(weights, 0)` | Normalizes user profiles; detects equal ratings and uses equal weights instead. |

Implementation notes: Each categorical/list block is L2-normalized, then concatenated with ABV / 25 and the complete wine vector is normalized again. Profiles use signed `(rating - user mean)` weights from training ratings only. With a single/equal rating history, rated wine vectors have equal weights. Unknown users/items return zero similarity. `model.feature_encoder` retains fitted vocabularies for later datasets. Content scores are similarities, so evaluate ranking metrics rather than RMSE/MAE.


Experiment helper (`experiments/compare_hybrid_weights.py`): `copy.copy(obj)` shallow-copies the fitted model attribute dictionary so both weight policies share the exact same fitted MF/Content objects. `np.allclose(a, 0)` verifies that regular-user metric differences are zero. `Axes.bar(x, height, width)` with width=0.36 draws paired fixed/adaptive bars; `Axes.set_xticks(x, labels)` includes group sample sizes.

## Member 4 (신우엽) – evaluation & analysis

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| `numpy` | `np.bincount(x, weights, minlength)` | `x`: user row numbers, `weights`: ratings, `minlength`: number of users | Sum and count of ratings per user in one pass → per-user mean rating (ItemKNN). |
| `numpy` | `np.divide(a, b, out, where)` | `where=b > 0`: divide only where the denominator is non-zero; `out`: default value elsewhere | Safe division: users/items with no ratings get a fallback value instead of NaN. |
| `numpy` | `np.fill_diagonal(a, val)` | `val=0.0` | Sets sim(i, i) = 0 so a wine is never its own neighbor. |
| `numpy` | `np.searchsorted(a, v)` | `a`: sorted user row numbers, `v`: `0..n_users` | Finds where each user's block starts in the sorted rating list (fast per-user grouping). |
| `numpy` | `np.ix_(rows, cols)` | target item rows, rated item columns | Selects the sub-matrix sim[targets, rated items] in one indexing step. |
| `numpy` | `np.argpartition(a, kth, axis)` | `kth=k-1`, `axis=1` | Finds the k most similar neighbors per row without fully sorting (faster than `argsort`). |
| `numpy` | `np.put_along_axis(arr, indices, values, axis)` | `values=True`, `axis=1` | Marks the k selected neighbors in a boolean mask. |
| `pandas` | `Series.map(arg)` | `arg`: dict id → row number / count | Converts user/item ids to matrix positions; attaches train counts to test ratings. |
| `pandas` | `Series.value_counts()` | – | Number of train ratings per wine (`item_n_train` in error analysis). |
| `pandas` | `DataFrame.nlargest(n, columns)` | `columns="abs_error"` | Picks the n worst predictions for the failure analysis. |
| `pandas` | `DataFrame.agg(**named)` | `MAE=("abs_error", "mean")`, … | Several summary statistics per user/item group in `error_by_group`. |
| built-in | `getattr(obj, name, default)` | `name="b_i"`, `default=None` | Uses MF's item bias for the cross-dataset score when the model has one. |
| `pandas` | `DataFrame.pivot(index, columns, values)` | `index="group"`, `columns="model"`, `values="MAE"` | Reshapes the per-group error table to one column per model so it can be drawn as grouped bars (`failure_analysis.py`). |
| `pandas` | `DataFrame.plot(kind, figsize, rot)` | `kind="bar"`, `rot=0`: horizontal x labels | Grouped bar chart of MAE per user/wine group, one bar per model (`failure_analysis.py`). |
| `matplotlib` | `plt.subplots(figsize)`, `Axes.plot(x, y, marker, label)`, `Figure.savefig(path, dpi)` | `marker="o"`, `dpi=150` | Line chart of the mean signed error per true rating, saved as PNG (`failure_analysis.py`). |
| `sys` | `sys.path.insert(0, path)`, `sys.stdout.reconfigure(errors)` | `path`: project root; `errors="replace"` | Lets a script in `experiments/` import the `recsys` package; prints wine names with accents without crashing on any console. |
| `numpy` | `np.isin(element, test_elements)` | `element`: all wine ids, `test_elements`: wines the user rated in train | Removes already-rated wines before the random baseline draws its top-k (`ranking_analysis.py`). |
| `numpy` | `Generator.choice(a, size, replace)` | `size=k`, `replace=False` | Random top-k recommendation used as the lower bound for the ranking metrics (`ranking_analysis.py`). |
| `pandas` | `Series.isin(values)` | `values`: set of the 50 most-rated wines | Share of recommendations / hits / relevant test ratings that fall on the popular head (`ranking_analysis.py`). |
| `matplotlib` | `Axes.bar(x, height, width, label)`, `Axes.set_xticks(ticks, labels)` | `width=0.4`, bars shifted by ±0.2 | Side-by-side bars of overall vs long-tail NDCG per model (`ranking_analysis.py`). |
| `pandas` | `DataFrame.groupby(by).indices` | `by="user_id"` | Gets the row positions for each user before assigning ratings to validation folds (`cross_validate.py`). |
| `numpy` | `Generator.permutation(x)` | `x`: one user's rating row positions | Shuffles the positions reproducibly before distributing them across folds (`cross_validate.py`). |
| `pandas` | `DataFrameGroupBy.agg(func)` | `func=["mean", "std", "count"]` | Reports the average, sample standard deviation, and number of available fold scores for each model (`cross_validate.py`). |
