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
| | | | |

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
| | | | |

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
