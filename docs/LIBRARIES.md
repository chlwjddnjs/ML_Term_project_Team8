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
| | | | |

## Member 3 (최정원) – content-based & hybrid

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| | | | |

## Member 4 (신우엽) – evaluation & analysis

| Module | Function / class | Parameters used | Purpose |
|---|---|---|---|
| | | | |
