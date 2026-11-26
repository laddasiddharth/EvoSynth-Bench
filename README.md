# EvoSynth-Bench: Real-World Benchmark Splits for AutoML Meta-Optimizers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23645119.svg)](https://doi.org/10.5281/zenodo.23645119)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Version](https://img.shields.io/badge/Version-v2.0.0-emerald.svg)](VERSION.md)

**EvoSynth-Bench** is the benchmark definition used to evaluate **EvoSynth**, a dynamic
multi-armed-bandit meta-optimizer that runs GA, PSO, DE and CMA-ES side by side for model
selection and hyperparameter tuning.

Version 2 defines **eight real-world datasets** (seven tabular classification datasets and
Fashion-MNIST for neural architecture search) and the **exact rows** every method is searched,
retrained and tested on. Anyone can therefore reproduce the experiments on identical data and
compare a new optimizer on the same footing.

> Version 1.0.0 (synthetic "pathological" dataset and early optimization trajectories) has been
> retired and remains available in the Zenodo version history. See [VERSION.md](VERSION.md).
> Optimization trajectories from the re-run experiments will be added in a later version.

---

## Datasets

| Dataset | Source | Rows | Features | Classes | Metric | Test / search / train rows |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| Covertype | [UCI](https://archive.ics.uci.edu/dataset/31/covertype) (scikit-learn `fetch_covtype`) | 581,012 | 54 | 7 | accuracy | 20,000 / 20,000 / 60,000 |
| HIGGS (98k version) | [OpenML 23512](https://www.openml.org/d/23512) | 98,050 | 28 | 2 | ROC AUC | 20,000 / 20,000 / 60,000 |
| Adult (Census Income) | [OpenML 1590](https://www.openml.org/d/1590) | 48,842 | 14 | 2 | ROC AUC | 12,000 / 20,000 / 36,000 |
| Airlines delay | [OpenML 1169](https://www.openml.org/d/1169) | 539,383 | 7 | 2 | ROC AUC | 20,000 / 20,000 / 60,000 |
| Poker Hand (v2) | [OpenML 1569](https://www.openml.org/d/1569) | 1,025,000 | 10 | 9 | accuracy | 20,000 / 20,000 / 60,000 |
| MiniBooNE | [OpenML 41150](https://www.openml.org/d/41150) | 130,064 | 50 | 2 | ROC AUC | 20,000 / 20,000 / 60,000 |
| Jannis | [OpenML 41168](https://www.openml.org/d/41168) | 83,733 | 54 | 4 | accuracy | 20,000 / 20,000 / 60,000 |
| Fashion-MNIST | [torchvision](https://github.com/zalandoresearch/fashion-mnist) | 70,000 images | 28×28 | 10 | accuracy | official 10,000 test / 10,000 search-train / 2,500 search-validation |

Binary datasets are scored with ROC AUC (several are imbalanced, e.g. Adult 76/24);
multiclass datasets with accuracy. The full catalogue, with source licences, is
[`data/datasets.csv`](data/datasets.csv).

**The data itself is not redistributed.** Each dataset is downloaded from its original source
under that source's licence (listed in `data/datasets.csv`); this release contributes the split
definitions, preprocessing and loader.

## How the splits are made

For each tabular dataset (all with `data_seed = 42`):

1. **test**: a stratified sample of *test rows*, split off first. No search ever sees these rows.
2. **search**: a stratified sample from the remaining rows; optimizers cross-validate candidate
   configurations on it.
3. **train**: a stratified sample from the remaining rows; the selected configuration (and every
   baseline) is retrained on it and scored once on **test**.

`search` and `train` may overlap (both are training data); neither overlaps `test`.
Fashion-MNIST keeps its official test set and draws the search-train and search-validation
images from the official training set.

Row indices refer to the row order of the source as downloaded (scikit-learn / OpenML /
torchvision), counting from 0.

## Preprocessing

Applied once to the whole dataset, identically for every method (no labels are used):

- labels re-coded to `0..K-1` (sorted original labels);
- categorical / text columns → integer codes (categories sorted), missing → their own code;
- missing numeric values → the column median (HIGGS has 1 such row).

---

## Repository organization

```text
EvoSynth-Bench/
├── data/
│   ├── datasets.csv            # catalogue: source, OpenML id, licence, size, metric, split sizes
│   └── splits/
│       ├── <dataset>.csv       # row_index, split (test | search | train | search+train)
│       └── fashion_mnist.csv   # row_index, split (search_train | search_val)
├── scripts/
│   └── load_dataset.py         # downloads a dataset, applies the preprocessing and the splits
├── metadata.json               # schema.org metadata
├── VERSION.md                  # version history
├── requirements.txt
├── LICENSE                     # CC BY 4.0 (this release's files)
└── README.md
```

## Quickstart

```bash
git clone https://github.com/laddasiddharth/EvoSynth-Bench.git
cd EvoSynth-Bench
pip install -r requirements.txt
python scripts/load_dataset.py                 # validate the split files (no download)
python scripts/load_dataset.py --fetch adult   # download Adult and build its splits
```

```python
import sys; sys.path.insert(0, "scripts")
from load_dataset import load_splits

parts = load_splits("higgs")            # downloads once (OpenML cache), then splits
X_search, y_search = parts["search"]    # tune on these rows
X_train,  y_train  = parts["train"]     # retrain the chosen configuration
X_test,   y_test   = parts["test"]      # score once
```

The same splits are produced inside EvoSynth by `evosynth.datasets.partition` (exported with
`tools/export_splits.py` in the EvoSynth repository).

---

## License

The files in this release (split definitions, catalogue, scripts, documentation) are licensed
under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The datasets remain under
their own licences; see `data/datasets.csv` and each source page.

## Citation

```bibtex
@dataset{evosynth_bench_2026,
  author    = {Loya, Kaushal and Ladda, Siddharth and Reddy, Paarthu and Narang, Priyansh and Kharat, Shivam},
  title     = {EvoSynth-Bench: Real-World Benchmark Splits for AutoML Meta-Optimizers},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v2.0.0},
  doi       = {10.5281/zenodo.23645119},
  url       = {https://doi.org/10.5281/zenodo.23645119}
}
```
