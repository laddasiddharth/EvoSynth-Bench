# EvoSynth-Bench: Multi-Algorithm HPO Trajectories & Pathological Landscapes Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10892002.svg)](https://doi.org/10.5281/zenodo.10892002)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Version](https://img.shields.io/badge/Version-v1.0.0-emerald.svg)](VERSION.md)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

**EvoSynth-Bench** is an open-source, research-grade benchmark dataset developed to evaluate automated machine learning (AutoML) meta-optimizers, multi-armed bandit dispatchers, and surrogate-assisted hyperparameter search algorithms. 

It provides two complementary assets:
1. **A Non-Convex Pathological Classification Benchmark (`data/pathological_classification_benchmark.csv`)**: A 2,000-instance landscape featuring 30% label-flip noise and high feature collinearity designed specifically to induce local stagnation and stress-test escape mechanisms in black-box optimizers.
2. **Multi-Algorithm Optimization Trajectory Logs (`data/evosynth_hpo_trajectories.csv`)**: 120 discrete evaluation episodes mapping 8-dimensional continuous/discrete hyperparameter vectors across 4 evolutionary heuristics (GA, PSO, DE, CMA-ES) and 3 model backends (XGBoost, LightGBM, Random Forest).

---

## 📁 Repository Organization

```text
evosynth-bench/
├── data/
│   ├── pathological_classification_benchmark.csv  # 2,000 samples, 20 features, 30% label flip
│   ├── evosynth_hpo_trajectories.csv              # 120 complete evaluations across 4 algorithms
│   ├── tier1_trajectories.csv                     # 40% Tier-1 exploitation tournament logs
│   ├── tier2_trajectories.csv                     # 60% Tier-2 exploration reserve logs
│   └── warmup_trajectories.csv                    # Latin Hypercube Sampling initial seed set
├── scripts/
│   └── load_dataset.py                            # Standalone validation and loading script
├── metadata.json                                  # Schema.org & DataCite machine-readable metadata
├── VERSION.md                                     # Semantic version history and release notes
├── requirements.txt                               # Minimal runtime dependencies
├── LICENSE                                        # Creative Commons Attribution 4.0 International
└── README.md                                      # Dataset specification and user documentation
```

---

## 📊 Dataset Specifications & Schemas

### 1. Pathological Classification Benchmark
- **File**: `data/pathological_classification_benchmark.csv`
- **Samples**: 2,000 instances
- **Features**: 20 continuous numerical variables (`feat_00` to `feat_19`)
  - 5 Informative features (informative linear signals)
  - 10 Redundant features (linear combinations of informative signals)
  - 5 Random Gaussian noise features
- **Target**: Binary classification label `target` $\in \{0, 1\}$
- **Label Noise**: 30% random asymmetric label flip (`flip_y=0.30`) to simulate high real-world measurement corruption and penalize greedy memorization.

### 2. Multi-Algorithm HPO Trajectories
- **File**: `data/evosynth_hpo_trajectories.csv`
- **Episodes**: 120 recorded evaluations across two co-evolving tiers
- **Column Descriptions**:
  | Column Name | Type | Description |
  | :--- | :--- | :--- |
  | `Eval_ID` | Integer | Global sequential evaluation index |
  | `Algorithm` | String | Metaheuristic operator (`T1_GA`, `T1_PSO`, `T1_DE`, `T1_CMAES`, `T2_PSO`, etc.) |
  | `Fitness` | Float | 3-Fold Stratified Cross-Validation Accuracy score (%) |
  | `model_type` | Categorical | Classifier family (`XGBoost`, `LightGBM`, `RandomForest`) |
  | `learning_rate` | Float | Boosting shrinkage factor $[0.01, 0.30]$ |
  | `n_estimators` | Integer | Number of boosting trees $[50, 500]$ |
  | `max_depth` | Integer | Maximum tree depth $[3, 10]$ |
  | `subsample` | Float | Row subsampling ratio $[0.5, 1.0]$ |
  | `colsample_bytree` | Float | Feature subsampling ratio $[0.3, 1.0]$ |
  | `reg_alpha` | Float | L1 regularization parameter $[0.0, 10.0]$ |
  | `booster` | Categorical | Tree booster type (`gbtree`, `dart`) |
  | `tier` | String | Co-evolutionary hierarchy (`Warmup`, `Tier-1 Exploitation`, `Tier-2 Exploration`) |

---

## 🚀 Quickstart & Usage

### Installation
Clone this repository and install minimal dependencies:
```bash
git clone https://github.com/laddasiddharth/evosynth-bench.git
cd evosynth-bench
pip install -r requirements.txt
```

### Loading with Pandas
```python
import pandas as pd

# Load the pathological dataset
df = pd.read_csv("data/pathological_classification_benchmark.csv")
X = df.drop(columns=["target"])
y = df["target"]
print(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

# Load HPO trajectories
trajectories = pd.read_csv("data/evosynth_hpo_trajectories.csv")
print(trajectories.groupby(["Algorithm", "model_type"])["Fitness"].agg(["count", "mean", "max"]))
```

### Automated Verification
Run the built-in loader script:
```bash
python scripts/load_dataset.py
```

---

## 🎯 Annotation & Generation Guidelines
1. **Label Generation**: The dataset targets are generated synthetically via an underlying non-linear hypercube manifold corrupted by controlled Bernoulli noise ($p=0.30$).
2. **No Human Annotator Bias**: All labels and evaluations are generated programmatically with explicit random seeds (`seed=42`) ensuring $100\%$ reproducibility.
3. **Reproducibility Guarantee**: Every hyperparameter configuration in the trajectory table can be evaluated deterministically using scikit-learn, XGBoost, and LightGBM.

---

## 📜 License
This dataset is distributed under the **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**. You are free to share, copy, distribute, and adapt the dataset for academic and commercial purposes with appropriate attribution to Group G002.

---

## 📚 Citation
If you use EvoSynth-Bench in your research or benchmarks, please cite:

```bibtex
@dataset{evosynth_bench_2026,
  author    = {Ladda, Siddharth and Group G002 Research Team},
  title     = {EvoSynth-Bench: Multi-Algorithm Hyperparameter Optimization Trajectories and Pathological Landscapes Benchmark},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.10892002},
  url       = {https://github.com/laddasiddharth/evosynth-bench}
}
```
