# EvoSynth-Bench: Version History & Changelog

### Version 2.0.0 (Release Date: 2026-11-26)
- **Real-world datasets replace the synthetic benchmark.** Eight datasets with their exact
  evaluation splits (`data_seed = 42`): Covertype, HIGGS (OpenML 98k), Adult, Airlines,
  Poker Hand (v2), MiniBooNE, Jannis, and Fashion-MNIST for neural architecture search.
- **Split definitions** (`data/splits/`): held-out test rows split off first, then the search and
  retraining rows from the rest, so test rows are never seen by any search.
- **Catalogue** (`data/datasets.csv`): source, OpenML id, source licence, size, metric, split sizes.
- **New `scripts/load_dataset.py`**: downloads each dataset from its original source, applies
  EvoSynth's preprocessing and the splits; validates the split files.
- **Removed:** the synthetic pathological classification benchmark and the v1 optimization
  trajectories (recorded on that benchmark with an earlier engine). Both remain available in
  version 1.0.0 on Zenodo.
- **Fixed:** the DOI in the README and metadata now points to this record (10.5281/zenodo.23645119).
- Optimization trajectories from the re-run experiments will be added in a later version.

### Version 1.0.0 (Release Date: 2026-10-25)
- Initial release: a 2,000-instance synthetic pathological classification benchmark (30% label
  noise) and 120 multi-algorithm HPO trajectory records (GA, PSO, DE, CMA-ES), with
  `scripts/load_dataset.py`.
