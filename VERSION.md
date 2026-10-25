# EvoSynth-Bench: Version History & Changelog

### Version 1.0.0 (Release Date: 2026-10-25)
- **Initial Public Dataset Release** for automated machine learning meta-optimizer benchmarking.
- **Pathological Synthetic Benchmark:**
  - 2,000 samples, 20 continuous features (5 informative, 10 redundant linear combinations, 5 random noise channels).
  - 30% label flip noise (`flip_y=0.30`) to test algorithm resilience against deceptive optima.
- **HPO Optimization Trajectory Logs:**
  - 120 recorded evaluation checkpoints across Tier-1 (Exploitation) and Tier-2 (Exploration).
  - Hyperparameter coverage for XGBoost (`gbtree`, `dart`), LightGBM, and Random Forest.
  - Multi-heuristic evaluation tracking: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE), and CMA-ES.
- **Validation Script:**
  - Added standalone `scripts/load_dataset.py` for automated schema integrity validation.
