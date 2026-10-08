# Machine Learning II — Workshop Rules & Policies (`RULES.md`)

This document defines the **hard architectural, pedagogical, and technical standards** governing all workshop assignments, materials, codebases, and student submissions for **Machine Learning II** at Lazarski University.

---

## 1. Absolute Language Policy: Pure Python

> [!CAUTION]
> **NO R CODE, R SYNTAX, OR R PACKAGES MAY EVER APPEAR IN THIS COURSE.**

- All workflows, assignments, pipelines, and solutions are implemented **strictly and exclusively in Python** (Python $\ge 3.10$).
- Prohibited dependencies: `rpy2`, any R binaries, or R-centric terminology.
- When adapting classic textbooks or literature (such as *Forecasting: Principles and Practice* (FPP3) by Hyndman & Athanasopoulos), all concepts, formulas, and exercises must be faithfully translated into modern idiomatic Python using standard scientific stack tools (`pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `torch`, `matplotlib`, `seaborn`).

---

## 2. Temporal & Longitudinal Integrity (Zero Data Leakage)

1. **No Shuffling on Time-Dependent Data**:
   - Standard random train/test splits (`train_test_split(..., shuffle=True)`) and standard random K-Fold cross-validation (`KFold(shuffle=True)`) are strictly forbidden on sequential, longitudinal, or panel data.
   - Any evaluation must respect the arrow of time: use contiguous temporal splits, expanding/rolling windows (`TimeSeriesSplit`), or group-aware partitioning (`GroupKFold` by entity).
2. **Leak-Free Preprocessing**:
   - All transformations (imputation, scaling, one-hot encoding, PCA) must be fitted **strictly on the training split** and applied (`transform`) to the test split.
   - Leak-free construction using `sklearn.pipeline.Pipeline` or `ColumnTransformer` is mandatory.
3. **No Look-Ahead in Feature Engineering**:
   - Lagged features, rolling aggregations, and difference calculations must only access past or contemporaneous information. Never use future values $y_{t+k}$ to predict $y_t$.

---

## 3. Reproducibility & Determinism

1. **Global Seeds**:
   - Every script and notebook must set explicit random seeds at the top:
     ```python
     import numpy as np
     import random
     import torch

     RANDOM_SEED = 42
     np.random.seed(RANDOM_SEED)
     random.seed(RANDOM_SEED)
     torch.manual_seed(RANDOM_SEED)
     ```
2. **Pinned Environments**:
   - Dependencies must be versioned in `pyproject.toml` and mirrored in `environment.yml`.
   - Never use untracked or globally installed user packages.

---

## 4. Code Quality & Vectorization

1. **Idiomatic Pandas & NumPy**:
   - Never write iterative Python `for` loops across DataFrame rows (`iterrows()`, `itertuples()`) when a vectorized method (`apply`, `groupby`, `shift`, `rolling`, or boolean masking) is available.
2. **Type Annotations & Documentation**:
   - Functions in solution scripts and utility modules must have clear docstrings (describing input types, output types, and assumptions) and Python type hints.
3. **Visualization Standards**:
   - Every visualization must have labelled axes (with units), an informative title, legible font sizes, and a diagnostic reading takeaway.

---

## 5. Course Structure & Instructor/Student Separation

1. **Directory Topology**:
   - Each workshop lives in `workshop_NN_topic/`.
   - `instructor/`: Contains the complete, runnable solution notebook (`workshop_NN_solution.ipynb`) and its corresponding `.py` script.
   - `student/`: Contains the auto-generated student walkthrough notebook (`workshop_NN.ipynb`) with `# YOUR CODE HERE` stubs.
   - `tests/`: Contains `pytest` assertions verifying student implementation correctness.
2. **Scaffold Automation**:
   - Never manually maintain diverging copies of instructor and student notebooks. Run `make scaffold` to auto-strip solutions from `instructor/` into `student/`.

---

## 6. Academic Integrity & Grading

1. **Autograding Compliance**:
   - Test suites in `tests/` run automatically on GitHub Actions on every student push.
   - Tests evaluate function signatures, return types, tensor/array shapes, mathematical properties, and deterministic outputs within appropriate numerical tolerances (`np.isclose` or `pytest.approx`).
