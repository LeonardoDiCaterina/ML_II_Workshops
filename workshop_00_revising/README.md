# Workshop 00: Machine Learning Foundations & Panel Data Revision

Welcome to **Workshop 00**! This workshop serves as a **diagnostic and foundational bridge** between the tabular, cross-sectional modeling of *Machine Learning I* and the longitudinal, dynamic, and temporal modeling of *Machine Learning II*.

---

## 🎯 Learning Objectives

By completing this workshop, you will:
1. **Understand Panel Data Structure**: Master the distinction between cross-sectional data ($N \times P$), pure univariate time series ($T \times 1$), and longitudinal panel data ($N \text{ entities} \times T \text{ periods} \times P \text{ features}$).
2. **Master Idiomatic Pandas for Sequential Data**:
   - Work seamlessly with hierarchical multi-indexes `['entity_id', 'date']`.
   - Calculate within-entity lags (`shift(1)`) and rolling window aggregations without cross-entity leakage.
   - Reshape data between long and wide formats using `pivot`, `melt`, and `unstack`.
3. **Compare Manifold Learning & Dimensionality Reduction**:
   - Contrast linear projections (**PCA**) with non-linear embeddings (**t-SNE** and **UMAP**).
   - Interpret scree plots, explained variance ratios, and topological neighborhood preservation.
4. **Build Leak-Free Scikit-Learn Pipelines**:
   - Construct robust `ColumnTransformer` workflows handling heterogeneous columns (numeric scaling, missing value imputation, categorical one-hot encoding).
   - Chain preprocessing with estimators inside `Pipeline`.
   - Implement **GroupKFold** cross-validation grouped by entity to prevent inter-entity data snooping.

---

## 📁 Folder Structure

```
workshop_00_revising/
├── README.md                           # This document (instructions & rubric)
├── instructor/
│   ├── workshop_00_solution.ipynb      # Instructor master reference notebook
│   └── workshop_00_solution.py         # Complete Python reference module
├── student/
│   └── workshop_00.ipynb               # Your interactive walkthrough notebook
└── tests/
    └── test_workshop_00.py             # Autograding unit tests (pytest)
```

---

## 🛠️ Instructions & Tasks Overview

Open `student/workshop_00.ipynb` in Jupyter Lab:

```bash
jupyter lab workshop_00_revising/student/workshop_00.ipynb
```

The notebook guides you through 4 sequential tasks:
- **Task 1: Synthetic Panel Generation (`generate_panel_data`)**: Build a synthetic dataset with $N$ entities over $T$ monthly periods, injecting realistic missing values and categoricals.
- **Task 2: Pandas Feature Engineering (`engineer_panel_features`)**: Compute within-entity lags and rolling statistics safely.
- **Task 3: Manifold Learning Projections (`run_dimensionality_reduction`)**: Fit PCA, t-SNE, and UMAP on standardized features.
- **Task 4: Scikit-Learn Pipeline & Group CV (`build_panel_pipeline`, `evaluate_panel_pipeline`)**: Assemble a `ColumnTransformer` pipeline and evaluate it via entity-aware `GroupKFold`.

---

## 🧪 Testing Your Solution Locally

Before submitting to GitHub Classroom, test your implementation locally using `pytest`:

```bash
pytest workshop_00_revising/tests/
```

All 5 test assertions should pass cleanly.

---

## 📊 Grading Rubric (100 Points Total)

| Criterion | Points | Verification |
|:---|:---:|:---|
| **Task 1: Panel Data Generation** | 25 | Shape matches $(N \cdot T, P)$, valid MultiIndex, correct column types |
| **Task 2: Leak-Free Pandas Features** | 25 | Lags and rolling means calculated strictly within entity boundaries |
| **Task 3: Manifold Embeddings** | 25 | PCA, t-SNE, and UMAP return 2D projections of correct shape |
| **Task 4: Pipeline & Group CV** | 25 | Unfitted ColumnTransformer pipeline + GroupKFold evaluation with no leakage |
