# Workshop 00: Machine Learning Foundations & Panel Data Revision

Welcome to **Workshop 00**! This workshop serves as a **2-hour foundational bridge** between the tabular, cross-sectional modeling of *Machine Learning I* and the longitudinal, dynamic, and temporal modeling of *Machine Learning II: Forecasting and Dynamic Modeling*.

---

## ⏱️ 120-Minute Laboratory Map

| Time | Block | Pedagogical Focus | Core Deliverable |
|---|---|---|---|
| **00:00 – 00:20** (20m) | **Part 1** | **Synthetic Panel Generation** | MultiIndex `['entity_id', 'date']`, realistic flaws (5% NaNs), dynamic continuous target. |
| **00:20 – 00:45** (25m) | **Part 2** | **Pandas Fluency & Feature Engineering** | Within-entity lags (`shift(1)`) & rolling stats (`rolling(3).mean()`) without cross-entity leakage. |
| **00:45 – 01:10** (25m) | **Part 3** | **Manifold Learning & Projections** | Linear vs non-linear projections: **PCA** (scree plot) vs **t-SNE** vs **UMAP**. |
| **01:10 – 01:35** (25m) | **Part 4** | **Tree Model Spectrum: Tree → Forest → Boost** | `Pipeline` with `ColumnTransformer` & `GroupKFold`. Comparing **DecisionTree**, **RandomForest**, and **XGBoost**. |
| **01:35 – 02:00** (25m) | **Part 5** | **Residual Diagnostics & Time-Series Bridge** | Residual analysis ($e_t = y_t - \hat{y}_t$): Actual vs Fitted, error distributions, **Residual Autocorrelation (ACF)**, and Feature Importances. |

---

## 🎯 Learning Objectives

By completing this workshop, you will:
1. **Understand Panel Data Structure**: Master the distinction between cross-sectional data ($N \times P$), pure univariate time series ($T \times 1$), and longitudinal panel data ($N \text{ entities} \times T \text{ periods} \times P \text{ features}$).
2. **Master Idiomatic Pandas for Sequential Data**:
   - Work seamlessly with hierarchical multi-indexes `['entity_id', 'date']`.
   - Calculate within-entity lags (`shift(1)`) and rolling window aggregations without cross-entity leakage.
3. **Compare Manifold Learning Paradigms**:
   - Contrast linear projections (**PCA**) with non-linear embeddings (**t-SNE** and **UMAP**).
   - Interpret scree plots, explained variance ratios, and topological neighborhood preservation.
4. **Deploy the Tree-Based Learning Spectrum**:
   - Trace the progression from a baseline `DecisionTreeRegressor` (high variance) to `RandomForestRegressor` (bagging) and `XGBRegressor` (gradient boosting).
   - Construct robust `ColumnTransformer` workflows handling heterogeneous columns (numeric scaling, missing value imputation, categorical one-hot encoding).
   - Implement **GroupKFold** cross-validation grouped by entity to prevent inter-entity data snooping.
5. **Connect Residual Analysis to Dynamic Forecasting**:
   - Understand why static cross-sectional tree models leave **autocorrelated residuals** when applied to temporal panel processes.
   - Plot a 4-panel diagnostic dashboard: Actual vs. Fitted, Residuals vs. Fitted, Error Distribution, and Lag-1 Residual Autocorrelation.
   - Compare feature importance metrics (MDI Gini/variance reduction vs. XGBoost Gain).

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
    ├── nb_loader.py                    # Submission loader
    └── test_workshop_00.py             # Autograding unit tests (pytest)
```

---

## 🛠️ Instructions & Tasks Overview

Open `student/workshop_00.ipynb` in Jupyter Lab:

```bash
jupyter lab workshop_00_revising/student/workshop_00.ipynb
```

The notebook guides you through 5 sequential tasks:
- **Task 1: Synthetic Panel Generation (`generate_panel_data`)**: Build a synthetic panel with $N$ entities over $T$ monthly periods, injecting realistic missing values, categoricals, and dynamic targets.
- **Task 2: Pandas Feature Engineering (`engineer_panel_features`)**: Compute within-entity lags and rolling statistics safely.
- **Task 3: Manifold Learning Projections (`run_dimensionality_reduction`)**: Fit PCA, t-SNE, and UMAP on standardized features.
- **Task 4: Tree Model Hierarchy & Group CV (`build_tree_pipeline`, `train_and_evaluate_trees`)**: Assemble `ColumnTransformer` pipelines for Decision Tree, Random Forest, and XGBoost, evaluating via `GroupKFold`.
- **Task 5: Residual Diagnostics (`analyze_residuals`)**: Calculate residual errors ($y - \hat{y}$), RMSE, MAE, $R^2$, and lag-1 autocorrelation.

---

## 🧪 Testing Your Solution Locally

Before submitting to GitHub, test your implementation locally using `pytest`:

```bash
pytest workshop_00_revising/tests/
```

All 6 test assertions should pass cleanly.

---

## 📊 Grading Rubric (100 Points Total)

| Criterion | Points | Verification |
|:---|:---:|:---|
| **Task 1: Panel Data Generation** | 20 | Shape matches $(N \cdot T, P)$, valid MultiIndex, correct column types, NaNs present |
| **Task 2: Leak-Free Pandas Features** | 20 | Lags and rolling means calculated strictly within entity boundaries (no cross-entity bleed) |
| **Task 3: Manifold Embeddings** | 20 | PCA, t-SNE, and UMAP return 2D projections of correct shape |
| **Task 4: Tree Pipelines & Group CV** | 20 | Valid pipelines for DT, RF, XGBoost + GroupKFold evaluation with no entity leakage |
| **Task 5: Residual Diagnostics** | 20 | Accurate calculation of residuals, RMSE, MAE, R², and lag-1 residual autocorrelation |
