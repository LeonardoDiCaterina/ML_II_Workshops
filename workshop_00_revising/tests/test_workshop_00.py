# =============================================================================
# Autograding Tests for Workshop 00: Revising
# =============================================================================

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Load submission module (student notebook by default, or instructor solution)
from nb_loader import load_submission

module = load_submission()

generate_panel_data = getattr(module, "generate_panel_data", None)
engineer_panel_features = getattr(module, "engineer_panel_features", None)
run_dimensionality_reduction = getattr(module, "run_dimensionality_reduction", None)
build_panel_pipeline = getattr(module, "build_panel_pipeline", None)
evaluate_panel_pipeline = getattr(module, "evaluate_panel_pipeline", None)

def test_generate_panel_data_shape_and_index():
    assert callable(generate_panel_data), "generate_panel_data is not defined or callable"
    n_ent = 8
    n_per = 12
    n_feat = 5
    df = generate_panel_data(n_entities=n_ent, n_periods=n_per, n_features=n_feat, random_state=42)
    
    assert df.shape == (n_ent * n_per, n_feat + 2), f"Expected shape {(n_ent * n_per, n_feat + 2)}, got {df.shape}"
    assert isinstance(df.index, pd.MultiIndex), "DataFrame index must be a MultiIndex"
    assert df.index.names == ["entity_id", "date"], f"Index levels must be [entity_id, date], got {df.index.names}"
    assert "sector" in df.columns, "sector column must be present"
    assert "target" in df.columns, "target column must be present"

def test_engineer_panel_features_no_cross_entity_leakage():
    assert callable(engineer_panel_features), "engineer_panel_features is not defined or callable"
    df = generate_panel_data(n_entities=4, n_periods=10, n_features=4, random_state=42)
    featured = engineer_panel_features(df)
    
    assert "feat_0_lag1" in featured.columns, "feat_0_lag1 must be created"
    assert "feat_0_roll3_mean" in featured.columns, "feat_0_roll3_mean must be created"
    
    # Critical test: First period of each entity MUST have NaN for lag1 (no cross-entity bleed!)
    first_periods = featured.groupby(level="entity_id").head(1)
    assert first_periods["feat_0_lag1"].isna().all(), (
        "Data leakage detected! First period of each entity must have NaN for 1-step lag."
    )

def test_run_dimensionality_reduction_outputs():
    assert callable(run_dimensionality_reduction), "run_dimensionality_reduction is not defined or callable"
    X = np.random.randn(50, 6)
    results = run_dimensionality_reduction(X, n_components=2, random_state=42)
    
    assert "pca_emb" in results and results["pca_emb"].shape == (50, 2)
    assert "tsne_emb" in results and results["tsne_emb"].shape == (50, 2)
    assert "umap_emb" in results and results["umap_emb"].shape == (50, 2)
    assert "pca_explained_var" in results and len(results["pca_explained_var"]) == 2
    assert np.all(results["pca_explained_var"] >= 0)

def test_build_panel_pipeline_structure():
    assert callable(build_panel_pipeline), "build_panel_pipeline is not defined or callable"
    num_cols = ["feat_0", "feat_1", "feat_2"]
    cat_cols = ["sector"]
    pipe = build_panel_pipeline(num_cols, cat_cols)
    
    assert "preprocessor" in pipe.named_steps, "Pipeline must contain preprocessor step"
    assert "classifier" in pipe.named_steps, "Pipeline must contain classifier step"

def test_evaluate_panel_pipeline_runs_group_kfold():
    assert callable(evaluate_panel_pipeline), "evaluate_panel_pipeline is not defined or callable"
    df = generate_panel_data(n_entities=6, n_periods=12, n_features=4, random_state=42)
    num_cols = ["feat_0", "feat_1", "feat_2", "feat_3"]
    pipe = build_panel_pipeline(num_cols, ["sector"])
    
    score = evaluate_panel_pipeline(pipe, df, feature_cols=num_cols + ["sector"], n_splits=3)
    assert isinstance(score, float), "Score must be a float"
    assert 0.0 <= score <= 1.0, f"ROC-AUC score must be between 0 and 1, got {score}"
