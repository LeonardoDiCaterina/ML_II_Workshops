# =============================================================================
# Autograding Tests for Workshop 00: Revising & Foundations
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
build_tree_pipeline = getattr(module, "build_tree_pipeline", None)
build_panel_pipeline = getattr(module, "build_panel_pipeline", None)
train_and_evaluate_trees = getattr(module, "train_and_evaluate_trees", None)
analyze_residuals = getattr(module, "analyze_residuals", None)


def test_generate_panel_data_shape_and_index():
    assert callable(generate_panel_data), "generate_panel_data is not defined or callable"
    n_ent = 8
    n_per = 12
    n_feat = 5
    df = generate_panel_data(n_entities=n_ent, n_periods=n_per, n_features=n_feat, random_state=42)
    
    assert df.shape[0] == n_ent * n_per, f"Expected {n_ent * n_per} rows, got {df.shape[0]}"
    assert isinstance(df.index, pd.MultiIndex), "DataFrame index must be a MultiIndex"
    assert df.index.names == ["entity_id", "date"], f"Index levels must be [entity_id, date], got {df.index.names}"
    assert "sector" in df.columns, "sector column must be present"
    assert "target" in df.columns, "target column must be present"
    assert "target_growth" in df.columns, "target_growth continuous target must be present"
    
    # Check that numeric feature columns have ~5% NaNs
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert len(feat_cols) == n_feat, f"Expected {n_feat} feature columns, got {len(feat_cols)}"
    assert df[feat_cols].isna().sum().sum() > 0, "Expected injected missing values in numeric columns"


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


def test_build_tree_pipeline_models():
    assert callable(build_tree_pipeline) or callable(build_panel_pipeline), (
        "build_tree_pipeline is not defined or callable"
    )
    builder = build_tree_pipeline if callable(build_tree_pipeline) else build_panel_pipeline
    num_cols = ["feat_0", "feat_1", "feat_2"]
    cat_cols = ["sector"]
    
    for model_name in ["decision_tree", "random_forest", "xgboost"]:
        if callable(build_tree_pipeline):
            pipe = build_tree_pipeline(model_name, num_cols, cat_cols)
        else:
            pipe = build_panel_pipeline(num_cols, cat_cols)
            
        assert "preprocessor" in pipe.named_steps, f"Pipeline for {model_name} must contain preprocessor"
        assert ("regressor" in pipe.named_steps) or ("classifier" in pipe.named_steps), (
            f"Pipeline for {model_name} must contain regressor step"
        )


def test_train_and_evaluate_trees_group_kfold():
    assert callable(train_and_evaluate_trees), "train_and_evaluate_trees is not defined or callable"
    df = generate_panel_data(n_entities=6, n_periods=12, n_features=4, random_state=42)
    num_cols = ["feat_0", "feat_1", "feat_2", "feat_3"]
    
    res = train_and_evaluate_trees(df, feature_cols=num_cols, categorical_cols=["sector"], n_splits=3)
    assert "models" in res, "Result dictionary must contain 'models' key"
    assert "y_true" in res, "Result dictionary must contain 'y_true' key"
    
    for m in ["decision_tree", "random_forest", "xgboost"]:
        assert m in res["models"], f"Model {m} must be evaluated"
        info = res["models"][m]
        assert "rmse" in info and isinstance(info["rmse"], float) and info["rmse"] > 0
        assert "mae" in info and isinstance(info["mae"], float) and info["mae"] > 0
        assert "oof_predictions" in info and len(info["oof_predictions"]) == len(df)
        assert not np.isnan(info["oof_predictions"]).any(), f"{m} OOF predictions must not contain NaNs"


def test_analyze_residuals_metrics():
    assert callable(analyze_residuals), "analyze_residuals is not defined or callable"
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    y_pred = np.array([9.5, 12.5, 13.0, 16.5, 17.0])
    
    metrics = analyze_residuals(y_true, y_pred)
    assert "residuals" in metrics, "residuals array must be present"
    assert "rmse" in metrics and isinstance(metrics["rmse"], float)
    assert "mae" in metrics and isinstance(metrics["mae"], float)
    assert "r2" in metrics and isinstance(metrics["r2"], float)
    assert "lag1_autocorr" in metrics and isinstance(metrics["lag1_autocorr"], float)
    
    expected_residuals = y_true - y_pred
    np.testing.assert_allclose(metrics["residuals"], expected_residuals)
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(np.mean(expected_residuals**2))), rel=1e-4)
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(expected_residuals))), rel=1e-4)
