# =============================================================================
# Course   : CS 513 - Data Analytics & Machine Learning
# Purpose  : Tests for feature engineering pipeline
# =============================================================================

import pytest

from src.data.enrichment import PropertyEnricher
from src.data.synthetic import generate_synthetic_dataset
from src.features.engineering import FeatureEngineer, LABEL_NAMES
from src.features.preprocessor import (
    build_preprocessor,
    get_X_y,
    split_feature_columns,
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def features_df():
    df = generate_synthetic_dataset(n_samples=3000, random_seed=42)
    df = PropertyEnricher().enrich_all(df)
    return FeatureEngineer().run(df)


def test_labels_created(features_df):
    assert "valuation_label" in features_df.columns
    assert "valuation_label_name" in features_df.columns
    assert features_df["valuation_label"].isin([0, 1, 2]).all()


def test_labels_have_all_three_classes(features_df):
    counts = features_df["valuation_label"].value_counts()
    assert 0 in counts.index, "missing undervalued class"
    assert 1 in counts.index, "missing fairly_valued class"
    assert 2 in counts.index, "missing overvalued class"


def test_label_names_consistent(features_df):
    """Numeric label and string label must agree."""
    for k, name in LABEL_NAMES.items():
        sub = features_df[features_df["valuation_label"] == k]
        assert (sub["valuation_label_name"] == name).all()


def test_engineered_features_exist(features_df):
    expected = {"sqft_per_unit", "amenity_score",
                "log_subway_dist", "market_cycle_index"}
    assert expected.issubset(set(features_df.columns))


def test_preprocessor_no_label_leak(features_df):
    """Critical: preprocessor MUST drop price and price_per_sqft."""
    numeric, categorical = split_feature_columns(features_df)
    leak_cols = {"price", "price_per_sqft", "valuation_label",
                 "ppsf_neighborhood_percentile", "ppsf_neighborhood_median"}
    for col in leak_cols:
        assert col not in numeric, f"LEAK: {col} in numeric features"
        assert col not in categorical, f"LEAK: {col} in categorical features"


def test_get_X_y_shapes(features_df):
    X, y = get_X_y(features_df)
    assert len(X) == len(features_df)
    assert len(y) == len(features_df)
    # X must NOT contain the target columns
    for col in ["valuation_label", "valuation_label_name", "price",
                "price_per_sqft"]:
        assert col not in X.columns


def test_preprocessor_transforms_without_error(features_df):
    sample = features_df.head(200)
    preprocessor, _, _ = build_preprocessor(sample)
    X = sample.drop(columns=["valuation_label", "valuation_label_name"], errors="ignore")
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 200
    assert transformed.shape[1] > 10
