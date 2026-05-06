from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


# Drop columns that leak the label or are identifiers / display-only
LEAKAGE_COLS = {
    "valuation_label",
    "valuation_label_name",
    "peer_median_ppsf",
    "peer_std_ppsf",
    "peer_z",
    "ppsf_neighborhood_percentile",   # this is computed FROM ppsf, leaks
    "ppsf_neighborhood_median",        # ditto
    "price",                            # ppsf already encodes price-per-sqft
    "price_per_sqft",                   # this is what we used to build labels
    "price_per_unit",
}

ID_DISPLAY_COLS = {
    "property_id",
    "nta_proxy",
    "sale_year", "sale_quarter",   # already encoded in market_cycle_index
    "year_built",                   # already encoded in building_age
    "unit_bucket",
}


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols), excluding leakage and IDs."""
    drop = LEAKAGE_COLS | ID_DISPLAY_COLS
    feature_df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    numeric = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that handles imputation + scaling/encoding
    for both numeric and categorical features.
    """
    numeric, categorical = split_feature_columns(df)
    log.info(f"Numeric features ({len(numeric)}): {numeric}")
    log.info(f"Categorical features ({len(categorical)}): {categorical}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric, categorical


def get_X_y(df: pd.DataFrame, target_col: str = "valuation_label"):
    """Return (X, y) - X has only the features we want to model on."""
    y = df[target_col].astype(int).values
    drop = LEAKAGE_COLS | ID_DISPLAY_COLS
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()
    return X, y
