from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import load_parquet, save_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


LABEL_NAMES = {0: "undervalued", 1: "fairly_valued", 2: "overvalued"}
LABEL_TO_INT = {v: k for k, v in LABEL_NAMES.items()}


class FeatureEngineer:
    """End-to-end feature engineering. Produces a model-ready DataFrame."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.thresh = self.cfg["features"]["label_thresholds"]

    # Build derived/engineered features
    def build_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Building age (recompute even if present, stay consistent)
        df["building_age"] = 2024 - df["year_built"]

        # Per-unit metrics
        df["sqft_per_unit"] = (df["sqft"] / df["num_units"]).round(2)
        df["price_per_unit"] = (df["price"] / df["num_units"]).round(2)

        # Lot coverage / Floor Area Ratio (FAR)
        df["floor_area_ratio"] = (df["sqft"] / df["lot_sqft"]).round(3)
        df["lot_coverage"] = (df["sqft"] /
                               (df["lot_sqft"] * df["num_floors"].clip(lower=1))).round(3)

        # Composite "amenity score" - simple sum of 0-1 normalized features
        df["amenity_score"] = (
            df["walk_score"] / 100.0
            + df["transit_score"] / 100.0
            + df["school_quality_score"] / 10.0
            - df["crime_rate_per_1k"] / 80.0
        ).round(3)

        # Distance penalty (closer is better -> log-transform)
        df["log_subway_dist"] = np.log1p(df["nearest_subway_m"]).round(3)
        df["log_park_dist"] = np.log1p(df["nearest_park_m"]).round(3)

        # Market cycle index: simple proxy = sale_year + quarter/4 - 2022
        df["market_cycle_index"] = (df["sale_year"] - 2022 +
                                     (df["sale_quarter"] - 1) / 4.0).round(3)

        log.info(f"Built engineered features: shape now {df.shape}")
        return df

    # Create the target label via peer-group z-score
    def create_valuation_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Peer group: same nta_proxy + similar building
        df["unit_bucket"] = pd.cut(
            df["num_units"],
            bins=[0, 1, 2, 4, 8, 1000],
            labels=["1u", "2u", "3-4u", "5-8u", "9+u"],
        ).astype(str)

        peer_keys = ["nta_proxy", "unit_bucket"]
        grp = df.groupby(peer_keys)["price_per_sqft"]
        df["peer_median_ppsf"] = grp.transform("median")
        df["peer_std_ppsf"] = grp.transform("std").fillna(grp.transform("std").mean())

        # Z-score
        df["peer_z"] = (
            (df["price_per_sqft"] - df["peer_median_ppsf"]) /
            df["peer_std_ppsf"].replace(0, np.nan)
        ).fillna(0.0)

        # Borough-level fallback for tiny peer groups
        peer_size = grp.transform("size")
        small_mask = peer_size < 5
        if small_mask.any():
            log.info(f"Reassigning {small_mask.sum():,} rows with tiny peer groups "
                     f"to borough-level z-score")
            bgrp = df.groupby("borough")["price_per_sqft"]
            bmed = bgrp.transform("median")
            bstd = bgrp.transform("std").replace(0, np.nan)
            df.loc[small_mask, "peer_z"] = (
                (df.loc[small_mask, "price_per_sqft"] - bmed[small_mask]) / bstd[small_mask]
            ).fillna(0.0)

        # Apply thresholds
        lo = self.thresh["undervalued_z"]
        hi = self.thresh["overvalued_z"]
        conditions = [df["peer_z"] < lo, df["peer_z"] > hi]
        choices = [0, 2]
        df["valuation_label"] = np.select(conditions, choices, default=1)
        df["valuation_label_name"] = df["valuation_label"].map(LABEL_NAMES)

        # Distribution
        dist = df["valuation_label_name"].value_counts(normalize=True).round(3)
        log.info(f"Label distribution:\n{dist.to_string()}")
        return df

    # Final selection / cleanup
    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with any critical nulls
        critical = ["price", "sqft", "price_per_sqft", "valuation_label"]
        before = len(df)
        df = df.dropna(subset=critical).reset_index(drop=True)
        if len(df) < before:
            log.warning(f"Dropped {before - len(df):,} rows with null critical fields")
        return df

    # Public pipeline
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.build_engineered_features(df)
        df = self.create_valuation_labels(df)
        df = self.finalize(df)
        return df


def main():
    """CLI entrypoint: python -m src.features.engineering"""
    cfg = load_config()
    in_path = f"{cfg['paths']['data_interim']}/properties_enriched.parquet"
    out_path = f"{cfg['paths']['data_processed']}/properties_features.parquet"
    df = load_parquet(in_path)
    fe = FeatureEngineer(cfg)
    final = fe.run(df)
    save_parquet(final, out_path)
    print(f"Final feature dataset shape: {final.shape}")
    print(f"\nValuation label distribution:")
    print(final["valuation_label_name"].value_counts())


if __name__ == "__main__":
    main()
