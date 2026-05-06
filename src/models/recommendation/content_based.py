from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.io import save_pickle, load_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


# Features used for content similarity (NOT including price/ppsf to avoid
# label leakage and to allow finding similar properties at any price tier)
CONTENT_FEATURES = [
    "sqft", "lot_sqft", "num_units", "num_floors", "building_age",
    "walk_score", "transit_score", "bike_score",
    "nearest_subway_m", "nearest_park_m",
    "crime_rate_per_1k", "school_quality_score",
    "median_household_income", "population_density",
    "amenity_score", "floor_area_ratio",
]


class ContentBasedRecommender:
    """Cosine-similarity recommender over a fixed property catalog."""

    def __init__(self, features: list[str] | None = None):
        self.features = features or CONTENT_FEATURES
        self.scaler = StandardScaler()
        self.df: pd.DataFrame | None = None
        self.feature_matrix: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        usable = [c for c in self.features if c in df.columns]
        missing = set(self.features) - set(usable)
        if missing:
            log.warning(f"Content features missing from df: {missing}")
        self.df = df.reset_index(drop=True).copy()
        X = self.df[usable].fillna(self.df[usable].median()).values
        self.feature_matrix = self.scaler.fit_transform(X)
        log.info(f"Content recommender fitted on {len(df):,} properties "
                 f"x {len(usable)} features")
        return self

    def recommend_similar(self, seed_idx: int, top_n: int = 10,
                            filter_label: str | None = None,
                            max_price: float | None = None,
                            same_borough: bool = False) -> pd.DataFrame:
        """Return top-N most similar properties to the seed, with filters."""
        if self.feature_matrix is None:
            raise RuntimeError("Recommender not fitted")
        if seed_idx < 0 or seed_idx >= len(self.df):
            raise IndexError(f"seed_idx {seed_idx} out of range")

        seed_vec = self.feature_matrix[seed_idx].reshape(1, -1)
        sims = cosine_similarity(seed_vec, self.feature_matrix).ravel()

        order = np.argsort(-sims)
        order = order[order != seed_idx]

        candidates = self.df.iloc[order].copy()
        candidates["similarity"] = sims[order]

        # Filters
        if filter_label is not None:
            candidates = candidates[candidates["valuation_label_name"] == filter_label]
        if max_price is not None:
            candidates = candidates[candidates["price"] <= max_price]
        if same_borough:
            candidates = candidates[candidates["borough"] == self.df.iloc[seed_idx]["borough"]]

        return candidates.head(top_n)

    def recommend_from_preferences(self, preferences: dict,
                                     top_n: int = 10) -> pd.DataFrame:
        """
        Recommend properties matching natural-language-ish preferences.

        preferences: dict with optional keys:
            - budget_max: float
            - bedrooms_min: int  (mapped to num_units)
            - borough: str
            - prefer_walkable: bool
            - prefer_safe: bool
            - prefer_school: bool
            - prefer_transit: bool
            - prefer_undervalued: bool
        """
        if self.df is None:
            raise RuntimeError("Recommender not fitted")
        candidates = self.df.copy()

        # Hard filters
        if preferences.get("budget_max"):
            candidates = candidates[candidates["price"] <= preferences["budget_max"]]
        if preferences.get("borough"):
            candidates = candidates[candidates["borough"] == preferences["borough"].upper()]
        if preferences.get("bedrooms_min"):
            candidates = candidates[candidates["num_units"] >= preferences["bedrooms_min"]]
        if preferences.get("prefer_undervalued"):
            candidates = candidates[candidates["valuation_label_name"] == "undervalued"]

        if candidates.empty:
            log.warning("No properties match the hard filters")
            return candidates.head(0)

        # Soft scoring
        score = pd.Series(0.0, index=candidates.index)
        if preferences.get("prefer_walkable"):
            score += candidates["walk_score"] / 100.0
        if preferences.get("prefer_safe"):
            score += (1.0 - candidates["crime_rate_per_1k"] / 80.0)
        if preferences.get("prefer_school"):
            score += candidates["school_quality_score"] / 10.0
        if preferences.get("prefer_transit"):
            score += candidates["transit_score"] / 100.0

        # Always reward value (undervalued first, fairly_valued second)
        label_bonus = candidates["valuation_label_name"].map({
            "undervalued": 0.5, "fairly_valued": 0.2, "overvalued": 0.0,
        }).fillna(0.0)
        score += label_bonus

        candidates = candidates.assign(preference_score=score)
        return candidates.sort_values("preference_score", ascending=False).head(top_n)

    def save(self, path: str | Path) -> None:
        save_pickle({
            "df": self.df,
            "feature_matrix": self.feature_matrix,
            "scaler": self.scaler,
            "features": self.features,
        }, path)

    def load(self, path: str | Path) -> "ContentBasedRecommender":
        payload = load_pickle(path)
        self.df = payload["df"]
        self.feature_matrix = payload["feature_matrix"]
        self.scaler = payload["scaler"]
        self.features = payload["features"]
        return self
