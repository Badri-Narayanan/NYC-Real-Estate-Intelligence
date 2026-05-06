from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import save_pickle, load_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


# Buyer simulation
BUYER_PROFILES = [
    {"name": "young_professional",
     "weights": {"walk_score": 0.30, "transit_score": 0.30,
                  "amenity_score": 0.20, "crime_rate_per_1k": -0.20}},
    {"name": "family_with_kids",
     "weights": {"school_quality_score": 0.40, "crime_rate_per_1k": -0.30,
                  "sqft": 0.20, "lot_sqft": 0.10}},
    {"name": "investor_value",
     "weights": {"price_per_sqft": -0.50, "amenity_score": 0.20,
                  "school_quality_score": 0.15, "median_household_income": 0.15}},
    {"name": "luxury_buyer",
     "weights": {"sqft": 0.30, "median_household_income": 0.30,
                  "amenity_score": 0.20, "school_quality_score": 0.20}},
    {"name": "retiree",
     "weights": {"crime_rate_per_1k": -0.30, "nearest_park_m": -0.20,
                  "school_quality_score": 0.10, "transit_score": 0.20,
                  "amenity_score": 0.20}},
]


def simulate_buyer_interactions(df: pd.DataFrame,
                                  n_buyers: int = 500,
                                  interactions_per_buyer: int = 30,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic buyer-property rating data (1-5 scale).
    Each buyer follows a weighted preference profile; their "rating"
    of a property is computed from the property's features and that profile.
    """
    rng = np.random.default_rng(random_seed)
    df = df.reset_index(drop=True).copy()

    # Standardize features used for scoring (z-score, so sign matters)
    feature_cols = list({k for p in BUYER_PROFILES for k in p["weights"]})
    feature_cols = [c for c in feature_cols if c in df.columns]
    Z = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
    Z = Z.fillna(0.0)

    rows = []
    for buyer_id in range(n_buyers):
        profile = BUYER_PROFILES[rng.integers(0, len(BUYER_PROFILES))]
        # Build per-property score using profile weights
        score = pd.Series(0.0, index=Z.index)
        for k, w in profile["weights"].items():
            if k in Z.columns:
                score += Z[k] * w
        score += rng.normal(0, 0.3, size=len(score))   # buyer noise

        # Map z-scores to 1-5 ratings (clip + scale)
        rating_raw = 3 + score
        ratings = np.clip(np.round(rating_raw), 1, 5).astype(int)

        # Sample top-K + a random sample of "browsed" properties
        # (this mimics the implicit-feedback structure of real systems)
        top_idx = score.nlargest(interactions_per_buyer // 2).index
        random_idx = rng.choice(df.index, size=interactions_per_buyer // 2,
                                 replace=False)
        idx = list(set(top_idx) | set(random_idx))[:interactions_per_buyer]

        for i in idx:
            rows.append({
                "buyer_id": f"buyer_{buyer_id:04d}",
                "property_id": df.iloc[i]["property_id"],
                "rating": int(ratings[i]),
                "buyer_profile": profile["name"],
            })

    interactions = pd.DataFrame(rows)
    log.info(f"Simulated {len(interactions):,} interactions for "
             f"{n_buyers} buyers across {interactions['property_id'].nunique():,} properties")
    return interactions


# Collaborative recommender
class CollaborativeRecommender:
    """SVD-based collaborative filtering. Falls back to user-mean if
    `surprise` is not installed."""

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                  lr_all: float = 0.005, reg_all: float = 0.02,
                  random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = None
        self.trainset = None
        self.interactions: pd.DataFrame | None = None
        self._mode = "surprise"


    def _try_surprise(self):
        """Attempt to import scikit-surprise. Returns None if it's not
        installed OR if the import fails for any other reason (e.g. NumPy
        2.x ABI incompatibility, which raises ImportError-via-RuntimeError
        depending on the platform). All failures fall back to user-mean."""
        try:
            from surprise import SVD, Dataset, Reader
            return SVD, Dataset, Reader
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            log.warning(f"scikit-surprise unavailable ({type(e).__name__}): {str(e)[:120]}")
            return None


    def fit(self, interactions: pd.DataFrame) -> "CollaborativeRecommender":
        """interactions: DataFrame with columns [buyer_id, property_id, rating]"""
        self.interactions = interactions.copy()
        surprise = self._try_surprise()

        if surprise is not None:
            SVD, Dataset, Reader = surprise
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                interactions[["buyer_id", "property_id", "rating"]], reader
            )
            self.trainset = data.build_full_trainset()
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=self.random_state,
            )
            self.model.fit(self.trainset)
            log.info(f"Trained SVD: {self.n_factors} factors, {self.n_epochs} epochs")
        else:
            log.warning("scikit-surprise not available -> using user-mean fallback")
            self._mode = "user_mean"
            self.model = (interactions.groupby("buyer_id")["rating"].mean(),
                          interactions.groupby("property_id")["rating"].mean(),
                          interactions["rating"].mean())
        return self


    def predict_rating(self, buyer_id: str, property_id: str) -> float:
        if self.model is None:
            raise RuntimeError("Recommender not fitted")
        if self._mode == "surprise":
            return float(self.model.predict(buyer_id, property_id).est)
        # Fallback: user_mean + item_mean - global_mean
        user_mean, item_mean, global_mean = self.model
        u = float(user_mean.get(buyer_id, global_mean))
        i = float(item_mean.get(property_id, global_mean))
        return float(np.clip(u + i - global_mean, 1, 5))

    def recommend_for_buyer(self, buyer_id: str, all_property_ids: list,
                              top_n: int = 10) -> pd.DataFrame:
        scores = []
        seen = set(self.interactions[self.interactions["buyer_id"] == buyer_id]
                    ["property_id"]) if self.interactions is not None else set()
        for pid in all_property_ids:
            if pid in seen:
                continue
            scores.append({"property_id": pid,
                            "predicted_rating": self.predict_rating(buyer_id, pid)})
        result = pd.DataFrame(scores).sort_values(
            "predicted_rating", ascending=False
        ).head(top_n)
        return result


    def save(self, path: str | Path) -> None:
        save_pickle({
            "model": self.model,
            "interactions": self.interactions,
            "mode": self._mode,
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
        }, path)

    def load(self, path: str | Path) -> "CollaborativeRecommender":
        payload = load_pickle(path)
        self.model = payload["model"]
        self.interactions = payload["interactions"]
        self._mode = payload["mode"]
        return self