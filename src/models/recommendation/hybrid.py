from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.recommendation.collaborative import CollaborativeRecommender
from src.models.recommendation.content_based import ContentBasedRecommender
from src.utils.io import save_pickle, load_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1]. Returns 0.5 if constant."""
    if s.empty:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


class HybridRecommender:
    """Blend content-based and collaborative scores."""

    def __init__(self, content_rec: ContentBasedRecommender,
                  collab_rec: CollaborativeRecommender,
                  alpha: float = 0.6):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.alpha = alpha

    def recommend(self, buyer_id: str, seed_property_idx: int,
                   top_n: int = 10,
                   filter_label: str | None = None,
                   max_price: float | None = None) -> pd.DataFrame:
        # Content-based pool: take top 200 candidates
        content = self.content_rec.recommend_similar(
            seed_idx=seed_property_idx,
            top_n=200,
            filter_label=filter_label,
            max_price=max_price,
        )
        if content.empty:
            return content

        # Score each with collab
        collab_scores = []
        for pid in content["property_id"]:
            collab_scores.append(self.collab_rec.predict_rating(buyer_id, pid))

        content = content.assign(
            collab_score=collab_scores,
            content_score=content["similarity"],
        )

        # Normalize and blend
        c_norm = _minmax(content["content_score"])
        l_norm = _minmax(content["collab_score"])
        content["hybrid_score"] = self.alpha * c_norm + (1.0 - self.alpha) * l_norm

        return content.sort_values("hybrid_score", ascending=False).head(top_n)

    def recommend_from_preferences(self, buyer_id: str, preferences: dict,
                                     top_n: int = 10) -> pd.DataFrame:
        # Use content recommender's preference filter, then re-rank with collab
        candidates = self.content_rec.recommend_from_preferences(
            preferences, top_n=200
        )
        if candidates.empty:
            return candidates

        collab_scores = [
            self.collab_rec.predict_rating(buyer_id, pid)
            for pid in candidates["property_id"]
        ]
        candidates = candidates.assign(collab_score=collab_scores)

        c_norm = _minmax(candidates["preference_score"])
        l_norm = _minmax(candidates["collab_score"])
        candidates["hybrid_score"] = self.alpha * c_norm + (1.0 - self.alpha) * l_norm
        return candidates.sort_values("hybrid_score", ascending=False).head(top_n)

    def save(self, path: str | Path) -> None:
        save_pickle({"alpha": self.alpha}, path)

    @classmethod
    def from_components(cls, content_path, collab_path, alpha: float = 0.6):
        content = ContentBasedRecommender().load(content_path)
        collab = CollaborativeRecommender().load(collab_path)
        return cls(content, collab, alpha=alpha)
