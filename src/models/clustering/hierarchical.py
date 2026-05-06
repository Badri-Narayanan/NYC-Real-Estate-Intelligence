from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.utils.io import save_pickle, load_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


class NeighborhoodHierarchical:
    """Agglomerative clustering. Used to compare structure with k-Means."""

    def __init__(self, n_clusters: int = 8, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.scaler = StandardScaler()
        self.features: list[str] = []
        self.fitted_labels_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, features: list[str],
              max_samples: int = 10000) -> "NeighborhoodHierarchical":
        """
        Hierarchical clustering is O(n^2) memory; we sub-sample
        to a tractable size for fitting and refit borough-aware later.
        """
        self.features = [f for f in features if f in df.columns]
        sample = df.sample(min(max_samples, len(df)), random_state=42)
        X = sample[self.features].fillna(sample[self.features].median()).values
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.fitted_labels_ = self.model.labels_
        log.info(f"Hierarchical fit on {len(sample):,} sample (linkage={self.linkage})")
        return self

    def get_dendrogram_data(self, df: pd.DataFrame, max_samples: int = 500):
        """Return linkage matrix for dendrogram plotting."""
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        sample = df.sample(min(max_samples, len(df)), random_state=42)
        X = sample[self.features].fillna(sample[self.features].median()).values
        Xs = self.scaler.transform(X)
        return scipy_linkage(Xs, method=self.linkage)

    def save(self, path: str | Path) -> None:
        save_pickle({
            "model": self.model, "scaler": self.scaler,
            "features": self.features, "n_clusters": self.n_clusters,
            "linkage": self.linkage,
        }, path)

    def load(self, path: str | Path) -> "NeighborhoodHierarchical":
        payload = load_pickle(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.features = payload["features"]
        self.n_clusters = payload["n_clusters"]
        self.linkage = payload["linkage"]
        return self
