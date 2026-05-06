from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.io import save_pickle, load_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


CLUSTER_NAME_TEMPLATE = {
    # Will be auto-named by the profiler based on archetype features
}


class NeighborhoodKMeans:
    """k-Means on neighborhood-level features. Discovers urban archetypes."""

    def __init__(self, n_clusters: int = 8, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state,
                              n_init=n_init)
        self.scaler = StandardScaler()
        self.features: list[str] = []
        self.cluster_profiles_: pd.DataFrame | None = None
        self.cluster_names_: dict[int, str] = {}

    def fit(self, df: pd.DataFrame, features: list[str]) -> "NeighborhoodKMeans":
        self.features = [f for f in features if f in df.columns]
        X = df[self.features].fillna(df[self.features].median()).values
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)

        df_with_cluster = df.copy()
        df_with_cluster["cluster"] = self.model.labels_
        self.cluster_profiles_ = self._profile_clusters(df_with_cluster)
        self.cluster_names_ = self._auto_name_clusters(self.cluster_profiles_)

        try:
            sil = silhouette_score(Xs, self.model.labels_, sample_size=5000,
                                    random_state=self.random_state)
            log.info(f"k-Means k={self.n_clusters}, silhouette={sil:.3f}")
        except Exception as e:
            log.warning(f"Could not compute silhouette: {e}")

        return self

    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].fillna(df[self.features].median()).values
        return self.model.predict(self.scaler.transform(X))

    def find_optimal_k(self, df: pd.DataFrame, features: list[str],
                         k_range: range = range(3, 13)) -> dict:
        feats = [f for f in features if f in df.columns]
        X = df[feats].fillna(df[feats].median()).values
        Xs = StandardScaler().fit_transform(X)

        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(Xs)
            inertias.append(km.inertia_)
            try:
                sil = silhouette_score(Xs, labels, sample_size=3000,
                                        random_state=self.random_state)
            except Exception:
                sil = float("nan")
            silhouettes.append(sil)
        return {"k_range": list(k_range),
                "inertias": inertias,
                "silhouettes": silhouettes}

    def _profile_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in [
            "price_per_sqft", "walk_score", "transit_score",
            "school_quality_score", "crime_rate_per_1k",
            "median_household_income", "population_density",
            "amenity_score",
        ] if c in df.columns]
        agg = df.groupby("cluster")[cols].median().round(2)
        agg["n_properties"] = df.groupby("cluster").size()
        return agg

    def _auto_name_clusters(self, profile: pd.DataFrame) -> dict[int, str]:
        """Auto-generate human-readable cluster names from features."""
        names = {}
        for cluster_id, row in profile.iterrows():
            tags = []
            ppsf = row.get("price_per_sqft", 0)
            walk = row.get("walk_score", 50)
            crime = row.get("crime_rate_per_1k", 25)
            school = row.get("school_quality_score", 5)

            if ppsf > 1200:
                tags.append("Luxury")
            elif ppsf > 700:
                tags.append("MidTier")
            else:
                tags.append("Affordable")

            if walk > 85:
                tags.append("HighlyWalkable")
            elif walk < 60:
                tags.append("CarOriented")

            if crime < 15:
                tags.append("LowCrime")
            elif crime > 35:
                tags.append("HighCrime")

            if school > 7:
                tags.append("TopSchools")

            names[int(cluster_id)] = "_".join(tags) if tags else f"Cluster{cluster_id}"
        return names

    def save(self, path: str | Path) -> None:
        save_pickle({
            "model": self.model, "scaler": self.scaler,
            "features": self.features,
            "cluster_profiles": self.cluster_profiles_,
            "cluster_names": self.cluster_names_,
            "n_clusters": self.n_clusters,
        }, path)

    def load(self, path: str | Path) -> "NeighborhoodKMeans":
        payload = load_pickle(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.features = payload["features"]
        self.cluster_profiles_ = payload.get("cluster_profiles")
        self.cluster_names_ = payload.get("cluster_names", {})
        self.n_clusters = payload["n_clusters"]
        return self


def main():
    """CLI entrypoint: python -m src.models.clustering.kmeans"""
    cfg = load_config()
    from src.utils.io import load_parquet, save_parquet
    df = load_parquet(f"{cfg['paths']['data_processed']}/properties_features.parquet")

    km = NeighborhoodKMeans(
        n_clusters=cfg["clustering"]["kmeans"]["n_clusters"],
        random_state=cfg["project"]["random_seed"],
        n_init=cfg["clustering"]["kmeans"]["n_init"],
    )
    km.fit(df, cfg["clustering"]["features"])
    df["cluster"] = km.predict(df)
    df["cluster_name"] = df["cluster"].map(km.cluster_names_)

    save_parquet(df, f"{cfg['paths']['data_processed']}/properties_clustered.parquet")
    km.save(Path(cfg["paths"]["models"]) / "clustering_kmeans.pkl")

    print("\n=== CLUSTER PROFILES ===")
    print(km.cluster_profiles_.to_string())
    print("\n=== AUTO NAMES ===")
    for k, v in km.cluster_names_.items():
        print(f"Cluster {k}: {v}")


if __name__ == "__main__":
    main()
