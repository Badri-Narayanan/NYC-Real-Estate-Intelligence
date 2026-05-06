from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from src.models.classification.base import BaseClassifier


class KNNValuationClassifier(BaseClassifier):
    name = "knn"

    def _build_estimator(self):
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)

    @property
    def param_grid(self) -> dict:
        return self.config["classifiers"]["knn"]["param_grid"]
