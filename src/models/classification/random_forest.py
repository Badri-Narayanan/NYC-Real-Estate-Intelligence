from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.models.classification.base import BaseClassifier


class RandomForestValuationClassifier(BaseClassifier):
    name = "random_forest"

    def _build_estimator(self):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        return self.config["classifiers"]["random_forest"]["param_grid"]
