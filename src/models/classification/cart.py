from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from src.models.classification.base import BaseClassifier


class CARTValuationClassifier(BaseClassifier):
    name = "cart"

    def _build_estimator(self):
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            criterion="gini",
            random_state=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        return self.config["classifiers"]["cart"]["param_grid"]
