from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.preprocessor import build_preprocessor
from src.utils.io import load_pickle, save_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseClassifier(ABC):
    """
    Abstract base class for all valuation classifiers.

    Subclasses must implement:
        - name (class attribute)
        - _build_estimator() returning a sklearn-compatible estimator
        - default_params dict for the estimator
    """

    name: str = "base"

    def __init__(self, config: dict, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.best_params_: dict | None = None
        self.cv_results_: dict | None = None
        self.feature_names_in_: list[str] | None = None

    # MUST IMPLEMENT
    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return a fresh estimator instance."""
        ...

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        """Return param search grid (keys are pipeline-prefixed)."""
        ...

    # Pipeline construction
    def build_pipeline(self, df_for_schema: pd.DataFrame) -> Pipeline:
        """Build (preprocessor + classifier) pipeline from a sample DataFrame."""
        preprocessor, _, _ = build_preprocessor(df_for_schema)
        estimator = self._build_estimator()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        self.pipeline = pipe
        self.feature_names_in_ = list(df_for_schema.columns)
        return pipe

    # Train
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              tune: bool = True, n_iter: int | None = None,
              cv: int | None = None) -> "BaseClassifier":
        if self.pipeline is None:
            self.build_pipeline(X_train)

        n_iter = n_iter or self.config["classifiers"][self.name].get("n_iter", 10)
        cv = cv or self.config["classifiers"]["cv_folds"]
        scoring = self.config["classifiers"]["scoring"]

        if tune:
            log.info(f"[{self.name}] RandomizedSearchCV: n_iter={n_iter}, cv={cv}")
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            search = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_grid,
                n_iter=n_iter,
                cv=skf,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=2,           # was n_jobs=-1  ← this is the key fix
                refit=True,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            self.pipeline = search.best_estimator_
            self.best_params_ = search.best_params_
            self.cv_results_ = {
                "best_score": search.best_score_,
                "best_params": search.best_params_,
            }
            log.info(f"[{self.name}] Best CV {scoring}: {search.best_score_:.4f}")
            log.info(f"[{self.name}] Best params: {search.best_params_}")
        else:
            self.pipeline.fit(X_train, y_train)
            log.info(f"[{self.name}] Trained without hyperparam search")
        return self

    # Predict
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} not trained yet")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} not trained yet")
        return self.pipeline.predict_proba(X)

    # Persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "pipeline": self.pipeline,
            "name": self.name,
            "best_params": self.best_params_,
            "cv_results": self.cv_results_,
            "feature_names_in": self.feature_names_in_,
        }
        save_pickle(payload, path)

    def load(self, path: str | Path) -> "BaseClassifier":
        payload = load_pickle(path)
        self.pipeline = payload["pipeline"]
        self.best_params_ = payload.get("best_params")
        self.cv_results_ = payload.get("cv_results")
        self.feature_names_in_ = payload.get("feature_names_in")
        return self
