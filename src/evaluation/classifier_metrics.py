from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ClassifierResult:
    """All metrics + artifacts from one classifier evaluation."""
    name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: float
    cohen_kappa: float
    confusion_matrix: np.ndarray
    classification_report: str
    train_seconds: float
    predict_seconds: float
    best_params: dict | None = None
    extras: dict = field(default_factory=dict)

    def as_summary_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "roc_auc_ovr": round(self.roc_auc_ovr, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "train_sec": round(self.train_seconds, 2),
            "predict_sec": round(self.predict_seconds, 3),
        }


class ClassifierEvaluator:
    """Compute the full metric suite for a fitted classifier."""

    def __init__(self, label_names: list[str] | None = None):
        self.label_names = label_names or ["undervalued", "fairly_valued", "overvalued"]

    def evaluate(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                 train_seconds: float = 0.0,
                 best_params: dict | None = None,
                 name: str | None = None) -> ClassifierResult:
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
        predict_seconds = time.perf_counter() - t0

        # Multi-class ROC-AUC (one-vs-rest, weighted)
        if y_proba is not None and y_proba.shape[1] >= 2:
            try:
                roc = roc_auc_score(y_test, y_proba, multi_class="ovr",
                                     average="weighted")
            except ValueError:
                roc = float("nan")
        else:
            roc = float("nan")

        result = ClassifierResult(
            name=name or getattr(model, "name", "model"),
            accuracy=accuracy_score(y_test, y_pred),
            f1_macro=f1_score(y_test, y_pred, average="macro"),
            f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            precision_macro=precision_score(y_test, y_pred, average="macro",
                                              zero_division=0),
            recall_macro=recall_score(y_test, y_pred, average="macro",
                                        zero_division=0),
            roc_auc_ovr=roc,
            cohen_kappa=cohen_kappa_score(y_test, y_pred),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(
                y_test, y_pred, target_names=self.label_names, digits=3,
                zero_division=0,
            ),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            best_params=best_params,
        )
        return result


def build_comparison_table(results: list[ClassifierResult]) -> pd.DataFrame:
    """Combine ClassifierResults into a single tidy comparison DataFrame."""
    rows = [r.as_summary_row() for r in results]
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("f1_macro", ascending=False)
