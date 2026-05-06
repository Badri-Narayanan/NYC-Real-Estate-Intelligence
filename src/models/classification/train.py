from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.classifier_metrics import (
    ClassifierEvaluator,
    ClassifierResult,
    build_comparison_table,
)
from src.features.engineering import LABEL_NAMES
from src.features.preprocessor import get_X_y
from src.models.classification.cart import CARTValuationClassifier
from src.models.classification.knn import KNNValuationClassifier
from src.models.classification.random_forest import RandomForestValuationClassifier
from src.models.classification.xgboost_model import XGBoostValuationClassifier
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_parquet, save_pickle
from src.utils.logger import get_logger

log = get_logger(__name__)


CLASSIFIER_REGISTRY = {
    "knn": KNNValuationClassifier,
    "cart": CARTValuationClassifier,
    "random_forest": RandomForestValuationClassifier,
    "xgboost": XGBoostValuationClassifier,
}


class ClassifierTrainer:
    """Trains all 4 classifiers, evaluates them uniformly, persists artifacts."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.seed = self.cfg["project"]["random_seed"]
        self.evaluator = ClassifierEvaluator(
            label_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
        )

    def _split(self, df: pd.DataFrame):
        X, y = get_X_y(df)
        test_size = self.cfg["split"]["test_size"]
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.seed
        )

    def _try_mlflow(self):
        """Attempt to import mlflow. Return module or None."""
        try:
            import mlflow
            mlflow.set_tracking_uri("file:" + str(self.cfg["paths"]["mlruns"]))
            mlflow.set_experiment("real_estate_classifiers")
            return mlflow
        except ImportError:
            log.warning("mlflow not installed - skipping experiment tracking")
            return None

    def train_one(self, model_key: str, df: pd.DataFrame,
                  tune: bool = True) -> tuple[ClassifierResult, object]:
        cls = CLASSIFIER_REGISTRY[model_key]
        clf = cls(self.cfg, random_state=self.seed)

        X_train, X_test, y_train, y_test = self._split(df)

        log.info(f"Training [{model_key}] on {X_train.shape[0]:,} rows, "
                 f"{X_train.shape[1]} features")

        t0 = time.perf_counter()
        clf.train(X_train, y_train, tune=tune)
        train_seconds = time.perf_counter() - t0

        result = self.evaluator.evaluate(
            clf, X_test, y_test,
            train_seconds=train_seconds,
            best_params=clf.best_params_,
            name=model_key,
        )

        # Persist
        models_dir = Path(self.cfg["paths"]["models"])
        ensure_dir(models_dir)
        clf.save(models_dir / f"classifier_{model_key}.pkl")

        return result, clf

    def train_all(self, df: pd.DataFrame, tune: bool = True,
                  models_to_train: list[str] | None = None
                  ) -> tuple[pd.DataFrame, dict]:
        models_to_train = models_to_train or list(CLASSIFIER_REGISTRY.keys())
        mlflow = self._try_mlflow()

        results: list[ClassifierResult] = []
        fitted: dict[str, object] = {}

        for key in models_to_train:
            log.info(f"==== Training {key} ====")

            if mlflow:
                with mlflow.start_run(run_name=key):
                    result, clf = self.train_one(key, df, tune=tune)
                    mlflow.log_params(result.best_params or {})
                    mlflow.log_metrics({
                        "accuracy": result.accuracy,
                        "f1_macro": result.f1_macro,
                        "f1_weighted": result.f1_weighted,
                        "roc_auc_ovr": result.roc_auc_ovr,
                        "cohen_kappa": result.cohen_kappa,
                        "train_seconds": result.train_seconds,
                    })
            else:
                result, clf = self.train_one(key, df, tune=tune)

            results.append(result)
            fitted[key] = clf
            log.info(f"[{key}] accuracy={result.accuracy:.4f}, "
                     f"f1_macro={result.f1_macro:.4f}, "
                     f"roc_auc={result.roc_auc_ovr:.4f}")

        comparison = build_comparison_table(results)

        # Save comparison artifact
        out = Path(self.cfg["paths"]["reports"]) / "classifier_comparison.csv"
        ensure_dir(out.parent)
        comparison.to_csv(out)
        log.info(f"Saved comparison table -> {out}")

        # Save full result objects
        save_pickle({"results": results, "comparison": comparison},
                    Path(self.cfg["paths"]["models"]) / "classifier_results.pkl")

        return comparison, fitted


def main():
    """CLI entrypoint: python -m src.models.classification.train"""
    cfg = load_config()
    df = load_parquet(f"{cfg['paths']['data_processed']}/properties_features.parquet")
    trainer = ClassifierTrainer(cfg)
    comparison, _ = trainer.train_all(df, tune=True)
    print("\n" + "=" * 70)
    print("FINAL CLASSIFIER COMPARISON")
    print("=" * 70)
    print(comparison.to_string())


if __name__ == "__main__":
    main()
