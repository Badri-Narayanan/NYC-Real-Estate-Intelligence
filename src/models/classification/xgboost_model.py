from __future__ import annotations

from src.models.classification.base import BaseClassifier
from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    _HAVE_XGB = False
    log.warning("xgboost not installed - falling back to "
                "sklearn.ensemble.GradientBoostingClassifier")


class XGBoostValuationClassifier(BaseClassifier):
    name = "xgboost"

    def _build_estimator(self):
        if _HAVE_XGB:
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=self.random_state,
                verbosity=0,
            )
        # Fallback - sklearn GradientBoosting (slower but works)
        return _GBC(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        if _HAVE_XGB:
            return self.config["classifiers"]["xgboost"]["param_grid"]
    
        return {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        }
