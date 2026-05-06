from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.eda_plots import save_or_show
from src.utils.logger import get_logger

log = get_logger(__name__)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str,
                            save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                  xticklabels=labels, yticklabels=labels, ax=ax,
                  cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_classifier_comparison(comparison: pd.DataFrame, save_path=None):
    """Bar chart comparing the headline metric (f1_macro) across models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    metrics = ["accuracy", "f1_macro", "roc_auc_ovr"]
    plot_df = comparison[metrics].reset_index().melt(id_vars="model")

    sns.barplot(data=plot_df, x="model", y="value", hue="variable",
                  ax=axes[0], palette="Set2")
    axes[0].set_title("Classifier Performance Comparison")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].legend(title="metric")

    # Train time comparison
    sns.barplot(data=comparison.reset_index(), x="model", y="train_sec",
                  ax=axes[1], palette="Set3")
    axes[1].set_title("Training Time (seconds)")
    axes[1].set_ylabel("seconds")

    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_feature_importance(model, feature_names: list[str],
                              top_n: int = 20, save_path=None):
    """Plot feature importance from tree-based models."""
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier", model)
    else:
        clf = model

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        log.warning(f"Model {type(clf).__name__} has no feature_importances_")
        return

    n = min(len(importances), len(feature_names), top_n)
    idx = np.argsort(importances)[::-1][:n]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.3)))
    ax.barh(range(n), importances[idx][::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_title(f"Top {n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_kmeans_optimal_k(scan: dict, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(scan["k_range"], scan["inertias"], marker="o", color="#1f77b4")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Inertia (lower = tighter)")

    axes[1].plot(scan["k_range"], scan["silhouettes"], marker="s", color="#ff7f0e")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Silhouette (higher = better)")

    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_dendrogram(linkage_matrix, save_path=None, max_d: float | None = None):
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, ax=ax, truncate_mode="lastp", p=30,
                 leaf_rotation=90, leaf_font_size=8, show_contracted=True)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward)")
    ax.set_xlabel("Property cluster")
    ax.set_ylabel("Distance")
    if max_d:
        ax.axhline(y=max_d, color="r", linestyle="--")
    fig.tight_layout()
    save_or_show(fig, save_path)
