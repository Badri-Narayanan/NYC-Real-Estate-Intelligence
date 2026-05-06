from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

log = get_logger(__name__)

sns.set_theme(style="whitegrid", context="talk")


def save_or_show(fig, save_path: str | Path | None):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        log.info(f"Saved figure -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_price_distribution(df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["price"] / 1e6, bins=60, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Price ($M)")
    axes[0].set_xlabel("Price (millions USD)")
    sns.histplot(np.log10(df["price"]), bins=60, ax=axes[1], color="#ff7f0e")
    axes[1].set_title("Log10(Price)")
    axes[1].set_xlabel("log10(price)")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_ppsf_by_borough(df: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    order = df.groupby("borough")["price_per_sqft"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="borough", y="price_per_sqft", order=order,
                  ax=ax, showfliers=False, palette="viridis")
    ax.set_title("Price-per-sqft Distribution by Borough")
    ax.set_ylabel("Price per sqft ($)")
    ax.set_xlabel("")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] | None = None,
                              save_path=None):
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap="RdBu_r", center=0,
                  linewidths=0.5, ax=ax, square=True,
                  cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_label_distribution(df: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["valuation_label_name"].value_counts()
    counts.plot(kind="bar", ax=ax, color=["#2ca02c", "#7f7f7f", "#d62728"])
    ax.set_title("Valuation Label Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}\n({v/counts.sum()*100:.1f}%)",
                ha="center", fontsize=11)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_walkscore_vs_ppsf(df: pd.DataFrame, save_path=None):
    sample = df.sample(min(10000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=sample, x="walk_score", y="price_per_sqft",
                      hue="borough", alpha=0.4, s=15, ax=ax)
    ax.set_title("Walk Score vs Price-per-sqft (10K sample)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_crime_vs_ppsf(df: pd.DataFrame, save_path=None):
    sample = df.sample(min(10000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=sample, x="crime_rate_per_1k", y="price_per_sqft",
                      hue="borough", alpha=0.4, s=15, ax=ax)
    ax.set_title("Crime Rate vs Price-per-sqft")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1))
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_missingness(df: pd.DataFrame, save_path=None):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        log.info("No missing values - skipping missingness plot")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    miss.plot(kind="barh", ax=ax, color="#d62728")
    ax.set_title("Missing-value rate by column")
    ax.set_xlabel("Fraction missing")
    fig.tight_layout()
    save_or_show(fig, save_path)
