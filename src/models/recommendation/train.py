from __future__ import annotations

from pathlib import Path

from src.models.recommendation.collaborative import (
    CollaborativeRecommender,
    simulate_buyer_interactions,
)
from src.models.recommendation.content_based import ContentBasedRecommender
from src.models.recommendation.hybrid import HybridRecommender
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_parquet, save_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


class RecommenderTrainer:

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()

    def train_all(self):
        cfg = self.cfg
        df = load_parquet(f"{cfg['paths']['data_processed']}/properties_features.parquet")
        log.info(f"Loaded {len(df):,} properties for recommender training")

        models_dir = Path(cfg["paths"]["models"])
        ensure_dir(models_dir)

        # 1) Content-based
        log.info("Training content-based recommender...")
        content = ContentBasedRecommender().fit(df)
        content.save(models_dir / "recommender_content.pkl")

        log.info("Simulating buyer interactions...")
        interactions = simulate_buyer_interactions(
            df,
            n_buyers=cfg["recommender"]["simulated_buyers"],
            interactions_per_buyer=cfg["recommender"]["interactions_per_buyer"],
            random_seed=cfg["project"]["random_seed"],
        )
        save_parquet(interactions,
                      f"{cfg['paths']['data_processed']}/interactions.parquet")

        log.info("Training collaborative SVD recommender...")
        collab = CollaborativeRecommender(
            n_factors=cfg["recommender"]["collaborative"]["n_factors"],
            n_epochs=cfg["recommender"]["collaborative"]["n_epochs"],
            lr_all=cfg["recommender"]["collaborative"]["lr_all"],
            reg_all=cfg["recommender"]["collaborative"]["reg_all"],
            random_state=cfg["project"]["random_seed"],
        ).fit(interactions)
        collab.save(models_dir / "recommender_collab.pkl")

        log.info("Building hybrid recommender...")
        hybrid = HybridRecommender(content, collab,
                                     alpha=cfg["recommender"]["hybrid"]["alpha"])
        hybrid.save(models_dir / "recommender_hybrid.pkl")

        log.info("All recommenders saved")
        return content, collab, hybrid


def main():
    """CLI entrypoint: python -m src.models.recommendation.train"""
    trainer = RecommenderTrainer()
    content, collab, hybrid = trainer.train_all()

    # Smoke-demo
    print("\n=== DEMO: Top 5 recommendations for first buyer ===")
    df = content.df
    seed_idx = 0
    rec = hybrid.recommend(
        buyer_id="buyer_0000", seed_property_idx=seed_idx, top_n=5,
        filter_label="undervalued"
    )
    show_cols = ["property_id", "borough", "price", "valuation_label_name",
                  "walk_score", "school_quality_score", "hybrid_score"]
    print(rec[show_cols].to_string())


if __name__ == "__main__":
    main()
