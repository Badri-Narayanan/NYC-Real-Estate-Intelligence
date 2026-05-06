# =============================================================================
# Usage:
#   python main.py --step ingest         # download/generate raw data
#   python main.py --step enrich         # add neighborhood-percentile features
#   python main.py --step features       # feature engineering + labels
#   python main.py --step train          # train all 4 classifiers
#   python main.py --step recommend      # build hybrid recommender
#   python main.py --step cluster        # geospatial clustering
#   python main.py --step all            # run everything in order
# =============================================================================

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import load_parquet, save_parquet
from src.utils.logger import get_logger

log = get_logger("pipeline")


# Stage runners
def stage_ingest(cfg, force: bool = False):
    log.info("=== STAGE: INGEST ===")
    from src.data.ingestion import NYCDataIngester
    ingester = NYCDataIngester(cfg)
    df = ingester.ingest(force=force)
    log.info(f"INGEST done: shape={df.shape}")
    return df


def stage_enrich(cfg):
    log.info("=== STAGE: ENRICH ===")
    from src.data.enrichment import PropertyEnricher
    raw_path = f"{cfg['paths']['data_raw']}/properties.parquet"
    df = load_parquet(raw_path)
    enricher = PropertyEnricher(cfg)
    enriched = enricher.enrich_all(df)
    save_parquet(enriched, f"{cfg['paths']['data_interim']}/properties_enriched.parquet")
    log.info(f"ENRICH done: shape={enriched.shape}")
    return enriched


def stage_features(cfg):
    log.info("=== STAGE: FEATURES ===")
    from src.features.engineering import FeatureEngineer
    df = load_parquet(f"{cfg['paths']['data_interim']}/properties_enriched.parquet")
    fe = FeatureEngineer(cfg)
    final = fe.run(df)
    save_parquet(final, f"{cfg['paths']['data_processed']}/properties_features.parquet")
    log.info(f"FEATURES done: shape={final.shape}")
    return final


def stage_train(cfg, models: list[str] | None = None, tune: bool = True):
    log.info("=== STAGE: TRAIN CLASSIFIERS ===")
    from src.models.classification.train import ClassifierTrainer
    df = load_parquet(f"{cfg['paths']['data_processed']}/properties_features.parquet")
    trainer = ClassifierTrainer(cfg)
    comparison, _ = trainer.train_all(df, tune=tune, models_to_train=models)
    log.info(f"TRAIN done. Comparison:\n{comparison.to_string()}")
    return comparison


def stage_recommend(cfg):
    log.info("=== STAGE: RECOMMENDATION SYSTEM ===")
    from src.models.recommendation.train import RecommenderTrainer
    trainer = RecommenderTrainer(cfg)
    return trainer.train_all()


def stage_cluster(cfg):
    log.info("=== STAGE: CLUSTERING ===")
    from src.models.clustering.kmeans import NeighborhoodKMeans
    df = load_parquet(f"{cfg['paths']['data_processed']}/properties_features.parquet")
    km = NeighborhoodKMeans(
        n_clusters=cfg["clustering"]["kmeans"]["n_clusters"],
        random_state=cfg["project"]["random_seed"],
        n_init=cfg["clustering"]["kmeans"]["n_init"],
    )
    km.fit(df, cfg["clustering"]["features"])
    df["cluster"] = km.predict(df)
    df["cluster_name"] = df["cluster"].map(km.cluster_names_)
    save_parquet(df, f"{cfg['paths']['data_processed']}/properties_clustered.parquet")
    km.save(Path(cfg["paths"]["models"]) / "clustering_kmeans.pkl")
    log.info(f"CLUSTER done. Cluster names: {km.cluster_names_}")
    return km


def stage_all(cfg, **kwargs):
    log.info("=== STAGE: ALL ===")
    t0 = time.time()
    stage_ingest(cfg, force=kwargs.get("force_ingest", False))
    stage_enrich(cfg)
    stage_features(cfg)
    stage_train(cfg, models=kwargs.get("models"), tune=kwargs.get("tune", True))
    stage_recommend(cfg)
    stage_cluster(cfg)
    log.info(f"FULL PIPELINE DONE in {time.time()-t0:.1f}s")


# CLI entrypoint
def main():
    parser = argparse.ArgumentParser(description="NYC Real-Estate ML Pipeline")
    parser.add_argument("--step", required=True,
                          choices=["ingest", "enrich", "features",
                                    "train", "recommend", "cluster", "all"])
    parser.add_argument("--force", action="store_true",
                          help="Force re-ingest if raw data exists")
    parser.add_argument("--no-tune", action="store_true",
                          help="Skip hyperparameter tuning (fast mode)")
    parser.add_argument("--models", nargs="*", default=None,
                          choices=["knn", "cart", "random_forest", "xgboost"],
                          help="Subset of models to train")
    args = parser.parse_args()

    cfg = load_config()
    log.info(f"Config loaded. Mode: {cfg['data']['mode']}")

    runners = {
        "ingest": lambda: stage_ingest(cfg, force=args.force),
        "enrich": lambda: stage_enrich(cfg),
        "features": lambda: stage_features(cfg),
        "train": lambda: stage_train(cfg, models=args.models, tune=not args.no_tune),
        "recommend": lambda: stage_recommend(cfg),
        "cluster": lambda: stage_cluster(cfg),
        "all": lambda: stage_all(cfg, force_ingest=args.force,
                                    models=args.models, tune=not args.no_tune),
    }

    runners[args.step]()
    log.info("Pipeline stage finished successfully")


if __name__ == "__main__":
    sys.exit(main() or 0)
