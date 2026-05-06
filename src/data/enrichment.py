from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import save_parquet, load_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


class PropertyEnricher:
    """Enrich raw property data with derived neighborhood-relative features."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()

    def add_neighborhood_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["nta_proxy"] = (
            df["borough"].astype(str)
            + "_"
            + (df["lat"].round(2)).astype(str)
            + "_"
            + (df["lng"].round(2)).astype(str)
        )
        n_neighborhoods = df["nta_proxy"].nunique()
        log.info(f"Created {n_neighborhoods:,} neighborhood proxy zones")
        return df

    # Percentile rankings within neighborhood
    def add_percentile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ppsf_neighborhood_percentile"] = df.groupby("nta_proxy")[
            "price_per_sqft"
        ].transform(lambda x: x.rank(pct=True))
        df["crime_neighborhood_percentile"] = df.groupby("nta_proxy")[
            "crime_rate_per_1k"
        ].transform(lambda x: x.rank(pct=True))
        df["school_neighborhood_percentile"] = df.groupby("nta_proxy")[
            "school_quality_score"
        ].transform(lambda x: x.rank(pct=True))
        df["income_neighborhood_percentile"] = df.groupby("nta_proxy")[
            "median_household_income"
        ].transform(lambda x: x.rank(pct=True))
        log.info("Added 4 neighborhood-percentile features")
        return df

    # Aggregate neighborhood medians
    def add_neighborhood_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        med = df.groupby("nta_proxy")["price_per_sqft"].median().rename(
            "ppsf_neighborhood_median"
        )
        df = df.merge(med, on="nta_proxy", how="left")
        return df

    
    # Public pipeline
    def enrich_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_neighborhood_id(df)
        df = self.add_percentile_features(df)
        df = self.add_neighborhood_medians(df)
        return df


def main():
    """CLI entrypoint: python -m src.data.enrichment"""
    cfg = load_config()
    raw_path = f"{cfg['paths']['data_raw']}/properties.parquet"
    out_path = f"{cfg['paths']['data_interim']}/properties_enriched.parquet"
    df = load_parquet(raw_path)
    enricher = PropertyEnricher(cfg)
    enriched = enricher.enrich_all(df)
    save_parquet(enriched, out_path)
    print(f"Enriched dataset shape: {enriched.shape}")
    print(enriched.head())


if __name__ == "__main__":
    main()
