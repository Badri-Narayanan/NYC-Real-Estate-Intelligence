# =============================================================================
# Purpose  : Data ingestion. Supports two modes:
#              "synthetic" - generates realistic fake data (works offline)
#              "real"      - downloads NYC DOF rolling sales (Excel)
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data.synthetic import generate_synthetic_dataset
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_parquet, load_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


class NYCDataIngester:
    """
    Pulls raw NYC property data from configured sources.
    Saves the canonical raw dataset to data/raw/properties.parquet.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.raw_dir = Path(self.cfg["paths"]["data_raw"])
        ensure_dir(self.raw_dir)

    # Public API
    def ingest(self, force: bool = False) -> pd.DataFrame:
        """Run ingestion. Returns the raw DataFrame."""
        out_path = self.raw_dir / "properties.parquet"
        if out_path.exists() and not force:
            log.info(f"Raw data exists at {out_path} (use force=True to re-ingest)")
            return load_parquet(out_path)

        mode = self.cfg["data"]["mode"]
        log.info(f"Ingesting in mode = '{mode}'")

        if mode == "synthetic":
            df = self._ingest_synthetic()
        elif mode == "real":
            df = self._ingest_real()
        else:
            raise ValueError(f"Unknown data mode: {mode}")

        save_parquet(df, out_path)
        return df

    # Synthetic mode
    def _ingest_synthetic(self) -> pd.DataFrame:
        n = self.cfg["data"]["synthetic_size"]
        seed = self.cfg["project"]["random_seed"]
        years = self.cfg["data"]["years"]
        return generate_synthetic_dataset(n_samples=n, random_seed=seed, years=years)

    # Real mode - now uses NYC Open Data Socrata API
    def _ingest_real(self) -> pd.DataFrame:
        """
        Pull recent sales from NYC Open Data (NYC DOF), enrich with synthetic
        hyperlocal features, then top-up with synthetic rows to reach the
        configured size target. This gives us a hybrid dataset that is
        partially REAL (price/sqft/borough/year_built) and partially
        SYNTHETIC (the hyperlocal columns NYC DOF doesn't publish).

        Falls back to fully-synthetic if the live feed is unreachable.
        """
        from src.data.live_integration import fetch_and_integrate

        target = self.cfg["data"]["synthetic_size"]
        log.info(f"Real-mode ingestion: target size = {target:,} rows")

        # Fetch as much as Socrata will give us in one call (max 50k)
        live_df, result = fetch_and_integrate(
            limit=min(target, 50_000),
            since=None,           # let server return whatever's most recent
            random_seed=self.cfg["project"]["random_seed"],
        )

        if live_df.empty or result.error:
            log.warning(
                f"Live feed returned no usable data ({result.error}). "
                f"Falling back to synthetic."
            )
            return self._ingest_synthetic()

        log.info(f"Pulled {len(live_df):,} live records from "
                 f"NYC Open Data (source={result.source})")

        # If we have fewer real rows than target, top up with synthetic
        if len(live_df) < target:
            extra = target - len(live_df)
            log.info(f"Topping up with {extra:,} synthetic rows")
            synth = self._ingest_synthetic_n(extra)
            synth["data_source"] = "synthetic"
            # Align columns
            common = [c for c in synth.columns if c in live_df.columns]
            combined = pd.concat([live_df[common], synth[common]],
                                   ignore_index=True)
            return combined

        return live_df

    def _ingest_synthetic_n(self, n: int) -> pd.DataFrame:
        """Override of synthetic generator to produce a specific row count."""
        from src.data.synthetic import generate_synthetic_dataset
        return generate_synthetic_dataset(
            n_samples=n,
            random_seed=self.cfg["project"]["random_seed"],
            years=self.cfg["data"]["years"],
        )


def main():
    """CLI entrypoint: python -m src.data.ingestion"""
    ingester = NYCDataIngester()
    df = ingester.ingest(force=True)
    print(df.head())
    print(f"\nShape: {df.shape}")


if __name__ == "__main__":
    main()
