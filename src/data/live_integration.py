from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.socrata_client import SocrataClient, FetchResult
from src.data.synthetic import BOROUGH_PROFILES, _truncated_normal
from src.data.hyperlocal_enricher import enrich_dataframe
from src.utils.logger import get_logger

log = get_logger(__name__)


def integrate_live_with_features(live_df: pd.DataFrame,
                                    random_seed: int = 42) -> pd.DataFrame:
    """
    Take a DataFrame of real NYC DOF sales (from SocrataClient.fetch_recent_sales)
    and add the hyperlocal features needed by the ML pipeline.

    Real fields (preserved as-is):
      borough, sqft, lot_sqft, year_built, num_units, price, price_per_sqft,
      sale_date, address, zip_code, neighborhood

    Synthesized fields (drawn from the borough's empirical distribution):
      lat, lng, walk_score, transit_score, bike_score, nearest_subway_m,
      nearest_park_m, crime_rate_per_1k, school_quality_score,
      median_household_income, population_density, flood_zone_flag,
      building_class (if missing)

    A `data_source` column is added:
      - 'live_nyc_dof' for the price/sqft/borough/year_built fields
      - synthesized fields are still drawn from real-borough distributions
    """
    if live_df.empty:
        log.warning("integrate_live_with_features: empty input")
        return live_df

    rng = np.random.default_rng(random_seed)
    df = live_df.copy().reset_index(drop=True)
    df["data_source"] = "live_nyc_dof"

    # Default any missing required fields
    if "num_units" not in df.columns or df["num_units"].isna().all():
        df["num_units"] = 1
    df["num_units"] = df["num_units"].fillna(1).clip(lower=1).astype(int)

    if "year_built" not in df.columns or df["year_built"].isna().all():
        df["year_built"] = 1970
    df["year_built"] = df["year_built"].fillna(1970).clip(1800, 2024).astype(int)
    df["building_age"] = 2024 - df["year_built"]

    if "lot_sqft" not in df.columns:
        df["lot_sqft"] = (df["sqft"] * 1.5).fillna(2000).astype(int)
    df["lot_sqft"] = df["lot_sqft"].fillna(df["sqft"] * 1.5).clip(lower=500).astype(int)

    df["num_floors"] = df["num_units"].clip(lower=1).apply(
        lambda u: max(1, int(u / 2))
    )

    if "building_class" not in df.columns or df["building_class"].isna().all():
        df["building_class"] = "Unknown"

    # Spatial coordinates: derive from borough bounding box if missing
    # (NYC DOF data does not include lat/lng — we approximate from borough)
    n = len(df)
    if "lat" not in df.columns or df["lat"].isna().all():
        lat_arr = np.empty(n, dtype=float)
        lng_arr = np.empty(n, dtype=float)
        for i, row in df.iterrows():
            borough = str(row.get("borough") or "BROOKLYN").upper()
            prof = BOROUGH_PROFILES.get(borough, BOROUGH_PROFILES["BROOKLYN"])
            lat_arr[i] = rng.uniform(*prof["lat_range"])
            lng_arr[i] = rng.uniform(*prof["lng_range"])
        df["lat"] = lat_arr
        df["lng"] = lng_arr

    # Distances: synthetic — no public API provides these per-property
    nearest_subway = np.clip(np.abs(rng.exponential(scale=400, size=n)), 50, 5000)
    nearest_park   = np.clip(np.abs(rng.exponential(scale=600, size=n)), 50, 4000)
    flood          = rng.choice([0, 1], size=n, p=[0.92, 0.08])
    df["nearest_subway_m"] = nearest_subway.astype(int)
    df["nearest_park_m"]   = nearest_park.astype(int)
    df["flood_zone_flag"]  = flood

    # Hyperlocal features: use real APIs (Walk Score, Census, NYC Open Data)
    # with graceful per-field fallback to synthetic borough averages.
    log.info("Fetching real hyperlocal features via Walk Score, Census, and NYC Open Data APIs...")
    df = enrich_dataframe(df, random_seed=random_seed)

    # Derived columns expected by the rest of the pipeline
    if "price_per_sqft" not in df.columns:
        df["price_per_sqft"] = (df["price"] / df["sqft"]).round(2)
    df["floor_area_ratio"] = (df["sqft"] / df["lot_sqft"]).round(3)

    if "sale_date" in df.columns:
        df["sale_year"] = df["sale_date"].dt.year.fillna(2024).astype(int)
        df["sale_quarter"] = df["sale_date"].dt.quarter.fillna(1).astype(int)
    else:
        df["sale_year"] = 2024
        df["sale_quarter"] = 1

    if "property_id" not in df.columns:
        df["property_id"] = [
            f"LIVE_{i:08d}_{int(row['price']):d}"
            for i, row in df.iterrows()
        ]

    log.info(f"Integrated {len(df):,} live NYC DOF rows with synthesized hyperlocal features")
    return df


def fetch_and_integrate(limit: int = 5000,
                          borough: str | None = None,
                          since: str | None = None,
                          random_seed: int = 42) -> tuple[pd.DataFrame, FetchResult]:
    """
    One-shot helper: query Socrata, integrate features, return ready-to-model DataFrame.
    Returns (df, FetchResult). df is empty on failure.
    """
    client = SocrataClient()
    result = client.fetch_recent_sales(limit=limit, borough=borough, since=since)
    if result.df.empty:
        return pd.DataFrame(), result
    integrated = integrate_live_with_features(result.df, random_seed=random_seed)
    return integrated, result