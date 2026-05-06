# =============================================================================
# Course   : CS 513 - Data Analytics & Machine Learning
# Purpose  : Tests for data ingestion + enrichment + validation
# =============================================================================

import pytest
import pandas as pd

from src.data.synthetic import generate_synthetic_dataset
from src.data.enrichment import PropertyEnricher
from src.data.validation import validate_properties


@pytest.fixture(scope="module")
def small_df():
    return generate_synthetic_dataset(n_samples=2000, random_seed=42)


def test_synthetic_size_and_shape(small_df):
    assert len(small_df) == 2000
    assert "price" in small_df.columns
    assert "borough" in small_df.columns


def test_borough_distribution(small_df):
    boroughs = set(small_df["borough"].unique())
    assert boroughs.issubset(
        {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}
    )


def test_no_negative_prices(small_df):
    assert (small_df["price"] > 0).all()
    assert (small_df["sqft"] > 0).all()


def test_lat_lng_in_nyc_bounds(small_df):
    assert small_df["lat"].between(40.4, 40.95).all()
    assert small_df["lng"].between(-74.30, -73.65).all()


def test_walkscore_in_range(small_df):
    assert small_df["walk_score"].between(0, 100).all()
    assert small_df["transit_score"].between(0, 100).all()


def test_price_correlates_with_features(small_df):
    """Price should positively correlate with walk/school, negatively with crime."""
    assert small_df["price_per_sqft"].corr(small_df["walk_score"]) > 0
    assert small_df["price_per_sqft"].corr(small_df["school_quality_score"]) > 0
    assert small_df["price_per_sqft"].corr(small_df["crime_rate_per_1k"]) < 0


def test_enrichment_adds_percentile_features(small_df):
    enriched = PropertyEnricher().enrich_all(small_df)
    expected = {
        "nta_proxy",
        "ppsf_neighborhood_percentile",
        "crime_neighborhood_percentile",
        "school_neighborhood_percentile",
        "income_neighborhood_percentile",
        "ppsf_neighborhood_median",
    }
    assert expected.issubset(set(enriched.columns))
    # percentiles bounded in [0, 1]
    for c in ["ppsf_neighborhood_percentile", "crime_neighborhood_percentile"]:
        assert enriched[c].between(0, 1).all()


def test_schema_validation_passes_on_clean_data(small_df):
    ok, msg = validate_properties(small_df, raise_on_error=False)
    assert ok, msg
