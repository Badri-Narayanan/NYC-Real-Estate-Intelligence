# =============================================================================
# Course   : CS 513 - Data Analytics & Machine Learning
# Purpose  : Tests for live data integration (Socrata client + integrator)
#
# These tests are designed to PASS in a sandboxed/offline environment by
# verifying graceful degradation. They also exercise the data shape contract
# so that on a machine WITH internet, a live response will be cleanly
# integrated into the pipeline.
# =============================================================================

import pytest
import pandas as pd

from src.data.socrata_client import SocrataClient, FetchResult
from src.data.live_integration import integrate_live_with_features


@pytest.fixture(scope="module")
def offline_client():
    """A client that will fail to reach the network; tests its graceful degradation."""
    return SocrataClient(timeout=2, max_retries=1)


def test_socrata_client_instantiates():
    c = SocrataClient()
    assert c is not None
    assert c.cache_dir.exists()


def test_socrata_returns_fetchresult_on_failure(offline_client):
    """When network fails, must return a FetchResult, never raise."""
    result = offline_client.fetch_recent_sales(limit=5, use_cache=False)
    assert isinstance(result, FetchResult)
    assert result.source in ("synthetic_fallback", "live", "cache")
    if result.source == "synthetic_fallback":
        assert result.error is not None
        assert isinstance(result.df, pd.DataFrame)
        assert result.df.empty


def test_freshness_probe_returns_dict(offline_client):
    """Metadata probe must return a dict even on network failure."""
    meta = offline_client.get_dataset_freshness()
    assert isinstance(meta, dict)
    assert "available" in meta


def test_borough_code_mapping():
    """The numeric borough codes used by NYC DOF must round-trip correctly."""
    from src.data.socrata_client import BOROUGH_CODE_MAP
    assert BOROUGH_CODE_MAP["1"] == "MANHATTAN"
    assert BOROUGH_CODE_MAP["3"] == "BROOKLYN"
    assert SocrataClient._borough_code("BROOKLYN") == "3"


def test_integrate_with_synthetic_features_shape():
    """If we hand the integrator a fake live row, it must add hyperlocal cols."""
    fake_live = pd.DataFrame([
        {"borough": "BROOKLYN", "sqft": 1500, "lot_sqft": 2200, "year_built": 1990,
         "num_units": 2, "price": 950_000, "price_per_sqft": 633.33,
         "sale_date": pd.Timestamp("2025-09-01")},
        {"borough": "QUEENS",  "sqft": 1200, "lot_sqft": 1800, "year_built": 1985,
         "num_units": 1, "price": 600_000, "price_per_sqft": 500.00,
         "sale_date": pd.Timestamp("2025-08-15")},
    ])
    integrated = integrate_live_with_features(fake_live)
    assert len(integrated) == 2
    must_exist = {"lat", "lng", "walk_score", "transit_score", "crime_rate_per_1k",
                  "school_quality_score", "median_household_income", "data_source"}
    assert must_exist.issubset(integrated.columns), \
        f"Missing: {must_exist - set(integrated.columns)}"
    assert (integrated["data_source"] == "live_nyc_dof").all()
    # Price preserved exactly (real data is preserved as-is)
    assert integrated.iloc[0]["price"] == 950_000


def test_integrate_handles_empty_input():
    integrated = integrate_live_with_features(pd.DataFrame())
    assert integrated.empty
