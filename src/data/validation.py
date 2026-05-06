from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# Optional dep - pandera. Provide a degraded validator if it's not installed.
try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    _HAVE_PANDERA = True
except ImportError:
    _HAVE_PANDERA = False
    log.warning("pandera not installed - using lightweight built-in validator")


# Canonical schema for the post-ingestion DataFrame
if _HAVE_PANDERA:
    PROPERTY_SCHEMA = DataFrameSchema(
        {
            "property_id": Column(str, nullable=False),
            "borough": Column(str, Check.isin(
                ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]
            )),
            "lat": Column(float, Check.in_range(40.4, 40.95)),
            "lng": Column(float, Check.in_range(-74.30, -73.65)),
            "sqft": Column(int, Check.gt(0)),
            "lot_sqft": Column(int, Check.gt(0)),
            "year_built": Column(int, Check.in_range(1800, 2025)),
            "num_units": Column(int, Check.ge(1)),
            "num_floors": Column(int, Check.ge(1)),
            "walk_score": Column(float, Check.in_range(0, 100)),
            "transit_score": Column(float, Check.in_range(0, 100)),
            "bike_score": Column(float, Check.in_range(0, 100)),
            "crime_rate_per_1k": Column(float, Check.ge(0)),
            "school_quality_score": Column(float, Check.in_range(0, 10)),
            "median_household_income": Column(int, Check.gt(0)),
            "price": Column(int, Check.gt(0)),
            "price_per_sqft": Column(float, Check.gt(0)),
            "flood_zone_flag": Column(int, Check.isin([0, 1])),
        },
        strict=False,
        coerce=True,
    )
else:
    PROPERTY_SCHEMA = None


def _builtin_validate(df: pd.DataFrame) -> tuple[bool, str]:
    """Lightweight validator used when pandera isn't available."""
    errors = []
    required = ["property_id", "borough", "lat", "lng", "sqft", "price"]
    for c in required:
        if c not in df.columns:
            errors.append(f"missing required column: {c}")

    if "borough" in df.columns:
        valid = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}
        bad = set(df["borough"].dropna().unique()) - valid
        if bad:
            errors.append(f"invalid boroughs: {bad}")

    if "price" in df.columns and (df["price"] <= 0).any():
        errors.append("found non-positive prices")

    if "sqft" in df.columns and (df["sqft"] <= 0).any():
        errors.append("found non-positive sqft")

    if "lat" in df.columns and not df["lat"].between(40.4, 40.95).all():
        errors.append("latitudes out of NYC bounds")

    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def validate_properties(df: pd.DataFrame, raise_on_error: bool = False
                          ) -> tuple[bool, str]:
    """Validate a properties DataFrame. Returns (ok, message)."""
    if _HAVE_PANDERA:
        try:
            PROPERTY_SCHEMA.validate(df, lazy=True)
            log.info(f"Schema validation passed (pandera) for {len(df):,} rows.")
            return True, "ok"
        except pa.errors.SchemaErrors as exc:
            msg = f"Schema validation failed:\n{exc.failure_cases.head(20)}"
            log.error(msg)
            if raise_on_error:
                raise
            return False, msg
    # Fallback
    ok, msg = _builtin_validate(df)
    if ok:
        log.info(f"Schema validation passed (builtin) for {len(df):,} rows.")
    else:
        log.error(f"Schema validation failed: {msg}")
        if raise_on_error:
            raise ValueError(msg)
    return ok, msg
