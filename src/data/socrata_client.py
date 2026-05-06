from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.logger import get_logger

log = get_logger(__name__)


# Public, official endpoint for NYC Citywide Rolling Calendar Sales
SOCRATA_DATASET_ID = "usep-8jbt"
SOCRATA_BASE = f"https://data.cityofnewyork.us/resource/{SOCRATA_DATASET_ID}.json"
SOCRATA_METADATA = f"https://data.cityofnewyork.us/api/views/{SOCRATA_DATASET_ID}"

# Borough codes used in NYC DOF data
BOROUGH_CODE_MAP = {
    "1": "MANHATTAN", "2": "BRONX", "3": "BROOKLYN",
    "4": "QUEENS", "5": "STATEN ISLAND",
    1: "MANHATTAN", 2: "BRONX", 3: "BROOKLYN",
    4: "QUEENS", 5: "STATEN ISLAND",
    "MANHATTAN": "MANHATTAN", "BRONX": "BRONX", "BROOKLYN": "BROOKLYN",
    "QUEENS": "QUEENS", "STATEN ISLAND": "STATEN ISLAND",
}


@dataclass
class FetchResult:
    """Bundle returned by the client. Keeps metadata about freshness + source."""
    df: pd.DataFrame
    source: str             # "live" | "cache" | "synthetic_fallback"
    fetched_at: datetime    # UTC
    record_count: int
    last_dataset_update: datetime | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class SocrataClient:
    """
    Thin, defensive wrapper around the NYC Open Data SODA API.

    Features:
      - Automatic retry with exponential backoff
      - Local disk cache (so repeated calls during one demo are instant)
      - Pagination (Socrata caps single requests at 50K rows)
      - Graceful degradation: returns FetchResult with `error` set, never raises
      - Dataset-freshness probe (separate metadata endpoint)
    """

    def __init__(self,
                  app_token: str | None = None,
                  cache_dir: str | Path = "data/external/socrata_cache",
                  cache_ttl_seconds: int = 3600,        # 1 hour default
                  timeout: int = 20,
                  max_retries: int = 3):
        self.app_token = app_token or os.getenv("SOCRATA_APP_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_seconds
        self.timeout = timeout
        self.max_retries = max_retries

    # Headers
    def _headers(self) -> dict:
        h = {"Accept": "application/json", "User-Agent": "CS513-RealEstate-ML/1.0"}
        if self.app_token:
            h["X-App-Token"] = self.app_token
        return h

    # Cache
    def _cache_path(self, key: str) -> Path:
        # sanitize: only alnum + dash
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:120]
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> tuple[list[dict], datetime] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.cache_ttl:
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cached_at = datetime.fromtimestamp(payload["cached_at"], tz=timezone.utc)
            return payload["records"], cached_at
        except Exception as e:
            log.warning(f"Cache read failed for {key}: {e}")
            return None

    def _write_cache(self, key: str, records: list[dict]) -> None:
        p = self._cache_path(key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"cached_at": time.time(), "records": records}, f)
        except Exception as e:
            log.warning(f"Cache write failed for {key}: {e}")

    # Low-level fetch with retry
    def _fetch_page(self, params: dict) -> list[dict]:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(SOCRATA_BASE, params=params,
                                      headers=self._headers(),
                                      timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    log.warning(f"Rate-limited (429). Attempt {attempt}/{self.max_retries}")
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                last_err = e
                log.warning(f"Fetch failed (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"All {self.max_retries} retries exhausted: {last_err}")


    # Dataset metadata (for "data freshness" badge in UI)

    def get_dataset_freshness(self) -> dict[str, Any]:
        """
        Return basic metadata about the source dataset (last update time,
        row count, etc). Returns {'available': False, 'error': ...} on failure.
        """
        try:
            resp = requests.get(SOCRATA_METADATA, headers=self._headers(),
                                  timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return {
                "available": True,
                "name": data.get("name"),
                "row_count": data.get("rowsUpdatedAt"),
                "last_update_utc": datetime.fromtimestamp(
                    data.get("rowsUpdatedAt", 0), tz=timezone.utc
                ).isoformat() if data.get("rowsUpdatedAt") else None,
                "category": data.get("category"),
                "view_url": f"https://data.cityofnewyork.us/d/{SOCRATA_DATASET_ID}",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


    # Public: query sales

    def fetch_recent_sales(self,
                            limit: int = 5000,
                            borough: str | None = None,
                            min_price: int = 100_000,
                            min_sqft: int = 200,
                            since: str | None = None,
                            use_cache: bool = True) -> FetchResult:
        """
        Pull recent NYC DOF sales. Returns a FetchResult.

        Parameters
        ----------
        limit : int       Max rows to return (Socrata API max per request: 50000)
        borough : str     One of MANHATTAN/BROOKLYN/QUEENS/BRONX/STATEN_ISLAND
                          or None for all
        min_price : int   Filter out tiny / family-transfer sales
        min_sqft : int    Filter out missing-sqft rows
        since : str       ISO date "YYYY-MM-DD" - only sales on/after this date
        use_cache : bool  Hit local disk cache first if not stale
        """
        cache_key = f"sales_{borough or 'all'}_{since or 'any'}_{limit}"
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                records, cached_at = cached
                log.info(f"Loaded {len(records):,} live records from cache "
                         f"(age: {datetime.now(timezone.utc) - cached_at})")
                df = self._to_dataframe(records)
                df = self._post_filter(df, min_price, min_sqft)
                return FetchResult(
                    df=df, source="cache", fetched_at=cached_at,
                    record_count=len(df),
                    notes=[f"Cache-hit; TTL={self.cache_ttl}s"],
                )

        # SoQL query
        wheres = []
        if borough:
            code = self._borough_code(borough)
            if code:
                wheres.append(f"borough='{code}'")
        if since:
            wheres.append(f"sale_date >= '{since}'")
        # Always exclude clearly-bogus rows server-side
        wheres.append(f"sale_price > {min_price}")
        params = {
            "$limit": min(limit, 50_000),
            "$order": "sale_date DESC",
        }
        if wheres:
            params["$where"] = " AND ".join(wheres)

        try:
            log.info(f"Querying NYC Open Data (Socrata) -> {params}")
            records = self._fetch_page(params)
            self._write_cache(cache_key, records)
            df = self._to_dataframe(records)
            df = self._post_filter(df, min_price, min_sqft)
            return FetchResult(
                df=df, source="live", fetched_at=datetime.now(timezone.utc),
                record_count=len(df),
                notes=[f"Fetched {len(records):,} raw records; "
                       f"{len(df):,} after sqft filter"],
            )
        except Exception as e:
            log.error(f"Live fetch failed: {e}")
            return FetchResult(
                df=pd.DataFrame(),
                source="synthetic_fallback",
                fetched_at=datetime.now(timezone.utc),
                record_count=0,
                error=str(e),
                notes=[
                    "NYC Open Data unreachable (network/firewall/rate-limit).",
                    "Pipeline will fall back to synthetic dataset.",
                ],
            )


    # Helpers

    @staticmethod
    def _borough_code(borough: str) -> str | None:
        """NYC DOF stores borough as a numeric code 1-5."""
        b = borough.upper().strip()
        rev = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3",
               "QUEENS": "4", "STATEN ISLAND": "5"}
        return rev.get(b)

    @staticmethod
    def _to_dataframe(records: list[dict]) -> pd.DataFrame:
        """Convert raw Socrata records -> clean typed DataFrame."""
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)

        # Map Socrata field names to our canonical schema
        rename = {
            "borough": "borough_code",
            "neighborhood": "neighborhood",
            "building_class_category": "building_class",
            "tax_class_at_present": "tax_class",
            "block": "block",
            "lot": "lot",
            "address": "address",
            "zip_code": "zip_code",
            "residential_units": "residential_units",
            "commercial_units": "commercial_units",
            "total_units": "num_units",
            "land_square_feet": "lot_sqft",
            "gross_square_feet": "sqft",
            "year_built": "year_built",
            "sale_price": "price",
            "sale_date": "sale_date",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Type coercion
        numeric_cols = ["price", "sqft", "lot_sqft", "year_built",
                         "num_units", "residential_units", "commercial_units"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

        if "borough_code" in df.columns:
            df["borough"] = df["borough_code"].map(BOROUGH_CODE_MAP)

        # Derived
        if "price" in df.columns and "sqft" in df.columns:
            with pd.option_context("mode.use_inf_as_na", True):
                df["price_per_sqft"] = df["price"] / df["sqft"]

        return df

    @staticmethod
    def _post_filter(df: pd.DataFrame, min_price: int, min_sqft: int) -> pd.DataFrame:
        """Apply sanity filters that are awkward server-side."""
        if df.empty:
            return df
        if "price" in df.columns:
            df = df[df["price"] >= min_price]
        if "sqft" in df.columns:
            df = df[df["sqft"] >= min_sqft]
        if "price_per_sqft" in df.columns:
            df = df[(df["price_per_sqft"] >= 50) & (df["price_per_sqft"] <= 5000)]
        return df.reset_index(drop=True)


# Stub clients for other providers (kept for transparency)
class StreetEasyStub:
    """
    StreetEasy does NOT offer a free public API as of 2024.
    Their official statement on their forums confirms this.

    Known options (all paid / restricted):
      - StreetEasy Premier Agent program (broker-only)
      - Third-party scrapers (RapidAPI, Apify) - violate ToS, not academically defensible

    If you have a paid agent account, override fetch() to call your endpoint.
    """
    available = False

    def fetch(self, *args, **kwargs):
        raise NotImplementedError(
            "StreetEasy has no free public API. "
            "See src/data/socrata_client.py docstring for details."
        )


class ZillowStub:
    """
    Zillow deprecated their public API in September 2021. The 'Zillow API'
    on RapidAPI is third-party scraping and violates Zillow's ToS.

    Bridge Interactive (Zillow's official partner program) is for licensed
    brokers/MLS members only.

    If you have a Bridge Interactive account, set ZILLOW_BRIDGE_TOKEN and
    override fetch().
    """
    available = False

    def fetch(self, *args, **kwargs):
        raise NotImplementedError(
            "Zillow public API was retired in 2021. "
            "Bridge Interactive requires broker/MLS credentials."
        )


class RedfinStub:
    """
    Redfin has no public API. They expose data only via their website's
    internal endpoints (not stable, not allowed for redistribution per ToS).
    """
    available = False

    def fetch(self, *args, **kwargs):
        raise NotImplementedError(
            "Redfin has no public API. Internal endpoints are not "
            "ToS-permitted for third-party use."
        )
