from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.synthetic import BOROUGH_PROFILES
from src.utils.logger import get_logger

log = get_logger(__name__)

CACHE_DIR = Path("data/external/hyperlocal_cache")
CACHE_TTL  = 7 * 24 * 3600   # 7 days

# Precinct → zip lookup (used for crime rate)
PRECINCT_TO_ZIPS: dict[int, list[str]] = {
    1:["10006","10007","10038"], 5:["10002","10012","10013"],
    6:["10011","10014"], 7:["10002","10009"], 9:["10003","10009"],
    10:["10011","10001"], 13:["10003","10010","10016"],
    14:["10036","10019"], 17:["10017","10022"], 18:["10019","10020"],
    19:["10021","10028","10065"], 20:["10023","10024"],
    22:["10024","10025"], 23:["10026","10027","10029","10030"],
    24:["10025","10031"], 25:["10029","10035"],
    26:["10026","10030","10037"], 28:["10027","10037","10039"],
    30:["10031","10032","10033"], 32:["10027","10039"],
    33:["10034","10040"], 34:["10032","10033","10034"],
    40:["10454","10455"], 41:["10454","10459"],
    42:["10454","10455","10456"], 43:["10455","10459","10460"],
    44:["10452","10453","10456"], 45:["10461","10462"],
    46:["10452","10453"], 47:["10471","10463"], 48:["10457","10458"],
    49:["10469","10470","10475"], 50:["10463","10468"],
    52:["10460","10462","10473"],
    60:["11220","11232"], 61:["11204","11219"],
    62:["11220","11230","11234"], 63:["11226","11210","11225"],
    66:["11215","11217","11231"], 67:["11213","11216","11233"],
    68:["11209","11220"], 69:["11210","11225","11226"],
    70:["11228","11214"], 71:["11226","11203"],
    72:["11218","11219"], 73:["11207","11212"],
    75:["11207","11208"], 76:["11201","11215","11231"],
    77:["11217","11238"], 78:["11201","11215","11217"],
    79:["11212","11213","11233"], 81:["11221","11237"],
    83:["11385","11237"], 84:["11201","11205","11217"],
    88:["11206","11211","11221"], 90:["11211","11222"],
    94:["11222","11211"],
    100:["11414","11415"], 101:["11417","11420"],
    102:["11418","11419","11421"], 103:["11415","11418"],
    104:["11375","11374"], 105:["11411","11413","11422"],
    106:["11412","11433","11434"], 107:["11364","11365","11366"],
    108:["11368","11369","11370"], 109:["11367","11375","11379"],
    110:["11354","11355","11356"], 111:["11357","11358","11360"],
    112:["11373","11374"],
    113:["11432","11433","11434","11435"],
    114:["11101","11102","11103","11104","11105","11106"],
    115:["11368","11372","11373"],
    120:["10301","10302","10303"], 121:["10305","10306","10307","10308"],
    122:["10309","10310","10311","10312"],
    123:["10303","10304","10314"],
}

ZIP_TO_PRECINCTS: dict[str, list[int]] = {}
for _pct, _zips in PRECINCT_TO_ZIPS.items():
    for _z in _zips:
        ZIP_TO_PRECINCTS.setdefault(_z, []).append(_pct)


# Cache helpers
def _cache_get(key: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _cache_set(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(data))
    except Exception as e:
        log.warning(f"Cache write failed {key}: {e}")

def _safe_key(s: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(s))[:80]


# Borough fallback
def _fallback(borough: str, field: str, rng: np.random.Generator) -> float:
    p = BOROUGH_PROFILES.get(str(borough).upper(),
                              BOROUGH_PROFILES["BROOKLYN"])
    M = {
        "walk_score":              (p["walk_mean"],    p["walk_std"],    10,   100),
        "transit_score":           (p["transit_mean"], p["transit_std"],  5,   100),
        "bike_score":              (p["walk_mean"]-5,  15,                0,   100),
        "crime_rate_per_1k":       (p["crime_mean"],   p["crime_std"],    1,    80),
        "school_quality_score":    (p["school_mean"],  p["school_std"],   1,    10),
        "median_household_income": (p["income_mean"],  p["income_std"], 20000, 350000),
        "population_density":      (25000, 12000, 1000, 110000),
    }
    if field not in M:
        return 0.0
    mn, sd, lo, hi = M[field]
    return float(np.clip(rng.normal(mn, sd), lo, hi))

def _socrata_headers() -> dict:
    h = {"Accept": "application/json",
         "User-Agent": "CS513-RealEstate-ML/1.0"}
    tok = os.getenv("SOCRATA_APP_TOKEN", "")
    if tok:
        h["X-App-Token"] = tok
    return h


# Walk Score API

_WS_FAILURE_COUNT = 0
_WS_DISABLED      = False
_WS_FAILURE_THRESHOLD = 10


def fetch_walk_scores(lat: float, lng: float,
                       address: str = "", borough: str = "BROOKLYN",
                       rng: np.random.Generator | None = None) -> dict:
    """
    Call Walk Score API per property. Includes circuit breaker: if the API
    times out 10 times in a row, the rest of the batch falls back to synthetic
    to avoid multi-hour runtimes.
    """
    global _WS_FAILURE_COUNT, _WS_DISABLED

    rng = rng or np.random.default_rng(42)
    key = os.getenv("WALKSCORE_API_KEY", "")

    def _synthetic():
        return {"walk_score":    round(_fallback(borough, "walk_score",    rng), 1),
                "transit_score": round(_fallback(borough, "transit_score", rng), 1),
                "bike_score":    round(_fallback(borough, "bike_score",    rng), 1),
                "source": "synthetic_fallback"}

    if not key or _WS_DISABLED:
        return _synthetic()

    ck = _safe_key(f"ws_{lat:.4f}_{lng:.4f}")
    cached = _cache_get(ck)
    if cached:
        return cached

    try:
        r = requests.get("https://api.walkscore.com/score", timeout=15, params={
            "wsapikey": key, "address": address or f"{lat},{lng}",
            "lat": lat, "lon": lng, "format": "json", "transit": 1, "bike": 1,
        })
        r.raise_for_status()
        d = r.json()
        if d.get("status") not in (1, 2):
            raise ValueError(f"status={d.get('status')}")
        result = {
            "walk_score":    float(d.get("walkscore") or
                                    _fallback(borough, "walk_score", rng)),
            "transit_score": float((d.get("transit") or {}).get("score") or
                                    _fallback(borough, "transit_score", rng)),
            "bike_score":    float((d.get("bike") or {}).get("score") or
                                    _fallback(borough, "bike_score", rng)),
            "source": "walkscore_api",
        }
        _cache_set(ck, result)
        _WS_FAILURE_COUNT = 0   # reset on success
        return result
    except Exception as e:
        _WS_FAILURE_COUNT += 1
        if _WS_FAILURE_COUNT <= 3:
            log.warning(f"WalkScore API failed ({lat:.4f},{lng:.4f}): {e}")
        elif _WS_FAILURE_COUNT == _WS_FAILURE_THRESHOLD:
            _WS_DISABLED = True
            log.warning(
                f"WalkScore API has failed {_WS_FAILURE_THRESHOLD} times consecutively — "
                f"disabling for the rest of this run. Remaining properties will use "
                f"borough-calibrated synthetic walk scores."
            )
        return _synthetic()


# Census ACS5 API
def fetch_census_data(zip_code: str, borough: str = "BROOKLYN",
                       rng: np.random.Generator | None = None) -> dict:
    """
    Census ACS5 2022. Variables B19013_001E (median income) and
    B01003_001E (total population). Geography: zip code tabulation area.
    """
    rng = rng or np.random.default_rng(42)
    key = os.getenv("CENSUS_API_KEY", "")
    z   = str(zip_code).strip().zfill(5) if zip_code else ""

    if not z or not key:
        if not key:
            log.debug("CENSUS_API_KEY not set")
        return {"median_household_income": int(_fallback(borough, "median_household_income", rng)),
                "population_density":      int(_fallback(borough, "population_density", rng)),
                "source": "synthetic_fallback"}

    ck = _safe_key(f"census_{z}")
    cached = _cache_get(ck)
    if cached:
        return cached

    try:
        r = requests.get("https://api.census.gov/data/2022/acs/acs5", timeout=12,
                          params={"get": "B19013_001E,B01003_001E",
                                  "for": f"zip code tabulation area:{z}",
                                  "key": key})
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 2:
            raise ValueError("no data rows")
        row = dict(zip(rows[0], rows[1]))
        income = int(row.get("B19013_001E", -1) or -1)
        pop    = int(row.get("B01003_001E",  -1) or -1)
        if income < 0:
            income = int(_fallback(borough, "median_household_income", rng))
        if pop < 0:
            pop = 50000
        result = {"median_household_income": income,
                  "population_density":      int(min(pop * 12, 110000)),
                  "source": "census_api"}
        _cache_set(ck, result)
        log.debug(f"Census zip={z} income={income:,}")
        return result
    except Exception as e:
        log.warning(f"Census failed zip={z}: {e}")
        return {"median_household_income": int(_fallback(borough, "median_household_income", rng)),
                "population_density":      int(_fallback(borough, "population_density", rng)),
                "source": "synthetic_fallback"}


# Crime rate — NYPD Complaint Data Historic
_CRIME_COUNT_CACHE: dict[str, int] | None = None


def _bulk_load_crime_counts() -> dict[str, int]:
    """One bulk SoQL query that returns 2023 complaint counts by precinct."""
    global _CRIME_COUNT_CACHE
    if _CRIME_COUNT_CACHE is not None:
        return _CRIME_COUNT_CACHE

    disk = _cache_get("crime_bulk_v1")
    if disk and isinstance(disk, dict) and "counts" in disk:
        _CRIME_COUNT_CACHE = disk["counts"]
        log.info(f"Loaded crime data from cache: {len(_CRIME_COUNT_CACHE)} precincts")
        return _CRIME_COUNT_CACHE

    
    log.info("Bulk-fetching NYPD complaint counts by precinct (one-time)...")
    counts: dict[str, int] = {}
    last_err: Exception | None = None
    
    for dataset_id, label in [("5uac-w243", "YTD"), ("qgea-i56i", "Historic")]:
        try:
            r = requests.get(
                f"https://data.cityofnewyork.us/resource/{dataset_id}.json",
                headers=_socrata_headers(), timeout=120,
                params={
                    "$select": "addr_pct_cd,count(*) AS n",
                    "$group":  "addr_pct_cd",
                    "$limit":  500,
                }
            )
            r.raise_for_status()
            rows = r.json()
            for row in rows:
                pct = str(row.get("addr_pct_cd", "")).strip()
                n   = int(row.get("n", 0) or 0)
                if pct:
                    counts[pct] = n
            if counts:
                _CRIME_COUNT_CACHE = counts
                _cache_set("crime_bulk_v1", {"counts": counts})
                log.info(f"Crime data loaded ({label} dataset): {sum(counts.values()):,} complaints across {len(counts)} precincts")
                return counts
        except Exception as e:
            last_err = e
            log.warning(f"Crime fetch from {dataset_id} ({label}) failed: {e}")
            continue
    
    log.warning(f"All crime data fetches failed (last error: {last_err}) — using synthetic fallback")
    _CRIME_COUNT_CACHE = {}
    return _CRIME_COUNT_CACHE


def fetch_crime_rate(zip_code: str, borough: str = "BROOKLYN",
                      rng: np.random.Generator | None = None) -> dict:
    """
    Crime rate per 1k residents for a zip code.
    Uses bulk-prefetched precinct counts; aggregates over precincts that
    cover this zip via ZIP_TO_PRECINCTS.
    """
    rng = rng or np.random.default_rng(42)
    z   = str(zip_code).strip().zfill(5) if zip_code else ""
    if not z:
        return {"crime_rate_per_1k": round(_fallback(borough, "crime_rate_per_1k", rng), 2),
                "source": "synthetic_fallback"}

    precincts = ZIP_TO_PRECINCTS.get(z, [])
    if not precincts:
        return {"crime_rate_per_1k": round(_fallback(borough, "crime_rate_per_1k", rng), 2),
                "source": "synthetic_fallback_no_precinct"}

    counts = _bulk_load_crime_counts()
    if not counts:
        return {"crime_rate_per_1k": round(_fallback(borough, "crime_rate_per_1k", rng), 2),
                "source": "synthetic_fallback"}

    total = sum(counts.get(str(p), 0) for p in precincts)
    pop_est = 50000
    rate = float(np.clip(round(total / pop_est * 1000, 2), 1.0, 80.0))

    return {"crime_rate_per_1k": rate,
            "complaint_count": total,
            "precincts": precincts,
            "source": "nyc_open_data_complaints"}


# School quality — DOE School Locations
_SCHOOL_COUNT_CACHE: dict[str, int] | None = None


def _bulk_load_school_counts() -> dict[str, int]:
    """One-time bulk fetch of all NYC schools, grouped by zip.
    Cached in module memory for the lifetime of the process."""
    global _SCHOOL_COUNT_CACHE
    if _SCHOOL_COUNT_CACHE is not None:
        return _SCHOOL_COUNT_CACHE

    # Try disk cache first
    disk = _cache_get("school_bulk_v1")
    if disk and isinstance(disk, dict) and "counts" in disk:
        _SCHOOL_COUNT_CACHE = disk["counts"]
        log.info(f"Loaded school directory from cache: {len(_SCHOOL_COUNT_CACHE)} zips")
        return _SCHOOL_COUNT_CACHE

    log.info("Bulk-fetching NYC DOE school directory (one-time, ~2-3 sec)...")
    try:
        r = requests.get(
            "https://data.cityofnewyork.us/resource/s3k6-pzi2.json",
            headers=_socrata_headers(), timeout=60,    # generous one-time timeout
            params={"$select": "dbn,zip", "$limit": 5000}
        )
        r.raise_for_status()
        schools = r.json()
        counts: dict[str, int] = {}
        for s in schools:
            z = str(s.get("zip", "")).strip().zfill(5)
            if z and z != "00000":
                counts[z] = counts.get(z, 0) + 1
        _SCHOOL_COUNT_CACHE = counts
        _cache_set("school_bulk_v1", {"counts": counts})
        log.info(f"School directory loaded: {len(schools)} schools across {len(counts)} zips")
        return counts
    except Exception as e:
        log.warning(f"Bulk school fetch failed: {e} — will use per-zip fallback")
        _SCHOOL_COUNT_CACHE = {}
        return _SCHOOL_COUNT_CACHE


def fetch_school_quality(zip_code: str, borough: str = "BROOKLYN",
                          rng: np.random.Generator | None = None) -> dict:
    """
    School quality proxy from school count per zip.
    Uses bulk-prefetched count map to avoid 173 individual API calls.
    Score formula:
        baseline (borough average) + density adjustment
            +1.0 if 5+ schools
            +0.5 if 3-4 schools
             0.0 if 1-2 schools
            -1.0 if 0 schools
    """
    rng = rng or np.random.default_rng(42)
    z   = str(zip_code).strip().zfill(5) if zip_code else ""
    if not z:
        return {"school_quality_score": round(_fallback(borough, "school_quality_score", rng), 2),
                "source": "synthetic_fallback"}

    counts = _bulk_load_school_counts()

    if not counts:
        # Bulk fetch failed entirely — fall back
        return {"school_quality_score": round(_fallback(borough, "school_quality_score", rng), 2),
                "source": "synthetic_fallback"}

    n_schools = counts.get(z, 0)
    baseline  = float(BOROUGH_PROFILES.get(str(borough).upper(),
                       BOROUGH_PROFILES["BROOKLYN"])["school_mean"])

    if   n_schools >= 5: adj = +1.0
    elif n_schools >= 3: adj = +0.5
    elif n_schools >= 1: adj =  0.0
    else:                adj = -1.0

    score = float(np.clip(round(baseline + adj, 2), 1.0, 10.0))
    return {"school_quality_score": score,
            "n_schools": n_schools,
            "source": "nyc_open_data_doe_directory"}


# Batch enricher
def enrich_dataframe(df: pd.DataFrame,
                      random_seed: int = 42,
                      rate_limit_delay: float = 0.2) -> pd.DataFrame:
    """
    Enrich a DataFrame with real hyperlocal features.

    - Census, crime, school quality: one call per unique zip code.
    - Walk Score: one call per property (lat/lng specific); cached.

    All calls fall back gracefully to borough-calibrated synthetic values
    when the API is unreachable or returns no data.
    """
    rng = np.random.default_rng(random_seed)
    df  = df.copy()

    if "zip_code" not in df.columns:
        df["zip_code"] = ""
    df["zip_code"] = (df["zip_code"].fillna("").astype(str)
                       .str.strip().str.zfill(5))
    df["zip_code"] = df["zip_code"].replace("00000", "")

    # Per-zip fetches
    unique_zips = [z for z in df["zip_code"].unique() if z and z != "00000"]
    log.info(f"Fetching Census + crime + school for {len(unique_zips)} unique zip codes...")

    zip_census: dict[str, dict] = {}
    zip_crime:  dict[str, dict] = {}
    zip_school: dict[str, dict] = {}

    for z in unique_zips:
        rows = df[df["zip_code"] == z]
        borough = (str(rows["borough"].iloc[0]).upper()
                   if not rows.empty else "BROOKLYN")
        zip_census[z] = fetch_census_data(z, borough, rng)
        zip_crime[z]  = fetch_crime_rate(z, borough, rng)
        zip_school[z] = fetch_school_quality(z, borough, rng)

    # Walk Score per property
    ws_key = os.getenv("WALKSCORE_API_KEY", "")
    if ws_key:
        log.info(f"Fetching Walk Scores for {len(df)} properties via API...")
    else:
        log.info("WALKSCORE_API_KEY not set — using synthetic walk scores")

    walk_sc, transit_sc, bike_sc = [], [], []
    for _, row in df.iterrows():
        borough = str(row.get("borough", "BROOKLYN")).upper()
        ws = fetch_walk_scores(
            float(row.get("lat", 40.7)),
            float(row.get("lng", -74.0)),
            str(row.get("address", "")),
            borough, rng
        )
        walk_sc.append(ws["walk_score"])
        transit_sc.append(ws["transit_score"])
        bike_sc.append(ws["bike_score"])
        if ws_key:
            time.sleep(rate_limit_delay)

    df["walk_score"]    = np.round(walk_sc, 1)
    df["transit_score"] = np.round(transit_sc, 1)
    df["bike_score"]    = np.round(bike_sc, 1)

    # Map zip-level results back to rows
    def _get(z, mapping, field, borough, fb):
        e = mapping.get(str(z).strip(), {})
        return e[field] if field in e else _fallback(borough, fb, rng)

    df["median_household_income"] = df.apply(
        lambda r: int(_get(r["zip_code"], zip_census,
                           "median_household_income",
                           r.get("borough","BROOKLYN"),
                           "median_household_income")), axis=1)

    df["population_density"] = df.apply(
        lambda r: int(_get(r["zip_code"], zip_census,
                           "population_density",
                           r.get("borough","BROOKLYN"),
                           "population_density")), axis=1)

    df["crime_rate_per_1k"] = df.apply(
        lambda r: round(float(_get(r["zip_code"], zip_crime,
                                    "crime_rate_per_1k",
                                    r.get("borough","BROOKLYN"),
                                    "crime_rate_per_1k")), 2), axis=1)

    df["school_quality_score"] = df.apply(
        lambda r: round(float(_get(r["zip_code"], zip_school,
                                    "school_quality_score",
                                    r.get("borough","BROOKLYN"),
                                    "school_quality_score")), 2), axis=1)

    # Summary log
    n_cen  = sum(1 for v in zip_census.values() if v.get("source") == "census_api")
    n_cri  = sum(1 for v in zip_crime.values()
                   if "nyc_open_data" in v.get("source",""))
    n_sch  = sum(1 for v in zip_school.values()
                   if "nyc_open_data" in v.get("source", ""))

    log.info(
        f"Hyperlocal enrichment done — "
        f"Census: {n_cen}/{len(unique_zips)} real | "
        f"Crime:  {n_cri}/{len(unique_zips)} real | "
        f"School: {n_sch}/{len(unique_zips)} real | "
        f"WalkScore: {'API' if ws_key else 'synthetic'}"
    )
    return df