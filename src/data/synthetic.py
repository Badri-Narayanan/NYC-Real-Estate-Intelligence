from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# NYC borough characteristics
BOROUGH_PROFILES = {
    "MANHATTAN": {
        "lat_range": (40.700, 40.880),
        "lng_range": (-74.020, -73.910),
        "ppsf_mean": 1450, "ppsf_std": 550,
        "income_mean": 110000, "income_std": 60000,
        "crime_mean": 28, "crime_std": 12,
        "walk_mean": 92, "walk_std": 6,
        "transit_mean": 95, "transit_std": 4,
        "school_mean": 7.0, "school_std": 1.8,
        "weight": 0.18,
    },
    "BROOKLYN": {
        "lat_range": (40.570, 40.740),
        "lng_range": (-74.040, -73.860),
        "ppsf_mean": 950, "ppsf_std": 420,
        "income_mean": 78000, "income_std": 42000,
        "crime_mean": 22, "crime_std": 10,
        "walk_mean": 84, "walk_std": 11,
        "transit_mean": 82, "transit_std": 12,
        "school_mean": 6.5, "school_std": 1.7,
        "weight": 0.30,
    },
    "QUEENS": {
        "lat_range": (40.540, 40.800),
        "lng_range": (-73.970, -73.700),
        "ppsf_mean": 720, "ppsf_std": 280,
        "income_mean": 72000, "income_std": 35000,
        "crime_mean": 16, "crime_std": 8,
        "walk_mean": 72, "walk_std": 14,
        "transit_mean": 70, "transit_std": 16,
        "school_mean": 6.8, "school_std": 1.5,
        "weight": 0.28,
    },
    "BRONX": {
        "lat_range": (40.785, 40.915),
        "lng_range": (-73.935, -73.765),
        "ppsf_mean": 460, "ppsf_std": 200,
        "income_mean": 45000, "income_std": 25000,
        "crime_mean": 35, "crime_std": 14,
        "walk_mean": 75, "walk_std": 13,
        "transit_mean": 76, "transit_std": 13,
        "school_mean": 5.4, "school_std": 1.6,
        "weight": 0.16,
    },
    "STATEN ISLAND": {
        "lat_range": (40.495, 40.650),
        "lng_range": (-74.260, -74.050),
        "ppsf_mean": 540, "ppsf_std": 180,
        "income_mean": 88000, "income_std": 32000,
        "crime_mean": 12, "crime_std": 6,
        "walk_mean": 50, "walk_std": 18,
        "transit_mean": 45, "transit_std": 18,
        "school_mean": 7.2, "school_std": 1.3,
        "weight": 0.08,
    },
}

BUILDING_CLASSES = [
    "A1_1FAM", "A2_1FAM", "B1_2FAM", "B2_2FAM",
    "C0_WALKUP", "C1_WALKUP", "D1_ELEVATOR", "D4_ELEVATOR",
    "R1_CONDO", "R2_CONDO", "R4_CONDO", "S1_MIXED",
]


def _truncated_normal(mean: float, std: float, low: float, high: float,
                       size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from a truncated normal distribution."""
    samples = rng.normal(mean, std, size * 2)
    samples = samples[(samples >= low) & (samples <= high)]
    if len(samples) < size:
        # Fall back: clip
        extra = rng.normal(mean, std, size)
        samples = np.concatenate([samples, np.clip(extra, low, high)])
    return samples[:size]


def generate_synthetic_dataset(n_samples: int = 100_000,
                                random_seed: int = 42,
                                years: list[int] | None = None) -> pd.DataFrame:
    """
    Generate synthetic NYC real-estate data with realistic distributions
    AND realistic feature correlations (e.g. price correlates with walk_score,
    school_quality, and inversely with crime_rate).

    Parameters
    ----------
    n_samples : int
        Total rows to generate.
    random_seed : int
    years : list[int] or None
        Sale years to spread the data across. Defaults to [2022, 2023, 2024].

    Returns
    -------
    pd.DataFrame
    """
    if years is None:
        years = [2022, 2023, 2024]

    rng = np.random.default_rng(random_seed)
    log.info(f"Generating {n_samples:,} synthetic NYC properties (seed={random_seed})")

    # Allocate samples across boroughs by weight
    borough_names = list(BOROUGH_PROFILES.keys())
    weights = np.array([BOROUGH_PROFILES[b]["weight"] for b in borough_names])
    weights = weights / weights.sum()
    borough_counts = rng.multinomial(n_samples, weights)

    frames = []
    for borough, n in zip(borough_names, borough_counts):
        if n == 0:
            continue
        prof = BOROUGH_PROFILES[borough]

        # Spatial coordinates
        lat = rng.uniform(*prof["lat_range"], size=n)
        lng = rng.uniform(*prof["lng_range"], size=n)

        # Structural attributes
        sqft = _truncated_normal(1400, 700, 300, 8000, n, rng).astype(int)
        lot_sqft = _truncated_normal(2500, 1500, 500, 25000, n, rng).astype(int)
        year_built = rng.integers(1900, 2024, size=n)
        building_age = 2024 - year_built
        num_units = rng.choice([1, 2, 3, 4, 6, 8, 12, 20], size=n,
                                p=[0.30, 0.25, 0.10, 0.08, 0.10, 0.07, 0.06, 0.04])
        num_floors = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, 20], size=n,
                                 p=[0.10, 0.30, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02, 0.02])
        building_class = rng.choice(BUILDING_CLASSES, size=n)

        # Hyperlocal features (with realistic correlations)
        # Crime: borough baseline + spatial noise
        crime = _truncated_normal(prof["crime_mean"], prof["crime_std"], 1, 80, n, rng)
        # Walk score: borough baseline
        walk = _truncated_normal(prof["walk_mean"], prof["walk_std"], 10, 100, n, rng)
        # Transit: correlated with walk
        transit = np.clip(walk + rng.normal(0, 8, n), 5, 100)
        bike = np.clip(walk - rng.normal(5, 12, n), 0, 100)
        # School quality
        school = _truncated_normal(prof["school_mean"], prof["school_std"], 1, 10, n, rng)
        # Income (loosely correlated with school + inversely with crime)
        income = _truncated_normal(prof["income_mean"], prof["income_std"],
                                     20000, 350000, n, rng)
        # Distances
        nearest_subway_m = np.abs(rng.exponential(scale=400, size=n))
        nearest_subway_m = np.clip(nearest_subway_m, 50, 5000)
        nearest_park_m = np.abs(rng.exponential(scale=600, size=n))
        nearest_park_m = np.clip(nearest_park_m, 50, 4000)
        # Population density (people/km²)
        pop_density = _truncated_normal(25000, 12000, 1000, 110000, n, rng)
        # Flood zone
        flood = rng.choice([0, 1], size=n, p=[0.92, 0.08])

        # === Price computation: economically realistic ===
        # Base ppsf depends on borough + walk + transit + school - crime
        ppsf_base = prof["ppsf_mean"]
        ppsf = (
            ppsf_base
            + (walk - prof["walk_mean"]) * 4.0           # +$4 per walk score point
            + (transit - prof["transit_mean"]) * 3.0
            + (school - prof["school_mean"]) * 35.0      # +$35 per school point
            - (crime - prof["crime_mean"]) * 6.0         # -$6 per crime unit
            + (income - prof["income_mean"]) * 0.0015    # income premium
            - building_age * 1.2                         # age depreciation
            - flood * 60.0                                # flood discount
            - (nearest_subway_m - 400) * 0.05            # transit penalty
        )
        # Add a stochastic component (this is what creates the
        # under/fair/over-valued classes when we compute z-scores later)
        noise = rng.normal(0, prof["ppsf_std"] * 0.6, n)
        ppsf = ppsf + noise
        ppsf = np.clip(ppsf, 200, 5000)

        price = (ppsf * sqft).astype(int)

        # Sale dates
        sale_year = rng.choice(years, size=n)
        sale_quarter = rng.integers(1, 5, size=n)

        df = pd.DataFrame({
            "property_id": [f"{borough[:2]}{rng.integers(10**8, 10**9)}_{i:06d}"
                            for i in range(n)],
            "borough": borough,
            "lat": lat, "lng": lng,
            "sqft": sqft, "lot_sqft": lot_sqft,
            "year_built": year_built, "building_age": building_age,
            "num_units": num_units, "num_floors": num_floors,
            "building_class": building_class,
            "walk_score": walk.round(1), "transit_score": transit.round(1),
            "bike_score": bike.round(1),
            "nearest_subway_m": nearest_subway_m.round(0).astype(int),
            "nearest_park_m": nearest_park_m.round(0).astype(int),
            "crime_rate_per_1k": crime.round(2),
            "school_quality_score": school.round(2),
            "median_household_income": income.round(0).astype(int),
            "population_density": pop_density.round(0).astype(int),
            "flood_zone_flag": flood,
            "price": price,
            "price_per_sqft": (price / sqft).round(2),
            "sale_year": sale_year,
            "sale_quarter": sale_quarter,
            "floor_area_ratio": (sqft / lot_sqft).round(3),
        })
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    # Shuffle
    full = full.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    log.info(f"Generated dataset shape: {full.shape}")
    log.info(f"Borough distribution:\n{full['borough'].value_counts().to_string()}")
    return full
