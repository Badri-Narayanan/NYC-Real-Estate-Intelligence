from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.classification.base import BaseClassifier
from src.models.recommendation.content_based import ContentBasedRecommender
from src.models.recommendation.collaborative import CollaborativeRecommender
from src.models.recommendation.hybrid import HybridRecommender
from src.models.clustering.kmeans import NeighborhoodKMeans
from src.utils.config import load_config
from src.utils.io import load_pickle, load_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


# Singleton registry
class _Registry:
    _instance = None

    def __init__(self):
        self.cfg = load_config()
        self.models_dir = Path(self.cfg["paths"]["models"])
        self._properties: pd.DataFrame | None = None
        self._classifier = None
        self._content = None
        self._collab = None
        self._hybrid = None
        self._kmeans = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def properties(self) -> pd.DataFrame:
        if self._properties is None:
            path = f"{self.cfg['paths']['data_processed']}/properties_features.parquet"
            self._properties = load_parquet(path)
        return self._properties

    @property
    def classifier(self):
        """Load best classifier (xgboost > random_forest > cart > knn)."""
        if self._classifier is None:
            for name in ["xgboost", "random_forest", "cart", "knn"]:
                p = self.models_dir / f"classifier_{name}.pkl"
                if p.exists():
                    payload = load_pickle(p)
                    self._classifier = payload["pipeline"]
                    log.info(f"Loaded classifier: {name}")
                    break
            if self._classifier is None:
                raise FileNotFoundError("No trained classifier found")
        return self._classifier

    @property
    def content(self) -> ContentBasedRecommender:
        if self._content is None:
            self._content = ContentBasedRecommender().load(
                self.models_dir / "recommender_content.pkl"
            )
        return self._content

    @property
    def collab(self) -> CollaborativeRecommender:
        if self._collab is None:
            self._collab = CollaborativeRecommender().load(
                self.models_dir / "recommender_collab.pkl"
            )
        return self._collab

    @property
    def hybrid(self) -> HybridRecommender:
        if self._hybrid is None:
            alpha = self.cfg["recommender"]["hybrid"]["alpha"]
            self._hybrid = HybridRecommender(self.content, self.collab, alpha=alpha)
        return self._hybrid

    @property
    def kmeans(self) -> NeighborhoodKMeans:
        if self._kmeans is None:
            self._kmeans = NeighborhoodKMeans().load(
                self.models_dir / "clustering_kmeans.pkl"
            )
        return self._kmeans


# Tool definitions
TOOL_SCHEMAS = [
    {
        "name": "classify_property",
        "description": (
            "Classify a single property as undervalued, fairly_valued, or "
            "overvalued. Returns the predicted class label and probabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borough": {"type": "string", "enum": [
                    "MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]},
                "sqft": {"type": "number", "description": "Square footage"},
                "price": {"type": "number", "description": "List or sale price (USD)"},
                "num_units": {"type": "integer", "default": 1},
                "year_built": {"type": "integer", "default": 1980},
                "walk_score": {"type": "number", "default": 70},
                "transit_score": {"type": "number", "default": 70},
                "crime_rate_per_1k": {"type": "number", "default": 20},
                "school_quality_score": {"type": "number", "default": 6},
                "median_household_income": {"type": "number", "default": 70000},
            },
            "required": ["borough", "sqft", "price"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Return top-N recommended properties matching the buyer's preferences. "
            "Use this whenever a user describes what they're looking for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_max": {"type": "number"},
                "borough": {"type": "string"},
                "bedrooms_min": {"type": "integer"},
                "prefer_walkable": {"type": "boolean", "default": False},
                "prefer_safe": {"type": "boolean", "default": False},
                "prefer_school": {"type": "boolean", "default": False},
                "prefer_transit": {"type": "boolean", "default": False},
                "prefer_undervalued": {"type": "boolean", "default": False},
                "top_n": {"type": "integer", "default": 5},
            },
            "required": [],
        },
    },
    {
        "name": "get_neighborhood_profile",
        "description": (
            "Return aggregate statistics (median price/sqft, crime, schools, walkability) "
            "for a NYC borough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borough": {"type": "string", "enum": [
                    "MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]},
            },
            "required": ["borough"],
        },
    },
    {
        "name": "compare_properties",
        "description": "Compare multiple properties side-by-side by their property_ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "property_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["property_ids"],
        },
    },
    {
        "name": "get_market_summary",
        "description": "Return overall NYC market statistics: total properties, label distribution, median prices.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_live_market_data",
        "description": (
            "Query the NYC Department of Finance LIVE feed (NYC Open Data / Socrata) "
            "for the most recent real-estate sales transactions. Use when the user "
            "asks about RECENT sales, what's happening NOW, or up-to-date market "
            "activity. Returns aggregated stats grouped by borough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borough": {
                    "type": "string",
                    "enum": ["MANHATTAN", "BROOKLYN", "QUEENS",
                              "BRONX", "STATEN ISLAND"],
                    "description": "Filter to a single borough; omit for all 5",
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to look. Defaults to 90.",
                    "default": 90,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records to fetch. Default 1000.",
                    "default": 1000,
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_data_freshness",
        "description": (
            "Check when the NYC Open Data sales dataset was last updated. "
            "Use when the user asks 'how recent is this data?' or wants "
            "to verify the live feed is working."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# Tool implementations
def _format_results(df: pd.DataFrame, columns: list[str]) -> str:
    keep = [c for c in columns if c in df.columns]
    if df.empty:
        return "(no results)"
    return df[keep].to_string(index=False)


def classify_property(borough: str, sqft: float, price: float,
                        num_units: int = 1, year_built: int = 1980,
                        walk_score: float = 70, transit_score: float = 70,
                        crime_rate_per_1k: float = 20,
                        school_quality_score: float = 6,
                        median_household_income: float = 70000,
                        **kwargs) -> str:
    reg = _Registry.instance()
    pipeline = reg.classifier
    df = reg.properties

    # Build a DataFrame row matching the training schema
    template = df.drop(columns=["valuation_label", "valuation_label_name",
                                  "peer_median_ppsf", "peer_std_ppsf",
                                  "peer_z", "ppsf_neighborhood_percentile",
                                  "ppsf_neighborhood_median",
                                  "price", "price_per_sqft", "price_per_unit",
                                  "property_id", "nta_proxy",
                                  "sale_year", "sale_quarter", "year_built",
                                  "unit_bucket"], errors="ignore").iloc[[0]].copy()

    template.iloc[0] = template.iloc[0]   # initialize to a real row

    # Override the user-provided fields
    overrides = {
        "borough": borough.upper(),
        "sqft": float(sqft),
        "lot_sqft": max(float(sqft) * 1.5, 1000),
        "num_units": int(num_units),
        "num_floors": max(1, int(num_units / 2)),
        "building_age": 2024 - int(year_built),
        "walk_score": float(walk_score),
        "transit_score": float(transit_score),
        "bike_score": float(walk_score),
        "crime_rate_per_1k": float(crime_rate_per_1k),
        "school_quality_score": float(school_quality_score),
        "median_household_income": float(median_household_income),
        "amenity_score": (walk_score / 100 + transit_score / 100
                          + school_quality_score / 10 - crime_rate_per_1k / 80),
    }
    for k, v in overrides.items():
        if k in template.columns:
            template[k] = v

    pred = pipeline.predict(template)[0]
    proba = pipeline.predict_proba(template)[0]
    labels = ["undervalued", "fairly_valued", "overvalued"]
    return json.dumps({
        "predicted_label": labels[int(pred)],
        "probabilities": {l: round(float(p), 3) for l, p in zip(labels, proba)},
        "input_summary": {
            "borough": borough, "sqft": sqft, "price": price,
            "implied_ppsf": round(price / sqft, 2),
        },
    }, indent=2)


def get_recommendations(top_n: int = 5, **prefs) -> str:
    reg = _Registry.instance()
    rec = reg.content.recommend_from_preferences(prefs, top_n=top_n)
    if rec.empty:
        return "(no properties match the constraints; try relaxing the budget or borough)"
    cols = ["property_id", "borough", "price", "sqft", "num_units",
            "valuation_label_name", "walk_score", "transit_score",
            "school_quality_score", "crime_rate_per_1k", "preference_score"]
    return _format_results(rec, cols)


def get_neighborhood_profile(borough: str, **kwargs) -> str:
    reg = _Registry.instance()
    df = reg.properties
    sub = df[df["borough"] == borough.upper()]
    if sub.empty:
        return f"No data for borough '{borough}'"
    profile = {
        "borough": borough.upper(),
        "n_properties": int(len(sub)),
        "median_price_usd": int(sub["price"].median()),
        "median_price_per_sqft": float(round(sub["price_per_sqft"].median(), 2)),
        "median_walk_score": float(round(sub["walk_score"].median(), 1)),
        "median_transit_score": float(round(sub["transit_score"].median(), 1)),
        "median_school_score": float(round(sub["school_quality_score"].median(), 2)),
        "median_crime_rate_per_1k": float(round(sub["crime_rate_per_1k"].median(), 2)),
        "median_household_income": int(sub["median_household_income"].median()),
        "label_distribution": sub["valuation_label_name"].value_counts(
            normalize=True
        ).round(3).to_dict(),
    }
    return json.dumps(profile, indent=2)


def compare_properties(property_ids: list, **kwargs) -> str:
    reg = _Registry.instance()
    df = reg.properties
    sub = df[df["property_id"].isin(property_ids)]
    if sub.empty:
        return "No matching property IDs found."
    cols = ["property_id", "borough", "price", "sqft", "price_per_sqft",
            "num_units", "walk_score", "transit_score",
            "school_quality_score", "crime_rate_per_1k",
            "valuation_label_name"]
    return _format_results(sub, cols)


def get_market_summary(**kwargs) -> str:
    reg = _Registry.instance()
    df = reg.properties
    return json.dumps({
        "total_properties": int(len(df)),
        "boroughs": df["borough"].value_counts().to_dict(),
        "median_price": int(df["price"].median()),
        "median_ppsf": float(round(df["price_per_sqft"].median(), 2)),
        "label_distribution": df["valuation_label_name"].value_counts(
            normalize=True
        ).round(3).to_dict(),
    }, indent=2)


def get_live_market_data(borough: str | None = None,
                           days_back: int = 90,
                           limit: int = 1000,
                           **kwargs) -> str:
    """Query NYC Open Data live feed for recent sales."""
    from datetime import datetime, timedelta
    from src.data.socrata_client import SocrataClient

    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    client = SocrataClient()
    result = client.fetch_recent_sales(limit=limit, borough=borough, since=since)

    if result.df.empty:
        return json.dumps({
            "live_data_available": False,
            "source": result.source,
            "error": result.error or "no records returned",
            "notes": result.notes,
            "advice": "The NYC Open Data feed appears unreachable. "
                       "I can answer using the historical/synthetic dataset instead.",
        }, indent=2)

    df = result.df
    summary = {
        "live_data_available": True,
        "source": "NYC Open Data (Socrata) - dataset usep-8jbt",
        "fetched_at_utc": result.fetched_at.isoformat(),
        "data_freshness": result.source,   # "live" or "cache"
        "filter": {"borough": borough or "all", "days_back": days_back},
        "n_transactions": int(len(df)),
        "median_price": int(df["price"].median()) if "price" in df else None,
        "median_ppsf": float(round(df["price_per_sqft"].median(), 2))
                          if "price_per_sqft" in df else None,
        "by_borough": {},
    }
    if "borough" in df.columns:
        for b, sub in df.groupby("borough"):
            if b is None or pd.isna(b):
                continue
            summary["by_borough"][str(b)] = {
                "n": int(len(sub)),
                "median_price": int(sub["price"].median()),
                "median_ppsf": float(round(sub["price_per_sqft"].median(), 2)),
            }
    if "sale_date" in df.columns and not df["sale_date"].isna().all():
        summary["date_range"] = {
            "earliest": str(df["sale_date"].min().date()),
            "latest": str(df["sale_date"].max().date()),
        }
    return json.dumps(summary, indent=2, default=str)


def check_data_freshness(**kwargs) -> str:
    """Probe NYC Open Data for dataset metadata."""
    from src.data.socrata_client import SocrataClient
    client = SocrataClient()
    meta = client.get_dataset_freshness()
    return json.dumps(meta, indent=2)


# Dispatcher
TOOL_DISPATCH = {
    "classify_property": classify_property,
    "get_recommendations": get_recommendations,
    "get_neighborhood_profile": get_neighborhood_profile,
    "compare_properties": compare_properties,
    "get_market_summary": get_market_summary,
    "get_live_market_data": get_live_market_data,
    "check_data_freshness": check_data_freshness,
}


def execute_tool(name: str, params: dict) -> str:
    if name not in TOOL_DISPATCH:
        return f"Unknown tool: {name}"
    try:
        return TOOL_DISPATCH[name](**params)
    except Exception as e:
        log.error(f"Tool '{name}' failed: {e}")
        return f"Tool execution error: {e}"
