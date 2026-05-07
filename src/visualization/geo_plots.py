

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# Cluster color palette (8 distinct colors)
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

NYC_CENTER = [40.7128, -74.0060]


def _try_import_folium():
    try:
        import folium
        return folium
    except ImportError:
        log.warning("folium not installed - skipping geo plots")
        return None


def plot_cluster_map(df: pd.DataFrame, cluster_col: str = "cluster",
                       sample_size: int = 5000, save_path=None,
                       zoom_start: int = 11):
    folium = _try_import_folium()
    if folium is None:
        return None

    sample = df.sample(min(sample_size, len(df)), random_state=42)
    m = folium.Map(location=NYC_CENTER, zoom_start=zoom_start, tiles="cartodbpositron")

    for _, row in sample.iterrows():
        c = int(row[cluster_col]) if pd.notna(row[cluster_col]) else 0
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        popup = (
            f"<b>{row.get('borough', '')}</b><br>"
            f"Cluster: {c} ({row.get('cluster_name', '')})<br>"
            f"${row.get('price', 0):,.0f}<br>"
            f"${row.get('price_per_sqft', 0):.0f}/sqft<br>"
            f"Walk: {row.get('walk_score', 0):.0f} | "
            f"School: {row.get('school_quality_score', 0):.1f}<br>"
            f"Label: {row.get('valuation_label_name', '')}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(m)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info(f"Saved cluster map -> {save_path}")
    return m


def plot_valuation_map(df: pd.DataFrame, sample_size: int = 5000,
                        save_path=None, zoom_start: int = 11):
    folium = _try_import_folium()
    if folium is None:
        return None

    color_map = {
        "undervalued": "#2ca02c",
        "fairly_valued": "#7f7f7f",
        "overvalued": "#d62728",
    }
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    m = folium.Map(location=NYC_CENTER, zoom_start=zoom_start, tiles="cartodbpositron")

    for _, row in sample.iterrows():
        label = row.get("valuation_label_name", "fairly_valued")
        color = color_map.get(label, "#7f7f7f")
        popup = (
            f"<b>{row.get('borough', '')}</b><br>"
            f"${row.get('price', 0):,.0f} ({row.get('sqft', 0)} sqft)<br>"
            f"${row.get('price_per_sqft', 0):.0f}/sqft<br>"
            f"<span style='color:{color}'><b>{label.upper()}</b></span>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=3, color=color, fill=True, fill_opacity=0.6,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(m)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info(f"Saved valuation map -> {save_path}")
    return m


def plot_recommendation_map(seed: pd.Series, recommendations: pd.DataFrame,
                              save_path=None, zoom_start: int = 12):
    folium = _try_import_folium()
    if folium is None:
        return None

    m = folium.Map(location=[seed["lat"], seed["lng"]],
                    zoom_start=zoom_start, tiles="cartodbpositron")

    folium.Marker(
        location=[seed["lat"], seed["lng"]],
        popup=f"SEED: ${seed['price']:,.0f}",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(m)

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
        popup = (
            f"<b>#{i}</b><br>"
            f"${row['price']:,.0f}<br>"
            f"Score: {row.get('hybrid_score', row.get('similarity', 0)):.3f}<br>"
            f"Label: {row['valuation_label_name']}"
        )
        folium.Marker(
            location=[row["lat"], row["lng"]],
            popup=folium.Popup(popup, max_width=200),
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info(f"Saved recommendation map -> {save_path}")
    return m

def plot_price_heatmap(df: pd.DataFrame,
                        metric: str = "price_per_sqft",
                        sample_size: int = 8000,
                        save_path=None,
                        zoom_start: int = 11) -> object | None:
    """
    Render a Folium HeatMap showing the spatial intensity of a chosen metric
    across NYC. Supports four metrics:

        price_per_sqft    — price density (hot = expensive)
        crime_rate_per_1k — crime density (hot = unsafe)
        walk_score        — walkability (hot = walkable)
        undervalued       — concentration of undervalued properties

    The map uses folium.plugins.HeatMap with a LayerControl so the viewer
    can toggle the heatmap on/off over the base tiles.

    Parameters
    ----------
    df          : DataFrame with lat, lng, and the chosen metric column
    metric      : one of the four named metrics above
    sample_size : rows to sample (keep ≤ 10K for browser speed)
    save_path   : optional path to save the .html file
    zoom_start  : initial zoom level

    Returns
    -------
    folium.Map or None (if folium is not installed)
    """
    folium = _try_import_folium()
    if folium is None:
        return None

    try:
        from folium.plugins import HeatMap
    except ImportError:
        log.warning(
            "folium.plugins.HeatMap not available. "
            "Upgrade folium: pip install --upgrade folium"
        )
        return None

    # Metric configuration
    METRIC_CONFIG = {
        "price_per_sqft": {
            "label":      "Price per sqft ($)",
            "title":      "NYC Property Price Density",
            "subtitle":   "Hotter areas = higher price per square foot",
            "gradient":   {0.2: "#313695", 0.4: "#4575b4",
                           0.6: "#fdae61", 0.8: "#f46d43", 1.0: "#a50026"},
            "radius":     14, "blur": 18, "min_opacity": 0.35,
            "col":        "price_per_sqft",
        },
        "crime_rate_per_1k": {
            "label":      "Crime rate (per 1,000 people)",
            "title":      "NYC Crime Rate Density",
            "subtitle":   "Hotter areas = higher crime rate",
            "gradient":   {0.2: "#1a9850", 0.4: "#91cf60",
                           0.6: "#fee08b", 0.8: "#fc8d59", 1.0: "#d73027"},
            "radius":     14, "blur": 18, "min_opacity": 0.35,
            "col":        "crime_rate_per_1k",
        },
        "walk_score": {
            "label":      "Walk Score (0–100)",
            "title":      "NYC Walkability Density",
            "subtitle":   "Hotter areas = more walkable neighborhoods",
            "gradient":   {0.2: "#d73027", 0.4: "#fc8d59",
                           0.6: "#fee08b", 0.8: "#91cf60", 1.0: "#1a9850"},
            "radius":     14, "blur": 18, "min_opacity": 0.35,
            "col":        "walk_score",
        },
        "undervalued": {
            "label":      "Undervalued property concentration",
            "title":      "NYC Undervalued Property Hotspots",
            "subtitle":   "Hotter areas = more undervalued deals here",
            "gradient":   {0.2: "#f7f7f7", 0.5: "#74add1",
                           0.8: "#313695", 1.0: "#023858"},
            "radius":     16, "blur": 20, "min_opacity": 0.30,
            "col":        "valuation_label_name",
        },
    }

    if metric not in METRIC_CONFIG:
        log.error(
            f"Unknown metric '{metric}'. "
            f"Choose from: {list(METRIC_CONFIG)}"
        )
        return None

    cfg_m = METRIC_CONFIG[metric]

    # Prepare weighted data points
    needed = ["lat", "lng", cfg_m["col"]]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        log.error(f"plot_price_heatmap: missing columns {missing_cols}")
        return None

    sample = (
        df[needed]
        .dropna()
        .sample(min(sample_size, len(df)), random_state=42)
        .copy()
    )

    if metric == "undervalued":
        # Binary weight: 1 for undervalued, skip everything else
        sample["_weight"] = (
            sample[cfg_m["col"]] == "undervalued"
        ).astype(float)
        sample = sample[sample["_weight"] > 0]
    else:
        col_vals = sample[cfg_m["col"]].astype(float)
        v_min = col_vals.quantile(0.02)
        v_max = col_vals.quantile(0.98)
        if v_max - v_min < 1e-6:
            v_min, v_max = col_vals.min(), col_vals.max()
        sample["_weight"] = (
            (col_vals - v_min) / (v_max - v_min)
        ).clip(0, 1)

    heat_data = sample[["lat", "lng", "_weight"]].values.tolist()

    if not heat_data:
        log.warning("No data points for heatmap after filtering")
        return None

    # Build Folium map with named tile layers + HeatMap layer
    m = folium.Map(
        location=NYC_CENTER,
        zoom_start=zoom_start,
        tiles=None,   # tiles added as named layers below
    )

    folium.TileLayer(
        "cartodbpositron", name="Light (CartoDB)", control=True
    ).add_to(m)
    folium.TileLayer(
        "cartodbdark_matter", name="Dark (CartoDB)", control=True
    ).add_to(m)

    HeatMap(
        data=heat_data,
        name=cfg_m["label"],
        min_opacity=cfg_m["min_opacity"],
        radius=cfg_m["radius"],
        blur=cfg_m["blur"],
        gradient=cfg_m["gradient"],
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend HTML overlay
    gradient_stops = sorted(cfg_m["gradient"].items())
    gradient_css = ", ".join(
        f"{color} {int(stop * 100)}%"
        for stop, color in gradient_stops
    )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(255,255,255,0.92); padding: 12px 16px;
        border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif; min-width: 210px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:5px;">
        {cfg_m['title']}
      </div>
      <div style="font-size:11px; color:#555; margin-bottom:8px;">
        {cfg_m['subtitle']}
      </div>
      <div style="height:14px; border-radius:4px;
                  background: linear-gradient(to right, {gradient_css});
                  margin-bottom:4px;"></div>
      <div style="display:flex; justify-content:space-between;
                  font-size:10px; color:#333;">
        <span>Low</span><span>High</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        log.info(f"Saved heatmap ({metric}) -> {save_path}")

    return m