import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from app.components.theme import inject_global_css, section_header
from src.utils.config import load_config
from src.utils.io import load_parquet
from src.visualization.geo_plots import plot_cluster_map, plot_valuation_map

st.set_page_config(page_title="Neighborhood Map", page_icon="🗺️", layout="wide")
inject_global_css()

# Hero
st.markdown(
    """
    <div class="hero-card" style="padding:28px 32px; background: linear-gradient(135deg, #fb8c00 0%, #ffa726 50%, #ffb74d 100%);">
      <div style="display:flex; align-items:center; gap:16px;">
        <div>
          <div class="hero-eyebrow">See the patterns at a glance</div>
          <h1 class="hero-title" style="font-size:32px; margin-bottom:6px;">NYC, visualized.</h1>
          <p class="hero-sub" style="font-size:15px;">
            Color every property by what kind of neighborhood it sits in, or whether it's a deal.
            Spot trends a spreadsheet would never show you.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clustered():
    cfg = load_config()
    return load_parquet(f"{cfg['paths']['data_processed']}/properties_clustered.parquet")


# Controls
try:
    df = load_clustered()
except FileNotFoundError:
    st.error("🛑 Clustering output not found. Run `python main.py --step cluster` first.")
    st.stop()

section_header("Map controls", icon="layers")
c1, c2, c3 = st.columns(3)

mode = c1.radio(
    "🎨 Color by",
    ["Cluster archetype", "Valuation",
     "Price heatmap", "Crime heatmap",
     "Walkability heatmap", "Undervalued hotspots"],
    horizontal=True,
)

borough_filter = c2.selectbox("📍 Borough", ["All"] + sorted(df["borough"].unique()))
sample_size = c3.slider("🔢 Properties shown", 500, 10000, 3000, step=500)

filtered = df if borough_filter == "All" else df[df["borough"] == borough_filter]

# Map
section_header("The map", icon="public")

try:
    from streamlit_folium import folium_static
    from src.visualization.geo_plots import plot_price_heatmap

    HEATMAP_METRICS = {
        "Price heatmap":        "price_per_sqft",
        "Crime heatmap":        "crime_rate_per_1k",
        "Walkability heatmap":  "walk_score",
        "Undervalued hotspots": "undervalued",
    }

    if mode == "Cluster archetype":
        m = plot_cluster_map(filtered, sample_size=sample_size)
    elif mode == "Valuation":
        m = plot_valuation_map(filtered, sample_size=sample_size)
    else:
        m = plot_price_heatmap(
            filtered,
            metric=HEATMAP_METRICS[mode],
            sample_size=sample_size,
        )

    if m is not None:
        folium_static(m, width=1200, height=560)
    else:
        st.info("Install `folium` and `streamlit-folium` for a richer map.")
        st.map(filtered.sample(min(sample_size, len(filtered)))[["lat", "lng"]]
                  .rename(columns={"lat": "latitude", "lng": "longitude"}))
except ImportError:
    st.info("Install `streamlit-folium` for a richer map. Showing basic Streamlit map below.")
    st.map(filtered.sample(min(sample_size, len(filtered)))[["lat", "lng"]]
              .rename(columns={"lat": "latitude", "lng": "longitude"}))

# Cluster profiles
if "cluster_name" in df.columns:
    section_header("What each archetype looks like", icon="category")

    profile_cols = ["price", "price_per_sqft", "walk_score",
                    "transit_score", "school_quality_score",
                    "crime_rate_per_1k", "median_household_income"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profile = (df.groupby("cluster_name")[profile_cols]
                  .median().round(2)
                  .sort_values("price_per_sqft", ascending=False))

    pc1, pc2 = st.columns([1.5, 1])
    with pc1:
        st.markdown("##### Median values per cluster")
        st.dataframe(profile, use_container_width=True)
    with pc2:
        st.markdown("##### Cluster size")
        sizes = df["cluster_name"].value_counts().reset_index()
        sizes.columns = ["cluster", "count"]
        fig = px.pie(
            sizes, names="cluster", values="count", hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
