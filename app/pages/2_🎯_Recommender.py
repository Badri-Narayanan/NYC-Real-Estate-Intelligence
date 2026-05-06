import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from app.components.theme import inject_global_css, section_header
from src.models.recommendation.content_based import ContentBasedRecommender
from src.utils.config import load_config

st.set_page_config(page_title="Smart Recommender", page_icon="🎯", layout="wide")
inject_global_css()


# Hero
st.markdown(
    """
    <div class="hero-card" style="padding:28px 32px; background: linear-gradient(135deg, #43a047 0%, #66bb6a 50%, #81c784 100%);">
      <div style="display:flex; align-items:center; gap:16px;">
        <div>
          <div class="hero-eyebrow">Personalized matches in seconds</div>
          <h1 class="hero-title" style="font-size:32px; margin-bottom:6px;">Find homes that fit your life.</h1>
          <p class="hero-sub" style="font-size:15px;">
            Set your budget and what matters most. We rank thousands of properties and surface the
            ones that genuinely match — with a clear reason for each.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_recommender():
    cfg = load_config()
    return ContentBasedRecommender().load(
        Path(cfg["paths"]["models"]) / "recommender_content.pkl"
    )


# Form
section_header("What are you looking for?", icon="tune")

with st.form("rec_form"):
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("💰 Max budget ($)", 200_000, 10_000_000, 1_200_000, step=50_000)
    borough = c2.selectbox(
        "📍 Borough",
        ["Any", "MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"],
    )
    bedrooms = c3.number_input("🛏️ Min bedrooms / units", 1, 10, 2)

    st.markdown("##### What matters most to you?")
    p1, p2, p3, p4, p5 = st.columns(5)
    walkable = p1.checkbox("🚶 Walkability", True)
    safe = p2.checkbox("🛡️ Low crime", True)
    school = p3.checkbox("🎓 Top schools", True)
    transit = p4.checkbox("🚇 Good transit", False)
    undervalued = p5.checkbox("💎 Undervalued only", True)

    top_n = st.slider("How many results?", 3, 30, 10)
    submitted = st.form_submit_button("🔍 Find matches", type="primary", use_container_width=True)


# Results
if submitted:
    try:
        rec = load_recommender()
    except FileNotFoundError:
        st.error("🛑 Recommender not built. Run `python main.py --step recommend` first.")
        st.stop()

    prefs = {
        "budget_max": budget, "bedrooms_min": bedrooms,
        "prefer_walkable": walkable, "prefer_safe": safe,
        "prefer_school": school, "prefer_transit": transit,
        "prefer_undervalued": undervalued,
    }
    if borough != "Any":
        prefs["borough"] = borough

    with st.spinner("Searching..."):
        results = rec.recommend_from_preferences(prefs, top_n=top_n)

    if results.empty:
        st.warning("😅 No properties match. Try increasing your budget or unchecking 'Undervalued only'.")
        st.stop()

    # Top metrics
    section_header(f"Found {len(results)} matches", icon="auto_awesome")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg price", f"${results['price'].mean():,.0f}")
    m2.metric("Avg $/sqft", f"${results['price_per_sqft'].mean():,.0f}")
    m3.metric("Avg school score", f"{results['school_quality_score'].mean():.1f}/10")
    m4.metric("Avg walk score", f"{results['walk_score'].mean():.0f}/100")

    # Map + chart side-by-side
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    cm1, cm2 = st.columns([1.2, 1])
    with cm1:
        st.markdown("##### 🗺️ Where are they?")
        st.map(
            results[["lat", "lng"]].rename(columns={"lat": "latitude", "lng": "longitude"}),
            zoom=10,
        )
    with cm2:
        st.markdown("##### 🏙️ Borough mix")
        bor_counts = results["borough"].value_counts().reset_index()
        bor_counts.columns = ["borough", "count"]
        fig = px.bar(
            bor_counts, x="count", y="borough", orientation="h",
            color="borough",
            color_discrete_sequence=["#1976d2", "#43a047", "#fb8c00", "#8e24aa", "#e53935"],
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            xaxis_title=None, yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Pretty results table
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    section_header("Top matches", icon="format_list_numbered")

    show = results[[
        "borough", "price", "sqft", "num_units",
        "walk_score", "transit_score", "school_quality_score",
        "crime_rate_per_1k", "valuation_label_name", "preference_score"
    ]].copy()

    show.insert(0, "Rank", range(1, len(show) + 1))
    show["price"] = show["price"].apply(lambda x: f"${x:,.0f}")
    show["preference_score"] = show["preference_score"].round(2)

    show.columns = [
        "#", "Borough", "Price", "Sqft", "Units",
        "Walk", "Transit", "School", "Crime/1k", "Tag", "Match"
    ]
    st.dataframe(show, use_container_width=True, hide_index=True)
