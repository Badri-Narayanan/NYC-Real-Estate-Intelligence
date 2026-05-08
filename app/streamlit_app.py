import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from app.components.theme import (
    inject_global_css,
    section_header,
    feature_card,
    pill_badge,
    animated_counter_html,
)
from app.components.data_freshness import freshness_badge

# Page config
st.set_page_config(
    page_title="NYC Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 4px 0;">
          <div style="font-size:32px; line-height:1;">🏙️</div>
          <div style="font-weight:600; color:#1976d2; font-size:18px; margin-top:4px;">
            NYC RE Intelligence
          </div>
          <div style="font-size:11px; color:#5f6368;">CS 513 Final Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("##### 🧭 Navigate")
    st.caption("Use the pages below to explore each part of the system.")
    st.markdown(
        """
        - **🏡 Property Valuator** — instant price-fairness check
        - **🎯 Smart Recommender** — find your match
        - **🗺️ Neighborhood Map** — visual NYC explorer
        - **🤖 AI Assistant** — ask anything in plain English
        """
    )
    st.markdown("---")
    freshness_badge()

# Hero header
st.markdown(
    """
    <div class="hero-card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
          <div class="hero-eyebrow">
            &nbsp;NEW YORK CITY  &nbsp;|&nbsp;  POWERED BY MACHINE LEARNING + AI
          </div>
          <h1 class="hero-title">Buy smarter. Sell sharper.</h1>
          <p class="hero-sub">
            We analyze every property in NYC against its neighborhood peers, tell you whether it's a good deal,
            and answer questions in plain English. No spreadsheets. No jargon.
          </p>
        </div>
        <div style="font-size: 84px; opacity: 0.85;">
          <span class="material-icons" style="font-size:96px; color: rgba(255,255,255,0.95);">apartment</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top-row animated counters
counters = [
    {"value": 100000, "label": "Properties analyzed", "icon": "domain", "suffix": "+", "color": "#1976d2"},
    {"value": 4, "label": "ML models compared", "icon": "model_training", "color": "#43a047"},
    {"value": 8, "label": "Neighborhood archetypes", "icon": "explore", "color": "#fb8c00"},
    {"value": 7, "label": "AI assistant tools", "icon": "smart_toy", "color": "#8e24aa"},
]
components.html(animated_counter_html(counters), height=150)

st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

# What it does (3 plain-English use cases)
section_header("What you can do here", icon="rocket_launch")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    feature_card(
        icon="check",
        title="Check if a price is fair",
        body="Type in any property's basics — borough, size, list price, walkability — "
              "and we tell you whether it's priced below, at, or above what similar nearby "
              "properties go for.",
        cta="Try the Valuator →",
        color="#1976d2",
    )
with c2:
    feature_card(
        icon="explore",
        title="Find homes that match your life",
        body="Set your budget and what you care about — schools, safety, commute, walkability. "
              "Our recommender ranks the best-fit properties and explains why each one made the list.",
        cta="Open Recommender →",
        color="#43a047",
    )
with c3:
    feature_card(
        icon="chat",
        title="Just ask, in plain English",
        body="Talk to our AI assistant the way you'd talk to a savvy friend in real estate. "
              "It taps live NYC sales data and our ML models to give you grounded, honest answers.",
        cta="Chat with the AI →",
        color="#8e24aa",
    )

st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

# Data sources (transparency about real vs synthetic)
section_header("Where our data comes from", icon="source")

dc1, dc2 = st.columns([1.2, 1])
with dc1:
    st.markdown(
        """
        <div class="surface-card">
          <h3 style="margin-top:0;">
            <span class="material-icons" style="vertical-align:middle; color:#1976d2;">verified</span>
            NYC Department of Finance
            <span class="pill pill-green">LIVE</span>
          </h3>
          <p style="margin:8px 0 12px 0; color:#3c4043;">
            Every recorded property sale in New York City, published by the city's official data portal
            (<code>data.cityofnewyork.us</code>). We pull this directly from their open API in real time.
          </p>
          <ul style="color:#5f6368; margin:0; padding-left: 20px;">
            <li>Sale prices, addresses, square footage, year built, units</li>
            <li>Updated regularly by NYC; no scraping, fully ethical</li>
            <li>Fallback to our 100K-record reference dataset if the live feed is briefly down</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with dc2:
    st.markdown(
        """
        <div class="surface-card">
          <h3 style="margin-top:0;">
            <span class="material-icons" style="vertical-align:middle; color:#fb8c00;">layers</span>
            Hyperlocal context
            <span class="pill pill-orange">DERIVED</span>
          </h3>
          <p style="margin:8px 0 12px 0; color:#3c4043;">
            NYC DOF only publishes the basics. We layer in walkability, transit access, school
            quality, crime rates, and median income — calibrated to each borough's real averages.
          </p>
          <ul style="color:#5f6368; margin:0; padding-left: 20px;">
            <li>Walk Score, Transit Score, Bike Score</li>
            <li>NYC Open Data crime & schools</li>
            <li>Census income & density</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <details style="margin-top:18px; padding:12px 16px; background:#f8f9fa; border-radius:8px; border:1px solid #e0e0e0;">
      <summary style="cursor:pointer; font-weight:500; color:#5f6368;">
        ❓ Why don't we use StreetEasy, Zillow, or Redfin?
      </summary>
      <p style="margin-top:12px; color:#5f6368; font-size:14px;">
        None of them offer a free public API. Zillow shut theirs down in 2021. StreetEasy never had one.
        Redfin doesn't either. The only alternatives are paid scraping services that violate those sites'
        terms of service — not acceptable for an academic project. NYC Open Data is the official,
        free, ethical source for NYC sales — and it covers <strong>every recorded transaction in the city</strong>,
        not just listings on one platform.
      </p>
    </details>
    """,
    unsafe_allow_html=True,
)

# How it works (visual)
section_header("How it works", icon="account_tree")

st.markdown(
    """
    <div class="flow-container">
      <div class="flow-step">
        <div class="flow-icon" style="background:#e3f2fd; color:#1976d2;">
          <span class="material-icons">cloud_download</span>
        </div>
        <div class="flow-title">1. Ingest</div>
        <div class="flow-body">Pull live NYC DOF sales + reference data</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">
        <div class="flow-icon" style="background:#fff3e0; color:#fb8c00;">
          <span class="material-icons">tune</span>
        </div>
        <div class="flow-title">2. Enrich</div>
        <div class="flow-body">Layer in walkability, schools, crime, transit</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">
        <div class="flow-icon" style="background:#e8f5e9; color:#43a047;">
          <span class="material-icons">model_training</span>
        </div>
        <div class="flow-title">3. Learn</div>
        <div class="flow-body">Train 4 ML models, pick the best</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">
        <div class="flow-icon" style="background:#f3e5f5; color:#8e24aa;">
          <span class="material-icons">smart_toy</span>
        </div>
        <div class="flow-title">4. Answer</div>
        <div class="flow-body">AI assistant translates results to plain English</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

# After the "How it works" section, add:
section_header("Model performance", icon="leaderboard")

try:
    comparison_path = Path(__file__).parent.parent / "reports" / "classifier_comparison.csv"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path, index_col=0)
        m1, m2, m3, m4 = st.columns(4)
        best = comp.index[0]
        m1.metric("Best model", best.upper())
        m2.metric("Accuracy", f"{comp.loc[best,'accuracy']:.1%}")
        m3.metric("F1-Macro", f"{comp.loc[best,'f1_macro']:.3f}")
        m4.metric("ROC-AUC", f"{comp.loc[best,'roc_auc_ovr']:.3f}")
        st.dataframe(comp[['accuracy','f1_macro','roc_auc_ovr','cohen_kappa']].round(4),
                     use_container_width=True)
    else:
        st.info("Run `python main.py --step train` to see live model metrics here.")
except Exception:
    pass

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:16px 0; color:#9aa0a6; font-size:13px;">
      Built for CS 513 (Data Analytics &amp; Machine Learning, Spring 2026) ·
      Powered by Anthropic Claude · Data: NYC Open Data
    </div>
    """,
    unsafe_allow_html=True,
)
