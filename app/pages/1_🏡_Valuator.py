import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json

import plotly.graph_objects as go
import streamlit as st

from app.components.theme import inject_global_css, section_header
from src.agent.tools import classify_property

st.set_page_config(page_title="Property Valuator", page_icon="🏡", layout="wide")
inject_global_css()

# Header
st.markdown(
    """
    <div class="hero-card" style="padding: 28px 32px;">
      <div style="display:flex; align-items:center; gap:16px;">
        <div>
          <div class="hero-eyebrow">Step 1 of your home search</div>
          <h1 class="hero-title" style="font-size:32px; margin-bottom:6px;">Is the price right?</h1>
          <p class="hero-sub" style="font-size:15px;">
            Plug in any property's basics. We compare it to thousands of nearby sales and tell you
            if it's a steal, fair, or overpriced — instantly.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Form
section_header("Tell us about the property", icon="home_work")

with st.form("classifier_form"):
    c1, c2, c3 = st.columns(3)
    borough = c1.selectbox("📍 Borough", ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"])
    sqft = c2.number_input("📏 Square Feet", 200, 20000, 1500, step=50)
    price = c3.number_input("💰 List Price ($)", 100_000, 20_000_000, 900_000, step=10_000)

    c4, c5, c6 = st.columns(3)
    num_units = c4.number_input("🛏️ Units / Bedrooms", 1, 20, 2)
    year_built = c5.number_input("🏗️ Year Built", 1800, 2024, 1985)
    income = c6.number_input("💵 Median income (neighborhood)", 20000, 400000, 85000, step=5000)

    st.markdown("##### 🧭 Neighborhood characteristics")
    c7, c8, c9, c10 = st.columns(4)
    walk = c7.slider("🚶 Walk Score", 0, 100, 80)
    transit = c8.slider("🚇 Transit Score", 0, 100, 80)
    crime = c9.slider("🚨 Crime / 1k people", 0.0, 80.0, 18.0)
    school = c10.slider("🎓 School Score (0-10)", 0.0, 10.0, 6.5)

    submitted = st.form_submit_button("🔍 Analyze this property", type="primary", use_container_width=True)

# Result
if submitted:
    with st.spinner("Running ML model..."):
        try:
            result = json.loads(classify_property(
                borough=borough, sqft=sqft, price=price,
                num_units=num_units, year_built=year_built,
                walk_score=walk, transit_score=transit,
                crime_rate_per_1k=crime, school_quality_score=school,
                median_household_income=income,
            ))
        except FileNotFoundError:
            st.error(
                "🛑 No trained model found. Run `python main.py --step all` first to train the classifiers."
            )
            st.stop()

    label = result["predicted_label"]
    proba = result["probabilities"]
    summary = result["input_summary"]

    config = {
        "undervalued":   {"emoji": "🟢", "color": "#43a047",
                            "bg": "#e8f5e9", "headline": "GREAT DEAL",
                            "msg": "This property is priced below comparable nearby sales — a good buy."},
        "fairly_valued": {"emoji": "🔵", "color": "#1976d2",
                            "bg": "#e3f2fd", "headline": "FAIR PRICE",
                            "msg": "This property is priced in line with comparable nearby sales."},
        "overvalued":    {"emoji": "🔴", "color": "#e53935",
                            "bg": "#ffebee", "headline": "OVERPRICED",
                            "msg": "This property is priced above comparable nearby sales — negotiate or look elsewhere."},
    }[label]

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    section_header("Verdict", icon="gavel")

    col_main, col_chart = st.columns([1, 1.2], gap="large")

    with col_main:
        st.markdown(
            f"""
            <div style="background: {config['bg']}; border-left: 6px solid {config['color']};
                        border-radius: 10px; padding: 28px;">
              <div style="font-size: 13px; font-weight: 600; color: {config['color']};
                          letter-spacing: 1px; margin-bottom: 6px;">
                {config['emoji']} {config['headline']}
              </div>
              <div style="font-size: 28px; font-weight: 700; color: #202124;
                          margin-bottom: 14px;">
                {label.replace('_', ' ').title()}
              </div>
              <div style="color:#3c4043; font-size: 15px; line-height: 1.5; margin-bottom: 18px;">
                {config['msg']}
              </div>
              <div style="display:flex; gap:24px; padding-top: 14px;
                          border-top: 1px solid rgba(0,0,0,0.08); font-size:13px;">
                <div>
                  <div style="color:#5f6368;">Implied $/sqft</div>
                  <div style="font-weight:600; font-size:18px; color:#202124;">
                    ${summary['implied_ppsf']:,.0f}
                  </div>
                </div>
                <div>
                  <div style="color:#5f6368;">Borough</div>
                  <div style="font-weight:600; font-size:18px; color:#202124;">
                    {summary['borough'].title()}
                  </div>
                </div>
                <div>
                  <div style="color:#5f6368;">Total price</div>
                  <div style="font-weight:600; font-size:18px; color:#202124;">
                    ${summary['price']:,.0f}
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        labels_disp = ["Undervalued", "Fairly Valued", "Overvalued"]
        values = [proba["undervalued"], proba["fairly_valued"], proba["overvalued"]]
        colors = ["#43a047", "#1976d2", "#e53935"]

        fig = go.Figure(go.Bar(
            x=labels_disp, y=values,
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            textfont=dict(size=14, color="#202124"),
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="<b>How the model sees it</b>", font=dict(size=16, color="#202124")),
            yaxis=dict(tickformat=".0%", range=[0, max(values) * 1.25],
                         showgrid=True, gridcolor="#e0e0e0"),
            xaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=30),
            height=320,
            font=dict(family="Roboto, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Drivers explanation
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    with st.expander("🔬 What drove this result?"):
        ppsf = summary['implied_ppsf']
        # Borough median ppsf approximation (from synthetic profile)
        from src.data.synthetic import BOROUGH_PROFILES
        prof = BOROUGH_PROFILES.get(borough, BOROUGH_PROFILES["BROOKLYN"])
        bor_med = prof["ppsf_mean"]
        delta = (ppsf - bor_med) / bor_med * 100

        st.markdown(
            f"""
            - The implied **$/sqft is ${ppsf:,.0f}**; the typical borough median is around **${bor_med:,.0f}**
            - That's **{delta:+.1f}%** vs. borough average
            - Walk Score {walk} ({"high" if walk > 80 else "moderate" if walk > 60 else "low"}) →
              the model rewards walkability
            - School Score {school:.1f}/10 ({"strong" if school > 7 else "average" if school > 5 else "weak"}) →
              schools matter a lot to peer pricing
            - Crime rate {crime:.1f}/1k ({"low" if crime < 15 else "moderate" if crime < 30 else "elevated"})
              → safety pulls comparable prices up or down
            """
        )
