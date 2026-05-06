from __future__ import annotations
import json
import streamlit as st


def inject_global_css() -> None:
    """Inject Material UI CSS + load fonts via JS to avoid Streamlit sanitizer."""

    # Load Google Fonts via JavaScript (avoids Streamlit eating <link> tags)
    st.markdown(
        """
        <script>
        (function() {
            var fonts = [
                'https://fonts.googleapis.com/icon?family=Material+Icons',
                'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
            ];
            fonts.forEach(function(href) {
                if (!document.querySelector('link[href="' + href + '"]')) {
                    var link = document.createElement('link');
                    link.rel = 'stylesheet';
                    link.href = href;
                    document.head.appendChild(link);
                }
            });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }
        .hero-card {
            background: linear-gradient(135deg, #1976d2 0%, #42a5f5 50%, #64b5f6 100%);
            color: white;
            padding: 40px 44px;
            border-radius: 16px;
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(25,118,210,0.25);
        }
        .hero-eyebrow {
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            opacity: 0.92;
            margin-bottom: 8px;
        }
        .hero-title {
            font-size: 42px !important;
            line-height: 1.1 !important;
            margin: 4px 0 14px 0 !important;
            font-weight: 700 !important;
            color: white !important;
        }
        .hero-sub {
            font-size: 17px !important;
            line-height: 1.5;
            opacity: 0.94;
            margin: 0 !important;
            max-width: 720px;
            color: white !important;
        }
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 32px 0 18px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        .section-header .material-icons { color: #1976d2; font-size: 28px; }
        .section-header .title { font-size: 22px; font-weight: 500; color: #202124; }
        .surface-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #e8eaed;
            height: 100%;
            transition: box-shadow 200ms ease, transform 200ms ease;
        }
        .surface-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        .surface-card h3 { margin-top: 0 !important; font-size: 18px !important; color: #202124 !important; }
        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 28px 24px 24px 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #e8eaed;
            height: 100%;
            display: flex;
            flex-direction: column;
            transition: box-shadow 200ms ease, transform 200ms ease;
            border-top: 4px solid var(--accent, #1976d2);
        }
        .feature-card:hover {
            box-shadow: 0 6px 18px rgba(0,0,0,0.12), 0 3px 6px rgba(0,0,0,0.08);
            transform: translateY(-3px);
        }
        .feature-icon-wrap {
            width: 54px; height: 54px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            margin-bottom: 16px;
        }
        .feature-icon-wrap .material-icons { font-size: 30px; }
        .feature-title { font-size: 18px; font-weight: 500; color: #202124; margin-bottom: 10px; }
        .feature-body { font-size: 14px; color: #5f6368; line-height: 1.55; flex-grow: 1; margin-bottom: 16px; }
        .feature-cta { font-size: 13px; font-weight: 500; color: var(--accent, #1976d2); letter-spacing: 0.3px; }
        .pill {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
            margin-left: 8px; text-transform: uppercase; vertical-align: middle;
        }
        .pill-green { background: #e6f4ea; color: #1e8e3e; }
        .pill-orange { background: #fef7e0; color: #b06000; }
        .pill-blue { background: #e8f0fe; color: #1967d2; }
        .pill-red { background: #fce8e6; color: #c5221f; }
        .flow-container {
            display: flex; align-items: center; justify-content: space-between;
            gap: 8px; flex-wrap: wrap; padding: 8px 0;
        }
        .flow-step {
            flex: 1; min-width: 160px; text-align: center; background: white;
            border-radius: 10px; padding: 18px 12px; border: 1px solid #e8eaed;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .flow-icon {
            width: 56px; height: 56px; border-radius: 50%;
            margin: 0 auto 10px auto; display: flex; align-items: center; justify-content: center;
        }
        .flow-icon .material-icons { font-size: 28px; }
        .flow-title { font-size: 14px; font-weight: 600; color: #202124; margin-bottom: 4px; }
        .flow-body { font-size: 12px; color: #5f6368; line-height: 1.4; }
        .flow-arrow { font-size: 24px; color: #bdbdbd; font-weight: 300; align-self: center; }
        @media (max-width: 800px) { .flow-arrow { display: none; } }
        .freshness-badge {
            background: #f1f8e9; border-left: 3px solid #43a047;
            padding: 10px 12px; border-radius: 4px; font-size: 12px; color: #33691e;
        }
        .freshness-badge.stale { background: #fff3e0; border-left-color: #fb8c00; color: #e65100; }
        .freshness-badge.error { background: #fce8e6; border-left-color: #c5221f; color: #b71c1c; }
        .freshness-dot {
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block; margin-right: 6px; vertical-align: middle; background: #43a047;
        }
        .freshness-dot.stale { background: #fb8c00; }
        .freshness-dot.error { background: #c5221f; }
        .freshness-pulse { animation: pulse 1.6s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .counter-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
        }
        @media (max-width: 800px) { .counter-grid { grid-template-columns: repeat(2, 1fr); } }
        .counter-card {
            background: white; border-radius: 12px; padding: 20px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #e8eaed; display: flex; align-items: center; gap: 14px;
        }
        .counter-icon-wrap {
            width: 48px; height: 48px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        }
        .counter-icon-wrap .material-icons { font-size: 26px; }
        .counter-value { font-size: 28px; font-weight: 700; color: #202124; line-height: 1; margin-bottom: 4px; }
        .counter-label { font-size: 12px; color: #5f6368; letter-spacing: 0.2px; }
        section[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e8eaed; }
        </style>
        """,
        unsafe_allow_html=True,
    )



# Section header

def section_header(title: str, icon: str = "info") -> None:
    """Render a Material-icons-prefixed section header."""
    st.markdown(
        f"""
        <div class="section-header">
          <span class="material-icons">{icon}</span>
          <span class="title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Feature card

def feature_card(icon: str, title: str, body: str,
                  cta: str | None = None,
                  color: str = "#1976d2") -> None:
    """Render a single feature card with icon, title, body, CTA."""
    cta_html = f'<div class="feature-cta">{cta}</div>' if cta else ""
    bg = color + "1A"   # rough 10% alpha
    st.markdown(
        f"""
        <div class="feature-card" style="--accent: {color};">
          <div class="feature-icon-wrap" style="background: {bg};">
            <span class="material-icons" style="color: {color};">{icon}</span>
          </div>
          <div class="feature-title">{title}</div>
          <div class="feature-body">{body}</div>
          {cta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



# Pill badge

def pill_badge(text: str, color: str = "blue") -> str:
    """Return inline HTML for a colored pill. Doesn't print on its own."""
    return f'<span class="pill pill-{color}">{text}</span>'



# Animated counters (CountUp.js)

def animated_counter_html(counters: list[dict]) -> str:
    """
    Build the HTML+JS for a row of animated counters.
    counters: list of dicts {value, label, icon, color, suffix?, prefix?}
    """
    cards_html = ""
    for i, c in enumerate(counters):
        bg = c["color"] + "1A"
        prefix = c.get("prefix", "")
        suffix = c.get("suffix", "")
        cards_html += f"""
        <div class="counter-card">
          <div class="counter-icon-wrap" style="background: {bg};">
            <span class="material-icons" style="color: {c['color']};">{c['icon']}</span>
          </div>
          <div>
            <div class="counter-value">
              <span style="color: {c['color']};">{prefix}</span>
              <span id="counter_{i}" class="counter-num"
                    data-target="{c['value']}"
                    data-duration="1800">0</span>
              <span style="color: {c['color']};">{suffix}</span>
            </div>
            <div class="counter-label">{c['label']}</div>
          </div>
        </div>
        """

    return f"""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
      body {{ margin: 0; font-family: 'Roboto', sans-serif; }}
      .counter-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
      }}
      @media (max-width: 800px) {{
        .counter-grid {{ grid-template-columns: repeat(2, 1fr); }}
      }}
      .counter-card {{
        background: white;
        border-radius: 12px;
        padding: 20px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #e8eaed;
        display: flex;
        align-items: center;
        gap: 14px;
      }}
      .counter-icon-wrap {{
        width: 48px; height: 48px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
      }}
      .counter-icon-wrap .material-icons {{ font-size: 26px; }}
      .counter-value {{
        font-size: 28px;
        font-weight: 700;
        color: #202124;
        line-height: 1;
        margin-bottom: 4px;
      }}
      .counter-label {{
        font-size: 12px;
        color: #5f6368;
        letter-spacing: 0.2px;
      }}
    </style>
    <div class="counter-grid">
      {cards_html}
    </div>
    <script>
      function easeOutCubic(t) {{ return 1 - Math.pow(1 - t, 3); }}
      function animateCounter(el) {{
        const target = parseFloat(el.dataset.target);
        const duration = parseFloat(el.dataset.duration) || 1500;
        const start = performance.now();
        function step(now) {{
          const t = Math.min(1, (now - start) / duration);
          const eased = easeOutCubic(t);
          const current = target * eased;
          el.textContent = formatNumber(current, target);
          if (t < 1) requestAnimationFrame(step);
          else el.textContent = formatNumber(target, target);
        }}
        requestAnimationFrame(step);
      }}
      function formatNumber(n, target) {{
        // Decide formatting based on target magnitude
        if (target >= 1000) return Math.round(n).toLocaleString();
        if (target % 1 === 0) return Math.round(n).toString();
        return n.toFixed(1);
      }}
      document.querySelectorAll('.counter-num').forEach(animateCounter);
    </script>
    """
