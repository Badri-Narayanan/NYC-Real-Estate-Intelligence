from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st


@st.cache_data(ttl=600)   # ping at most once per 10 minutes
def _probe_freshness() -> dict:
    """Return dataset metadata from NYC Open Data, cached for 10 min."""
    from src.data.socrata_client import SocrataClient
    return SocrataClient().get_dataset_freshness()


def freshness_badge() -> None:
    """Render a small colored badge in the sidebar."""
    st.markdown("##### 📡 Live Data")

    try:
        meta = _probe_freshness()
    except Exception as e:
        meta = {"available": False, "error": str(e)}

    if meta.get("available"):
        last = meta.get("last_update_utc")
        try:
            dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            age_hours = None

        if age_hours is None:
            tone = ""
            label = "Connected"
        elif age_hours < 24 * 90:
            tone = ""
            label = f"Updated ~{int(age_hours/24)} days ago"
        else:
            tone = "stale"
            label = f"Updated ~{int(age_hours/24)} days ago"

        st.markdown(
            f"""
            <div class="freshness-badge {tone}">
              <span class="freshness-dot {tone} freshness-pulse"></span>
              <strong>NYC Open Data:</strong> connected<br>
              <span style="font-size:11px; opacity:0.85;">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        err = meta.get("error", "unknown")
        # Shorten long stack traces
        err_short = err if len(err) < 60 else err[:57] + "..."
        st.markdown(
            f"""
            <div class="freshness-badge error">
              <span class="freshness-dot error"></span>
              <strong>NYC Open Data:</strong> unreachable<br>
              <span style="font-size:11px; opacity:0.85;">Using local dataset.<br>{err_short}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
