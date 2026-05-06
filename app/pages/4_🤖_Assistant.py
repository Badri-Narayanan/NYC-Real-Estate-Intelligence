import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.theme import inject_global_css, section_header
from src.agent.llm_agent import RealEstateAgent

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
inject_global_css()

# Hero
st.markdown(
    """
    <div class="hero-card" style="padding:28px 32px; background: linear-gradient(135deg, #8e24aa 0%, #ab47bc 50%, #ce93d8 100%);">
      <div style="display:flex; align-items:center; gap:16px;">
        <div>
          <div class="hero-eyebrow">Powered by Claude + your trained models</div>
          <h1 class="hero-title" style="font-size:32px; margin-bottom:6px;">Just ask.</h1>
          <p class="hero-sub" style="font-size:15px;">
            Your personal NYC real-estate analyst. It calls live data, runs the ML models, and
            explains everything in plain English.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Agent setup
@st.cache_resource
def get_agent():
    return RealEstateAgent()


agent = get_agent()

if not agent.is_available():
    st.error(
        "⚠️ **Agent unavailable.** Add your `ANTHROPIC_API_KEY` to "
        "`config/.env` and restart the app. Everything else still works."
    )
    st.stop()

# Conversation state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.tool_log = []

# Sidebar - tool call log
with st.sidebar:
    st.markdown("##### 🔧 Tool calls")
    st.caption("Watch the assistant reason in real time.")
    if not st.session_state.tool_log:
        st.markdown(
            """<div style="padding:12px; background:#f8f9fa; border-radius:6px;
                           color:#9aa0a6; font-size:12px; text-align:center;">
                  Send a message to see what tools the AI uses.
                </div>""",
            unsafe_allow_html=True,
        )
    for entry in st.session_state.tool_log[-10:]:
        kind = entry["kind"]
        color = "#1976d2" if kind == "CALL" else "#43a047"
        st.markdown(
            f"""<div style="padding:8px 10px; border-left:3px solid {color};
                           background:#fafafa; margin-bottom:6px; font-size:12px;">
                  <strong style="color:{color};">{kind}</strong>
                  <code style="font-size:11px;">{entry['name']}</code>
                </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("details", expanded=False):
            st.code(entry.get("preview", ""), language="json")

    st.markdown("---")
    if st.button("🔄 Reset conversation", use_container_width=True):
        agent.reset()
        st.session_state.chat_messages = []
        st.session_state.tool_log = []
        st.rerun()

# Sample prompts
section_header("Try one of these to start", icon="lightbulb")

samples = [
    ("🏠 Family budget search",
      "I have a $1M budget for a 2-bedroom in Brooklyn. I prioritize schools and safety. What should I look at?"),
    ("📈 Live market check",
      "What's been happening in the NYC market over the last 90 days? Any boroughs trending differently?"),
    ("💎 Investor scan",
      "Show me undervalued properties under $700K. What are the trade-offs?"),
    ("🔍 Quick valuation",
      "Is a 1500 sqft Brooklyn property listed at $900K with walk score 85 a good deal?"),
]
sample_cols = st.columns(2)
for i, (title, prompt) in enumerate(samples):
    if sample_cols[i % 2].button(f"{title}\n\n_{prompt[:70]}..._",
                                    key=f"sample_{i}",
                                    use_container_width=True):
        st.session_state.pending_prompt = prompt
        st.rerun()

# Chat history
st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
section_header("Conversation", icon="chat")

for m in st.session_state.chat_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# Chat input
prompt = st.chat_input("Ask the assistant anything about NYC real estate...") or st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("⏳ thinking...")

        def on_event(kind, payload):
            if kind == "tool_call":
                preview = str(payload.get("args", ""))[:300]
                st.session_state.tool_log.append({
                    "kind": "CALL", "name": payload["name"], "preview": preview
                })
                thinking.markdown(f"🔧 calling **`{payload['name']}`**...")
            elif kind == "tool_result":
                preview = str(payload.get("result", ""))[:500]
                st.session_state.tool_log.append({
                    "kind": "RESULT", "name": payload["name"], "preview": preview
                })

        try:
            answer = agent.chat_streaming_callback(prompt, on_event=on_event)
        except Exception as e:
            answer = f"❌ Agent error: {e}"

        thinking.empty()
        st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
