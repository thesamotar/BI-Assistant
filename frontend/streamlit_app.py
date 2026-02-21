"""
BI Assistant — Streamlit Frontend
===================================
Run:
    streamlit run frontend/streamlit_app.py
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BI Assistant",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "query_history" not in st.session_state:
    st.session_state.query_history = []          # list of {query, result}
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False
if "feedback_type" not in st.session_state:
    st.session_state.feedback_type = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""


# ---------------------------------------------------------------------------
# Helper: backend calls
# ---------------------------------------------------------------------------


def check_health() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable", "supabase": "unreachable"}


def ask_question(query: str, top_k: int) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        st.error(f"Backend error: {exc.response.status_code} — {exc.response.text}")
    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
    return None


def send_feedback(result: dict, query: str, feedback_type: str) -> bool:
    try:
        r = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
                "feedback": feedback_type,
            },
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        st.error(f"Failed to submit feedback: {exc}")
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    # Health indicator
    health = check_health()
    be_ok = health.get("status") == "ok"
    sb_ok = health.get("supabase") == "ok"

    st.markdown("**Backend**")
    st.markdown(
        "🟢 Connected" if be_ok else "🔴 Unreachable",
    )
    st.markdown("**Supabase**")
    st.markdown(
        "🟢 Connected" if sb_ok else f"🔴 {health.get('supabase', 'unknown')}",
    )

    st.divider()

    top_k = st.slider(
        "Number of sources (top_k)",
        min_value=1,
        max_value=20,
        value=5,
        help="How many source documents to retrieve and pass to the LLM.",
    )

    st.divider()

    # Query history
    st.markdown("**Query History**")
    if not st.session_state.query_history:
        st.caption("No queries yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            short = item["query"][:50] + ("…" if len(item["query"]) > 50 else "")
            if st.button(short, key=f"hist_{i}", use_container_width=True):
                st.session_state.current_result = item["result"]
                st.session_state.current_query = item["query"]
                st.session_state.feedback_given = False
                st.session_state.feedback_type = None

    if st.session_state.query_history:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.query_history = []
            st.session_state.current_result = None
            st.session_state.current_query = ""
            st.session_state.feedback_given = False
            st.rerun()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("📊 Business Intelligence Assistant")
st.caption("Powered by LangChain · Gemini · Supabase (pgvector) · UCB1 Re-ranking")

# Query form
with st.form("query_form", clear_on_submit=False):
    query_input = st.text_area(
        "Ask a business intelligence question",
        value=st.session_state.current_query,
        placeholder="e.g. What are OpenAI's latest product announcements?",
        height=100,
    )
    submitted = st.form_submit_button("🔍 Search", use_container_width=True)

if submitted and query_input.strip():
    with st.spinner("Retrieving and generating answer…"):
        result = ask_question(query_input.strip(), top_k)

    if result:
        st.session_state.current_result = result
        st.session_state.current_query = query_input.strip()
        st.session_state.feedback_given = False
        st.session_state.feedback_type = None

        # Persist to history (avoid duplicates)
        existing_queries = [h["query"] for h in st.session_state.query_history]
        if query_input.strip() not in existing_queries:
            st.session_state.query_history.append(
                {"query": query_input.strip(), "result": result}
            )

elif submitted and not query_input.strip():
    st.warning("Please enter a question before searching.")


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

result = st.session_state.current_result
current_query = st.session_state.current_query

if result:
    st.divider()

    # Answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Sources + scores
    st.subheader("Sources")
    sources = result.get("sources", [])
    scores = result.get("scores", [])

    if sources:
        table_data = []
        for idx, (url, score) in enumerate(zip(sources, scores), start=1):
            table_data.append(
                {
                    "#": idx,
                    "URL": url,
                    "Score": round(score, 4),
                }
            )
        st.dataframe(
            table_data,
            column_config={
                "URL": st.column_config.LinkColumn("URL"),
                "Score": st.column_config.NumberColumn("Score", format="%.4f"),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No sources returned.")

    st.caption(f"Model: `{result.get('model', 'unknown')}`")

    # ---------------------------------------------------------------------------
    # Feedback
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("Was this answer helpful?")

    if st.session_state.feedback_given:
        emoji = "👍" if st.session_state.feedback_type == "positive" else "👎"
        st.success(f"{emoji} Thank you for your feedback!")
    else:
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Yes", use_container_width=True):
                if send_feedback(result, current_query, "positive"):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_type = "positive"
                    st.rerun()
        with col2:
            if st.button("👎 No", use_container_width=True):
                if send_feedback(result, current_query, "negative"):
                    st.session_state.feedback_given = True
                    st.session_state.feedback_type = "negative"
                    st.rerun()
