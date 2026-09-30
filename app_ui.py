"""
app_ui.py - Streamlit chat UI for the Agentic AI eBook RAG Chatbot.

Run with:
    python -m streamlit run app_ui.py
"""

import streamlit as st
from rag_pipeline import run_rag_pipeline

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    :root {
        --brand-primary:   #6C63FF;
        --brand-secondary: #48CFD0;
        --bg-dark:         #0F1117;
        --bg-card:         #1A1D2E;
        --bg-card2:        #21253A;
        --text-primary:    #E8EAED;
        --text-muted:      #9AA0B4;
        --border:          rgba(108,99,255,0.25);
        --shadow:          0 8px 32px rgba(0,0,0,0.4);
        --radius:          14px;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: var(--bg-dark); color: var(--text-primary); }
    header[data-testid="stHeader"] { background: var(--bg-dark) !important; }

    section[data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }

    /* ── chunk cards ── */
    .chunk-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        color: var(--text-muted);
        position: relative;
    }
    .chunk-score {
        position: absolute; top: 10px; right: 14px;
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
        color: #fff; font-size: 11px; font-weight: 700;
        padding: 2px 8px; border-radius: 20px;
    }

    /* ── confidence badge ── */
    .confidence-badge {
        display: inline-block; padding: 4px 14px;
        border-radius: 20px; font-size: 13px; font-weight: 700; margin-top: 6px;
    }
    .conf-high { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
    .conf-mid  { background:#3a2e1a; color:#fbbf24; border:1px solid #fbbf24; }
    .conf-low  { background:#3a1a1a; color:#f87171; border:1px solid #f87171; }

    /* ── hero ── */
    .hero { text-align:center; padding: 20px 0 10px; }
    .hero h1 {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero p { color: var(--text-muted); font-size: 1rem; max-width: 560px; margin: 0 auto; }

    /* ── chat message overrides ── */
    [data-testid="stChatMessage"] {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta}]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_confidence = st.toggle("Show confidence score", value=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "What is Agentic AI?",
        "How does Agentic AI differ from traditional AI?",
        "What are the key components of an AI agent?",
        "What are the use cases of Agentic AI?",
        "What challenges or risks are mentioned?",
        "Who is the prime minister of India?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sq_{q[:25]}"):
            st.session_state["prefill"] = q
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph · Pinecone · Gemini · FastAPI")

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🤖 Agentic AI eBook Chatbot</h1>
        <p>Ask anything about the <strong>Agentic AI eBook</strong>.
           Answers are grounded exclusively in the PDF — no hallucinations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            if show_confidence:
                conf = meta.get("confidence", 0)
                cls = "conf-high" if conf >= 0.75 else "conf-mid" if conf >= 0.5 else "conf-low"
                st.markdown(
                    f'<span class="confidence-badge {cls}">Confidence: {conf:.2%}</span>',
                    unsafe_allow_html=True,
                )
            if show_chunks and meta.get("retrieved_chunks"):
                with st.expander("📄 Retrieved Context Chunks", expanded=False):
                    for i, chunk in enumerate(meta["retrieved_chunks"], 1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<span class="chunk-score">#{i} · {chunk["score"]:.4f}</span>'
                            f'<p style="margin-top:6px;padding-right:60px">{chunk["text"]}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ── Chat input ────────────────────────────────────────────────────────────────
# Handle sidebar sample question prefill
default_input = st.session_state.pop("prefill", None)

prompt = st.chat_input(
    placeholder="e.g. What is Agentic AI?",
    key="chat_input",
)

# Use prefill if no direct input (sidebar button clicked)
if default_input and not prompt:
    prompt = default_input

if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Run pipeline and stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking… 🔍"):
            try:
                result = run_rag_pipeline(prompt)
                answer = result["answer"]
                meta = {
                    "confidence": result["confidence"],
                    "retrieved_chunks": result["retrieved_chunks"],
                }

                st.markdown(answer)

                if show_confidence:
                    conf = meta["confidence"]
                    cls = "conf-high" if conf >= 0.75 else "conf-mid" if conf >= 0.5 else "conf-low"
                    st.markdown(
                        f'<span class="confidence-badge {cls}">Confidence: {conf:.2%}</span>',
                        unsafe_allow_html=True,
                    )

                if show_chunks and meta["retrieved_chunks"]:
                    with st.expander("📄 Retrieved Context Chunks", expanded=False):
                        for i, chunk in enumerate(meta["retrieved_chunks"], 1):
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<span class="chunk-score">#{i} · {chunk["score"]:.4f}</span>'
                                f'<p style="margin-top:6px;padding-right:60px">{chunk["text"]}</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except Exception as exc:
                err_str = str(exc)
                if "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                    friendly = (
                        "⚠️ **Gemini API temporarily overloaded.** "
                        "Please wait a few seconds and try again."
                    )
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    friendly = (
                        "⚠️ **Rate limit reached** on the Gemini free tier. "
                        "Please wait ~60 seconds and try again."
                    )
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    friendly = (
                        f"⚠️ **Model not found.** Check your `LLM_MODEL` setting in `.env`.\n\n"
                        f"Details: `{err_str[:300]}`"
                    )
                else:
                    friendly = f"⚠️ **An error occurred:**\n\n```\n{err_str[:500]}\n```"

                st.markdown(friendly)
                st.session_state.messages.append(
                    {"role": "assistant", "content": friendly, "meta": None}
                )
