import sys
from pathlib import Path
from unittest import result
sys.path.append(str(Path(__file__).parent))

# import graph
import streamlit as st
import threading


from config import settings
from src.ingestion import DocumentIngestion
from src.embeddings import EmbeddingStore
from src.graph import CRAGGraph

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Pipeline (once, cached) ────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Loads the full CRAG pipeline once and caches it.
    Returns a ready-to-use CRAGGraph instance.
    """
    with st.spinner("Loading SkillVistaar... (first load takes ~30 seconds)"):
        # Check if vectorstore already exists
        index_file = settings.VECTORSTORE_DIR / "index.faiss"
        store = EmbeddingStore()

        if not index_file.exists():
            # First time — need to ingest and embed
            st.info("First run detected — downloading and indexing documents...")
            ingestion = DocumentIngestion()
            chunks = ingestion.run()
            vectorstore = store.build(chunks)
        else:
            # Already indexed — just load
            vectorstore = store.load()

        return CRAGGraph(vectorstore)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/320px-Flag_of_Europe.svg.png", width=50)
        st.title("SkillVistaar")
        st.caption("AI Career Coach for Indian Students")

        st.divider()

        st.markdown("### 📚 Knowledge Base")
        st.markdown("""
        This system is trained on:
        - 📄 NEP 2020
        - 📄 NCF-SE 2023
        - 📄 India Skills Report 2026
        """)

        st.divider()

        st.markdown("### 🔄 How CRAG Works")
        st.markdown("""
        1. **Retrieve** — finds relevant chunks
        2. **Grade** — checks if chunks are useful
        3. **Rewrite** — improves query if needed
        4. **Generate** — produces final answer
        """)

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Built with LangGraph + Ollama + FAISS")
        st.caption(f"Model: {settings.LLM_MODEL}")
        st.caption(f"Embeddings: {settings.EMBEDDING_MODEL}")


# ─── Sample Questions ─────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "What does NEP 2020 say about vocational education?",
    "What are the top skills required for jobs in India in 2026?",
    "How does NEP 2020 plan to improve foundational literacy?",
    "What career opportunities exist in AI for Indian students?",
    "What is the 5+3+3+4 structure in NEP 2020?",
]


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Render sidebar
    render_sidebar()

    # Header
    st.title("🎓 SkillVistaar — AI Career Coach")
    st.caption(settings.APP_DESCRIPTION)

    st.divider()

    # Sample questions row
    st.markdown("**Try a sample question:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, SAMPLE_QUESTIONS)):
        with col:
            if st.button(
                question[:40] + "...",
                key=f"sample_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = question

    st.divider()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if it was an assistant message
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📎 Sources"):
                        for src in message["sources"]:
                            st.caption(
                                f"📄 {src['file']} — Page {src['page']}"
                            )

            # Show CRAG path info
            if message["role"] == "assistant" and message.get("was_rewritten"):
                st.info(
                    f"🔄 Query was rewritten for better retrieval:\n"
                    f"_{message.get('rewritten_question', '')}_"
                )

    # Load the pipeline
    graph = load_pipeline()

    # Handle sample question clicks
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.user_input = question

    # Chat input
    user_input = st.chat_input(
        "Ask about NEP 2020, career guidance, AI literacy..."
    )

    # Process input
    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        # with st.chat_message("assistant"):
        #     with st.spinner("Thinking..."):
        #       result = graph.run(user_input)

        #     # Display answer
        #     st.markdown(result["answer"])

        #     # Display sources
        #     if result["sources"]:
        #         with st.expander("📎 Sources"):
        #             for src in result["sources"]:
        #                 st.caption(
        #                     f"📄 {src['file']} — Page {src['page']}"
        #                 )

        #     # Show if query was rewritten (CRAG correction happened)
        #     if result["was_rewritten"]:
        #         st.info(
        #             f"🔄 Query was rewritten for better retrieval:\n"
        #             f"_{result['rewritten_question']}_"
        #         )

        if user_input:
         st.session_state.messages.append({
          "role": "user",
          "content": user_input
         })

         with st.chat_message("user"):
          st.markdown(user_input)

         with st.chat_message("assistant"):

          # ── Stop button ──────────────────────────────────────
          # Session state flag to track if user stopped generation
          st.session_state.setdefault("stop_generation", False)
          st.session_state["stop_generation"] = False

          stop_col, _ = st.columns([1, 5])
          with stop_col:
            if st.button("⏹ Stop", key="stop_btn"):
                st.session_state["stop_generation"] = True

          # ── Run pipeline ─────────────────────────────────────
          status = st.status("Running CRAG pipeline...", expanded=True)

          with status:
            st.write("🔍 Retrieving documents...")
            result = graph.run(user_input)
            
            if result["was_rewritten"]:
                st.write("✏️ Query rewritten for better retrieval...")
            
            if result["used_web_search"]:
                st.write("🌐 Local docs insufficient — searching web...")
            
            st.write("✅ Generating answer...")
        
          status.update(label="Done!", state="complete", expanded=False)

          # ── Stream answer word by word ────────────────────────
          if not st.session_state.get("stop_generation", False):
            answer_placeholder = st.empty()
            displayed = ""

            for word in result["answer"].split(" "):
                # Check stop flag on every word
                if st.session_state.get("stop_generation", False):
                    answer_placeholder.markdown(displayed + "\n\n_⏹ Generation stopped._")
                    break
                displayed += word + " "
                answer_placeholder.markdown(displayed + "▌")

            # Final answer without cursor
            if not st.session_state.get("stop_generation", False):
                answer_placeholder.markdown(result["answer"])

          # ── Sources ───────────────────────────────────────────
          if result["sources"]:
            with st.expander("📎 Sources"):
                for src in result["sources"]:
                    if src.get("file") == "web_search":
                        # Web result — show URL
                        st.caption(f"🌐 Web: {src.get('url', 'unknown')}")
                    else:
                        # Local doc — show file + page
                        st.caption(f"📄 {src['file']} — Page {src['page']}")

          # ── CRAG path badges ──────────────────────────────────
          cols = st.columns(3)

          with cols[0]:
            st.success("✅ Retrieved")

          with cols[1]:
            if result["was_rewritten"]:
                st.warning("✏️ Query rewritten")
            else:
                st.success("✅ Query kept")

          with cols[2]:
            if result["used_web_search"]:
                st.warning("🌐 Used web search")
            else:
                st.success("✅ Used local docs")

         # Save to history
         st.session_state.messages.append({
           "role":               "assistant",
           "content":            result["answer"],
           "sources":            result["sources"],
           "was_rewritten":      result["was_rewritten"],
           "rewritten_question": result.get("rewritten_question", ""),
           "used_web_search":    result["used_web_search"]
            })

         # Save assistant message to history
         st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "was_rewritten": result["was_rewritten"],
            "rewritten_question": result.get("rewritten_question", "")
         })


if __name__ == "__main__":
    main()