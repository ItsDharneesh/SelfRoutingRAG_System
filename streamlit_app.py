"""Streamlit UI for Agentic RAG System (Router-based)"""

import streamlit as st
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.doc_ingestion.doc_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# page config
st.set_page_config(
    page_title="🤖 Agentic RAG Search",
    page_icon="🔍",
    layout="centered"
)

# simple css
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def init_session_state():
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    llm = Config.get_llm()

    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    vector_store = VectorStore()

    documents = doc_processor.process_urls(
        Config.DEFAULT_URLS
    )

    vector_store.create_vectorstore(documents)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )

    rag_graph = graph_builder.build()
    return rag_graph, len(documents)


def main():
    init_session_state()

    st.title("🔍 Agentic RAG Document Search")
    st.markdown("Docs first. Web fallback only if needed.")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_graph, num_chunks = initialize_rag()
            st.session_state.rag_graph = rag_graph
            st.session_state.initialized = True
            st.success(
                f"✅ System ready ({num_chunks} document chunks indexed)"
            )

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Enter your question",
            placeholder="Ask something..."
        )
        submit = st.form_submit_button("🔍 Search")

    if submit and question and st.session_state.rag_graph:
        with st.spinner("Thinking..."):
            start_time = time.time()

            result = st.session_state.rag_graph.invoke(
                {"question": question}
            )

            elapsed = time.time() - start_time

            answer = result.get("answer", "")
            docs = result.get("retrieved_docs", [])
            used_web = result.get("use_web", False)

            st.markdown("### 💡 Answer")
            st.success(answer)

            if used_web:
                st.caption("🌐 Used web fallback")
            else:
                st.caption("📄 Answered from documents")

            if docs:
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(docs, 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )

            st.caption(f"⏱️ Response time: {elapsed:.2f}s")

            # 🔍 DEBUG LOGS (SAFE)
            with st.expander("🪵 Debug logs"):
                st.write("Retrieved docs count:", result.get("debug_retrieved_count"))
                st.write("Judge decision:", result.get("debug_judge_decision"))
                st.write("Used web:", used_web)

                if used_web:
                    st.write("Raw Tavily response:")
                    st.code(result.get("debug_web_raw", "")[:2000])

                    st.write("Web context passed to LLM:")
                    st.code(result.get("debug_web_context") or "EMPTY")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")


if __name__ == "__main__":
    main()
