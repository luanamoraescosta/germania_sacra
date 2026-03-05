"""
ui.py
-----
Streamlit user interface for the Germania Sacra Monastery Explorer.

This module only handles layout, widget state, and display logic.
All data-fetching and LLM work is delegated to rag_pipeline.py.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

from config import (
    AVAILABLE_MODELS,
    EXAMPLE_QUERIES,
    HF_TOKEN,
    PAGE_CSS,
)
from rag_pipeline import GermaniaSacraPipeline


# ---------------------------------------------------------------------------
#  Page setup (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

def configure_page() -> None:
    st.set_page_config(
        page_title="Germania Sacra Monastery Explorer",
        page_icon="(church)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, bool]:
    """
    Draw the sidebar and return the user's choices.

    Returns
    -------
    hf_token : str
    model_name : str
    show_query : bool
    """
    with st.sidebar:
        st.header("Configuration")
        hf_token: str = st.text_input(
            "HuggingFace Token",
            value=HF_TOKEN,
            type="password",
            help="Your HuggingFace API token.",
        )
        model_name: str = st.selectbox("Language Model", AVAILABLE_MODELS)
        show_query: bool = st.checkbox("Show generated SPARQL query", value=False)

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            "Query FactGrid for Germania Sacra monastery data using "
            "natural-language questions."
        )

        st.markdown("---")
        st.subheader("Example Questions")
        for i, q in enumerate(EXAMPLE_QUERIES):
            if st.button(q, key=f"example_{i}"):
                st.session_state.pending_question = q

    return hf_token, model_name, show_query


# ---------------------------------------------------------------------------
#  Main question form
# ---------------------------------------------------------------------------

def render_question_input() -> str:
    """Render the text input and return the current question string."""
    st.subheader("Ask a question about Germania Sacra monasteries")
    default = st.session_state.get("pending_question", "")
    question: str = st.text_input("Your question:", value=default)
    return question


# ---------------------------------------------------------------------------
#  Results display
# ---------------------------------------------------------------------------

def render_sparql(query: str) -> None:
    st.subheader("Generated SPARQL Query")
    st.markdown(
        f'<div class="query-box"><pre>{query}</pre></div>',
        unsafe_allow_html=True,
    )


def render_results(results: list[dict] | None) -> None:
    st.subheader("Results")
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(results)} row(s) returned.")
    else:
        st.info("No results returned from FactGrid.")


# ---------------------------------------------------------------------------
#  Pipeline loader (cached so the model is not reloaded on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str, hf_token: str) -> GermaniaSacraPipeline:
    pipe = GermaniaSacraPipeline(model_name=model_name, hf_token=hf_token)
    for msg in pipe.status_messages:
        st.sidebar.info(msg)
    return pipe


# ---------------------------------------------------------------------------
#  Main entry point (called from app.py)
# ---------------------------------------------------------------------------

def run_ui() -> None:
    configure_page()

    st.title("Germania Sacra Monastery Explorer")
    st.markdown(
        "*Explore historical monasteries documented in the Germania Sacra project "
        "through FactGrid queries.*"
    )

    hf_token, model_name, show_query = render_sidebar()
    question = render_question_input()

    # Determine the CSV path (expected next to app.py)
    csv_path = pathlib.Path(__file__).with_name(
        "dataset_monastery_items_and_properties.csv"
    )
    if not csv_path.exists():
        csv_path = None

    col1, _ = st.columns([1, 5])
    search_clicked = col1.button("Search", type="primary")

    # Trigger on button click or on example selection
    trigger = search_clicked or bool(st.session_state.get("pending_question"))

    if trigger:
        active_question = question or st.session_state.get("pending_question", "")
        st.session_state.pending_question = ""   # clear after use

        if not active_question.strip():
            st.warning("Please enter a question before searching.")
            return

        if not hf_token:
            st.warning("Please enter your HuggingFace token in the sidebar.")
            return

        with st.spinner("Running pipeline..."):
            pipe = load_pipeline(model_name, hf_token)
            sparql, results = pipe.run(active_question, csv_path=csv_path)

        if show_query:
            render_sparql(sparql)

        render_results(results)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;'>"
        "Germania Sacra Monastery Explorer | Powered by FactGrid and HuggingFace"
        "</div>",
        unsafe_allow_html=True,
    )
