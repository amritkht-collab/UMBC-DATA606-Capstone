"""Streamlit entry point.

Phase 4 Week 7. Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")


def render_sidebar() -> None:
    """TODO Phase 4 Day 1-2: New Query | History | Settings."""
    with st.sidebar:
        st.header("Research Assistant")
        st.caption("UMBC DATA606 Capstone")
        st.divider()
        st.write("Sidebar placeholder — wired up in Phase 4.")


def render_main() -> None:
    """TODO Phase 4 Day 3: tabs for Report, Sources, Charts + export buttons."""
    st.title("Autonomous Research Assistant")
    st.info("Scaffold only. The agent backend is implemented across Phases 1-3.")
    query = st.chat_input("Ask a research question...")
    if query:
        st.chat_message("user").write(query)
        st.chat_message("assistant").write("Agent not yet wired up.")


def main() -> None:
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
