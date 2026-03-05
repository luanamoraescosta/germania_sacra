"""
sparql_client.py
----------------
Low-level helpers for communicating with the FactGrid REST API
and the FactGrid SPARQL endpoint.

All methods are stateless and do not depend on Streamlit; they can
be reused outside the UI (e.g., in scripts or tests).
"""

from __future__ import annotations

import streamlit as st
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from config import FACTGRID_API_URL, FACTGRID_SPARQL_ENDPOINT, FACTGRID_USER_AGENT


# ---------------------------------------------------------------------------
#  FactGrid entity search
# ---------------------------------------------------------------------------

def search_entities(term: str, language: str = "en", limit: int = 5) -> list[dict]:
    """
    Search the FactGrid Wikibase API for items matching *term*.

    Returns a list of dicts with keys:
        id          – Q-identifier  (e.g. "Q469419")
        label       – human-readable label
        description – short description (may be empty)
    """
    params: dict[str, str | int] = {
        "action": "wbsearchentities",
        "format": "json",
        "search": term,
        "language": language,
        "limit": limit,
        "type": "item",
    }
    try:
        response = requests.get(FACTGRID_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "id": item["id"],
                "label": item["label"],
                "description": item.get("description", ""),
            }
            for item in data.get("search", [])
        ]
    except Exception as exc:
        st.error(f"FactGrid API error: {exc}")
        return []


# ---------------------------------------------------------------------------
#  SPARQL execution
# ---------------------------------------------------------------------------

def execute_sparql(query: str) -> list[dict] | None:
    """
    Execute a SPARQL SELECT query against the FactGrid endpoint.

    Returns a flat list of row-dicts  {variable_name: value_string},
    or None if the query fails.
    """
    if not query:
        return None

    wrapper = SPARQLWrapper(FACTGRID_SPARQL_ENDPOINT, agent=FACTGRID_USER_AGENT)
    wrapper.setQuery(query)
    wrapper.setReturnFormat(JSON)

    try:
        raw = wrapper.query().convert()
        rows: list[dict] = []
        for binding in raw["results"]["bindings"]:
            row: dict[str, str] = {}
            for var in raw["head"]["vars"]:
                row[var] = binding[var]["value"] if var in binding else ""
            rows.append(row)
        return rows
    except Exception as exc:
        st.error(f"SPARQL execution failed: {exc}")
        return None
