"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation pipeline for the Germania Sacra explorer.

Responsibilities
    - Load the HuggingFace model once and reuse it.
    - Extract entities from a natural-language question.
    - Map those entities to FactGrid Q-ids.
    - Generate a FactGrid SPARQL query.

The class has NO Streamlit imports; all UI feedback is returned as
strings so the caller (ui.py) decides how to display them.
"""

from __future__ import annotations

import csv
import itertools
import pathlib
import re
from typing import Any

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ResponseSchema / StructuredOutputParser moved to langchain_community in newer releases;
# we ship a minimal replacement here to avoid the extra dependency.
import json as _json
import re as _re

class ResponseSchema:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class StructuredOutputParser:
    def __init__(self, schemas: list[ResponseSchema]):
        self._schemas = schemas

    @classmethod
    def from_response_schemas(cls, schemas: list[ResponseSchema]):
        return cls(schemas)

    def get_format_instructions(self) -> str:
        keys = ", ".join(f'"{s.name}"' for s in self._schemas)
        return f'Return a JSON object with keys: {keys}'

    def parse(self, text: str) -> dict:
        # Extract the first JSON object found in the text
        match = _re.search(r'\{.*\}', text, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group())
            except _json.JSONDecodeError:
                pass
        return {s.name: [] for s in self._schemas}

from config import (
    FACTGRID_PROPERTIES,
    FALLBACK_ENTITY_TERMS,
    HF_TOKEN,
)
from sparql_client import search_entities


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _extract_sparql_block(text: str) -> str | None:
    """Return the content of the first ```sparql … ``` block in *text*."""
    match = re.search(r"```sparql(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _query_looks_valid(query: str | None) -> bool:
    """Basic sanity check: the query must have SELECT/ASK and WHERE."""
    if not query:
        return False
    if "..." in query or "\u2026" in query:   # ellipsis placeholder
        return False
    uq = query.upper()
    return ("SELECT" in uq or "ASK" in uq) and "WHERE" in uq


def _build_props_block() -> str:
    """Serialise FACTGRID_PROPERTIES into a compact JSON-like string for prompts."""
    return ", ".join(
        f'{{"label":"{lbl}","id":"{pid}","description":"{desc}"}}'
        for lbl, pid, desc in FACTGRID_PROPERTIES
    )


def _build_items_block(csv_path: pathlib.Path, max_rows: int = 10) -> str:
    """
    Read the first *max_rows* rows from the monastery CSV and return a
    compact JSON-like string suitable for prompt injection.
    """
    examples: list[str] = []
    try:
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in itertools.islice(reader, max_rows):
                if row.get("item", "").startswith("Q"):
                    examples.append(
                        f'{{"id":"{row["item"]}","label":"{row["itemLabel"]}"}}'
                    )
    except FileNotFoundError:
        pass
    return ", ".join(examples) or '"<no-items>"'


# ---------------------------------------------------------------------------
#  Pipeline class
# ---------------------------------------------------------------------------

class GermaniaSacraPipeline:
    """
    Encapsulates model loading and all LLM-based steps of the RAG pipeline.

    Parameters
    ----------
    hf_token : str
        HuggingFace access token.
    model_name : str
        HuggingFace model identifier.
    """

    def __init__(self, model_name: str, hf_token: str = HF_TOKEN) -> None:
        self.model_name = model_name
        self.model_initialized = False
        self.tokenizer: Any = None
        self.model: Any = None
        self._cached_pipe: Any = None          # single pipeline instance, reused every call
        self._status_messages: list[str] = []  # collected for the caller to display

        self._load_model(hf_token)

    # ------------------------------------------------------------------
    #  Public: status messages accumulated during __init__
    # ------------------------------------------------------------------

    @property
    def status_messages(self) -> list[str]:
        return list(self._status_messages)

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------

    def _load_model(self, hf_token: str) -> None:
        self._status_messages.append("Configuring model for efficient CPU usage...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            # Build the HuggingFace pipeline ONCE and reuse it for every query.
            # Recreating it on every call was the main cause of slowness.
            self._cached_pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                truncation=True,
                max_length=2048,
            )
            self.model_initialized = True
            self._status_messages.append("Model loaded successfully on CPU.")
        except Exception as exc:
            self._status_messages.append(f"Model loading failed: {exc}")
            self.model_initialized = False

    # ------------------------------------------------------------------
    #  Internal: return the cached pipeline with updated generation params
    # ------------------------------------------------------------------

    def _make_llm(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        max_length: int = 2048,
        truncation: bool = True,
    ) -> HuggingFacePipeline:
        return HuggingFacePipeline(
            pipeline=self._cached_pipe,
            pipeline_kwargs={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
            },
        )

    # ------------------------------------------------------------------
    #  Step 1: entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, question: str) -> list[str]:
        """
        Extract noun entities from *question*.

        Falls back to keyword matching when the model is unavailable.
        """
        if not self.model_initialized:
            found = [
                t.capitalize()
                for t in FALLBACK_ENTITY_TERMS
                if t in question.lower()
            ]
            return found or ["Monastery"]

        template = (
            "## INSTRUCTIONS\n"
            "- Extract the entities from the given question.\n"
            "- Return ONLY a JSON object with a single key **entities** "
            "that contains a list of the extracted nouns.\n"
            "- Do NOT return anything else.\n\n"
            "## OUTPUT FORMAT\n"
            "{format_instructions}\n\n"
            "## EXAMPLES\n"
            "Question: how much 1 tablespoon of water?\n"
            'Entity: ```json{{"entities": ["Tablespoon"]}}```\n\n'
            "Question: what country is Jakarta in?\n"
            'Entity: ```json{{"entities": ["Jakarta"]}}```\n\n'
            "## QUESTION\n"
            "{question}\n"
            "Entity:"
        )

        parser = StructuredOutputParser.from_response_schemas(
            [ResponseSchema(name="entities", description="list of extracted entities")]
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = self._make_llm(max_new_tokens=256)
        chain = prompt | llm
        raw = chain.invoke({"question": question})
        json_part = raw.split("Entity:")[-1]

        try:
            return parser.parse(json_part).get("entities", [])
        except Exception:
            return ["Monastery"]

    # ------------------------------------------------------------------
    #  Step 2: map entities to FactGrid Q-ids
    # ------------------------------------------------------------------

    def resolve_entity_ids(
        self, question: str, entities: list[str]
    ) -> list[dict]:
        """
        For each entity in *entities*, fetch FactGrid candidates and use
        the LLM to choose the best match.

        Returns a list of dicts: {id, label, description}.
        """
        if not entities:
            entities = ["Monastery"]

        candidates: dict[str, list[dict]] = {
            e: search_entities(e) for e in entities
        }

        # Simple fallback (no LLM)
        def _first_candidates() -> list[dict]:
            return [items[0] for items in candidates.values() if items]

        if not self.model_initialized:
            return _first_candidates()

        template = (
            "## INSTRUCTIONS\n"
            "- For each entity supplied, choose the most appropriate FactGrid item.\n"
            "- Return ONLY a JSON object with a single key **ids** containing a list.\n"
            "- Do NOT hallucinate extra items.\n\n"
            "## OUTPUT FORMAT\n"
            "{format_instructions}\n\n"
            "## EXAMPLE\n"
            "Question: Monasteries in Germany\n"
            'Entities: ["Monastery","Germany"]\n'
            'FactGrid Entities: ```json{{"Monastery":[{{"id":"Q469419","label":"Carmelite monastery Marienau","description":""}}],'
            '"Germany":[{{"id":"Q183","label":"Germany","description":"country in Central Europe"}}]}}```\n'
            'Entity IDs: ```json{{"ids":[{{"id":"Q469419","label":"Carmelite monastery Marienau","description":""}},'
            '{{"id":"Q183","label":"Germany","description":"country in Central Europe"}}]}}```\n\n'
            "## QUESTION\n"
            "{question}\n"
            "Entities: {entities}\n"
            "FactGrid Entities: ```json{{{candidates}}}```\n"
            "Entity IDs:"
        )

        parser = StructuredOutputParser.from_response_schemas(
            [ResponseSchema(name="ids", description="list of chosen FactGrid items")]
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "entities", "candidates"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = self._make_llm(max_new_tokens=512)
        chain = prompt | llm
        raw = chain.invoke(
            {"question": question, "entities": entities, "candidates": candidates}
        )
        json_part = raw.split("Entity IDs:")[-1]

        try:
            return parser.parse(json_part).get("ids", [])
        except Exception:
            cleaned = (
                json_part.replace("'", '"')
                .replace("\n", "")
                .replace("```json", "")
                .replace("```", "")
            )
            try:
                return parser.parse(cleaned).get("ids", [])
            except Exception:
                return _first_candidates()

    # ------------------------------------------------------------------
    #  Step 3: generate SPARQL
    # ------------------------------------------------------------------

    def generate_sparql(
        self, question: str, entity_ids: list[dict], csv_path: pathlib.Path | None = None
    ) -> str:
        """
        Generate a FactGrid SPARQL query for *question*.

        If the LLM produces an invalid query after two attempts,
        a safe hard-coded fallback is returned.
        """
        if not self.model_initialized:
            return self._fallback_sparql(entity_ids)

        # Load example data for the prompt
        items_block = _build_items_block(csv_path) if csv_path else '"<no-items>"'
        props_block = _build_props_block()

        template = (
            "## INSTRUCTIONS\n"
            "You are an expert on FactGrid and the Germania Sacra monastery dataset.\n\n"
            "Write ONE SPARQL query that answers the user question.\n\n"
            "* Use only the properties listed in the Properties block.\n"
            "* You may reference any of the monastery Q-ids shown in the Items block.\n"
            "* Do NOT add LIMIT, ORDER BY, FILTER, or aggregation unless the user asks.\n"
            "* Return ONLY the query wrapped in a markdown block:\n\n"
            "```sparql\nSELECT ... WHERE {\n    ...\n}\n```\n\n"
            "## ITEMS (sample monastery Q-ids)\n"
            "[{{ items_block }}]\n\n"
            "## PROPERTIES (FactGrid P-ids)\n"
            "[{{ props_block }}]\n\n"
            "## QUESTION\n"
            "{{ question }}\n\n"
            "## SELECTED ENTITY IDs\n"
            "{{ entity_ids }}"
        )

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "entity_ids", "items_block", "props_block"],
            template_format="jinja2",
        )
        llm = self._make_llm(
            max_new_tokens=512,
            temperature=0.2,
            max_length=2048,
            truncation=True,
        )
        chain = prompt | llm

        for _ in range(2):
            raw = chain.invoke(
                {
                    "question": question,
                    "entity_ids": entity_ids,
                    "items_block": items_block,
                    "props_block": props_block,
                }
            )

            sparql = _extract_sparql_block(raw)
            if sparql:
                sparql = sparql.strip().strip("`")

            if _query_looks_valid(sparql):
                # Remove random ordering if the user did not ask for it
                if not any(
                    kw in question.lower() for kw in ("order", "sort", "random")
                ):
                    for clause in (
                        "ORDER BY RAND()",
                        "ORDER BY DESC(RAND())",
                        "ORDER BY ASC(RAND())",
                    ):
                        sparql = sparql.replace(clause, "")
                return sparql

        return self._fallback_sparql(entity_ids)

    # ------------------------------------------------------------------
    #  Fallback query
    # ------------------------------------------------------------------

    def _fallback_sparql(self, entity_ids: list[dict]) -> str:
        """
        Return a minimal, syntactically correct FactGrid query.
        Restricts results to items that have a Klosterdatenbank ID (P471).
        """
        values_clause = ""
        if entity_ids:
            ids = " ".join(
                f"wd:{e['id']}" for e in entity_ids if e.get("id")
            )
            if ids:
                values_clause = (
                    f"  VALUES ?filterItem {{ {ids} }}\n"
                    "  ?monastery ?p ?filterItem .\n"
                )

        return (
            "SELECT ?monastery ?monasteryLabel\n"
            "WHERE {\n"
            "  ?monastery wdt:P471 ?klosterdatenbankID .\n"
            f"{values_clause}"
            "  OPTIONAL { ?monastery wdt:P2 ?instance . }\n"
            '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }\n'
            "}\n"
            "LIMIT 50"
        )

    # ------------------------------------------------------------------
    #  Convenience: run the full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        csv_path: pathlib.Path | None = None,
    ) -> tuple[str, list[dict] | None]:
        """
        Execute the full pipeline for *question*.

        Returns
        -------
        sparql : str
            The generated (or fallback) SPARQL query.
        results : list[dict] | None
            Rows returned by FactGrid, or None on error.
        """
        from sparql_client import execute_sparql  # local import to avoid circularity

        entities = self.extract_entities(question)
        entity_ids = self.resolve_entity_ids(question, entities)
        sparql = self.generate_sparql(question, entity_ids, csv_path=csv_path)
        results = execute_sparql(sparql)
        return sparql, results