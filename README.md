# Germania Sacra Monastery Explorer

A local web interface for querying the [FactGrid](https://database.factgrid.de) knowledge base using natural-language questions. Built as part of the **RAGesten** research group at the **Niedersächsische Akademie der Wissenschaften zu Göttingen**.

---

## What it does

This tool allows researchers to ask questions in plain English about monasteries documented in the [Germania Sacra](https://www.germania-sacra.de/) project — without needing to write SPARQL queries manually.

Under the hood, it uses a local language model to:
1. Extract entities from the question
2. Map them to FactGrid Q-identifiers
3. Generate a SPARQL query
4. Execute it against the FactGrid endpoint and display the results

---

## Status

> **This application is currently running locally for testing purposes only.**
> It is part of an ongoing NLP-to-SPARQL experiment within the RAGesten project.

The goal is to evaluate how well small, open-weight language models can translate natural-language questions into valid SPARQL queries for a specialized humanities knowledge base.

---

## Project context

**RAGesten** is a research initiative at the Niedersächsische Akademie der Wissenschaften zu Göttingen exploring Retrieval-Augmented Generation (RAG) techniques applied to historical and scholarly datasets. This interface is one of the testbeds for that investigation.

---

## Tech stack

| Component | Technology |
|---|---|
| Interface | [Streamlit](https://streamlit.io) |
| Knowledge base | [FactGrid](https://database.factgrid.de) (Wikibase) |
| Query language | SPARQL |
| Language models | HuggingFace Transformers (local, CPU) |
| LLM orchestration | LangChain |

---

## Running locally

**1. Install dependencies**
```bash
pip install streamlit SPARQLWrapper transformers langchain langchain-core langchain-huggingface torch pandas
```

**2. Set your HuggingFace token**
```powershell
$env:HF_TOKEN = "your_token_here"
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project structure

```
germania_sacra/
    app.py          — entry point
    config.py       — constants, model list, CSS
    sparql_client.py — FactGrid API and SPARQL execution
    rag_pipeline.py  — entity extraction, ID mapping, SPARQL generation
    ui.py           — Streamlit interface
```

---

## Example questions

- List all monasteries in Germany
- Show Benedictine monasteries founded before 1200
- Find Cistercian monasteries with their coordinates
- What monasteries were part of the Germania Sacra project in Bavaria?
- Show monasteries dissolved during the Reformation

---

## License

This project is part of an academic research initiative. Please contact the RAGesten group at the Niedersächsische Akademie der Wissenschaften zu Göttingen for information on reuse and collaboration.
