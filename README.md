# Oak Training Assistant

Internal "ChatGPT for company projects" using **Hybrid Retrieval (GraphRAG + Vector RAG)**:

- **GraphRAG**: ensures **complete project context** (especially full workflows/steps) is always retrieved.
- **Vector RAG (ChromaDB)**: brings in supporting snippets for nuanced questions and comparisons.

## What you get

- Data ingestion pipeline: `data/` → **graph_store** + **vector_store**
- Hybrid retrieval: project detection → graph retrieval → vector retrieval → LLM reasoning → structured answer
- Streamlit chat UI

## Prerequisites

- **Python 3.12+** (Python 3.14 recommended — latest stable).
- **Groq API key** (recommended): set `GROQ_API_KEY`
  - Alternative: **OpenAI API key** (`OPENAI_API_KEY`) or **Ollama** (local).

## Install

From the repository root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional env file:

- Copy `.env.example` → `.env`
- Set one of:
  - `GROQ_API_KEY=...` and optionally `GROQ_MODEL=llama-3.3-70b-versatile`
  - `OPENAI_API_KEY=...` and optionally `OPENAI_MODEL=gpt-4.1-mini`
  - Ollama runs automatically as fallback (no key needed)

## Add project documents

Drop YAML files in `data/projects/`. See the included example:

- `data/projects/raman_drug_detection.yaml`

## Build indexes (vector + graph)

```bash
python scripts/run_ingestion.py
```

This writes:

- `vector_store/` (ChromaDB persistent store)
- `graph_store/oak_graph.json` (project knowledge graph)

## Run the chat app

```bash
streamlit run src/app/streamlit_app.py
```

## LLM provider priority

1. **Groq** — if `GROQ_API_KEY` is set (fast, free tier available). Uses `llama-index-llms-groq`.
2. **OpenAI** — if `OPENAI_API_KEY` is set.
3. **Ollama** — local fallback (requires Ollama running on `localhost:11434`).

## Notes on the Hybrid Retrieval design

- **Project detection** identifies the most likely project name (fuzzy match + metadata hints).
- **Graph retrieval** returns **all fields** and **all workflow steps** for that project (no chunk-loss).
- **Vector retrieval** adds supporting excerpts across docs, filtered by project when possible.

## Extending

1. Add new YAML docs to `data/projects/`
2. Re-run ingestion: `python scripts/run_ingestion.py`
3. Restart Streamlit (if needed)
