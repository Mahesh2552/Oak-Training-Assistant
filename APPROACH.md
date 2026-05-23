# Oak Training Assistant — Approach Document

## 1. Purpose

Oak Training Assistant is an internal **question–answering application** over **company project documentation**. It helps new and existing employees explore what each project does, how it works, and how pieces fit together, using **retrieval-augmented generation (RAG)** so answers stay grounded in approved source material rather than model hallucination.

---

## 2. Goals

| Goal | How it is addressed |
|------|----------------------|
| Faithful answers | Context is assembled from ingested docs; prompts instruct the model not to invent facts. |
| Complete workflows when needed | Structured project data is stored in a graph artifact so **workflow steps are not lost** to chunking. |
| Flexible depth | System and user prompts encourage **ChatGPT-style** responses: short overviews when asked briefly, full steps when the user asks for explanation or walkthrough. |
| Low friction for authors | Projects can be authored as **YAML** (rich structure), **TXT**, or **PDF** under a single folder; ingestion rebuilds indexes. |
| Practical deployment | **Groq** (or OpenAI / Ollama) for generation; **local embeddings** for retrieval; no requirement to run large language models on employee laptops. |

---

## 3. Problem With “Vector-Only” RAG

Classic RAG **splits documents into chunks** and retrieves the top *k* chunks by embedding similarity. That works well for FAQs and long prose, but it **fragments** structured content. In particular, **ordered workflow steps** may be split across chunks; similarity search might return only a subset of steps, so the model never sees the full procedure.

---

## 4. Solution: Hybrid RAG (Graph + Vector)

The approach combines two retrieval paths:

1. **Graph / structured retrieval (GraphRAG-style)**  
   - At ingest time, each project is normalized into a **canonical record** (name, overview, problem statement, architecture list, **workflow** as ordered steps, technologies, challenges, use cases).  
   - That record is written to **`graph_store/oak_graph.json`**.  
   - At query time, the app **detects which project** the user likely means (substring match and **fuzzy matching** on project names).  
   - If a project is identified, the **entire structured context** for that project—including **all workflow steps in order**—is passed to the model. Nothing is dropped because of chunk boundaries.

2. **Vector retrieval**  
   - The same material (plus section-level variants) is **chunked**, embedded with **`sentence-transformers/all-MiniLM-L6-v2`**, and stored in **ChromaDB** under **`vector_store/`**.  
   - At query time, the question is embedded with the **same model family**, and the **top‑k** similar chunks are retrieved.  
   - When a project is already known, retrieval can be **scoped** with metadata (`project_name`) and **k** is increased to pull more supporting excerpts.

**Why both:** The graph path guarantees **completeness** for structured fields and workflows. The vector path improves **nuanced** questions, comparisons, and phrasing that does not align with rigid fields.

---

## 5. High-Level Architecture

```
data/projects/  →  run_ingestion.py  →  graph_store/oak_graph.json
                                   →  vector_store/ (Chroma)

User  →  Streamlit UI  →  project detection (graph metadata)
                      →  graph context (if project found)
                      →  vector top‑k (Chroma, optional filter)
                      →  LLM (Groq / OpenAI / Ollama)  →  answer
```

- **Offline:** Ingestion only reads files and writes JSON + Chroma; **no LLM** is required for indexing.  
- **Online:** Each question triggers retrieval, then a **single** (or continued) chat call to the configured LLM with **system prompt + optional history + graph block + vector block**.

---

## 6. Data Sources

| Format | Role |
|--------|------|
| **YAML** | Preferred for **structured** projects: explicit `workflow` steps, lists for architecture, technologies, etc. Drives both graph and vector indexes strongly. |
| **TXT** | Unstructured narrative. Optional first line `project_name: Name` or markdown `# Title`; otherwise the **filename stem** is turned into a display name. Content becomes the **overview** body; other graph fields stay empty. |
| **PDF** | Text extracted with **pypdf**; project name from filename stem; full text as **overview**. Same graph limitations as TXT. |

All sources under **`data/projects/`** are discovered by extension and processed in one ingestion run.

---

## 7. Ingestion Pipeline

**Entry point:** `scripts/run_ingestion.py`

1. **Load** each supported file into a normalized **`ProjectDoc`** (structured from YAML, or “loose” from TXT/PDF).  
2. **Graph build** (`src/ingestion/build_graph.py`): emit `oak_graph.json` with per-project records and simple triples for debugging/navigation.  
3. **Vector build** (`src/ingestion/build_vector_db.py`):  
   - Render each project as markdown-style text (full document plus **per-section** documents for better retrieval).  
   - **SentenceSplitter** (chunk size **900**, overlap **120** in token-oriented units as configured in LlamaIndex).  
   - Embed with **HuggingFaceEmbedding** (`all-MiniLM-L6-v2`).  
   - Write to Chroma collection **`oak_projects`** with metadata such as **`project_name`**, **`section`**, **`source_path`**.  
4. Vector index is **rebuilt** by replacing the collection on each full ingestion run (simple, consistent baseline).

---

## 8. Runtime Query Pipeline

**Entry point:** `src/app/streamlit_app.py`

1. User submits a question (chat).  
2. **GraphRetriever.detect_project** scores the query against known `project_name` values (substring + RapidFuzz, threshold ~78).  
3. If detected: **full** project markdown context from the graph record.  
4. **VectorRetriever.retrieve** with `top_k` from settings (**6** default, **10** when project is known); optional **ExactMatchFilter** on `project_name`.  
5. **User message** bundles question + both context blocks; **recent chat history** is included for follow-ups.  
6. **LLM** returns the answer; UI appends to session history.

---

## 9. LLM Selection (`src/llm/llm_config.py`)

Priority:

1. **`GROQ_API_KEY`** — OpenAI-compatible client with base URL `https://api.groq.com/openai/v1`. A small **subclass** of LlamaIndex’s OpenAI integration avoids OpenAI-only model-name validation so Groq model IDs (e.g. `llama-3.3-70b-versatile`) work.  
2. **`OPENAI_API_KEY`** — standard OpenAI models.  
3. **Ollama** — local models (subject to host RAM).

---

## 10. Prompting Strategy

**System prompt** (`src/prompts/system_prompt.py`): Ground answers in provided context; when the user asks for explanation or steps, **preserve full workflow order** from graph context; when they ask only for an overview or a single topic, **match length and scope**—no mandatory fixed report template.

**User prompt:** Explicit reminder to tailor the answer to the **current question**; includes labeled **graph** and **vector** sections.

---

## 11. Configuration (`config.py`)

| Setting | Meaning |
|---------|---------|
| `PROJECT_DOCS_DIR` | `data/projects/` |
| `GRAPH_JSON_PATH` | `graph_store/oak_graph.json` |
| `VECTOR_STORE_DIR` | Chroma persistence root |
| `CHROMA_COLLECTION` | `oak_projects` |
| `VECTOR_TOP_K` / `VECTOR_TOP_K_WHEN_PROJECT_KNOWN` | Default retrieval breadth |

Environment variables (`.env`): **`GROQ_API_KEY`**, **`GROQ_MODEL`**, optional **`GROQ_CONTEXT_WINDOW`**; or OpenAI / Ollama variables per `llm_config.py`.

---

## 12. Limitations and Tradeoffs

- **TXT/PDF** projects do not populate structured workflow lists in the graph; **graph completeness** is strongest with **YAML**.  
- **Project detection** can miss or mis-attach if names are ambiguous; fuzzy thresholds can be tuned in `GraphRetriever.detect_project`.  
- **Full re-ingestion** rebuilds the vector collection; very large corpora may need incremental strategies not implemented here.  
- **Embedding and LLM** are separate stacks; cost and rate limits apply to the chosen LLM provider.

---

## 13. Operational Checklist

1. Add or edit files in **`data/projects/`**.  
2. Run **`python scripts/run_ingestion.py`**.  
3. Start **`streamlit run src/app/streamlit_app.py`**.  
4. Ensure **`.env`** contains a working **`GROQ_API_KEY`** (or fallback provider).

---

## 14. Repository Map (conceptual)

| Area | Location |
|------|----------|
| Settings | `config.py` |
| Ingestion | `scripts/run_ingestion.py`, `src/ingestion/` |
| Graph retrieval | `src/retrieval/graph_retriever.py` |
| Vector retrieval | `src/retrieval/vector_retriever.py` |
| LLM wiring | `src/llm/llm_config.py` |
| Prompts | `src/prompts/system_prompt.py` |
| UI | `src/app/streamlit_app.py` |
| Doc loading / formats | `src/utils/helpers.py` |

---

*Document version: aligned with the hybrid RAG design and multi-format ingestion described in this repository.*
