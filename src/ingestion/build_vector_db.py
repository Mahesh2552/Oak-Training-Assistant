from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings
from src.utils.helpers import ProjectDoc, ensure_dir, format_project_as_markdown, load_project_docs


def _project_to_documents(doc: ProjectDoc) -> list[Document]:
    """
    Create LlamaIndex Documents with metadata so we can filter by project.
    We store both:
    - full structured markdown
    - section-specific documents (helps comparisons and targeted answers)
    """
    base_meta = {"project_name": doc.project_name, "source_path": doc.source_path}

    docs: list[Document] = []
    docs.append(
        Document(
            text=format_project_as_markdown(doc),
            metadata={**base_meta, "section": "full"},
        )
    )

    def add_section(title: str, text: str) -> None:
        if text and text.strip():
            docs.append(Document(text=f"{doc.project_name}\n\n## {title}\n{text.strip()}\n", metadata={**base_meta, "section": title}))

    add_section("Overview", doc.overview)
    add_section("Problem Statement", doc.problem_statement)
    if doc.architecture:
        add_section("Architecture", "\n".join([f"- {x}" for x in doc.architecture]))
    if doc.workflow:
        wf = []
        for item in doc.workflow:
            step = item.get("step")
            title = str(item.get("title", "")).strip()
            details = str(item.get("details", "")).strip()
            prefix = f"Step {step}" if step not in (None, "", "None") else "Step"
            wf.append(f"- {prefix}: {title}".strip())
            if details:
                wf.append(f"  - {details}")
        add_section("Workflow (step-by-step)", "\n".join(wf))
    if doc.technologies_used:
        add_section("Technologies Used", "\n".join([f"- {x}" for x in doc.technologies_used]))
    if doc.challenges:
        add_section("Challenges", "\n".join([f"- {x}" for x in doc.challenges]))
    if doc.use_cases:
        add_section("Use Cases", "\n".join([f"- {x}" for x in doc.use_cases]))

    return docs


def build_vector_db(
    *,
    projects_dir: Path | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    projects_dir = projects_dir or settings.PROJECT_DOCS_DIR
    persist_dir = persist_dir or settings.VECTOR_STORE_DIR
    collection_name = collection_name or settings.CHROMA_COLLECTION

    ensure_dir(persist_dir)

    # Fresh (re)build by deleting any existing collection
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=120)

    project_docs: list[ProjectDoc] = load_project_docs(projects_dir)
    documents: list[Document] = []
    for p in project_docs:
        documents.extend(_project_to_documents(p))

    nodes = splitter.get_nodes_from_documents(documents)

    _ = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)


if __name__ == "__main__":
    build_vector_db()
    print("Vector DB build complete.")
